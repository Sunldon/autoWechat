"""基于文件系统的长期记忆存储。

每人一个目录 `memory_files/{user_id}/`，内含 3 个分类 Markdown 文件：
- 个人特征.md
- 喜好偏好.md
- 关系性事实.md

写入时每个分类独立调 LLM 去重合并，检索时 LLM 选择器直接选出最相关条目。
"""

import json
import logging
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger(__name__)


def _text_similar(a: str, b: str) -> float:
    """两段文本的相似度 (0~1)。短文本用字符包含率，长文本用 SequenceMatcher。"""
    if a == b:
        return 1.0
    if len(a) < 6 or len(b) < 6:
        # 短文本：一个包含另一个即视为相似
        return 0.85 if (a in b or b in a) else 0.0
    return SequenceMatcher(None, a, b).ratio()


def _parse_json(text: str):
    """JSON 解析容错（与 memory_manager._parse_json_safe 功能一致，避循环导入）。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder(strict=False)
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("]", "}"):
            try:
                return decoder.raw_decode(text[: i + 1])[0]
            except json.JSONDecodeError:
                continue
    return None


class FileMemoryStore:
    """文件记忆存储：每人一个目录，3 个分类文件。"""

    CATEGORIES = ["个人特征", "喜好偏好", "关系性事实"]

    def __init__(self, base_dir: str = "./memory_files", max_lines: int = 60):
        self.base_dir = Path(base_dir)
        self.max_lines = max_lines

    # ── 路径工具 ──────────────────────────────────────────────

    def _user_dir(self, user_id: str) -> Path:
        safe = re.sub(r"[\\/:*?\"<>|]", "_", user_id)
        return self.base_dir / safe

    def _ensure_user_dir(self, user_id: str):
        self._user_dir(user_id).mkdir(parents=True, exist_ok=True)

    def _category_path(self, user_id: str, category: str) -> Path:
        return self._user_dir(user_id) / f"{category}.md"

    # ── 读取 ──────────────────────────────────────────────────

    def load(self, user_id: str, categories: list[str] | None = None) -> list[dict]:
        """读取所有/指定分类文件，返回 [{"text": str, "category": str}, ...]。"""
        if categories is None:
            categories = self.CATEGORIES
        results = []
        for cat in categories:
            path = self._category_path(user_id, cat)
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        results.append({"text": line[2:], "category": cat})
        return results

    def load_all_texts(self, user_id: str) -> list[str]:
        """返回所有记忆纯文本行（用于 LLM 选择器）。"""
        return [item["text"] for item in self.load(user_id)]

    # ── 写入（分类别独立合并）───────────────────────────────────

    def merge(
        self,
        user_id: str,
        new_items: list[dict],
        llm_client,
    ) -> dict[str, list[str]]:
        """LLM 分类合并：每个分类独立调用 LLM 去重，原子写入。

        Args:
            user_id: 联系人名称
            new_items: [{"subject": "对方"|"自己"|"关系", "text": "..."}, ...]
            llm_client: OpenAI 兼容客户端

        Returns:
            {"个人特征": [text, ...], "喜好偏好": [...], "关系性事实": [...]}
        """
        self._ensure_user_dir(user_id)

        # 按 subject 预筛
        for_identity = [i for i in new_items if i["subject"] in ("对方", "自己")]
        for_relation = [i for i in new_items if i["subject"] == "关系"]

        results = {}
        for cat in self.CATEGORIES:
            items = for_relation if cat == "关系性事实" else for_identity
            if items or self._category_path(user_id, cat).exists():
                results[cat] = self._merge_one(cat, user_id, items, llm_client)
            else:
                results[cat] = []
        return results

    def _merge_one(
        self,
        category: str,
        user_id: str,
        new_items: list[dict],
        llm_client,
    ) -> list[str]:
        """对单个分类做 LLM 去重合并。

        核心原则：现有记忆永不删除，LLM 只负责筛选新增条目。
        最终 = 现有 + LLM 筛选后的新增（去重、去无关）。
        """
        path = self._category_path(user_id, category)
        existing_lines = []
        if path.exists():
            existing_lines = [
                l[2:] for l in path.read_text(encoding="utf-8").split("\n")
                if l.startswith("- ")
            ]

        if not new_items:
            return existing_lines

        # ── 第一层：文本级去重（先于 LLM，避免语义重复堆积）──
        deduped = []
        existing_set = set(e.lower() for e in existing_lines)
        for item in new_items:
            text = item["text"].strip()
            t_lower = text.lower()
            # 精确匹配
            if t_lower in existing_set:
                continue
            # 子串匹配（如"朋友"已存在，跳过"朋友"）
            if any(t_lower in e or e in t_lower for e in existing_set):
                continue
            # 模糊匹配（如"队友兼朋友" vs "朋友兼游戏队友"）
            if any(_text_similar(t_lower, e) > 0.75 for e in existing_set):
                continue
            # 新增内部去重
            if any(_text_similar(t_lower, d["text"].lower()) > 0.8 for d in deduped):
                continue
            deduped.append(item)

        skipped = len(new_items) - len(deduped)
        if skipped:
            logger.info(f"[merge] {user_id}/{category}: 文本去重跳过 {skipped} 条，剩余 {len(deduped)}")

        if not deduped:
            return existing_lines

        # 格式化新增（只传筛选后的给 LLM）
        new_block = "\n".join(
            f"[{i['subject']}] {i['text']}" for i in deduped
        )

        cat_defs = {
            "个人特征": (
                "稳定的身份和自我认知：职业、学历、性格、自称的身份角色、自我评价。\n"
                "保留：像「自称XX」「性格XX」这类稳定的自我描述或他人评价。\n"
                "丢弃：具体的时间安排（如几点下班）、一次性事件（如今晚要做什么）。"
            ),
            "喜好偏好": (
                "长期稳定的兴趣：口味、爱好、娱乐方式、品牌偏好。\n"
                "严格排除：一次性事件、工作安排、临时决定。"
            ),
            "关系性事实": (
                "两人共同的经历、约定、互评、关系定位（如队友/同学/同事）。\n"
                "包括：双方互动中表现出来的关系性质、共同参与的活动。"
            ),
        }
        def_text = cat_defs.get(category, "")

        prompt = f"""你是记忆档案管理员。请筛选 {user_id} 的【{category}】档案的新增条目。

【{category} — 严格定义】
{def_text}

【现有档案】（这些已经永久保存，你绝不能删除它们）
{chr(10).join(f"- {l}" for l in existing_lines) if existing_lines else "（空）"}

【本轮新增】（共{len(deduped)}条，已预去重）
{new_block}

【任务 — 只筛选新增】
1. 筛选：每条新增必须完全符合【{category}】定义。不符合的丢弃。
2. 去重：与【现有档案】语义高度相似的丢弃（现有已覆盖）。
3. 去重：新增之间重复的，保留更完整那条。
4. 去掉 [对方]/[自己]/[关系] 前缀。

仅返回 JSON 字符串数组（只输出值得加入的新条目），不要其他内容：
["新条目1", "新条目2", ...]"""

        try:
            response = llm_client.chat.completions.create(
                model=llm_client.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            logger.info(f"[merge] {user_id}/{category} LLM响应({len(content)}字符): {content[:300]}")

            result = _parse_json(content)
            accepted_raw = [str(x) for x in result] if isinstance(result, list) else []

            # 后置去重：LLM 返回的条目也不应互重复或与现有重复
            accepted = []
            for text in accepted_raw:
                t = text.strip().lower()
                if not t:
                    continue
                if t in existing_set:
                    continue
                if any(_text_similar(t, e) > 0.8 for e in existing_set):
                    continue
                if any(_text_similar(t, a.lower()) > 0.8 for a in accepted):
                    continue
                accepted.append(text)

            # 最终 = 现有（永不删除）+ LLM 筛选的新增
            merged = list(existing_lines) + accepted
            logger.info(
                f"[merge] {user_id}/{category}: "
                f"{len(existing_lines)} existing + {len(deduped)} new → "
                f"+{len(accepted)} added = {len(merged)} total"
            )
        except Exception as e:
            logger.warning(f"[merge] {user_id}/{category} LLM 调用失败: {e}")
            merged = existing_lines  # 兜底：保留现有

        # ── 超限压缩：让 LLM 合并去重，而非简单裁尾 ──
        if len(merged) > self.max_lines:
            merged = self._compress_category(category, user_id, merged, llm_client)

        # 原子写入
        if merged:
            content = "\n".join(f"- {t}" for t in merged if t)
            tmp_path = path.with_suffix(".md.tmp")
            tmp_path.write_text(content + "\n", encoding="utf-8")
            os.replace(str(tmp_path), str(path))
        elif path.exists():
            path.write_text("", encoding="utf-8")

        return merged

    # ── 压缩（超限时 LLM 智能合并）─────────────────────────────

    def _compress_category(
        self,
        category: str,
        user_id: str,
        entries: list[str],
        llm_client,
    ) -> list[str]:
        """LLM 将超限的条目压缩到 max_lines 以内：合并相似、删除冗余。"""
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(entries))

        prompt = f"""你是记忆档案整理助手。{user_id} 的【{category}】档案已超过上限，请压缩到不超过 {self.max_lines} 条。

【当前全部条目】（共{len(entries)}条）
{numbered}

【压缩规则】
1. 合并：描述同一件事的不同表述合并为一条最完整的。
   例如 "队友兼朋友"+"朋友兼游戏队友" → "游戏队友兼好友"
2. 去冗余：明显重复或过于宽泛的条目删除（如已有"游戏队友"，删除单纯的"朋友"）。
3. 保留核心：最关键、最具体的事实优先保留。
4. 最终不超过 {self.max_lines} 条。

仅返回 JSON 字符串数组，不要其他内容：
["条目1", "条目2", ...]"""

        try:
            response = llm_client.chat.completions.create(
                model=llm_client.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            logger.info(f"[compress] {user_id}/{category} LLM响应({len(content)}字符): {content[:300]}")

            result = _parse_json(content)
            compressed = [str(x) for x in result] if isinstance(result, list) else []
            if compressed:
                logger.info(
                    f"[compress] {user_id}/{category}: "
                    f"{len(entries)} → {len(compressed)} 条"
                )
                return compressed[:self.max_lines]
        except Exception as e:
            logger.warning(f"[compress] {user_id}/{category} 失败: {e}")

        # 兜底：简单裁尾
        logger.info(f"[compress] {user_id}/{category}: LLM 压缩失败，回退裁尾 → {self.max_lines} 条")
        return entries[:self.max_lines]

    # ── 检索（LLM 选择器）─────────────────────────────────────

    def llm_select(
        self,
        user_id: str,
        query: str,
        top_k: int,
        llm_client,
    ) -> list[str]:
        """LLM 选择器：读取全部记忆 → LLM 直接选出最相关 top_k 条。"""
        all_texts = self.load_all_texts(user_id)
        if not all_texts:
            return []

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(all_texts))

        prompt = f"""你需要从以下记忆库中，为一段微信对话筛选最相关的长期记忆。

【记忆库】（共{len(all_texts)}条）
{numbered}

【当前对话内容】
{query}

【筛选规则 — 严格】
1. 逐条判断：这条记忆与当前对话有实质关联吗？
   - 直接关联：对话主题和记忆描述的是同一件事/同一个人/同类话题 → 选
   - 间接关联：对话可能暗示了记忆中的背景信息 → 选
   - 完全无关：对话和记忆完全是两码事 → 不选
2. 如果某条记忆与对话毫无关系，绝对不要选它。
3. 最多选{top_k}条，不足也没关系——宁缺毋滥。

仅输出选中的条目编号和简要理由，格式：「序号. 记忆文本 | 理由」
每行一条，无其他内容。"""

        try:
            response = llm_client.chat.completions.create(
                model=llm_client.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            logger.info(f"[select] {user_id} LLM响应({len(content)}字符): {content[:300]}")
            selected = []
            for line in content.split("\n"):
                m = re.match(r"(\d+)[\.\、\s]+(.+?)\s*[|｜]", line)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(all_texts):
                        selected.append(all_texts[idx])
            logger.info(f"[select] {user_id}: selected {len(selected)}/{len(all_texts)}")
            return selected[:top_k]
        except Exception as e:
            logger.error(f"[llm_select] {user_id}: {e}")
            return []
