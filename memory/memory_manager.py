"""记忆编排层：LLM 提取 + 文件存储 + LLM 选择器检索。

架构（去掉 mem0/ChromaDB/BM25/reranker）：
- _extract_memories(): LLM 从对话中提取双方特征+关系性事实
- FileMemoryStore.merge(): LLM 分类去重 → 原子写入 4 个 .md 文件
- _retrieve_memories(): LLM 选择器从全部记忆中选出最相关条目
- 保留 HyDE 查询改写（可选）和短期记忆窗口
"""

import json as _json
import logging
import re
from collections import OrderedDict
from typing import Optional

from openai import OpenAI

from memory.file_memory import FileMemoryStore
from memory.short_term_memory import ShortTermMemory
from logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# JSON 解析容错
# ═══════════════════════════════════════════════════════════════

def _parse_json_safe(text: str):
    """解析 LLM 的 JSON 输出，自动修复截断和 markdown 代码块。"""
    text = text.strip()
    # 去 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # 1. 直接解析
    try:
        return _json.loads(text, strict=False)
    except _json.JSONDecodeError:
        pass

    # 2. 从右往左找最后一个完整 ']' 或 '}'
    decoder = _json.JSONDecoder(strict=False)
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("]", "}"):
            try:
                obj, _ = decoder.raw_decode(text[: i + 1])
                return obj
            except _json.JSONDecodeError:
                continue
    return None


# ═══════════════════════════════════════════════════════════════
# MemoryManager
# ═══════════════════════════════════════════════════════════════

class MemoryManager:
    """记忆编排层——所有外部代码只与这个类交互。

    用法:
        mm = MemoryManager()
        ctx = mm.read_context("最近忙什么", "张三")     # 检索
        mm.store_context(messages, "张三")               # 存储
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        window_size: int = 10,
    ):
        import config as _app_config

        mc = config or _app_config.MEMORY_CONFIG
        self._enabled = mc.get("enabled", True)

        if not self._enabled:
            self._file_store = None
            self._llm = None
            self.short_term = ShortTermMemory(window_size)
            self._top_k = 5
            self._hyde_max_chars = 3
            self._stored_fingerprints: dict[str, OrderedDict] = {}
            self._max_fingerprints = 50
            logger.info("MemoryManager: 长期记忆已禁用")
            return

        # 文件存储
        fmc = mc.get("file_memory", {})
        self._file_store = FileMemoryStore(
            fmc.get("path", "./memory_files"),
            max_lines=fmc.get("max_lines", 60),
        )

        # LLM 客户端（三合一：提取 + 合并 + 选择器）
        self._llm = self._build_llm_client(mc.get("llm", {}))

        # 短期记忆
        self.short_term = ShortTermMemory(window_size=window_size)

        # 指纹去重
        self._stored_fingerprints: dict[str, OrderedDict] = {}
        self._max_fingerprints = 50

        # 消息缓冲区：攒够再 merge，避免每次对话都调 LLM
        self._pending_messages: dict[str, list[dict]] = {}
        self._merge_threshold = mc.get("merge_threshold", 50)

        # 检索配置
        sc = mc.get("search", {})
        self._hyde_max_chars = sc.get("hyde_max_chars", 3)
        self._top_k = sc.get("top_k", 5)

        logger.info(
            f"MemoryManager 初始化完成 | "
            f"存储={fmc.get('path', './memory_files')} | "
            f"top_k={self._top_k}"
        )

    # ── LLM 客户端 ──────────────────────────────────────────

    def _build_llm_client(self, llm_cfg: dict) -> OpenAI:
        """从配置字典创建 OpenAI 兼容客户端。"""
        base_url = llm_cfg.get("openai_base_url", "http://localhost:1234/v1")
        model = llm_cfg.get("model", "qwen3-vl-8b-instruct")
        api_key = llm_cfg.get("api_key", "not-needed")
        client = OpenAI(base_url=base_url, api_key=api_key)
        client.model = model  # 附加 model 名方便调用
        return client

    # ── 指纹去重 ────────────────────────────────────────────

    def _make_fingerprint(self, msg: dict) -> int:
        """生成消息指纹：(sender, text) 的 hash。"""
        return hash((msg.get("sender", ""), msg.get("text", "")))

    # ── HyDE 查询改写（保留）─────────────────────────────────

    def _should_use_hyde(self, query: str) -> bool:
        """判断是否对当前查询使用 HyDE 改写。"""
        stripped = query.strip()
        if not stripped:
            return False
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", stripped))
        total_chars = len(stripped)
        if chinese_chars == 0 and total_chars <= 10:
            return False
        if 0 < chinese_chars <= self._hyde_max_chars:
            return False
        if total_chars <= 3:
            return False
        return True

    def _hyde_rewrite(self, query: str, user_id: str) -> str:
        """HyDE：用 LLM 生成假设文档，弥补短查询的语义鸿沟。"""
        if not self._should_use_hyde(query) or not self._llm:
            return query

        try:
            hyde_prompt = (
                "你是一个记忆检索助手。给定一条微信聊天消息，请想象一下："
                "如果要检索与该消息相关的用户长期记忆，这些记忆可能包含哪些事实？\n\n"
                "请用一段陈述性文字描述这些潜在相关记忆的内容，不要写对话或评价，"
                "一句话直接输出事实陈述。\n\n"
                f"聊天消息: {query}\n"
                "相关记忆:"
            )
            response = self._llm.chat.completions.create(
                model=self._llm.model,
                messages=[
                    {"role": "system", "content": "你是一个记忆检索助手，输出简洁的相关事实描述。"},
                    {"role": "user", "content": hyde_prompt},
                ],
                max_tokens=256,
                temperature=0.3,
            )
            hyde_doc = response.choices[0].message.content.strip()
            if hyde_doc:
                logger.info(f"[HyDE] {query} → {hyde_doc}...")
                return f"{query}\n\n{hyde_doc}"
        except Exception as e:
            logger.warning(f"[HyDE] 生成失败: {e}")
        return query

    # ── 记忆提取（替代原 _patched_add）────────────────────────

    def _extract_memories(
        self,
        messages: list[dict],
        user_id: str,
    ) -> list[dict]:
        """LLM 从对话消息中提取双方特征+关系性事实。

        Args:
            messages: [{"sender": "self"/"other", "text": "..."}, ...]
            user_id: 聊天对方名称

        Returns:
            [{"subject": "对方"|"自己"|"关系", "text": "..."}, ...]
        """
        if not self._llm:
            return []

        # 格式化对话
        lines = []
        for msg in messages:
            sender = msg.get("sender", "")
            text = msg.get("text", "")
            if not text:
                continue
            if sender == "self":
                lines.append(f"我: {text}")
            elif sender == "other":
                lines.append(f"{user_id}: {text}")
        dialog = "\n".join(lines)

        if not dialog.strip():
            return []

        # 截断确保不超 context
        dialog = dialog[-3000:] if len(dialog) > 3000 else dialog

        system_prompt = f"""你是一个微信聊天记录分析器。从对话中提取值得长期记忆的事实。

对话角色："我"=本人，"{user_id}"=聊天对方。找出所有透露个人信息或两人关系的线索。

规则：只要提到具体事物（设备、游戏、工作、时间、地点）就值得提取。每条一句话，15字以内。严禁返回空数组，除非只有纯表情。

返回 JSON 数组（不要代码块）：
[{{"subject": "对方", "text": "xxx"}}, {{"subject": "自己", "text": "xxx"}}, {{"subject": "关系", "text": "xxx"}}]

subject: "对方"=关于{user_id}、"自己"=关于我、"关系"=两人之间"""

        try:
            response = self._llm.chat.completions.create(
                model=self._llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": dialog},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            logger.info(f"[extract] {user_id} LLM 响应({len(content)}字符): {content[:500]}")

            parsed = _parse_json_safe(content)
            if isinstance(parsed, list):
                items = [
                    item
                    for item in parsed
                    if isinstance(item, dict)
                    and item.get("text")
                    and item.get("subject") in ("对方", "自己", "关系")
                ]
                logger.info(f"[extract] {user_id}: 解析成功, 提取 {len(items)} 条记忆")
                for it in items:
                    logger.info(f"  [{it['subject']}] {it['text']}")
                return items
            else:
                logger.warning(f"[extract] {user_id}: LLM 返回非数组: {content[:200]}")
        except Exception as e:
            logger.error(f"[extract] {user_id}: {e}")

        return []

    # ── 记忆检索（替代原 _hybrid_search）───────────────────────

    def _retrieve_memories(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
    ) -> list[str]:
        """LLM 选择器检索：从全部记忆中选出最相关条目。

        Args:
            query: 检索查询
            user_id: 联系人名称
            top_k: 返回条数，默认 self._top_k

        Returns:
            选中的记忆文本列表
        """
        if not self._file_store or not self._llm:
            return []
        top_k = top_k or self._top_k
        return self._file_store.llm_select(user_id, query, top_k, self._llm)

    # ── 公开接口 ────────────────────────────────────────────

    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        use_hyde: bool = True,
    ) -> list[str]:
        """外部调用的检索管道：HyDE → LLM 选择器 → Top-K。

        Args:
            query: 用户原始查询
            user_id: 联系人名称
            top_k: 返回的记忆条数
            use_hyde: 是否使用 HyDE 查询改写

        Returns:
            选中的记忆文本列表
        """
        search_query = self._hyde_rewrite(query, user_id) if use_hyde else query
        return self._retrieve_memories(search_query, user_id, top_k)

    def read_context(self, current_msg: str, user_id: str) -> str:
        """『读』阶段：检索长期记忆 + 短期记忆，返回格式化的上下文文本。

        Args:
            current_msg: 当前对方的最新消息文本
            user_id: 联系人名称

        Returns:
            格式化的上下文文本，供注入 system prompt
        """
        if not self._enabled:
            return ""

        search_query = self._hyde_rewrite(current_msg, user_id)
        parts = []

        # 1. 长期记忆检索
        results = self._retrieve_memories(search_query, user_id)
        if results:
            parts.append("[长期记忆]\n" + "\n".join(results))
            logger.info(f"检索完成 [{user_id}]: 注入 {len(results)} 条记忆")
            logger.info(f"检索记忆：{results}")
        else:
            logger.info(f"检索完成 [{user_id}]: 无相关记忆")

        # 2. 短期记忆摘要
        summary = self.short_term.get_summary(user_id)
        if summary:
            parts.append(summary)

        return "\n\n".join(parts) if parts else ""

    def store_context(self, messages: list[dict], user_id: str):
        """『写』阶段：消息先攒缓冲区，攒够阈值才提取+合并。

        Args:
            messages: [{"sender": "self"/"other", "text": "..."}, ...]
            user_id: 联系人名称
        """
        if not self._enabled or not self._file_store:
            return

        # 指纹去重
        if user_id not in self._stored_fingerprints:
            self._stored_fingerprints[user_id] = OrderedDict()
        stored = self._stored_fingerprints[user_id]

        new_messages = []
        for msg in messages:
            fp = self._make_fingerprint(msg)
            if fp not in stored:
                new_messages.append(msg)
                stored[fp] = True
                if len(stored) > self._max_fingerprints:
                    stored.popitem(last=False)

        if not new_messages:
            logger.debug(f"无新消息需要存储 [{user_id}]")
            return

        # 更新短期记忆
        self.short_term.update(new_messages, user_id)

        # 攒消息到缓冲区
        buf = self._pending_messages.setdefault(user_id, [])
        buf.extend(new_messages)
        logger.debug(
            f"缓冲 [{user_id}]: {len(buf)}/{self._merge_threshold} 条消息待合并"
        )

        # 攒够了 → 触发提取+合并
        if len(buf) >= self._merge_threshold:
            self._flush_pending(user_id)

    def _flush_pending(self, user_id: str):
        """将缓冲区的消息一次性提取+合并写入文件。"""
        buf = self._pending_messages.pop(user_id, [])
        if not buf:
            return

        logger.info(f"触发合并 [{user_id}]: 缓冲区 {len(buf)} 条消息")

        extracted = self._extract_memories(buf, user_id)
        if extracted:
            logger.info(f"LLM 提取 {len(extracted)} 条记忆 [{user_id}]")
            self._file_store.merge(user_id, extracted, self._llm)
        else:
            logger.info(f"LLM 未提取到新记忆 [{user_id}]")

    def get_all_memories(self, user_id: str) -> list[str]:
        """获取用户全部记忆（用于调试）。"""
        if not self._file_store:
            return []
        return self._file_store.load_all_texts(user_id)
