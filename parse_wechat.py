"""从微信 Markdown 聊天记录生成长期记忆档案。

工作流：
1. 读取文件 → 按 other:/self: 配对
2. 每 20 组分一批 → 调 MemoryManager._extract_memories()
3. 收集所有批次 → MemoryManager._file_store.merge()
4. 原子写入 4 个分类文件

用法：
    python parse_wechat.py --generate --user-id 张三
    python parse_wechat.py --generate --user-id 张三 --file wechat_cleaned_test.md --reset
"""

import os
import re
import shutil
import sys

from logger import setup_logger, get_logger

logger = get_logger(__name__)

BATCH_SIZE = 50


def parse_wechat_markdown(md_text: str) -> dict | None:
    """解析 "self: xxx" 或 "other: xxx" 格式的 Markdown 行。"""
    md_text = md_text.strip()
    match = re.match(r"^(self|other):\s*(.*)$", md_text)
    if match:
        role_raw, content = match.groups()
        return {"clean_text": content, "metadata": {"sender": role_raw}}
    logger.warning(f"解析失败: [{repr(md_text[:60])}]")
    return None


def generate_memory_files(file_path: str, user_id: str, reset: bool = False):
    """从聊天记录生成记忆档案 — 全部复用 MemoryManager 的提取+合并逻辑。

    Args:
        file_path: Markdown 聊天记录路径
        user_id: 联系人名称（对方的名字）
        reset: 是否清空旧档案重新生成
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return

    # ── 复用 MemoryManager（含 LLM 客户端和 FileMemoryStore）──
    from memory.memory_manager import MemoryManager, _parse_json_safe

    mm = MemoryManager()
    if not mm._enabled or not mm._llm or not mm._file_store:
        logger.error("MemoryManager 未启用，无法生成记忆档案")
        return

    llm_client = mm._llm
    file_store = mm._file_store

    # ── 重置旧档案 ──
    if reset:
        user_dir = file_store._user_dir(user_id)
        if user_dir.exists():
            shutil.rmtree(str(user_dir))
            logger.info(f"已清空旧档案: {user_dir}")

    # ── 读取和解析 ──
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    logger.info(f"读取 {len(lines)} 行记录")

    # 按 other: + self: 配对
    pairs = []
    for i in range(len(lines) - 1):
        if lines[i].startswith("other:") and lines[i + 1].startswith("self:"):
            parsed_other = parse_wechat_markdown(lines[i])
            parsed_self = parse_wechat_markdown(lines[i + 1])
            if parsed_other and parsed_self:
                pairs.append({
                    "other": parsed_other["clean_text"],
                    "self": parsed_self["clean_text"],
                })

    logger.info(f"解析到 {len(pairs)} 组对话对")

    # ── 分批提取（复用 _extract_memories）──
    all_items = []
    total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start:start + BATCH_SIZE]
        batch_num = start // BATCH_SIZE + 1

        # 转换为 _extract_memories 要求的格式
        messages = []
        for p in batch:
            messages.append({"sender": "other", "text": p["other"]})
            messages.append({"sender": "self", "text": p["self"]})

        logger.info(f"[batch {batch_num}/{total_batches}] {len(messages)} 条消息")

        items = mm._extract_memories(messages, user_id)
        all_items.extend(items)
        logger.info(f"  批次 {batch_num}/{total_batches}: 提取 {len(items)} 条记忆")

    logger.info(f"共提取 {len(all_items)} 条记忆事实，开始分类合并...")

    # ── 去重（同 subject+text 去重）──
    seen = set()
    unique_items = []
    for item in all_items:
        key = (item.get("subject", ""), item.get("text", ""))
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
    logger.info(f"去重后: {len(unique_items)} 条（去除 {len(all_items) - len(unique_items)} 条重复）")

    # ── 分类合并写入（复用 file_store.merge）──
    if unique_items:
        results = file_store.merge(user_id, unique_items, llm_client)
        for cat, texts in results.items():
            logger.info(f"  {cat}: {len(texts)} 条")
    else:
        logger.warning("没有提取到任何记忆事实")

    logger.info(f"记忆档案生成完成 [{user_id}]")


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_logger(console_level=20)
    import argparse

    parser = argparse.ArgumentParser(
        description="从微信聊天记录生成长期记忆档案（文件存储）"
    )
    parser.add_argument("--generate", action="store_true", help="生成记忆档案")
    parser.add_argument("--user-id", default="", type=str, help="联系人名称，如 张三")
    parser.add_argument(
        "--file", default="wechat_cleaned.md", type=str,
        help="聊天记录文件路径（默认 wechat_cleaned.md）",
    )
    parser.add_argument("--reset", action="store_true", help="清空旧档案重新生成")

    args = parser.parse_args()

    if args.generate:
        if not args.user_id:
            logger.error("--generate 需要 --user-id 参数指定联系人名称")
            sys.exit(1)
        generate_memory_files(file_path=args.file, user_id=args.user_id, reset=args.reset)
