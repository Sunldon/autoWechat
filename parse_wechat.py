import re
import os
import sys

from logger import setup_logger, get_logger
logger = get_logger(__name__)


def parse_wechat_markdown(md_text):
    """解析 self: xxx 或 other: xxx 格式的 Markdown 行"""
    md_text = md_text.strip()
    pattern = r"^(self|other):\s*(.*)$"
    match = re.match(pattern, md_text)
    if match:
        role_raw, content = match.groups()
        return {
            "clean_text": content,
            "metadata": {"sender": role_raw},
        }
    logger.warning(f"解析失败 - 原始文本: [{repr(md_text)}]")
    return None


BATCH_SIZE = 20  # 每批对话对数，给 LLM 更多上下文做提取


def import_markdown_file(file_path, user_id="", memory=None):
    """导入 Markdown 聊天记录到 Mem0（分批打包）

    Args:
        file_path: Markdown 文件路径
        user_id: 联系人名称
        memory: Mem0 Memory 实例
    """
    if not os.path.exists(file_path):
        return
    if not memory or not user_id:
        logger.warning("需要 --user-id 来指定存储目标")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    logger.info(f"总共读取到 {len(lines)} 行 Markdown 记录, 开始解析...")

    # 解析所有对话对
    pairs = []
    for i in range(len(lines) - 1):
        if "other" in lines[i] and "self" in lines[i + 1]:
            parsed_other = parse_wechat_markdown(lines[i])
            parsed_self = parse_wechat_markdown(lines[i + 1])
            if parsed_other and parsed_self:
                pairs.append((parsed_other["clean_text"], parsed_self["clean_text"]))

    logger.info(f"解析到 {len(pairs)} 组对话对，分批导入（每批 {BATCH_SIZE} 组）...")

    total_imported = 0
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start : start + BATCH_SIZE]
        # 将 batch 内的所有消息组装成一个长对话，给 LLM 更多上下文
        batch_messages = []
        for other_text, self_text in batch:
            batch_messages.append({"role": "user", "content": f"我: {self_text}"})
            batch_messages.append({"role": "other", "content": f"{user_id}: {other_text}"})
        try:
            memory.add(batch_messages, user_id=user_id)
            total_imported += len(batch)
            logger.info(f"  批次 {start // BATCH_SIZE + 1}: 导入 {len(batch)} 组")
        except Exception as e:
            logger.error(f"  批次 {start // BATCH_SIZE + 1} 失败: {e}")

    logger.info(f"成功导入 {total_imported} 组 [问答对] 记忆到 Mem0（user_id={user_id}）")


if __name__ == "__main__":
    setup_logger(console_level=20)  # INFO for standalone CLI
    import argparse

    parser = argparse.ArgumentParser(description="导入聊天记录到 Mem0")
    parser.add_argument("--parse", default=False, action="store_true", help="导入 wechat_cleaned.md")
    parser.add_argument("--user-id", default="", type=str, help="联系人名称，如 张三（--parse 时必需）")
    parser.add_argument("--test", default="", type=str, help="导入后测试搜索关键词")

    args = parser.parse_args()

    try:
        from memory.memory_manager import DEFAULT_MEM0_CONFIG
        from mem0 import Memory

        memory = Memory.from_config(DEFAULT_MEM0_CONFIG)
        logger.info("Mem0 已初始化")
    except Exception as e:
        logger.error(f"Mem0 初始化失败: {e}")
        sys.exit(1)

    if args.parse:
        if not args.user_id:
            logger.warning("--parse 需要 --user-id 参数指定联系人名称")
            sys.exit(1)

        import_markdown_file("wechat_cleaned.md", user_id=args.user_id, memory=memory)

    if args.test and "memory" in dir() and memory:
        logger.info(f"测试搜索: {args.test}")
        try:
            r = memory.search(args.test, filters={"user_id": args.user_id}, top_k=3)
            results = r.get("results", [])
            logger.info(f"找到 {len(results)} 条结果:")
            for i, mem in enumerate(results):
                logger.info(f"  [{i+1}] {mem.get('memory', '')[:80]}")
        except Exception as e:
            logger.error(f"搜索失败: {e}")

        logger.info(f"测试读取上下文: {args.test}")
        try:
            from memory.memory_manager import MemoryManager
            memory_manager = MemoryManager()            
            r = memory_manager.read_context(args.test, user_id=args.user_id)
            logger.info(f"上下文信息:\n{r}")
        except Exception as e:
            logger.error(f"搜索失败: {e}")

