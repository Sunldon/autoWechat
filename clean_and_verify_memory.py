"""
清理旧记忆数据 + 验证 ChromaDB 存储内容是否均为 LLM 提取的记忆

用法:
  1. 清理旧数据:  python clean_and_verify_memory.py --clean
  2. 验证数据:    python clean_and_verify_memory.py --verify --user-id "张三"
  3. 重新导入:    python parse_wechat.py --parse --user-id "张三"
  4. 三步一起:    python clean_and_verify_memory.py --clean --verify --reimport --user-id "张三"
"""
import argparse
import os
import sys

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

from logger import setup_logger, get_logger
logger = get_logger(__name__)


# ============================================================
#  清理旧数据
# ============================================================
def clean_chromadb():
    """删除 ChromaDB 中 wechat_history 相关的 collections"""
    import chromadb
    import shutil

    client = chromadb.PersistentClient(path=os.path.join(_BASE, "chat_db"))

    targets = ["wechat_history", "wechat_memories", "wechat_history_entities"]
    deleted = []
    for name in targets:
        try:
            client.delete_collection(name)
            deleted.append(name)
            logger.info(f"  [删除] collection: {name}")
        except Exception:
            logger.info(f"  [跳过] collection {name} 不存在或已删除")

    # 清理 ChromaDB 数据目录中的子目录
    cpath = os.path.join(_BASE, "chat_db")
    for item in os.listdir(cpath):
        item_path = os.path.join(cpath, item)
        if item != "chroma.sqlite3" and os.path.isdir(item_path):
            shutil.rmtree(item_path)
            logger.info(f"  [删除] 数据目录: {item_path}")

    if not deleted:
        logger.info("  [结果] 所有目标 collection 均已不存在，无需清理")
    else:
        logger.info(f"  [结果] 已清理 {len(deleted)} 个 collection: {deleted}")
    return True


# ============================================================
#  验证存储的记忆
# ============================================================
def verify_memories(user_id: str):
    """验证 ChromaDB 中的记忆是否均为 LLM 提取的，无 raw 消息残留"""
    from memory.memory_manager import MemoryManager

    mm = MemoryManager()
    if not mm.mem0:
        logger.error("Mem0 未初始化")
        return False

    resp = mm.mem0.search("", filters={"user_id": user_id}, top_k=500, threshold=0.0)
    all_raw = (
        resp.get("results", [])
        if isinstance(resp, dict)
        else (resp or [])
    )

    if not all_raw:
        logger.info("ChromaDB 中无任何记忆数据（空数据库）")
        return True

    logger.info(f"共 {len(all_raw)} 条原始记录")

    # --- 检查1: 是否存在 raw 原始消息 ---
    raw_keywords = ["self:", "other:", "一起走啊"]
    raw_found = []
    for m in all_raw:
        text = m.get("memory", "")
        for kw in raw_keywords:
            if kw in text:
                raw_found.append(text)

    if raw_found:
        logger.warning("发现 raw 原始消息残留:")
        for t in raw_found[:10]:
            logger.warning(f"       - {t[:80]}")
    else:
        logger.info("未发现任何 raw 原始消息残留")

    # --- 检查2: 是否每条记忆都是 LLM 提炼的特征 ---
    llm_like = 0
    non_llm = 0
    non_llm_samples = []
    llm_indicators = ["用户", "User", "提到", "表示", "喜欢", "正在", "计划", "希望"]

    for m in all_raw:
        text = (m.get("memory") or "").strip()
        if not text:
            continue
        if len(text) >= 15 and any(kw in text for kw in llm_indicators):
            llm_like += 1
        else:
            non_llm += 1
            non_llm_samples.append(text)

    logger.info(f"LLM 提取特征匹配: {llm_like} 条")
    logger.info(f"非 LLM 特征  : {non_llm} 条")
    if non_llm_samples:
        logger.info("非 LLM 特征样本:")
        for s in non_llm_samples[:10]:
            logger.info(f"  - {s[:80]}")

    # --- 打印全部记忆清单 ---
    logger.info("全部记忆清单:")
    seen = set()
    for i, m in enumerate(all_raw):
        text = (m.get("memory") or "").strip()
        if not text or len(text) < 4:
            continue
        if text not in seen:
            seen.add(text)
            role = m.get("role", "?")
            logger.info(f"  [{i:3d}] [{role:10s}] {text[:100]}")

    logger.info(f"总计: {len(seen)} 条去重记忆")

    success = not raw_found and non_llm == 0
    if success:
        logger.info("全部通过！所有记忆均为 LLM 提取的结构化事实")
    else:
        logger.warning("有异常数据，请检查")
    return success


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    setup_logger(console_level=20)  # INFO for standalone CLI
    parser = argparse.ArgumentParser(description="清理/验证记忆数据")
    parser.add_argument("--clean", action="store_true", help="删除旧的 ChromaDB collection")
    parser.add_argument("--verify", action="store_true", help="验证存储的记忆是否均为 LLM 提取的")
    parser.add_argument("--reimport", action="store_true", help="自动执行 parse_wechat.py --parse")
    parser.add_argument("--user-id", default="", type=str, help="联系人名称（verify/reimport 时必需）")
    args = parser.parse_args()

    if args.clean:
        logger.info("=== 清理旧数据 ===")
        clean_chromadb()

    if args.reimport:
        if not args.user_id:
            logger.warning("--reimport 需要 --user-id 参数")
            sys.exit(1)
        logger.info(f"=== 重新导入 (user_id={args.user_id}) ===")
        from parse_wechat import import_markdown_file
        from memory.memory_manager import DEFAULT_MEM0_CONFIG
        from mem0 import Memory

        memory = Memory.from_config(DEFAULT_MEM0_CONFIG)
        import_markdown_file("wechat_cleaned.md", user_id=args.user_id, memory=memory)

    if args.verify:
        if not args.user_id:
            logger.warning("--verify 需要 --user-id 参数")
            sys.exit(1)
        logger.info(f"=== 验证存储记忆 (user_id={args.user_id}) ===")
        verify_memories(args.user_id)

    if not any([args.clean, args.verify, args.reimport]):
        parser.print_help()
