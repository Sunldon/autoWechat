"""
Mem0 记忆检索评估脚本：Hit@K

用 MemoryManager 的完整混合检索管道（HyDE + 向量检索 + BM25 + RRF + Reranker）
评估不同 K 值下的命中率。

用法:
  python eval_mem0.py                          # 默认评估
  python eval_mem0.py --user-id "张三"           # 指定用户
  python eval_mem0.py --sample 50 --k 1 3 5     # 50 条测试，评估 K=1,3,5
  python eval_mem0.py --no-llm-queries           # 仅用启发式生成 query，不调用 LLM
"""
import os
import sys
import random
import argparse
import contextlib
import io

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory_manager import MemoryManager
import config as _app_config


def search_quiet(mm, query, user_id, top_k):
    """执行搜索时压制 pipeline 的调试输出"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        # eval 使用较小的 search_top_k 以加速 reranker
        # use_hyde=False 因为 LM Studio 可能不可用
        result = mm.search(query, user_id=user_id, top_k=top_k, search_top_k=12, use_hyde=False)
    finally:
        sys.stdout = old_stdout
    return result


def parse_args():
    p = argparse.ArgumentParser(description="记忆检索 Hit@K 评估")
    p.add_argument("--user-id", default=None, help="联系人名称（默认从配置读取）")
    p.add_argument("--sample", type=int, default=50, help="测试集大小，0=全部")
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], help="评估的 K 值列表")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--no-llm-queries", action="store_true", help="不使用 LLM 生成 query")
    return p.parse_args()


def _load_user_id(args_user_id) -> str:
    if args_user_id:
        return args_user_id
    return _app_config.USER_CONFIG.get("name", "哈哈")


def _make_llm_client():
    mc = _app_config.MEMORY_CONFIG.get("llm", {})
    from openai import OpenAI
    client = OpenAI(
        base_url=mc.get("openai_base_url", "http://localhost:1234/v1"),
        api_key=mc.get("api_key", "not-needed"),
    )
    client.model = mc.get("model", "qwen3.5-9b")
    return client


def generate_query_llm(memory_text: str, llm_client) -> str | None:
    prompt = (
        "你是一名微信用户。以下是关于你的某条记忆：\n"
        f"【记忆】{memory_text}\n\n"
        "请生成一个简短的微信聊天问题或陈述，这个问题/语句最可能触发检索到这条记忆。\n"
        "要求：\n"
        "1. 用第一人称（我），像在跟朋友聊天\n"
        "2. 不超过 30 个字\n"
        "3. 只输出语句本身，不要任何解释\n\n"
        "语句："
    )
    try:
        resp = llm_client.chat.completions.create(
            model=llm_client.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.7,
        )
        q = resp.choices[0].message.content.strip().strip("'\"\u201c\u201d\u2018\u2019")
        if 3 <= len(q) <= 60:
            return q
    except Exception:
        pass
    return None


def generate_query_heuristic(text: str) -> str:
    t = text.strip().rstrip("。，,.!！")
    if len(t) <= 4:
        return t
    for kw, q_template in [
        ("喜欢", "你是不是喜欢{}"),
        ("觉得", "你是不是觉得{}"),
        ("认为", "你觉得{}"),
        ("计划", "你计划{}吗"),
        ("打算", "你打算{}吗"),
        ("想去", "你想去{}吗"),
        ("要吃", "你要吃{}吗"),
        ("想", "你想不想{}"),
        ("要去", "你要去{}吗"),
        ("去过", "你去过{}吗"),
        ("吃过", "你吃过{}吗"),
        ("看过", "你看过{}吗"),
        ("喜欢", "你喜欢{}吗"),
        ("是", "你是不是{}"),
        ("在", "你现在在{}吗"),
        ("有", "你有没有{}"),
    ]:
        if kw in t:
            after = t.split(kw, 1)[1].strip().rstrip("。，,.!！")
            return q_template.format(after)
    return f"我{t[:25]}是吧？"


def build_query(memory_text: str, use_llm: bool, llm_client=None) -> str:
    if use_llm and llm_client:
        q = generate_query_llm(memory_text, llm_client)
        if q:
            return q
    return generate_query_heuristic(memory_text)


def is_hit(expected: str, retrieved_list: list[dict], min_span: int = 8) -> bool:
    """判断 expected 是否出现在检索结果中（8 字连续片段匹配，避免误判）"""
    if not expected or not retrieved_list:
        return False

    retrieved_texts = [r.get("memory", "") for r in retrieved_list if isinstance(r, dict)]

    for ret in retrieved_texts:
        if not ret:
            continue
        if len(expected) < min_span:
            if expected in ret:
                return True
            continue
        spans = {expected[i:i + min_span] for i in range(len(expected) - min_span + 1)}
        for span in spans:
            if span in ret:
                return True

    return False


def print_config(mm: MemoryManager):
    sc = _app_config.SEARCH_CONFIG
    print(f"  混合检索:     {'开' if sc.get('hybrid_search', True) else '关'}")
    print(f"  BM25 权重:     {sc.get('bm25_weight', 0.3)}")
    print(f"  Reranker:      {'开' if sc.get('reranker', {}).get('enabled', False) else '关'}")
    print(f"  HyDE 跳过:     ≤{sc.get('hyde_max_chars', 3)} 个中文字符")
    print(f"  分数阈值:      {sc.get('score_threshold', 0.96)}")
    print(f"  最终 top_k:    {sc.get('top_k', 5)}")


def main():
    args = parse_args()
    user_id = _load_user_id(args.user_id)
    random.seed(args.seed)
    use_llm = not args.no_llm_queries

    print("=" * 60)
    print(f"评估用户: {user_id}")
    print("=" * 60)

    # ---- 1. 初始化（压制模型加载的输出） ----
    with contextlib.redirect_stdout(io.StringIO()):
        mm = MemoryManager()
    if not mm.mem0:
        print("Mem0 未初始化")
        sys.exit(1)

    llm_client = _make_llm_client() if use_llm else None

    print("\n--- 检索管道配置 ---")
    print_config(mm)

    # ---- 2. 拉取全部记忆 ----
    print("\n正在拉取所有记忆...")
    all_raw = mm.get_all_memories(user_id, top_k=1000)

    seen = set()
    all_memories = []
    for m in all_raw:
        text = (m.get("memory") or "").strip()
        if not text or len(text) < 4:
            continue
        if text not in seen:
            seen.add(text)
            all_memories.append(text)

    print(f"共获取到 {len(all_raw)} 条原始记录，去重后 {len(all_memories)} 条不同记忆")
    if not all_memories:
        print("没有记忆数据，请先运行: python parse_wechat.py --parse --user-id", user_id)
        sys.exit(0)

    # ---- 3. 构建测试集 ----
    test_dataset = [(build_query(t, use_llm, llm_client), t) for t in all_memories]
    if args.sample and len(test_dataset) > args.sample:
        test_dataset = random.sample(test_dataset, args.sample)

    print(f"\n测试集: {len(test_dataset)} 条")
    print(f"  例如: '{test_dataset[0][0][:30]}' → '{test_dataset[0][1][:30]}'")

    # ---- 4. 评估每个 K ----
    results: dict[int, dict] = {}

    for k in sorted(args.k):
        print(f"\n{'─' * 50}")
        print(f"Hit@{k}  评估中...")
        print(f"{'─' * 50}")

        hit_count = 0
        misses = []

        for i, (q, expected) in enumerate(test_dataset):
            try:
                retrieved = search_quiet(mm, q, user_id, top_k=k)
            except Exception as e:
                print(f"  搜索失败: {e}")
                continue

            hit = is_hit(expected, retrieved)
            if hit:
                hit_count += 1
            else:
                misses.append((q, expected, retrieved))

            cur = i + 1
            if cur % 10 == 0 or cur == len(test_dataset):
                pct = hit_count / cur * 100
                print(f"  进度: {cur}/{len(test_dataset)} | 命中: {hit_count}/{cur} ({pct:.0f}%)")

        hit_rate = hit_count / len(test_dataset) if test_dataset else 0
        results[k] = {
            "hit_count": hit_count,
            "total": len(test_dataset),
            "hit_rate": hit_rate,
            "misses": misses,
        }

        print(f"\n  Hit@{k} = {hit_count}/{len(test_dataset)} = {hit_rate:.2%}")

    # ---- 5. 汇总报告 ----
    print("\n" + "=" * 60)
    print("  汇总结果")
    print("=" * 60)
    for k in sorted(args.k):
        r = results[k]
        bar_len = 30
        filled = int(r["hit_rate"] * bar_len)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"  Hit@{k}  {bar}  {r['hit_count']:>3}/{r['total']:<3} = {r['hit_rate']:.2%}")

    best_k = max(results, key=lambda k: results[k]["hit_rate"])
    print(f"\n  最佳 K: {best_k} (Hit@{best_k} = {results[best_k]['hit_rate']:.2%})")

    # ---- 6. 未命中样本 ----
    worst_k = max(results, key=lambda k: len(results[k]["misses"]))
    misses = results[worst_k]["misses"]
    if misses:
        print(f"\n未命中样本 (Hit@{worst_k}, 共 {len(misses)} 条, 显示前 10):")
        for q, exp, ret in misses[:10]:
            print(f"  Q: {q}")
            print(f"  E: {exp}")
            first = (ret[0].get("memory", "") if isinstance(ret[0], dict) else str(ret[0])) if ret else "(空)"
            print(f"  R[0]: {first[:60]}")
            print()

    # ---- 7. 保存报告 ----
    report = f"eval_report_{user_id}.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"用户: {user_id}  测试集: {len(test_dataset)}\n")
        for k in sorted(args.k):
            f.write(f"Hit@{k}: {results[k]['hit_rate']:.2%} ({results[k]['hit_count']}/{results[k]['total']})\n")
        f.write(f"\n--- 详细记录 ---\n")
        for i, (q, e) in enumerate(test_dataset):
            tags = []
            for k in sorted(args.k):
                r = results[k]
                is_miss = any(m[0] == q and m[1] == e for m in r["misses"])
                tags.append("MISS" if is_miss else "HIT ")
            f.write(f"[{' '.join(tags)}] Q: {q}\n     E: {e}\n\n")
    print(f"详细报告: {report}")


if __name__ == "__main__":
    main()
