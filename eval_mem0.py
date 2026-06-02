"""
Mem0 记忆检索评估脚本：Hit@K

用 search() 拉取用户的所有记忆（走 ChromaDB，数据完整），
构造测试集，评估真实 Query 能否命中对应记忆。
"""
import os
import sys
import random

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.memory_manager import MemoryManager

# ============================================================
# 配置
# ============================================================
USER_ID = "张三"
K = 3
SAMPLE_SIZE = 30  # 测试集大小

# ============================================================
# 1. 用 search() 拉取所有记忆（走 ChromaDB）
# ============================================================
print("=" * 60)
print(f"评估用户: {USER_ID}  Hit@{K}")
print("=" * 60)

mm = MemoryManager()
if not mm.mem0:
    print("Mem0 未初始化")
    sys.exit(1)

print("\n正在拉取所有记忆...")
resp = mm.mem0.search("", filters={"user_id": USER_ID}, top_k=200, threshold=0.0)
all_raw = resp.get("results", []) if isinstance(resp, dict) else (resp or [])

# 去重：同一段文本可能以 user/assistant 角色各存一次，只保留一次
seen = set()
all_memories = []
for m in all_raw:
    # print(f"原始记录: {m}")
    text = (m.get("memory") or "").strip()
    if not text or len(text) < 4:
        continue
    if text not in seen:
        seen.add(text)
        all_memories.append(text)

print(f"共获取到 {len(all_raw)} 条原始记录，去重后 {len(all_memories)} 条不同记忆\n")

if not all_memories:
    print("没有记忆数据，请先运行: python parse_wechat.py --parse --user-id", USER_ID)
    sys.exit(0)

# ============================================================
# 2. 构建测试集：从记忆自动生成 Query
# ============================================================
def build_query(text: str) -> str:
    """从记忆文本生成自然的聊天 Query"""
    t = text
    if len(t) <= 8:
        return t
    for kw, q in [
        ("喜欢", "你是不是喜欢"), ("觉得", "你是不是觉得"),
        ("认为", "你觉得"), ("计划", "你计划"), ("打算", "你打算"),
        ("想", "你想"), ("要去", "你要去"), ("去了", "你去"),
    ]:
        if kw in t:
            after = t.split(kw, 1)[1].strip().rstrip("。，,.")
            return f"{q}{after}？"
    return t


test_dataset = [(build_query(t), t) for t in all_memories]
if SAMPLE_SIZE and len(test_dataset) > SAMPLE_SIZE:
    test_dataset = random.sample(test_dataset, SAMPLE_SIZE)

print(f"测试集: {len(test_dataset)} 条")
print(f"  例如: '{test_dataset[0][0]}' → '{test_dataset[0][1]}'") if test_dataset else None
print()

# ============================================================
# 3. 评估 Hit@K
# ============================================================
hit_count = 0
misses = []

for q, expected in test_dataset:
    try:
        resp = mm.mem0.search(q, filters={"user_id": USER_ID}, top_k=K)
    except Exception as e:
        print(f"  搜索失败: {e}")
        continue

    results = resp.get("results", []) if isinstance(resp, dict) else (resp or [])
    retrieved = [r.get("memory", "") for r in results]

    # 判断：expected 中任意 >=5 字的片段是否出现在检索结果中
    def check(text, candidates):
        # 提取所有 5 字段落
        segs = {text[i:i+5] for i in range(len(text) - 4)}
        for c in candidates:
            for s in segs:
                if s in c:
                    return True
        return False

    hit = check(expected, retrieved)
    if hit:
        hit_count += 1
    else:
        misses.append((q, expected, retrieved))

    # 进度
    cur = len(misses) + hit_count
    if cur % 10 == 0 or cur == len(test_dataset):
        pct = hit_count / cur * 100
        print(f"  进度: {cur}/{len(test_dataset)} | 命中: {hit_count}/{cur} ({pct:.0f}%)")

# ============================================================
# 4. 结果
# ============================================================
hit_at_k = hit_count / len(test_dataset) if test_dataset else 0

print("\n" + "=" * 60)
print(f" Hit@{K} = {hit_at_k:.2%}")
print("=" * 60)
if hit_at_k < 0.5:
    print("⚠️ 检索率极低")
elif hit_at_k < 0.7:
    print("⚠️ 低于 70%，可能经常‘健忘’")
else:
    print("🎉 良好")

if misses:
    print(f"\n未命中样本 (共 {len(misses)} 条, 显示前 10):")
    for q, exp, ret in misses[:10]:
        print(f"  Q: {q}")
        print(f"  E: {exp}")
        print(f"  R: {(ret[0] if ret else '(空)')}")
        print()

# 报告
report = f"eval_report_{USER_ID}_hit@{K}.txt"
with open(report, "w", encoding="utf-8") as f:
    f.write(f"用户: {USER_ID}  测试集: {len(test_dataset)}  Hit@{K}: {hit_at_k:.2%}\n\n")
    for i, (q, e) in enumerate(test_dataset):
        hit_str = "HIT" if (q, e) not in [(m[0], m[1]) for m in misses] else "MISS"
        f.write(f"[{hit_str}] Q: {q}\n     E: {e}\n\n")
print(f"报告: {report}")
