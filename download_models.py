from modelscope import snapshot_download
import os

if not os.path.exists('./models'):
    os.makedirs('./models')

models = [
    ("BAAI/bge-m3", "嵌入模型 (Embedding)"),
    ("BAAI/bge-reranker-v2-m3", "精排模型 (Cross-Encoder Reranker)"),
]

for model_id, desc in models:
    print(f"正在下载 {desc}: {model_id} ...")
    model_dir = snapshot_download(
        model_id,
        cache_dir='./models'
    )
    print(f"  ✓ 已下载到: {model_dir}")

print("\n所有模型下载完成")