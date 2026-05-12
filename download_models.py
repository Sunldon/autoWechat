from modelscope import snapshot_download
import os

if not os.path.exists('./models'):
    os.makedirs('./models')

# 下载 BAAI/bge-m3 模型到指定目录
model_dir = snapshot_download(
    'BAAI/bge-m3', 
    cache_dir='./models'
)

print(f"模型已成功下载到: {model_dir}")