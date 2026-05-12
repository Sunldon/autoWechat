import re, time, os, random
import chromadb
from sentence_transformers import SentenceTransformer


# --- 1. 定义兼容 ChromaDB 的 Embedding 函数类 ---
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# --- 修改后的 Embedding 函数类 ---
class BgeM3EmbeddingFunction(EmbeddingFunction): # 继承基类
    def __init__(self, model_path='./models/bge-m3'):
        print(f"正在加载 Embedding 模型: {model_path}...")
        self.model = SentenceTransformer(model_path)

    def __call__(self, input_texts: Documents) -> Embeddings:
        # ChromaDB 传入的是 Documents 类型（本质是 List[str]）
        # 返回必须是 Embeddings 类型（本质是 List[List[float]]）
        embeddings = self.model.encode(input_texts)
        return embeddings.tolist()

class ChatKnowledgeBase:
    def __init__(self, db_path="./chat_db"):
        # 初始化本地数据库
        self.client = chromadb.PersistentClient(path=db_path)
        
        # --- 核心修改：切换到 sentence-transformers ---
        self.embedding_fn = BgeM3EmbeddingFunction()
        
        # 获取或创建 Collection，并指定 Embedding 函数
        # 注意：如果你之前已经用默认模型创建过这个 Collection，
        # 建议删除 chat_db 文件夹重新运行，因为向量维度和模型空间必须一致。
        self.collection = self.client.get_or_create_collection(
            name="wechat_history", 
            embedding_function=self.embedding_fn
        )

    def add_json_messages(self, messages):
        """将 AI 识别到的新消息存入数据库"""
        documents = []
        ids = []
        metadatas = []
        
        for msg in messages:
            content = msg['text']
            sender = msg['sender']
            
            documents.append(content)
            metadatas.append({
                "sender": sender
            })
            ids.append(f"msg_{int(time.time() * 1000)}_{random.randint(0,999)}")
            
        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    def query_context(self, current_query, n_results=5):
        """搜索记忆并返回匹配得分"""
        results = self.collection.query(
            query_texts=[current_query],
            n_results=n_results
        )
        
        # 打印调试信息，看看模型眼里的距离是多少
        # print(f"DEBUG - Distances: {results['distances'][0]}")

        formatted_memories = []
        
        if results['documents'] and len(results['documents'][0]) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            dist = results['distances'][0]  # 获取距离列表

            for i in range(len(docs)):
                score = 1 - dist[i]  # 简单转换成直观的“相似分数”
                reply = metas[i].get('my_reply', '')
                
                # 过滤：如果相似度太低（距离太大），就不放进提示词了
                print(f"DEBUG - Score: {score:.2f}, Distance: {dist[i]:.4f} docs: {docs[i]} Reply: {reply}")
                if dist[i] < 0.5: 
                    formatted_memories.append(f"{reply}")

        return "\n".join(formatted_memories) if formatted_memories else ""

# --- 以下解析函数保持不变，但确保调用时逻辑正确 ---

def parse_wechat_markdown(md_text):
    # 匹配格式：self: 24 或 other: 消息内容
    md_text = md_text.strip()  # 移除首尾空白
    pattern = r"^(self|other):\s*(.*)$"  # 添加行首行尾锚点
    match = re.match(pattern, md_text)
    if match:
        role_raw, content = match.groups()
        role = role_raw  # 直接使用匹配到的self或other
        return {
            "clean_text": content,
            "metadata": {"sender": role}
        }
    print(f"解析失败 - 原始文本: [{repr(md_text)}]")
    return None

def add_to_knowledge_base(kb, md_line, line_num=None):
    parsed = parse_wechat_markdown(md_line)
    if not parsed:
        return False

    msg_id = f"md_{line_num}" if line_num is not None else f"msg_{int(time.time() * 1000)}"
    
    # upsert 内部会自动调用 kb.embedding_fn 将 clean_text 转化为向量
    kb.collection.upsert(
        documents=[parsed['clean_text']],
        metadatas=[parsed['metadata']],
        ids=[msg_id]
    )
    return True

def import_markdown_file(kb, file_path):
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    print(f"总共读取到 {len(lines)} 行 Markdown 记录, 开始解析...")
    added_count = 0
    # 遍历每一行，寻找“对方说”紧跟“我回”的结构
    for i in range(len(lines) - 1):
        current_line = lines[i]
        next_line = lines[i+1]
        # print(f"DEBUG - Line {i}: {current_line}")
        # print(f"DEBUG - Line {i+1}: {next_line}")   
        # 匹配模式：当前行是【对方】，下一行是【我】
        if "other" in current_line and "self" in next_line:
            parsed_other = parse_wechat_markdown(current_line)
            parsed_self = parse_wechat_markdown(next_line)
            
            if parsed_other and parsed_self:
                # --- 关键修改：存储逻辑 ---
                # 我们以“对方的话”作为 document（用于被搜索）
                # 把“我的回复”存入 metadata（用于被提取）
                kb.collection.upsert(
                    documents=[parsed_other['clean_text']], 
                    metadatas=[{
                        "my_reply": parsed_self['clean_text']
                    }],
                    ids=[f"pair_{i}"]
                )
                added_count += 1
    
    print(f"成功导入 {added_count} 组 [问答对] 记忆")

def debug_random_sample(db_path="./chat_db", count=3):
    """
    高效随机抽检：只从数据库中读取指定数量的记录
    """
    client = chromadb.PersistentClient(path=db_path)
    try:
        # 必须指定 embedding_function，否则无法正确读取数据
        collection = client.get_collection(
            name="wechat_history", 
            embedding_function=BgeM3EmbeddingFunction()
        )
    except Exception as e:
        print(f"❌ 获取 Collection 失败: {e}")
        return

    # 1. 先获取数据库里的总记录数
    total_count = collection.count()
    if total_count == 0:
        print("Empty Database.")
        return

    # 2. 随机生成一个起始索引 (Offset)
    # 确保 offset + count 不会超过总数
    max_offset = max(0, total_count - count)
    random_offset = random.randint(0, max_offset)

    # 3. 仅查询这一小部分数据
    results = collection.get(
        limit=count,
        offset=random_offset,
        include=['documents', 'metadatas']
    )

    print(f"\n{'='*15} 随机抽检 (总数: {total_count}) {'='*15}")
    
    docs = results.get('documents', [])
    metas = results.get('metadatas', [])

    for i in range(len(docs)):
        other_say = docs[i]
        my_reply = metas[i].get('my_reply', 'N/A')
        
        print(f" 👤 对方: {other_say}")
        print(f" 🤖 我回: {my_reply}")
        print("-" * 40)

if __name__ == "__main__":
    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser()
    # 是否解析微信记录
    parser.add_argument("--parse", default=False, action="store_true")
    # 测试文本
    parser.add_argument("--test", default="test", type=str)

    args = parser.parse_args()
    # 初始化
    kb = ChatKnowledgeBase()

    if args.parse:
        import_markdown_file(kb, "wechat_cleaned.md")

    print("test query:", args.test)
    context = kb.query_context(args.test)
    print(f"--- 检索到的相关记忆 ---\n{context}")