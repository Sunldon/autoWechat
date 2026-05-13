import base64
import time
import json
import sys
import os
import logging
from typing import Optional
from difflib import SequenceMatcher
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 导入配置
from config import MODEL_CONFIG, DEBUG_CONFIG, USER_CONFIG
from parse_wechat import ChatKnowledgeBase

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.WARNING,  # 默认抑制第三方库 HTTP 请求等 INFO 日志
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 抑制第三方库无用日志（HTTP 请求、模型下载等）
for lib in ("httpx", "chromadb", "sentence_transformers", "urllib3", "requests"):
    logging.getLogger(lib).setLevel(logging.WARNING)

# ============ 初始化 ============
# 延迟初始化知识库（只在需要时才初始化）
kb = None

def get_kb():
    """获取知识库实例（延迟初始化）"""
    global kb
    if kb is None:
        kb = ChatKnowledgeBase()
    return kb

# 初始化 LangChain ChatOpenAI（兼容 LM Studio）
llm = ChatOpenAI(
    model=MODEL_CONFIG["model_name"],
    openai_api_base=MODEL_CONFIG["api_base"],
    openai_api_key=MODEL_CONFIG["api_key"],
    max_tokens=MODEL_CONFIG["max_tokens"],
    temperature=MODEL_CONFIG["temperature"],
)


# ============ LangChain Tools ============
@tool
def get_current_time() -> str:
    """
    获取当前日期和时间。

    Returns:
        当前时间字符串，格式为 YYYY-MM-DD HH:MM
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# 绑定工具到 LLM（Tool Calling）
llm_with_tools = llm.bind_tools([get_current_time])


# ============ 辅助函数 ============

def is_similar(a: str, b: str) -> bool:
    """返回 0-1 之间的相似度得分"""
    ratio = SequenceMatcher(None, a, b).ratio()
    logger.debug(f"相似度计算: {ratio:.2f}")
    return ratio > 0.5


def image_to_base64(image_path: str) -> str:
    """将本地图片路径转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ============ 消息解析（多模态） ============

def ai_get_messages() -> Optional[str]:
    """
    调用 Qwen3-VL-8B-Instruct-GGUF 多模态模型解析聊天截图
    返回 JSON 字符串
    """
    screenshot_path = DEBUG_CONFIG["screenshot_path"]

    if not os.path.exists(screenshot_path):
        logger.error(f"截图不存在: {screenshot_path}")
        return None

    base64_image = image_to_base64(screenshot_path)

    prompt = """# Role
你是一个幽默、直率且极具观察力的微信分析助手。

# Workflow
请按照以下步骤分析提供的图片：

## 第一步：视觉解析 (Visual Parsing)
1. 识别对话框的水平对齐方式（这是判定的最重要标准）：
   - 气泡整体靠右对齐，头像位于右侧，气泡颜色为绿色 -> "self" (自己)
   - 气泡整体靠左对齐，头像位于左侧，气泡颜色为白色 -> "other" (对方)
2. 识别顺序：从上到下，按照时间轴逻辑提取。
3. 文字提取：
   - 只提取气泡内的文本内容。
   - 忽略气泡上方的群昵称（如 "罗成"）和时间戳（如 "18:52"）。
4. 非文本处理：
   - 如果是表情包，简要描述其画面内容。
   - 如果是纯图片，简要描述图片内容。
   - 保留消息中的原生 emoji 符号。
5. 记住，在微信界面中，右侧气泡是你自己，左侧白色气泡是你的朋友。

## Output Format
严格按 JSON 数组格式返回，并且严格按照消息顺序返回，不要包含任何 Markdown 代码块标签或解释文字：
[
  {
    "text": "原始消息内容",
    "sender": "other/self"
  }
]
"""

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ]
        )
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        format_content = ""
        for item in json.loads(content):
            format_content += f"{item['sender']}: {item['text']}\n"
        logger.info(f"消息解析结果: {format_content}")
        return content
    except Exception as e:
        logger.error(f"消息解析失败: {e}")
        return None

# ============ 数字分身加载 ============
def load_digital_twin() -> dict:
    """加载数字分身内容"""
    base_path = "./person/"

    files_to_load = {
        "skill": "SKILL.md",
        "personality": "personality.md",
        "interaction": "interaction.md",
        "memory": "memory.md",
    }

    content = {}
    for key, filename in files_to_load.items():
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content[key] = f.read()
        else:
            content[key] = None
            logger.warning(f"数字分身文件不存在: {filename}")

    return content


def create_system_prompt(twin_content: dict, retrieved_memories: str = "", abandon: str = "") -> str:
    """
    优化后的系统提示词

    Args:
        twin_content: 基础设定
        retrieved_memories: 从向量数据库检索出来的"当年我曾回过：xxx"的内容
        abandon: 不良回复示例
    """
    # 读取系统提示词模板
    template_path = os.path.join(os.path.dirname(__file__), "system_prompt_template.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # 准备模板替换的变量
    user_name = USER_CONFIG['name']
    skill = twin_content.get('skill', '')
    personality_detail = "- 性格细节:" + twin_content.get('personality', '') if twin_content.get('personality', '') else ""
    interaction_habit = "- 说话习惯:" + twin_content.get('interaction', '') if twin_content.get('interaction', '') else ""
    memories = retrieved_memories if retrieved_memories else "暂无相关往事记录，请根据性格自行发挥。"
    abandon_reply = abandon if abandon else "暂无禁止回复"
    
    # 使用字符串格式化填充模板
    return template.format(
        user_name=user_name,
        skill=skill,
        personality_detail=personality_detail,
        interaction_habit=interaction_habit,
        retrieved_memories=memories,
        abandon_reply=abandon_reply
    )


# ============ 核心对话功能 ============
def chat_with_digital_twin(user_input: list, abandon: str = "", use_history: bool = True) -> str:
    # 加载数字分身
    twin_content = load_digital_twin()

    # 根据参数决定是否使用历史记录
    if use_history:
        # 从向量数据库检索记忆
        text = ""
        for msg in reversed(user_input):
            if msg["sender"] == "other":
                text += msg["text"] + " "
            else:
                break
        retrieved_memories = get_kb().query_context(text)
    else:
        retrieved_memories = ""

    system_prompt = create_system_prompt(twin_content, retrieved_memories, abandon)

    # 构建消息
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"聊天记录：{user_input}"),
    ]

    # 普通对话模式
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"对话生成失败: {e}")
        return "我刚才走神了，再说一遍？"

# ============ 重复检测 ============

def check_duplicate_messages(user_input: list, reply: str) -> bool:
    for msg in user_input:
        if msg["sender"] == "self" and is_similar(msg["text"], reply):
            return True
    return False

# ============ 主函数 ============
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("普通对话测试")
    print("=" * 50)

    import argparse
    parser = argparse.ArgumentParser(description="微信自动回复系统")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="不使用历史聊天记录进行回复"
    )
    args = parser.parse_args()

    screenshot_path = DEBUG_CONFIG["screenshot_path"]
    if os.path.exists(screenshot_path):
        model_output = ai_get_messages()

        if model_output:
            try:
                messages = json.loads(model_output)
                print("当前消息结构:")
                for msg in messages:
                    print(msg)

                reply = chat_with_digital_twin(messages, "我真服了", use_history=not args.no_history)
                print(f"回复: {reply}")

                is_duplicate = check_duplicate_messages(messages, "你好，去打球")
                print(f"是否有重复消息: {is_duplicate}")
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print("原始输出:", model_output)
    else:
        print(f"截图不存在: {screenshot_path}")

    sys.exit(0)