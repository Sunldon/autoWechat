import base64
import json
import sys
import os
import logging
from typing import Optional
from difflib import SequenceMatcher
from datetime import datetime

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from sqlalchemy import text
import re

# 导入配置
from config import CHAT_MODEL_CONFIG, VISION_MODEL_CONFIG, DEBUG_CONFIG, USER_CONFIG
from parse_wechat import ChatKnowledgeBase

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 抑制第三方库日志
for lib in (
    "httpx",
    "chromadb",
    "sentence_transformers",
    "urllib3",
    "requests",
):
    logging.getLogger(lib).setLevel(logging.WARNING)

# ============ 初始化 ============
kb = None


def get_kb():
    """延迟初始化知识库"""
    global kb

    if kb is None:
        kb = ChatKnowledgeBase()

    return kb


# ============ 初始化 LLM ============
# 聊天模型（用于回复消息）
chat_llm = ChatOpenAI(
    model=CHAT_MODEL_CONFIG["model_name"],
    openai_api_base=CHAT_MODEL_CONFIG["api_base"],
    openai_api_key=CHAT_MODEL_CONFIG["api_key"],
    max_tokens=CHAT_MODEL_CONFIG["max_tokens"],
    temperature=CHAT_MODEL_CONFIG["temperature"],
)

# 视觉模型（用于解析图片）
vision_llm = ChatOpenAI(
    model=VISION_MODEL_CONFIG["model_name"],
    openai_api_base=VISION_MODEL_CONFIG["api_base"],
    openai_api_key=VISION_MODEL_CONFIG["api_key"],
    max_tokens=VISION_MODEL_CONFIG["max_tokens"],
    temperature=VISION_MODEL_CONFIG["temperature"],
)

# ============ Tools ============
@tool
def get_current_time() -> str:
    """
    获取当前日期和时间
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M")


@tool
def web_search(query: str) -> str:
    """搜索网络获取实时信息，当遇到不懂的名词、概念或需要最新资讯时使用"""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"未找到关于「{query}」的相关信息"
        formatted = "\n\n".join(
            f"[{i+1}] {r['title']}\n   {r['body']}"
            for i, r in enumerate(results)
        )
        return f"关于「{query}」的搜索结果:\n{formatted}"
    except Exception as e:
        return f"搜索「{query}」时遇到网络问题，已跳过搜索"


TOOLS = {
    "get_current_time": get_current_time,
    "web_search": web_search,
}

# 为聊天模型绑定工具
chat_llm_with_tools = chat_llm.bind_tools(list(TOOLS.values()))

# 为视觉模型也绑定相同的工具（如果需要）
vision_llm_with_tools = vision_llm.bind_tools(list(TOOLS.values()))

# ============ 辅助函数 ============
def is_similar(a: str, b: str) -> bool:
    """判断两句话是否相似"""

    ratio = SequenceMatcher(None, a, b).ratio()

    logger.debug(f"相似度: {ratio:.2f}")

    return ratio > 0.5


def image_to_base64(image_path: str) -> str:
    """图片转 base64"""

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ============ 多模态解析 ============
def ai_get_messages() -> Optional[str]:
    """
    调用多模态模型解析微信截图
    """

    screenshot_path = DEBUG_CONFIG["screenshot_path"]

    if not os.path.exists(screenshot_path):
        logger.error(f"截图不存在: {screenshot_path}")
        return None

    base64_image = image_to_base64(screenshot_path)

    prompt = """
# Role
你是一个微信聊天截图解析助手。

# Workflow

## 第一步：过滤非气泡文本（核心规则）
请首先扫描整张截图，**必须严格忽略**以下内容：
1. 居中显示的灰色时间戳（例如：16:31、16:37）。
2. 居中显示、没有气泡背景的灰色系统提示文本（例如：“你已添加了...”、“以上是打招呼的消息...”、“对方开启了朋友验证...”、“文本已撤回”等）。

## 第二步：识别气泡消息归属
仅对**拥有对话气泡**的文本进行识别，根据气泡的颜色和左右绝对位置判断：
- **other**：位于屏幕【左侧】的【白色气泡】，头像在左侧。
- **self**：位于屏幕【右侧】的【绿色气泡】，头像在右侧。

## 第三步：提取内容
1. 按照从上到下的时间顺序提取。
2. 只提取气泡内的纯文本。
3. 保留文本中的 emoji。
4. 若气泡内是图片或自定义表情包，简要描述或用 `[图片]` / `[表情]` 代替。

# Output Format
严格返回符合标准的 JSON 数组，禁止包含 Markdown 标记（如 ```json），禁止任何前后解释性文字：

[
  {
    "text": "消息内容",
    "sender": "self/other"
  }
]
"""

    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]
        )
    ]

    try:
        # 使用视觉模型进行图片解析
        response = vision_llm.invoke(messages)

        content = response.content.strip()

        format_content = ""

        for item in json.loads(content):
            format_content += f"{item['sender']}: {item['text']}\n"

        logger.info(f"消息解析结果:\n{format_content}")

        return content

    except Exception as e:
        logger.error(f"消息解析失败: {e}")
        return None


# ============ 数字分身 ============
def load_digital_twin() -> dict:
    """加载数字分身"""

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
            content[key] = ""
            logger.warning(f"数字分身文件不存在: {filename}")

    return content


# ============ Prompt ============
def create_system_prompt(
    twin_content: dict,
    retrieved_memories: str = "",
    abandon: str = "",
    advices: str = "",
) -> str:
    """
    构造系统提示词
    """

    template_path = os.path.join(
        os.path.dirname(__file__),
        "system_prompt_template_react.txt",
    )

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    user_name = USER_CONFIG["name"]

    skill = twin_content.get("skill", "")

    personality_detail = (
        "- 性格细节:\n" + twin_content.get("personality", "")
        if twin_content.get("personality", "")
        else ""
    )

    interaction_habit = (
        "- 说话习惯:\n" + twin_content.get("interaction", "")
        if twin_content.get("interaction", "")
        else ""
    )

    memories = (
        retrieved_memories  
        if retrieved_memories
        else "暂无相关往事记录，请根据性格发挥。"
    )

    abandon_reply = (
        abandon
        if abandon
        else "暂无禁止回复"
    )

    return template.format(
        user_name=user_name,
        skill=skill,
        personality_detail=personality_detail,
        interaction_habit=interaction_habit,
        retrieved_memories=memories,
        abandon_reply=abandon_reply,
        advices=advices,
    )


# ============ ReAct 执行器 ============
def invoke_react(messages: list, max_steps: int = 3, llm_model_with_tools=None) -> str:
    """
    ReAct Tool Calling 循环
    """
    
    # 如果没有指定模型，默认使用聊天模型
    if llm_model_with_tools is None:
        llm_model_with_tools = chat_llm_with_tools

    history = messages[:]

    for step in range(max_steps):
        logger.info(f"ReAct Step: {step + 1}")

        response = llm_model_with_tools.invoke(history)
        # 打印完整 AIMessage
        print("\n[AI RESPONSE]")
        print(response)

        tool_calls = getattr(response, "tool_calls", None)

        # 不需要工具
        if not tool_calls:
            # final_text = response.content.strip()
            # 解析最终回复文本
            if "Final:" in response.content:
                final_reply = response.content.split("Final:")[-1].strip()
            elif "Final" in response.content:
                final_reply = response.content.split("Final")[-1].strip()
            else:
                # 兜底：若未按格式输出，提取最后一行非空行
                lines = [line.strip() for line in response.content.split("\n") if line.strip()]
                final_reply = lines[-1]

            logger.info(f"react 回复: {final_reply}")
            return final_reply

        history.append(response)

        # 执行工具
        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})
            tool_call_id = call.get("id")

            logger.info(f"调用工具: {tool_name}")

            tool_func = TOOLS.get(tool_name)

            if tool_func is None:
                result = f"未知工具: {tool_name}"
            else:
                try:
                    result = tool_func.invoke(tool_args)
                except Exception as e:
                    result = f"工具调用失败: {e}"

            logger.info(f"工具返回: {result}")

            history.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                )
            )

    return "我刚卡了一下"

def reflect_and_refine(draft_reply: str, chat_text: str, abandon: str) -> str:
    """
    反思与修正模块：对初版回复进行人设、口语化、红线的二次审查
    """
    logger.info("=== 进入智能体反思阶段 ===")
    reflect_prompt = f"""
# Role
你是反思审查员。你需要站在真人的视角，用极度挑剔的眼光，严厉审视初版回复是否有一丝一毫的“AI味”、“书面公文感”或不符合微信号主人设的地方。

# 当前谈话背景（微信聊天记录）
{chat_text}

# 绝对禁止回复的方向/红线
{abandon if abandon else "暂无"}

# 待审查的初版回复草稿
"{draft_reply}"

# 反思检查清单 (Critique Checklist)
1. **AI腔调过滤**：有没有“作为AI...”、“请问有什么可以帮您”、“理解您的感受”等机械客套话？
2. **大模型说教味/小作文**：真人微信聊天讲究短小、随性、碎片化。初版回复是否太长、太严谨？
3. **书面语/公文感剔除（重点）**：检查是否使用了虽然通顺但偏书面的词汇（例如：将“刚忙完”写成“处理完事/处理事务”，将“没空”写成“时间不允许”）。真人微信聊天应极其口语化。
4. **红线审查**：是否提及了[绝对禁止回复]的内容？
5. **语气与上下文还原度**：是否完全符合前文的说话习惯、亲疏关系和性格？标点符号的使用是否自然（如：真人很少在微信短句末尾加句号）？

# 输出要求
**请注意：只要你觉得这句话“不够地道”、“像是一个AI在找借口/回复”，就请拒绝执行 PASS。**
请严格按照以下格式输出，并且不要添加任何多余的解释或文本：
[是否通过]: PASS / FAIL
[存在问题]：如果未通过，请详细指出初版回复中存在的所有问题
[改进建议]：如果未通过，请给出的改进建议
"""

    try:
        # 反思阶段不需要工具调用，直接使用基础 chat_llm 即可
        response = chat_llm.invoke([HumanMessage(content=reflect_prompt)])
        content = response.content.strip()
        logger.info(f"反思链完整输出:\n{content}")

        # 解析最终回复文本
        return content
    except Exception as e:
        logger.error(f"反思修正阶段抛出异常: {e}")
        return draft_reply  # 异常时兜底返回初版草稿


# ============ 核心对话 ============
def chat_with_digital_twin(
    user_input: list,
    abandon: str = "",
    use_history: bool = True,
) -> str:
    """
    数字分身聊天
    """

    twin_content = load_digital_twin()

    # ===== 记忆检索 =====
    if use_history:
        text = ""

        for msg in reversed(user_input):
            if msg["sender"] == "other":
                text += msg["text"] + " "
            else:
                break

        retrieved_memories = get_kb().query_context(text)

    else:
        retrieved_memories = ""

    try:
        advices = ""
        for i in range(3):
            logger.info(f"对话生成尝试 {i + 1}/3")
                # ===== Prompt =====
            system_prompt = create_system_prompt(
                twin_content,
                retrieved_memories,
                abandon,
                advices,
            )

            # ===== 聊天文本 =====
            chat_text = "\n".join(
                [
                    f'{msg["sender"]}: {msg["text"]}'
                    for msg in user_input
                ]
            )
            print(f"系统提示:\n{system_prompt}")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"聊天记录如下：\n{chat_text}"
                ),
            ]            
            reply = invoke_react(messages)
            reflect_reply = "PASS"
            # 2. 核心改动：接入反思链（剔除异常兜底词）
            if reply not in ["我刚卡了一下", "我刚才走神了，再说一遍？"]:
                reflect_reply = reflect_and_refine(reply, chat_text, abandon)
            
            # 使用正则匹配核心标签，兼容中英文冒号和换行
            pass_match = re.search(r"\[是否通过\][:：]\s*(.*)", reflect_reply)
            issues_match = re.search(r"\[存在问题\][:：]\s*([\s\S]*?)(?=\[改进建议\]|$)", reflect_reply)
            suggestions_match = re.search(r"\[改进建议\][:：]\s*([\s\S]*)", reflect_reply)

            # 提取并清洗数据（去掉首尾空格）
            is_pass = pass_match.group(1).strip() if pass_match else ""
            issues = issues_match.group(1).strip() if issues_match else ""
            suggestions = suggestions_match.group(1).strip() if suggestions_match else ""

            # ---- 打印结果测试 ----
            print(f"【是否通过】:\n{is_pass}\n" + "-"*30)
            print(f"【存在问题】:\n{issues}\n" + "-"*30)
            print(f"【改进建议】:\n{suggestions}")
            
            if is_pass == "PASS":
                logger.info("反思链审核通过，保持原回复")
                return reply.strip()
            advices = suggestions
            if i == 2:
                logger.warning("已达最大尝试次数，使用最后一次反思建议作为回复")
                if "Action" in reply:
                    logger.info("检测到回复中包含工具调用，保留原回复以确保功能执行")
                    return None
                return reply.strip()
        return None

    except Exception as e:
        logger.error(f"对话生成失败: {e}")
        return "我刚才走神了，再说一遍？"


# ============ 重复检测 ============
def check_duplicate_messages(
    user_input: list,
    reply: str,
) -> bool:
    """
    检测是否回复过类似内容
    """

    for msg in user_input:
        if (
            msg["sender"] == "self"
            and is_similar(msg["text"], reply)
        ):
            return True

    return False


# ============ Main ============
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("微信数字分身测试")
    print("=" * 50)

    import argparse

    parser = argparse.ArgumentParser(
        description="微信自动回复系统"
    )

    parser.add_argument(
        "--no-history",
        action="store_true",
        help="不使用历史聊天记录",
    )

    args = parser.parse_args()

    screenshot_path = DEBUG_CONFIG["screenshot_path"]

    if not os.path.exists(screenshot_path):
        print(f"截图不存在: {screenshot_path}")
        sys.exit(0)

    # ===== 解析截图 =====
    model_output = ai_get_messages()

    if not model_output:
        print("消息解析失败")
        sys.exit(0)

    try:
        messages = json.loads(model_output)
        messages[-1]["text"] = "去打球吗"
        messages[-1]["sender"] = "other"
        print("\n当前消息结构:")
        for msg in messages:
            print(msg)

        # ===== 回复 =====
        reply = chat_with_digital_twin(
            messages,
            abandon="现在几点了",
            use_history=not args.no_history,
        )

        print(f"\n回复: {reply}")

        # ===== 重复检测 =====
        duplicate = check_duplicate_messages(
            messages,
            reply,
        )

        print(f"重复检测: {duplicate}")

    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print("原始输出:")
        print(model_output)

    sys.exit(0)