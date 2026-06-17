from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import CHAT_MODEL_CONFIG
from logger import get_logger
logger = get_logger(__name__)


class ShortTermMemory:
    """短期记忆：滑动窗口 + 摘要压缩

    每个 user_id 维护一个最近 N 轮对话的滑动窗口。
    当窗口溢出时，调用 LLM 将早期对话压缩为摘要。
    检索时拼接: [摘要] + [最近 N 轮原始消息]
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        # user_id -> list[dict]，每个 dict 含 "sender" 和 "text"
        self._windows: dict[str, list[dict]] = {}
        # user_id -> 摘要字符串
        self._summaries: dict[str, str] = {}
        # 用于摘要的 LLM（延迟初始化）
        self._summary_llm: Optional[ChatOpenAI] = None

    def _get_llm(self) -> ChatOpenAI:
        if self._summary_llm is None:
            self._summary_llm = ChatOpenAI(
                model=CHAT_MODEL_CONFIG["model_name"],
                openai_api_base=CHAT_MODEL_CONFIG["api_base"],
                openai_api_key=CHAT_MODEL_CONFIG["api_key"],
                max_tokens=512,
                temperature=0.3,
            )
        return self._summary_llm

    def update(self, messages: list[dict], user_id: str):
        """更新短期记忆窗口

        Args:
            messages: 当前轮次的对话消息列表，格式 [{"sender": "self"/"other", "text": "..."}]
            user_id: 联系人名称
        """
        if user_id not in self._windows:
            self._windows[user_id] = []
            self._summaries[user_id] = ""

        self._windows[user_id].extend(messages)
        logger.debug(
            f"短期记忆更新 [{user_id}]: 当前窗口 {len(self._windows[user_id])} 条消息"
        )

        # 当窗口大小超过阈值（2 倍窗口大小）时压缩
        if len(self._windows[user_id]) > self.window_size * 2:
            self._compress(user_id)

    def _compress(self, user_id: str):
        """压缩早期对话为摘要"""
        overflow = self._windows[user_id][: -self.window_size]
        self._windows[user_id] = self._windows[user_id][-self.window_size :]

        # 将溢出内容格式化为文本
        overflow_lines = [f'{m["sender"]}: {m["text"]}' for m in overflow]
        overflow_text = "\n".join(overflow_lines)

        # 如果已有摘要，一起压缩
        previous_summary = self._summaries.get(user_id, "")
        if previous_summary:
            overflow_text = f"之前摘要：{previous_summary}\n新内容：{overflow_text}"

        summary = self._llm_summarize(overflow_text, user_id)
        self._summaries[user_id] = summary
        logger.info(
            f"短期记忆压缩完成 [{user_id}]: {summary[:60]}..."
        )

    def _llm_summarize(self, text: str, user_id: str) -> str:
        """调用 LLM 压缩文本为摘要"""
        prompt = (
            f"请将以下与 [{user_id}] 的聊天记录浓缩为一段简短的摘要（100字以内），"
            f"保留关键事实、话题和人物信息：\n\n{text}\n\n摘要："
        )
        try:
            response = self._get_llm().invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"摘要压缩失败: {e}")
            # 兜底：截取前 100 字
            return text[:100]

    def get_summary(self, user_id: str) -> str:
        """获取当前窗口的完整上下文（摘要 + 最近消息）

        Returns:
            格式化的上下文文本，供注入 system prompt 使用
        """
        summary = self._summaries.get(user_id, "")
        recent = self._windows.get(user_id, [])

        parts = []
        if summary:
            parts.append(f"[近期对话摘要] {summary}")
        if recent:
            recent_lines = [f'{m["sender"]}: {m["text"]}' for m in recent]
            parts.append("[最近对话]\n" + "\n".join(recent_lines))

        return "\n\n".join(parts)

    def clear(self, user_id: str):
        """清除指定用户的短期记忆"""
        self._windows.pop(user_id, None)
        self._summaries.pop(user_id, None)

    def clear_all(self):
        """清除所有用户的短期记忆"""
        self._windows.clear()
        self._summaries.clear()
