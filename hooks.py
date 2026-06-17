import re
import logging
from typing import Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class HookResult:
    def __init__(self, passed: bool, reply: str, reason: str = ""):
        self.passed = passed
        self.reply = reply
        self.reason = reason


class LengthHook:
    def __init__(self, max_chars: int = 15):
        self.max_chars = max_chars

    def check(self, reply: str, **kwargs) -> HookResult:
        if len(reply) > self.max_chars:
            logger.warning(f"[LengthHook] 超长: {len(reply)} > {self.max_chars}")
            return HookResult(False, reply[:self.max_chars], f"截断至{self.max_chars}字")
        return HookResult(True, reply)


class BannedWordsHook:
    def __init__(self, banned_words: Optional[list[str]] = None):
        self.banned_words = banned_words or []

    def check(self, reply: str, **kwargs) -> HookResult:
        for word in self.banned_words:
            if word in reply:
                logger.warning(f"[BannedWordsHook] 命中禁止词: {word}")
                return HookResult(False, reply, f"包含禁止词「{word}」")
        return HookResult(True, reply)


class FormatHook:
    AI_PATTERNS = [
        r"作为AI",
        r"作为一个人工智能",
        r"有什么可以帮您",
        r"请问有什么",
        r"我来帮你",
        r"我来为您",
        r"理解您的感受",
        r"我理解你",
    ]

    def check(self, reply: str, **kwargs) -> HookResult:
        for pattern in self.AI_PATTERNS:
            if re.search(pattern, reply):
                logger.warning(f"[FormatHook] AI腔命中: {pattern}")
                return HookResult(False, reply, f"AI腔: {pattern}")

        if re.search(r"```|^\{|^\[.*\]$", reply, re.MULTILINE):
            logger.warning(f"[FormatHook] 疑似代码/JSON格式")
            return HookResult(False, reply, "包含代码或JSON格式")

        return HookResult(True, reply)


class DuplicateHook:
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold
        self._recent_replies: dict[str, list[str]] = {}

    def check(self, reply: str, user_id: str = "", **kwargs) -> HookResult:
        recent = self._recent_replies.get(user_id, [])
        for prev in recent:
            ratio = SequenceMatcher(None, reply, prev).ratio()
            if ratio > self.similarity_threshold:
                logger.warning(f"[DuplicateHook] 与历史回复相似: {ratio:.2f}")
                return HookResult(False, reply, f"与近期回复重复(相似度{ratio:.0%})")

        if user_id:
            self._recent_replies.setdefault(user_id, []).append(reply)
            if len(self._recent_replies[user_id]) > 10:
                self._recent_replies[user_id].pop(0)

        return HookResult(True, reply)


class HookChain:
    def __init__(self, hooks: Optional[list] = None):
        self.hooks = hooks or []

    def add(self, hook):
        self.hooks.append(hook)
        return self

    def run(self, reply: str, **kwargs) -> HookResult:
        current = reply
        for hook in self.hooks:
            result = hook.check(current, **kwargs)
            if not result.passed:
                logger.info(f"[HookChain] 被 {type(hook).__name__} 拦截: {result.reason}")
                return result
            current = result.reply
        return HookResult(True, current)

    def run_with_retry(self, reply: str, max_retries: int = 0, **kwargs) -> HookResult:
        current = reply
        for attempt in range(max_retries + 1):
            result = self.run(current, **kwargs)
            if result.passed:
                return result
            if attempt < max_retries:
                logger.info(f"[HookChain] 第{attempt + 1}次拦截，等待重试")
                current = reply
        return result
