import traceback
import sys
import os

# ============ 强制离线：必须在 HuggingFace 库导入前设置 ============
os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import json
import argparse
from datetime import datetime

from logger import setup_logger, get_logger
setup_logger()
logger = get_logger(__name__)

from operate_wechat import (
    get_wechat_window,
    search_and_click_contact,
    capture_chat_area,
    send_message_chinese,
)
from ai_analyse import (
    ai_get_messages,
    chat_with_digital_twin,
    check_duplicate_messages,
)

from config import CONTACTS, MONITOR_CONFIG

# ============ 记忆模块初始化 ============
from memory.memory_manager import MemoryManager
memory_manager = MemoryManager()

# ============ 命令行参数解析 ============
parser = argparse.ArgumentParser(description="微信自动回复系统")
parser.add_argument(
    "--no-history",
    action="store_true",
    help="不使用历史聊天记录进行回复"
)
args = parser.parse_args()

# 全局变量：是否使用历史记录
USE_HISTORY = not args.no_history


# ============ 自定义异常 ============

class WechatOperationError(Exception):
    """微信操作异常"""
    pass


class MessageParseError(Exception):
    """消息解析异常"""
    pass


# ============ 主流程 ============

def main_flow():
    """
    微信自动化主流程

    遍历配置的联系人列表，监听新消息并自动回复
    """
    # 初始化状态
    last_msg_cache = {contact: None for contact in CONTACTS}
    sleep_time = MONITOR_CONFIG["base_sleep_time"]
    idle_time = 0
    reset = False
    cycle_count = 0

    logger.info("=" * 50)
    logger.info("微信自动化启动")
    logger.info(f"监控联系人: {CONTACTS}")
    logger.info("=" * 50)

    while True:
        cycle_count += 1
        TARGET_USER = ""  # 默认第一个
        need_reset = False
        for _idx, target_user in enumerate(CONTACTS):
            TARGET_USER = target_user
            logger.info(f"\n--- 轮次 {cycle_count} | 正在处理: {TARGET_USER} ---")

            try:
                win = get_wechat_window()
                if not win:
                    logger.warning("未获取到微信窗口，跳过当前联系人")
                    continue

                if not search_and_click_contact(win, TARGET_USER):
                    logger.error(f"搜索框未找到 {TARGET_USER}")
                    sys.exit(1)

                win = get_wechat_window()
                if not win:
                    logger.warning("获取窗口失败")
                    continue

                capture_chat_area(win)

                model_output = ai_get_messages()
                if not model_output:
                    logger.warning("消息解析返回为空")
                    time.sleep(MONITOR_CONFIG["base_sleep_time"])
                    continue

                try:
                    messages = json.loads(model_output)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {e}")
                    logger.debug(f"原始输出: {model_output[:200]}...")
                    time.sleep(MONITOR_CONFIG["base_sleep_time"])
                    continue

                # 打印消息结构
                logger.info(f"消息数量: {len(messages)}")
                for msg in messages:
                    sender = "我" if msg.get("sender") == "self" else "对方"
                    text = msg.get("text", "")[:30]
                    logger.debug(f"  [{sender}]: {text}")

                need_reply = messages[-1].get("sender", "") == "other"

                if need_reply and last_msg_cache[TARGET_USER] != messages[-1]["text"]:
                    logger.info(f"检测到新消息，需要回复")
                    need_reset = True
                    # 重试机制：最多尝试3次
                    last_reply = ""
                    reply_success = False

                    for attempt in range(MONITOR_CONFIG["retry_attempts"]):
                        reply = chat_with_digital_twin(
                            messages,
                            last_reply,
                            use_history=USE_HISTORY,
                            memory_manager=memory_manager,
                            user_id=TARGET_USER,
                        )
                        if reply is None:
                            continue
                        # 检查是否重复
                        if check_duplicate_messages(messages, reply) and attempt < MONITOR_CONFIG["retry_attempts"] - 1:
                            logger.info(f"回复重复，第 {attempt + 1} 次重试")
                            last_reply = reply
                            continue

                        # 发送回复
                        try:
                            send_message_chinese(reply)
                            logger.info(f"已发送回复: {reply[:30]}...")
                            reply_success = True
                            break
                        except Exception as e:
                            logger.error(f"发送消息失败: {e}")
                            last_reply = reply

                    if reply_success:
                        last_msg_cache[TARGET_USER] = messages[-1]["text"]
                        reset = True

                else:
                    idle_time += min(sleep_time, MONITOR_CONFIG["max_sleep_time"])
                    logger.info(f"空闲中，累积空闲时间: {idle_time}秒")

                # 每轮结束休眠
                time.sleep(10)

            # ========== 异常处理 ==========
            except Exception as e:
                error_type = type(e).__name__
                logger.error(f"[{error_type}] 操作异常: {e}")

                # 根据异常类型做不同处理
                if "PyGetWindowException" in error_type or "Win32Exception" in error_type:
                    # 窗口异常，可能是窗口被关闭或最小化，尝试恢复
                    logger.warning("窗口异常，尝试恢复...")
                    time.sleep(5)
                    continue
                elif "PermissionError" in error_type or "Access" in str(e):
                    # 权限异常，可能被其他程序占用
                    logger.warning("权限异常，等待后重试...")
                    time.sleep(10)
                    continue
                elif "TimeoutError" in error_type or "timeout" in str(e).lower():
                    # 超时异常，网络或响应慢
                    logger.warning("操作超时，等待后重试...")
                    time.sleep(15)
                    continue
                else:
                    # 其他异常，打印堆栈后继续
                    logger.exception("未预期的异常")
                    time.sleep(MONITOR_CONFIG["base_sleep_time"])
                    continue
        
        if need_reset:
            sleep_time = min(sleep_time * 2, MONITOR_CONFIG["max_sleep_time"])

        # ========== 循环结束：重置状态 ==========
        if reset:
            logger.info("一轮结束，重置状态")
            idle_time = 0
            sleep_time = MONITOR_CONFIG["base_sleep_time"]
            reset = False

        logger.info(f"等待 {sleep_time} 秒后开始下一轮...")
        time.sleep(sleep_time)


# ============ 入口 ============

if __name__ == "__main__":
    try:
        main_flow()
    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"致命异常，程序退出: {e}")
        sys.exit(1)