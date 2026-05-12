import base64
import sys
import pyautogui
import pygetwindow as gw
from PIL import ImageGrab
import time
import pyperclip
import json
import argparse
import logging
from pathlib import Path

# 配置
import os

# 导入项目配置
from config import WINDOW_CONFIG, DEBUG_CONFIG, CAPTURE_CONFIG, SEARCH_CONFIG

# ============ 日志配置 ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============ 核心函数 ============

def capture_chat_area(win):
    """
    截取聊天区域，跳过头像和标题栏
    """
    left = win.left + CAPTURE_CONFIG["left_offset"]
    top = win.top + CAPTURE_CONFIG["top_offset"]
    right = win.left + win.width
    bottom = win.top + win.height - CAPTURE_CONFIG["bottom_offset"]

    img = ImageGrab.grab(bbox=(left, top, right, bottom))
    img.save(DEBUG_CONFIG["screenshot_path"])
    logger.debug(f"已保存聊天截图: {DEBUG_CONFIG['screenshot_path']}")
    return img


def send_message_chinese(text: str):
    """发送中文消息（通过剪贴板粘贴）"""
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    pyautogui.press("enter")
    logger.info(f"已发送消息: {text[:20]}...")


def get_wechat_window(width: int = None, height: int = None):
    """
    获取微信窗口并设置为固定大小和位置

    Args:
        width: 窗口宽度，默认使用配置值
        height: 窗口高度，默认使用配置值

    Returns:
        微信窗口对象，失败返回 None
    """
    if width is None:
        width = WINDOW_CONFIG["width"]
    if height is None:
        height = WINDOW_CONFIG["height"]

    wins = gw.getWindowsWithTitle("微信")

    if not wins:
        logger.warning("未找到微信窗口")
        return None

    # 找到主微信窗口
    index = 0
    for i, win in enumerate(wins):
        if win.title == "微信":
            index = i
            break

    win = wins[index]
    logger.info(f"已获取微信窗口: {win}")

    # 1. 如果最小化了，先恢复
    if win.isMinimized:
        win.restore()
        logger.info("窗口已从最小化恢复")

    # 2. 激活窗口
    for attempt in range(3):
        try:
            win.activate()
            time.sleep(3)
            break
        except Exception as e:
            logger.warning(f"窗口激活尝试 {attempt + 1}/3 失败: {e}")

    # 3. 强制调整为固定大小
    win.resizeTo(width, height)
    logger.info(f"窗口已调整为 {width}x{height}")

    # 4. 强制移动到屏幕左上角 (0, 0)
    # 这样做可以让坐标计算变得极其稳定
    win.moveTo(0, 0)
    logger.info("窗口已移动到 (0, 0)")

    # 给窗口一点缓冲时间来完成动画
    time.sleep(1)

    return win


def search_and_click_contact(win, target_name: str) -> bool:
    """
    通过搜索框查找并点击联系人

    Args:
        win: 微信窗口对象
        target_name: 目标联系人名称

    Returns:
        是否成功找到并点击
    """
    # 搜索框在窗口中的绝对坐标（窗口固定在 0,0）
    search_box_x = win.left + SEARCH_CONFIG["x_offset"]
    search_box_y = win.top + SEARCH_CONFIG["y_offset"]
    
    # 1. 点击搜索框
    pyautogui.click(search_box_x, search_box_y)
    time.sleep(0.5)

    # 2. 清空搜索框（先全选，再删除）
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyautogui.press("delete")
    time.sleep(0.3)

    # 3. 输入联系人名称（通过剪贴板支持中文）
    pyperclip.copy(target_name)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(1)

    # 4. 按回车确认搜索
    pyautogui.press("enter")
    time.sleep(1)

    # 5. 点击第一个搜索结果
    # 搜索结果列表的第一个条目位置（相对于窗口）
    result_x = win.left + 200
    result_y = win.top + 100

    pyautogui.click(result_x, result_y)
    time.sleep(0.5)

    logger.info(f"已通过搜索框查找并点击联系人: {target_name}")
    return True


# ============ 主函数 ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="微信自动化 - 窗口操作")
    parser.add_argument(
        "--target", type=str, default="张三", help="目标联系人"
    )
    args = parser.parse_args()

    TARGET_USER = args.target

    logger.info(f"开始操作，目标联系人: {TARGET_USER}")

    win = get_wechat_window()
    if not win:
        sys.exit(1)

    # 2. 通过搜索框定位联系人
    # click_icon_tray()
    time.sleep(0.5)

    if not search_and_click_contact(win, TARGET_USER):
        logger.error(f"搜索框未找到 {TARGET_USER}")
        sys.exit(1)

    time.sleep(0.5)

    win = get_wechat_window()
    if not win:
        logger.error("未找到微信窗口")
        sys.exit(1)

    capture_chat_area(win)
    logger.info("操作完成")
