import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# 获取项目根目录
BASE_DIR = Path(__file__).parent


def load_yaml_config(filename: str) -> dict:
    """加载 YAML 配置文件"""
    config_path = BASE_DIR / filename
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# ============ 配置加载 ============

_config = load_yaml_config("config.yaml")

# 模型配置
MODEL_CONFIG = {
    "api_base": os.getenv("API_BASE", _config.get("model", {}).get("api_base", "http://localhost:1234/v1")),
    "api_key": os.getenv("API_KEY", _config.get("model", {}).get("api_key", "lm-studio")),
    "model_name": os.getenv("MODEL_NAME", _config.get("model", {}).get("model_name", "qwen3-vl-8b-instruct")),
    "max_tokens": _config.get("model", {}).get("max_tokens", 51200),
    "temperature": _config.get("model", {}).get("temperature", 0.85)
}

# 微信配置
WECHAT_CONFIG = _config.get("wechat", {})
CONTACTS = WECHAT_CONFIG.get("contacts", ["李四", "张三"])
WINDOW_CONFIG = WECHAT_CONFIG.get("window", {"width": 1000, "height": 800})
# 聊天区域截取配置
CAPTURE_CONFIG = WECHAT_CONFIG.get("capture", {
    "left_offset": 300,      # 左侧偏移（跳过联系人栏）
    "top_offset": 90,        # 顶部偏移（跳过标题栏）
    "bottom_offset": 205,   # 底部偏移（跳过输入栏）
})
MONITOR_CONFIG = WECHAT_CONFIG.get("monitor", {
    "base_sleep_time": 20,
    "max_sleep_time": 600,
    "idle_trigger_seconds": 600,
    "retry_attempts": 3,
})

# 用户配置（本人信息）
USER_CONFIG = WECHAT_CONFIG.get("user", {
    "name": "张三",       # 用户姓名
})

# 调试配置
DEBUG_CONFIG = _config.get("debug", {
    "screenshot_path": "debug_last_msg.png"
})