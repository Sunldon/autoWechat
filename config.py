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

# 聊天模型配置
CHAT_MODEL_CONFIG = {
    "api_base": os.getenv("CHAT_API_BASE", _config.get("chat_model", {}).get("api_base", "http://localhost:1234/v1")),
    "api_key": os.getenv("CHAT_API_KEY", _config.get("chat_model", {}).get("api_key", "lm-studio")),
    "model_name": os.getenv("CHAT_MODEL_NAME", _config.get("chat_model", {}).get("model_name", "qwen3.5-9b")),
    "max_tokens": _config.get("chat_model", {}).get("max_tokens", 2048),
    "temperature": _config.get("chat_model", {}).get("temperature", 0.7)
}

# 视觉模型配置（用于图片解析）
VISION_MODEL_CONFIG = {
    "api_base": os.getenv("VISION_API_BASE", _config.get("vision_model", {}).get("api_base", "http://localhost:1234/v1")),
    "api_key": os.getenv("VISION_API_KEY", _config.get("vision_model", {}).get("api_key", "lm-studio")),
    "model_name": os.getenv("VISION_MODEL_NAME", _config.get("vision_model", {}).get("model_name", "qwen3-vl-8b-instruct")),
    "max_tokens": _config.get("vision_model", {}).get("max_tokens", 4096),
    "temperature": _config.get("vision_model", {}).get("temperature", 0.3)
}

# 保持向后兼容性
MODEL_CONFIG = CHAT_MODEL_CONFIG  # 默认使用聊天模型

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

SEARCH_CONFIG = WECHAT_CONFIG.get("search_position", {
    "x_offset": 150,       # 搜索框相对于窗口左边的水平偏移
    "y_offset": 40,        # 搜索框相对于窗口顶部的垂直偏移
})

# 记忆模块配置（文件存储）
MEMORY_CONFIG = _config.get("memory", {
    "enabled": True,
    "window_size": 10,
    "llm": {
        "model": "qwen3-vl-8b-instruct",
        "openai_base_url": "http://localhost:1234/v1",
        "api_key": "not-needed",
    },
    "file_memory": {
        "path": "./memory_files",
        "max_lines": 60,
    },
    "search": {
        "hyde_max_chars": 3,
        "top_k": 5,
    },
})

# 环境变量覆盖记忆 LLM 配置
_mem_env_model = os.getenv("MEMORY_MODEL_NAME")
if _mem_env_model:
    MEMORY_CONFIG.setdefault("llm", {})["model"] = _mem_env_model
_mem_env_url = os.getenv("MEMORY_API_BASE")
if _mem_env_url:
    MEMORY_CONFIG.setdefault("llm", {})["openai_base_url"] = _mem_env_url
_mem_env_key = os.getenv("MEMORY_API_KEY")
if _mem_env_key:
    MEMORY_CONFIG.setdefault("llm", {})["api_key"] = _mem_env_key

# 调试配置
DEBUG_CONFIG = _config.get("debug", {
    "screenshot_path": "debug_last_msg.png"
})
