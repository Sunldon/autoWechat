from openai import OpenAI

from config import CHAT_MODEL_CONFIG, VISION_MODEL_CONFIG


def create_chat_client() -> OpenAI:
    return OpenAI(
        api_key=CHAT_MODEL_CONFIG["api_key"],
        base_url=CHAT_MODEL_CONFIG["api_base"],
    )


def create_vision_client() -> OpenAI:
    return OpenAI(
        api_key=VISION_MODEL_CONFIG["api_key"],
        base_url=VISION_MODEL_CONFIG["api_base"],
    )
