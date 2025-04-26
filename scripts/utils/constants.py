import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class APIProvider(Enum):
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    QWEN = "qwen"
    OPENAI = "openai"

@dataclass
class APIConfig:
    api_key: str
    base_url: Optional[str] = None
    model_name: str = ""
    http_options: Optional[Dict[str, Any]] = None

API_CONFIGS = {
    APIProvider.DEEPSEEK: APIConfig(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        model_name="deepseek-chat"
    ),
    APIProvider.GEMINI: APIConfig(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash-thinking-exp",
        http_options={'api_version': 'v1alpha'}
    ),
    APIProvider.QWEN: APIConfig(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus"
    ),
    APIProvider.OPENAI: APIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o"
    )
}

LANGS = {
    "AR": "ARABIC", "FI": "FINNISH", "FA": "FARSI", "EU": "BASQUE",
    "IT": "ITALIANO", "SV": "SWEDISH", "CA": "CATALAN", "CS": "CZECH",
    "EN": "ENGLISH", "DE": "GERMAN", "ES": "SPANISH", "ZH": "CHINESE",
    "HI": "HINDI", "FR": "FRENCH"
}

