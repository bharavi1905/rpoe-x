from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")    # NO default — must be set externally

_client = OpenAI(
    api_key  = HF_TOKEN or "sk-placeholder",
    base_url = API_BASE_URL,
)


def call_model(messages: list, max_tokens: int = 20) -> str:
    response = _client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = 0.0,
    )
    return response.choices[0].message.content.strip()
