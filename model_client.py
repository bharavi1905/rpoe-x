from __future__ import annotations

import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen3-1.7B")
HF_TOKEN     = os.getenv("HF_TOKEN")    # NO default — must be set externally
MODEL_PATH   = os.getenv("MODEL_PATH")  # local model directory (optional)

_client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)

_local_model     = None
_local_tokenizer = None


def init_model() -> None:
    """Load a local transformers model if MODEL_PATH is set, otherwise use API."""
    global _local_model, _local_tokenizer

    if not MODEL_PATH:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        _local_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _local_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=dtype,
        ).to(device)
        print(f"[INFO] Local model ready on {device}: {MODEL_PATH}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to load local model: {e}", file=sys.stderr)


def call_model(messages: list, max_tokens: int = 20) -> str:
    """Run inference via local transformers model or OpenAI-compatible API."""

    if _local_model is not None and _local_tokenizer is not None:
        import torch
        inputs = _local_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            enable_thinking=False,
        ).to(_local_model.device)
        with torch.no_grad():
            outputs = _local_model.generate(
                **inputs,
                max_new_tokens=max_tokens * 2,
                do_sample=False,
                pad_token_id=_local_tokenizer.eos_token_id,
            )
        raw = _local_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        ).strip()
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    response = _client.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        max_tokens=max_tokens, temperature=0.0,
    )
    return response.choices[0].message.content.strip()
