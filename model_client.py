"""
model_client — auto-selects the best available inference backend.

Priority: MLX (Apple Silicon) → CUDA → MPS → CPU → OpenAI-compatible API

Usage:
    import model_client
    model_client.init_model()
    response = model_client.call_model(messages, max_tokens=20)
"""
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
MODEL_PATH   = os.getenv("MODEL_PATH")  # base model HF ID or local directory (optional)
ADAPTER_PATH = os.getenv("ADAPTER_PATH")  # LoRA adapter directory (optional, MLX only)

_client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)

_BACKEND         = "api"   # resolved by init_model()
_local_model     = None    # transformers model  (cuda / mps / cpu)
_local_tokenizer = None    # transformers tokenizer
_mlx_model       = None    # mlx_lm model        (Apple Silicon)
_mlx_tokenizer   = None    # mlx_lm tokenizer


def _detect_backend() -> str:
    if not MODEL_PATH and not ADAPTER_PATH:
        return "api"
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            return "mlx"
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def init_model() -> None:
    """Load the model on the best available backend. Call once at startup."""
    global _BACKEND, _local_model, _local_tokenizer, _mlx_model, _mlx_tokenizer

    _BACKEND = _detect_backend()
    if _BACKEND == "api":
        return

    print(f"[INFO] Backend: {_BACKEND}  model: {MODEL_PATH}", file=sys.stderr)

    if _BACKEND == "mlx":
        try:
            from mlx_lm import load as mlx_load
            base = MODEL_PATH or MODEL_NAME
            kwargs = {"adapter_path": ADAPTER_PATH} if ADAPTER_PATH else {}
            _mlx_model, _mlx_tokenizer = mlx_load(base, **kwargs)
            adapter_msg = f" + adapter {ADAPTER_PATH}" if ADAPTER_PATH else ""
            print(f"[INFO] MLX model ready: {base}{adapter_msg}", file=sys.stderr)
            return
        except Exception as e:
            print(f"[WARN] MLX load failed, falling back to CPU: {e}", file=sys.stderr)
            _BACKEND = "cpu"

    # Transformers path — cuda / mps / cpu
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dtype = torch.float16 if _BACKEND == "cuda" else torch.float32
        _local_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _local_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=dtype,
        ).to(_BACKEND)
        print(f"[INFO] Transformers model ready on {_BACKEND}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Failed to load local model: {e}", file=sys.stderr)


def call_model(messages: list, max_tokens: int = 20) -> str:
    """Route inference to the active backend: MLX → transformers → API."""

    if _BACKEND == "mlx" and _mlx_model is not None:
        from mlx_lm import generate as mlx_generate
        prompt = _mlx_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,              # disable Qwen3 chain-of-thought
        )
        raw = mlx_generate(
            _mlx_model, _mlx_tokenizer,
            prompt=prompt, max_tokens=max_tokens, verbose=False,
        )
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if not raw:
            raise ValueError("empty response from MLX model")
        return raw

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

    # OpenAI-compatible API — HF Inference Router, OpenAI, local vLLM, etc.
    response = _client.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        max_tokens=max_tokens, temperature=0.0,
    )
    return response.choices[0].message.content.strip()
