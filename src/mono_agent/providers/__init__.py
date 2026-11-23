"""Chat model factory. A model is named "provider:model" (e.g. "ollama:qwen3:8b",
"anthropic:claude-haiku-5-5", "openrouter:qwen/qwen3-32b"), so each graph node can use a
different provider via config."""

import os

from langchain.chat_models import init_chat_model
from langchain_core.caches import BaseCache
from langchain_core.language_models import BaseChatModel

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def chat_model(spec: str, timeout: float = 120, cache: BaseCache | None = None) -> BaseChatModel:
    """`cache` lets the eval record and replay responses (eval/recording.py)."""
    model = _build(spec, timeout)
    if cache is not None:
        model.cache = cache
    return model


def _build(spec: str, timeout: float) -> BaseChatModel:
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        # Qwen3 "thinking" off: about 2x faster, and tool calls didn't need it in testing.
        # num_predict caps output: without it a looping model generates until the context is
        # full, and the timeout never fires because each streamed chunk resets it.
        # num_ctx 32k instead of the model's 262k: our prompts are ~12k tokens at most, and a
        # smaller KV cache is faster and uses ~8x less memory.
        return init_chat_model(
            spec,
            temperature=0,
            reasoning=False,
            num_predict=2048,
            num_ctx=32_768,
            client_kwargs={"timeout": timeout},
        )
    if provider == "anthropic":
        # Current Claude models reject non-default sampling parameters: no temperature.
        return init_chat_model(spec, timeout=timeout, max_retries=2, max_tokens=2048)
    if provider == "openrouter":
        # OpenRouter speaks the OpenAI API; one key gives access to many vendors' models.
        return init_chat_model(
            f"openai:{model}",
            base_url=OPENROUTER_URL,
            temperature=0,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            timeout=timeout,
            max_retries=2,
        )
    return init_chat_model(spec, temperature=0, timeout=timeout, max_retries=2)


REQUIRED_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def missing_key(spec: str) -> str | None:
    """Name of the env var this model needs but doesn't have, if any."""
    key = REQUIRED_KEYS.get(spec.partition(":")[0])
    return key if key and not os.environ.get(key) else None
