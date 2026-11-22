"""Chat model factory. A model is named "provider:model" (e.g. "ollama:qwen3:8b",
"anthropic:claude-haiku-4-5"), so each graph node can use a different provider via config."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def chat_model(spec: str) -> BaseChatModel:
    provider = spec.split(":", 1)[0]
    extra = {}
    if provider == "ollama":
        extra["reasoning"] = False  # Qwen3 "thinking" off: slower, and not needed for tools
    return init_chat_model(spec, temperature=0, **extra)
