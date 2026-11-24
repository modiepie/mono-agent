"""All runtime configuration, read from environment / .env."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOCAL_MODEL = "ollama:qwen3.6:35b-a3b-q4_K_M"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr keeps the token out of repr() and logs.
    mono_token: SecretStr | None = None
    mono_base_url: str = "https://api.monobank.ua"

    data_dir: Path = Path("data")
    synthetic_seed: int = 42
    # The agent's "today". Synthetic data covers 2025, so it defaults to 2025-12-31 there.
    as_of: date | None = None

    # "provider:model" per node, as accepted by langchain's init_chat_model.
    router_model: str = LOCAL_MODEL
    agent_model: str = LOCAL_MODEL
    composer_model: str = LOCAL_MODEL
    # Used when the primary model of a node fails (timeout, rate limit, provider down).
    # Empty = no fallback. Example: "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free".
    fallback_model: str = ""

    # Retrieval (each stage switchable for the eval's ablation).
    qdrant_url: str = "http://localhost:6333"
    docs_dir: Path = Path("data/docs")
    embeddings: str = "local:BAAI/bge-m3"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    chunking: Literal["fixed", "structure"] = "structure"
    retrieval_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rerank: bool = True
    top_k: int = 5

    # Service
    log_level: str = "INFO"
    # Tracing is on when both keys are set (self-hosted Langfuse in docker compose, or cloud).
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "http://localhost:3000"

    @field_validator("mono_token", "langfuse_secret_key", mode="before")
    @classmethod
    def _blank_is_none(cls, v: object) -> object:
        # `MONO_TOKEN=` (as in .env.example) means "not set", not "an empty token".
        return None if isinstance(v, str) and not v.strip() else v

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def use_real_api(self) -> bool:
        return self.mono_token is not None

    @property
    def today(self) -> date:
        if self.as_of:
            return self.as_of
        return date.today() if self.use_real_api else date(2025, 12, 31)

    @property
    def cache_db(self) -> Path:
        # Separate files so synthetic data can never mix with real data.
        return self.data_dir / ("cache.sqlite" if self.use_real_api else "synthetic.sqlite")

    @property
    def checkpoint_db(self) -> Path:
        # Per mode too: a demo conversation must not continue on real data, or the reverse.
        return self.data_dir / (
            "checkpoints-real.sqlite" if self.use_real_api else "checkpoints.sqlite"
        )


def get_settings() -> Settings:
    return Settings()
