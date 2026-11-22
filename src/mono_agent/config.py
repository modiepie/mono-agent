"""All runtime configuration, read from environment / .env."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr keeps the token out of repr() and logs.
    mono_token: SecretStr | None = None
    mono_base_url: str = "https://api.monobank.ua"

    data_dir: Path = Path("data")
    synthetic_seed: int = 42

    @property
    def use_real_api(self) -> bool:
        return self.mono_token is not None

    @property
    def cache_db(self) -> Path:
        return self.data_dir / "cache.sqlite"


def get_settings() -> Settings:
    return Settings()
