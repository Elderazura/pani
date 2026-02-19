"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from .env. Never hardcode API keys."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Binance API
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET: str = ""
    BINANCE_TESTNET: bool = True

    # Local LLM (Ollama)
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b"

    # Cloud LLM
    CLOUD_LLM_PROVIDER: str = "anthropic"
    CLOUD_LLM_API_KEY: str = ""
    CLOUD_LLM_MODEL: str = "claude-haiku-4-5-20251001"

    # LLM preference
    USE_LOCAL_LLM: bool = True

    # Trading mode
    DRY_RUN: bool = True

    # Risk parameters
    MIN_VOLUME_USD: float = 100_000_000
    MAX_LEVERAGE: int = 3
    STOP_LOSS_PCT: float = 1.5
    TAKE_PROFIT_PCT: float = 4.5
    DAILY_LOSS_LIMIT_PCT: float = 3.0
    MAX_OPEN_POSITIONS: int = 3

    # Persistence
    DB_PATH: str = "data/futures_intel.duckdb"


def get_settings() -> Settings:
    """Return application settings. Loads from .env on first call."""
    return Settings()
