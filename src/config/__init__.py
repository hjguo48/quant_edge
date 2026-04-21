from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLFLOW_DB_PATH = REPO_ROOT / "mlflow.db"
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{DEFAULT_MLFLOW_DB_PATH.as_posix()}"
DEFAULT_MLFLOW_ARTIFACT_ROOT = REPO_ROOT / "mlartifacts"


class Settings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str = "quantedge"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    MLFLOW_TRACKING_URI: str = DEFAULT_MLFLOW_TRACKING_URI
    MLFLOW_ARTIFACT_ROOT: str = str(DEFAULT_MLFLOW_ARTIFACT_ROOT)

    POLYGON_API_KEY: str = ""
    POLYGON_S3_KEY: str = ""
    POLYGON_S3_SECRET: str = ""
    POLYGON_S3_ENDPOINT: str = "https://files.massive.com"
    POLYGON_S3_BUCKET: str = "flatfiles"
    FRED_API_KEY: str = ""
    FMP_API_KEY: str = ""

    TRAIN_WINDOW_YEARS: int = 3
    VAL_WINDOW_MONTHS: int = 6
    TEST_WINDOW_MONTHS: int = 6
    ROLL_FORWARD_MONTHS: int = 6
    REBALANCE_FREQUENCY: str = "weekly"
    UNIVERSE_REBALANCE_FREQUENCY: str = "monthly"

    MAX_SINGLE_STOCK_WEIGHT: float = 0.10
    MAX_SECTOR_DEVIATION: float = 0.10
    BETA_RANGE: tuple[float, float] = (0.8, 1.2)
    MAX_WEEKLY_TURNOVER: float = 0.40
    MIN_HOLDINGS: int = 20
    CVAR_THRESHOLD: float = -0.05

    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    ENABLE_TRADE_MICROSTRUCTURE_FEATURES: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @property
    def database_url(self) -> str:
        user = quote_plus(self.POSTGRES_USER)
        password = quote_plus(self.POSTGRES_PASSWORD)
        return (
            f"postgresql://{user}:{password}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def async_database_url(self) -> str:
        user = quote_plus(self.POSTGRES_USER)
        password = quote_plus(self.POSTGRES_PASSWORD)
        return (
            f"postgresql+asyncpg://{user}:{password}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
