from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = Field(..., description="PostgreSQL connection string")
    redis_url: str = Field(default="redis://localhost:6379/0")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    schema_config_path: str = Field(default="src/shipsentinel/schema/default.yaml")
    model_registry_name: str = Field(default="shipsentinel-lgbm")
    sla_breach_threshold: float = Field(default=0.5)


def get_settings() -> Settings:
    return Settings()
