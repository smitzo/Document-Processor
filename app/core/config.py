"""
Application Configuration
==========================
Reads environment variables (from .env or shell).
"""

from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google Generative AI (Gemini)
    google_api_key: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"

settings = Settings()
