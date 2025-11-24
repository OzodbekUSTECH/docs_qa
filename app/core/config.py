from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
)


class Settings(BaseSettings):
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    ECHO: bool

    ALLOWED_ORIGINS: list[str] = ["*"]
    ALLOWED_HOSTS: list[str] = ["*"]
    
    LOG_LEVEL: int = logging.INFO
    
    DOCS_USERNAME: str = "admin"
    DOCS_PASSWORD: str = "admin"
    
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    
    DOC_AI_PROJECT_ID: str | None = "docsqa-478609"
    DOC_AI_LOCATION: str | None = "us"
    DOC_AI_PROCESSOR_ID: str | None = "c58c3016e8dd7dd0"
    
    @property
    def database_url(self):
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def sessions_db_path(self) -> Path:
        """Путь к файлу базы данных SQLite для сессий чата"""
        sessions_dir = Path("storage") / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir / "chat_sessions.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
