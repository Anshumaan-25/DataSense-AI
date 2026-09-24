"""Configuration management for DataSense AI."""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    # Base paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"

    # DuckDB Configuration
    duckdb_path: str = os.getenv("DUCKDB_PATH", str(BASE_DIR / "data" / "tpch.duckdb"))
    tpch_scale_factor: float = float(os.getenv("TPCH_SCALE_FACTOR", "0.1"))

    # LLM Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").lower()

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def get_effective_provider(self) -> str:
        """Resolve active LLM provider based on settings and available API keys."""
        if self.llm_provider in ("gemini", "google"):
            return "gemini" if self.gemini_api_key else "none"
        if self.llm_provider in ("openai", "chatgpt"):
            return "openai" if self.openai_api_key else "none"

        # Auto detection
        if self.gemini_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        return "none"


settings = Settings()
