"""Configuration management for DataSense AI."""

import os
import math
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

    # Relative paths are anchored to the repository, independent of the launch directory.
    chroma_storage_path: Path = BASE_DIR / os.getenv("CHROMA_STORAGE_PATH", "data/chroma_storage")

    # LLM Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto").lower()

    # Gemini
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    gemini_timeout_seconds: float = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
    gemini_max_attempts: int = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
    gemini_thinking_level: str = os.getenv("GEMINI_THINKING_LEVEL", "low").lower()
    gemini_max_output_tokens: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
    gemini_requests_per_minute: float = float(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "4"))

    # Groq's OpenAI-compatible endpoint; local pacing stays below its free-tier caps.
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_timeout_seconds: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "120"))
    groq_max_attempts: int = int(os.getenv("GROQ_MAX_ATTEMPTS", "3"))
    groq_max_output_tokens: int = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "2048"))
    groq_requests_per_minute: float = float(os.getenv("GROQ_REQUESTS_PER_MINUTE", "6"))
    groq_tokens_per_minute: int = int(os.getenv("GROQ_TOKENS_PER_MINUTE", "7000"))

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def __post_init__(self):
        if not math.isfinite(self.gemini_timeout_seconds) or self.gemini_timeout_seconds <= 0:
            raise ValueError("GEMINI_TIMEOUT_SECONDS must be finite and positive.")
        if self.gemini_max_attempts < 1:
            raise ValueError("GEMINI_MAX_ATTEMPTS must be positive.")
        if self.gemini_max_output_tokens < 1:
            raise ValueError("GEMINI_MAX_OUTPUT_TOKENS must be positive.")
        if self.gemini_thinking_level not in {"auto", "minimal", "low", "medium", "high"}:
            raise ValueError("GEMINI_THINKING_LEVEL must be auto, minimal, low, medium, or high.")
        if not math.isfinite(self.gemini_requests_per_minute) or self.gemini_requests_per_minute < 0:
            raise ValueError("GEMINI_REQUESTS_PER_MINUTE must be finite and non-negative (0 disables pacing).")
        if not math.isfinite(self.groq_timeout_seconds) or self.groq_timeout_seconds <= 0:
            raise ValueError("GROQ_TIMEOUT_SECONDS must be finite and positive.")
        if self.groq_max_attempts < 1 or self.groq_max_output_tokens < 1:
            raise ValueError("GROQ_MAX_ATTEMPTS and GROQ_MAX_OUTPUT_TOKENS must be positive.")
        if not math.isfinite(self.groq_requests_per_minute) or self.groq_requests_per_minute < 0:
            raise ValueError("GROQ_REQUESTS_PER_MINUTE must be finite and non-negative.")
        if self.groq_tokens_per_minute < 0:
            raise ValueError("GROQ_TOKENS_PER_MINUTE must be non-negative (0 disables token pacing).")

    def get_effective_provider(self) -> str:
        """Resolve active LLM provider based on settings and available API keys."""
        if self.llm_provider in ("gemini", "google"):
            return "gemini" if self.gemini_api_key else "none"
        if self.llm_provider in ("openai", "chatgpt"):
            return "openai" if self.openai_api_key else "none"
        if self.llm_provider == "groq":
            return "groq" if self.groq_api_key else "none"

        # Auto detection
        if self.groq_api_key and not self.groq_api_key.startswith("your_"):
            return "groq"
        if self.gemini_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        return "none"


settings = Settings()
