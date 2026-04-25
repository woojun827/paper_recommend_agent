from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .utils import ensure_dir

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4")
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    openalex_api_key: str = os.getenv("OPENALEX_API_KEY", "")
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "ko")
    cache_dir: Path = ensure_dir(os.getenv("CACHE_DIR", ".cache"))
    timeout_seconds: int = int(os.getenv("TIMEOUT_SECONDS", "25"))
    top_n_from_heuristic: int = int(os.getenv("TOP_N_FROM_HEURISTIC", "12"))


SETTINGS = Settings()
