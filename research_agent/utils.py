from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

STOPWORDS = {
    "a", "an", "and", "the", "for", "of", "to", "in", "on", "with", "using",
    "based", "from", "via", "by", "into", "at", "is", "are", "be", "or", "as",
    "that", "this", "we", "our", "can", "will", "how", "what", "why", "when",
    "paper", "study", "method", "approach", "model", "models", "system", "systems"
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def slugify(text: str, max_len: int = 60) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^a-z0-9가-힣\s_-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:max_len] or "run"


def tokenize(text: str) -> List[str]:
    text = normalize_space(text).lower()
    tokens = re.findall(r"[a-zA-Z0-9\-\+]{2,}", text)
    return [t for t in tokens if t not in STOPWORDS]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_score(query_terms: List[str], title: str, abstract: str, extra_terms: List[str] | None = None) -> float:
    target_tokens = tokenize(title) + tokenize(abstract)
    if extra_terms:
        target_tokens += [t.lower() for t in extra_terms]
    return jaccard(query_terms, target_tokens)


def current_year() -> int:
    return datetime.now().year


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hash_key(parts: List[str]) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def clip(text: str, max_chars: int = 600) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def format_bullets(items: List[str]) -> str:
    return "\n".join([f"- {i}" for i in items]) if items else "- 없음"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}
