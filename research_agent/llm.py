from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .config import SETTINGS

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class LLMClient:
    def __init__(self) -> None:
        self.enabled = bool(SETTINGS.openai_api_key and OpenAI is not None)
        self.client = OpenAI(api_key=SETTINGS.openai_api_key) if self.enabled else None

    def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.client is None:
            return None

        response = self.client.responses.create(
            model=SETTINGS.openai_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(response, "output_text", "")
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None
