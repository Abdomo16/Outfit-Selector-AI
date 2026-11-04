import json
import os
from typing import List

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"


class AssistantError(Exception):
    """The model could not be reached or did not return usable JSON."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def complete_json(messages: List[dict], timeout: float = 20.0) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise AssistantError("OPENROUTER_API_KEY is not set")

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "Outfit Selector AI",
            },
            json={
                "model": os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(_strip_fences(content or ""))
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as error:
        raise AssistantError(str(error)) from error

    if not isinstance(parsed, dict):
        raise AssistantError("model returned JSON that is not an object")
    return parsed
