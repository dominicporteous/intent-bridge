"""Pure normalization and OpenAI message parsing policies."""

from __future__ import annotations

import re
from typing import Any


def normalize_command(text: str) -> str:
    text = text.strip().lower().replace("â€™", "'")
    text = re.sub(r"[.!?,;:]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_search_text(value: Any) -> str:
    """Normalize identifiers and labels for deterministic fuzzy matching."""
    text = str(value or "").casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "input_text", "output_text"}
            and isinstance(item.get("text"), str)
        ]
        return " ".join(parts).strip()
    return ""


def extract_client_history(body: dict[str, Any], turns: int) -> list[dict[str, str]]:
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return []
    last_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict) and messages[index].get("role") == "user"
        ),
        None,
    )
    compact = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or index == last_user_index:
            continue
        role = message.get("role")
        value = message_text(message)
        if role in {"user", "assistant"} and value:
            compact.append({"role": role, "content": value})
    return compact[-(turns * 2) :]


def sanitise_spoken_response(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if not text:
        return text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"^[#>*\-]+\s*", "", text)
    text = re.sub(r"\s*\([^()]{0,160}\)", "", text)
    text = re.sub(r"\s*\[[^\[\]]{0,160}\]", "", text)
    text = text.replace("Â°C", " degrees Celsius").replace("Â°F", " degrees Fahrenheit")
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    limitations = (
        "i don't have access",
        "i do not have access",
        "through the available tools",
        "available home assistant tools",
        "unable to access",
        "cannot access",
    )
    if any(marker in text.casefold() for marker in limitations):
        return "I can't check that."
    text = re.split(
        r"\b(?:if you'd like|if you would like|if you want|you could|you can also|would you like)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,;:-")
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-") + "."
    return text
