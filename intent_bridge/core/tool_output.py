"""Integration-neutral normalization of agent tool outputs."""

import json
from typing import Any


def serialise_tool_output(output: Any) -> str:
    if output is None:
        return ""
    if hasattr(output, "model_dump"):
        try:
            output = output.model_dump()
        except Exception:
            pass
    if isinstance(output, (dict, list, tuple)):
        try:
            return json.dumps(output, ensure_ascii=False, default=str)
        except Exception:
            return str(output)
    return str(output)


def tool_output_mapping(output: Any) -> dict | None:
    if output is None:
        return None
    if hasattr(output, "model_dump"):
        try:
            output = output.model_dump()
        except Exception:
            pass
    if isinstance(output, dict):
        if set(output) >= {"text"} and isinstance(output.get("text"), str):
            nested = tool_output_mapping(output["text"])
            return nested if nested is not None else output
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            nested = tool_output_mapping(item)
            if nested is not None:
                return nested
        return None
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
        except Exception:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def tool_output_failed(output: Any) -> bool:
    payload = tool_output_mapping(output)
    if isinstance(payload, dict):
        if payload.get("success") is False:
            return True
        if payload.get("success") is True:
            return False
        if payload.get("error"):
            return True
    text = serialise_tool_output(output).casefold()
    return any(
        marker in text
        for marker in (
            '"success": false',
            "'success': false",
            "error calling tool",
            "toolerror",
            "unauthorized",
            "forbidden",
            "invalid parameter",
            "timed out",
        )
    )


__all__ = ["serialise_tool_output", "tool_output_failed", "tool_output_mapping"]
