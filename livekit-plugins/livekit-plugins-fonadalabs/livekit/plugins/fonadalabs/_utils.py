from __future__ import annotations

import json
from typing import Any


def parse_error(body: str | dict[str, Any]) -> tuple[str, str]:
    """Return ``(error_code, message)`` from a FonadaLabs error payload.

    HTTP errors look like ``{"detail": {"error": ..., "message": ...}}``;
    WebSocket errors look like ``{"type": "error", "error": ..., "message": ...}``.
    """
    if isinstance(body, str):
        try:
            data: Any = json.loads(body)
        except ValueError:
            return "", body[:300]
    else:
        data = body

    if not isinstance(data, dict):
        return "", str(data)[:300]

    detail = data.get("detail", data)
    if isinstance(detail, dict):
        return str(detail.get("error") or ""), str(detail.get("message") or detail)
    return "", str(detail)[:300]
