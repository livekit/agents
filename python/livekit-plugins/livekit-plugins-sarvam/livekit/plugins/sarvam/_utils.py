from __future__ import annotations


def looks_like_error_text(value: object) -> bool:
    """Heuristic to detect server-side error hints in text payloads/reasons."""
    if not isinstance(value, str):
        return False

    lowered = value.lower()
    error_hints = (
        "error",
        "invalid",
        "failed",
        "forbidden",
        "unauthorized",
        "not found",
        "rate limit",
        "timeout",
    )
    return any(hint in lowered for hint in error_hints)
