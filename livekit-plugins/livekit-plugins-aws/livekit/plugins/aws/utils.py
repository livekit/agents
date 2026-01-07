from __future__ import annotations

DEFAULT_REGION = "us-east-1"


def _strip_nones(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}
