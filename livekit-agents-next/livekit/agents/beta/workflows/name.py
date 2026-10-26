from __future__ import annotations

from dataclasses import dataclass


def _clean_name_arg(value: str | None) -> str | None:
    # Some models (e.g. gemma) fill optional args with placeholder strings like
    # "null"/"NULL" instead of omitting them, or wrap values in literal quotes.
    # Normalize those to None/clean values so they hit the required-field
    # validation below instead of being recorded as the user's name.
    if value is None:
        return None
    value = value.strip().strip("'\"")
    if not value or value.casefold() in ("null", "none", "nil", "n/a", "unknown", "unspecified"):
        return None
    return value


@dataclass
class GetNameResult:
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None
