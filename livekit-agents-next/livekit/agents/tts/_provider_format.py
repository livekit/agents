"""Expressive markup helpers over the core's per-provider tables (``_core.Markup``)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypedDict

from .. import _core

if TYPE_CHECKING:
    pass


class ExpressiveTag(TypedDict):
    """An expressive markup tag stripped from a transcript: the tag name (or the expr marker's
    type) and its payload (the ``value`` attribute, the expr ``label`` or the inner text)."""

    type: str
    value: str


_EXPR_MARKER_RE = re.compile(r"<expr\b([^>]*?)/?\s*>")
_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


def split_all_markup(text: str) -> tuple[str, list[ExpressiveTag]]:
    """Strip every provider's expressive XML markup; brackets are prose and stay."""
    clean, tags = _core.split_all_markup(text)
    return clean, [ExpressiveTag(type=t["type"], value=t["value"]) for t in tags]


def strip_all_markup(text: str) -> str:
    return _core.strip_all_markup(text)  # type: ignore[no-any-return]


def split_expr_markup(text: str) -> tuple[str, list[ExpressiveTag]]:
    """Strip only the ``<expr/>`` dialect, returning the clean text and its markers; the
    provider-native tags stay in the words."""
    tags = [
        ExpressiveTag(type=attrs.get("type", ""), value=attrs.get("label", ""))
        for attrs in (dict(_ATTR_RE.findall(m)) for m in _EXPR_MARKER_RE.findall(text))
    ]
    if not tags:
        return text, tags
    clean = _core.strip_expr_markup(text).rstrip(" \t")
    # a marker heading the text takes the space after it along
    return (clean.lstrip(" \t") if text.lstrip().startswith("<expr") else clean), tags


def strip_expr_markup(text: str) -> str:
    return split_expr_markup(text)[0]
