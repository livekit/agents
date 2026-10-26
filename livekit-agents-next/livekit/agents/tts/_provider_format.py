"""Expressive markup helpers over the core's per-provider tables (``_core.Markup``)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypedDict

from .. import _core
from ..types import TimedString

if TYPE_CHECKING:
    from .. import tokenize


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


def sentence_tokenizer(provider: str, *, expressive: bool) -> tokenize.SentenceTokenizer:
    """The blingfire tokenizer for a provider's streamed input: capped at its max chunk length,
    batching sentences up to it when ``expressive``."""
    from .. import tokenize

    max_len, min_len, xml_aware = _core.sentence_tokenizer_config(provider, expressive=expressive)
    return tokenize.blingfire.SentenceTokenizer(
        max_token_len=max_len, min_token_len=min_len, xml_aware=xml_aware
    )


def drop_bracket_cues(
    tokens: list[TimedString], held: list[TimedString], *, final: bool = False
) -> list[TimedString]:
    """Remove bracket cues from TTS-aligned tokens, keeping the survivors' timings.

    ``held`` carries an unclosed span across calls; pass the same list each time and call once
    more with ``final=True`` at the end of the stream.
    """
    tokens = held + tokens
    held.clear()
    emit, keep = _core.drop_bracket_cues([str(t) for t in tokens], final)
    held.extend(_retext(tokens[i], text) for i, text in keep)
    return [_retext(tokens[i], text) for i, text in emit]


def _retext(token: TimedString, text: str | None) -> TimedString:
    if text is None:
        return token
    return TimedString(
        text,
        start_time=token.start_time,
        end_time=token.end_time,
        confidence=token.confidence,
        start_time_offset=token.start_time_offset,
        speaker_id=token.speaker_id,
    )
