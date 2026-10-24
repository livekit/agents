"""Emotion markers in LLM text, found while the text streams to the TTS.

Two marker styles (:data:`MarkerStyle`):

- ``"expr"``: LiveKit's expressive markup, ``<expr type="expression" label="happy"/>``.
  The label sets the emotion it names as a whole word, so a free-form (Inworld) label
  such as "warm, concerned" works too. Other ``<expr>`` types (break, sound, prosody,
  spell) are not emotions; the words inside a wrapping marker still count as spoken.
- ``"brackets"``: audio tags, ``[happy]``, as ElevenLabs v3/v4 voices read natively. A
  tag sets an emotion only when it is exactly the emotion's name; other tags
  (``[whispers]``) steer the voice only. A bracketed run longer than
  ``MAX_TAG_LENGTH``, or spanning a line, is text.

``<expr>`` markup is read in either style, since no prose looks like it. Square brackets
are tags only in the brackets style; in any other they are text the TTS reads ("[1]").

Each marker is reported with an *anchor*: the number of normalized spoken characters
(letters and digits, lowercased) before it. The planner in ``_tts_node`` resolves the
anchor to a time once the TTS reports where those characters were spoken.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from .types import EMOTIONS, Emotion

MarkerStyle = Literal["expr", "brackets"]
"""How the LLM writes emotion markers: ``<expr>`` markup, or ``[tag]`` audio tags."""

MAX_TAG_LENGTH = 32
"""Longer bracketed runs are text, not tags."""

# cap on the attributes of an unclosed "<expr" held before it is released as text
_MAX_EXPR_HOLD = 256

_BRACKET_TAG_RE = re.compile(rf"\[(?P<name>[^\[\]\n]{{1,{MAX_TAG_LENGTH}}})\]")
_EXPR_TAG_RE = re.compile(r"<expr\b(?P<attrs>[^>]*?)/?\s*>|</expr\s*>")
_EXPR_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class Marker:
    """An emotion marker found in the text."""

    anchor: int
    """Normalized spoken characters before the marker."""
    emotion: Emotion


def normalize(text: str) -> str:
    """Letters and digits only, lowercased, so TTS alignment can be compared with text."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def emotion_for_label(label: str, allowed: Sequence[Emotion] = EMOTIONS) -> Emotion | None:
    """The first emotion in ``allowed`` that an ``<expr>`` label names as a whole word.

    "warm, concerned" is concerned; "excitedly", "unhappy" and "sympathetic" name none.
    """
    text = label.lower()
    found = [(m.start(), e) for e in allowed if (m := re.search(rf"\b{e}\b", text))]
    return min(found)[1] if found else None


def emotion_for_tag(name: str, allowed: Sequence[Emotion] = EMOTIONS) -> Emotion | None:
    """The emotion in ``allowed`` that a ``[tag]`` name is exactly, ignoring case."""
    name = name.strip().lower()
    return next((e for e in allowed if e == name), None)


@dataclass(frozen=True)
class _Syntax:
    """One way of writing a marker."""

    whole: re.Pattern[str]
    """A complete marker."""
    partial: re.Pattern[str]
    """A marker cut off by the end of the text, which the next chunk may complete."""
    emotion: Callable[[re.Match[str], Sequence[Emotion]], Emotion | None]
    """The emotion in the allowed set that a complete marker sets, if any."""


def _expr_emotion(match: re.Match[str], allowed: Sequence[Emotion]) -> Emotion | None:
    attrs = dict(_EXPR_ATTR_RE.findall(match.group("attrs") or ""))
    if attrs.get("type") != "expression":
        return None
    return emotion_for_label(attrs.get("label", ""), allowed)


_EXPR = _Syntax(
    whole=_EXPR_TAG_RE,
    partial=re.compile(rf"</?(?:e|ex|exp|expr(?:\W[^>]{{0,{_MAX_EXPR_HOLD}}})?)?\Z"),
    emotion=_expr_emotion,
)
_BRACKETS = _Syntax(
    whole=_BRACKET_TAG_RE,
    partial=re.compile(rf"\[[^\[\]\n]{{0,{MAX_TAG_LENGTH}}}\Z"),
    emotion=lambda match, allowed: emotion_for_tag(match.group("name"), allowed),
)


class TagParser:
    """Streaming parser: feed text chunks, get the text to forward and the markers found.

    A marker split across chunks is held back until it completes (or can no longer be a
    marker), so the forwarded text never carries half a tag.

    Args:
        style: The markers to read. ``<expr>`` markup is read in either style.
        emotions: Emotions a marker may resolve to.
        forward_expr: Pass ``<expr>`` markup on to the TTS. ``True`` when the session TTS
            lowers it (expressive mode), ``False`` to strip it so the TTS does not read
            it aloud. ``[tag]`` markers always go on: the TTS reads them.
    """

    def __init__(
        self,
        *,
        style: MarkerStyle = "expr",
        emotions: Sequence[Emotion] = EMOTIONS,
        forward_expr: bool,
    ) -> None:
        if style not in get_args(MarkerStyle):
            raise ValueError(f"unknown marker style {style!r}")
        # each syntax by the character that opens it, with whether its markers go on
        self._syntaxes: dict[str, tuple[_Syntax, bool]] = {"<": (_EXPR, forward_expr)}
        if style == "brackets":
            self._syntaxes["["] = (_BRACKETS, True)
        self._opener = re.compile("|".join(map(re.escape, self._syntaxes)))
        self._emotions = emotions
        self._buf = ""
        self._spoken = ""

    @property
    def spoken(self) -> str:
        """The normalized spoken text forwarded so far; markers are anchored into it."""
        return self._spoken

    @property
    def anchor(self) -> int:
        """Normalized spoken characters forwarded so far."""
        return len(self._spoken)

    def push(self, text: str) -> tuple[str, list[Marker]]:
        self._buf += text
        return self._consume(final=False)

    def flush(self) -> tuple[str, list[Marker]]:
        return self._consume(final=True)

    def _consume(self, *, final: bool) -> tuple[str, list[Marker]]:
        buf, pos = self._buf, 0
        out: list[str] = []
        markers: list[Marker] = []
        while (opener := self._opener.search(buf, pos)) is not None:
            i = opener.start()
            self._emit_text(buf[pos:i], out)
            syntax, forward = self._syntaxes[buf[i]]
            if match := syntax.whole.match(buf, i):
                if (emotion := syntax.emotion(match, self._emotions)) is not None:
                    markers.append(Marker(self.anchor, emotion))
                if forward:
                    out.append(match.group())
                pos = match.end()
            elif not final and syntax.partial.match(buf, i):
                self._buf = buf[i:]
                return "".join(out), markers
            else:
                self._emit_text(buf[i], out)  # not a marker after all
                pos = i + 1

        self._emit_text(buf[pos:], out)
        self._buf = ""
        return "".join(out), markers

    def _emit_text(self, text: str, out: list[str]) -> None:
        if text:
            out.append(text)
            self._spoken += normalize(text)
