"""Emotion markers in LLM text, found while the text streams to the TTS.

Two marker styles are recognized:

- LiveKit's expressive markup, ``<expr type="expression" label="happy"/>``. The label
  sets the avatar emotion it names as a whole word, so a free-form (Inworld) label such
  as "warm, concerned" works too. Other ``<expr>`` types (break, sound, prosody, spell)
  are not emotions; the words inside a wrapping marker still count as spoken.
- Bracketed audio tags, ``[happy]``, as ElevenLabs v3/v4 voices read natively. A tag sets
  an emotion only when it is exactly the emotion's name. Longer bracketed runs, and ones
  spanning a newline, are text (``MAX_TAG_LENGTH``).

Each marker is reported with an *anchor*: the number of normalized spoken characters
(letters and digits, lowercased) before it. The planner in ``_tts_node`` resolves the
anchor to a time once the TTS reports where those characters were spoken.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .types import EMOTIONS, Emotion

MAX_TAG_LENGTH = 32
"""Longer bracketed runs are text, not tags."""

# cap on how long an unclosed "<expr" is held before it is released as plain text
_MAX_EXPR_HOLD = 256

_EXPR_TAG_RE = re.compile(r"<expr\b(?P<attrs>[^>]*?)/?\s*>|</expr\s*>")
_EXPR_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')
_BRACKET_TAG_RE = re.compile(rf"\[(?P<name>[^\[\]\n]{{1,{MAX_TAG_LENGTH}}})\]")
_EXPR_PREFIXES = ("<expr", "</expr")


@dataclass(frozen=True)
class Marker:
    """An emotion marker found in the text."""

    anchor: int
    """Normalized spoken characters before the marker."""
    emotion: Emotion


def normalize(text: str) -> str:
    """Letters and digits only, lowercased, so TTS alignment can be compared with text."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _may_be_expr(rest: str) -> bool:
    """Whether ``rest`` (which starts with "<") opens an expr tag that has not closed yet."""
    for prefix in _EXPR_PREFIXES:
        if len(rest) <= len(prefix):
            if prefix.startswith(rest):
                return True
        elif rest.startswith(prefix) and not (
            rest[len(prefix)].isalnum() or rest[len(prefix)] == "_"
        ):
            return True
    return False


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


class TagParser:
    """Streaming parser: feed text chunks, get the text to forward and the markers found.

    A marker split across chunks is held back until it completes (or can no longer be a
    marker), so the forwarded text never carries half a tag.

    Args:
        emotions: Emotions a marker may resolve to.
        forward_expr: Pass ``<expr>`` markup on to the TTS. ``True`` when the session TTS
            lowers it (expressive mode), ``False`` to strip it so the TTS does not read
            it aloud.
        forward_brackets: Pass ``[tag]`` markers on to the TTS.
    """

    def __init__(
        self,
        *,
        emotions: Sequence[Emotion] = EMOTIONS,
        forward_expr: bool,
        forward_brackets: bool = True,
    ) -> None:
        self._emotions = emotions
        self._forward_expr = forward_expr
        self._forward_brackets = forward_brackets
        self._buf = ""
        self._spoken = ""
        self._saw_brackets = False

    @property
    def spoken(self) -> str:
        """The normalized spoken text forwarded so far; markers are anchored into it."""
        return self._spoken

    @property
    def anchor(self) -> int:
        """Normalized spoken characters forwarded so far."""
        return len(self._spoken)

    @property
    def saw_brackets(self) -> bool:
        """Whether any ``[tag]`` marker was forwarded."""
        return self._saw_brackets

    def push(self, text: str) -> tuple[str, list[Marker]]:
        self._buf += text
        return self._consume(final=False)

    def flush(self) -> tuple[str, list[Marker]]:
        return self._consume(final=True)

    def _consume(self, *, final: bool) -> tuple[str, list[Marker]]:
        buf = self._buf
        out: list[str] = []
        markers: list[Marker] = []
        pos = 0
        while pos < len(buf):
            lt, lb = buf.find("<", pos), buf.find("[", pos)
            candidates = [i for i in (lt, lb) if i != -1]
            if not candidates:
                self._emit_text(buf[pos:], out)
                pos = len(buf)
                break
            i = min(candidates)
            self._emit_text(buf[pos:i], out)
            pos = i

            if buf[i] == "<":
                consumed = self._consume_expr(buf, i, out, markers, final=final)
            else:
                consumed = self._consume_bracket(buf, i, out, markers, final=final)
            if consumed is None:
                break  # held for the next chunk
            pos = consumed

        self._buf = buf[pos:]
        return "".join(out), markers

    def _emit_text(self, text: str, out: list[str]) -> None:
        if text:
            out.append(text)
            self._spoken += normalize(text)

    def _consume_expr(
        self, buf: str, i: int, out: list[str], markers: list[Marker], *, final: bool
    ) -> int | None:
        """Handle a "<" at ``i``; return where to continue, or None to hold."""
        rest = buf[i:]
        match = _EXPR_TAG_RE.match(rest)
        if match is None:
            if _may_be_expr(rest) and not final and len(rest) <= _MAX_EXPR_HOLD:
                return None
            self._emit_text("<", out)
            return i + 1

        attrs = match.group("attrs")
        if attrs is not None:  # an opening or self-closing marker
            parsed = dict(_EXPR_ATTR_RE.findall(attrs))
            if parsed.get("type") == "expression":
                emotion = emotion_for_label(parsed.get("label", ""), self._emotions)
                if emotion is not None:
                    markers.append(Marker(self.anchor, emotion))
        if self._forward_expr:
            out.append(match.group(0))
        return i + match.end()

    def _consume_bracket(
        self, buf: str, i: int, out: list[str], markers: list[Marker], *, final: bool
    ) -> int | None:
        """Handle a "[" at ``i``; return where to continue, or None to hold."""
        rest = buf[i:]
        match = _BRACKET_TAG_RE.match(rest)
        if match is None:
            # an unclosed tag may still close in the next chunk, unless it is already
            # too long or spans a line
            open_run = rest[1:]
            if (
                not final
                and len(open_run) <= MAX_TAG_LENGTH
                and "\n" not in open_run
                and "]" not in open_run
            ):
                return None
            self._emit_text("[", out)
            return i + 1

        name = match.group("name").strip()
        if not name:
            self._emit_text(match.group(0), out)
            return i + match.end()

        emotion = emotion_for_tag(name, self._emotions)
        if emotion is not None:
            markers.append(Marker(self.anchor, emotion))
        if self._forward_brackets:
            self._saw_brackets = True
            out.append(match.group(0))
        return i + match.end()


def strip_markers(text: str) -> str:
    """Remove ``<expr>`` markup and ``[tag]`` markers from ``text``, keeping the words.

    Used on TTS-aligned text before matching it against the parsed text, so the two
    normalize to the same characters.
    """
    text = _EXPR_TAG_RE.sub("", text)
    return _BRACKET_TAG_RE.sub("", text)
