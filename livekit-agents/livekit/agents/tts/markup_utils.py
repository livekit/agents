from __future__ import annotations

import re

# Permissive regex: allows extra trailing attributes between `value="..."` and the
# tag close (e.g. `<expression value="X" extra/>`), and captures inner content for
# the wrapping form `<expression value="X">content</expression>`. Smaller LLMs
# occasionally emit both malformations; the looser pattern keeps them from leaking
# raw XML to the TTS provider.
_EXPRESSION_RE = re.compile(
    r'<expression\s+value="([^"]*)"[^>]*?(?:/>|>(.*?)</expression\s*>)',
    re.DOTALL,
)
_SOUND_RE = re.compile(
    r'<sound\s+value="([^"]*)"[^>]*?(?:/>|>(.*?)</sound\s*>)',
    re.DOTALL,
)
# Orphan closing tags left behind when `normalize_markup` rewrites a wrapping
# opener (`<expression value="X" >…`) to self-closing — the trailing
# `</expression>` no longer pairs with anything and would otherwise reach the
# TTS provider as raw XML.
_ORPHAN_CLOSE_RE = re.compile(r"</(?:expression|sound)\s*>", re.IGNORECASE)
_BREAK_RE = re.compile(r'<break\s+time="[^"]*"\s*/>')
_BREAK_TIME_RE = re.compile(r'<break\s+time="([^"]*)"\s*/>')


def convert_expression_tags(text: str) -> str:
    """Convert ``<expression>`` and ``<sound>`` XML tags to ``[...]`` bracket format.

    Tolerates malformed shapes smaller LLMs occasionally emit:
      - Extra trailing attributes (``<expression value="X" extra/>``) are ignored.
      - Wrapping form (``<expression value="X">content</expression>``) preserves
        the inner content, replacing the wrapper with ``[X]``.
      - Orphan closing tags left by ``normalize_markup`` rewriting a wrapping
        opener to self-closing are stripped.
    """

    def _sub(m: re.Match[str]) -> str:
        value = m.group(1)
        content = m.group(2)
        return f"[{value}]{content}" if content is not None else f"[{value}]"

    text = _EXPRESSION_RE.sub(_sub, text)
    text = _SOUND_RE.sub(_sub, text)
    text = _ORPHAN_CLOSE_RE.sub("", text)
    return text


def convert_expression_to_fish(text: str) -> str:
    """Fish-specific variant of `convert_expression_tags`.

    Same shape handling, but each ``<expression>`` value is intensified with a leading
    "very" so the emotion lands harder in Fish's audio, e.g.
    ``<expression value="regretful"/>foo`` → ``[very regretful]foo``. ``<sound>`` values
    pass through unchanged (``[laughing]``), and an already-"very" value isn't doubled.
    """

    def _expr(m: re.Match[str]) -> str:
        value = m.group(1).strip()
        if value and not value.lower().startswith("very "):
            value = f"very {value}"
        content = m.group(2)
        return f"[{value}]{content}" if content is not None else f"[{value}]"

    def _sound(m: re.Match[str]) -> str:
        value = m.group(1)
        content = m.group(2)
        return f"[{value}]{content}" if content is not None else f"[{value}]"

    text = _EXPRESSION_RE.sub(_expr, text)
    text = _SOUND_RE.sub(_sound, text)
    text = _ORPHAN_CLOSE_RE.sub("", text)
    return text


def convert_break_to_ellipsis(text: str) -> str:
    """Replace ``<break time="..."/>`` tags with an ellipsis (``...``).

    Used for providers whose pacing is best expressed through punctuation
    (e.g. Inworld) rather than explicit silence directives.
    """
    return _BREAK_RE.sub("...", text)


def _break_seconds(value: str) -> float | None:
    """Parse a break duration like ``500ms``, ``1s``, or ``1.5`` into seconds."""
    value = value.strip().lower()
    try:
        if value.endswith("ms"):
            return float(value[:-2]) / 1000.0
        if value.endswith("s"):
            return float(value[:-1])
        return float(value)
    except ValueError:
        return None


def convert_break_to_fish(text: str) -> str:
    """Replace ``<break time="..."/>`` tags with Fish Audio's native pause markers.

    Fish Audio exposes two pause primitives, ``[break]`` and ``[long-break]``; map
    any pause of roughly a second or longer to the longer marker.
    """

    def _sub(m: re.Match[str]) -> str:
        seconds = _break_seconds(m.group(1))
        return "[long-break]" if seconds is not None and seconds >= 1.0 else "[break]"

    return _BREAK_TIME_RE.sub(_sub, text)


_EMPHASIS_RE = re.compile(r"<emphasis(?:\s[^>]*)?>([^<]*)</emphasis>", re.IGNORECASE)


def convert_emphasis_to_fish(text: str) -> str:
    """Convert ``<emphasis>word</emphasis>`` wrappers to Fish Audio's ``[emphasis] word``.

    Fish exposes ``[emphasis]`` as an inline marker that stresses the word that
    immediately follows it. The framework wraps the emphasized word in XML so the
    same syntax stays consistent with the other tags; this rewrites the pair into
    Fish's prefix-marker form.
    """
    return _EMPHASIS_RE.sub(lambda m: f"[emphasis] {m.group(1).strip()}", text)


def strip_bracket_tags(text: str) -> str:
    """Strip square bracket tags like ``[laughs]``, ``[whisper]`` from text."""
    return re.sub(r"\[[^\]]+\]", "", text)


def strip_xml_tags(text: str, tags: list[str]) -> str:
    """Strip specific XML-style tags from text, preserving their inner content.

    Handles opening/closing tag pairs (``<tag ...>content</tag>``) and
    self-closing tags (``<tag .../>``, ``<tag />``).

    Args:
        text: The text containing XML-style markup.
        tags: List of tag names to strip (e.g. ``["emotion", "speed"]``).

    Returns:
        The text with the specified tags removed but their content preserved.
    """
    if not tags:
        return text

    tag_pattern = "|".join(re.escape(tag) for tag in tags)

    # self-closing: <tag ... />
    text = re.sub(rf"<(?:{tag_pattern})\b[^>]*/\s*>", "", text)

    # opening: <tag ...>
    text = re.sub(rf"<(?:{tag_pattern})\b[^>]*>", "", text)

    # closing: </tag>
    text = re.sub(rf"</(?:{tag_pattern})\s*>", "", text)

    return text
