from __future__ import annotations

import re

_EXPRESSION_RE = re.compile(r'<expression\s+value="([^"]*)"(?:\s*/>|>(?:.*?)</expression>)')
_SOUND_RE = re.compile(r'<sound\s+value="([^"]*)"(?:\s*/>|>(?:.*?)</sound>)')
_BREAK_RE = re.compile(r'<break\s+time="[^"]*"\s*/>')
_BREAK_TIME_RE = re.compile(r'<break\s+time="([^"]*)"\s*/>')


def convert_expression_tags(text: str) -> str:
    """Convert ``<expression>`` and ``<sound>`` XML tags to ``[...]`` bracket format."""
    text = _EXPRESSION_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _SOUND_RE.sub(lambda m: f"[{m.group(1)}]", text)
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
