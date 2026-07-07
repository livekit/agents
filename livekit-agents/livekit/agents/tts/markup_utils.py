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


_VALUE_ATTR_RE = re.compile(r'\b[\w-]+\s*=\s*"([^"]*)"')


def extract_and_strip(
    text: str, *, xml_tags: list[str], brackets: bool
) -> tuple[str, list[tuple[str, str]]]:
    """Strip markup and collect the stripped tags in a single pass.

    One regex scan both removes the markup and records each removed tag, so
    stripping and extraction can never disagree about what counts as a tag.

    Returns ``(clean_text, tags)`` where ``tags`` is a list of ``(type, value)``
    pairs in order of appearance:

    - ``type`` is the XML tag name, or ``""`` for square-bracket tags.
    - ``value`` is a wrapping tag's inner text (``<spell>A7X9</spell>`` ->
      ``"A7X9"``), else its first quoted attribute value
      (``<emotion value="happy"/>`` -> ``"happy"``), else the bracket content,
      falling back to ``""``.

    Wrapping tags keep their inner content in ``clean_text`` (only the delimiters
    are removed); self-closing, lone, and bracket tags are removed entirely.

    Args:
        text: The text containing markup.
        xml_tags: XML tag names to handle (e.g. ``["emotion", "sound"]``).
        brackets: Whether to also handle square-bracket tags like ``[laughs]``.
    """
    if not xml_tags and not brackets:
        return text, []

    alternatives: list[str] = []
    if xml_tags:
        tag_pattern = "|".join(re.escape(tag) for tag in xml_tags)
        # <tag .../> or <tag ...> optionally followed by inner</tag>
        alternatives.append(
            rf"<(?P<tag>{tag_pattern})\b(?P<attrs>[^>]*?)\s*/?\s*>"
            rf"(?:(?P<inner>.*?)</(?P=tag)\s*>)?"
        )
        # lone closing tag: </tag>
        alternatives.append(rf"</(?:{tag_pattern})\s*>")
    if brackets:
        alternatives.append(r"\[(?P<bracket>[^\]]+)\]")

    pattern = re.compile("|".join(alternatives), re.DOTALL)
    tags: list[tuple[str, str]] = []

    def _repl(m: re.Match[str]) -> str:
        groups = m.groupdict()
        tag = groups.get("tag")
        if tag is not None:
            inner = groups.get("inner")
            if inner is not None and inner.strip():
                value = inner.strip()
            else:
                attr_match = _VALUE_ATTR_RE.search(groups.get("attrs") or "")
                value = attr_match.group(1) if attr_match else ""
            tags.append((tag, value))
            # wrapping tags keep their inner content; self-closing/lone tags vanish
            return inner if inner is not None else ""

        bracket = groups.get("bracket")
        if bracket is not None:
            tags.append(("", bracket.strip()))
            return ""

        return ""  # lone closing tag

    # iterate to a fixed point so nested wrapping tags are fully removed: a single pass
    # strips only the outer tag (e.g. <excited><loud>hi</loud></excited> -> keeps the
    # inner <loud>hi</loud>), so repeat until the text stops changing. Each pass removes
    # at least the matched delimiters, so this always terminates.
    clean = text
    prev = None
    while clean != prev:
        prev = clean
        clean = pattern.sub(_repl, clean)
    return clean, tags


def strip_bracket_tags(text: str) -> str:
    """Strip square bracket tags like ``[laughs]``, ``[whisper]`` from text."""
    return extract_and_strip(text, xml_tags=[], brackets=True)[0]


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
    return extract_and_strip(text, xml_tags=tags, brackets=False)[0]
