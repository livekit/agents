from __future__ import annotations

import re

_EXPRESSION_RE = re.compile(r'<expression\s+value="([^"]*)"(?:\s*/>|>(?:.*?)</expression>)')
_SOUND_RE = re.compile(r'<sound\s+value="([^"]*)"(?:\s*/>|>(?:.*?)</sound>)')


def convert_expression_tags(text: str) -> str:
    """Convert ``<expression>`` and ``<sound>`` XML tags to ``[...]`` bracket format."""
    text = _EXPRESSION_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _SOUND_RE.sub(lambda m: f"[{m.group(1)}]", text)
    return text


_VALUE_ATTR_RE = re.compile(r'\b[\w-]+\s*=\s*"([^"]*)"')


def extract_and_strip(text: str, *, xml_tags: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """Strip XML markup tags and collect the stripped tags in a single pass.

    One regex scan both removes the markup and records each removed tag, so
    stripping and extraction can never disagree about what counts as a tag.

    Only XML-shaped markup is recognized. Square brackets are left alone: in LLM output
    they are prose (``[text](url)`` links) that a strip would mangle, and provider-native
    ones are removed at their source by ``_provider_format.drop_bracket_cues``.

    Returns ``(clean_text, tags)`` where ``tags`` is a list of ``(type, value)``
    pairs in order of appearance:

    - ``type`` is the XML tag name.
    - ``value`` is a wrapping tag's inner text (``<spell>A7X9</spell>`` ->
      ``"A7X9"``), else its first quoted attribute value
      (``<emotion value="happy"/>`` -> ``"happy"``), falling back to ``""``.

    Wrapping tags keep their inner content in ``clean_text`` (only the delimiters
    are removed); self-closing and lone tags are removed entirely.

    Args:
        text: The text containing markup.
        xml_tags: XML tag names to handle (e.g. ``["emotion", "sound"]``).
    """
    if not xml_tags:
        return text, []

    tag_pattern = "|".join(re.escape(tag) for tag in xml_tags)
    pattern = re.compile(
        # <tag .../> or <tag ...> optionally followed by inner</tag>
        rf"<(?P<tag>{tag_pattern})\b(?P<attrs>[^>]*?)\s*/?\s*>"
        rf"(?:(?P<inner>.*?)</(?P=tag)\s*>)?"
        # lone closing tag: </tag>
        rf"|</(?:{tag_pattern})\s*>",
        re.DOTALL,
    )
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
