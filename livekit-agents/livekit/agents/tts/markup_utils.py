from __future__ import annotations

import re
from functools import lru_cache

_EXPRESSION_RE = re.compile(r'<expression\s+value="([^"]*)"(?:\s*/>|>(?:.*?)</expression>)')
_SOUND_RE = re.compile(r'<sound\s+value="([^"]*)"(?:\s*/>|>(?:.*?)</sound>)')


def convert_expression_tags(text: str) -> str:
    """Convert ``<expression>`` and ``<sound>`` XML tags to ``[...]`` bracket format."""
    text = _EXPRESSION_RE.sub(lambda m: f"[{m.group(1)}]", text)
    text = _SOUND_RE.sub(lambda m: f"[{m.group(1)}]", text)
    return text


_VALUE_ATTR_RE = re.compile(r'\b[\w-]+\s*=\s*"([^"]*)"')


@lru_cache(maxsize=32)
def _compile_markup(xml_tags: tuple[str, ...]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Compile the strip and delimiter patterns for a tag set, once per set.

    ``markup`` matches a whole tag (with its inner content, for a wrapping pair);
    ``delimiters`` matches tag delimiters individually, so a single pass over a tag's
    inner content reduces it to text without needing the fixed-point loop.
    """
    tag_pattern = "|".join(re.escape(tag) for tag in xml_tags)
    markup = re.compile(
        # <tag .../> or <tag ...> optionally followed by inner</tag>
        rf"<(?P<tag>{tag_pattern})\b(?P<attrs>[^>]*?)\s*/?\s*>"
        rf"(?:(?P<inner>.*?)</(?P=tag)\s*>)?"
        # lone closing tag: </tag>
        rf"|</(?:{tag_pattern})\s*>",
        re.DOTALL,
    )
    delimiters = re.compile(rf"<(?:{tag_pattern})\b[^>]*>|</(?:{tag_pattern})\s*>")
    return markup, delimiters


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

    pattern, delimiters = _compile_markup(tuple(xml_tags))
    tags: list[tuple[str, str]] = []

    def _repl(m: re.Match[str]) -> str:
        groups = m.groupdict()
        tag = groups.get("tag")
        if tag is not None:
            inner = groups.get("inner")
            # a wrapping tag's value is its inner *text*, so nested markup is stripped out
            # of it -- those inner tags are recorded on their own by the later pass that
            # sweeps the raw inner content returned below. deleting delimiters is enough
            # here and keeps this linear: recursing would rescan each nesting level again.
            inner_text = delimiters.sub("", inner).strip() if inner else ""
            if inner_text:
                value = inner_text
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
