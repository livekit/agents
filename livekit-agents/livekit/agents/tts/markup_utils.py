from __future__ import annotations

import re


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
