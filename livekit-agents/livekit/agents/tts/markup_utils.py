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


class TagAwareBuffer:
    """Buffers streaming text to ensure XML tags are never split mid-chunk.

    When streaming text that contains markup to a TTS, a chunk boundary can land
    inside a tag (e.g. ``<emotion name="hap`` | ``py"/>Hello``). This buffer holds
    back an incomplete trailing tag and flushes it once the tag is fully received.
    """

    def __init__(self) -> None:
        self._pending: str = ""

    def push(self, token: str) -> str:
        """Push a token and return the text that is safe to forward.

        Any incomplete tag at the end of the accumulated text is held back until
        the next push completes it.
        """
        self._pending += token
        return self._flush()

    def finish(self) -> str:
        """Flush any remaining buffered text (including incomplete tags)."""
        result = self._pending
        self._pending = ""
        return result

    def _flush(self) -> str:
        last_open = self._pending.rfind("<")
        if last_open == -1:
            result = self._pending
            self._pending = ""
            return result

        last_close = self._pending.rfind(">")
        if last_close > last_open:
            # tag is complete
            result = self._pending
            self._pending = ""
            return result

        # incomplete tag at the end — hold it back
        result = self._pending[:last_open]
        self._pending = self._pending[last_open:]
        return result
