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

# horizontal whitespace immediately before a tag. Every removal pattern captures it as
# ``pre`` so :func:`_dedup_removal_space` can decide whether to keep it; newlines are
# excluded so paragraph breaks are never touched.
LEADING_WS = r"(?P<pre>[^\S\r\n]*)"


def _dedup_removal_space(m: re.Match[str], kept: str) -> str:
    """Replacement text for a stripped tag, minus the space its removal would double.

    The instructions place an expression marker before *every* sentence, so a turn is
    written with a marker delimited like a word — a space on each side — at every sentence
    boundary::

        <expr type="expression" label="sincere and concerned"/> Oh no, I'm sorry to hear
        that. <expr type="expression" label="warm and grounded"/> I can certainly see what
        we have available for you. <expr type="expression" label="upbeat, warm questioning"/>
        Just to be sure, are you looking to check out tomorrow, Friday the seventeenth?

    Both spaces are correct while the marker is there. Stripping it for the transcript
    collapses its width to zero and leaves both behind, so every sentence lands with two
    spaces after its punctuation (``"to hear that.  I can certainly"``) — every sentence
    of every expressive turn, not an edge case.

    When nothing of the tag survives and whitespace follows the match, the whitespace
    captured *before* it (the ``pre`` group) is therefore dropped so a single separator
    remains — matching what ``_provider_format.drop_bracket_cues`` already does for
    bracket cues.

    Whitespace before a tag at the very end of *text* is kept: it may be the separator
    for words still streaming in, and the sinks dedup that seam themselves.

    Args:
        m: A match whose pattern captured the leading whitespace as ``pre``.
        kept: The text that survives the removal (a wrapping tag's inner text or the
            native tag it lowers to), or ``""`` when the tag vanishes entirely.
    """
    pre = m.group("pre")
    if kept:
        return pre + kept
    if not pre:
        return ""
    nxt = m.string[m.end() : m.end() + 1]
    return "" if nxt.isspace() else pre


def _strip_one(out: list[str], text: str, pos: int, m: re.Match[str], kept: str) -> int:
    """Append what survives a removal, and return the position to resume scanning from.

    A tag heading a line has no space before it to pair with, so the one after it is the
    stranded half and goes with it. Anywhere else exactly one separator stays: the space
    before is dropped when one already follows, kept when none does. Newlines are never
    touched, so paragraph structure survives.
    """
    pre = m.group("pre")
    if kept:
        out.append(pre + kept)
        return pos
    if not "".join(out) or "".join(out).endswith("\n"):
        return pos + len(text[pos:]) - len(text[pos:].lstrip(" \t"))
    if pre and text[pos : pos + 1] not in (" ", "\t"):
        out.append(pre)
    return pos


def extract_and_strip(
    text: str, *, xml_tags: list[str], at_line_start: bool = True
) -> tuple[str, list[tuple[str, str]]]:
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
        at_line_start: Whether *text* begins a line. ``False`` for a chunk picked up
            mid-line, where leading whitespace is a real separator between two words.
    """
    if not xml_tags:
        return text, []

    tag_pattern = "|".join(re.escape(tag) for tag in xml_tags)
    pattern = re.compile(
        # leading space is part of the match so removing a tag can't double the separator
        LEADING_WS + "(?:"
        # <tag .../> or <tag ...> optionally followed by inner</tag>
        rf"<(?P<tag>{tag_pattern})\b(?P<attrs>[^>]*?)\s*/?\s*>"
        rf"(?:(?P<inner>.*?)</(?P=tag)\s*>)?"
        # lone closing tag: </tag>
        rf"|</(?:{tag_pattern})\s*>"
        ")",
        re.DOTALL,
    )
    tags: list[tuple[str, str]] = []

    def _pass(text: str) -> str:
        out: list[str] = ["" if at_line_start else "\u0000"]
        pos = 0
        for m in pattern.finditer(text):
            out.append(text[pos : m.start()])
            pos = m.end()
            groups = m.groupdict()
            inner = groups.get("inner")
            if (tag := groups.get("tag")) is not None:
                if inner is not None and inner.strip():
                    value = inner.strip()
                else:
                    attr_match = _VALUE_ATTR_RE.search(groups.get("attrs") or "")
                    value = attr_match.group(1) if attr_match else ""
                tags.append((tag, value))
            # wrapping tags keep their inner content; self-closing/lone tags vanish
            pos = _strip_one(out, text, pos, m, inner or "")
        out.append(text[pos:])
        return "".join(out)

    # iterate to a fixed point so nested wrapping tags are fully removed: a single pass
    # strips only the outer tag (e.g. <excited><loud>hi</loud></excited> -> keeps the
    # inner <loud>hi</loud>), so repeat until the text stops changing. Each pass removes
    # at least the matched delimiters, so this always terminates.
    clean = text
    prev = None
    while clean != prev:
        prev = clean
        clean = _pass(clean).lstrip("\u0000")
    return clean, tags
