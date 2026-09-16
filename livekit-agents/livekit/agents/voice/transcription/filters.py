import re
from collections.abc import AsyncIterable

# Scripts that put an emphasis delimiter flush against a word character: CJK,
# kana and Thai take no spaces at all, Korean attaches particles to the closing
# run. A neighbour from one of these ranges is still a valid boundary.
_FLUSH_EMPHASIS_SCRIPTS = (
    r"\u0e00-\u0e7f"  # thai
    r"\u1100-\u11ff"  # hangul jamo
    r"\u3040-\u30ff"  # hiragana, katakana
    r"\u3130-\u318f"  # hangul compatibility jamo
    r"\u3400-\u4dbf"  # cjk unified ideographs extension a
    r"\u4e00-\u9fff"  # cjk unified ideographs
    r"\uac00-\ud7af"  # hangul syllables
    r"\uf900-\ufaff"  # cjk compatibility ideographs
    r"\uff66-\uff9d"  # halfwidth katakana
)
# a word character that would make emphasis intra-word: ``\w`` less those scripts
_INTRAWORD = rf"[^\W{_FLUSH_EMPHASIS_SCRIPTS}]"

# emphasis *text*, **text**, ***text***: the closing run has to match the
# opening one. Asterisks are also arithmetic, exponents and globs, so each side
# needs a boundary that is neither intra-word nor another asterisk.
_ASTERISK_EMPHASIS = re.compile(
    rf"(?<!{_INTRAWORD})(?<!\*)"  # opening boundary
    r"(\*{1,3})(?!\s)"  # opening delimiter run
    r"([^*\n]+?)"  # body
    r"(?<!\s)\1"  # closing run, same width
    rf"(?!{_INTRAWORD})(?!\*)"  # closing boundary
)
# the same for underscores. CommonMark has no intra-word underscore emphasis,
# so the boundary here stays the full word class -- that is what keeps
# ``snake_case`` and ``__dunder__`` intact, in every script.
_UNDERSCORE_EMPHASIS = re.compile(r"(?<!\w)(_{1,3})(?!\s)([^_\n]+?)(?<!\s)\1(?!\w)")

# Anchored at the start of a line only, so they are safe to apply to a prefix
# of a line that is still being streamed.
LINE_PATTERNS = [
    # headers: remove # and following spaces
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # list markers: remove -, +, * and following spaces
    (re.compile(r"^\s*[-+*]\s+", re.MULTILINE), ""),
    # block quotes: remove > and following spaces
    (re.compile(r"^\s*>\s+", re.MULTILINE), ""),
]

# These match a whole line, so they may only be applied once the line is known
# to be complete -- a prefix of ``____ and so on`` is not a horizontal rule. They
# run before LINE_PATTERNS, which would read ``* * *`` as a list item.
FULL_LINE_PATTERNS = [
    # horizontal rules carry no spoken content. per CommonMark the markers may be
    # spaced apart, and a fourth column of indent makes the line code
    (
        re.compile(
            r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$",
            re.MULTILINE,
        ),
        "",
    ),
]

INLINE_PATTERNS = [
    # images: keep alt text ![alt](url) -> alt
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # links: keep text part [text](url) -> text
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # emphasis: remove the delimiters, keep the body. Runs before the code and
    # strikethrough patterns, which would otherwise consume the body and strand
    # the delimiters around it (``**~~text~~**``).
    (_ASTERISK_EMPHASIS, r"\2"),
    (_UNDERSCORE_EMPHASIS, r"\2"),
    # nesting is order-dependent -- ``**_text_**`` needs the asterisks off
    # first, ``_**text**_`` the underscores -- so the asterisks get a second
    # look once the underscores are gone
    (_ASTERISK_EMPHASIS, r"\2"),
    # code blocks: remove ``` from ```text```
    (re.compile(r"`{3,4}[\S]*"), ""),
    # inline code: remove ` from `text`
    (re.compile(r"`([^`]+?)`"), r"\1"),
    # strikethrough: drop ~~text~~ entirely, struck text is not spoken
    (re.compile(r"~~(?!\s)[^~]*?(?<!\s)~~"), ""),
]
INLINE_SPLIT_TOKENS = " ,.?!;，。？！；"

# text without one of these cannot match any inline pattern
INLINE_MARKERS = re.compile(r"[*_`~\[]")

COMPLETE_LINKS_PATTERN = re.compile(r"\[[^\]]*\]\([^)]*\)")  # links [text](url)
COMPLETE_IMAGES_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # images ![text](url)


def _unbalanced(buffer: str, delimiter: str) -> bool:
    """Whether a delimiter's occurrences cannot pair up yet."""
    doubles = buffer.count(delimiter * 2)
    if doubles % 2 == 1:
        return True
    return (buffer.count(delimiter) - doubles * 2) % 2 == 1


async def filter_markdown(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Filter out markdown symbols from the text.
    """

    def has_incomplete_pattern(buffer: str) -> bool:
        """Check if buffer might contain incomplete markdown patterns that need more text."""

        if buffer.endswith(("#", "-", "+", "*", "_", ">", "!", "`", "~", " ")):
            return True

        # emphasis delimiters that cannot pair up yet
        if _unbalanced(buffer, "*") or _unbalanced(buffer, "_"):
            return True

        # incomplete code (`text`) or strikethrough (~~text~~)
        if buffer.count("`") % 2 == 1 or buffer.count("~~") % 2 == 1:
            return True

        # incomplete links [text](url) or images ![text](url)
        open_brackets = buffer.count("[")
        complete_links = len(COMPLETE_LINKS_PATTERN.findall(buffer))
        complete_images = len(COMPLETE_IMAGES_PATTERN.findall(buffer))

        return open_brackets - complete_links - complete_images > 0

    def process_complete_text(text: str, *, is_newline: bool, is_line_end: bool) -> str:
        if is_newline:
            if is_line_end:
                for pattern, replacement in FULL_LINE_PATTERNS:
                    text = pattern.sub(replacement, text)

            for pattern, replacement in LINE_PATTERNS:
                text = pattern.sub(replacement, text)

        # most spoken text carries no inline markup at all
        if not INLINE_MARKERS.search(text):
            return text

        for pattern, replacement in INLINE_PATTERNS:
            text = pattern.sub(replacement, text)

        return text

    buffer = ""
    buffer_is_newline = True  # track if buffer is at start of line

    async for chunk in text:
        buffer += chunk

        if "\n" in buffer:
            lines = buffer.split("\n")
            buffer = lines[-1]  # keep last incomplete line

            for i, line in enumerate(lines[:-1]):
                is_newline = buffer_is_newline if i == 0 else True
                processed_line = process_complete_text(
                    line, is_newline=is_newline, is_line_end=True
                )
                yield processed_line + "\n"

            buffer_is_newline = True
            continue

        # split at the position after the split token
        last_split_pos = max(map(buffer.rfind, INLINE_SPLIT_TOKENS))

        if last_split_pos >= 1:
            processable = buffer[:last_split_pos]  # exclude the split token
            rest = buffer[last_split_pos:]
            if not has_incomplete_pattern(processable):
                yield process_complete_text(
                    processable, is_newline=buffer_is_newline, is_line_end=False
                )
                buffer = rest
                buffer_is_newline = False

    if buffer:
        yield process_complete_text(buffer, is_newline=buffer_is_newline, is_line_end=True)


# Unicode block ranges from: https://unicode.org/Public/UNIDATA/Blocks.txt
EMOJI_PATTERN = re.compile(
    r"[\U0001F000-\U0001FBFF]"  # Emoji blocks: Mahjong Tiles through Symbols for Legacy Computing
    r"|[\U00002600-\U000026FF]"  # Miscellaneous Symbols
    r"|[\U00002700-\U000027BF]"  # Dingbats
    r"|[\U00002B00-\U00002BFF]"  # Miscellaneous Symbols and Arrows
    r"|[\U0000FE00-\U0000FE0F]"  # Variation selectors
    r"|\U0000200D"  # Zero width joiner
    r"|\U000020E3"  # Combining enclosing keycap
    r"+",
    re.UNICODE,
)


async def filter_emoji(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Filter out emojis from the text.
    """

    async for chunk in text:
        filtered_chunk = EMOJI_PATTERN.sub("", chunk)
        yield filtered_chunk
