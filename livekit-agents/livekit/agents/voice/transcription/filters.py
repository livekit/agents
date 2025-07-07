import re
from collections.abc import AsyncIterable

LINE_PATTERNS = [
    # headers: remove # and following spaces
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # list markers: remove -, +, * and following spaces
    (re.compile(r"^\s*[-+*]\s+", re.MULTILINE), ""),
    # block quotes: remove > and following spaces
    (re.compile(r"^\s*>\s+", re.MULTILINE), ""),
]

INLINE_PATTERNS = [
    # images: keep alt text ![alt](url) -> alt
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # links: keep text part [text](url) -> text
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),
    # bold: remove asterisks from **text**
    (re.compile(r"\*\*([^*]+?)\*\*"), r"\1"),
    # italic: remove asterisks from *text*
    (re.compile(r"\*([^*]+?)\*"), r"\1"),
    # bold with underscores: remove underscores from __text__
    (re.compile(r"__([^_]+?)__"), r"\1"),
    # italic with underscores: remove underscores from _text_
    (re.compile(r"_([^_]+?)_"), r"\1"),
    # code blocks: remove ``` from ```text```
    (re.compile(r"`{3,4}[\S]*"), ""),
    # inline code: remove ` from `text`
    (re.compile(r"`([^`]+?)`"), r"\1"),
    # strikethrough: remove ~~text~~
    (re.compile(r"~~([^~]*?)~~"), ""),
]

COMPLETE_LINKS_PATTERN = re.compile(r"\[[^\]]*\]\([^)]*\)")  # links [text](url)
COMPLETE_IMAGES_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # images ![text](url)


async def filter_markdown(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """
    Filter out markdown symbols from the text.
    """

    def has_incomplete_pattern(buffer: str) -> bool:
        """Check if buffer might contain incomplete markdown patterns that need more text."""

        if buffer.endswith(("#", "-", "+", "*", ">", "!", "`", "~", " ")):
            return True

        # check for incomplete bold (**text** or *text*)
        double_asterisks = buffer.count("**")
        if double_asterisks % 2 == 1:
            return True

        single_asterisks = buffer.count("*") - (double_asterisks * 2)
        if single_asterisks % 2 == 1:
            return True

        # check for incomplete underscores (__text__ or _text_)
        double_underscores = buffer.count("__")
        if double_underscores % 2 == 1:
            return True
        single_underscores = buffer.count("_") - (double_underscores * 2)
        if single_underscores % 2 == 1:
            return True

        # check for incomplete code (`text`)
        backticks = buffer.count("`")
        if backticks % 2 == 1:
            return True

        # check for incomplete strikethrough (~~text~~)
        double_tildes = buffer.count("~~")
        if double_tildes % 2 == 1:
            return True

        # check for incomplete links [text](url) or images ![text](url)
        open_brackets = buffer.count("[")
        complete_links = len(COMPLETE_LINKS_PATTERN.findall(buffer))
        complete_images = len(COMPLETE_IMAGES_PATTERN.findall(buffer))

        remaining_brackets = open_brackets - complete_links - complete_images
        if remaining_brackets > 0:
            return True

        return False

    def process_complete_text(text: str, is_newline: bool = False) -> str:
        if is_newline:
            for pattern, replacement in LINE_PATTERNS:
                text = pattern.sub(replacement, text)

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
                processed_line = process_complete_text(line, is_newline=is_newline)
                yield processed_line + "\n"

            buffer_is_newline = True
        elif not has_incomplete_pattern(buffer):
            yield process_complete_text(buffer, is_newline=buffer_is_newline)
            buffer = ""
            buffer_is_newline = False

    if buffer:
        yield process_complete_text(buffer, is_newline=buffer_is_newline)
