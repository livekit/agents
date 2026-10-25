import re

from . import tokenizer


def split_words(
    text: str,
    *,
    ignore_punctuation: bool = True,
    split_character: bool = False,
    retain_format: bool = False,
) -> list[tuple[str, int, int]]:
    """
    Split text into words, supporting both space-separated languages (like English)
    and character-based languages (like Chinese, Japanese, Korean, Thai).

    For non-spaced scripts, each character is treated as a separate word if split_character is True.
    For other languages, words are split by whitespace.

    Returns a list of words with their start and end indices of the original text.
    """
    words: list[tuple[str, int, int]] = []

    # CJK: \u4e00-\u9fff, \u3040-\u30ff, \u3400-\u4dbf
    # Thai: \u0E00-\u0E7F
    char_based_codes = (
        re.compile(
            r"[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf"  # CJK scripts
            r"\u0E00-\u0E7F]"  # Thai
        )
        if split_character
        else None
    )

    pos = 0
    word_start = 0

    def _add_current_word(start: int, end: int) -> None:
        word = text[start:end]
        if ignore_punctuation and word:
            word = "".join(c for c in word if not tokenizer.is_punctuation(c))

        if word:
            words.append((word, start, end))

    for pos, char in enumerate(text):
        if char.isspace():
            if retain_format and not text[word_start:pos].strip():
                continue

            # reached whitespace, commit current word
            _add_current_word(word_start, pos)
            word_start = pos if retain_format else pos + 1

        elif char_based_codes and char_based_codes.match(char):
            if word_start < pos:
                _add_current_word(word_start, pos)

            # commit character as a word
            _add_current_word(pos, pos + 1)
            word_start = pos + 1

    # add the last word if there is one
    _add_current_word(word_start, len(text))

    return words
