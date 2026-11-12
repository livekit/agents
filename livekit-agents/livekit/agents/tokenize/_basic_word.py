import re

from . import tokenizer

# full-width clause and sentence punctuation. CJK text has no spaces, so with
# split_cjk_clauses these are the only word boundaries a run of it has
_CJK_CLAUSE_PUNCTUATION = frozenset("，。！？；：、")
# ideographs and kana, the scripts written without spaces between words
_CJK_SCRIPTS = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
# split_character makes each of these characters a word
_CHAR_BASED_CODES = re.compile(rf"[{_CJK_SCRIPTS}\u0e00-\u0e7f]")  # + thai
# such a mark only ends a word after CJK text, so it never splits markup that a TTS
# rejoins with spaces, e.g. an SSML attribute like ph="..."
_CJK_TEXT = re.compile(
    rf"[{_CJK_SCRIPTS}"
    r"\u3000-\u303f"  # cjk symbols and punctuation, e.g. 」
    r"\uf900-\ufaff"  # cjk compatibility ideographs
    r"\uac00-\ud7af"  # hangul syllables
    r"\uff00-\uffef]"  # halfwidth and fullwidth forms
)


def split_words(
    text: str,
    *,
    ignore_punctuation: bool = True,
    split_character: bool = False,
    retain_format: bool = False,
    split_cjk_clauses: bool = False,
) -> list[tuple[str, int, int]]:
    """
    Split text into words, supporting both space-separated languages (like English)
    and character-based languages (like Chinese, Japanese, Korean, Thai).

    For non-spaced scripts, each character is treated as a separate word if split_character is True.
    For other languages, words are split by whitespace. With split_cjk_clauses, a run of CJK
    text also ends a word at full-width punctuation such as "，" or "。" that follows it, and
    the mark stays on the word.

    Returns a list of words with their start and end indices of the original text.
    """
    words: list[tuple[str, int, int]] = []

    char_based_codes = _CHAR_BASED_CODES if split_character else None

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

        elif (
            split_cjk_clauses
            and char in _CJK_CLAUSE_PUNCTUATION
            and pos > 0
            and _CJK_TEXT.match(text[pos - 1])
            # a "，" or "：" inside a full-width number like １，０００ or １０：３０, which a
            # TTS would stop reading as one number once the words are joined with a space
            and not (char in "，：" and text[pos - 1].isdigit())
        ):
            # commit the word with its punctuation, otherwise a CJK reply is a single
            # word and a word stream (e.g. a TTS input) holds it until the end
            _add_current_word(word_start, pos + 1)
            word_start = pos + 1

    # add the last word if there is one
    _add_current_word(word_start, len(text))

    return words
