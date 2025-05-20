from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
from typing import overload

from . import _basic_word, tokenizer


@overload
def replace_words(
    *,
    text: str,
    replacements: dict[str, str],
) -> str: ...


@overload
def replace_words(
    *,
    text: AsyncIterable[str],
    replacements: dict[str, str],
) -> AsyncIterable[str]: ...


def replace_words(
    *,
    text: str | AsyncIterable[str],
    replacements: dict[str, str],
) -> str | AsyncIterable[str]:
    """
    Replace words in the given (async) text. The replacements are case-insensitive and the
    replacement will keep the case of the original word.
    Args:
        text: text to replace words in
        words: dictionary of words to replace
    """

    replacements = {k.lower(): v for k, v in replacements.items()}

    def _process_words(text: str, words: list[tuple[str, int, int]]) -> tuple[str, int]:
        offset = 0
        processed_index = 0
        for word, start_index, end_index in words:
            no_punctuation = word.rstrip("".join(tokenizer.PUNCTUATIONS))
            punctuation_off = len(word) - len(no_punctuation)
            replacement = replacements.get(no_punctuation.lower())
            if replacement:
                text = (
                    text[: start_index + offset]
                    + replacement
                    + text[end_index + offset - punctuation_off :]
                )
                offset += len(replacement) - len(word) + punctuation_off

            processed_index = end_index + offset

        return text, processed_index

    if isinstance(text, str):
        words = _basic_word.split_words(text, ignore_punctuation=False)
        text, _ = _process_words(text, words)
        return text
    else:

        async def _replace_words() -> AsyncGenerator[str, None]:
            buffer = ""
            async for chunk in text:
                buffer += chunk
                words = _basic_word.split_words(buffer, ignore_punctuation=False)

                if len(words) <= 1:
                    continue

                buffer, procesed_index = _process_words(buffer, words[:-1])
                yield buffer[:procesed_index]
                buffer = buffer[procesed_index:]

            if buffer:
                words = _basic_word.split_words(buffer, ignore_punctuation=False)
                buffer, _ = _process_words(buffer, words)
                yield buffer

        return _replace_words()
