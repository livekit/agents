from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
from typing import overload

from .. import _core


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

    if isinstance(text, str):
        return _core.replace_words(text, replacements)[0]  # type: ignore[no-any-return]

    async def _replace_words() -> AsyncGenerator[str, None]:
        # the last word may still grow, so it is held until the next chunk
        buffer = ""
        async for chunk in text:
            buffer += chunk
            if (replaced := _core.replace_words(buffer, replacements, True)) is not None:
                buffer, processed = replaced
                yield buffer[:processed]
                buffer = buffer[processed:]
        if buffer:
            yield _core.replace_words(buffer, replacements)[0]

    return _replace_words()
