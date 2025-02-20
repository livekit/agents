"""
Text processing utilities for word-level manipulation.

Provides functionality for replacing words in text while preserving original casing.
"""

from __future__ import annotations

from typing import AsyncIterable, overload

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
    """Replace words in text while preserving original casing and punctuation.
    
    Features:
    - Case-insensitive matching with case preservation
    - Handles trailing punctuation
    - Works with both static text and async streams
    - Maintains word boundaries
    
    Args:
        text: Input text or async stream of text chunks
        replacements: Dictionary of {lowercase_word: replacement} mappings
        
    Returns:
        Processed text in same format as input (string or async iterable)
        
    Example:
        replace_words(
            text="Hello world!",
            replacements={"hello": "hi", "world": "earth"}
        )  # Returns "Hi earth!"
    """
    # Normalize replacement keys to lowercase
    replacements = {k.lower(): v for k, v in replacements.items()}

    def _process_words(text: str, words: list[tuple[str, int, int]]) -> tuple[str, int]:
        """Internal word processing with offset tracking."""
        offset = 0  # Tracks cumulative changes in text length
        processed_index = 0  # Last processed character index
        
        for word, start_index, end_index in words:
            # Separate word from trailing punctuation
            no_punctuation = word.rstrip("".join(tokenizer.PUNCTUATIONS))
            punctuation_off = len(word) - len(no_punctuation)
            
            # Get replacement if exists
            replacement = replacements.get(no_punctuation.lower())
            
            if replacement:
                # Apply replacement with original punctuation
                text = (
                    text[: start_index + offset]
                    + replacement
                    + text[end_index + offset - punctuation_off :]
                )
                # Update offset for subsequent replacements
                offset += len(replacement) - len(word) + punctuation_off

            processed_index = end_index + offset

        return text, processed_index

    # Synchronous text processing
    if isinstance(text, str):
        words = _basic_word.split_words(text, ignore_punctuation=False)
        processed_text, _ = _process_words(text, words)
        return processed_text

    # Asynchronous text processing
    else:
        async def _replace_words() -> AsyncIterable[str]:
            """Async generator for processing text streams."""
            buffer = ""
            async for chunk in text:
                buffer += chunk
                words = _basic_word.split_words(buffer, ignore_punctuation=False)

                if len(words) <= 1:
                    continue  # Wait for more input to form complete words

                # Process complete words and yield completed portion
                buffer, procesed_index = _process_words(buffer, words[:-1])
                yield buffer[:procesed_index]
                buffer = buffer[procesed_index:]

            # Process remaining buffer
            if buffer:
                words = _basic_word.split_words(buffer, ignore_punctuation=False)
                buffer, _ = _process_words(buffer, words)
                yield buffer

        return _replace_words()
