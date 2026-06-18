import re
from collections.abc import AsyncIterable, Callable, Sequence
from typing import Literal

from .filters import filter_emoji, filter_markdown

TextTransforms = (
    Literal["filter_markdown", "filter_emoji"] | Callable[[AsyncIterable[str]], AsyncIterable[str]]
)

_BUILTIN_TRANSFORMS: dict[str, Callable[[AsyncIterable[str]], AsyncIterable[str]]] = {
    "filter_markdown": lambda text: filter_markdown(text),
    "filter_emoji": lambda text: filter_emoji(text),
}


def _apply_text_transforms(
    text: AsyncIterable[str], transforms: Sequence[TextTransforms]
) -> AsyncIterable[str]:
    for transform in transforms:
        if isinstance(transform, str):
            if transform not in _BUILTIN_TRANSFORMS:
                raise ValueError(
                    f"Invalid transform: {transform}, available transforms: {_BUILTIN_TRANSFORMS.keys()}"
                )
            text = _BUILTIN_TRANSFORMS[transform](text)
        elif callable(transform):
            text = transform(text)
        else:
            raise ValueError(f"Invalid transform: {transform}, must be a string or callable")
    return text


def replace(
    replacements: dict[str, str], case_sensitive: bool = False
) -> Callable[[AsyncIterable[str]], AsyncIterable[str]]:
    """Create a text transform that replaces words with new words.

    Buffers enough text to handle terms that might be split across token boundaries
    during streaming.

    Args:
        replacements: A dictionary mapping words to their new words.
        case_sensitive: Whether to match the case of the words.
    Returns:
        A callable that can be used as a ``TextTransforms`` entry.
    """
    tail_len = max(len(k) for k in replacements) - 1 if replacements else 0
    flags = re.IGNORECASE if not case_sensitive else 0

    async def _transform(text: AsyncIterable[str]) -> AsyncIterable[str]:
        buffer = ""
        async for chunk in text:
            buffer += chunk
            if len(buffer) <= tail_len:
                continue
            for old, new in replacements.items():
                buffer = re.sub(re.escape(old), lambda _, r=new: r, buffer, flags=flags)

            flush_to = len(buffer) - tail_len
            yield buffer[:flush_to]
            buffer = buffer[flush_to:]

        if buffer:
            for old, new in replacements.items():
                buffer = re.sub(re.escape(old), lambda _, r=new: r, buffer, flags=flags)
            yield buffer

    return _transform
