import re
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Sequence
from typing import Any, Literal

from .filters import filter_emoji, filter_markdown

TextTransforms = (
    Literal["filter_markdown", "filter_emoji"] | Callable[[AsyncIterable[str]], AsyncIterable[str]]
)

_BUILTIN_TRANSFORMS: dict[str, Callable[[AsyncIterable[str]], AsyncIterable[str]]] = {
    "filter_markdown": lambda text: filter_markdown(text),
    "filter_emoji": lambda text: filter_emoji(text),
}


def _apply_text_transforms(
    text: AsyncIterable[Any], transforms: Sequence[TextTransforms]
) -> AsyncIterable[Any]:
    # Resolve + validate eagerly so a bad transform name raises at call time.
    resolved: list[Callable[[AsyncIterable[str]], AsyncIterable[str]]] = []
    for transform in transforms:
        if isinstance(transform, str):
            if transform not in _BUILTIN_TRANSFORMS:
                raise ValueError(
                    f"Invalid transform: {transform}, available transforms: {_BUILTIN_TRANSFORMS.keys()}"
                )
            resolved.append(_BUILTIN_TRANSFORMS[transform])
        elif callable(transform):
            resolved.append(transform)
        else:
            raise ValueError(f"Invalid transform: {transform}, must be a string or callable")

    async def _gen() -> AsyncGenerator[Any, None]:
        aiter = text.__aiter__()
        try:
            first = await aiter.__anext__()
        except StopAsyncIteration:
            return

        async def _rechained() -> AsyncGenerator[Any, None]:
            yield first
            async for chunk in aiter:
                yield chunk

        # The transforms are str-only and stateful (they buffer across chunks). A
        # stream carrying non-str chunks — e.g. structured-output BaseModel deltas —
        # is passed through untouched; the producer emits homogeneous streams
        # (all-str plain text, or all-BaseModel structured output), so the first
        # chunk's type decides the whole stream.
        if not isinstance(first, str):
            async for chunk in _rechained():
                yield chunk
            return

        stream: AsyncIterable[str] = _rechained()
        for fn in resolved:
            stream = fn(stream)
        async for chunk in stream:
            yield chunk

    return _gen()


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
