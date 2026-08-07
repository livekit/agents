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

    Buffers only a trailing key-prefix while streaming, so a key split across
    chunks is still matched but ordinary words are never split.

    Args:
        replacements: A dictionary mapping words to their new words.
        case_sensitive: Whether to match the case of the words.
    Returns:
        A callable that can be used as a ``TextTransforms`` entry.
    """
    flags = re.IGNORECASE if not case_sensitive else 0
    keys = list(replacements)
    lookup = replacements if case_sensitive else {k.lower(): v for k, v in replacements.items()}
    # single-pass alternation, longest key first so overlaps prefer the longer match
    pattern = (
        re.compile("|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True)), flags)
        if keys
        else None
    )
    # longest end-anchored run that is a proper prefix of a key (a later chunk may complete it)
    max_prefix = max((len(k) - 1 for k in keys), default=0)
    prefixes = {k[:n] for k in keys for n in range(1, len(k))}
    holdback_re = (
        re.compile("(?:" + "|".join(re.escape(p) for p in prefixes) + r")\Z", flags)
        if prefixes
        else None
    )

    def _apply(text: str) -> str:
        if pattern is None:
            return text
        # map each match to its value literally (no backreference expansion)
        return pattern.sub(
            lambda m: lookup[m.group(0) if case_sensitive else m.group(0).lower()], text
        )

    def _holdback(buffer: str) -> int:
        if holdback_re is None:
            return 0
        # only the last max_prefix chars can begin a match
        m = holdback_re.search(buffer[-max_prefix:])
        return len(m.group()) if m else 0

    async def _transform(text: AsyncIterable[str]) -> AsyncIterable[str]:
        buffer = ""
        async for chunk in text:
            # substitute complete matches, then hold back a trailing partial-key run
            buffer = _apply(buffer + chunk)
            flush_to = len(buffer) - _holdback(buffer)
            if flush_to > 0:
                yield buffer[:flush_to]
                buffer = buffer[flush_to:]

        if buffer:
            yield buffer

    return _transform
