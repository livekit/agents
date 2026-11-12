from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable
from typing import overload

from . import _core, _generated
from ._generated.tokenizer import TokenData, TokenSpan
from .utils import aio, shortuuid

__all__ = [
    "SentenceStream",
    "WordStream",
    "BufferedSentenceStream",
    "BufferedWordStream",
    "replace_words",
]


class _TokenStream(ABC):
    def __init__(self) -> None:
        self._event_ch = aio.Chan[TokenData]()

    @abstractmethod
    def push_text(self, text: str) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def end_input(self) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    async def __anext__(self) -> TokenData:
        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[TokenData]:
        return self

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    @property
    def closed(self) -> bool:
        return self._event_ch.closed


class SentenceStream(_TokenStream):
    pass


class WordStream(_TokenStream):
    pass


# a tokenizer returns tokens, or tokens with their start and end offsets in the text
TokenizeCallable = Callable[[str], list[str] | list[TokenSpan]]


class _NativeTokenStream(_TokenStream):
    """The tokens a native buffer emits as the text streams in."""

    def __init__(
        self, buffer: _core.TokenBuffer | _core.SpeechSentenceStream | _core.SpeechWordStream
    ) -> None:
        super().__init__()
        self._buffer = buffer
        self._current_segment_id = shortuuid()

    def _send(self, tokens: list[str]) -> None:
        for token in tokens:
            self._event_ch.send_nowait(TokenData(token=token, segment_id=self._current_segment_id))

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        if text:
            self._send(self._buffer.push(text))

    def flush(self) -> None:
        self._check_not_closed()
        self._send(self._buffer.flush())
        self._current_segment_id = shortuuid()

    def end_input(self) -> None:
        self.flush()
        self._event_ch.close()

    async def aclose(self) -> None:
        self._event_ch.close()


class BufferedTokenStream(_NativeTokenStream):
    """Streamed text cut into tokens by a tokenizer."""

    def __init__(
        self,
        *,
        tokenize_fnc: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
    ) -> None:
        """
        Args:
            tokenize_fnc: Cuts a text into tokens.
            min_token_len: The shortest token emitted; shorter ones are batched.
            min_ctx_len: The text buffered before tokenizing.
            max_token_len: The longest token emitted.
        """
        super().__init__(
            _core.TokenBuffer(
                tokenize_fnc,
                min_token_len=min_token_len,
                min_ctx_len=min_ctx_len,
                max_token_len=max_token_len,
            )
        )


class BufferedSentenceStream(BufferedTokenStream, SentenceStream):
    def __init__(
        self,
        *,
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
            max_token_len=max_token_len,
        )


class BufferedWordStream(BufferedTokenStream, WordStream):
    def __init__(
        self, *, tokenizer: TokenizeCallable, min_token_len: int, min_ctx_len: int
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer, min_token_len=min_token_len, min_ctx_len=min_ctx_len
        )


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
    """Replace words in a text or a text stream, case-insensitively, keeping each original
    word's case.

    Args:
        text: The text, or a stream of text chunks.
        replacements: Words to replace, mapped to their replacements.
    """

    def replace_all(text: str) -> str:
        replaced = _core.replace_words(text, replacements)
        assert replaced is not None, "only the streaming mode holds a word back"
        return replaced.text

    if isinstance(text, str):
        return replace_all(text)

    async def _replace_words() -> AsyncGenerator[str, None]:
        # the last word may still grow, so it is held until the next chunk
        buffer = ""
        async for chunk in text:
            buffer += chunk
            if (replaced := _core.replace_words(buffer, replacements, True)) is not None:
                yield replaced.text[: replaced.processed]
                buffer = replaced.text[replaced.processed :]
        if buffer:
            yield replace_all(buffer)

    return _replace_words()


class _SpeechSentenceStream(_NativeTokenStream, SentenceStream):
    """`SentenceTokenizer`'s stream: a sentence goes out once no text to come could change it."""


class _SentenceTokenizerBase:
    """What `SentenceTokenizer` streams with: its own native stream, or the `tokenize` of a
    subclass that overrides it."""

    _native: _core.SpeechSentenceTokenizer

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        raise NotImplementedError

    def stream(self, *, language: str | None = None) -> SentenceStream:
        native = self._native
        # A subclass's own `tokenize` decides its sentences; the generated ones are native.
        if type(self).tokenize.__module__.startswith(_generated.__name__):
            return _SpeechSentenceStream(native._core_stream())
        # each of the subclass's sentences goes out once the next one starts
        return BufferedSentenceStream(
            tokenizer=functools.partial(self.tokenize, language=language),
            min_token_len=1,
            min_ctx_len=1,
            max_token_len=native._core_max_token_len,
        )


class _SpeechWordStream(_NativeTokenStream, WordStream):
    """`WordTokenizer`'s stream: a word goes out once the next one proves it whole."""


class _WordTokenizerBase:
    """What `WordTokenizer` streams with: its own native stream, or the `tokenize` of a
    subclass that overrides it."""

    _native: _core.SpeechWordTokenizer

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        raise NotImplementedError

    def stream(self, *, language: str | None = None) -> WordStream:
        # A subclass's own `tokenize` decides its words; the generated one is native.
        if type(self).tokenize.__module__.startswith(_generated.__name__):
            return _SpeechWordStream(self._native._core_stream())
        # each of the subclass's words goes out once the next one starts
        return BufferedWordStream(
            tokenizer=functools.partial(self.tokenize, language=language),
            min_token_len=1,
            min_ctx_len=1,
        )

    def format_words(self, words: list[str]) -> str:
        return " ".join(words)


# imported last: these modules subclass the definitions above
from ._generated import public_tokenize as _public  # noqa: E402
from ._generated.public_tokenize import *  # noqa: E402, F403

__all__ += _public.__all__
