from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, overload

from .. import _core
from ..utils import aio, shortuuid

__all__ = [
    "SentenceTokenizer",
    "SentenceStream",
    "WordTokenizer",
    "WordStream",
    "TokenData",
    "BufferedSentenceStream",
    "BufferedWordStream",
    "replace_words",
    "basic",
    "blingfire",
]


@dataclass
class TokenData:
    segment_id: str = ""
    token: str = ""


class SentenceTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        pass

    @abstractmethod
    def stream(self, *, language: str | None = None) -> SentenceStream:
        pass


class SentenceStream(ABC):
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


class WordTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        pass

    @abstractmethod
    def stream(self, *, language: str | None = None) -> WordStream:
        pass

    def format_words(self, words: list[str]) -> str:
        return " ".join(words)


class WordStream(ABC):
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


# a tokenizer returns tokens, or tokens with their start and end offsets in the text
TokenizeCallable = Callable[[str], list[str] | list[tuple[str, int, int]]]


class BufferedTokenStream:
    """Streamed text cut into tokens by a tokenizer; the batching is the core's
    (``_core.TokenBuffer``)."""

    def __init__(
        self,
        *,
        tokenize_fnc: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
        retain_format: bool = False,
        xml_aware: bool = False,
    ) -> None:
        """
        Args:
            tokenize_fnc: Cuts a text into tokens.
            min_token_len: The shortest token emitted; shorter ones are batched.
            min_ctx_len: The text buffered before tokenizing.
            max_token_len: The longest token emitted.
            retain_format: Unused; the tokenizer decides.
            xml_aware: Never split an XML tag across tokens; only when the input carries
                markup, as a stray "<" in plain text holds streaming until flush.
        """
        self._event_ch = aio.Chan[TokenData]()
        self._buffer = _core.TokenBuffer(
            tokenize_fnc,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
            max_token_len=max_token_len,
            xml_aware=xml_aware,
        )
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

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def __aiter__(self) -> BufferedTokenStream:
        return self

    async def __anext__(self) -> TokenData:
        return await self._event_ch.__anext__()


class BufferedSentenceStream(BufferedTokenStream, SentenceStream):
    def __init__(
        self,
        *,
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
        xml_aware: bool = False,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
            max_token_len=max_token_len,
            xml_aware=xml_aware,
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
    """
    Replace words in the given (async) text. The replacements are case-insensitive and the
    replacement will keep the case of the original word.
    Args:
        text: text to replace words in
        words: dictionary of words to replace
    """

    def replace_all(text: str) -> str:
        replaced = _core.replace_words(text, replacements)
        assert replaced is not None, "only the streaming mode holds a word back"
        return replaced[0]

    if isinstance(text, str):
        return replace_all(text)

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
            yield replace_all(buffer)

    return _replace_words()


class _CoreSentenceTokenizer(SentenceTokenizer):
    """A sentence tokenizer the core runs (``self._core``); a stream batches its spans."""

    _core: Any

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return self._core.tokenize(text, language=language)  # type: ignore[no-any-return]

    def stream(self, *, language: str | None = None) -> SentenceStream:
        return BufferedSentenceStream(
            tokenizer=self._core._core_split,
            min_token_len=self._core._core_min_token_len,
            min_ctx_len=self._core._core_min_ctx_len,
        )


class _CoreWordTokenizer(WordTokenizer):
    """A word tokenizer the core runs (``self._core``); a stream batches its spans."""

    _core: Any

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return self._core.tokenize(text, language=language)  # type: ignore[no-any-return]

    def stream(self, *, language: str | None = None) -> WordStream:
        return BufferedWordStream(
            tokenizer=self._core._core_split,
            min_token_len=self._core._core_min_token_len,
            min_ctx_len=self._core._core_min_ctx_len,
        )


# the tokenizers build on the definitions above
from .._generated import tokenize_basic as basic  # noqa: E402
from . import blingfire  # noqa: E402
