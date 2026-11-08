from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable
from typing import overload

from . import _core
from ._generated import public_tokenize as _public
from ._generated.public_tokenize import *  # noqa: F403
from .utils import aio, shortuuid

__all__ = [
    "SentenceTokenizer",
    "SentenceStream",
    "WordTokenizer",
    "WordStream",
    "BufferedSentenceStream",
    "BufferedWordStream",
    "replace_words",
    "basic",
    "blingfire",
]
__all__ += _public.__all__


class SentenceTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        pass

    @abstractmethod
    def stream(self, *, language: str | None = None) -> SentenceStream:
        pass


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


class WordTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        pass

    @abstractmethod
    def stream(self, *, language: str | None = None) -> WordStream:
        pass

    def format_words(self, words: list[str]) -> str:
        return " ".join(words)


class WordStream(_TokenStream):
    pass


# a tokenizer returns tokens, or tokens with their start and end offsets in the text
TokenizeCallable = Callable[[str], list[str] | list[TokenSpan]]


class BufferedTokenStream(_TokenStream):
    """Streamed text cut into tokens by a tokenizer."""

    def __init__(
        self,
        *,
        tokenize_fnc: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
        max_token_len: int | None = None,
        xml_aware: bool = False,
    ) -> None:
        """
        Args:
            tokenize_fnc: Cuts a text into tokens.
            min_token_len: The shortest token emitted; shorter ones are batched.
            min_ctx_len: The text buffered before tokenizing.
            max_token_len: The longest token emitted.
            xml_aware: Never split an XML tag across tokens; only when the input carries
                markup, as a stray "<" in plain text holds streaming until flush.
        """
        super().__init__()
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


class _CoreSentenceTokenizer(SentenceTokenizer):
    _native: _core.BasicSentenceTokenizer | _core.BlingfireSentenceTokenizer

    def stream(self, *, language: str | None = None) -> SentenceStream:
        return BufferedSentenceStream(
            tokenizer=self._native._core_split,
            min_token_len=self._native._core_min_token_len,
            min_ctx_len=self._native._core_min_ctx_len,
            max_token_len=self._native._core_max_token_len,
            xml_aware=self._native._core_xml_aware,
        )


class _CoreWordTokenizer(WordTokenizer):
    _native: _core.BasicWordTokenizer

    def stream(self, *, language: str | None = None) -> WordStream:
        return BufferedWordStream(
            tokenizer=self._native._core_split,
            min_token_len=self._native._core_min_token_len,
            min_ctx_len=self._native._core_min_ctx_len,
        )


# imported last: these modules subclass the definitions above
from ._generated import (  # noqa: E402
    public_tokenize_basic as basic,
    public_tokenize_blingfire as blingfire,
)
