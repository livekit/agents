from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..utils import aio

# fmt: off
PUNCTUATIONS = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',  # noqa: E501
                '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '±', '—', '‘', '’', '“', '”', '…']  # noqa: E501

# fmt: on


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

    def _do_close(self) -> None:
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")


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

    def _do_close(self) -> None:
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")
