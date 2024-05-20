from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class TokenEventType(Enum):
    STARTED = 0
    TOKEN = 1
    FINISHED = 2


@dataclass
class TokenEvent:
    type: TokenEventType
    token: str = ""


class TokenStream(Protocol):
    def push_text(self, text: str | None) -> None: ...

    def mark_segment_end(self) -> None:
        self.push_text(None)

    async def aclose(self, *, wait: bool = True) -> None: ...

    @abstractmethod
    async def __anext__(self) -> TokenEvent:
        pass

    def __aiter__(self) -> "WordStream":
        return self


class SentenceTokenizer(ABC):
    @abstractmethod
    def tokenize(self, *, text: str, language: str | None = None) -> list[str]:
        pass

    @abstractmethod
    def stream(self, *, language: str | None = None) -> "SentenceStream":
        pass


class SentenceStream(TokenStream, Protocol): ...


class WordTokenizer(ABC):
    @abstractmethod
    def tokenize(self, *, text: str, language: str | None = None) -> list[str]:
        pass

    @abstractmethod
    def stream(self, *, language: str | None = None) -> "WordStream":
        pass

    def format_words(self, words: list[str]) -> str:
        return " ".join(words)


class WordStream(TokenStream, Protocol): ...
