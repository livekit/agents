from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SegmentedSentence:
    text: str


class SentenceTokenizer(ABC):
    @abstractmethod
    def tokenize(
        self, *, text: str, language: Optional[str] = None
    ) -> List[SegmentedSentence]:
        pass

    @abstractmethod
    def stream(self, *, language: Optional[str] = None) -> "SentenceStream":
        pass


class SentenceStream(ABC):
    @abstractmethod
    def push_text(self, text: str) -> None:
        pass

    @abstractmethod
    async def flush(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    @abstractmethod
    async def __anext__(self) -> SegmentedSentence:
        pass

    def __aiter__(self) -> "SentenceStream":
        return self
