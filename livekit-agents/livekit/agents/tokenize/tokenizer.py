"""
Text tokenization interfaces and utilities.

Provides abstract classes for sentence and word tokenization with streaming support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from ..utils import aio

# Common punctuation characters for text processing
PUNCTUATIONS = [
    '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', 
    '|', '}', '~', '±', '—', '‘', '’', '“', '”', '…'
]


@dataclass
class TokenData:
    """Container for tokenized text data with metadata."""
    segment_id: str = ""  # Identifier for text segments
    token: str = ""       # Actual token content


class SentenceTokenizer(ABC):
    """Abstract base class for sentence tokenization implementations."""
    
    @abstractmethod
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Split text into sentences.
        
        Args:
            text: Input text to tokenize
            language: Optional language code for localization
            
        Returns:
            List of sentences
        """
        pass

    @abstractmethod
    def stream(self, *, language: str | None = None) -> "SentenceStream":
        """Create a streaming tokenizer instance.
        
        Args:
            language: Optional language code for localization
            
        Returns:
            Configured SentenceStream
        """
        pass


class SentenceStream(ABC):
    """Base class for streaming sentence tokenization."""
    
    def __init__(self) -> None:
        self._event_ch = aio.Chan[TokenData]()  # Async channel for token output

    @abstractmethod
    def push_text(self, text: str) -> None:
        """Add text to the tokenization buffer."""
        
    @abstractmethod
    def flush(self) -> None:
        """Force processing of current buffer as complete sentences."""
        
    @abstractmethod
    def end_input(self) -> None:
        """Signal end of input data."""
        
    @abstractmethod
    async def aclose(self) -> None:
        """Clean up resources and close the stream."""
        
    async def __anext__(self) -> TokenData:
        """Get next token from stream."""
        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[TokenData]:
        """Support async iteration over tokens."""
        return self

    def _do_close(self) -> None:
        """Internal method to close the event channel."""
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        """Verify stream is still active."""
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")


class WordTokenizer(ABC):
    """Abstract base class for word tokenization implementations."""
    
    @abstractmethod
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        """Split text into words.
        
        Args:
            text: Input text to tokenize
            language: Optional language code for localization
            
        Returns:
            List of words
        """
        pass

    @abstractmethod
    def stream(self, *, language: str | None = None) -> "WordStream":
        """Create a streaming word tokenizer.
        
        Args:
            language: Optional language code for localization
            
        Returns:
            Configured WordStream
        """
        pass

    def format_words(self, words: list[str]) -> str:
        """Reconstruct text from tokenized words.
        
        Args:
            words: List of tokenized words
            
        Returns:
            Properly spaced text string
        """
        return " ".join(words)


class WordStream(ABC):
    """Base class for streaming word tokenization."""
    
    def __init__(self) -> None:
        self._event_ch = aio.Chan[TokenData]()  # Async channel for token output

    @abstractmethod
    def push_text(self, text: str) -> None:
        """Add text to the tokenization buffer."""
        
    @abstractmethod
    def flush(self) -> None:
        """Force processing of current buffer as complete words."""
        
    @abstractmethod
    def end_input(self) -> None:
        """Signal end of input data."""
        
    @abstractmethod
    async def aclose(self) -> None:
        """Clean up resources and close the stream."""
        
    async def __anext__(self) -> TokenData:
        """Get next token from stream."""
        return await self._event_ch.__anext__()

    def __aiter__(self) -> AsyncIterator[TokenData]:
        """Support async iteration over tokens."""
        return self

    def _do_close(self) -> None:
        """Internal method to close the event channel."""
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        """Verify stream is still active."""
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")
