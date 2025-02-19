"""
Buffered tokenization streams for incremental text processing.

Implements buffered word and sentence tokenization with configurable thresholds.
"""

from __future__ import annotations

import typing
from typing import Callable, Union

from ..utils import aio, shortuuid
from .tokenizer import SentenceStream, TokenData, WordStream

# Type alias for tokenization functions that can return either:
# - List of tokens
# - List of (token, start_index, end_index) tuples
TokenizeCallable = Callable[[str], Union[list[str], list[tuple[str, int, int]]]]


class BufferedTokenStream:
    """Base class for buffered token streams with configurable thresholds.
    
    Features:
    - Accumulates text until minimum context length is reached
    - Splits text into tokens using provided tokenization function
    - Maintains output buffer until minimum token length is met
    - Handles punctuation and whitespace properly
    """

    def __init__(
        self,
        *,
        tokenize_fnc: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        """
        Args:
            tokenize_fnc: Function to split text into tokens
            min_token_len: Minimum length before emitting a token
            min_ctx_len: Minimum context length before processing
        """
        self._event_ch = aio.Chan[TokenData]()
        self._tokenize_fnc = tokenize_fnc
        self._min_ctx_len = min_ctx_len
        self._min_token_len = min_token_len
        self._current_segment_id = shortuuid()  # Unique ID for text segments

        # Buffers for input processing and output accumulation
        self._buf_tokens: list[str] = []  # Temporary token storage
        self._in_buf = ""  # Raw input buffer
        self._out_buf = ""  # Processed output buffer

    @typing.no_type_check
    def push_text(self, text: str) -> None:
        """Add text to the input buffer and process if possible."""
        self._check_not_closed()
        self._in_buf += text

        # Wait until enough context is available
        if len(self._in_buf) < self._min_ctx_len:
            return

        # Process input in chunks
        while True:
            tokens = self._tokenize_fnc(self._in_buf)
            if len(tokens) <= 1:
                break  # Need more input

            # Handle accumulated output
            if self._out_buf:
                self._out_buf += " "

            # Extract first token
            tok = tokens.pop(0)
            tok_text = tok
            if isinstance(tok, tuple):
                tok_text = tok[0]

            # Build output buffer
            self._out_buf += tok_text
            if len(self._out_buf) >= self._min_token_len:
                # Emit when reaching minimum token length
                self._event_ch.send_nowait(
                    TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                )
                self._out_buf = ""

            # Update input buffer based on token position
            if isinstance(tok, tuple):
                # Use explicit indices if available
                self._in_buf = self._in_buf[tok[2] :]
            else:
                # Fallback to string search
                tok_i = max(self._in_buf.find(tok), 0)
                self._in_buf = self._in_buf[tok_i + len(tok) :].lstrip()

    @typing.no_type_check
    def flush(self) -> None:
        """Force processing of remaining text in buffers."""
        self._check_not_closed()

        if self._in_buf or self._out_buf:
            tokens = self._tokenize_fnc(self._in_buf)
            if tokens:
                # Combine remaining tokens
                if self._out_buf:
                    self._out_buf += " "

                if isinstance(tokens[0], tuple):
                    self._out_buf += " ".join([tok[0] for tok in tokens])
                else:
                    self._out_buf += " ".join(tokens)

            # Emit any remaining output
            if self._out_buf:
                self._event_ch.send_nowait(
                    TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                )

            # Reset for next segment
            self._current_segment_id = shortuuid()

        # Clear buffers
        self._in_buf = ""
        self._out_buf = ""

    def end_input(self) -> None:
        """Finalize processing and close the stream."""
        self.flush()
        self._event_ch.close()

    async def aclose(self) -> None:
        """Clean up resources and close the channel."""
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        """Verify stream is still active."""
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def __aiter__(self) -> "BufferedTokenStream":
        """Support async iteration over tokens."""
        return self

    async def __anext__(self) -> TokenData:
        """Get next token from the stream."""
        return await self._event_ch.__anext__()


class BufferedSentenceStream(BufferedTokenStream, SentenceStream):
    """Buffered stream implementation for sentence tokenization."""
    
    def __init__(
        self,
        *,
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
        )


class BufferedWordStream(BufferedTokenStream, WordStream):
    """Buffered stream implementation for word tokenization."""
    
    def __init__(
        self,
        *,
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
        )
