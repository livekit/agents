from __future__ import annotations

from collections.abc import Callable

from .. import _core
from ..utils import aio, shortuuid
from .tokenizer import SentenceStream, TokenData, WordStream

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
