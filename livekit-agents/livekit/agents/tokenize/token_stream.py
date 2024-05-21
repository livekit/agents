from __future__ import annotations

import asyncio
from typing import Callable, Optional

from .tokenizer import TokenEvent, TokenEventType, TokenStream


class BufferedTokenStream(TokenStream):
    def __init__(
        self,
        *,
        tokenizer: Callable[[str], list[str]],
        min_token_len: int,
        ctx_len: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._ctx_len = ctx_len
        self._min_token_len = min_token_len
        self._event_queue = asyncio.Queue[Optional[TokenEvent]]()
        self._closed = False

        self._incomplete_tokens: list[str] = []  # <= min_token_len
        self._buffer = ""
        self._new_segment = True

    def push_text(self, text: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push text to closed stream")

        if self._new_segment:
            self._new_segment = False
            self._event_queue.put_nowait(TokenEvent(type=TokenEventType.STARTED))

        if text is None:
            self._flush()
            self._event_queue.put_nowait(TokenEvent(type=TokenEventType.FINISHED))
            self._new_segment = True
            return

        for char in text:
            self._buffer += char

            if len(self._buffer) < self._ctx_len:
                continue

            tokens = self._tokenizer(self._buffer)
            if len(tokens) < 2:
                continue

            new_token = tokens[0]
            self._incomplete_tokens.append(new_token)
            s = " ".join(self._incomplete_tokens)

            if len(s) >= self._min_token_len:
                self._put_token(s)
                self._incomplete_tokens = []

            real_len = self._buffer.find(new_token) + len(new_token)
            self._buffer = self._buffer[real_len:][1:]

    def mark_segment_end(self) -> None:
        self.push_text(None)

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        self._flush()
        self._event_queue.put_nowait(None)

    def _flush(self) -> None:
        # try to segment the remaining data inside self._text_buffer
        ibuff = " ".join(self._incomplete_tokens)
        buff = ibuff
        tokens = self._tokenizer(self._buffer)
        start = 0
        for t in tokens:
            if not ibuff:
                start = 1

            buff += " " + t
            if len(buff) >= self._min_token_len:
                self._put_token(buff[start:])
                buff = ""

        if buff:
            self._put_token(buff[start:])

        self._buffer = ""

    def _put_token(self, token: str) -> None:
        self._event_queue.put_nowait(TokenEvent(type=TokenEventType.TOKEN, token=token))

    def __aiter__(self) -> "BufferedTokenStream":
        return self

    async def __anext__(self) -> TokenEvent:
        event = await self._event_queue.get()
        if event is None:
            raise StopAsyncIteration

        return event
