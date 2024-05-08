import asyncio
from typing import Callable

from .tokenizer import TokenStream


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
        self._event_queue = asyncio.Queue[str | None]()
        self._closed = False

        self._incomplete_tokens: list[str] = []  # <= min_token_len
        self._buffer = ""

    def push_text(self, text: str | None) -> None:
        if self._closed:
            raise ValueError("cannot push text to closed stream")

        if text is None:
            self._flush()
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
                self._event_queue.put_nowait(s)
                self._incomplete_tokens = []

            self._buffer = self._buffer[len(new_token) :].lstrip()

    def mark_segment_end(self) -> None:
        self.push_text(None)

    def _flush(self) -> None:
        # try to segment the remaining data inside self._text_buffer
        buff = " ".join(self._incomplete_tokens)
        tokens = self._tokenizer(self._buffer)
        for t in tokens:
            buff += " " + t
            if len(buff) >= self._min_token_len:
                self._event_queue.put_nowait(buff)
                buff = ""

        if buff:
            self._event_queue.put_nowait(buff.lstrip())

    async def aclose(self, *, wait: bool = True) -> None:
        self._closed = True
        self._flush()
        self._event_queue.put_nowait(None)

    def __aiter__(self) -> "BufferedTokenStream":
        return self

    async def __anext__(self) -> str:
        event = await self._event_queue.get()
        if event is None:
            raise StopAsyncIteration

        return event
