from __future__ import annotations

from typing import Callable

from ..utils import aio, shortuuid
from .tokenizer import SentenceStream, TokenData, WordStream


class BufferedTokenStream:
    def __init__(
        self,
        *,
        tokenize_fnc: Callable[[str], list[str]],
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        self._event_ch = aio.Chan[TokenData]()
        self._tokenize_fnc = tokenize_fnc
        self._min_ctx_len = min_ctx_len
        self._min_token_len = min_token_len
        self._current_segment_id = shortuuid()

        self._buf_tokens: list[str] = []  # <= min_token_len
        self._buf = ""

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        self._buf += text

        if len(self._buf) < self._min_ctx_len:
            return

        tokens = self._tokenize_fnc(self._buf)

        buf_toks = []
        buf = ""
        while len(tokens) > 1:
            if buf:
                buf += " "

            tok = tokens.pop(0)
            buf += tok
            buf_toks.append(tok)
            if len(buf) >= self._min_token_len:
                self._event_ch.send_nowait(
                    TokenData(token=buf, segment_id=self._current_segment_id)
                )

                for i, tok in enumerate(buf_toks):
                    tok_i = self._buf.find(tok)
                    self._buf = self._buf[tok_i + len(tok) :].lstrip()

                buf_toks = []
                buf = ""

    def flush(self) -> None:
        self._check_not_closed()
        if self._buf:
            tokens = self._tokenize_fnc(self._buf)
            if tokens:
                buf = " ".join(tokens)
            else:
                buf = self._buf

            self._event_ch.send_nowait(
                TokenData(token=buf, segment_id=self._current_segment_id)
            )
            self._current_segment_id = shortuuid()

        self._buf = ""

    def end_input(self) -> None:
        self.flush()
        self._event_ch.close()

    async def aclose(self) -> None:
        self._event_ch.close()

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def __aiter__(self) -> "BufferedTokenStream":
        return self

    async def __anext__(self) -> TokenData:
        return await self._event_ch.__anext__()


class BufferedSentenceStream(BufferedTokenStream, SentenceStream):
    def __init__(
        self,
        *,
        tokenizer: Callable[[str], list[str]],
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
        )


class BufferedWordStream(BufferedTokenStream, WordStream):
    def __init__(
        self,
        *,
        tokenizer: Callable[[str], list[str]],
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
        )
