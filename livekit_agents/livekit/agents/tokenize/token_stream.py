from __future__ import annotations

import typing
from typing import Callable, Union

from ..utils import aio, shortuuid
from .tokenizer import SentenceStream, TokenData, WordStream

# Tokenizers can either provide us with a list of tokens or a list of tokens along with their start and end indices.  # noqa: E501
# If the start and end indices are not available, we attempt to locate the token within the text using str.find.  # noqa: E501
TokenizeCallable = Callable[[str], Union[list[str], list[tuple[str, int, int]]]]


class BufferedTokenStream:
    def __init__(
        self,
        *,
        tokenize_fnc: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
        retain_format: bool = False,
    ) -> None:
        self._event_ch = aio.Chan[TokenData]()
        self._tokenize_fnc = tokenize_fnc
        self._min_ctx_len = min_ctx_len
        self._min_token_len = min_token_len
        self._retain_format = retain_format
        self._current_segment_id = shortuuid()

        self._buf_tokens: list[str] = []  # <= min_token_len
        self._in_buf = ""
        self._out_buf = ""

    @typing.no_type_check
    def push_text(self, text: str) -> None:
        self._check_not_closed()
        self._in_buf += text

        if len(self._in_buf) < self._min_ctx_len:
            return

        while True:
            tokens = self._tokenize_fnc(self._in_buf)
            if len(tokens) <= 1:
                break

            if self._out_buf:
                self._out_buf += " "

            tok = tokens.pop(0)
            tok_text = tok
            if isinstance(tok, tuple):
                tok_text = tok[0]

            self._out_buf += tok_text
            if len(self._out_buf) >= self._min_token_len:
                self._event_ch.send_nowait(
                    TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                )

                self._out_buf = ""

            if isinstance(tok, tuple):
                self._in_buf = self._in_buf[tok[2] :]
            else:
                tok_i = max(self._in_buf.find(tok), 0)
                self._in_buf = self._in_buf[tok_i + len(tok) :].lstrip()

    @typing.no_type_check
    def flush(self) -> None:
        self._check_not_closed()

        if self._in_buf or self._out_buf:
            tokens = self._tokenize_fnc(self._in_buf)
            if tokens:
                if self._out_buf:
                    self._out_buf += " "

                if isinstance(tokens[0], tuple):
                    self._out_buf += " ".join([tok[0] for tok in tokens])
                else:
                    self._out_buf += " ".join(tokens)

            if self._out_buf:
                self._event_ch.send_nowait(
                    TokenData(token=self._out_buf, segment_id=self._current_segment_id)
                )

        self._current_segment_id = shortuuid()
        self._in_buf = ""
        self._out_buf = ""

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
        tokenizer: TokenizeCallable,
        min_token_len: int,
        min_ctx_len: int,
    ) -> None:
        super().__init__(
            tokenize_fnc=tokenizer,
            min_token_len=min_token_len,
            min_ctx_len=min_ctx_len,
        )
