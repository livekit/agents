from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from types import TracebackType
from typing import AsyncIterator, Union

from ..utils import aio


class TextTranscriber(ABC):
    def __init__(self) -> None:
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @abstractmethod
    def stream(self) -> TranscriptionStream:
        """Create a new streaming transcription session"""
        pass

    async def aclose(self) -> None:
        pass

    async def __aenter__(self) -> TextTranscriber:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class TranscriptionStream(ABC):
    """Stream interface for transcription processing"""

    class _FlushSentinel: ...

    def __init__(self, *, transcriber: TextTranscriber) -> None:
        super().__init__()
        self._transcriber = transcriber
        self._input_ch = aio.Chan[Union[str, TranscriptionStream._FlushSentinel]]()
        self._event_ch = aio.Chan[str]()

        self._task = asyncio.create_task(self._run(), name="Transcription._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _run(self) -> None:
        """Main task for processing transcription"""
        pass

    def push_text(self, token: str) -> None:
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(token)

    def flush(self) -> None:
        self._check_input_not_ended()
        self._check_not_closed()
        self._input_ch.send_nowait(self._FlushSentinel())

    def end_input(self) -> None:
        self.flush()
        self._input_ch.close()

    async def aclose(self) -> None:
        self._input_ch.close()
        await aio.cancel_and_wait(self._task)

    def _check_not_closed(self) -> None:
        if self._event_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} is closed")

    def _check_input_not_ended(self) -> None:
        if self._input_ch.closed:
            cls = type(self)
            raise RuntimeError(f"{cls.__module__}.{cls.__name__} input ended")

    async def __anext__(self) -> str:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __aenter__(self) -> TranscriptionStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class SimpleTextTranscriber(TextTranscriber):
    """A simple implementation of TextTranscriber that passes text through unchanged"""

    class SimpleStream(TranscriptionStream):
        async def _run(self) -> None:
            async for text in self._input_ch:
                if isinstance(text, str):
                    self._event_ch.send_nowait(text)

    def stream(self) -> TranscriptionStream:
        return SimpleTextTranscriber.SimpleStream(transcriber=self)
