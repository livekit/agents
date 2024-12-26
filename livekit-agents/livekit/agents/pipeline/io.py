from __future__ import annotations

from typing import (
    AsyncIterable,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    Union,
)

from livekit import rtc

from .. import llm, stt

STTNode = Callable[
    [AsyncIterable[rtc.AudioFrame]],
    Union[Awaitable[Optional[AsyncIterable[stt.SpeechEvent]]],],
]
LLMNode = Callable[
    [llm.ChatContext, Optional[llm.FunctionContext]],
    Union[
        Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]],
        Awaitable[
            Optional[Union[AsyncIterable[llm.ChatChunk], AsyncIterable[str], str]],
        ],
    ],
]
TTSNode = Callable[[AsyncIterable[str]], Optional[AsyncIterable[rtc.AudioFrame]]]


AudioStream = AsyncIterable[rtc.AudioFrame]
VideoStream = AsyncIterable[rtc.VideoFrame]


class AudioSink(Protocol):
    async def capture_frame(self, audio: rtc.AudioFrame) -> None: ...

    def flush(self) -> None: ...


class TextSink(Protocol):
    async def capture_text(self, text: str) -> None: ...

    def flush(self) -> None: ...


class VideoSink(Protocol):
    async def capture_frame(self, text: rtc.VideoFrame) -> None: ...

    def flush(self) -> None: ...
