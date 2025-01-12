from dataclasses import dataclass
from abc import ABC, abstractmethod

from livekit import rtc

from .. import llm

from typing import AsyncIterable, Union, Literal, Generic, TypeVar


@dataclass
class InputSpeechStartedEvent:
    pass


@dataclass
class InputSpeechStoppedEvent:
    pass


@dataclass
class GenerationCreatedEvent:
    message_id: str
    text_stream: AsyncIterable[str]
    audio_stream: AsyncIterable[rtc.AudioFrame]
    tool_calls: AsyncIterable[llm.FunctionCallInfo]


@dataclass
class ErrorEvent:
    type: str
    message: str


@dataclass
class RealtimeCapabilities:
    message_truncation: bool


class RealtimeModel:
    def __init__(self, capabilities: RealtimeCapabilities) -> None:
        self._capabilities = capabilities

    @property
    def capabilities(self) -> RealtimeCapabilities:
        return self._capabilities

    @abstractmethod
    def session(self) -> "RealtimeSession": ...


EventTypes = Literal[
    "input_speech_started",  # serverside VAD
    "input_speech_stopped",  # serverside VAD
    "generation_created",
    "error",
]

TEvent = TypeVar("TEvent")


class RealtimeSession(
    ABC,
    rtc.EventEmitter[Union[EventTypes, TEvent]],
    Generic[TEvent],
):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__()
        self._realtime_model = realtime_model

    @property
    def realtime_model(self) -> RealtimeModel:
        return self._realtime_model

    @property
    @abstractmethod
    def chat_ctx(self) -> llm.ChatContext: ...

    @abstractmethod
    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None: ...

    @abstractmethod
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    @abstractmethod
    def generate_reply(self) -> None: ...  # when VAD is disabled

    # message_id is the ID of the message to truncate (inside the ChatCtx)
    @abstractmethod
    def truncate(self, *, message_id: str, audio_end_ms: int) -> None: ...
