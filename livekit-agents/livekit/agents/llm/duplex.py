from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Literal, TypeVar

from livekit import rtc

from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .chat_context import ChatContext
from .realtime import RealtimeError
from .tool_context import Tool, ToolChoice, ToolContext


@dataclass
class DuplexAudioFrame:
    """One frame of a duplex model's output, tagged with the turn it belongs to."""

    frame: rtc.AudioFrame
    turn_id: str | None = None
    """The model's turn id, or None when the plugin does not know it yet.

    Never a per-fragment id: a change of id means a change of turn, so report None rather than
    invent one.
    """
    start_ms: int | None = None
    """Position on the model's own timeline."""


@dataclass
class DuplexTranscriptDelta:
    """A fragment of the model's transcript of its own speech."""

    text: str
    turn_id: str | None = None
    """The model's turn id, or None while the model has not announced one yet."""
    start_ms: int | None = None
    end_ms: int | None = None
    """End of the span covered, on the model's timeline.

    A turn closes once this reaches the audio it forwarded.
    """


@dataclass
class DuplexTurnStartedEvent:
    turn_id: str


@dataclass
class DuplexTurnEndedEvent:
    """Emitted after the turn's last transcript, so it also means fully transcribed."""

    turn_id: str


@dataclass
class DuplexCapabilities:
    """What varies between duplex providers.

    Barge-in and the absence of message truncation are properties of the model type rather than
    flags: no duplex model can do them.
    """

    user_transcription: bool
    """Whether the model emits user audio transcription events"""
    auto_tool_reply_generation: bool
    """Whether the model automatically continues speaking after receiving tool results"""
    manual_response_creation: bool = False
    """Whether the client can ask the model to speak, rather than only waiting for it to"""
    mutable_chat_context: bool = False
    """Whether the chat context can be updated mid-session"""
    mutable_instructions: bool = False
    """Whether the instructions can be updated mid-session"""
    mutable_tools: bool = False
    """Whether the tools can be updated mid-session"""


DuplexEventTypes = Literal[
    "transcript_delta",  # the model's transcript of its own speech
    "turn_started",  # assistant turn boundary, as the model reports it
    "turn_ended",
    "function_call",
    "input_speech_started",  # serverside turn detection over the user's audio
    "input_speech_stopped",
    "input_audio_transcription_completed",
    "session_reconnected",
    "metrics_collected",
    "error",
]

TEvent = TypeVar("TEvent")


class DuplexModel(ABC):
    """A speech model that listens and speaks at the same time.

    Its audio streams continuously whether or not it is speaking, and barge-in is its own. Run one
    inside an ``AgentSession`` with :class:`~livekit.agents.llm.DuplexRealtimeAdapter`.
    """

    def __init__(self, *, capabilities: DuplexCapabilities) -> None:
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    @property
    def capabilities(self) -> DuplexCapabilities:
        return self._capabilities

    @property
    def label(self) -> str:
        return self._label

    @abstractmethod
    def session(self) -> DuplexSession: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    async def __aenter__(self) -> DuplexModel:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class DuplexSession(ABC, rtc.EventEmitter[DuplexEventTypes | TEvent], Generic[TEvent]):
    def __init__(self, duplex_model: DuplexModel) -> None:
        super().__init__()
        self._duplex_model = duplex_model

    @property
    def duplex_model(self) -> DuplexModel:
        return self._duplex_model

    @property
    def capabilities(self) -> DuplexCapabilities:
        return self._duplex_model.capabilities

    @property
    @abstractmethod
    def audio_stream(self) -> AsyncIterable[DuplexAudioFrame]:
        """The model's output audio, streamed for the life of the session.

        Carries silence and untranscribed sound as well as speech; the consumer decides what plays.
        """

    @property
    @abstractmethod
    def chat_ctx(self) -> ChatContext: ...

    @property
    @abstractmethod
    def tools(self) -> ToolContext: ...

    @abstractmethod
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Feed a video frame to the model; ignored by models without video input."""
        pass

    @abstractmethod
    async def aclose(self) -> None: ...

    # underscored until the shape settles, so they stay free to change: this is the framework's
    # contract with the plugin, and apps use a plugin's own methods via Agent.duplex_session

    @abstractmethod
    async def _update_instructions(self, instructions: str) -> None: ...

    @abstractmethod
    async def _update_chat_ctx(self, chat_ctx: ChatContext) -> None: ...

    @abstractmethod
    async def _update_tools(self, tools: list[Tool]) -> None: ...

    @abstractmethod
    def _update_options(
        self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN
    ) -> None: ...

    def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[str | None]:
        """Ask the model to speak now, where the protocol allows it.

        Resolves once the ask lands, with the id of the turn that will answer it, or None where
        the protocol names none and the model's next turn should be taken as the reply. Fail it
        where the ask could not be delivered; raise :class:`RealtimeError` where the model cannot
        be asked at all.
        """
        raise RealtimeError(f"{type(self).__name__} decides for itself when to speak")

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        """Apply the agent's whole configuration at once, right after the session is created.

        A model whose configuration is immutable once started should compose it here.
        """
        if is_given(instructions):
            await self._update_instructions(instructions)

        if is_given(chat_ctx):
            await self._update_chat_ctx(chat_ctx)

        if is_given(tools):
            await self._update_tools(tools)

    def _report_connection_acquired(self, acquire_time: float) -> None:
        """Report connection timing as a RealtimeModelMetrics event with zero usage."""
        from ..metrics.base import Metadata, RealtimeModelMetrics

        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id="",
                timestamp=time.time(),
                acquire_time=acquire_time,
                connection_reused=False,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=Metadata(
                    model_name=self._duplex_model.model,
                    model_provider=self._duplex_model.provider,
                ),
            ),
        )
