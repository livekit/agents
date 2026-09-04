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
    """One frame of the model's output audio."""

    frame: rtc.AudioFrame
    start_ms: int | None = None
    """Position on the model's timeline, for a provider that stamps its audio."""


@dataclass
class DuplexTranscriptDelta:
    """A fragment of the model's transcript of its own speech."""

    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    """Span on the model's timeline; only its continuity from one fragment to the next is used."""


@dataclass
class DuplexCapabilities:
    """What varies between duplex providers; barge-in and the lack of truncation do not."""

    user_transcription: bool
    """Whether the model transcribes the user's speech"""
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
    "function_call",
    "input_speech_started",  # the user's turns, as the plugin detects them
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
    def session(self, *, wait_for_config: bool = False) -> DuplexSession:
        """Open a session; ``wait_for_config`` promises it a ``_update_session`` call before use."""

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
    def __init__(self, duplex_model: DuplexModel, *, wait_for_config: bool = False) -> None:
        super().__init__()
        self._duplex_model = duplex_model
        self._config_delivered = asyncio.Event()
        if not wait_for_config:
            self._config_delivered.set()

    async def _await_config(self) -> None:
        """Wait for the promised configuration; set ``_config_delivered`` from ``aclose`` too."""
        await self._config_delivered.wait()

    @property
    def duplex_model(self) -> DuplexModel:
        return self._duplex_model

    @property
    def capabilities(self) -> DuplexCapabilities:
        return self._duplex_model.capabilities

    @property
    @abstractmethod
    def audio_stream(self) -> AsyncIterable[DuplexAudioFrame]:
        """The model's output audio for the life of the session, silence included."""

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

    # underscored until the shape settles: this is the framework's contract with the plugin, and
    # apps reach a plugin's own methods through Agent.duplex_session

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
    ) -> None:
        """Ask the model to speak now; its next speech is taken as the reply."""
        raise RealtimeError(f"{type(self).__name__} decides for itself when to speak")

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        """Apply the whole configuration at once, right after the session is created."""
        try:
            if is_given(instructions):
                await self._update_instructions(instructions)

            if is_given(chat_ctx):
                await self._update_chat_ctx(chat_ctx)

            if is_given(tools):
                await self._update_tools(tools)
        finally:
            self._config_delivered.set()

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
