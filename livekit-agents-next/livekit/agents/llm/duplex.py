from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from types import TracebackType
from typing import TYPE_CHECKING, Generic, Literal, Never

from typing_extensions import TypeVar

from livekit import rtc
from livekit.agents._generated.duplex import (
    DuplexAudioFrame as DuplexAudioFrame,
    DuplexCapabilities as DuplexCapabilities,
    DuplexOutputTranscriptDelta as DuplexOutputTranscriptDelta,
)

from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .chat_context import ChatContext, ChatItem
from .realtime import RealtimeError
from .tool_context import Tool, ToolChoice, ToolContext

if TYPE_CHECKING:
    from .duplex_adapter import AudioGate


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

TEvent = TypeVar("TEvent", default=Never, contravariant=True)


class DuplexModel(ABC):
    """A speech model that listens and speaks at the same time, handling interruptions itself.

    Run one inside an ``VoiceSession`` with :class:`~livekit.agents.llm.DuplexRealtimeAdapter`.
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

    def audio_gate(self) -> AudioGate | None:
        """The gate for this model's output, or None to let the adapter infer one."""
        return None

    @abstractmethod
    def session(self) -> DuplexSession:
        """Open a session; the adapter configures it with ``_update_session`` before use."""

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
        self._closing = False
        # set once the whole configuration is applied; a model whose configuration is immutable
        # once started waits on it before connecting
        self._configured = asyncio.Event()

    @property
    def duplex_model(self) -> DuplexModel:
        return self._duplex_model

    @property
    def capabilities(self) -> DuplexCapabilities:
        return self._duplex_model.capabilities

    @property
    @abstractmethod
    def audio_stream(self) -> AsyncIterable[DuplexAudioFrame]:
        """The model's output audio, its own silence included; it may stop between bursts."""

    @property
    @abstractmethod
    def tools(self) -> ToolContext: ...

    @abstractmethod
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Feed a video frame to the model; ignored by models without video input."""
        pass

    async def aclose(self) -> None:
        """Close the session; an override calls ``super().aclose()`` first.

        It releases a model that waits on ``_configured``, which then reads ``_closing`` to see
        that the configuration was abandoned rather than applied.
        """
        self._closing = True
        self._configured.set()

    # underscored until the shape settles: this is the framework's contract with the plugin, and
    # apps reach a plugin's own methods through Agent.duplex_session

    @abstractmethod
    async def _update_instructions(self, instructions: str) -> None: ...

    @abstractmethod
    async def _append_items(self, items: list[ChatItem]) -> None:
        """Tell the model about new chat items; the adapter owns the context and sends only these."""

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
        if is_given(instructions):
            await self._update_instructions(instructions)

        if is_given(chat_ctx):
            await self._append_items(chat_ctx.items)

        if is_given(tools):
            await self._update_tools(tools)

        # only a complete configuration releases a model that cannot be reconfigured later
        self._configured.set()

    def _report_connection_acquired(self, acquire_time: float) -> None:
        """Report connection timing as a RealtimeModelMetrics event with zero usage."""
        from ..metrics import Metadata, RealtimeModelMetrics

        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id="",
                timestamp=time.time(),
                acquire_time=acquire_time,
                connection_reused=False,
                metadata=Metadata(
                    model_name=self._duplex_model.model,
                    model_provider=self._duplex_model.provider,
                ),
            ),
        )
