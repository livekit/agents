from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable
from types import TracebackType
from typing import TYPE_CHECKING, Generic, Literal, Never, Protocol

from typing_extensions import TypeVar

from livekit import rtc
from livekit.agents._generated.duplex import (
    DuplexAudioFrame as DuplexAudioFrame,
    DuplexCapabilities as DuplexCapabilities,
    DuplexInputGap as DuplexInputGap,
    DuplexInputMute as DuplexInputMute,
    DuplexOutputTranscriptDelta as DuplexOutputTranscriptDelta,
)

from .. import _core
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .chat_context import ChatContext, ChatItem
from .realtime import RealtimeError, RealtimeModel, RealtimeSession
from .tool_context import Tool, ToolChoice, ToolContext

if TYPE_CHECKING:
    pass


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
        # the caller's audio, its gaps and mute changes, in order: pulled with ``async for`` while
        # connected and flushed on reconnect; a plugin more than 250 ms behind skips to the newest
        # audio after a gap
        self._input = _core.DuplexInput()

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

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Queue the caller's audio; the plugin pulls it from ``_input``."""
        self._input.push(frame)

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


# a provider may stop sending rather than stream its own silence; waiting out the frame in hand
# plus this long is the silence the gate never received
_AUDIO_TIMEOUT = 0.8


class AudioGate(Protocol):
    """Decides which frames of the model's output carry speech worth playing."""

    def update(self, frame: rtc.AudioFrame) -> bool:
        """True while the frame belongs to an open burst of output."""
        ...

    def deactivate(self) -> None:
        """End the open burst, keeping whatever has been learned about the model."""
        ...


FixedGate = _core.FixedGate
"""Opens on output that stands out from a silence the plugin already knows."""

AdaptiveNoiseGate = _core.AdaptiveNoiseGate
"""Opens on output that stands out from the model's own silence: the floor is the quietest
``min_silence_duration`` stretch it produced while not speaking, within ``window``."""


class DuplexRealtimeAdapter(RealtimeModel):
    """Runs a :class:`DuplexModel` inside an ``VoiceSession``.

    The core (``_core.DuplexRealtimeModel``) gates the model's continuous output, segments it
    into generations and attaches its transcript; output the model never transcribes still plays,
    it just has no chat item.
    """

    def __init__(
        self,
        duplex_model: DuplexModel,
        *,
        gate: Callable[[], AudioGate] | None = None,
        audio_timeout: float = _AUDIO_TIMEOUT,
    ) -> None:
        """Run a duplex model as an ordinary realtime session.

        Args:
            duplex_model: The model to wrap.
            gate: Decides which output frames carry speech. Defaults to the model's own gate,
                or an adaptive one that learns the model's silence.
            audio_timeout: How long to wait for the next frame before the output counts as
                finished, for a provider that stops sending rather than streaming its silence.
        """
        caps = _core.duplex_realtime_capabilities(duplex_model.capabilities)
        # This declared field is host-only; all native capability policy comes from the engine.
        caps.supports_overlapping_speech = True
        super().__init__(capabilities=caps)
        self._duplex_model = duplex_model
        self._gate = gate
        self._audio_timeout = audio_timeout

    @property
    def duplex_model(self) -> DuplexModel:
        return self._duplex_model

    @property
    def model(self) -> str:
        return self._duplex_model.model

    @property
    def provider(self) -> str:
        return self._duplex_model.provider

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        raise NotImplementedError(
            "DuplexRealtimeAdapter sessions run inside a VoiceSession (the core segments them)"
        )

    async def aclose(self) -> None:
        await self._duplex_model.aclose()
