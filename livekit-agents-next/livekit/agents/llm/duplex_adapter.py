from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from livekit import rtc

from .. import _core
from .duplex import DuplexCapabilities, DuplexModel
from .realtime import RealtimeCapabilities, RealtimeModel, RealtimeSession

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
        caps: DuplexCapabilities = duplex_model.capabilities
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=caps.user_transcription,
                auto_tool_reply_generation=caps.auto_tool_reply_generation,
                audio_output=True,
                manual_function_calls=False,
                supports_overlapping_speech=True,
                mutable_chat_context=caps.mutable_chat_context,
                mutable_instructions=caps.mutable_instructions,
                mutable_tools=caps.mutable_tools,
                per_response_tool_choice=False,
                supports_say=False,
            )
        )
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
