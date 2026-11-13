# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from .realtime import RealtimeModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from livekit.agents.llm import AudioGate
    from livekit.agents.llm.duplex import AudioGate as HostAudioGate, DuplexModel


class DuplexRealtimeAdapter(_core.DuplexRealtimeModel, RealtimeModel):
    """`llm.DuplexRealtimeAdapter`: runs a `DuplexModel` inside a `VoiceSession`. The core gates the
    model's continuous output, segments it into generations and attaches its transcript; output
    the model never transcribes still plays, it just has no chat item.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(cls, ("session",), "session.llm_node")

    def __init__(
        self,
        duplex_model: DuplexModel,
        *,
        gate: Callable[[], HostAudioGate] | None = None,
        audio_timeout: float = 0.8,
    ) -> None:
        """Run a duplex model as an ordinary realtime session.

        Args:
            duplex_model: The model to wrap.
            gate: Decides which output frames carry speech. Defaults to the model's own gate,
                or an adaptive one that learns the model's silence.
            audio_timeout: How long to wait for the next frame before the output counts as
                finished, for a provider that stops sending rather than streaming its silence.
        """
        self._duplex_model = duplex_model
        _core.DuplexRealtimeModel.__init__(
            self,
            duplex_model,
            gate=gate,
            audio_timeout=audio_timeout,
        )
        RealtimeModel.__init__(self, capabilities=self.capabilities)

    @property
    def model(self) -> str:
        return self._duplex_model.model

    @property
    def provider(self) -> str:
        return self._duplex_model.provider

    async def aclose(self) -> None:
        await self._duplex_model.aclose()
