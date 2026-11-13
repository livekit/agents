# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .. import _bindings, _core
from ..log import logger
from . import tts_availability
from .tts import TTS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .chat_items import MetricsMetadata
    from .llm_fallback import AvailabilityChanged


class FallbackAdapter(_core.FallbackTts, TTS[Literal["tts_availability_changed"]]):
    """`tts.FallbackAdapter`: several TTS behind one; a failed one is skipped until it recovers."""

    if TYPE_CHECKING:
        _members: list[TTS[Any]]

        def _core_failed(self, error: BaseException, recoverable: bool) -> None: ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(cls, ("_synthesize", "_stream"), "session.tts_node")

    def __init__(
        self,
        tts: Sequence[TTS[Any]],
        *,
        max_retry_per_tts: int = 2,
        attempt_timeout: float | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """Several TTS behind one: a failed one is skipped until it recovers.

        Args:
            tts: The TTS, in order of preference.
            max_retry_per_tts: Retries of each before moving to the next.
            attempt_timeout: Caps each limit of one attempt; ``None`` keeps the caller's
                ``conn_options`` limits.
            sample_rate: The output rate; the highest of the TTS by default, the others
                resampled.
        """
        _core.FallbackTts.__init__(
            self,
            tts,
            max_retry_per_tts=max_retry_per_tts,
            attempt_timeout=attempt_timeout,
            sample_rate=sample_rate,
        )
        self._members = list(tts)
        TTS.__init__(
            self,
            capabilities=self.capabilities,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
        )

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "tts_availability_changed",
            tts_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all TTSs are unavailable, retrying..")

    @property
    def _active(self) -> TTS[Any]:
        return self._members[self._core_active_index]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def provider(self) -> str:
        return self._active.provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._active.metrics_metadata

    async def aclose(self) -> None:
        await TTS.aclose(self)
        await _core.FallbackTts.aclose(self)


class StreamAdapter(_core.TtsStreamAdapter, TTS):
    """`tts.StreamAdapter`: a non-streaming TTS as a streaming one."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(cls, ("_synthesize", "_stream"), "session.tts_node")

    def __init__(self, *, tts: TTS[Any]) -> None:
        """A non-streaming TTS as a streaming one: each chunk of the text, as its stage splits it,
        is synthesized on its own.

        Args:
            tts: The non-streaming TTS.
        """
        _core.TtsStreamAdapter.__init__(self, tts=tts)
        self._tts = tts
        TTS.__init__(
            self,
            capabilities=self.capabilities,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
        )
