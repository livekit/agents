# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from ..types import NOT_GIVEN
from ..utils import is_given
from .tts import TTS as BaseTTS

if TYPE_CHECKING:
    from ..types import NotGivenOr
    from .connect import APIConnectOptions
    from .inference_tts import FallbackModels, TTSEncoding, TTSModels
    from .tts import TTSCapabilities


class TTS(BaseTTS):
    """`inference.TTS`: the LiveKit Inference gateway's TTS."""

    if TYPE_CHECKING:
        def _core_native(self) -> _core.InferenceTts: ...

    def __init__(
        self,
        model: TTSModels | str,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[TTSEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, object]] = NOT_GIVEN,
        fallback: NotGivenOr[FallbackModels] = NOT_GIVEN,
        conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN,
    ) -> None:
        """LiveKit Inference TTS.

        Args:
            model: ``"provider/model[:voice]"``.
            voice: The voice; overrides the model string's.
            language: The language to synthesize.
            encoding: The audio encoding asked of the gateway.
            sample_rate: The sample rate asked of the gateway.
            base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
            api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
            extra_kwargs: Provider options passed through to the gateway.
            fallback: Fallback models, names or ``FallbackModel`` entries, run by the gateway.
            conn_options: The connection timeout, and the retries the gateway runs server-side.
        """
        self._native = _core.InferenceTts(
            model,
            voice=voice if is_given(voice) else None,
            language=language if is_given(language) else None,
            encoding=encoding if is_given(encoding) else None,
            sample_rate=sample_rate if is_given(sample_rate) else None,
            base_url=base_url if is_given(base_url) else None,
            api_key=api_key if is_given(api_key) else None,
            api_secret=api_secret if is_given(api_secret) else None,
            extra_kwargs=extra_kwargs if is_given(extra_kwargs) else None,
            fallback=fallback if is_given(fallback) else None,
            conn_options=conn_options if is_given(conn_options) else None,
        )
        super().__init__(
            capabilities=self._native.capabilities,
            sample_rate=self._native.sample_rate,
            num_channels=self._native.num_channels,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, TTS, "_synthesize", "_stream")
        return self if replaced else self._core_native()

    async def aclose(self) -> None:
        await super().aclose()
        await self._native.aclose()

    @property
    def model(self) -> str:
        return self._native.model

    @property
    def capabilities(self) -> TTSCapabilities:
        return self._native.capabilities

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, object]] = NOT_GIVEN,
    ) -> None:
        """Update the options the next synthesis uses."""
        self._native.update_options(
            voice=voice if is_given(voice) else None,
            model=model if is_given(model) else None,
            language=language if is_given(language) else None,
            extra_kwargs=extra_kwargs if is_given(extra_kwargs) else None,
        )
