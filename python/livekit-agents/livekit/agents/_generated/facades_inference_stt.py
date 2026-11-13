# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .. import _bindings, _core
from .._bindings import silero_model
from ..types import NOT_GIVEN
from ..utils import is_given
from .stt import RecognizeStream, STT as BaseSTT

if TYPE_CHECKING:
    from livekit.agents.vad import VAD

    from ..inference import stt
    from ..types import NotGivenOr
    from .connect import APIConnectOptions
    from .inference_stt import FallbackModels, STTEncoding, STTLanguages, STTModels


class STT(_core.InferenceStt, BaseSTT):
    """`inference.STT`: the LiveKit Inference gateway's STT."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(
            cls,
            ("stream", "recognize", "_recognize_impl"),
            "session.stt_node",
        )

    def __init__(
        self,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, object]] = NOT_GIVEN,
        fallback: NotGivenOr[FallbackModels] = NOT_GIVEN,
        conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN,
        vad: NotGivenOr[VAD[Any]] = NOT_GIVEN,
    ) -> None:
        """LiveKit Inference STT.

        Args:
            model: ``"provider/model[:language]"``.
            language: The language to transcribe; overrides the model string's.
            base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
            encoding: The audio encoding sent to the gateway.
            sample_rate: The sample rate sent to the gateway. Defaults to 24000 for OpenAI,
                16000 for other providers.
            api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
            extra_kwargs: Provider options passed through to the gateway.
            fallback: Fallback models, names or ``FallbackModel`` entries, run by the gateway.
            conn_options: The connection timeout, and the retries the gateway runs server-side.
            vad: A VAD whose end of speech finalizes the transcript (Speechmatics and OpenAI).
        """
        _core.InferenceStt.__init__(
            self,
            silero_model,
            model if is_given(model) else None,
            language=language if is_given(language) else None,
            base_url=base_url if is_given(base_url) else None,
            encoding=encoding if is_given(encoding) else None,
            sample_rate=sample_rate if is_given(sample_rate) else None,
            api_key=api_key if is_given(api_key) else None,
            api_secret=api_secret if is_given(api_secret) else None,
            extra_kwargs=extra_kwargs if is_given(extra_kwargs) else None,
            fallback=fallback if is_given(fallback) else None,
            conn_options=conn_options if is_given(conn_options) else None,
            vad=vad if is_given(vad) else None,
        )
        BaseSTT.__init__(self, capabilities=self.capabilities)

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        extra: NotGivenOr[dict[str, object]] = NOT_GIVEN,
    ) -> None:
        """Update the options; open streams get them as a mid-stream ``session.update``."""
        _core.InferenceStt._core_update_options(
            self,
            model=model if is_given(model) else None,
            language=language if is_given(language) else None,
            extra=extra if is_given(extra) else None,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN,
    ) -> RecognizeStream:
        return SpeechStream(
            stt=cast("stt.STT", self),
            language=language if is_given(language) else None,
            conn_options=conn_options if is_given(conn_options) else self._core_conn_options(),
        )


class SpeechStream(RecognizeStream):
    """`inference.STT.stream()`: one gateway recognition session; the client reconnects it under its
    ``conn_options``.
    """

    _core_native_retries = True
    _stt: stt.STT

    def __init__(
        self,
        *,
        stt: stt.STT,
        language: STTLanguages | str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._language: STTLanguages | str | None = language

    async def _core_open(self) -> _core.InferenceSttStream:
        return self._stt._core_native_stream(
            language=self._language,
            start_time_offset=self._start_time_offset,
            conn_options=self._conn_options,
        )

    _run = RecognizeStream._core_run_native


SpeechStream.__module__ = "livekit.agents.inference.stt"
