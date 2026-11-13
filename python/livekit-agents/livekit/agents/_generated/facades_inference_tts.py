# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import _bindings, _core
from ..tokenize import SentenceTokenizer
from ..types import NOT_GIVEN
from ..utils import is_given
from .tts import TTS as BaseTTS

if TYPE_CHECKING:
    from ..types import NotGivenOr
    from .connect import APIConnectOptions
    from .inference_tts import FallbackModels, TTSEncoding, TTSModels
    from .stream_pacer import SentenceStreamPacer


class TTS(_core.InferenceTts, BaseTTS):
    """`inference.TTS`: the LiveKit Inference gateway's TTS."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _bindings.refuse_overrides(cls, ("_synthesize", "_stream"), "session.tts_node")

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
        _core.InferenceTts.__init__(
            self,
            SentenceTokenizer(),
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
        BaseTTS.__init__(
            self,
            capabilities=self.capabilities,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
        )

    async def aclose(self) -> None:
        await BaseTTS.aclose(self)
        await _core.InferenceTts.aclose(self)

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, object]] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update the options the next synthesis uses: the gateway's, and how the framework prepares
        the text (``tts.TTS.update_options``).
        """
        _core.InferenceTts._core_update_options(
            self,
            voice=voice if is_given(voice) else None,
            model=model if is_given(model) else None,
            language=language if is_given(language) else None,
            extra_kwargs=extra_kwargs if is_given(extra_kwargs) else None,
        )
        BaseTTS.update_options(self, sentence_tokenizer=sentence_tokenizer, pacing=pacing)
