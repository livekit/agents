"""FonadaLabs STT plugin for LiveKit — Kapture V3 REST implementation.

The framework buffers each utterance with VAD and calls ``_recognize_impl``
with the complete segment, which is sent as a WAV file to:

    POST https://api.fonada.ai/v3/asr/transcribe
    Authorization: Bearer <api_key>
    multipart: file=<wav>, language_id=<code>

Docs: https://fonadalabs.ai/docs/speech-to-text/kapture-v3
"""

from __future__ import annotations

import asyncio
import os

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from ._utils import parse_error
from .log import logger
from .models import STT_LANGUAGES, STTLanguages, STTModels, resolve_language

FONADALABS_API_BASE_URL = "https://api.fonada.ai"


class STT(stt.STT):
    """FonadaLabs Speech-to-Text plugin for LiveKit (Kapture V3 by default).

    Uses VAD-buffered (non-streaming) recognition: each complete utterance is
    transcribed with a single REST request.

    Args:
        api_key:      FonadaLabs API key (or ``FONADALABS_API_KEY`` env var).
        model:        ``"v3"`` (Kapture V3, the default and only documented model).
        language:     Language code or display name — ``"hi"``, ``"en"``,
                      ``"Tamil"``, ... See :data:`~.models.KAPTURE_LANGUAGES`.
        base_url:     Override the API base URL.
        http_session: Reuse an existing :class:`aiohttp.ClientSession`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: STTModels | str = "v3",
        language: STTLanguages | str = "hi",
        base_url: str = FONADALABS_API_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )

        self._api_key = api_key or os.environ.get("FONADALABS_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FonadaLabs API key is required. "
                "Pass it directly or set the FONADALABS_API_KEY environment variable."
            )

        if model not in STT_LANGUAGES:
            raise ValueError(
                f"Unsupported STT model {model!r}. Supported: {', '.join(STT_LANGUAGES)}"
            )
        self._model = model
        self._language = self._resolve_language(language)
        self._url = f"{base_url.rstrip('/')}/{model}/asr/transcribe"
        self._session = http_session

    @property
    def model(self) -> str:
        return f"kapture_{self._model}"

    @property
    def provider(self) -> str:
        return "FonadaLabs"

    def _resolve_language(self, language: str) -> str:
        supported = STT_LANGUAGES[self._model]
        code = resolve_language(language, supported)
        if code is None:
            raise ValueError(
                f"Language {language!r} is not supported by Kapture {self._model.upper()}. "
                f"Supported: {', '.join(f'{c} ({n})' for c, n in supported.items())}"
            )
        return code

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = self._resolve_language(language) if utils.is_given(language) else self._language

        combined = rtc.combine_audio_frames(buffer)
        wav_bytes = combined.to_wav_bytes()
        duration = combined.samples_per_channel / combined.sample_rate

        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        form.add_field("language_id", lang)

        logger.debug(
            f"STT sending {len(wav_bytes)} WAV bytes ({duration:.2f}s, {combined.sample_rate}Hz)"
        )

        try:
            async with self._ensure_session().post(
                self._url,
                data=form,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    code, msg = parse_error(body)
                    raise APIStatusError(
                        message=f"FonadaLabs STT error [{code}]: {msg}",
                        status_code=resp.status,
                        body=body,
                    )
                data = await resp.json()
        except asyncio.TimeoutError as e:
            raise APITimeoutError("FonadaLabs STT request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"FonadaLabs STT connection failed: {e}") from e

        text = (data.get("text") or "").strip()
        processing_ms = data.get("processing_time_ms")
        # Never log the transcript itself: it is user speech.
        logger.debug(
            f"STT done: {len(text)} chars "
            f"(lang={data.get('language')}, duration={data.get('audio_duration_s')}s"
            f"{f', processing={processing_ms:.1f}ms' if processing_ms else ''})"
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=data.get("job_id") or "",
            alternatives=[
                stt.SpeechData(language=LanguageCode(data.get("language") or lang), text=text)
            ],
        )
