from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import STTModels
from .version import __version__

DEFAULT_BASE_URL = "https://api.fish.audio"
DEFAULT_STT_MODEL: STTModels = "transcribe-1"
USER_AGENT = f"livekit-plugins-fishaudio/{__version__}"


@dataclass
class _STTOptions:
    api_key: str
    base_url: str
    language: LanguageCode | None
    ignore_timestamps: bool
    model: STTModels | str


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        ignore_timestamps: bool = True,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Fish Audio STT.

        Args:
            api_key (NotGivenOr[str]): Fish Audio API key. Reads ``FISH_API_KEY`` if unset.
            base_url (NotGivenOr[str]): Custom base URL. Defaults to
                ``https://api.fish.audio``.
            language (NotGivenOr[str | None]): Optional spoken language hint. ``None``,
                ``"auto"``, ``"multi"``, and ``""`` omit the language field and let Fish
                Audio auto-detect.
            ignore_timestamps (bool): Request plain transcription without timestamp details.
                Set to ``False`` to preserve provider timestamp segments in metadata when
                Fish Audio returns them. Defaults to ``True``.
            http_session (aiohttp.ClientSession | None): Optional aiohttp session.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                aligned_transcript=False,
                offline_recognize=True,
            )
        )

        fish_api_key = api_key if is_given(api_key) else os.getenv("FISH_API_KEY")
        if not fish_api_key:
            raise ValueError(
                "Fish Audio API key is required, either as argument or set "
                "FISH_API_KEY environment variable"
            )

        self._opts = _STTOptions(
            api_key=fish_api_key,
            base_url=(base_url if is_given(base_url) else DEFAULT_BASE_URL).rstrip("/"),
            language=_normalize_language(language),
            ignore_timestamps=ignore_timestamps,
            model=DEFAULT_STT_MODEL,
        )
        self._session = http_session

    @property
    def model(self) -> STTModels | str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "FishAudio"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        ignore_timestamps: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = _normalize_language(language)
        if is_given(ignore_timestamps):
            self._opts.ignore_timestamps = ignore_timestamps

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        request_language = (
            _normalize_language(language) if is_given(language) else self._opts.language
        )
        form = self._build_form(
            audio=utils.merge_frames(buffer).to_wav_bytes(),
            language=request_language,
        )

        try:
            async with self._ensure_session().post(
                url=f"{self._opts.base_url}/v1/asr",
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "User-Agent": USER_AGENT,
                },
                data=form,
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                request_id = response.headers.get("x-request-id", "")
                payload = await _read_response(response)

                if response.status >= 400:
                    _log_stt_error(
                        "Fish Audio STT request failed",
                        request_id=request_id,
                        provider=self.provider,
                        model=self.model,
                        language=request_language,
                        status_code=response.status,
                        body=payload,
                    )
                    raise APIStatusError(
                        "Fish Audio ASR request failed",
                        status_code=response.status,
                        request_id=request_id,
                        body=payload,
                    )

        except asyncio.TimeoutError as exc:
            _log_stt_error(
                "Fish Audio STT request timed out",
                request_id="",
                provider=self.provider,
                model=self.model,
                language=request_language,
                error=exc,
            )
            raise APITimeoutError() from exc
        except aiohttp.ClientError as exc:
            _log_stt_error(
                "Fish Audio STT connection error",
                request_id="",
                provider=self.provider,
                model=self.model,
                language=request_language,
                error=exc,
            )
            raise APIConnectionError(str(exc)) from exc
        except Exception as exc:
            raise APIConnectionError() from exc

        event = _speech_event_from_response(
            payload,
            language=request_language,
            request_id=request_id,
        )
        transcript = event.alternatives[0].text if event.alternatives else ""
        logger.info(
            "Fish Audio STT transcript: %s",
            transcript,
            extra={
                "request_id": request_id,
                "stt_provider": self.provider,
                "stt_model": self.model,
                "stt_language": str(request_language) if request_language else None,
                "stt_duration": event.alternatives[0].end_time if event.alternatives else 0.0,
            },
        )
        return event

    def _build_form(
        self,
        *,
        audio: bytes,
        language: LanguageCode | None,
    ) -> aiohttp.FormData:
        form = aiohttp.FormData()
        form.add_field(
            "audio",
            audio,
            filename="audio.wav",
            content_type="audio/wav",
        )
        if language is not None:
            form.add_field("language", language.language)
        form.add_field(
            "ignore_timestamps",
            "true" if self._opts.ignore_timestamps else "false",
        )
        return form


def _normalize_language(language: NotGivenOr[str | None]) -> LanguageCode | None:
    if not is_given(language) or language is None:
        return None
    if language in ("", "auto", "multi"):
        return None
    return LanguageCode(language)


async def _read_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
    try:
        payload = await response.json()
    except aiohttp.ContentTypeError:
        return {"error": await response.text()}

    if isinstance(payload, dict):
        return payload
    return {"response": payload}


def _log_stt_error(
    message: str,
    *,
    request_id: str,
    provider: str,
    model: str,
    language: LanguageCode | None,
    status_code: int | None = None,
    body: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    extra: dict[str, Any] = {
        "request_id": request_id,
        "stt_provider": provider,
        "stt_model": model,
        "stt_language": str(language) if language else None,
    }
    if status_code is not None:
        extra["stt_status_code"] = status_code
    if body is not None:
        extra["stt_error_body"] = body
    if error is not None:
        extra["stt_error"] = repr(error)

    logger.error(message, extra=extra)


def _speech_event_from_response(
    payload: dict[str, Any],
    *,
    language: LanguageCode | None,
    request_id: str,
) -> stt.SpeechEvent:
    text = str(payload.get("text") or "")
    duration = _coerce_float(payload.get("duration"))
    segments = payload.get("segments") or []

    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=request_id,
        alternatives=[
            stt.SpeechData(
                language=language or LanguageCode(str(payload.get("language") or "")),
                text=text,
                start_time=0.0,
                end_time=duration,
                metadata={"segments": segments} if segments else None,
            )
        ],
    )


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
