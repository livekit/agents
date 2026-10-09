from __future__ import annotations

import asyncio
import math
import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, replace

import httpx
import numpy as np

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given


@dataclass
class _Request:
    id: str
    conn_options: APIConnectOptions
    retry_after: float = 0


class STT(stt.STT):
    """Final-utterance transcription through the authenticated Oruk API.

    Spectra-2 uses your existing Oruk API key and shared plan minutes. Use a VAD
    StreamAdapter for a live agent; this file API does not produce partial words.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: str = "oruk-spectra-2",
        base_url: str = "https://speech-api.oruk.ai",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        key = api_key if is_given(api_key) else os.environ.get("ORUK_API_KEY")
        if not key:
            raise ValueError("Set ORUK_API_KEY or pass api_key")
        if not model.strip():
            raise ValueError("model must not be empty")
        self._key = key
        self._model = model
        self._url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
        self._client = http_client
        self._owns_client = http_client is None
        # Each concurrent utterance has its own ID; transport retries retain it.
        self._request: ContextVar[_Request] = ContextVar("oruk_stt_request")

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "Oruk"

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        request = _Request(str(uuid.uuid4()), conn_options)
        token = self._request.set(request)
        try:
            # Keep one framework call per utterance so a retryable HTTP attempt
            # cannot emit a terminal STT error before our retries finish. This
            # also preserves retryable handling with the supported 1.8.3 floor.
            return await super().recognize(
                buffer,
                language=language,
                conn_options=replace(conn_options, max_retry=0),
            )
        finally:
            self._request.reset(token)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        request = self._request.get()
        options = request.conn_options
        for attempt in range(options.max_retry + 1):
            try:
                return await self._recognize_once(buffer, language=language, conn_options=options)
            except APIError as exc:
                if not exc.retryable or attempt == options.max_retry:
                    raise
                self._emit_error(exc, recoverable=True)
                await asyncio.sleep(max(request.retry_after, options._interval_for_retry(attempt)))
                request.retry_after = 0
        raise RuntimeError("unreachable")

    async def _recognize_once(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        if is_given(language) and language:
            raise ValueError(
                "Oruk selects the language automatically; language forcing is unsupported"
            )
        wav = _wav(buffer)
        request = self._request.get()
        if self._client is None:
            self._client = httpx.AsyncClient()
        try:
            response = await self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._key}", "X-Request-ID": request.id},
                data={"model": self._model},
                files={"file": ("audio.wav", wav, "audio/wav")},
                timeout=httpx.Timeout(conn_options.timeout),
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            raise APITimeoutError("Oruk transcription timed out") from exc
        except httpx.RequestError as exc:
            raise APIConnectionError("Oruk transcription connection failed") from exc

        if not response.is_success:
            code = ""
            try:
                body = response.json()
                error = body.get("error", {}) if isinstance(body, dict) else {}
                code = error.get("code", "") if isinstance(error, dict) else ""
            except ValueError:
                pass
            retryable = response.status_code >= 500 or response.status_code == 429
            if response.status_code == 429:
                try:
                    delay = float(response.headers.get("Retry-After", "0"))
                    if math.isfinite(delay) and delay >= 0:
                        request.retry_after = delay
                except ValueError:
                    pass
                # model_busy closes the attempt without inference/charge. A new
                # ID is required; upload_busy and transport failures retain it.
                if code == "model_busy":
                    request.id = str(uuid.uuid4())
            raise APIStatusError(
                f"Oruk transcription failed ({code or response.status_code})",
                status_code=response.status_code,
                request_id=response.headers.get("X-Request-ID"),
                retryable=retryable,
            )
        try:
            result = response.json()
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                raise ValueError("missing transcript")
            language_code = result.get("language") or ""
            if not isinstance(language_code, str):
                raise ValueError("invalid language")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=response.headers.get("X-Request-ID", request.id),
                alternatives=[
                    stt.SpeechData(text=result["text"], language=LanguageCode(language_code))
                ],
            )
        except (ValueError, TypeError) as exc:
            raise APIError("Invalid Oruk transcription response", retryable=False) from exc


def _wav(buffer: AudioBuffer) -> bytes:
    frame = rtc.combine_audio_frames(buffer)
    duration = frame.samples_per_channel / frame.sample_rate
    if not 0.045 <= duration <= 60:
        raise ValueError("Oruk utterances must be between 45 ms and 60 seconds")
    if frame.num_channels != 1:
        samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)
        mono = np.rint(samples.astype(np.float32).mean(axis=1)).astype(np.int16)
        frame = rtc.AudioFrame(
            data=mono.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=1,
            samples_per_channel=frame.samples_per_channel,
        )
    if frame.sample_rate != 16000:
        resampler = rtc.AudioResampler(
            input_rate=frame.sample_rate, output_rate=16000, num_channels=1
        )
        frame = rtc.combine_audio_frames([*resampler.push(frame), *resampler.flush()])
    return frame.to_wav_bytes()
