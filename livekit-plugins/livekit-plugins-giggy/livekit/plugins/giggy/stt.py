from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

GIGGY_ORIGIN = "https://giggy.ai"
SAMPLE_RATE = 16000
MAX_JSON_BYTES = 1024 * 1024
LANGUAGES = {
    LanguageCode(name).language: name
    for name in (
        "Arabic",
        "Cantonese",
        "Chinese",
        "Czech",
        "Dutch",
        "English",
        "Filipino",
        "French",
        "German",
        "Greek",
        "Hindi",
        "Hungarian",
        "Indonesian",
        "Italian",
        "Japanese",
        "Korean",
        "Malay",
        "Persian",
        "Polish",
        "Portuguese",
        "Romanian",
        "Russian",
        "Spanish",
        "Swedish",
        "Thai",
        "Turkish",
        "Vietnamese",
    )
}
LANGUAGES.update({"yue": "Cantonese", "fil": "Filipino"})
LANGUAGE_CODES = {name: code for code, name in LANGUAGES.items()}


class STT(stt.STT):
    """Recognize complete utterances using Giggy's canonical Streaming WAV API.

    AgentSession uses LiveKit's VAD stream adapter for turn segmentation.
    This provider returns final transcripts and does not expose live-input STT.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str | None = None,
        recognition_timeout: float = 90.0,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create an utterance recognizer; language defaults to automatic detection.

        Args:
            api_key: Giggy API key, or GIGGY_API_KEY from the environment.
            language: Supported language name or BCP-47 code, or None for detection.
            recognition_timeout: Overall recognition deadline in seconds.
            http_session: Borrowed, credential-free session; never closed by this provider.
        """
        resolved_key = api_key or os.getenv("GIGGY_API_KEY")
        if not resolved_key:
            raise ValueError("Giggy API key is required.")
        if not math.isfinite(recognition_timeout) or recognition_timeout <= 0:
            raise ValueError("recognition_timeout must be finite and positive.")
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._api_key = resolved_key
        self._language = self._resolve_language(language)
        self._recognition_timeout = recognition_timeout
        self._http_session = http_session
        self._requests: set[asyncio.Task[stt.SpeechEvent]] = set()
        self._closed = False

    @property
    def provider(self) -> str:
        return "Giggy"

    @property
    def model(self) -> str:
        return "giggyspeech-stt"

    def _resolve_language(self, language: str | None) -> str | None:
        if language is None:
            return None
        name = LANGUAGES.get(LanguageCode(language).language)
        if name is None:
            raise ValueError("Unsupported Giggy STT language.")
        return name

    def _ensure_session(self) -> aiohttp.ClientSession:
        session = (
            self._http_session
            if self._http_session is not None
            else utils.http_context.http_session()
        )
        if (
            session.auth is not None
            or session.cookie_jar
            or any(
                name.lower()
                in {
                    "authorization",
                    "proxy-authorization",
                    "xi-api-key",
                    "x-api-key",
                    "x-giggy-input-token",
                    "cookie",
                }
                for name in session.headers
            )
        ):
            raise ValueError(
                "Giggy STT requires a credential-free HTTP session for signed artifacts."
            )
        return session

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Admit once; retries are disabled because a request may already be charged."""
        if self._closed:
            raise RuntimeError("Giggy STT provider is closed.")
        task = asyncio.create_task(
            super().recognize(
                buffer, language=language, conn_options=replace(conn_options, max_retry=0)
            )
        )
        self._requests.add(task)
        try:
            return await asyncio.wait_for(task, timeout=self._recognition_timeout)
        except asyncio.TimeoutError:
            error = APITimeoutError("Giggy STT recognition deadline expired.", retryable=False)
            self._emit_error(error, recoverable=False)
            raise error from None
        finally:
            self._requests.discard(task)

    async def aclose(self) -> None:
        """Cancel active recognition and leave borrowed HTTP sessions open."""
        self._closed = True
        requests = list(self._requests)
        for task in requests:
            task.cancel()
        if requests:
            await asyncio.gather(*requests, return_exceptions=True)

    async def _read_json(self, response: aiohttp.ClientResponse, *, status: int) -> dict:
        if response.status != status:
            raise APIStatusError(
                "Giggy STT HTTP request failed.",
                status_code=response.status,
                request_id=response.headers.get("x-request-id"),
                body=None,
                retryable=False,
            )
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise APIConnectionError(
                "Giggy STT returned an unexpected response format.", retryable=False
            )
        body = bytearray()
        async for chunk in response.content.iter_chunked(4096):
            body.extend(chunk)
            if len(body) > MAX_JSON_BYTES:
                raise APIConnectionError(
                    "Giggy STT response exceeded its size limit.", retryable=False
                )
        result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError("Invalid response object")
        return result

    async def _sse_events(
        self, response: aiohttp.ClientResponse
    ) -> AsyncIterator[tuple[str, dict]]:
        pending = b""
        event = ""
        data: list[str] = []
        size = 0
        async for chunk in response.content.iter_chunked(4096):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                size += len(line) + 1
                if size > MAX_JSON_BYTES:
                    raise ValueError("SSE event exceeds limit")
                line = line.rstrip(b"\r")
                if not line:
                    if data:
                        payload = json.loads("\n".join(data))
                        if not isinstance(payload, dict):
                            raise ValueError("Invalid SSE object")
                        yield event, payload
                    event, data, size = "", [], 0
                elif line.startswith(b"event:"):
                    event = line[6:].lstrip(b" ").decode("utf-8")
                elif line.startswith(b"data:"):
                    data.append(line[5:].lstrip(b" ").decode("utf-8"))
            if len(pending) + size > MAX_JSON_BYTES:
                raise ValueError("SSE line exceeds limit")
        if pending or data:
            raise ValueError("Incomplete SSE event")

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        selected_language = (
            self._resolve_language(language) if is_given(language) else self._language
        )
        frame = rtc.combine_audio_frames(buffer)
        if frame.num_channels != 1:
            raise ValueError("Giggy STT requires mono audio.")
        duration = frame.samples_per_channel / frame.sample_rate
        if not 0.5 <= duration <= 43200:
            raise ValueError("Giggy STT requires 0.5 seconds to 12 hours of audio.")
        if frame.sample_rate != SAMPLE_RATE:
            resampler = rtc.AudioResampler(frame.sample_rate, SAMPLE_RATE, num_channels=1)
            frame = rtc.combine_audio_frames(resampler.push(frame) + resampler.flush())
        session = self._ensure_session()
        headers = {"xi-api-key": self._api_key}
        request_id = str(uuid4())
        params = {"mode": "streaming", "client_request_id": request_id}
        if selected_language is not None:
            params["language"] = selected_language
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=conn_options.timeout,
            sock_connect=conn_options.timeout,
            sock_read=conn_options.timeout,
        )
        try:
            generation = None
            generation_id = None
            attempt = None
            last_sequence = -1
            final_text = None
            async with session.post(
                GIGGY_ORIGIN + "/v1/audio/transcriptions",
                params=params,
                headers={**headers, "Content-Type": "audio/wav", "Accept": "text/event-stream"},
                data=frame.to_wav_bytes(),
                timeout=timeout,
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise APIStatusError(
                        "Giggy STT HTTP request failed.",
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                        body=None,
                        retryable=False,
                    )
                if (
                    response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                    != "text/event-stream"
                ):
                    raise ValueError("Expected SSE response")
                async for event, payload in self._sse_events(response):
                    if event == "generation_created":
                        if generation_id is not None:
                            raise ValueError("Duplicate admission")
                        generation_id = payload.get("generation_uuid") or payload.get(
                            "generation", {}
                        ).get("generation_uuid")
                        if (
                            not isinstance(generation_id, str)
                            or str(UUID(generation_id)) != generation_id
                        ):
                            raise ValueError("Invalid admission identity")
                    elif event == "transcript_chunk":
                        sequence = payload.get("sequence")
                        if (
                            generation_id is None
                            or payload.get("generation_uuid") != generation_id
                            or (
                                type(payload.get("attempt")) is not int
                                or payload["attempt"] < 1
                                or (attempt is not None and payload["attempt"] != attempt)
                                or type(sequence) is not int
                                or sequence <= last_sequence
                                or payload.get("replace") is not True
                                or type(payload.get("complete")) is not bool
                                or not isinstance(payload.get("text"), str)
                                or final_text is not None
                                or generation is not None
                            )
                        ):
                            raise ValueError("Invalid transcript snapshot")
                        attempt, last_sequence = payload["attempt"], sequence
                        if payload["complete"]:
                            final_text = payload["text"]
                    elif event == "generation_updated":
                        if generation_id is None or payload.get("generation_uuid") != generation_id:
                            raise ValueError("Generation identity mismatch")
                        current = payload.get("generation", {})
                        if not isinstance(current, dict):
                            raise ValueError("Invalid generation metadata")
                        state = payload.get("status") or current.get("status")
                        if state in {"failed", "cancelled", "admission_failed"}:
                            raise APIConnectionError(
                                "Giggy STT operation did not complete.", retryable=False
                            )
                        if state == "completed":
                            if (
                                generation is not None
                                or final_text is None
                                or current.get("generation_uuid") != generation_id
                            ):
                                raise ValueError("Invalid completion metadata")
                            generation = current
                        elif state not in {"queued", "processing"}:
                            raise ValueError("Invalid generation state")
                    elif event == "error":
                        raise APIConnectionError("Giggy STT stream failed.", retryable=False)
            if generation is None or generation_id is None or final_text is None:
                raise ValueError("STT stream ended without canonical completion")
            # Resolve the retained result through the authenticated public projection.
            async with session.get(
                f"{GIGGY_ORIGIN}/v1/generations/{generation_id}",
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
            ) as response:
                generation = (await self._read_json(response, status=200))["generation"]
            if (
                not isinstance(generation, dict)
                or generation.get("generation_uuid") != generation_id
                or generation.get("status") != "completed"
            ):
                raise ValueError("Canonical completion was not confirmed")
            result_url = generation["result"]["url"]
            if not isinstance(result_url, str):
                raise ValueError("Invalid artifact URL")
            origin = urlsplit(result_url)
            if (
                origin.scheme != "https"
                or not origin.hostname
                or origin.username is not None
                or origin.password is not None
            ):
                raise ValueError("Invalid artifact origin")
            # Recheck mutable session defaults before crossing to a signed artifact origin.
            self._ensure_session()
            async with session.get(result_url, timeout=timeout, allow_redirects=False) as response:
                result = await self._read_json(response, status=200)
            if (
                not isinstance(result.get("text"), str)
                or result["text"] != final_text
                or (
                    type(result.get("audio_duration_ms")) is not int
                    or result["audio_duration_ms"] != math.ceil(frame.samples_per_channel / 16)
                )
            ):
                raise ValueError("Invalid canonical transcript")
            detected_languages = result.get("languages")
            if (
                not isinstance(detected_languages, list)
                or any(
                    not isinstance(item, str) or item not in LANGUAGE_CODES
                    for item in detected_languages
                )
                or (result["text"] and not detected_languages)
            ):
                raise ValueError("Invalid transcript language")
            languages = [LanguageCode(LANGUAGE_CODES[item]) for item in detected_languages]
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=generation_id,
                alternatives=[
                    stt.SpeechData(
                        text=result["text"],
                        language=languages[0]
                        if languages
                        else LanguageCode(selected_language or ""),
                        source_languages=languages,
                        end_time=frame.samples_per_channel / SAMPLE_RATE,
                    )
                ],
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Giggy STT request timed out.", retryable=False) from None
        except aiohttp.ClientError:
            raise APIConnectionError("Giggy STT network request failed.", retryable=False) from None
        except (ValueError, TypeError, KeyError, AttributeError):
            raise APIConnectionError(
                "Giggy STT returned an invalid response.", retryable=False
            ) from None
