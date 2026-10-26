from __future__ import annotations

import asyncio
import json
import math
import os
import re
import tempfile
import threading
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, BinaryIO
from urllib.parse import urlsplit
from uuid import UUID, uuid4
from weakref import WeakSet

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
    vad as lk_vad,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

GIGGY_ORIGIN = "https://api.giggy.ai"
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
    """Stream live PCM through Giggy, retaining explicit recording recognition."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str | None = None,
        recognition_timeout: float = 90.0,
        http_session: aiohttp.ClientSession | None = None,
        vad: lk_vad.VAD | None = None,
    ) -> None:
        """Create an utterance recognizer; language defaults to automatic detection.

        Args:
            api_key: Giggy API key, or GIGGY_API_KEY from the environment.
            language: Supported language name or BCP-47 code, or None for detection.
            recognition_timeout: Overall recognition deadline in seconds.
            http_session: Borrowed, credential-free session; never closed by this provider.
            vad: Native endpoint detector; defaults to Silero with 100 ms silence.
        """
        resolved_key = api_key or os.getenv("GIGGY_API_KEY")
        if not resolved_key:
            raise ValueError("Giggy API key is required.")
        if not math.isfinite(recognition_timeout) or recognition_timeout <= 0:
            raise ValueError("recognition_timeout must be finite and positive.")
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self._api_key = resolved_key
        self._language = self._resolve_language(language)
        self._recognition_timeout = recognition_timeout
        self._http_session = http_session
        self._requests: set[asyncio.Task[stt.SpeechEvent]] = set()
        self._closed = False
        self._vad = vad
        self._streams: WeakSet[SpeechStream] = WeakSet()

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Stream once per utterance; never replay ambiguous audio."""
        if self._closed:
            raise RuntimeError("Giggy STT provider is closed.")
        detector = self._vad
        if detector is None:
            from livekit.plugins import silero

            detector = self._vad = silero.VAD.load(
                min_speech_duration=0.2,
                min_silence_duration=0.1,
                prefix_padding_duration=1.0,
                max_buffered_speech=30.0,
                activation_threshold=0.7,
                deactivation_threshold=0.35,
            )
        selected = self._resolve_language(language) if is_given(language) else self._language
        stream = SpeechStream(self, detector, selected, replace(conn_options, max_retry=0))
        self._streams.add(stream)
        return stream

    @property
    def provider(self) -> str:
        """The provider name reported in LiveKit metrics."""
        return "Giggy"

    @property
    def model(self) -> str:
        """The canonical Giggy speech recognition model identifier."""
        return "giggyspeech-stt"

    def _resolve_language(self, language: str | None) -> str | None:
        """Return the supported language name, or None for automatic detection."""
        if language is None:
            return None
        name = LANGUAGES.get(LanguageCode(language).language)
        if name is None:
            raise ValueError("Unsupported Giggy STT language.")
        return name

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Return a credential-free borrowed or context-managed HTTP session."""
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
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))
        self._streams.clear()
        requests = list(self._requests)
        for task in requests:
            task.cancel()
        if requests:
            await asyncio.gather(*requests, return_exceptions=True)

    async def _read_json(self, response: aiohttp.ClientResponse, *, status: int) -> dict:
        """Validate the expected HTTP status and read a bounded JSON object."""
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
        """Yield complete SSE event names and JSON objects within the size limit."""
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
        """Validate and stage a recording, then return its verified final transcript event."""
        selected_language = (
            self._resolve_language(language) if is_given(language) else self._language
        )
        frames = [buffer] if isinstance(buffer, rtc.AudioFrame) else buffer
        if not frames or any(frame.num_channels != 1 for frame in frames):
            raise ValueError("Giggy STT requires mono audio.")
        rate = frames[0].sample_rate
        if rate <= 0 or any(frame.sample_rate != rate for frame in frames):
            raise ValueError("Giggy STT requires consistent positive input sample rates.")
        duration = sum(frame.samples_per_channel for frame in frames) / rate
        if not 0.5 <= duration <= 43200:
            raise ValueError("Giggy STT accepts recordings from 0.5 seconds to 12 hours.")
        session = self._ensure_session()
        cancelled = threading.Event()
        # Stage native WAV on disk without copying or merging the full recording.
        # Await the encoder on cancellation so it cannot write to a closed file.
        # A concrete binary file is recognized by aiohttp on Windows as well.
        with (
            tempfile.TemporaryFile(mode="w+b") as temporary,
            open(temporary.fileno(), "r+b", closefd=False) as recording,
        ):
            encoding = asyncio.create_task(
                asyncio.to_thread(self._encode_wav, frames, recording, cancelled)
            )
            try:
                samples = await asyncio.shield(encoding)
            except BaseException:
                cancelled.set()
                await asyncio.gather(encoding, return_exceptions=True)
                raise
            return await self._transcribe_wav(
                session, recording, samples, selected_language, conn_options
            )

    def _encode_wav(
        self, frames: list[rtc.AudioFrame], destination: BinaryIO, cancelled: threading.Event
    ) -> int:
        """Stage native WAV in bounded chunks and return the encoded sample count."""
        rate = frames[0].sample_rate
        resampler = (
            rtc.AudioResampler(rate, SAMPLE_RATE, num_channels=1) if rate != SAMPLE_RATE else None
        )
        samples = 0
        with wave.open(destination, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SAMPLE_RATE)
            for frame in frames:
                source = frame.data.cast("B")
                for start in range(0, len(source), 8192):
                    if cancelled.is_set():
                        raise asyncio.CancelledError
                    pcm = source[start : start + 8192]
                    if resampler is None:
                        writer.writeframesraw(pcm)
                        samples += len(pcm) // 2
                    else:
                        chunk = rtc.AudioFrame(
                            data=pcm,
                            sample_rate=rate,
                            num_channels=1,
                            samples_per_channel=len(pcm) // 2,
                        )
                        for output in resampler.push(chunk):
                            writer.writeframesraw(output.data)
                            samples += output.samples_per_channel
            if resampler is not None:
                for output in resampler.flush():
                    writer.writeframesraw(output.data)
                    samples += output.samples_per_channel
        destination.seek(0)
        return samples

    async def _transcribe_wav(
        self,
        session: aiohttp.ClientSession,
        recording: BinaryIO,
        samples: int,
        selected_language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Upload the staged WAV and verify the completed transcript artifact."""
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
                data=recording,
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
                or re.fullmatch(r"[a-f0-9]{32}\.r2\.cloudflarestorage\.com", origin.hostname or "")
                is None
                or origin.port not in (None, 443)
                or origin.fragment
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
                    or result["audio_duration_ms"] != math.ceil(samples / 16)
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
                        end_time=samples / SAMPLE_RATE,
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


class SpeechStream(stt.RecognizeStream):
    """Native LiveKit frames on the canonical Giggy one-socket input rail."""

    def __init__(
        self,
        owner: STT,
        detector: lk_vad.VAD,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> None:
        """Initialize bounded live capture, endpoint detection and ordered turn delivery."""
        self._owner, self._detector, self._language = owner, detector, language
        self._turns: set[asyncio.Task[None]] = set()
        self._failure: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active: dict[str, Any] | None = None
        self._previous: asyncio.Event | None = None
        self._pending_bytes = 0
        self._pre_roll = bytearray()
        self._unprocessed = bytearray()
        super().__init__(stt=owner, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._input_ch = utils.aio.Chan(maxsize=256)

    async def _run(self) -> None:
        """Drain live input and pending turns, cancelling tasks on failure or closure."""
        producer = asyncio.create_task(self._feed_input())
        try:
            done, _ = await asyncio.wait(
                {producer, self._failure}, return_when=asyncio.FIRST_COMPLETED
            )
            if self._failure in done:
                await self._failure
            await producer
            await asyncio.gather(*tuple(self._turns))
            if self._failure.done():
                await self._failure
        finally:
            producer.cancel()
            tasks = tuple(self._turns)
            for task in tasks:
                task.cancel()
            await asyncio.gather(producer, *tasks, return_exceptions=True)
            self._failure.cancel()

    async def _feed_input(self) -> None:
        """Feed bounded real PCM frames into VAD and flush the detector at input boundaries."""
        detector = self._detector.stream()
        consumer = asyncio.create_task(self._consume_vad(detector))
        consumer.add_done_callback(self._turn_done)
        try:
            async for frame in self._input_ch:
                if isinstance(frame, self._FlushSentinel):
                    # Flush real pending PCM and finalize; never manufacture silence.
                    detector.end_input()
                    await consumer
                    await detector.aclose()
                    detector = self._detector.stream()
                    consumer = asyncio.create_task(self._consume_vad(detector))
                    consumer.add_done_callback(self._turn_done)
                    continue
                if frame.num_channels != 1 or not 0 < len(frame.data) <= 16000:
                    raise ValueError("Giggy live STT requires bounded mono PCM frames.")
                self._unprocessed.extend(frame.data.cast("B"))
                if len(self._unprocessed) + self._pending_bytes > 960000:
                    raise ValueError("Giggy live STT capture queue is full.")
                detector.push_frame(frame)
                if consumer.done():
                    await consumer
            detector.end_input()
            await consumer
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
            await detector.aclose()

    async def _consume_vad(self, detector: lk_vad.VADStream) -> None:
        """Build utterances from ordered VAD events while preserving every captured PCM sample."""
        async for event in detector:
            if event.type == lk_vad.VADEventType.INFERENCE_DONE:
                # Silero publishes START after INFERENCE_DONE for the same
                # window. Let that START withdraw a short end before deciding
                # to finalize it on the following window.
                if (
                    self._active is not None
                    and self._active["short_end"]
                    and self._active["samples"] >= 8000
                ):
                    self._finish_turn()
                pcm = b"".join(frame.data.cast("B") for frame in event.frames)
                if not pcm or self._unprocessed[: len(pcm)] != pcm:
                    raise ValueError("VAD did not preserve the input PCM timeline.")
                del self._unprocessed[: len(pcm)]
                if self._active is None:
                    self._pre_roll.extend(pcm)
                    del self._pre_roll[:-32000]
                else:
                    self._queue_audio(self._active, pcm, event.raw_accumulated_silence > 0)
            elif event.type == lk_vad.VADEventType.START_OF_SPEECH:
                if self._active is None:
                    self._begin_turn()
                else:
                    self._active["short_end"] = False
                    self._active["boundary"] = None
                    self._active["speech_end"] = None
                self._event_ch.send_nowait(stt.SpeechEvent(stt.SpeechEventType.START_OF_SPEECH))
            elif event.type == lk_vad.VADEventType.END_OF_SPEECH and self._active is not None:
                self._active["speech_end"] = (
                    time.time() - event.silence_duration - event.inference_duration
                )
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        stt.SpeechEventType.END_OF_SPEECH,
                        speech_end_time=self._active["speech_end"],
                    )
                )
                if self._active["samples"] >= 8000:
                    self._finish_turn()
                else:
                    self._active["short_end"] = True
        # A VAD window may leave a real sub-window tail at clean source EOF.
        if self._active is not None:
            # EOF is not VAD-confirmed silence. A tentative boundary must not
            # discard resumed speech in the detector's last inference window.
            self._active["boundary"] = None
            if self._active["speech_end"] is None:
                self._event_ch.send_nowait(stt.SpeechEvent(stt.SpeechEventType.END_OF_SPEECH))
            if self._unprocessed:
                self._queue_audio(self._active, bytes(self._unprocessed), False)
            self._finish_turn()
        self._pre_roll.clear()
        self._unprocessed.clear()

    def _begin_turn(self) -> None:
        """Start a bounded transcription turn with pre-roll and ordered publication state."""
        if len(self._turns) >= 8:
            raise ValueError("Giggy live STT has too many pending turns.")
        turn: dict[str, Any] = {
            "queue": asyncio.Queue(maxsize=256),
            "samples": 0,
            "frames": 0,
            "boundary": None,
            "short_end": False,
            "ended": False,
            "sent_end": False,
            "ready": asyncio.Event(),
            "published": asyncio.Event(),
            "previous": self._previous,
            "speech_end": None,
            "id": None,
            "attempt": None,
        }
        self._active = turn
        self._queue_audio(turn, bytes(self._pre_roll), False)
        self._pre_roll.clear()
        self._previous = turn["published"]
        task = asyncio.create_task(
            asyncio.wait_for(self._transcribe_turn(turn), timeout=self._owner._recognition_timeout)
        )
        self._turns.add(task)
        task.add_done_callback(self._turn_done)

    def _queue_audio(self, turn: dict[str, Any], pcm: bytes, quiet: bool) -> None:
        """Queue real PCM and track tentative silence boundaries within the capture limits."""
        if not pcm:
            return
        turn["samples"] += len(pcm) // 2
        self._pending_bytes += len(pcm)
        if turn["samples"] > 30 * SAMPLE_RATE or self._pending_bytes > 960000:
            raise ValueError("Giggy live STT capture exceeds its bounded queue or 30-second turn.")
        if not quiet:
            turn["boundary"] = None
        elif turn["boundary"] is None or turn["samples"] - turn["boundary"] > 2400:
            turn["boundary"] = turn["samples"]
        for start in range(0, len(pcm), 16000):
            turn["queue"].put_nowait((pcm[start : start + 16000], turn["boundary"], False))

    def _finish_turn(self) -> None:
        """Validate the active turn's minimum duration and queue its final input marker."""
        turn = self._active
        if turn is None:
            return
        if turn["samples"] < 8000:
            raise ValueError("Giggy live STT needs at least 0.5 seconds of real PCM.")
        turn["ended"] = True
        turn["queue"].put_nowait((b"", turn["boundary"], True))
        self._active = None

    def _turn_done(self, task: asyncio.Task[None]) -> None:
        """Retire a completed turn task and propagate its first failure to the stream."""
        self._turns.discard(task)
        if (
            not task.cancelled()
            and (error := task.exception()) is not None
            and not self._failure.done()
        ):
            if isinstance(error, asyncio.TimeoutError):
                error = APITimeoutError("Giggy live STT deadline expired.", retryable=False)
            self._failure.set_exception(error)

    async def _send_audio(
        self, turn: dict[str, Any], socket: aiohttp.ClientWebSocketResponse
    ) -> None:
        """Send queued PCM once, update decode preparation and stop at the turn boundary."""
        await turn["ready"].wait()
        sent_samples, prepared = 0, None
        while True:
            pcm, candidate, complete = await turn["queue"].get()
            if candidate != prepared and (
                prepared is not None or candidate is not None and candidate <= sent_samples
            ):
                prepared = (
                    candidate if candidate is not None and candidate <= sent_samples else None
                )
                await socket.send_json({"type": "prepare", "decode_end_sample": prepared})
            if pcm:
                await socket.send_bytes(pcm)
                sent_samples += len(pcm) // 2
                turn["frames"] += 1
                self._pending_bytes -= len(pcm)
            if candidate != prepared and candidate is not None and candidate <= sent_samples:
                if candidate < sent_samples - 3200:
                    raise ValueError("Canonical preparation boundary expired.")
                await socket.send_json({"type": "prepare", "decode_end_sample": candidate})
                prepared = candidate
            if complete:
                if sent_samples != turn["samples"]:
                    raise ValueError("Captured PCM was not sent exactly once.")
                turn["sent_end"] = True
                await socket.send_json({"type": "stop"})
                return

    async def _transcribe_turn(self, turn: dict[str, Any]) -> None:
        """Deliver one WebSocket turn in order, then verify its durable transcript result."""
        sender = None
        try:
            session = self._owner._ensure_session()
            connected = time.perf_counter()
            async with await asyncio.wait_for(
                session.ws_connect(
                    f"{GIGGY_ORIGIN}/v1/audio/transcriptions",
                    headers={"Authorization": "Bearer " + self._owner._api_key},
                    max_msg_size=16 * 1024 * 1024,
                    compress=0,
                    receive_timeout=min(30, self._conn_options.timeout),
                ),
                timeout=self._conn_options.timeout,
            ) as socket:
                self._report_connection_acquired(time.perf_counter() - connected, False)
                await socket.send_json(
                    {
                        "type": "start",
                        "format": "raw",
                        "encoding": "LINEAR16",
                        "sampleRateHz": SAMPLE_RATE,
                        "interimResults": True,
                        "options": {"endpointing": "external"},
                        **({"language": LANGUAGE_CODES[self._language]} if self._language else {}),
                    }
                )
                sender = asyncio.create_task(self._send_audio(turn, socket))
                sender.add_done_callback(self._turn_done)
                acknowledged, completed, final_text, last_sequence = False, False, None, -1
                async for message in socket:
                    if sender.done():
                        await sender
                    if message.type != aiohttp.WSMsgType.TEXT:
                        raise ValueError("Expected a live STT control event.")
                    payload = json.loads(message.data)
                    kind = payload.get("type")
                    if kind == "generation_created":
                        gid = payload.get("generation_uuid") or payload.get("generation", {}).get(
                            "generation_uuid"
                        )
                        if (
                            turn["id"] is not None
                            or not isinstance(gid, str)
                            or str(UUID(gid)) != gid
                        ):
                            raise ValueError("Invalid admission identity.")
                        turn["id"] = gid
                        continue
                    if not turn["id"] or payload.get("generation_uuid") != turn["id"]:
                        raise ValueError("Live STT generation identity changed.")
                    if kind == "ready":
                        if (
                            turn["ready"].is_set()
                            or type(payload.get("attempt")) is not int
                            or payload["attempt"] < 1
                            or payload.get("sampleRateHz") != SAMPLE_RATE
                        ):
                            raise ValueError("Invalid live STT readiness.")
                        turn["attempt"] = payload["attempt"]
                        turn["ready"].set()
                    elif kind == "input_complete":
                        if (
                            acknowledged
                            or not turn["sent_end"]
                            or type(payload.get("sequence")) is not int
                            or payload["sequence"] != turn["frames"]
                        ):
                            raise ValueError("Invalid final input acknowledgement.")
                        acknowledged = True
                    elif kind == "transcription":
                        alternatives = payload.get("alternatives")
                        text = (
                            alternatives[0].get("transcript")
                            if isinstance(alternatives, list)
                            and alternatives
                            and isinstance(alternatives[0], dict)
                            else None
                        )
                        sequence = payload.get("sequence")
                        if (
                            not turn["ready"].is_set()
                            or type(payload.get("attempt")) is not int
                            or payload["attempt"] != turn["attempt"]
                            or type(sequence) is not int
                            or sequence <= last_sequence
                            or payload.get("replace") is not True
                            or type(payload.get("is_final")) is not bool
                            or not isinstance(text, str)
                            or final_text is not None
                        ):
                            raise ValueError("Invalid transcript snapshot.")
                        last_sequence = sequence
                        if payload["is_final"]:
                            if not acknowledged:
                                raise ValueError("Final transcript preceded input acceptance.")
                            if turn["previous"] is not None:
                                await turn["previous"].wait()
                            final_text = text
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                stt.SpeechEventType.FINAL_TRANSCRIPT
                                if payload["is_final"]
                                else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                request_id=turn["id"],
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        language=LanguageCode(
                                            payload.get("language")
                                            or LANGUAGE_CODES.get(self._language or "", "")
                                        ),
                                        end_time=turn["samples"] / SAMPLE_RATE,
                                    )
                                ],
                                speech_end_time=turn["speech_end"] if payload["is_final"] else None,
                            )
                        )
                        if payload["is_final"]:
                            turn["published"].set()
                    elif kind == "generation_updated":
                        status = payload.get("status") or payload.get("generation", {}).get(
                            "status"
                        )
                        if status == "completed":
                            if final_text is None or not acknowledged:
                                raise ValueError("Completion preceded a fenced final.")
                            completed = True
                            break
                        if status not in {"queued", "processing"}:
                            raise ValueError("Live STT operation did not complete.")
                    else:
                        raise ValueError("Unexpected live STT event.")
                if not completed or final_text is None:
                    raise ValueError("Live STT closed without canonical completion.")
                await sender
            await self._verify_result(turn, final_text)
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    stt.SpeechEventType.RECOGNITION_USAGE,
                    request_id=turn["id"],
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=turn["samples"] / SAMPLE_RATE
                    ),
                )
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Giggy live STT deadline expired.", retryable=False) from None
        except (aiohttp.ClientError, ValueError, TypeError, KeyError, AttributeError):
            raise APIConnectionError(
                "Giggy live STT protocol or verification failed.", retryable=False
            ) from None
        finally:
            if sender is not None:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

    async def _verify_result(self, turn: dict[str, Any], final_text: str) -> None:
        """Verify canonical completion and match the signed artifact's text and PCM duration."""
        session = self._owner._ensure_session()
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(
            f"{GIGGY_ORIGIN}/v1/generations/{turn['id']}",
            headers={"xi-api-key": self._owner._api_key},
            timeout=timeout,
            allow_redirects=False,
        ) as response:
            generation = (await self._owner._read_json(response, status=200))["generation"]
        if (
            generation.get("generation_uuid") != turn["id"]
            or generation.get("status") != "completed"
        ):
            raise ValueError("Canonical completion was not confirmed.")
        result_url = generation["result"]["url"]
        origin = urlsplit(result_url)
        if (
            origin.scheme != "https"
            or re.fullmatch(r"[a-f0-9]{32}\.r2\.cloudflarestorage\.com", origin.hostname or "")
            is None
            or origin.port not in (None, 443)
            or origin.fragment
            or origin.username is not None
            or origin.password is not None
        ):
            raise ValueError("Invalid signed artifact origin.")
        self._owner._ensure_session()
        async with session.get(result_url, timeout=timeout, allow_redirects=False) as response:
            result = await self._owner._read_json(response, status=200)
        if (
            result.get("text") != final_text
            or type(result.get("audio_duration_ms")) is not int
            or result["audio_duration_ms"] != math.ceil(turn["samples"] / 16)
        ):
            raise ValueError("Saved transcript or captured duration mismatch.")
