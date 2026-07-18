# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Blaze STT Plugin for LiveKit Voice Agent

Speech-to-Text plugin that interfaces with Blaze's transcription service.

Batch API: POST /v1/stt/transcribe (model default: v2.0)
Realtime API: WS /v1/stt/realtime (model default: stt-stream-1.5)
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid

import httpx
import websockets

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    NotGivenOr,
    stt,
)
from livekit.agents.stt import StreamAdapter
from livekit.agents.utils import AudioBuffer, shortuuid

from ._config import BlazeConfig
from ._utils import apply_normalization_rules, convert_pcm_to_wav, effective_connect_timeout
from .log import logger

# Latest public Blaze STT model identifiers.
DEFAULT_BATCH_MODEL = "v2.0"
DEFAULT_STREAM_MODEL = "stt-stream-1.5"


class STT(stt.STT):
    """
    Blaze Speech-to-Text Plugin.

    Converts speech to text using Blaze's transcription service.

    * Batch recognition via ``recognize()`` / ``_recognize_impl`` uses
      ``POST /v1/stt/transcribe`` with model ``v2.0`` by default.
    * Native streaming via ``stream()`` uses WebSocket ``/v1/stt/realtime``
      with model ``stt-stream-1.5`` by default (partial + final transcripts).
    * ``with_streaming(vad)`` remains available for VAD-segmented batch STT.

    Args:
        api_url: Base URL for the STT service.
        language: Language code for transcription (default: "vi").
        auth_token: Bearer token for authentication.
        sample_rate: Audio sample rate in Hz (default: 16000).
        model: Batch STT model id (default: v2.0).
        stream_model: Realtime STT model id (default: stt-stream-1.5).
        normalization_rules: Dict mapping input strings to replacements.
        timeout: Request timeout in seconds (default: 30.0).
        config: Optional BlazeConfig for centralized configuration.

    Example:
        >>> from livekit.plugins import blaze
        >>> stt = blaze.STT(language="vi")  # streaming-capable, latest models
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        language: str = "vi",
        auth_token: str | None = None,
        sample_rate: int = 16000,
        model: str = DEFAULT_BATCH_MODEL,
        stream_model: str = DEFAULT_STREAM_MODEL,
        normalization_rules: dict[str, str] | None = None,
        timeout: float | None = None,
        config: BlazeConfig | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )

        self._config = config or BlazeConfig()
        self._api_url = api_url or self._config.api_url
        self._language = language
        self._auth_token = auth_token or self._config.api_token
        self._sample_rate = sample_rate
        self._model = model
        self._stream_model = stream_model
        self._timeout = timeout if timeout is not None else self._config.stt_timeout
        self._normalization_rules = normalization_rules
        self._transcribe_url = f"{self._api_url}/v1/stt/transcribe"
        ws_base = self._api_url.replace("https://", "wss://").replace("http://", "ws://")
        self._ws_url = f"{ws_base}/v1/stt/realtime"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._timeout, connect=5.0))

        # Frame accumulation: buffer PCM from empty STT segments so short
        # leading fragments (hesitant speech) are prepended to the next segment.
        self._pending_pcm: bytes = b""
        self._pending_sample_rate: int = sample_rate
        self._pending_num_channels: int = 1
        self._pending_empty_count: int = 0
        self._last_recognize_time: float = 0.0

        # Safety limits
        self._max_pending_duration: float = 5.0  # seconds of buffered audio
        self._max_pending_segments: int = 3  # consecutive empty segments
        self._pending_idle_timeout: float = 10.0  # auto-clear after idle gap

        logger.info(
            "BlazeSTT initialized: url=%s, language=%s, batch_model=%s, stream_model=%s",
            self._api_url,
            self._language,
            self._model,
            self._stream_model,
        )

    @property
    def provider(self) -> str:
        return "Blaze"

    @property
    def model(self) -> str:
        """Primary model id (batch). Streaming uses ``stream_model``."""
        return self._model

    @property
    def stream_model(self) -> str:
        return self._stream_model

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def aclose(self) -> None:
        await self._client.aclose()
        await super().aclose()

    def update_options(
        self,
        *,
        language: str | None = None,
        auth_token: str | None = None,
        model: str | None = None,
        stream_model: str | None = None,
        normalization_rules: dict[str, str] | None = None,
    ) -> None:
        """Update STT options at runtime."""
        if language is not None:
            self._language = language
        if auth_token is not None:
            self._auth_token = auth_token
        if model is not None:
            self._model = model
        if stream_model is not None:
            self._stream_model = stream_model
        if normalization_rules is not None:
            self._normalization_rules = normalization_rules

    def with_streaming(self, vad: object) -> StreamAdapter:
        """Create a VAD-segmented streaming STT over the batch API.

        Prefer native ``stream()`` (``stt-stream-1.5``) for low-latency realtime
        STT. This adapter is useful when you want VAD-driven utterance cuts with
        the batch model (``v2.0``).

        Args:
            vad: A VAD instance (e.g. ``silero.VAD.load()``).

        Returns:
            A ``StreamAdapter`` with ``streaming=True`` capability.
        """
        from livekit.agents.vad import VAD

        if not isinstance(vad, VAD):
            raise TypeError(f"Expected a VAD instance, got {type(vad).__name__}")
        return StreamAdapter(stt=self, vad=vad)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Open a realtime STT WebSocket stream (model: stt-stream-1.5 by default)."""
        return SpeechStream(
            stt=self,
            conn_options=conn_options,
            language=language if isinstance(language, str) else self._language,
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech from audio buffer.

        This method makes a single HTTP request per invocation. Retry logic is
        handled by the base class ``recognize()`` which wraps this method in a
        retry loop and catches ``APIError`` subclasses.

        Empty results are buffered and prepended to the next call so that
        short hesitant speech fragments are not silently dropped.
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        # Merge audio frames from the buffer
        pcm_parts: list[bytes] = []
        sample_rate = self._sample_rate
        num_channels = 1
        frames = [buffer] if not isinstance(buffer, list) else buffer
        for frame in frames:
            pcm_parts.append(bytes(frame.data))
            sample_rate = frame.sample_rate
            num_channels = frame.num_channels
        segment_pcm = b"".join(pcm_parts)

        # Auto-clear stale pending buffer if too much time elapsed
        now = time.monotonic()
        if self._pending_pcm and self._last_recognize_time > 0:
            idle_gap = now - self._last_recognize_time
            if idle_gap > self._pending_idle_timeout:
                logger.debug(
                    "[%s] Clearing stale pending buffer (%.1fs idle)",
                    request_id,
                    idle_gap,
                )
                self._pending_pcm = b""
                self._pending_empty_count = 0
        self._last_recognize_time = now

        # Prepend any buffered PCM from previous empty segments
        if self._pending_pcm:
            logger.info(
                "[%s] Prepending %d bytes pending PCM to %d bytes new segment",
                request_id,
                len(self._pending_pcm),
                len(segment_pcm),
            )
            pcm_data = self._pending_pcm + segment_pcm
        else:
            pcm_data = segment_pcm

        if len(pcm_data) == 0:
            logger.warning("[%s] Empty audio buffer received, skipping", request_id)
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[],
            )

        lang = language if isinstance(language, str) else self._language
        logger.info(
            "[%s] STT request: language=%s, audio=%d bytes",
            request_id,
            lang,
            len(pcm_data),
        )

        wav_data = convert_pcm_to_wav(
            pcm_data,
            sample_rate=sample_rate,
            channels=num_channels,
            bits_per_sample=16,
        )

        params = {
            "language": lang,
            "enable_segments": "false",
            "enable_refinement": "false",
            "model": self._model,
        }
        headers: dict[str, str] = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        files = {
            "audio_file": ("audio.wav", io.BytesIO(wav_data), "audio/wav"),
        }

        try:
            response = await self._client.post(
                self._transcribe_url,
                files=files,
                params=params,
                headers=headers,
                timeout=effective_connect_timeout(conn_options, self._timeout),
            )
        except httpx.TimeoutException as e:
            raise APITimeoutError(f"STT request timed out: {e}") from e
        except httpx.NetworkError as e:
            raise APIConnectionError(f"STT network error: {e}") from e
        except Exception as e:
            raise APIConnectionError(f"STT connection error: {e}") from e

        if response.status_code != 200:
            error_text = response.text
            raise APIStatusError(
                f"STT service error {response.status_code}: {error_text}",
                status_code=response.status_code,
                request_id=request_id,
                body=error_text,
            )

        result = response.json()
        raw_text = result.get("transcription", "")
        text = apply_normalization_rules(raw_text, self._normalization_rules)
        # Explicit null confidence must not become None (breaks %.3f logging).
        confidence = float(result.get("confidence") or 1.0)
        latency = time.monotonic() - start_time

        # --- Frame accumulation logic ---
        # Compute duration of the current segment (not including pending)
        bytes_per_sample = 2 * num_channels  # 16-bit PCM
        segment_duration = (
            len(segment_pcm) / (sample_rate * bytes_per_sample)
            if sample_rate and bytes_per_sample
            else 0.0
        )
        pending_duration = (
            len(self._pending_pcm) / (sample_rate * bytes_per_sample) if self._pending_pcm else 0.0
        )

        if not text.strip():
            # Empty result — decide whether to buffer or discard
            self._pending_empty_count += 1
            total_pending_duration = pending_duration + segment_duration

            if (
                self._pending_empty_count <= self._max_pending_segments
                and total_pending_duration <= self._max_pending_duration
            ):
                # Buffer this segment's PCM for the next call
                self._pending_pcm = pcm_data  # includes already-prepended pending
                self._pending_sample_rate = sample_rate
                self._pending_num_channels = num_channels
                logger.info(
                    "[%s] STT empty → buffered (count=%d, duration=%.1fs, latency=%.3fs)",
                    request_id,
                    self._pending_empty_count,
                    total_pending_duration,
                    latency,
                )
            else:
                # Safety limit reached — discard buffer
                logger.info(
                    "[%s] STT empty → discarded pending buffer "
                    "(count=%d, duration=%.1fs, latency=%.3fs)",
                    request_id,
                    self._pending_empty_count,
                    total_pending_duration,
                    latency,
                )
                self._pending_pcm = b""
                self._pending_empty_count = 0

            # Return empty so StreamAdapter skips this segment as usual
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[
                    stt.SpeechData(
                        text="",
                        language=LanguageCode(lang),
                        confidence=0.0,
                    )
                ],
            )

        # Got real text — clear pending buffer
        had_pending = self._pending_empty_count > 0
        self._pending_pcm = b""
        self._pending_empty_count = 0

        logger.info(
            "[%s] STT completed: text='%s', confidence=%.3f, latency=%.3fs%s",
            request_id,
            text[:80],
            confidence,
            latency,
            f" (included {pending_duration:.1f}s pending audio)" if had_pending else "",
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=request_id,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=LanguageCode(lang),
                    confidence=confidence,
                )
            ],
        )


class SpeechStream(stt.SpeechStream):
    """Realtime STT over Blaze WebSocket ``/v1/stt/realtime`` (stt-stream-1.5).

    Protocol:
      1. Connect WS, send ``{token, language, model}``
      2. Wait for ``{type: "ready"}`` (or connection-ready messages)
      3. Stream binary PCM (s16le mono, typically 16 kHz)
      4. Receive ``{type: "partial"|"final"|"error", text: "..."}``
    """

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        language: str = "vi",
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._sample_rate)
        self._blaze_stt = stt
        self._language = language
        self._request_id = shortuuid()

    async def _run(self) -> None:
        stt_cfg = self._blaze_stt
        if not stt_cfg._auth_token:
            raise APIConnectionError("Blaze STT streaming requires an auth token (BLAZE_API_TOKEN)")

        timeout = effective_connect_timeout(self._conn_options, stt_cfg._timeout)
        logger.info(
            "[%s] STT WS connecting: url=%s model=%s language=%s",
            self._request_id,
            stt_cfg._ws_url,
            stt_cfg._stream_model,
            self._language,
        )

        try:
            async with websockets.connect(
                stt_cfg._ws_url,
                open_timeout=timeout,
                close_timeout=5,
                max_size=8 * 1024 * 1024,
            ) as ws:
                init_msg = {
                    "token": stt_cfg._auth_token,
                    "language": self._language,
                    "model": stt_cfg._stream_model,
                }
                await ws.send(json.dumps(init_msg))

                # Wait for ready / auth ack before streaming audio.
                ready = False
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and not ready:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=max(0.1, deadline - time.monotonic())
                    )
                    if isinstance(raw, bytes):
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    mtype = msg.get("type")
                    if mtype in (
                        "ready",
                        "successful-connection",
                        "successful-authentication",
                    ):
                        ready = True
                        break
                    if mtype == "error":
                        raise APIConnectionError(
                            f"STT realtime auth error: {msg.get('text') or msg}"
                        )

                if not ready:
                    raise APITimeoutError("STT realtime: timed out waiting for ready")

                send_task = asyncio.create_task(self._send_audio(ws))
                speaking = False
                try:
                    async for raw in ws:
                        if isinstance(raw, (bytes, bytearray)):
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        mtype = msg.get("type")
                        text = msg.get("text") or ""
                        if mtype == "error":
                            raise APIConnectionError(f"STT realtime error: {text or msg}")

                        if mtype not in ("partial", "final", "interim"):
                            continue

                        text = apply_normalization_rules(text, stt_cfg._normalization_rules)
                        if not text.strip() and mtype != "final":
                            continue

                        # AgentSession stt turn-detection requires START_OF_SPEECH to open
                        # the user turn (see audio_recognition.py). Emit on first non-empty
                        # transcript; re-arm after END_OF_SPEECH for multi-utterance streams.
                        if text.strip() and not speaking:
                            speaking = True
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                            )

                        event_type = (
                            stt.SpeechEventType.FINAL_TRANSCRIPT
                            if mtype == "final"
                            else stt.SpeechEventType.INTERIM_TRANSCRIPT
                        )
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=event_type,
                                request_id=self._request_id,
                                alternatives=[
                                    stt.SpeechData(
                                        text=text,
                                        language=LanguageCode(self._language),
                                        confidence=float(msg.get("confidence") or 1.0),
                                    )
                                ],
                            )
                        )
                        if mtype == "final":
                            if speaking:
                                speaking = False
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                                )
                finally:
                    send_task.cancel()
                    try:
                        await send_task
                    except asyncio.CancelledError:
                        pass

        except APIConnectionError:
            raise
        except APITimeoutError:
            raise
        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"STT WebSocket closed: {e}") from e
        except Exception as e:
            raise APIConnectionError(f"STT stream error: {e}") from e

    async def _send_audio(self, ws: websockets.ClientConnection) -> None:
        """Forward audio frames from the LiveKit input channel to the WS."""
        try:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    # Blaze/Soniox finalize on silence; no explicit flush frame.
                    continue
                # AudioFrame → raw PCM bytes
                pcm = bytes(item.data)
                if pcm:
                    await ws.send(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[%s] STT send_audio stopped: %s", self._request_id, e)
            raise
