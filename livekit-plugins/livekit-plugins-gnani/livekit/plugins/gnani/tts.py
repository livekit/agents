"""Text-to-Speech implementation for Gnani Vachana

This module provides a TTS implementation that uses the Gnani Vachana API,
supporting both chunked synthesis (REST) and real-time streaming (WebSocket).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, replace
from typing import Literal

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger

GNANI_TTS_BASE_URL = "https://api.vachana.ai"

GnaniTTSVoices = Literal[
    "sia",
    "raju",
    "kanika",
    "nikita",
    "ravan",
    "simran",
    "karan",
    "neha",
]

SUPPORTED_VOICES: set[str] = {
    "sia",
    "raju",
    "kanika",
    "nikita",
    "ravan",
    "simran",
    "karan",
    "neha",
}

GnaniTTSEncodings = Literal["linear_pcm", "oggopus"]
GnaniTTSContainers = Literal["raw", "mp3", "wav", "mulaw", "ogg"]


SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 44100)


@dataclass
class GnaniTTSOptions:
    api_key: str
    voice: str = "sia"
    model: str = "vachana-voice-v2"
    sample_rate: int = 16000
    encoding: str = "linear_pcm"
    container: str = "wav"
    num_channels: int = 1
    sample_width: int = 2
    base_url: str = GNANI_TTS_BASE_URL
    language: str = "IND-IN"


class TTS(tts.TTS):
    """Gnani Vachana Text-to-Speech implementation.

    Provides text-to-speech functionality using Gnani's Vachana platform.
    Supports batch synthesis via REST API and real-time streaming via WebSocket.

    Args:
        voice: Voice to use for synthesis (sia, raju, kanika, etc.).
        model: TTS model name (default: vachana-voice-v2).
        sample_rate: Audio output sample rate (8000-44100).
        encoding: Audio encoding (linear_pcm or oggopus).
        container: Audio container format (raw, mp3, wav, mulaw, ogg).
        api_key: Gnani API key (falls back to GNANI_API_KEY env var).
        base_url: Vachana API base URL.
        language: Language code for WebSocket TTS (default: IND-IN).
    """

    def __init__(
        self,
        *,
        voice: GnaniTTSVoices | str = "sia",
        model: str = "vachana-voice-v2",
        sample_rate: int = 16000,
        num_channels: int = 1,
        encoding: GnaniTTSEncodings | str = "linear_pcm",
        container: GnaniTTSContainers | str = "wav",
        api_key: str | None = None,
        base_url: str = GNANI_TTS_BASE_URL,
        language: str = "IND-IN",
    ) -> None:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {sample_rate}"
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._api_key = api_key or os.environ.get("GNANI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gnani API key is required. "
                "Provide it directly or set GNANI_API_KEY environment variable."
            )

        if voice not in SUPPORTED_VOICES:
            raise ValueError(
                f"Voice '{voice}' not supported. Choose from: {sorted(SUPPORTED_VOICES)}"
            )

        self._opts = GnaniTTSOptions(
            api_key=self._api_key,
            voice=voice,
            model=model,
            sample_rate=sample_rate,
            encoding=encoding,
            container=container,
            num_channels=num_channels,
            base_url=base_url,
            language=language,
        )
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Gnani"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def update_options(
        self,
        *,
        voice: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ) -> None:
        if voice is not None:
            if voice not in SUPPORTED_VOICES:
                raise ValueError(
                    f"Voice '{voice}' not supported. Choose from: {sorted(SUPPORTED_VOICES)}"
                )
            self._opts.voice = voice
        if model is not None:
            self._opts.model = model
        if language is not None:
            self._opts.language = language

    async def aclose(self) -> None:
        pass


class ChunkedStream(tts.ChunkedStream):
    """REST-based chunked TTS for Gnani Vachana.

    Uses POST /api/v1/tts/inference to synthesize text in a single request.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload = {
            "text": self._input_text,
            "voice": self._opts.voice,
            "model": self._opts.model,
            "audio_config": {
                "sample_rate": self._opts.sample_rate,
                "encoding": self._opts.encoding,
                "num_channels": self._opts.num_channels,
                "sample_width": self._opts.sample_width,
                "container": self._opts.container,
            },
        }

        headers = {
            "X-API-Key-ID": self._opts.api_key,
            "Content-Type": "application/json",
        }

        mime_type = f"audio/{self._opts.container}"
        if self._opts.container == "raw":
            mime_type = "audio/pcm"

        try:
            async with self._tts._ensure_session().post(
                url=f"{self._opts.base_url}/api/v1/tts/inference",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    logger.error(f"Gnani TTS API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Gnani TTS API Error ({res.status}): {error_text}",
                        status_code=res.status,
                        body=error_text,
                    )

                audio_bytes = await res.read()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type=mime_type,
                )
                output_emitter.push(audio_bytes)

        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani TTS API request timed out") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani TTS error: {e}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """WebSocket-based streaming TTS for Gnani Vachana.

    Opens a WebSocket to wss://api.vachana.ai/api/v1/tts and streams
    audio chunks back as they are synthesized.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _build_ws_url(self) -> str:
        base = self._opts.base_url
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://") :]
        else:
            ws_base = "wss://" + base
        return f"{ws_base}/api/v1/tts"

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import websockets

        mime_type = f"audio/{self._opts.container}"
        if self._opts.container == "raw":
            mime_type = "audio/pcm"

        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type=mime_type,
            stream=True,
        )

        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        text_parts: list[str] = []

        async for data in self._input_ch:
            if isinstance(data, str):
                text_parts.append(data)
            elif isinstance(data, self._FlushSentinel):
                break

        full_text = "".join(text_parts).strip()
        if not full_text:
            output_emitter.end_segment()
            return

        self._mark_started()

        ws_url = self._build_ws_url()
        headers = {
            "Content-Type": "application/json",
            "X-API-Key-ID": self._opts.api_key,
        }

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as ws:
                request_body = {
                    "text": full_text,
                    "voice": self._opts.voice,
                    "model": self._opts.model,
                    "language": self._opts.language,
                    "audio_config": {
                        "sample_rate": self._opts.sample_rate,
                        "encoding": self._opts.encoding,
                        "num_channels": self._opts.num_channels,
                        "sample_width": self._opts.sample_width,
                        "container": self._opts.container,
                    },
                }
                await ws.send(json.dumps(request_body))

                async for msg in ws:
                    if isinstance(msg, bytes):
                        output_emitter.push(msg)
                        continue

                    payload = json.loads(msg)
                    msg_type = payload.get("type", "")

                    if msg_type == "audio":
                        inner = payload.get("data", {})
                        audio_b64 = inner.get("audio", "")
                        if audio_b64:
                            output_emitter.push(base64.b64decode(audio_b64))

                    elif msg_type == "complete":
                        inner = payload.get("data")
                        if inner is not None:
                            audio_b64 = inner.get("audio", "")
                            if audio_b64:
                                output_emitter.push(base64.b64decode(audio_b64))
                        break

                    elif msg_type == "error":
                        error_msg = payload.get("message", "Unknown error")
                        logger.error(f"Gnani TTS stream error: {error_msg}")
                        raise APIStatusError(
                            message=f"Gnani TTS stream error: {error_msg}",
                            status_code=500,
                            body=error_msg,
                        )

        except websockets.exceptions.ConnectionClosed as e:
            raise APIConnectionError(f"Gnani TTS WebSocket closed: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani TTS WebSocket timed out") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani TTS WebSocket error: {e}") from e
        finally:
            output_emitter.end_segment()
