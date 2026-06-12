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

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    create_api_error_from_http,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .models import TTSEncoding, TTSModels
from .version import __version__

NUM_CHANNELS = 1
SMALLEST_BASE_URL = "https://api.smallest.ai/waves/v1"
SMALLEST_WS_URL = "wss://api.smallest.ai/waves/v1/tts/live"


@dataclass
class _TTSOptions:
    model: TTSModels | str
    api_key: str
    voice_id: str
    sample_rate: int
    speed: float
    language: LanguageCode
    output_format: TTSEncoding | str
    word_timestamps: bool
    base_url: str
    ws_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = "lightning_v3.1_pro",
        voice_id: str | None = None,
        sample_rate: int = 24000,
        speed: float = 1.0,
        language: str = "en",
        output_format: TTSEncoding | str = "pcm",
        word_timestamps: bool = False,
        base_url: str = SMALLEST_BASE_URL,
        ws_url: str = SMALLEST_WS_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Smallest AI Lightning TTS.

        Args:
            api_key: Your Smallest AI API key.
            model: The TTS model to use. Use "lightning_v3.1" for the standard model with
                217 voices across 12 languages, or "lightning_v3.1_pro" (default) for the
                premium pool with curated American, British, and Indian voices at 44.1 kHz.
            voice_id: The voice ID to use for synthesis. Defaults to "meher" for
                "lightning_v3.1_pro" and "sophia" for all other models. Pro voices must be
                paired with "lightning_v3.1_pro"; standard voices with "lightning_v3.1".
            sample_rate: Sample rate for the audio output. Both models are natively 44.1 kHz;
                supported rates are 8000, 16000, 24000, and 44100.
            speed: Speed of the speech synthesis (0.5–2.0).
            language: Language of the text to be synthesized. Use "auto" for automatic
                detection and code-switching. Pro supports "en", "hi", and "auto" only.
            output_format: Output format for HTTP synthesize() calls ("pcm", "mp3", "wav",
                "ulaw", "alaw"). WebSocket streaming always returns PCM.
            word_timestamps: Request per-word timing events from the server and emit them
                as timed transcript entries alongside audio. Applies to WebSocket streaming
                only; HTTP synthesize() returns raw audio without word events. Disabled by
                default. Supported on base-queue English + Hindi voices (meher, devansh,
                kartik, maithili, liam, avery); other voices silently emit no word events.
            base_url: Base URL for the Smallest AI HTTP API.
            ws_url: WebSocket URL for low-latency streaming synthesis.
            http_session: An existing aiohttp ClientSession to use.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError(
                "Smallest.ai API key is required, either as argument or set"
                " SMALLEST_API_KEY environment variable"
            )

        if voice_id is None:
            voice_id = "meher" if model == "lightning_v3.1_pro" else "sophia"

        self._opts = _TTSOptions(
            model=model,
            api_key=api_key,
            voice_id=voice_id,
            sample_rate=sample_rate,
            speed=speed,
            language=LanguageCode(language),
            output_format=output_format,
            word_timestamps=word_timestamps,
            base_url=base_url,
            ws_url=ws_url,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,
            mark_refreshed_on_get=False,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "SmallestAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._opts.ws_url,
                headers={
                    "Authorization": f"Bearer {self._opts.api_key}",
                    "X-Source": "livekit",
                    "X-LiveKit-Version": __version__,
                },
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        output_format: NotGivenOr[TTSEncoding | str] = NOT_GIVEN,
        word_timestamps: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update TTS options."""
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(language):
            self._opts.language = LanguageCode(language)
        if is_given(output_format):
            self._opts.output_format = output_format
        if is_given(word_timestamps):
            self._opts.word_timestamps = word_timestamps
            self._capabilities.aligned_transcript = word_timestamps

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """HTTP-based synthesis — used when synthesize() is called directly."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            data = _to_smallest_options(self._opts)
            data["text"] = self._input_text

            headers = {
                "Authorization": f"Bearer {self._opts.api_key}",
                "Content-Type": "application/json",
                "X-Source": "livekit",
                "X-LiveKit-Version": __version__,
            }
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/tts",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise create_api_error_from_http(body, status=resp.status)

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=f"audio/{self._opts.output_format}",
                )

                async for chunk, _ in resp.content.iter_chunks():
                    output_emitter.push(chunk)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise create_api_error_from_http(e.message, status=e.status) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """WebSocket-based streaming synthesis — primary path used by the agent pipeline."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        try:
            text_buffer = ""
            async for data in self._input_ch:
                if isinstance(data, str):
                    text_buffer += data
                elif isinstance(data, self._FlushSentinel):
                    if text_buffer.strip():
                        await self._run_ws(text_buffer.strip(), output_emitter)
                    text_buffer = ""
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_ws(self, text: str, output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        payload: dict[str, Any] = {
            "model": self._opts.model,
            "voice_id": self._opts.voice_id,
            "text": text,
            "sample_rate": self._opts.sample_rate,
            "speed": self._opts.speed,
            "language": self._opts.language.language
            if isinstance(self._opts.language, LanguageCode)
            else self._opts.language,
        }

        if self._opts.word_timestamps:
            payload["word_timestamps"] = True

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            self._acquire_time = self._tts._pool.last_acquire_time
            self._connection_reused = self._tts._pool.last_connection_reused
            self._mark_started()
            await ws.send_str(json.dumps(payload))

            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)

                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "SmallestAI WebSocket closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=str(msg.data),
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)
                status = event.get("status")

                if status == "chunk":
                    audio_b64 = event.get("data", {}).get("audio")
                    if audio_b64:
                        output_emitter.push(base64.b64decode(audio_b64))
                elif status == "word_timestamp":
                    data = event.get("data", {})
                    word = data.get("word")
                    start = data.get("start")
                    end = data.get("end")
                    if word is not None and start is not None and end is not None:
                        output_emitter.push_timed_transcript(
                            TimedString(text=word, start_time=start, end_time=end)
                        )
                elif status == "complete":
                    output_emitter.end_segment()
                    break
                elif status == "error":
                    raise APIConnectionError(
                        f"SmallestAI TTS error: {event.get('message', 'unknown error')}"
                    )


def _to_smallest_options(opts: _TTSOptions) -> dict[str, Any]:
    return {
        "model": opts.model,
        "voice_id": opts.voice_id,
        "sample_rate": opts.sample_rate,
        "speed": opts.speed,
        "language": opts.language.language
        if isinstance(opts.language, LanguageCode)
        else opts.language,
        "output_format": opts.output_format,
    }
