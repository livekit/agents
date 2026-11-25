# Copyright 2026 LiveKit, Inc.
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

"""Zoom text-to-speech for LiveKit Agents.

``zoom_ai.TTS()`` synthesizes speech with the Zoom AI Services Text-to-Speech API in Fast
mode: each request carries the complete text and the audio streams back in the response,
so playback starts before synthesis finishes. It isn't a streaming TTS, so ``AgentSession``
sends it one sentence at a time.

Built on the public docs: https://developers.zoom.us/docs/ai-services/tts/
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Literal

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
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .stt import USER_AGENT, _check_url, _tracking_id, _zoom_reason

BASE_URL = "https://api.zoom.us/v2/aiservices/tts"
SAMPLE_RATE = 44100  # Zoom TTS returns 16-bit mono PCM at 44.1 kHz only
NUM_CHANNELS = 1
# The longest input one request accepts, in Unicode code points.
MAX_INPUT_CHARS = 4000

# The voices listed in the docs. Any voice ID is accepted, so new voices work as is.
TTSVoices = Literal[
    "alex", "ariel", "ava", "brooke", "hazel", "jenna", "mark", "reid", "scott", "tony"
]  # fmt: skip


def _retry_after(headers: Any, body: str) -> float | None:
    """The wait Zoom asks for before the next request, in seconds, if it gives one.

    ``Retry-After`` comes with a 429 and a 503; a 503's JSON body also carries it as
    ``metadata.retry_after_seconds``.
    """
    value: Any = headers.get("Retry-After")
    if value is None:
        try:
            value = (json.loads(body).get("metadata") or {}).get("retry_after_seconds")
        except (TypeError, ValueError, AttributeError):  # no body, or not a JSON object
            return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):  # absent, or an HTTP date
        return None
    return seconds if seconds > 0 else None


def _split_text(text: str, limit: int = MAX_INPUT_CHARS) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` code points, at whitespace if possible."""
    pieces = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return pieces


@dataclass
class _TTSOptions:
    voice: str
    speed: float | None
    volume: float | None


class TTS(tts.TTS):
    """Zoom text-to-speech, one HTTPS request per sentence with the audio streamed back.

    ::

        session = AgentSession(
            stt=zoom_ai.STT(),
            tts=zoom_ai.TTS(voice="ava"),
            ...
        )
    """

    def __init__(
        self,
        *,
        voice: TTSVoices | str = "jenna",
        speed: float | None = None,
        volume: float | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a Zoom TTS.

        Args:
            voice: Zoom TTS voice ID (default ``"jenna"``). See
                https://developers.zoom.us/docs/ai-services/tts/#language-and-voices
            speed: Speaking rate multiplier, from 0.5 to 2.0. ``None`` uses the voice's
                normal rate.
            volume: Volume multiplier, greater than 0 and up to 10. ``None`` uses the
                voice's normal volume.
            api_key: Zoom AI Services API key (from https://www.zoom.ai/). Defaults to
                ``$ZOOM_AI_API_KEY``.
            base_url: Text-to-Speech API base URL; requests go to ``<base>/synthesize``.
            http_session: Optional aiohttp session; the job's session is used otherwise.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        api_key = api_key if is_given(api_key) else os.environ.get("ZOOM_AI_API_KEY", "")
        if not api_key:
            raise ValueError("Zoom AI API key is required: pass api_key or set ZOOM_AI_API_KEY")
        self._api_key = api_key
        base = base_url.rstrip("/")
        scheme, sep, _ = base.partition("://")
        if not sep or scheme.lower() not in ("http", "https"):
            raise ValueError(f"base_url must be an http(s) URL, got {base_url!r}")
        self._url = _check_url(base, "base_url", data="text") + "/synthesize"
        self._session = http_session
        self._opts = _TTSOptions(voice=voice, speed=speed, volume=volume)
        # Monotonic time before which no request is sent: the Retry-After Zoom last gave.
        self._not_before = 0.0

    @property
    def model(self) -> str:
        """``"tts-fast"``: the API has one model, in Fast mode."""
        return "tts-fast"

    @property
    def provider(self) -> str:
        """The provider name, ``"Zoom"``."""
        return "Zoom"

    @property
    def _http(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def _headers(self, tracking_id: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "x-zm-trackingid": tracking_id,
        }

    def update_options(
        self,
        *,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        volume: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update the voice, speed or volume; it applies from the next sentence.

        Args:
            voice: Zoom TTS voice ID.
            speed: Speaking rate multiplier, from 0.5 to 2.0, or ``None`` for normal.
            volume: Volume multiplier, greater than 0 and up to 10, or ``None`` for normal.
        """
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed
        if is_given(volume):
            self._opts.volume = volume

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Synthesizes one text with Zoom TTS, streaming its audio as it arrives."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    def _payload(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": text,
            "voice": self._opts.voice,
            "response_format": "pcm_44100",
        }
        if self._opts.speed is not None:
            payload["speed"] = self._opts.speed
        if self._opts.volume is not None:
            payload["volume"] = self._opts.volume
        return payload

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        pieces = _split_text(self._input_text) if self._input_text.strip() else []
        # Each request has its own tracking ID; it is the request_id of an error, so a
        # failure can be matched in Zoom's logs.
        tracking_ids = [_tracking_id() for _ in pieces] or [_tracking_id()]
        output_emitter.initialize(
            request_id=tracking_ids[0],
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        for piece, tracking_id in zip(pieces, tracking_ids, strict=True):
            await self._synthesize(piece, tracking_id, output_emitter)
        output_emitter.flush()

    async def _synthesize(
        self, text: str, tracking_id: str, output_emitter: tts.AudioEmitter
    ) -> None:
        # Non-ASCII text stays UTF-8 rather than \u escapes, which keeps a request of the
        # longest input within the API's 32 KiB body limit.
        body = json.dumps(self._payload(text), ensure_ascii=False).encode("utf-8")
        # Zoom asks clients to wait out Retry-After before trying again; a request sooner
        # than that is refused, and would use up the retries before the wait is over.
        if (wait := self._tts._not_before - time.monotonic()) > 0:
            if wait > self._conn_options.timeout:
                # longer than a reply can wait: stop the retries (a FallbackAdapter moves on)
                raise APIConnectionError(
                    f"Zoom TTS asked to wait {wait:.1f}s before the next request",
                    retryable=False,
                )
            await asyncio.sleep(wait)
        try:
            # Leaving this block mid-stream, as an interruption does, closes the
            # connection, which stops synthesis on the server.
            async with self._tts._http.post(
                self._tts._url,
                data=body,
                headers=self._tts._headers(tracking_id),
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    # Zoom explains a refusal (a bad API key or voice, say) in the JSON body.
                    reason = _zoom_reason(error)
                    if (retry_after := _retry_after(resp.headers, error)) is not None:
                        self._tts._not_before = max(
                            self._tts._not_before, time.monotonic() + retry_after
                        )
                    raise APIStatusError(
                        f"Zoom TTS error (HTTP {resp.status})" + (f": {reason}" if reason else ""),
                        status_code=resp.status,
                        request_id=tracking_id,
                        body=error,
                    )
                async for chunk in resp.content.iter_any():
                    output_emitter.push(chunk)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientPayloadError as e:
            # Once audio is streaming the status can't change, so a synthesis failure ends
            # the response early instead. A retry synthesizes the text from the beginning.
            raise APIConnectionError("Zoom TTS stream ended early") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError() from e
