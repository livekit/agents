# Copyright 2023 LiveKit, Inc.
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

"""GMI Cloud text-to-speech plugin for LiveKit Agents.

Synthesizes speech through GMI Cloud's multimodal inference engine
(https://www.gmicloud.ai) via its request-queue HTTP API
(``POST /api/v1/ie/requestqueue/apikey/requests``). Every documented
request family is supported behind one adapter:

- **Inworld** (``inworld-tts-2``, ``inworld-tts-2-flash``,
  ``inworld-tts-1.5-mini``, ``inworld-tts-1.5-max``): Inworld Realtime TTS.
- **MiniMax** (``minimax-tts-speech-*``): MiniMax speech synthesis.
- **Chatterbox** (``Chatterbox-tts``): Resemble AI's Chatterbox model,
  which requires a Resemble ``voice_uuid`` (pass it as ``voice``).

Responses are handled in both documented shapes: synchronous (``outcome``
returned inline) and queued (``queued`` / ``dispatched`` followed by polling
``GET /api/v1/ie/requestqueue/apikey/requests/{request_id}``).
"""

from __future__ import annotations

import asyncio
import io
import os
import wave
from dataclasses import dataclass, field, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger

DEFAULT_BASE_URL = "https://console.gmicloud.ai"
QUEUE_PATH = "/api/v1/ie/requestqueue/apikey/requests"

DEFAULT_MODEL = "inworld-tts-2"

# Per-family request defaults.
FAMILY_DEFAULTS: dict[str, dict[str, str | None]] = {
    "inworld": {"voice": "Dennis"},
    "minimax": {"voice": "English_expressive_narrator"},
    "chatterbox": {"voice": None},
}


def model_family(model: str) -> str:
    """Map a GMI Cloud model id to its request family.

    Returns one of ``"inworld"``, ``"minimax"`` or ``"chatterbox"``.
    Unknown ids default to the Inworld request shape.
    """
    m = (model or "").lower()
    if m.startswith("minimax"):
        return "minimax"
    if m.startswith("chatterbox"):
        return "chatterbox"
    return "inworld"


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    model: str
    voice: str | None
    audio_format: str
    speed: float
    temperature: float | None
    timestamp_type: str | None
    extra_payload: dict[str, Any] = field(default_factory=dict)
    poll_interval_secs: float = 2.0
    poll_attempts: int = 90


def build_payload(text: str, opts: _TTSOptions) -> dict[str, Any]:
    """Build the GMI Cloud request payload for the model's request family.

    Raises:
        ValueError: Chatterbox models without a Resemble ``voice_uuid``.
    """
    family = model_family(opts.model)
    defaults = FAMILY_DEFAULTS[family]
    voice = opts.voice if opts.voice is not None else defaults["voice"]
    encoding = (opts.audio_format or "WAV").lower()
    speed = opts.speed

    if family == "inworld":
        payload: dict[str, Any] = {
            "text": text,
            "audio_encoding": encoding.upper(),
            "speaking_rate": speed,
        }
        if voice:
            payload["voice_id"] = voice
        if opts.temperature is not None:
            payload["temperature"] = opts.temperature
        if opts.timestamp_type is not None:
            payload["timestamp_type"] = opts.timestamp_type
    elif family == "minimax":
        payload = {
            "text": text,
            "speed": str(speed),
            "format": "wav" if encoding == "wav" else encoding,
        }
        if voice:
            payload["voice_id"] = voice
    else:  # chatterbox
        if not voice:
            raise ValueError(
                "Chatterbox models require a Resemble voice UUID (voice_uuid): "
                "pass voice='<your-voice-uuid>' to livekit.plugins.gmicloud.TTS "
                "(or use a different model)."
            )
        payload = {
            "data": text,
            "voice_uuid": voice,
            "output_format": "wav" if encoding == "wav" else encoding,
        }

    if opts.extra_payload:
        payload.update(opts.extra_payload)
    return payload


def _extract_audio_url(data: dict[str, Any]) -> str | None:
    """Extract the first audio URL from a queue response outcome."""
    outcome = data.get("outcome") or {}
    if not isinstance(outcome, dict):
        return None
    audio_url = outcome.get("audio_url")
    if isinstance(audio_url, str) and audio_url:
        return audio_url
    for key in ("media_urls", "medias", "media"):
        for item in outcome.get(key) or []:
            if isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str) and url:
                    return url
    return None


def _wav_info(data: bytes) -> tuple[int, int]:
    """Return (sample_rate, num_channels) from a WAV header, or (0, 0)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            return wav.getframerate(), wav.getnchannels()
    except Exception:
        return 0, 0


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str | None = None,
        speed: float = 1.0,
        temperature: float | None = None,
        timestamp_type: str | None = None,
        extra_payload: dict[str, Any] | None = None,
        audio_format: str = "WAV",
        sample_rate: int = 48000,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        poll_interval_secs: float = 2.0,
        poll_attempts: int = 90,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """GMI Cloud TTS plugin.

        Synthesizes speech through GMI Cloud's request-queue HTTP API,
        automatically using the correct request shape for the model's family
        (Inworld, MiniMax or Chatterbox) and resolving both synchronous and
        queued/dispatched responses to WAV audio.

        Args:
            model: GMI Cloud model id (e.g. ``inworld-tts-2``,
                ``minimax-tts-speech-2.8-turbo``, ``Chatterbox-tts``).
            voice: Voice for the model's family. Inworld/MiniMax: a voice
                name/id (defaults to a family default). Chatterbox: a
                Resemble AI ``voice_uuid`` (required).
            speed: Speech speed multiplier (Inworld ``speaking_rate``,
                MiniMax ``speed``). Defaults to 1.0.
            temperature: Synthesis randomness (Inworld family only).
            timestamp_type: Alignment metadata request, e.g. ``"WORD"``
                (Inworld family only).
            extra_payload: Provider-specific payload fields merged into the
                request unchanged.
            audio_format: Output container requested from GMI Cloud.
                Defaults to ``"WAV"``.
            sample_rate: Declared sample rate of this TTS instance. The
                emitted audio uses the rate of the WAV returned by the model
                (read from its header), so this is only a nominal default.
            api_key: GMI Cloud API key. Falls back to the
                ``GMI_CLOUD_API_KEY`` or ``GMI_API_KEY`` environment
                variables.
            base_url: GMI Cloud console API base URL.
            poll_interval_secs: Seconds between status polls for queued
                generations. Defaults to 2.0.
            poll_attempts: Maximum status polls before giving up.
                Defaults to 90 (about 3 minutes at the default interval;
                queued MiniMax generations can stay ``processing`` for
                minutes).
            http_session: An existing aiohttp ClientSession to use. If not
                provided, the LiveKit Agents job-context session is used;
                outside an agent worker (standalone scripts) you must pass
                your own session or synthesis raises an error.

        Raises:
            ValueError: If no API key is available.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("GMI_CLOUD_API_KEY") or os.environ.get("GMI_API_KEY")
        if not api_key:
            raise ValueError(
                "GMI Cloud API key is required, either as argument or set"
                " GMI_CLOUD_API_KEY (or GMI_API_KEY) environment variable"
            )

        self._opts = _TTSOptions(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            voice=voice,
            audio_format=audio_format,
            speed=speed,
            temperature=temperature,
            timestamp_type=timestamp_type,
            extra_payload=extra_payload or {},
            poll_interval_secs=poll_interval_secs,
            poll_attempts=poll_attempts,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "GMI Cloud"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            try:
                self._session = utils.http_context.http_session()
            except RuntimeError as e:
                raise APIConnectionError(
                    "No aiohttp session available. Inside an AgentSession the "
                    "LiveKit Agents job-context session is used automatically; "
                    "outside an agent worker, pass http_session= to "
                    "livekit.plugins.gmicloud.TTS."
                ) from e
        return self._session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _resolve_result(
        self, session: aiohttp.ClientSession, data: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        """Resolve a response that may be synchronous or queued.

        Returns the final response dict and an audio URL (or None).
        """
        status = data.get("status")
        request_id = data.get("request_id")
        audio_url = _extract_audio_url(data)

        if not audio_url and status != "failed" and request_id:
            for _ in range(self._opts.poll_attempts):
                await asyncio.sleep(self._opts.poll_interval_secs)
                async with session.get(
                    f"{self._opts.base_url}{QUEUE_PATH}/{request_id}",
                    headers={"Authorization": f"Bearer {self._opts.api_key}"},
                ) as response:
                    if response.status != 200:
                        continue
                    data = await response.json(content_type=None)
                status = data.get("status")
                audio_url = _extract_audio_url(data)
                if status in ("success", "failed") or audio_url:
                    break

        return data, audio_url

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if not self._input_text.strip():
            # Nothing to synthesize: complete the stream as a clean zero-audio
            # segment. The emitter must still be initialized (the base class
            # ends it after _run); event consumers see a no-op.
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=self._tts.sample_rate,
                num_channels=1,
                mime_type="audio/wav",
            )
            return

        # may raise ValueError (e.g. Chatterbox without a voice_uuid) — a
        # clear configuration error, deliberately not an APIError so the
        # retry loop doesn't re-attempt it
        body = {"model": self._opts.model, "payload": build_payload(self._input_text, self._opts)}

        url = f"{self._opts.base_url}{QUEUE_PATH}"
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }
        request_id = utils.shortuuid()
        timeout = aiohttp.ClientTimeout(
            total=max(30.0, self._conn_options.timeout),
            sock_connect=self._conn_options.timeout,
        )

        try:
            session = self._tts._ensure_session()

            async with session.post(url, json=body, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise APIStatusError(
                        f"GMI Cloud API error: HTTP {resp.status} - {error_text}",
                        status_code=resp.status,
                        request_id=request_id,
                        body=error_text,
                    )
                data = await resp.json(content_type=None)

            data, audio_url = await self._resolve_result(session, data)
            if data.get("status") != "success" or not audio_url:
                raise APIError(
                    f"GMI Cloud TTS failed: status={data.get('status')!r}, "
                    f"request_id={data.get('request_id')!r}",
                    retryable=False,
                )

            async with session.get(audio_url) as audio_resp:
                if audio_resp.status != 200:
                    error_text = await audio_resp.text()
                    raise APIStatusError(
                        f"GMI Cloud audio download error: HTTP {audio_resp.status} - {error_text}",
                        status_code=audio_resp.status,
                        request_id=request_id,
                        body=error_text,
                    )
                wav_bytes = await audio_resp.read()

        except APIError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError(
                f"GMI Cloud TTS request timed out after {timeout.total}s"
            ) from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                f"GMI Cloud HTTP error: {e.status}: {e.message}",
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(
                f"GMI Cloud TTS connection failed: {type(e).__name__}: {e}"
            ) from e
        except Exception as e:
            raise APIConnectionError(
                f"GMI Cloud TTS unexpected error: {type(e).__name__}: {e}"
            ) from e

        sample_rate, num_channels = _wav_info(wav_bytes)
        if not sample_rate:
            logger.warning(
                "GMI Cloud TTS: could not read WAV header, falling back to declared rate",
                extra={"tts": self._tts._label},
            )
            sample_rate, num_channels = self._tts.sample_rate, 1

        output_emitter.initialize(
            request_id=str(data.get("request_id") or request_id),
            sample_rate=sample_rate,
            num_channels=num_channels,
            mime_type="audio/wav",
        )
        output_emitter.push(wav_bytes)
        output_emitter.flush()
