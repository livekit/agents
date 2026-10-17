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

"""Text-to-speech for Avaz.

Each synthesis stream opens its own WebSocket to ``/tts/stream-input``: the plugin
sends the voice settings, streams one sentence per line, then asks Avaz to flush.
Avaz answers with a base64 WAV per sentence and a final ``closed`` status, then
closes the socket.
"""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import json
import os
import time
import wave
import weakref
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlparse, urlunparse

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import DEFAULT_BASE_URL, DEFAULT_MODEL, MODEL_SAMPLE_RATES, TTSModels
from .version import __version__

NUM_CHANNELS = 1
SAMPLE_WIDTH = 2
USER_AGENT = f"livekit-plugins-avaz/{__version__}"
STREAM_PATH = "/tts/stream-input"
# Avaz cuts its text buffer at `chunk_notation`. Sending one sentence per line keeps
# abbreviations and numbers such as "1.500" inside their sentence.
CHUNK_NOTATION = "\n"
# Each sentence comes back as a single base64 WAV message, larger than aiohttp's 4 MiB
# default for long sentences.
MAX_MESSAGE_SIZE = 16 * 1024 * 1024
# Avaz can be busy synthesizing when a stream is interrupted, so don't wait long for
# its reply to the close frame.
WS_CLOSE_TIMEOUT = 1.0

_WS_SCHEMES = {"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"}


@dataclass
class _TTSOptions:
    model: TTSModels
    speaker_id: int | None
    api_key: str | None
    ws_url: str

    @property
    def sample_rate(self) -> int:
        return MODEL_SAMPLE_RATES[self.model]


class TTS(tts.TTS):
    """Turkish text-to-speech with Avaz, streamed one sentence at a time."""

    def __init__(
        self,
        *,
        model: TTSModels = DEFAULT_MODEL,
        speaker_id: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Avaz TTS.

        Args:
            model: The Avaz model to use. ``"avaz3"`` returns 48 kHz audio and ``"avaz2"``
                returns 24 kHz audio. Defaults to ``"avaz3"``.
            speaker_id: The voice to use, as listed by the Avaz ``/tts/voices`` endpoint.
                Defaults to the model's default voice.
            api_key: The Avaz API key. Defaults to the ``AVAZ_API_KEY`` environment
                variable, unless ``base_url`` points to a different endpoint than the hosted
                API. A key is only accepted with an ``https``/``wss`` or loopback ``base_url``.
            base_url: The Avaz API base URL, for example of an on-premises deployment.
                Defaults to the ``AVAZ_BASE_URL`` environment variable, or to the hosted API.
            tokenizer: The sentence tokenizer used to split streamed text. Defaults to
                ``livekit.agents.tokenize.blingfire.SentenceTokenizer``.
            http_session: The aiohttp session to use. Defaults to the job's shared session.
        """
        _validate_model(model)
        if is_given(speaker_id):
            _validate_speaker_id(speaker_id)

        resolved_base_url = (
            base_url if is_given(base_url) else os.environ.get("AVAZ_BASE_URL") or DEFAULT_BASE_URL
        )
        ws_url = _to_ws_url(resolved_base_url)
        is_hosted = ws_url == _to_ws_url(DEFAULT_BASE_URL)
        resolved_api_key: str | None = None
        if is_given(api_key):
            resolved_api_key = api_key
        elif is_hosted or not is_given(base_url):
            # the key from the environment only goes to the hosted API or AVAZ_BASE_URL
            resolved_api_key = os.environ.get("AVAZ_API_KEY")

        if not resolved_api_key and is_hosted:
            raise ValueError(
                "Avaz API key is required, either as argument or set AVAZ_API_KEY"
                " environment variable"
            )
        if resolved_api_key and not _is_secure_or_loopback(ws_url):
            raise ValueError(
                "Avaz API key is only sent over https:// or wss://, use a secure base_url or"
                ' pass api_key="" if the deployment does not require a key'
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=MODEL_SAMPLE_RATES[model],
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            model=model,
            speaker_id=speaker_id if is_given(speaker_id) else None,
            api_key=resolved_api_key or None,
            ws_url=ws_url,
        )
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        """The Avaz model used for synthesis."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """The provider name, ``"Avaz"``."""
        return "Avaz"

    def update_options(self, *, speaker_id: NotGivenOr[int | None] = NOT_GIVEN) -> None:
        """Update the voice used by streams created after this call.

        Args:
            speaker_id: The voice to use, as listed by the Avaz ``/tts/voices`` endpoint, or
                ``None`` for the model's default voice.
        """
        if is_given(speaker_id):
            if speaker_id is not None:
                _validate_speaker_id(speaker_id)
            self._opts.speaker_id = speaker_id

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        """Synthesize a complete text through a single stream, created when synthesis starts.

        Args:
            text: The text to speak.
            conn_options: The connection and retry options.
        """
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a stream that speaks text as it is pushed.

        Args:
            conn_options: The connection and retry options.
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all streams created by this instance."""
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()

    def _ensure_session(self) -> aiohttp.ClientSession:
        return self._session or utils.http_context.http_session()


class SynthesizeStream(tts.SynthesizeStream):
    """A stream of Avaz speech for one utterance, using one WebSocket connection."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        """Create a stream with a copy of the TTS options.

        Args:
            tts: The TTS instance that owns this stream.
            conn_options: The connection and retry options.
        """
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        sentence_stream = self._tts._sentence_tokenizer.stream()
        has_text = asyncio.Event()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sentence_stream.flush()
                    continue

                if data.strip():
                    has_text.set()
                sentence_stream.push_text(data)

            sentence_stream.end_input()

        input_task = asyncio.create_task(_input_task())
        has_text_task = asyncio.create_task(has_text.wait())
        try:
            # connect once there is something to say, so a turn without text needs no socket
            await asyncio.wait([input_task, has_text_task], return_when=asyncio.FIRST_COMPLETED)
            if not has_text.is_set():
                await input_task
                return

            await self._run_ws(sentence_stream, input_task, output_emitter, request_id)
        finally:
            await utils.aio.gracefully_cancel(input_task, has_text_task)
            await sentence_stream.aclose()

    async def _run_ws(
        self,
        sentence_stream: tokenize.SentenceStream,
        input_task: asyncio.Task[None],
        output_emitter: tts.AudioEmitter,
        request_id: str,
    ) -> None:
        ws = await self._connect()
        sent = received = 0
        input_done = flush_sent = False
        # set while Avaz owes audio, so a slow LLM between sentences can't time out
        audio_owed = asyncio.Event()

        async def _send_task() -> None:
            nonlocal sent, input_done, flush_sent
            try:
                async for ev in sentence_stream:
                    # a newline inside a sentence would split it on the Avaz side
                    text = " ".join(ev.token.split())
                    if not text:
                        continue

                    if sent == 0:
                        output_emitter.start_segment(segment_id=request_id)
                        self._mark_started()

                    sent += 1
                    audio_owed.set()
                    await ws.send_str(json.dumps({"text": text + CHUNK_NOTATION}))

                if sent:
                    flush_sent = True
                    await ws.send_str(json.dumps({"flush": True, "force_flush": True}))
            finally:
                input_done = True
                audio_owed.set()

        async def _recv_task() -> None:
            nonlocal received
            while True:
                await audio_owed.wait()
                if input_done and sent == 0:
                    return

                data = await _receive_json(ws, timeout=self._conn_options.timeout)
                if "error" in data:
                    logger.warning(
                        "Avaz reported an error during synthesis",
                        extra={"request_id": request_id, "lk.pii.error": data["error"]},
                    )
                    raise APIError("Avaz reported an error during synthesis")

                if "audio" in data:
                    output_emitter.push(_decode_audio(data["audio"], self._opts.sample_rate))
                    received += 1
                    if received >= sent and not input_done:
                        audio_owed.clear()
                elif data.get("status") == "closed":
                    if not flush_sent:
                        raise APIConnectionError("Avaz ended the stream before all text was sent")
                    break

            # Avaz sends nothing for a sentence it fails to synthesize
            if received < sent:
                raise APIError(f"Avaz returned audio for {received} of {sent} sentences")

        tasks = [asyncio.create_task(_send_task()), asyncio.create_task(_recv_task())]
        try:
            await asyncio.gather(*tasks, input_task)
        except asyncio.TimeoutError:
            raise APITimeoutError("Avaz stopped sending audio") from None
        except APIError:
            raise
        except (aiohttp.ClientError, OSError) as e:
            raise APIConnectionError("Avaz connection error") from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await ws.close()

        if sent:
            output_emitter.end_segment()

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        """Open a WebSocket and send the voice settings, both within the connect timeout."""
        headers = {"User-Agent": USER_AGENT}
        if self._opts.api_key:
            # aiohttp drops Authorization, unlike custom headers, on cross-origin redirects
            headers["Authorization"] = f"Bearer {self._opts.api_key}"

        started_at = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._tts._ensure_session().ws_connect(
                    self._opts.ws_url,
                    headers=headers,
                    max_msg_size=MAX_MESSAGE_SIZE,
                    timeout=aiohttp.ClientWSTimeout(ws_close=WS_CLOSE_TIMEOUT),
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Avaz connection timed out") from None
        except aiohttp.WSServerHandshakeError as e:
            # the error keeps the request headers, including the API key
            if e.status == 499:  # 499 would end the stream silently
                raise APIError("Avaz rejected the connection", retryable=False) from None
            raise APIStatusError("Avaz rejected the connection", status_code=e.status) from None
        except (aiohttp.ClientError, OSError) as e:
            raise APIConnectionError(f"Avaz connection error ({type(e).__name__})") from None

        try:
            remaining = self._conn_options.timeout - (time.perf_counter() - started_at)
            await asyncio.wait_for(self._send_voice_settings(ws), max(remaining, 0.0))
        except BaseException as e:
            await ws.close()
            if isinstance(e, asyncio.TimeoutError):
                raise APITimeoutError("Avaz did not accept the voice settings in time") from None
            if isinstance(e, (aiohttp.ClientError, OSError)):
                raise APIConnectionError("Avaz connection error") from e
            raise

        self._acquire_time = time.perf_counter() - started_at
        return ws

    async def _send_voice_settings(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        model_settings: dict[str, Any] = {"model_id": self._opts.model}
        if self._opts.speaker_id is not None:
            model_settings["speaker_id"] = self._opts.speaker_id

        await ws.send_str(
            json.dumps(
                {
                    "model_settings": model_settings,
                    "voice_settings": {"chunk_notation": CHUNK_NOTATION},
                }
            )
        )
        data = await _receive_json(ws, timeout=None)
        if "error" in data:
            logger.warning(
                "Avaz rejected the voice settings",
                extra={
                    "model": self._opts.model,
                    "speaker_id": self._opts.speaker_id,
                    "lk.pii.error": data["error"],
                },
            )
            # an unknown model or voice, retrying will not help
            raise APIError(
                f"Avaz rejected the voice settings (model={self._opts.model},"
                f" speaker_id={self._opts.speaker_id})",
                retryable=False,
            )
        if data.get("status") != "initialized":
            raise APIError("Avaz sent an unexpected response to the voice settings")


async def _receive_json(
    ws: aiohttp.ClientWebSocketResponse, *, timeout: float | None
) -> dict[str, Any]:
    # ws.receive(timeout=...) restarts its timer on every ping from the server
    msg = await asyncio.wait_for(ws.receive(), timeout)
    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
        raise APIConnectionError(f"Avaz closed the connection unexpectedly (code {ws.close_code})")
    if msg.type == aiohttp.WSMsgType.ERROR:
        cause = msg.data if isinstance(msg.data, BaseException) else None
        raise APIConnectionError("Avaz connection error") from cause
    if msg.type != aiohttp.WSMsgType.TEXT:
        raise APIError(f"Avaz sent an unexpected {msg.type.name} message")

    try:
        data = json.loads(msg.data)
    except json.JSONDecodeError:
        raise APIError("Avaz sent a message that is not valid JSON") from None

    if not isinstance(data, dict):
        raise APIError("Avaz sent a message that is not a JSON object")

    return data


def _decode_audio(audio_b64: Any, sample_rate: int) -> bytes:
    """Return the PCM frames of a base64 WAV, checking it matches the stream format."""
    if not isinstance(audio_b64, str):
        raise APIError("Avaz sent an invalid audio payload", retryable=False)

    expected_format = (NUM_CHANNELS, SAMPLE_WIDTH, sample_rate)
    try:
        with wave.open(io.BytesIO(base64.b64decode(audio_b64, validate=True)), "rb") as wav:
            audio_format = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
            if audio_format != expected_format:
                raise APIError(
                    f"Avaz sent audio as (channels, sample width, sample rate) {audio_format},"
                    f" expected {expected_format}",
                    retryable=False,
                )

            num_frames = wav.getnframes()
            pcm = wav.readframes(num_frames)
    except (ValueError, wave.Error, EOFError):  # binascii.Error is a ValueError
        raise APIError("Avaz sent audio that is not a valid WAV file", retryable=False) from None

    if len(pcm) != num_frames * NUM_CHANNELS * SAMPLE_WIDTH:
        raise APIError("Avaz sent a truncated WAV file", retryable=False)

    return pcm


def _to_ws_url(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    scheme = _WS_SCHEMES.get(parsed.scheme.lower())
    if scheme is None or not parsed.netloc:
        raise ValueError("Avaz base_url must be an http(s):// or ws(s):// URL")

    path = parsed.path.rstrip("/") + STREAM_PATH
    return urlunparse((scheme, parsed.netloc, path, "", parsed.query, ""))


def _is_secure_or_loopback(ws_url: str) -> bool:
    parsed = urlparse(ws_url)
    if parsed.scheme == "wss":
        return True

    host = parsed.hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_model(model: str) -> None:
    if model not in MODEL_SAMPLE_RATES:
        raise ValueError(
            f"unsupported Avaz model {model!r}, expected one of {list(MODEL_SAMPLE_RATES)}"
        )


def _validate_speaker_id(speaker_id: int) -> None:
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or speaker_id < 0:
        raise ValueError(f"Avaz speaker_id must be a non-negative integer, got {speaker_id!r}")
