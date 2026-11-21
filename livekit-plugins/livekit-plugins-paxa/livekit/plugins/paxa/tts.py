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

"""Text-to-speech for Paxa.

``synthesize()`` streams one ``POST /v1/tts`` request. ``stream()`` runs one session on
the ``/v1/tts/live`` WebSocket. Both ask for pcm, 16-bit mono with no header, at the TTS's
sample rate, and pass it on as it arrives.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import weakref
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

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

from ._utils import (
    check_pcm_content_type,
    http_status_error,
    live_close_error,
    live_event_error,
    live_url,
    normalize_base_url,
    request_headers,
    take_frame,
    validate_options,
)
from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    NUM_CHANNELS,
    TTSLanguages,
    TTSModels,
)

# Stays well inside Paxa's per-leg ceiling (max_chars, 5000 today) and its 64 KiB frame
# limit.
_MAX_FRAME_CHARS = 2000
# How long a sent frame may go without starting a leg (a frame of tags alone starts
# none) before the next frame goes out anyway.
_LEG_START_GRACE = 0.5
# A direction tag, or one still open at a frame's end. Paxa starts no leg for a frame of
# these alone, and sends nothing back for it.
_TAG_RE = re.compile(r"\[[^\]]*(?:\]|$)|<[^>]*(?:>|$)")


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: str
    language: TTSLanguages | str
    speed: float
    sample_rate: int
    api_key: str
    base_url: str


class TTS(tts.TTS):
    """Paxa text-to-speech for LiveKit Agents.

    ``stream()`` uses the live WebSocket (``/v1/tts/live``) and ``synthesize()`` uses
    ``POST /v1/tts``. Audio is 16-bit mono PCM at ``sample_rate``, 24 kHz by default.
    """

    def __init__(
        self,
        *,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        language: TTSLanguages | str = DEFAULT_LANGUAGE,
        speed: float = DEFAULT_SPEED,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        api_key: str | None = None,
        base_url: str | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of Paxa TTS.

        ``api_key`` must be set to your Paxa API key, either using the argument or by
        setting the ``PAXA_API_KEY`` environment variable.

        Args:
            model: TTS model id. Defaults to ``paxa-tts-flash-v2``, whose voices each
                speak Thai, English and Mandarin and read direction tags such as
                ``[whispering]``.
            voice: Voice id from ``GET /v1/voices``, such as ``foithong`` or
                ``massaman``, or a designed voice id starting with ``voice_``.
            language: How digits, codes and symbols are read: ``auto`` (default),
                ``th``, ``en`` or ``zh``. Words always keep the language of their script,
                so mixed Thai, English and Mandarin text needs no setting.
            speed: Speaking rate, from 0.5 to 1.5. Defaults to 1.0.
            sample_rate: Output sample rate in Hz. Defaults to 24000. Paxa TTS Flash v2
                serves 8000, 16000, 22050, 24000, 32000, 44100 and 48000; Paxa TTS Flash
                serves 24000 only. Every rate costs the same.
            api_key: Paxa API key. Defaults to the ``PAXA_API_KEY`` environment variable.
            base_url: API origin. Defaults to the ``PAXA_BASE_URL`` environment
                variable, then ``https://api.paxalabs.com``.
            tokenizer: Sentence tokenizer that splits streamed text into text frames.
                Defaults to ``livekit.agents.tokenize.blingfire.SentenceTokenizer``.
            http_session: Optional aiohttp session to use for requests.
        """
        paxa_api_key = api_key or os.environ.get("PAXA_API_KEY")
        if not paxa_api_key:
            raise ValueError(
                "Paxa API key is required, either as argument or set PAXA_API_KEY "
                "environment variable"
            )
        validate_options(
            model=str(model),
            voice=voice,
            language=str(language),
            speed=speed,
            sample_rate=sample_rate,
        )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language=language,
            speed=speed,
            sample_rate=sample_rate,
            api_key=paxa_api_key,
            base_url=normalize_base_url(
                base_url or os.environ.get("PAXA_BASE_URL") or DEFAULT_BASE_URL
            ),
        )
        self._session = http_session
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "Paxa"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update options for synthesis started after this call.

        Args:
            model: TTS model id. It must serve the ``sample_rate`` the TTS was created with.
            voice: Voice id from ``GET /v1/voices``, or a designed voice id.
            language: ``auto``, ``th``, ``en`` or ``zh``.
            speed: Speaking rate, from 0.5 to 1.5.
        """
        opts = replace(self._opts)
        if is_given(model):
            opts.model = model
        if is_given(voice):
            opts.voice = voice
        if is_given(language):
            opts.language = language
        if is_given(speed):
            opts.speed = speed
        validate_options(
            model=str(opts.model),
            voice=opts.voice,
            language=str(opts.language),
            speed=opts.speed,
            sample_rate=opts.sample_rate,
        )
        self._opts = opts

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class ChunkedStream(tts.ChunkedStream):
    """Synthesizes one text with a streamed ``POST /v1/tts`` request."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        if not self._input_text.strip():
            # Paxa refuses text with nothing to speak, so blank text makes no request.
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
            )
            return

        payload = {
            "text": self._input_text,
            "voice": self._opts.voice,
            "model": self._opts.model,
            "format": "pcm",
            "sample_rate": self._opts.sample_rate,
            "stream": True,
            "speed": self._opts.speed,
            "language": self._opts.language,
        }
        try:
            async with self._tts._ensure_session().post(
                f"{self._opts.base_url}/v1/tts",
                headers=request_headers(self._opts.api_key),
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    sock_connect=self._conn_options.timeout,
                    sock_read=self._conn_options.timeout,
                ),
            ) as resp:
                request_id = resp.headers.get("x-request-id") or request_id
                if resp.status >= 400:
                    raise http_status_error(resp.status, await resp.read(), request_id=request_id)
                if not resp.content_type.startswith("audio/"):
                    raise APIStatusError(
                        "Paxa TTS returned a non-audio response",
                        status_code=502,
                        request_id=request_id,
                    )
                check_pcm_content_type(
                    resp.headers.get("Content-Type", ""), sample_rate=self._opts.sample_rate
                )

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)
                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError("Paxa TTS request timed out") from None
        except aiohttp.ClientError as e:
            # Paxa commits a streamed response with its first byte, so a failure after it
            # arrives here as a truncated body.
            raise APIConnectionError("Paxa TTS request failed") from e


class _FrameGate:
    """Decides when the next text frame may go out on a live session, and tracks the
    frames Paxa owes a leg.

    Paxa runs one synthesis leg at a time and closes a session that queues 32 frames
    behind one. A frame therefore goes out once the previous frame has started a leg and
    that leg has flushed; sentences that arrive in the meantime share the next frame.
    """

    def __init__(self) -> None:
        self.open_leg: int | None = None
        # When each frame still owed a leg went out. Paxa runs the frames in order, one leg
        # for each frame with something to say, so a leg pays off the oldest.
        self._owed: deque[float] = deque()
        self._unacked = 0
        self._last_sent = 0.0

    @property
    def owed_since(self) -> float | None:
        """When the oldest frame still owed a leg went out."""
        return self._owed[0] if self._owed else None

    def can_send(self, now: float) -> bool:
        if self.open_leg is not None:
            return False
        return self._unacked == 0 or now - self._last_sent >= _LEG_START_GRACE

    def wait_time(self, now: float) -> float | None:
        """Seconds until ``can_send`` turns true without a server event, if that is
        still ahead."""
        if self.open_leg is None and self._unacked:
            grace_left = self._last_sent + _LEG_START_GRACE - now
            if grace_left > 0:
                return grace_left
        return None

    def sent(self, now: float, text: str) -> None:
        self._unacked += 1
        self._last_sent = now
        if _TAG_RE.sub("", text).strip():
            self._owed.append(now)

    def leg_started(self, seq: int) -> None:
        self._unacked = 0
        if self._owed:
            self._owed.popleft()
        self.open_leg = seq

    def leg_flushed(self, seq: int) -> bool:
        # A flush with an empty buffer is acknowledged with the last leg's seq again.
        if seq != self.open_leg:
            return False
        self.open_leg = None
        return True


class SynthesizeStream(tts.SynthesizeStream):
    """Streams text to Paxa over the ``/v1/tts/live`` WebSocket.

    Each stream opens its own live session once text arrives: a ``start`` frame, a
    ``text`` frame per sentence, then ``end``. Every text frame ends with a newline, so
    Paxa starts a synthesis leg for it right away. The stream completes on the session's
    ``done`` event.

    Paxa sessions carry no context ids, so an interrupted stream closes its socket, which
    stops the leg in flight, and the next stream opens a new session.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
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
        pending: list[str] = []
        gate = _FrameGate()
        wake = asyncio.Event()
        text_or_end = asyncio.Event()
        has_text = False
        input_done = False
        end_sent_at: float | None = None
        started = False
        connection_id: str | None = None

        async def _forward_input() -> None:
            nonlocal has_text
            try:
                async for data in self._input_ch:
                    if isinstance(data, str):
                        if data.strip():
                            has_text = True
                            text_or_end.set()
                        sentence_stream.push_text(data)
                    elif has_text:
                        # A stream speaks a single segment, which its first flush ends.
                        break
                sentence_stream.end_input()
            finally:
                text_or_end.set()

        async def _collect_sentences() -> None:
            nonlocal input_done
            async for ev in sentence_stream:
                if ev.token.strip():
                    pending.append(ev.token)
                    wake.set()
            input_done = True
            wake.set()

        async def _send_loop(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal end_sent_at
            try:
                await ws.send_str(json.dumps(self._start_frame()))
                while True:
                    try:
                        await asyncio.wait_for(wake.wait(), gate.wait_time(time.monotonic()))
                    except asyncio.TimeoutError:
                        pass
                    wake.clear()

                    while pending and gate.can_send(time.monotonic()):
                        text = take_frame(pending, _MAX_FRAME_CHARS)
                        self._mark_started()
                        await ws.send_str(
                            json.dumps({"type": "text", "text": text + "\n"}, ensure_ascii=False)
                        )
                        gate.sent(time.monotonic(), text)

                    if input_done and not pending:
                        await ws.send_str(json.dumps({"type": "end"}))
                        end_sent_at = time.monotonic()
                        return
            except (aiohttp.ClientError, ConnectionError):
                # The socket is gone. The receive loop reports why it closed.
                return

        async def _recv_loop(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal started, connection_id
            timeout = self._conn_options.timeout
            wait = timeout
            while True:
                try:
                    msg = await ws.receive(timeout=wait)
                except asyncio.TimeoutError:
                    # Silence is expected while the LLM is still writing; it is a failure
                    # only while Paxa owes the session a response.
                    if not started or gate.open_leg is not None:
                        raise APITimeoutError("Paxa TTS live session timed out") from None
                    owed = [t for t in (gate.owed_since, end_sent_at) if t is not None]
                    # A frame or the end sent late in the silence gets the whole timeout.
                    wait = min(owed) + timeout - time.monotonic() if owed else timeout
                    if wait <= 0:
                        raise APITimeoutError("Paxa TTS live session timed out") from None
                    continue
                wait = timeout

                if msg.type == aiohttp.WSMsgType.BINARY:
                    if not started:
                        raise APIConnectionError("Paxa TTS sent audio before the session started")
                    output_emitter.push(msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    event = _parse_event(msg.data)
                    kind = event.get("type")
                    seq = event.get("seq")
                    if kind == "started":
                        started = True
                        connection = event.get("connection")
                        connection_id = connection if isinstance(connection, str) else request_id
                        # Paxa's connection id identifies the session in its support logs.
                        output_emitter.start_segment(segment_id=connection_id)
                    elif kind == "charged" and isinstance(seq, int):
                        gate.leg_started(seq)
                        wake.set()
                    elif kind == "flushed" and isinstance(seq, int):
                        if gate.leg_flushed(seq):
                            wake.set()
                    elif kind == "error":
                        raise live_event_error(event, connection_id=connection_id)
                    elif kind == "done":
                        output_emitter.end_segment()
                        return
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    # The close frame carries Paxa's code; ws.close_code reads 1006 when
                    # Paxa drops the connection right after sending it.
                    code = msg.data if msg.type == aiohttp.WSMsgType.CLOSE else ws.close_code
                    raise live_close_error(code, connection_id=connection_id)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError("Paxa TTS live connection failed")

        tasks = [
            asyncio.create_task(_forward_input(), name="paxa_tts_forward_input"),
            asyncio.create_task(_collect_sentences(), name="paxa_tts_collect_sentences"),
        ]
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            await text_or_end.wait()
            if has_text:
                # Connecting on the first text, not on stream creation, keeps a turn that
                # speaks nothing (a tool call) from opening a session at all.
                ws = await self._connect()
                tasks.append(asyncio.create_task(_send_loop(ws), name="paxa_tts_send"))
                tasks.append(asyncio.create_task(_recv_loop(ws), name="paxa_tts_recv"))
            await asyncio.gather(*tasks)
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError("Paxa TTS live session failed") from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await sentence_stream.aclose()
            if ws is not None:
                # Closing the socket stops a leg in flight, so an interrupted stream stops
                # synthesizing at once.
                await ws.close()

    def _start_frame(self) -> dict[str, Any]:
        return {
            "type": "start",
            "model": self._opts.model,
            "voice": self._opts.voice,
            "format": "pcm",
            "sample_rate": self._opts.sample_rate,
            "speed": self._opts.speed,
            "language": self._opts.language,
        }

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        try:
            return await asyncio.wait_for(
                self._tts._ensure_session().ws_connect(
                    live_url(self._opts.base_url), headers=request_headers(self._opts.api_key)
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("Paxa TTS live connection timed out") from None
        except aiohttp.WSServerHandshakeError as e:
            # The handshake error keeps the request headers, the API key among them, so it
            # is not chained.
            raise APIStatusError(
                "Paxa TTS live connection was refused", status_code=e.status
            ) from None
        except aiohttp.ClientError as e:
            raise APIConnectionError("Paxa TTS live connection failed") from e


def _parse_event(data: str) -> dict[str, Any]:
    try:
        event = json.loads(data)
    except ValueError:
        event = None
    if not isinstance(event, dict):
        raise APIConnectionError("Paxa TTS sent an unreadable live event")
    return event
