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

"""Text-to-speech over the Vartalaap API.

Two paths, both 16-bit PCM mono at 24 kHz:

* :class:`ChunkedStream` (``synthesize()``): one ``POST /v1/tts`` per text, the chunked body is
  forwarded as it arrives.
* :class:`SynthesizeStream` (``stream()``): a pooled ``WS /v1/tts/stream`` connection. Text is cut
  into sentences as the LLM produces it and every sentence is sent as its own utterance; the server
  queues them in order on the connection and streams the audio of each one back as binary frames,
  so sentence N+1 is synthesised while N is still playing.

The server answers with the first 80 ms frame as soon as it is decoded (tens of milliseconds on a
warm engine); nothing here waits for a whole utterance.
"""

from __future__ import annotations

import asyncio
import json
import os
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

from .log import logger
from .models import TTSDigits, TTSExpressions, TTSModels, TTSProfiles, TTSVoices
from .version import __version__

DEFAULT_BASE_URL = "https://api.vartaloop.ai"
DEFAULT_MODEL = "narad"
DEFAULT_VOICE = "saanvi"
API_KEY_ENV = "VARTALAAP_API_KEY"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
USER_AGENT = f"livekit-plugins-vartalaap/{__version__}"

# A pooled socket is reopened after this long so one call cannot pin a connection forever.
_MAX_SESSION_DURATION = 300.0
# WebSocket protocol pings while the socket idles between turns in the pool.
_WS_HEARTBEAT = 20.0
# Wall-time ceiling for one REST request (the server's own per-request limit is 120 s).
_HTTP_TOTAL_TIMEOUT = 120.0


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str
    expression: TTSExpressions | str
    digits: TTSDigits | str
    speed: float | None
    temperature: float | None
    seed: int | None
    postfx: bool | None
    profile: TTSProfiles | str | None
    raw: bool
    sample_rate: int
    api_key: str
    base_url: str

    def http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def ws_url(self) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}/v1/tts/stream"


def _check_options(o: _TTSOptions) -> None:
    if o.speed is not None and not 0.5 <= o.speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    if o.temperature is not None and not 0.0 <= o.temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")
    if o.digits not in ("auto", "hi", "mixed", "en"):
        raise ValueError("digits must be auto, hi, mixed or en")
    if o.profile is not None and o.profile not in ("default", "telephony"):
        raise ValueError("profile must be default or telephony")


def _request_body(opts: _TTSOptions, *, text: str, request_id: str) -> dict[str, Any]:
    """The JSON of one utterance, shared by the REST and the WebSocket path.

    Only fields that differ from the server's defaults are sent, so the request log stays
    readable and the server's own default for anything unset keeps applying.
    """
    body: dict[str, Any] = {
        "text": text,
        "voice": opts.voice,
        "format": "pcm16",
        "request_id": request_id,
    }
    if opts.expression and opts.expression != "neutral":
        body["style"] = opts.expression
    if opts.digits:
        body["digits"] = opts.digits
    if opts.speed is not None:
        body["speed"] = opts.speed
    if opts.temperature is not None:
        body["temperature"] = opts.temperature
    if opts.seed is not None:
        body["seed"] = opts.seed
    if opts.postfx is not None:
        body["postfx"] = opts.postfx
    if opts.profile is not None and opts.profile != "default":
        body["profile"] = opts.profile
    if opts.raw:
        body["raw"] = True
    return body


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = DEFAULT_MODEL,
        voice: TTSVoices | str = DEFAULT_VOICE,
        expression: TTSExpressions | str = "neutral",
        digits: TTSDigits | str = "auto",
        speed: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        postfx: bool | None = None,
        profile: TTSProfiles | str | None = None,
        raw: bool = False,
        sample_rate: int = SAMPLE_RATE,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of Vartalaap TTS.

        See https://docs.vartalaap.si/api/tts/ for the API this plugin speaks.

        Args:
            api_key (str, optional): The Vartalaap API key (``vt_live_…``). Read from the
                ``VARTALAAP_API_KEY`` environment variable when not given.
            model (TTSModels | str, optional): The model family. Defaults to ``narad``.
            voice (TTSVoices | str, optional): The voice id. Defaults to ``saanvi``. The voice
                decides the language and so the text normaliser: Hinglish text to a Hinglish voice,
                English text to an English one. ``GET /v1/voices`` lists the ids.
            expression (TTSExpressions | str, optional): Delivery of the voice: ``neutral``
                (default), ``excited``, ``sad`` or ``whisper``. A voice without that expression
                plays neutral.
            digits (TTSDigits | str, optional): The language digits are read in for Hinglish
                voices: ``auto`` (default), ``hi``, ``mixed`` or ``en``.
            speed (float, optional): Speaking pace, 0.5 to 2.0; the pitch stays the same. Unset
                plays the voice's natural pace.
            temperature (float, optional): Sampling noise, 0 to 2 (server default 0.3).
            seed (int, optional): Fixes the sampling noise for reproducible output.
            postfx (bool, optional): The server's output chain (loudness to -16 LUFS, high-pass,
                true-peak limiter). Unset keeps the server default (on).
            profile (TTSProfiles | str, optional): ``default`` or ``telephony``; the latter shapes
                the audio for 8 kHz mu-law delivery.
            raw (bool, optional): Skip the server's text normaliser (the text is already
                normalised). Defaults to False.
            sample_rate (int, optional): Output sample rate. The API produces 24 kHz; this must
                be 24000.
            base_url (str, optional): The API base URL. Defaults to ``https://api.vartaloop.ai``.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp session to
                use. If not provided, the agents' shared session is used.
            tokenizer (tokenize.SentenceTokenizer, optional): How streamed text is cut into
                utterances. Defaults to ``tokenize.basic.SentenceTokenizer``, which ends a sentence
                on the Devanagari danda as well as on ``.?!``.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        vartalaap_api_key = api_key or os.environ.get(API_KEY_ENV)
        if not vartalaap_api_key:
            raise ValueError(
                "Vartalaap API key is required, either as argument or set"
                f" {API_KEY_ENV} environment variable"
            )
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Vartalaap TTS produces {SAMPLE_RATE} Hz audio; sample_rate must be it"
            )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            expression=expression,
            digits=digits,
            speed=speed,
            temperature=temperature,
            seed=seed,
            postfx=postfx,
            profile=profile,
            raw=raw,
            sample_rate=sample_rate,
            api_key=vartalaap_api_key,
            base_url=base_url.rstrip("/"),
        )
        _check_options(self._opts)

        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=_MAX_SESSION_DURATION,
            mark_refreshed_on_get=False,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.basic.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Vartalaap"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._opts.api_key}", "User-Agent": USER_AGENT}

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._opts.ws_url(), headers=self._auth_headers(), heartbeat=_WS_HEARTBEAT
                ),
                timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            # the handshake's RequestInfo carries the Authorization header: do not chain it
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            # transport errors can quote the URL (which may carry a key): keep the type only
            raise APIConnectionError(type(e).__name__) from None

        logger.debug("Established new Vartalaap TTS WebSocket connection")
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def prewarm(self) -> None:
        """Open the pooled WebSocket ahead of the first turn so the greeting does not pay the
        TCP + TLS handshake."""
        self._pool.prewarm()

    async def list_voices(self) -> list[dict[str, Any]]:
        """``GET /v1/voices``: the voices this key may use, with ``language``, ``gender``,
        ``styles`` (the expressions a voice carries), ``speed`` and ``model``."""
        async with self._ensure_session().get(
            self._opts.http_url("/v1/voices"), headers=self._auth_headers()
        ) as resp:
            if resp.status != 200:
                raise APIStatusError(
                    message=await resp.text(), status_code=resp.status, request_id=None, body=None
                )
            data = await resp.json()
            voices: list[dict[str, Any]] = data.get("voices", [])
            return voices

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        expression: NotGivenOr[TTSExpressions | str] = NOT_GIVEN,
        digits: NotGivenOr[TTSDigits | str] = NOT_GIVEN,
        speed: NotGivenOr[float | None] = NOT_GIVEN,
        temperature: NotGivenOr[float | None] = NOT_GIVEN,
        seed: NotGivenOr[int | None] = NOT_GIVEN,
        postfx: NotGivenOr[bool | None] = NOT_GIVEN,
        profile: NotGivenOr[TTSProfiles | str | None] = NOT_GIVEN,
        raw: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """Update the TTS options.

        Every option travels in the body of each utterance, so a change applies from the next
        sentence on and the pooled WebSocket is kept. Options not given keep their value.

        Args:
            model (TTSModels | str, optional): The model family.
            voice (TTSVoices | str, optional): The voice id.
            expression (TTSExpressions | str, optional): ``neutral``, ``excited``, ``sad``, ``whisper``.
            digits (TTSDigits | str, optional): ``auto``, ``hi``, ``mixed`` or ``en``.
            speed (float | None, optional): Pace 0.5 to 2.0; ``None`` returns to the voice's pace.
            temperature (float | None, optional): Sampling noise 0 to 2.
            seed (int | None, optional): Fixed sampling seed.
            postfx (bool | None, optional): The output chain on or off.
            profile (TTSProfiles | str | None, optional): ``default`` or ``telephony``.
            raw (bool, optional): Skip the normaliser.
        """
        opts = replace(self._opts)
        if is_given(model):
            opts.model = model
        if is_given(voice):
            opts.voice = voice
        if is_given(expression):
            opts.expression = expression
        if is_given(digits):
            opts.digits = digits
        if is_given(speed):
            opts.speed = speed
        if is_given(temperature):
            opts.temperature = temperature
        if is_given(seed):
            opts.seed = seed
        if is_given(postfx):
            opts.postfx = postfx
        if is_given(profile):
            opts.profile = profile
        if is_given(raw):
            opts.raw = raw
        _check_options(opts)  # a rejected update leaves the current options untouched
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
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize one text over ``POST /v1/tts``, forwarding the chunked PCM body as it arrives."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        body = _request_body(self._opts, text=self._input_text, request_id=request_id)
        body["stream"] = True

        try:
            async with self._tts._ensure_session().post(
                self._opts.http_url("/v1/tts"),
                headers=self._tts._auth_headers(),
                json=body,
                timeout=aiohttp.ClientTimeout(
                    total=_HTTP_TOTAL_TIMEOUT, sock_connect=self._conn_options.timeout
                ),
            ) as resp:
                if resp.status != 200:
                    raise _status_error(
                        resp.status,
                        await resp.text(),
                        request_id=resp.headers.get("x-request-id", request_id),
                        retry_after=resp.headers.get("retry-after"),
                    )

                output_emitter.initialize(
                    request_id=resp.headers.get("x-request-id", request_id),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                # iter_chunks yields each transfer chunk as it lands: the first one is the first
                # 80 ms frame, the rest follow ahead of real time
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed synthesis over the pooled ``WS /v1/tts/stream`` connection.

    Text pushed by the agent is cut into sentences; each sentence is one utterance message with its
    own ``request_id``. The server serves them in order on the connection, so the next sentence is
    already being synthesised while the current one plays. Binary frames are the PCM of the
    utterance in progress; ``start`` and ``end`` events delimit it.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
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

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        # utterances sent and not yet ended: (request_id, segment_id), in order
        pending: deque[tuple[str, str]] = deque()
        # set once a message is on the wire, or when there is nothing to send
        sent_or_done = asyncio.Event()
        sender_done = asyncio.Event()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                async for ev in sent_tokenizer_stream:
                    text = ev.token.strip()
                    if not text:
                        continue
                    utt_id = utils.shortuuid()
                    # the owning TTS's current options: an update_options() between two
                    # sentences applies to the next one, the socket stays
                    body = _request_body(self._tts._opts, text=text, request_id=utt_id)
                    pending.append((utt_id, ev.segment_id or request_id))
                    self._mark_started()
                    await ws.send_str(json.dumps(body))
                    sent_or_done.set()
            finally:
                sender_done.set()
                sent_or_done.set()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await sent_or_done.wait()
            if sender_done.is_set() and not pending:
                return  # nothing was sent (empty input)

            current_segment: str | None = None
            current_utt: str | None = None
            while True:
                if sender_done.is_set() and not pending:
                    # the last utterance ended before the input closed: nothing more will come
                    output_emitter.end_input()
                    return
                msg = await _receive_or_sender_done(ws)
                if msg is None:
                    continue
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Vartalaap connection closed unexpectedly",
                        request_id=request_id,
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type == aiohttp.WSMsgType.BINARY:
                    if current_utt is None:
                        # audio of an utterance we did not start (a stale pooled socket)
                        continue
                    output_emitter.push(msg.data)
                    continue

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected Vartalaap message type %s", msg.type)
                    continue

                data = json.loads(msg.data)
                ev_type = data.get("type")
                ev_utt = data.get("request_id")

                ours = bool(pending) and ev_utt == pending[0][0]
                if ev_type == "error":
                    if ev_utt is not None and not ours:
                        # a late error of an utterance that is not ours (pooled socket leftover)
                        continue
                    raise _status_error(
                        int(data.get("status") or 500),
                        str(data.get("error", "unknown error")),
                        request_id=ev_utt or request_id,
                        retry_after=data.get("retry_after"),
                    )

                if not ours:
                    # an event of an utterance that is not ours (left over on a pooled socket)
                    continue

                if ev_type == "start":
                    current_utt, seg = pending[0]
                    if current_segment != seg:
                        if current_segment is not None:
                            output_emitter.end_segment()
                        output_emitter.start_segment(segment_id=seg)
                        current_segment = seg
                    output_emitter._note_provider_request_id(current_utt)
                elif ev_type == "end":
                    pending.popleft()
                    current_utt = None
                    if sender_done.is_set() and not pending:
                        output_emitter.end_input()
                        return
                # chunk, first_audio_ms, chunk_end: progress only

        async def _receive_or_sender_done(
            ws: aiohttp.ClientWebSocketResponse,
        ) -> aiohttp.WSMessage | None:
            """The next message, or None once the sender has finished (so the loop can check
            whether anything is still pending instead of waiting for a message that never comes)."""
            if sender_done.is_set():
                return await ws.receive(timeout=self._conn_options.timeout)
            # while an utterance is pending the server owes us a message within the timeout;
            # with nothing pending the pause is the LLM's (or the user's), not the server's
            timeout = self._conn_options.timeout if pending else None
            recv = asyncio.ensure_future(ws.receive(timeout=timeout))
            done_waiter = asyncio.ensure_future(sender_done.wait())
            try:
                await asyncio.wait({recv, done_waiter}, return_when=asyncio.FIRST_COMPLETED)
                if recv.done():
                    return recv.result()
                return None
            finally:
                if not recv.done():
                    # cancel the pending receive; the socket stays usable (aiohttp re-arms)
                    await utils.aio.gracefully_cancel(recv)
                if not done_waiter.done():
                    await utils.aio.gracefully_cancel(done_waiter)

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._tts._pool.last_acquire_time
                self._connection_reused = self._tts._pool.last_connection_reused
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_send_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    sent_or_done.set()
                    await sent_tokenizer_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


def _status_error(
    status: int, message: str, *, request_id: str | None, retry_after: Any = None
) -> APIStatusError:
    """An API error with the server's status. 503 (every worker busy, retry after ``Retry-After``)
    and 5xx are retryable; 4xx (bad text, voice, key, credits) are not."""
    try:
        body: object = json.loads(message)
    except (ValueError, TypeError):
        body = message
    if isinstance(body, dict) and "error" in body:
        message = str(body["error"])
    if retry_after is not None:
        message = f"{message} (retry after {retry_after}s)"
    return APIStatusError(
        message=message,
        status_code=status,
        request_id=request_id,
        body=body,
        retryable=status >= 500,
    )
