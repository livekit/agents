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

"""Qwen3-TTS protocol support for `TTS` (``model="qwen3-tts"``).

Qwen3-TTS is: session.config -> input.text* -> input.done (a flush, not a
close) -> audio.start / PCM / audio.done / session.done. That is a different
protocol from the Orpheus truss `TTS` also serves, so it gets its own backend
and stream implementation here; `TTS` picks between them.

Voices are clones, not presets: the Base checkpoint ships no built-in speakers.
Register one first (see `register_voice`), or pass ref_audio/ref_text to clone
inline. The server keeps uploaded voices on the container's local disk, so a
runtime-registered voice is scoped to one replica and lost on restart — bake it
into the truss via REQUIRED_VOICES for anything beyond single-replica tests.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts as tts_module,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
    TimedString,
)
from livekit.agents.utils import is_given

from ._endpoint import resolve_endpoint
from .log import logger

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
_BYTES_PER_SECOND = SAMPLE_RATE * NUM_CHANNELS * 2
_WS_MAX_MSG_SIZE = 16 * 1024 * 1024

# Server gives 10s for the first session.config, then 30s idle per message.
# Ours sit below so we drop a stale socket instead of dying mid-turn.
_UNCONFIGURED_TTL = 8.0
_CONFIGURED_TTL = 25.0
_KEEPALIVE_INTERVAL = 15.0
_KEEPALIVE_TIMEOUT = 3.0


class WordAlignment(BaseModel):
    model_config = ConfigDict(strict=True)

    words: list[str] = Field(default_factory=list[str])
    word_start_times_seconds: list[float] = Field(default_factory=list[float])
    word_end_times_seconds: list[float] = Field(default_factory=list[float])


class TimestampInfo(BaseModel):
    model_config = ConfigDict(strict=True)

    word_alignment: WordAlignment | None = None


class TTSResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    type: str
    sample_rate: int = SAMPLE_RATE
    error: str | bool | None = None
    sentence_index: int | None = None
    timestamp_info: TimestampInfo | None = None
    message: str = "unknown TTS error"


def _decode_response(payload: str) -> TTSResponse:
    try:
        return TTSResponse.model_validate_json(payload)
    except ValidationError as error:
        raise APIConnectionError(
            "Baseten returned an invalid synthesis response", retryable=False
        ) from error


_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


@dataclass
class _TTSOptions:
    voice: str
    task_type: str
    language: str | None
    speed: float
    instructions: str | None
    max_new_tokens: int | None
    initial_codec_chunk_frames: int | None
    x_vector_only_mode: bool | None
    ref_audio: str | None
    ref_text: str | None
    word_timestamps: bool
    extra_config: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])


@dataclass
class _WarmSocket:
    ws: aiohttp.ClientWebSocketResponse
    config: dict[str, JsonValue] | None
    last_activity: float
    conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS


@dataclass
class _TurnState:
    """Sender counts flushes issued, receiver counts the ones answered."""

    flushes_sent: int = 0
    flushes_done: int = 0
    sender_finished: asyncio.Event = field(default_factory=asyncio.Event)
    progress: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def settled(self) -> bool:
        return self.sender_finished.is_set() and self.flushes_done >= self.flushes_sent


class Qwen3Backend:
    """Connection state for a Qwen3-TTS deployment, owned by `TTS`.

    Not a `tts.TTS` itself — `TTS` selects this when ``model="qwen3-tts"`` and
    keeps the public surface. Holds one warm WebSocket across turns, since the
    session config is sticky and re-dialing would add a connect plus a config
    round trip to every agent response.
    """

    def __init__(
        self,
        *,
        model_endpoint: str | None = None,
        model_id: str | None = None,
        chain_id: str | None = None,
        api_key: str | None = None,
        voice: str,
        task_type: str = "Base",
        language: str | None = "Auto",
        speed: float = 1.0,
        instructions: str | None = None,
        max_new_tokens: int | None = None,
        initial_codec_chunk_frames: int | None = None,
        x_vector_only_mode: bool | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        word_timestamps: bool = False,
        extra_config: Mapping[str, JsonValue] | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Args:
            model_endpoint: Full ``wss://`` URL of the deployment. Takes priority
                over ``model_id`` and ``chain_id``; falls back to
                ``BASETEN_MODEL_ENDPOINT``.
            model_id: Baseten truss model ID; builds the endpoint URL for you.
            chain_id: Baseten chain ID; builds the endpoint URL for you.
            api_key: Baseten API key, or ``BASETEN_API_KEY``.
            voice: Name of a registered voice clone. Required — the Base
                checkpoint ships no built-in speakers, so there is no sensible
                default. See `register_voice`.
            task_type: ``Base`` for the voice-cloning checkpoint. ``CustomVoice``
                and ``VoiceDesign`` deployments use their own checkpoints and
                expose preset speaker names.
            language: Language hint; ``Auto`` lets the model decide.
            speed: Playback rate. Forced to 1.0 — progressive PCM is only offered
                at native speed and the server rejects anything else, which is
                fatal on a socket that has no session yet.
            instructions: Style prompt (``CustomVoice``/``VoiceDesign`` only).
            max_new_tokens: Cap on generated audio tokens per sentence.
            initial_codec_chunk_frames: First-chunk size. Larger trades
                time-to-first-audio for slightly better onset quality.
            x_vector_only_mode: Condition on the speaker embedding alone and skip
                in-context learning from the reference. Faster, less faithful.
            ref_audio: Reference clip for inline cloning — an ``http(s)`` URL or a
                local file path. Only needed when ``voice`` is not already
                registered on the deployment.
            ref_text: Transcript of ``ref_audio``. Improves fidelity; without it
                the server falls back to audio-only (x-vector).
            word_timestamps: Request word-level alignment and forward it to
                LiveKit as a timed transcript.
            extra_config: Merged into ``session.config`` last, for server-side
                fields newer than this plugin. Note that an unrecognized field
                invalidates the whole config.
            http_session: Optional aiohttp session to reuse.

        Raises:
            ValueError: If no API key or endpoint can be resolved, or if
                ``voice`` is empty.
        """
        api_key = api_key or os.environ.get("BASETEN_API_KEY")
        if not api_key:
            raise ValueError("Pass `api_key` or set BASETEN_API_KEY.")

        model_endpoint = resolve_endpoint(
            model_endpoint, model_id, chain_id, "BASETEN_MODEL_ENDPOINT"
        )
        if not voice:
            raise ValueError("`voice` is required: the Base checkpoint has no presets.")

        if speed != 1.0:
            logger.warning("progressive PCM requires speed=1.0; overriding %s", speed)
            speed = 1.0

        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._session = http_session
        self._opts = _TTSOptions(
            voice=voice,
            task_type=task_type,
            language=language,
            speed=speed,
            instructions=instructions,
            max_new_tokens=max_new_tokens,
            initial_codec_chunk_frames=initial_codec_chunk_frames,
            x_vector_only_mode=x_vector_only_mode,
            ref_audio=ref_audio,
            ref_text=ref_text,
            word_timestamps=word_timestamps,
            extra_config=dict(extra_config or {}),
        )

        self._ref_audio_b64: str | None = None
        self._warm: _WarmSocket | None = None
        self._warm_lock = asyncio.Lock()
        self._keepalive_task: asyncio.Task[None] | None = None
        self._closing = False

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        max_new_tokens: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """Applies from the next turn: config is sticky for a socket's life."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(instructions):
            self._opts.instructions = instructions
        if is_given(max_new_tokens):
            self._opts.max_new_tokens = max_new_tokens

    def make_stream(
        self, owner: tts_module.TTS, *, conn_options: APIConnectOptions
    ) -> Qwen3SynthesizeStream:
        return Qwen3SynthesizeStream(
            tts=owner, backend=self, opts=replace(self._opts), conn_options=conn_options
        )

    async def aclose(self) -> None:
        self._closing = True
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
        async with self._warm_lock:
            warm, self._warm = self._warm, None
        if warm:
            await self._shutdown_ws(warm.ws, notify=True)

    def _load_ref_audio(self, ref: str | None) -> str | None:
        if not ref or ref.startswith(("http://", "https://")):
            return ref
        if self._ref_audio_b64 is None:
            if not os.path.isfile(ref):
                logger.error("ref_audio %r is neither a URL nor a file", ref)
                self._opts.ref_audio = None
                return None
            with open(ref, "rb") as f:
                self._ref_audio_b64 = base64.b64encode(f.read()).decode()
        return self._ref_audio_b64

    def build_session_config(self, opts: _TTSOptions) -> dict[str, JsonValue]:
        """Doubles as the identity of the config applied to a parked socket.

        `split_granularity` is omitted on purpose: it is not in the streaming
        schema, and an unrecognized field invalidates the whole config.
        """
        o = opts
        config: dict[str, JsonValue] = {
            "task_type": o.task_type,
            "response_format": "pcm",
            "stream_audio": True,
            "speed": o.speed,
            "voice": o.voice,
        }
        optional: dict[str, JsonValue] = {
            "language": o.language,
            "instructions": o.instructions,
            "max_new_tokens": o.max_new_tokens,
            "initial_codec_chunk_frames": o.initial_codec_chunk_frames,
            "x_vector_only_mode": o.x_vector_only_mode,
        }
        config.update({k: v for k, v in optional.items() if v is not None})

        ref_audio = self._load_ref_audio(o.ref_audio)
        if ref_audio:
            config["ref_audio"] = ref_audio
            if o.ref_text:
                config["ref_text"] = o.ref_text
        if o.word_timestamps:
            # "sync" lands in audio.done, after that sentence's audio, so its
            # offset is known exactly; "async" arrives arbitrarily interleaved.
            config["timestamp_type"] = "word"
            config["timestamp_transport_strategy"] = "sync"
        config.update(o.extra_config)
        return config

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def acquire(
        self,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tuple[aiohttp.ClientWebSocketResponse, dict[str, JsonValue] | None]:
        """Reuse the parked socket if live, else dial. Returns its applied config."""
        async with self._warm_lock:
            warm, self._warm = self._warm, None

        if warm:
            ttl = _CONFIGURED_TTL if warm.config else _UNCONFIGURED_TTL
            if (
                not warm.ws.closed
                and (time.monotonic() - warm.last_activity) < ttl
                and warm.conn_options == conn_options
            ):
                return warm.ws, warm.config
            await self._shutdown_ws(warm.ws, notify=False)

        ws = await utils.http_context.ws_connect(
            self._ensure_session(),
            self._model_endpoint,
            conn_options=conn_options,
            headers={"Authorization": f"Api-Key {self._api_key}"},
            max_msg_size=_WS_MAX_MSG_SIZE,
            compress=0,
        )
        return ws, None

    async def release(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        config: dict[str, JsonValue] | None,
        *,
        discard: bool,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Park for the next turn, or discard.

        Discard is for a socket abandoned mid-generation (barge-in, error):
        closing it is what stops the server, and a polite session.close would
        wait on audio nobody will hear.
        """
        if discard or self._closing or ws.closed:
            await self._shutdown_ws(ws, notify=not discard)
            return

        async with self._warm_lock:
            stale, self._warm = self._warm, _WarmSocket(ws, config, time.monotonic(), conn_options)
        if stale and stale.ws is not ws:
            await self._shutdown_ws(stale.ws, notify=True)

        if not self._closing and (self._keepalive_task is None or self._keepalive_task.done()):
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self) -> None:
        """Keep the parked socket off the server's idle timeout.

        An empty ``input.done`` answers ``session.done`` with zero sentences.
        Unlike a protocol ping that proves the *session* is alive, not merely
        the TCP connection.

        The loop only ever exits on shutdown. It deliberately does not return
        when it finds nothing parked or when a flush fails: it releases
        ``_warm_lock`` for the round trip, so a turn can park a fresh socket at
        any point, and ``release`` cannot schedule a keepalive for it while
        this task is still running (not ``done()``). Returning early would
        leave that socket to idle out.
        """
        while not self._closing:
            await asyncio.sleep(_KEEPALIVE_INTERVAL)

            async with self._warm_lock:
                warm = self._warm
                if warm is None or not warm.config:
                    continue  # nothing parked, or no session on it yet
                if warm.ws.closed:
                    self._warm = None
                    continue
                # Take it out for the round trip so a turn starting now dials
                # its own socket instead of blocking on the lock.
                self._warm = None

            try:
                alive = await self._empty_flush(warm.ws)
            except asyncio.CancelledError:
                # This local is the only reference while the flush is in
                # flight, so aclose() cancelling here would leak it.
                await self._shutdown_ws(warm.ws, notify=False)
                raise

            if not alive:
                await self._shutdown_ws(warm.ws, notify=False)
                continue

            warm.last_activity = time.monotonic()
            async with self._warm_lock:
                surplus = self._warm is not None
                if not surplus:
                    self._warm = warm
            if surplus:
                # A turn parked its own socket while we were flushing; ours is
                # redundant.
                await self._shutdown_ws(warm.ws, notify=True)

    async def _empty_flush(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        try:
            await ws.send_str(json.dumps({"type": "input.done"}))
            deadline = time.monotonic() + _KEEPALIVE_TIMEOUT
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                msg = await ws.receive(timeout=remaining)
                if msg.type is aiohttp.WSMsgType.BINARY:
                    continue
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    return False
                event = _decode_response(msg.data)
                if event.type == "session.done":
                    return True
                if event.type == "error":
                    logger.warning("keepalive rejected: %s", event.message)
                    return False
        except Exception as e:  # noqa: BLE001
            logger.debug("keepalive failed (%s); dropping socket", e)
            return False

    async def _shutdown_ws(self, ws: aiohttp.ClientWebSocketResponse, *, notify: bool) -> None:
        with contextlib.suppress(Exception):
            if notify and not ws.closed:
                await ws.send_str(json.dumps({"type": "session.close"}))
        with contextlib.suppress(Exception):
            await ws.close()


class Qwen3SynthesizeStream(tts_module.SynthesizeStream):
    """One agent turn: text in over the input channel, PCM out via the emitter."""

    def __init__(
        self,
        *,
        tts: tts_module.TTS,
        backend: Qwen3Backend,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._backend = backend
        self._opts = opts

    async def _run(self, output_emitter: tts_module.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )

        try:
            ws, applied_config = await self._backend.acquire(self._conn_options)
        except APIError:
            raise
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e

        config = self._backend.build_session_config(self._opts)
        state = _TurnState()
        discard = True

        async def _send() -> None:
            buffered = False
            try:
                async for data in self.input:
                    if isinstance(data, self.Flush):
                        if not buffered:
                            continue
                        await ws.send_str(json.dumps({"type": "input.done"}))
                        state.flushes_sent += 1
                        state.progress.set()
                        buffered = False
                        continue
                    self.mark_started()
                    await ws.send_str(json.dumps({"type": "input.text", "text": data}))
                    buffered = True
                if buffered:
                    await ws.send_str(json.dumps({"type": "input.done"}))
                    state.flushes_sent += 1
            finally:
                state.sender_finished.set()
                state.progress.set()

        async def _recv() -> None:
            # One segment per stream: livekit drops push_text() after a flush
            # and raises if the emitter's segment count disagrees with its own.
            # So a flush means "synthesize what's buffered", never "new
            # segment", and the segment closes via end_input() below.
            segment_open = False
            segment_bytes = 0
            sentence_offset = 0.0

            def open_segment() -> None:
                nonlocal segment_open
                output_emitter.start_segment(segment_id=request_id)
                segment_open = True

            while True:
                msg = await ws.receive()

                if msg.type is aiohttp.WSMsgType.BINARY:
                    if not segment_open:
                        open_segment()
                    output_emitter.push(msg.data)
                    segment_bytes += len(msg.data)
                    # Audio counts as progress. The server only sends
                    # session.done once the whole utterance is synthesized, so
                    # Progress extends the read window while synthesis is pending;
                    # a long utterance must not expire on a healthy socket, and
                    # the framework won't retry once audio has been pushed.
                    state.progress.set()
                    continue
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIConnectionError("Baseten closed the TTS websocket")
                if msg.type is aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError(f"websocket error: {ws.exception()}")
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue

                event = _decode_response(msg.data)
                etype = event.type

                if etype == "audio.start":
                    rate = event.sample_rate
                    if rate != SAMPLE_RATE:
                        logger.warning(
                            "server sample_rate=%d, expected %d; audio will be mistimed",
                            rate,
                            SAMPLE_RATE,
                        )
                    if not segment_open:
                        open_segment()
                    sentence_offset = segment_bytes / _BYTES_PER_SECOND

                elif etype == "audio.done":
                    if event.error:
                        logger.error(
                            "synthesis failed for sentence %s",
                            event.sentence_index,
                        )
                    elif self._opts.word_timestamps:
                        self._emit_timestamps(output_emitter, event.timestamp_info, sentence_offset)

                elif etype == "session.done":
                    state.flushes_done += 1
                    state.progress.set()
                    if state.settled:
                        return

                elif etype == "error":
                    raise APIStatusError(
                        message=event.message,
                        status_code=500,
                        request_id=request_id,
                        body=None,
                    )

        try:
            if config != applied_config:
                await ws.send_str(json.dumps({"type": "session.config", **config}))

            send_task = asyncio.create_task(_send(), name="qwen3-tts-send")
            recv_task = asyncio.create_task(_recv(), name="qwen3-tts-recv")
            try:
                await asyncio.gather(send_task, self._drain(recv_task, state))
            finally:
                await utils.aio.gracefully_cancel(send_task, recv_task)

            output_emitter.end_input()
            discard = False
        except APIError:
            raise
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await self._backend.release(
                ws, config, discard=discard, conn_options=self._conn_options
            )

    async def _drain(self, recv_task: asyncio.Task[None], state: _TurnState) -> None:
        """Unblock a receiver parked on ws.receive() with nothing left to read.

        Happens when the sender's last chunk needed no extra flush, so the
        final session.done already landed.
        """
        while True:
            # Clear before testing so a session.done landing mid-check sets the
            # flag again rather than being swallowed.
            state.progress.clear()
            if recv_task.done():
                await recv_task
                return
            if state.settled:
                return

            waiter = asyncio.ensure_future(state.progress.wait())
            try:
                done, _ = await asyncio.wait(
                    {recv_task, waiter},
                    timeout=(
                        self._conn_options.read_timeout
                        if state.flushes_sent > state.flushes_done
                        else None
                    ),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not waiter.done():
                    waiter.cancel()
            if not done:
                raise APITimeoutError("Qwen3 synthesis read timed out", phase="read")

    def _emit_timestamps(
        self,
        output_emitter: tts_module.AudioEmitter,
        timestamp_info: TimestampInfo | None,
        offset: float,
    ) -> None:
        """Rebase one sentence's word times onto the segment timeline."""
        if not timestamp_info:
            return
        alignment = timestamp_info.word_alignment
        if alignment is None:
            return
        timed = [
            TimedString(
                text=f"{word} ",
                start_time=offset + float(start),
                end_time=offset + float(end),
            )
            # strict=False on purpose: a ragged alignment should cost timestamps,
            # not drop the call mid-turn.
            for word, start, end in zip(
                alignment.words,
                alignment.word_start_times_seconds,
                alignment.word_end_times_seconds,
                strict=False,
            )
        ]
        if timed:
            output_emitter.push_timed_transcript(timed)


async def _control(
    model_endpoint: str, api_key: str | None, message: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    api_key = api_key or os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise ValueError("Pass `api_key` or set BASETEN_API_KEY.")
    async with aiohttp.ClientSession() as session:
        async with await utils.http_context.ws_connect(
            session,
            model_endpoint,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            headers={"Authorization": f"Api-Key {api_key}"},
            max_msg_size=_WS_MAX_MSG_SIZE,
        ) as ws:
            await ws.send_str(json.dumps(message))
            try:
                msg = await ws.receive(timeout=DEFAULT_API_CONNECT_OPTIONS.read_timeout)
            except TimeoutError as e:
                raise APITimeoutError("Qwen3 control read timed out", phase="read") from e
            if msg.type is not aiohttp.WSMsgType.TEXT:
                # .data is None on CLOSED, the close code on CLOSE, an exception
                # on ERROR — json.loads would raise a bare TypeError and hide
                # what actually went wrong.
                raise RuntimeError(f"unexpected reply to {message.get('type')!r}: {msg.type.name}")
            return _JSON_OBJECT.validate_json(msg.data)


async def register_voice(
    *,
    model_endpoint: str,
    name: str,
    ref_audio_path: str,
    ref_text: str | None = None,
    api_key: str | None = None,
    consent: str = "user_consent",
) -> dict[str, JsonValue]:
    """Clone a voice from 10-20s of clean speech. Omitting ref_text is less faithful.

    The clone lives on one replica's local disk and dies with the container.
    """
    with open(ref_audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    message: dict[str, JsonValue] = {
        "type": "voice.add",
        "name": name,
        "consent": consent,
        "audio_data": audio_b64,
        "audio_format": os.path.splitext(ref_audio_path)[1].lstrip(".").lower() or "wav",
    }
    if ref_text:
        message["ref_text"] = ref_text

    response = await _control(model_endpoint, api_key, message)
    if response.get("type") == "error" or not response.get("success"):
        raise RuntimeError(f"voice.add failed: {response.get('message') or response}")
    voice = response.get("voice", response)
    if not isinstance(voice, dict):
        raise APIConnectionError(
            "Qwen3 voice.add returned an invalid voice record", retryable=False
        )
    return voice


async def list_voices(*, model_endpoint: str, api_key: str | None = None) -> dict[str, JsonValue]:
    """`voices` is built-ins (empty on Base); `uploaded_voices` is per-replica."""
    response = await _control(model_endpoint, api_key, {"type": "voice.list"})
    if response.get("type") == "error":
        raise RuntimeError(f"voice.list failed: {response.get('message')}")
    return response
