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

"""The Qwen3-TTS protocol `TTS` speaks for ``model="qwen3-tts"``.

Qwen3-TTS is: session.config -> input.text* -> input.done (a flush, not a close) -> audio.start /
PCM / audio.done / session.done. The session config is sticky for a socket's life, so a pool of
configured sockets carries one stream each.

Voices are clones, not presets: the Base checkpoint ships no built-in speakers. Register one first
(see `register_voice`), or pass ref_audio/ref_text to clone inline. The server keeps uploaded
voices on the container's local disk, so a runtime-registered voice is scoped to one replica and
lost on restart: bake it into the truss via REQUIRED_VOICES for anything beyond single-replica
tests.
"""

from __future__ import annotations

import base64
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict, assert_never

from pydantic import JsonValue, TypeAdapter, ValidationError

from livekit.agents import APIConnectOptions, APIError, http, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, TranscriptChunk

from .log import logger

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
_BYTES_PER_SECOND = SAMPLE_RATE * NUM_CHANNELS * 2
_WS_MAX_MSG_SIZE = 16 * 1024 * 1024

# the server drops a socket 30 s after its last message: a parked one is kept warm before, with
# an empty `input.done` (answered by a `session.done` of no sentences), which proves the session
# alive, not only the connection
_KEEPALIVE_INTERVAL = 15.0


@dataclass
class Qwen3Options:
    voice: str
    task_type: str
    language: str | None
    instructions: str | None
    max_new_tokens: int | None
    initial_codec_chunk_frames: int | None
    x_vector_only_mode: bool | None
    ref_audio: str | None
    ref_text: str | None
    word_timestamps: bool
    extra_config: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])


# Each message declares only the fields read.


class _WordAlignment(TypedDict, total=False):
    words: list[str]
    word_start_times_seconds: list[float]
    word_end_times_seconds: list[float]


class _TimestampInfo(TypedDict, total=False):
    word_alignment: _WordAlignment | None


class _AudioStart(TypedDict):
    type: Literal["audio.start"]
    sample_rate: NotRequired[int]


class _AudioDone(TypedDict):
    type: Literal["audio.done"]
    error: NotRequired[str | bool | None]
    sentence_index: NotRequired[int | None]
    timestamp_info: NotRequired[_TimestampInfo | None]


class _SessionDone(TypedDict):
    type: Literal["session.done"]


class _Error(TypedDict):
    type: Literal["error"]
    message: NotRequired[str]


_Message = bytes | _AudioStart | _AudioDone | _SessionDone | _Error


class _InputText(TypedDict):
    type: Literal["input.text"]
    text: str


class _InputDone(TypedDict):
    type: Literal["input.done"]


# session.config carries the plugin's fields and the caller's extra_config, merged last
_Command = _InputText | _InputDone | dict[str, JsonValue]


class Qwen3Pool(tts.WebSocketMultiplexer[_Message, _Command]):
    """Configured sockets, one stream each: re-dialing would add a connect and a config round
    trip to every agent response."""

    routed = False
    keepalive = tts.Keepalive(
        every=_KEEPALIVE_INTERVAL, frame=_InputDone(type="input.done"), only_while_unused=True
    )

    def __init__(self, *, model_endpoint: str, api_key: str, opts: Qwen3Options) -> None:
        self._model_endpoint = model_endpoint
        self._api_key = api_key
        self._opts = opts
        self._ref_audio_b64: str | None = None

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._model_endpoint,
            headers={"Authorization": f"Api-Key {self._api_key}"},
            max_message_size=_WS_MAX_MSG_SIZE,
        )

    def setup(self) -> dict[str, JsonValue]:
        """`split_granularity` is omitted on purpose: it is not in the streaming schema, and an
        unrecognized field invalidates the whole config."""
        o = self._opts
        config: dict[str, JsonValue] = {
            "type": "session.config",
            "task_type": o.task_type,
            "response_format": "pcm",
            "stream_audio": True,
            # progressive PCM is only offered at native speed
            "speed": 1.0,
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

        if ref_audio := self._ref_audio():
            config["ref_audio"] = ref_audio
            if o.ref_text:
                config["ref_text"] = o.ref_text
        if o.word_timestamps:
            # "sync" lands in audio.done, after that sentence's audio, so its offset is known
            # exactly; "async" arrives arbitrarily interleaved.
            config["timestamp_type"] = "word"
            config["timestamp_transport_strategy"] = "sync"
        config.update(o.extra_config)
        return config

    def route(self, message: _Message) -> str | None:
        if not isinstance(message, bytes) and message["type"] == "error":
            raise _error(message)
        return None

    def _ref_audio(self) -> str | None:
        ref = self._opts.ref_audio
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


class Qwen3SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    """Each segment is one flush: its text, then ``input.done``, answered by ``session.done``."""

    def __init__(
        self,
        *,
        tts: tts.TTS,
        pool: Qwen3Pool,
        word_timestamps: bool,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = pool
        self._word_timestamps = word_timestamps
        # the server answers the flushes in order: audio is the oldest one's
        self._open = deque[str]()
        # flushes sent and not yet answered: a parked socket's warm-keep answer is not a segment's
        self._flushes = 0
        self._audio_bytes = 0
        self._sentence_offset = 0.0

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                self._open.append(item.segment_id)
                return None
            case tts.SegmentText():
                return _InputText(type="input.text", text=f"{item.chunk.text} ")
            case tts.SegmentEnd():
                self._flushes += 1
                return _InputDone(type="input.done")
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if isinstance(message, bytes):
            if self._open:
                self._audio_bytes += len(message)
                self.report(tts.SegmentAudio(self._open[0], message))
        elif message["type"] == "audio.start":
            if (rate := message.get("sample_rate", SAMPLE_RATE)) != SAMPLE_RATE:
                logger.warning(
                    "server sample_rate=%d, expected %d; audio will be mistimed", rate, SAMPLE_RATE
                )
            self._sentence_offset = self._audio_bytes / _BYTES_PER_SECOND
        elif message["type"] == "audio.done":
            if message.get("error"):
                logger.error("synthesis failed for sentence %s", message.get("sentence_index"))
            elif self._word_timestamps and self._open:
                words = _words(message.get("timestamp_info"), self._sentence_offset)
                if words:
                    self.report(tts.SegmentWords(self._open[0], words))
        elif message["type"] == "session.done":
            if self._flushes and self._open:
                self._flushes -= 1
                self._audio_bytes = 0
                self.report(tts.SegmentDone(self._open.popleft()))
        elif message["type"] == "error":
            raise _error(message)
        else:
            assert_never(message)


def _words(info: _TimestampInfo | None, offset: float) -> list[TranscriptChunk]:
    """One sentence's word times, rebased onto the segment's timeline."""
    alignment = info.get("word_alignment") if info else None
    if not alignment:
        return []
    # strict=False on purpose: a ragged alignment should cost timestamps, not drop the call
    return [
        TranscriptChunk(text=f"{word} ", start_time=offset + start, end_time=offset + end)
        for word, start, end in zip(
            alignment.get("words", []),
            alignment.get("word_start_times_seconds", []),
            alignment.get("word_end_times_seconds", []),
            strict=False,
        )
    ]


def _error(message: _Error) -> APIError:
    return APIError("Baseten TTS returned an error", kind="server", detail=message.get("message"))


_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


async def _control(
    model_endpoint: str, api_key: str | None, message: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    api_key = api_key or os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise ValueError("Pass `api_key` or set BASETEN_API_KEY.")
    async with http.open() as session:
        ws = await session.ws_connect(
            model_endpoint,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            headers={"Authorization": f"Api-Key {api_key}"},
            max_msg_size=_WS_MAX_MSG_SIZE,
        )
        try:
            await ws.send_json(message)
            msg = await ws.receive(timeout=DEFAULT_API_CONNECT_OPTIONS.read_timeout)
            if msg.type is not http.WebSocketMessageType.TEXT:
                raise APIError(
                    f"Qwen3 sent no text reply to {message.get('type')}",
                    kind="protocol",
                    detail=msg.type.name,
                )
            try:
                return _JSON_OBJECT.validate_json(msg.data)
            except ValidationError as e:
                raise APIError(
                    f"Qwen3 sent a malformed reply to {message.get('type')}",
                    kind="protocol",
                    detail=str(e),
                ) from e
        finally:
            await ws.close()


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
        detail = str(response.get("message") or response)
        raise APIError("Qwen3 voice.add failed", kind="server", detail=detail)
    voice = response.get("voice", response)
    if not isinstance(voice, dict):
        raise APIError(
            "Qwen3 voice.add returned an invalid voice record", kind="protocol", detail=str(voice)
        )
    return voice


async def list_voices(*, model_endpoint: str, api_key: str | None = None) -> dict[str, JsonValue]:
    """`voices` is built-ins (empty on Base); `uploaded_voices` is per-replica."""
    response = await _control(model_endpoint, api_key, {"type": "voice.list"})
    if response.get("type") == "error":
        detail = str(response.get("message") or response)
        raise APIError("Qwen3 voice.list failed", kind="server", detail=detail)
    return response
