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

from __future__ import annotations

import os
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, ErrorKind, NotGivenOr
from livekit.agents.utils import is_given

DEFAULT_BASE_URL = "https://api.bland.ai/v2"
DEFAULT_VOICE_ID = "2f29fdbb-c55e-4add-9c7c-93437ebf379d"
DEFAULT_SAMPLE_RATE = 48000
NUM_CHANNELS = 1

SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

# Bland ends a session that goes 60s without a client message: a parked socket closes before.
_IDLE_TIMEOUT = 50.0

# What each error code means. An unlisted code (a synthesis failure, a busy concurrency pool) is
# the provider's failure and worth another attempt.
_ERROR_KINDS: dict[str, ErrorKind] = {
    "AUTH_FAILED": "auth",
    "INSUFFICIENT_CREDITS": "quota",
    "ORG_DELETED": "auth",
    "ORG_STRIPE_OVERDUE": "quota",
    "ORG_SUSPENDED": "auth",
    "USER_BANNED": "auth",
    "already_initialized": "invalid_request",
    "context_overflow": "invalid_request",
    "init_required": "invalid_request",
    "insufficient_credits": "quota",
    "invalid_message": "invalid_request",
    "invalid_request": "invalid_request",
    "unsupported_encoding": "invalid_request",
    "unsupported_sample_rate": "invalid_request",
    "unsupported_voice": "invalid_request",
    "voice_not_found": "invalid_request",
    "voice_not_live": "invalid_request",
}


@dataclass
class _TTSOptions:
    voice_id: str
    sample_rate: int
    expressiveness: NotGivenOr[float]
    stability: NotGivenOr[float]
    base_url: str


# Bland's realtime socket: `init` fixes the voice and format for the socket's life, then one turn
# (a context) at a time: a `speak` for a new context preempts the active one. Binary audio
# belongs to the turn between its utterance_start and utterance_end. Each message declares only
# the fields read.


class _Ready(TypedDict):
    type: Literal["ready"]
    encoding: NotRequired[str]
    sample_rate: NotRequired[int]


class _UtteranceStart(TypedDict):
    type: Literal["utterance_start"]
    context_id: str


class _UtteranceEnd(TypedDict):
    type: Literal["utterance_end"]
    context_id: str
    reason: str


class _Error(TypedDict):
    type: Literal["error"]
    context_id: NotRequired[str]
    code: NotRequired[str]
    message: NotRequired[str]


_Message = bytes | _Ready | _UtteranceStart | _UtteranceEnd | _Error
# the turn endings known to fail it; any other (complete, one Bland added since) ends it as said
_FAILED = frozenset({"cancelled", "error", "failed"})


class _Audio(TypedDict):
    encoding: Literal["pcm_s16le"]
    sample_rate: int


class _Controls(TypedDict, total=False):
    expressiveness: float
    stability: float


class _Init(TypedDict):
    type: Literal["init"]
    voice: str
    audio: _Audio
    controls: NotRequired[_Controls]


class _Speak(TypedDict):
    type: Literal["speak"]
    context_id: str
    text: str


class _Turn(TypedDict):
    type: Literal["end_of_turn", "cancel"]
    context_id: str


class _Close(TypedDict):
    type: Literal["close"]


class _SynthesizeBody(TypedDict):
    text: str
    voice: str
    audio: _Audio
    controls: NotRequired[_Controls]


_Command = _Init | _Speak | _Turn | _Close


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        expressiveness: NotGivenOr[float] = NOT_GIVEN,
        stability: NotGivenOr[float] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        streaming: bool = True,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """Create a new instance of the Bland TTS.

        Args:
            voice_id: Bland voice UUID; names are not accepted. Defaults to a ``BTTS_V3``
                voice. ``BTTS_V2`` voices work, but the controls below are calibrated for
                ``BTTS_V3``.
            sample_rate: Output sample rate in Hz, one of 8000, 16000, 24000, 44100, 48000.
                Defaults to 48000, the rate ``BTTS_V3`` renders natively.
            expressiveness: 0.0-1.0. Higher is more varied intonation.
            stability: 0.0-1.0. Higher is more consistent between renders.
            api_key: Bland API key. Falls back to the ``BLAND_API_KEY`` environment variable.
            base_url: Override the Bland API base URL.
            streaming: Stream text into one realtime WebSocket session, which is what a
                voice agent wants: audio starts before the sentence is finished, and a
                barge-in cancels the turn in place. Set False to synthesize each
                utterance with a single HTTP request instead — no session is held open,
                and no concurrency slot with it, which suits a pipeline that speaks
                rarely. ``synthesize()`` uses HTTP either way.
            http_session: Optional ``http.HttpSession`` to reuse.
        """
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}, got {sample_rate}")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=streaming),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        bland_api_key = api_key if is_given(api_key) else os.environ.get("BLAND_API_KEY")
        if not bland_api_key:
            raise ValueError(
                "Bland API key is required, either as `api_key` argument or "
                "`BLAND_API_KEY` environment variable"
            )

        self._api_key = bland_api_key
        self._opts = _TTSOptions(
            voice_id=voice_id,
            sample_rate=sample_rate,
            expressiveness=expressiveness,
            stability=stability,
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session.with_request_id("x-request-id") if http_session else None
        self._streams = weakref.WeakSet[SynthesizeStream]()
        # an open session holds a concurrency slot for as long as it lives: none is opened when
        # streaming is off
        self._mux = _Mux(self._api_key, self._opts) if streaming else None

    @property
    def provider(self) -> str:
        return "Bland"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session().with_request_id("x-request-id")
        return self._session

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        expressiveness: NotGivenOr[float] = NOT_GIVEN,
        stability: NotGivenOr[float] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            changed = False
            if is_given(voice_id):
                self._opts.voice_id = voice_id
                changed = True
            if is_given(expressiveness):
                self._opts.expressiveness = expressiveness
                changed = True
            if is_given(stability):
                self._opts.stability = stability
                changed = True

        if changed and self._mux is not None:
            # `init` fixes the voice and controls for the life of a session
            self._mux.invalidate()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            get_session=self._ensure_session,
            api_key=self._api_key,
            opts=replace(self._opts),
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        if self._mux is None:
            raise RuntimeError(
                "streaming is disabled on this Bland TTS instance; construct it with "
                "`streaming=True`, or wrap it in a `tts.StreamAdapter`"
            )
        stream = SynthesizeStream(mux=self._mux, tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        if self._mux is not None:
            self._mux.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        if self._mux is not None:
            await self._mux.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a complete string over HTTP."""

    def __init__(
        self,
        *,
        get_session: Callable[[], http.HttpSession],
        api_key: str,
        opts: _TTSOptions,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._get_session = get_session
        self._api_key = api_key

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        body = _SynthesizeBody(
            text=self._input_text,
            voice=self._opts.voice_id,
            audio=_Audio(encoding="pcm_s16le", sample_rate=self._opts.sample_rate),
        )
        if controls := _controls(self._opts):
            body["controls"] = controls

        try:
            async with self._get_session().post(
                f"{self._opts.base_url}/tts",
                headers={
                    "authorization": self._api_key,
                    "content-type": "application/json",
                },
                json=body,
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=resp.headers.get("x-request-id") or utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                output_emitter.flush()
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class _Mux(tts.WebSocketMultiplexer[_Message, _Command]):
    """Bland sessions, each carrying one stream at a time: its binary audio names no turn. A turn
    a barge-in left is cancelled in place, and the session speaks the next one once Bland ended
    it; a session the pool closes is closed on Bland's side first."""

    routed = False
    idle_timeout = _IDLE_TIMEOUT
    awaits_ready = True

    def __init__(self, api_key: str, opts: _TTSOptions) -> None:
        self._api_key = api_key
        self._opts = opts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        base_url = self._opts.base_url
        url = base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return tts.WebSocketTarget(
            f"{url}/tts/ws",
            headers={"Authorization": f"Bearer {self._api_key}"},
            request_id_header="x-request-id",
        )

    def setup(self) -> _Init:
        init = _Init(
            type="init",
            voice=self._opts.voice_id,
            audio=_Audio(encoding="pcm_s16le", sample_rate=self._opts.sample_rate),
        )
        if controls := _controls(self._opts):
            init["controls"] = controls
        return init

    def ready(self, message: _Message) -> bool:
        if isinstance(message, bytes) or message["type"] != "ready":
            return False
        # a field the acknowledgement leaves out is the format asked for
        if message.get("encoding", "pcm_s16le") != "pcm_s16le" or (
            message.get("sample_rate", self._opts.sample_rate) != self._opts.sample_rate
        ):
            raise APIError(
                "Bland acknowledged an unexpected audio format",
                body=message,
                kind="invalid_request",
            )
        return True

    def route(self, message: _Message) -> str | None:
        if not isinstance(message, bytes) and message["type"] == "error":
            # the session's own error, before it is ready
            raise _api_error(message)
        return None

    def close_context(self, context_id: str) -> _Turn:
        return _Turn(type="cancel", context_id=context_id)

    def context_ended(self, message: _Message) -> str | None:
        if isinstance(message, bytes) or message["type"] != "utterance_end":
            return None
        return message["context_id"]

    def goodbye(self) -> _Close:
        # Bland settles the session and closes it, so it holds no concurrency slot
        return _Close(type="close")


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    """Each segment is one Bland turn. Bland accumulates the text deltas server-side and picks its
    own synthesis boundaries, so text goes out as it arrives; a new turn would preempt the
    active one, so the next segment waits for the one before to end."""

    connection_per_segment = True

    def __init__(self, *, mux: _Mux, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = mux
        # the one turn in progress: the binary audio routed here is its
        self._turn = ""

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                self._turn = item.segment_id
                return None
            case tts.EndOfText():
                return None
            case tts.SegmentText():
                return _Speak(type="speak", context_id=item.segment_id, text=f"{item.chunk.text} ")
            case tts.SegmentEnd():
                # Bland holds a short tail back waiting for more context; this releases it
                return _Turn(type="end_of_turn", context_id=item.segment_id)
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        if isinstance(message, bytes):
            self.report(tts.SegmentAudio(self._turn, message))
        elif message["type"] == "utterance_end":
            if message["context_id"] != self._turn:
                return  # a turn this stream already left
            if message["reason"] in _FAILED:
                raise APIError(
                    f"Bland turn ended as {message['reason']}", kind="server", body=message
                )
            self.report(tts.SegmentDone(message["context_id"]))
        elif message["type"] == "error":
            raise _api_error(message)
        elif message["type"] == "ready" or message["type"] == "utterance_start":
            pass
        else:
            assert_never(message)


def _controls(opts: _TTSOptions) -> _Controls:
    controls = _Controls()
    if is_given(opts.expressiveness):
        controls["expressiveness"] = opts.expressiveness
    if is_given(opts.stability):
        controls["stability"] = opts.stability
    return controls


def _api_error(message: _Error) -> APIError:
    code = message.get("code")
    return APIError(
        f"Bland returned an error ({code})" if code else "Bland returned an error",
        body=message,
        kind=_ERROR_KINDS.get(code, "server") if code else "invalid_request",
        detail=message.get("message"),
    )
