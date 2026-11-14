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

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict, assert_never

from pydantic import Field

from livekit.agents import APIConnectOptions, APIError, tts
from livekit.agents.voice.io import TranscriptChunk

from ._stream_tts import StreamTTS

# Rime's WS3 socket: each segment is one context, ended by a flush. Each message declares only
# the fields read.


class _WordTimestamps(TypedDict):
    words: list[str]
    start: list[float]
    end: list[float]


class _Chunk(TypedDict):
    type: Literal["chunk"]
    contextId: str
    data: str


class _Timestamps(TypedDict):
    type: Literal["timestamps"]
    contextId: str
    word_timestamps: NotRequired[_WordTimestamps | None]


class _Done(TypedDict):
    type: Literal["done"]
    contextId: str


class _Error(TypedDict):
    type: Literal["error"]
    contextId: NotRequired[str]


_Message = Annotated[_Chunk | _Timestamps | _Done | _Error, Field(discriminator="type")]


class _Text(TypedDict):
    text: str
    contextId: str


class _Flush(TypedDict):
    operation: Literal["flush"]
    contextId: str


@dataclass(frozen=True)
class LegacySynthesisOptions:
    model: str
    websocket_url: str
    sample_rate: int


class LegacyWebSocketAdapter(tts.WebSocketMultiplexer[_Message, _Text | _Flush]):
    """Rime WS3 sockets bound to one endpoint, one stream at a time each."""

    max_session_duration = 300
    max_contexts = 1

    def __init__(self, *, websocket_url: str, api_key: str) -> None:
        self._websocket_url = websocket_url
        self._api_key = api_key

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._websocket_url, headers={"Authorization": f"Bearer {self._api_key}"}
        )

    def route(self, message: _Chunk | _Timestamps | _Done | _Error) -> str:
        if "contextId" not in message:
            # an error naming no context fails every stream on the socket
            raise _error()
        return message["contextId"]

    def stream(
        self,
        *,
        tts_instance: tts.TTS,
        options: LegacySynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> _LegacyWebSocketSynthesizeStream:
        if options.websocket_url != self._websocket_url:
            raise RuntimeError("Rime WS3 stream options do not match the active endpoint")
        return _LegacyWebSocketSynthesizeStream(
            tts_instance=tts_instance, mux=self, options=options, conn_options=conn_options
        )

    def update_endpoint(self, websocket_url: str) -> None:
        if websocket_url != self._websocket_url:
            self._websocket_url = websocket_url
            self.invalidate()


class _LegacyWebSocketSynthesizeStream(tts.WebSocketStream[_Message, _Text | _Flush]):
    """One LiveKit stream on a Rime WS3 socket, each segment its own context."""

    def __init__(
        self,
        *,
        tts_instance: tts.TTS,
        mux: LegacyWebSocketAdapter,
        options: LegacySynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=StreamTTS(tts_instance, model=options.model), conn_options=conn_options
        )
        self.multiplexer = mux

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Text | _Flush | None:
        match item:
            case tts.SegmentText():
                return {"text": item.chunk.text + " ", "contextId": item.segment_id}
            case tts.SegmentEnd():
                return {"operation": "flush", "contextId": item.segment_id}
            case tts.SegmentStart() | tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Chunk | _Timestamps | _Done | _Error) -> None:
        if message["type"] == "chunk":
            audio = base64.b64decode(message["data"])
            self.report(tts.SegmentAudio(message["contextId"], audio))
        elif message["type"] == "timestamps":
            timestamps = message.get("word_timestamps")
            words = (
                [
                    TranscriptChunk(text=word + " ", start_time=start, end_time=end)
                    for word, start, end in zip(
                        timestamps["words"], timestamps["start"], timestamps["end"], strict=False
                    )
                ]
                if timestamps
                else []
            )
            self.report(tts.SegmentWords(message["contextId"], words))
        elif message["type"] == "done":
            self.report(tts.SegmentDone(message["contextId"]))
        elif message["type"] == "error":
            raise _error()
        else:
            assert_never(message)


def _error() -> APIError:
    # auth fails the handshake, so an in-band error rejects the request itself
    return APIError("Rime WebSocket request failed", kind="invalid_request")
