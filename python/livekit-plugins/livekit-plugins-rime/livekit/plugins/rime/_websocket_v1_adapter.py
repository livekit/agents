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

from dataclasses import dataclass
from typing import Any

from rime_api import text_to_speech_pb2 as proto

from livekit.agents import APIConnectionError, APIConnectOptions, APIError, http, tts
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import _websocket_v1 as v1
from ._stream_tts import StreamTTS

_Response = proto.WebSocketResponse
# decode() reads protobuf envelopes: the generic Inbound, which the base validates with
# pydantic, stays Any
_Inbound = Any
_Frame = bytes | str


@dataclass(frozen=True)
class V1SynthesisOptions:
    """Model settings before conversion to the WebSocket v1 wire format."""

    model: str
    speaker: str
    language: NotGivenOr[str] = NOT_GIVEN
    audio_format: v1.RimeAudioFormat = v1.DEFAULT_AUDIO_FORMAT
    sampling_rate: NotGivenOr[int] = NOT_GIVEN
    time_scale_factor: NotGivenOr[float] = NOT_GIVEN
    pause_between_brackets: NotGivenOr[bool] = NOT_GIVEN
    phonemize_between_brackets: NotGivenOr[bool] = NOT_GIVEN


class WebSocketV1Adapter(http.Multiplexer[_Inbound, _Frame]):
    """Rime WebSocket v1 sockets, one stream at a time each: every segment is one Rime context,
    started with its first text and ended with its last."""

    max_session_duration = 300
    max_contexts = 1

    def __init__(
        self,
        *,
        websocket_v1_url: str,
        websocket_protocol: v1.WebSocketProtocol,
        api_key: str,
        allow_custom_endpoint: bool = False,
    ) -> None:
        v1.validate_websocket_url(websocket_v1_url, allow_custom_endpoint=allow_custom_endpoint)
        self._subprotocol = v1.subprotocol(websocket_protocol)
        self._websocket_v1_url = websocket_v1_url
        self.protocol: v1.WebSocketProtocol = websocket_protocol
        self._api_key = api_key
        self._allow_custom_endpoint = allow_custom_endpoint

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        return http.SocketTarget(
            self._websocket_v1_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            protocols=[self._subprotocol],
        )

    def decode(self, frame: bytes | str) -> _Response | None:
        return v1.decode(self.protocol, frame)

    def route(self, message: _Response) -> str | None:
        if message.context_id:
            return message.context_id
        payload = message.WhichOneof("payload")
        if payload == "error":
            raise v1.rime_error(message.error, fallback_request_id=None)
        if payload != "ready":
            raise v1.protocol_error(f"Rime v1 sent a {payload} event naming no context")
        if message.ready.protocol != v1.PROTOCOL_VERSION:
            raise v1.protocol_error("Rime v1 reported an unsupported protocol version")
        return None

    def close_context(self, context_id: str) -> _Frame:
        return v1.encode(self.protocol, v1.request(context_id, "cancel"))

    def stream(
        self,
        *,
        tts_instance: tts.TTS,
        options: V1SynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> _WebSocketV1SynthesizeStream:
        if not is_given(options.language):
            raise APIError("Rime v1 requires a language", kind="invalid_request")
        protocol_options = v1.SynthesisOptions(
            model=options.model,
            speaker=options.speaker,
            language=options.language,
            audio_format=options.audio_format,
            sampling_rate=options.sampling_rate if is_given(options.sampling_rate) else None,
            time_scale_factor=options.time_scale_factor
            if is_given(options.time_scale_factor)
            else None,
            pause_between_brackets=options.pause_between_brackets
            if is_given(options.pause_between_brackets)
            else None,
            phonemize_between_brackets=options.phonemize_between_brackets
            if is_given(options.phonemize_between_brackets)
            else None,
        )
        return _WebSocketV1SynthesizeStream(
            tts_instance=tts_instance, mux=self, options=protocol_options, conn_options=conn_options
        )

    def update_endpoint(self, websocket_v1_url: str, *, model_changed: bool) -> None:
        """Update the connection URL after validating its model binding."""
        v1.validate_websocket_url(
            websocket_v1_url, allow_custom_endpoint=self._allow_custom_endpoint
        )
        model_endpoint_changed = v1.model_endpoint_identity(
            websocket_v1_url
        ) != v1.model_endpoint_identity(self._websocket_v1_url)
        if model_changed and not model_endpoint_changed:
            raise ValueError("model cannot change without changing websocket_url")
        if websocket_v1_url == self._websocket_v1_url:
            return

        self._websocket_v1_url = websocket_v1_url
        self.invalidate()


class _WebSocketV1SynthesizeStream(tts.SocketStream[_Inbound, _Frame]):
    """One LiveKit stream on a Rime WebSocket v1 socket, each segment its own Rime context."""

    def __init__(
        self,
        *,
        tts_instance: tts.TTS,
        mux: WebSocketV1Adapter,
        options: v1.SynthesisOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=StreamTTS(tts_instance, model=options.model), conn_options=conn_options
        )
        self.multiplexer = mux
        self.mime_type = v1.emitter_mime_type(options.audio_format)
        self._protocol = mux.protocol
        self._start = v1.start_payload(options)
        self._request_ids: dict[str, str] = {}

    def encode(self, item: tts.SegmentInput | http.EndOfInput) -> list[_Frame] | _Frame | None:
        if isinstance(item, tts.SegmentStart):
            return self._frame(item.segment_id, "start", self._start)
        if isinstance(item, tts.SegmentText):
            return self._frame(item.segment_id, "text", item.chunk.text + " ")
        if isinstance(item, tts.SegmentEnd):
            return self._frame(item.segment_id, "end")
        return None

    def on_message(self, message: _Response) -> tts.SegmentOutput | None:
        context_id = message.context_id
        payload = message.WhichOneof("payload")
        if payload == "audio":
            return tts.SegmentAudio(context_id, message.audio)
        if payload == "done":
            self._request_ids.pop(context_id, None)
            return tts.SegmentDone(context_id)
        if payload == "error":
            request_id = self._request_ids.get(context_id)
            raise v1.rime_error(message.error, fallback_request_id=request_id)
        if payload == "cancelled":
            raise APIConnectionError("Rime v1 cancelled an active context unexpectedly")
        if payload != "started" or not message.started.request_id:
            raise v1.protocol_error(f"Unexpected Rime v1 event {payload!r}")
        self._request_ids[context_id] = message.started.request_id
        return None

    def _frame(self, context_id: str, payload: str, value: object = None) -> _Frame:
        return v1.encode(self._protocol, v1.request(context_id, payload, value))
