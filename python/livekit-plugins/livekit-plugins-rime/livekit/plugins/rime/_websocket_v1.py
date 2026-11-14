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
import binascii
import ipaddress
from dataclasses import dataclass
from typing import Literal, cast
from urllib.parse import unquote, urlsplit

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from pydantic import JsonValue, TypeAdapter, ValidationError
from rime_api import text_to_speech_pb2 as proto

from livekit.agents import APIError, APIStatusError, redacted_cause

from .models import is_mist_model

WebSocketProtocol = Literal["binary", "json"]
RimeAudioFormat = Literal[
    "audio/wav",
    "audio/mpeg",
    "audio/ogg;codecs=opus",
    "audio/pcm",
    "audio/pcmu",
    "audio/webm;codecs=opus",
]
BINARY_SUBPROTOCOL = "rime.v1.binary"
JSON_SUBPROTOCOL = "rime.v1.json"
PROTOCOL_VERSION = 1
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
NUM_CHANNELS = 1
DEFAULT_AUDIO_FORMAT: RimeAudioFormat = "audio/pcm"
SUPPORTED_AUDIO_FORMATS: frozenset[str] = frozenset(
    {
        "audio/wav",
        "audio/mpeg",
        "audio/ogg;codecs=opus",
        "audio/pcm",
        "audio/pcmu",
        "audio/webm;codecs=opus",
    }
)


@dataclass(frozen=True)
class SynthesisOptions:
    model: str
    speaker: str
    language: str
    audio_format: RimeAudioFormat = DEFAULT_AUDIO_FORMAT
    sampling_rate: int | None = None
    time_scale_factor: float | None = None
    pause_between_brackets: bool | None = None
    phonemize_between_brackets: bool | None = None


def validate_audio_format(audio_format: str) -> RimeAudioFormat:
    """Return a supported canonical Rime audio MIME type."""
    if audio_format not in SUPPORTED_AUDIO_FORMATS:
        choices = ", ".join(sorted(SUPPORTED_AUDIO_FORMATS))
        raise ValueError(
            f"unsupported Rime audio_format {audio_format!r}; choose one of: {choices}"
        )
    return cast(RimeAudioFormat, audio_format)


def emitter_mime_type(audio_format: RimeAudioFormat) -> str:
    return audio_format.partition(";")[0]


def subprotocol(protocol: WebSocketProtocol | str) -> str:
    if protocol == "binary":
        return BINARY_SUBPROTOCOL
    if protocol == "json":
        return JSON_SUBPROTOCOL
    raise ValueError('websocket_protocol must be "binary" or "json"')


def encode(protocol: WebSocketProtocol, request: proto.WebSocketRequest) -> bytes | str:
    """One request envelope as the negotiated encoding frames it."""
    if protocol == "binary":
        return request.SerializeToString()
    return json_format.MessageToJson(request, preserving_proto_field_name=False, indent=None)


def decode(protocol: WebSocketProtocol, frame: bytes | str) -> proto.WebSocketResponse | None:
    """One response envelope, or ``None`` for a frame or payload this client does not handle."""
    response = proto.WebSocketResponse()
    if protocol == "binary":
        if not isinstance(frame, bytes):
            return None
        try:
            response.ParseFromString(frame)
        except (DecodeError, TypeError) as cause:
            raise protocol_error("Rime v1 sent invalid protobuf") from redacted_cause(cause)
    else:
        if not isinstance(frame, str):
            return None
        try:
            envelope = _JSON_OBJECT.validate_json(frame)
        except (ValidationError, TypeError) as cause:
            raise protocol_error("Rime v1 sent invalid JSON", frame) from redacted_cause(cause)
        if not _validate_json_envelope(envelope):
            return None
        try:
            json_format.ParseDict(envelope, response, ignore_unknown_fields=True)
        except (json_format.ParseError, TypeError) as cause:
            raise protocol_error("Rime v1 sent invalid JSON", frame) from redacted_cause(cause)
    return response if response.WhichOneof("payload") is not None else None


_JSON_RESPONSE_PAYLOADS = ("ready", "started", "audio", "done", "cancelled", "error")


def _validate_json_envelope(envelope: dict[str, JsonValue]) -> bool:
    """Whether the envelope carries a payload this client handles, checked."""
    payloads = [name for name in _JSON_RESPONSE_PAYLOADS if name in envelope]
    if not payloads:
        return False
    if len(payloads) > 1:
        raise protocol_error("Rime v1 envelope must contain exactly one payload")

    payload = payloads[0]
    value = envelope[payload]
    if payload == "audio":
        if not isinstance(value, str):
            raise protocol_error("Rime v1 sent a non-string audio payload")
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as cause:
            raise protocol_error("Rime v1 sent invalid Base64 audio") from redacted_cause(cause)
    elif not isinstance(value, dict):
        raise protocol_error(f"Rime v1 sent a malformed {payload} event", str(value))
    return True


def protocol_error(message: str, payload: str | None = None) -> APIError:
    return APIError(message, kind="protocol", detail=payload)


def _is_loopback_host(hostname: str | None) -> bool:
    try:
        return hostname is not None and ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _is_trusted_rime_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    return normalized == "rime.ai" or normalized.endswith(".rime.ai")


def validate_endpoint_host(endpoint_url: str, *, allow_custom_endpoint: bool = False) -> None:
    """Reject endpoints that could receive Rime credentials without caller consent."""
    parts = urlsplit(endpoint_url)
    if not parts.netloc or parts.hostname is None:
        raise ValueError("Rime endpoint must be an absolute URL")
    if (
        not allow_custom_endpoint
        and not _is_loopback_host(parts.hostname)
        and not _is_trusted_rime_host(parts.hostname)
    ):
        raise ValueError(
            "Rime endpoint must use a trusted Rime host; "
            "set allow_custom_endpoint=True to send credentials to another host"
        )


def validate_websocket_url(websocket_url: str, *, allow_custom_endpoint: bool = False) -> None:
    """Validate a caller-supplied v1 WebSocket endpoint."""
    parts = urlsplit(websocket_url)
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        raise ValueError("Rime v1 websocket_url must be an absolute ws or wss URL")
    if parts.scheme == "ws" and not _is_loopback_host(parts.hostname):
        raise ValueError("Rime v1 websocket_url must use wss unless it uses a loopback IP address")
    validate_endpoint_host(websocket_url, allow_custom_endpoint=allow_custom_endpoint)


def model_endpoint_identity(websocket_url: str) -> tuple[str, str, int, str]:
    """Return the URL parts that identify the model behind an endpoint."""
    parts = urlsplit(websocket_url)
    assert parts.hostname is not None
    scheme = parts.scheme.lower()
    default_port = 443 if scheme == "wss" else 80
    return (
        scheme,
        parts.hostname.rstrip(".").lower(),
        parts.port or default_port,
        parts.path.rstrip("/"),
    )


def model_from_websocket_url(
    websocket_url: str, *, allow_custom_endpoint: bool = False
) -> str | None:
    """Return the model from /{model}/ws, or None for a dedicated /ws endpoint."""
    validate_websocket_url(websocket_url, allow_custom_endpoint=allow_custom_endpoint)
    path_segments = urlsplit(websocket_url).path.rstrip("/").split("/")
    if not path_segments or path_segments[-1] != "ws":
        raise ValueError("Rime v1 websocket_url path must end with /ws")
    if len(path_segments) < 2 or not path_segments[-2]:
        return None
    return unquote(path_segments[-2])


def start_payload(options: SynthesisOptions) -> proto.SynthesisRequest:
    audio_parameters = proto.AudioParameters(audio_format=options.audio_format)
    if options.sampling_rate is not None:
        audio_parameters.sampling_rate = options.sampling_rate
    if options.time_scale_factor is not None:
        audio_parameters.time_scale_factor = options.time_scale_factor

    request = proto.SynthesisRequest(
        speaker=options.speaker,
        language=options.language,
        text="",
        audio_parameters=audio_parameters,
    )
    if is_mist_model(options.model):
        if options.pause_between_brackets is not None:
            request.mist_parameters.pause_between_brackets = options.pause_between_brackets
        if options.phonemize_between_brackets is not None:
            request.mist_parameters.phonemize_between_brackets = options.phonemize_between_brackets
    return request


def request(context_id: str, payload: str, value: object = None) -> proto.WebSocketRequest:
    envelope = proto.WebSocketRequest(context_id=context_id)
    if payload == "start" and isinstance(value, proto.SynthesisRequest):
        envelope.start.CopyFrom(value)
    elif payload == "text" and isinstance(value, str):
        envelope.text = value
    elif payload == "end":
        envelope.end.SetInParent()
    elif payload == "cancel":
        envelope.cancel.SetInParent()
    else:
        raise ValueError(f"unsupported Rime v1 request payload {payload!r}")
    return envelope


def rime_error(error: proto.WebSocketError, *, fallback_request_id: str | None) -> APIError:
    kind = error.kind
    if not kind:
        return protocol_error("Rime v1 sent a malformed error")
    request_id = error.request_id if error.HasField("request_id") else fallback_request_id
    status_codes = {
        "invalid_input": 400,
        "unauthenticated": 401,
        "permission_denied": 403,
        "not_found": 404,
        "resource_exhausted": 429,
        "timeout": 504,
        "unavailable": 503,
        "unimplemented": 501,
        "internal": 500,
    }
    safe_kind = kind if kind in status_codes else "unknown"
    return APIStatusError(
        message="Rime v1 request failed",
        status_code=status_codes.get(kind, 500),
        request_id=request_id,
        body={"kind": safe_kind},
        # the server does not implement the request: a retry asks the same
        kind="invalid_request" if kind == "unimplemented" else None,
        detail=error.message or None,
    )
