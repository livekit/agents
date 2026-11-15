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

"""Rime's WebSocket v1 wire: protobuf envelopes, sent as binary or as proto3 JSON."""

from __future__ import annotations

import base64
import binascii
import ipaddress
from typing import Literal, TypeGuard, get_args
from urllib.parse import unquote, urlsplit

from google.protobuf import json_format
from google.protobuf.message import DecodeError
from pydantic import JsonValue, TypeAdapter, ValidationError
from rime_api import text_to_speech_pb2 as proto

from livekit.agents import APIError, APIStatusError, redacted_cause, tts
from livekit.agents.types import ErrorKind

WebSocketProtocol = Literal["binary", "json"]
RimeAudioFormat = Literal[
    "audio/wav",
    "audio/mpeg",
    "audio/ogg;codecs=opus",
    "audio/pcm",
    "audio/pcmu",
    "audio/webm;codecs=opus",
]
SUBPROTOCOLS: dict[WebSocketProtocol, str] = {"binary": "rime.v1.binary", "json": "rime.v1.json"}
PROTOCOL_VERSION = 1
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
# the payloads this client reads; any other is skipped
_JSON_RESPONSE_PAYLOADS = ("ready", "started", "audio", "done", "cancelled", "error")
# Rime's error kinds, as HTTP statuses and the kind a caller branches on
_STATUSES: dict[str, tuple[int, ErrorKind | None]] = {
    "invalid_input": (400, None),
    "unauthenticated": (401, None),
    "permission_denied": (403, None),
    "not_found": (404, None),
    "resource_exhausted": (429, None),
    "timeout": (504, None),
    "unavailable": (503, None),
    # the server does not implement the request: a retry asks the same
    "unimplemented": (501, "invalid_request"),
    "internal": (500, None),
}


def is_audio_format(audio_format: str) -> TypeGuard[RimeAudioFormat]:
    return audio_format in get_args(RimeAudioFormat)


def encode(protocol: WebSocketProtocol, request: proto.WebSocketRequest) -> bytes | str:
    """One request envelope as the negotiated encoding frames it."""
    if protocol == "binary":
        return request.SerializeToString()
    return json_format.MessageToJson(request, preserving_proto_field_name=False, indent=None)


def decode(protocol: WebSocketProtocol, frame: bytes | str) -> proto.WebSocketResponse | None:
    """One response envelope, or ``None`` for a frame or payload this client does not read."""
    response = proto.WebSocketResponse()
    if protocol == "binary":
        if not isinstance(frame, bytes):
            tts.log_skipped("Rime v1", "a text frame")
            return None
        try:
            response.ParseFromString(frame)
        except (DecodeError, TypeError) as cause:
            raise protocol_error("Rime v1 sent invalid protobuf") from redacted_cause(cause)
    else:
        if not isinstance(frame, str):
            tts.log_skipped("Rime v1", "a binary frame")
            return None
        try:
            envelope = _JSON_OBJECT.validate_json(frame)
        except (ValidationError, TypeError) as cause:
            raise protocol_error("Rime v1 sent invalid JSON", frame) from redacted_cause(cause)
        if not _check_json_envelope(envelope):
            return None
        try:
            json_format.ParseDict(envelope, response, ignore_unknown_fields=True)
        except (json_format.ParseError, TypeError) as cause:
            raise protocol_error("Rime v1 sent invalid JSON", frame) from redacted_cause(cause)
    payload = response.WhichOneof("payload")
    if payload not in _JSON_RESPONSE_PAYLOADS:
        # a payload Rime added since (unknown to the installed client, or not read): named once
        tts.log_skipped("Rime v1", str(payload))
        return None
    return response


def _check_json_envelope(envelope: dict[str, JsonValue]) -> bool:
    """Whether the envelope carries a payload this client reads, checked."""
    payloads = [name for name in _JSON_RESPONSE_PAYLOADS if name in envelope]
    if not payloads:
        tts.log_skipped("Rime v1", ",".join(sorted(envelope)) or "{}")
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


def validate_endpoint_host(endpoint_url: str, *, allow_custom_endpoint: bool) -> None:
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


def model_from_websocket_url(websocket_url: str, *, allow_custom_endpoint: bool) -> str | None:
    """The model a ``/{model}/ws`` path names, ``None`` for a ``/ws`` endpoint; the URL
    validated."""
    parts = urlsplit(websocket_url)
    if parts.scheme not in ("ws", "wss") or not parts.netloc:
        raise ValueError("Rime websocket_url must be an absolute ws or wss URL")
    if parts.scheme == "ws" and not _is_loopback_host(parts.hostname):
        raise ValueError("Rime websocket_url must use wss unless it uses a loopback IP address")
    validate_endpoint_host(websocket_url, allow_custom_endpoint=allow_custom_endpoint)
    path_segments = parts.path.rstrip("/").split("/")
    if path_segments[-1] != "ws":
        raise ValueError("Rime websocket_url path must end with /ws")
    if len(path_segments) < 2 or not path_segments[-2]:
        return None
    return unquote(path_segments[-2])


def rime_error(error: proto.WebSocketError, *, fallback_request_id: str | None) -> APIError:
    if not error.kind:
        return protocol_error("Rime v1 sent a malformed error")
    request_id = error.request_id if error.HasField("request_id") else fallback_request_id
    status, kind = _STATUSES.get(error.kind, (500, None))
    return APIStatusError(
        message="Rime v1 request failed",
        status_code=status,
        request_id=request_id,
        body={"kind": error.kind if error.kind in _STATUSES else "unknown"},
        kind=kind,
        detail=error.message or None,
    )
