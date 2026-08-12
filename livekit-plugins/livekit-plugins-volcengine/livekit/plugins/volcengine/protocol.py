from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class MessageType(IntEnum):
    """Message types used by the Volcengine V3 binary protocol."""

    FULL_CLIENT_REQUEST = 0x1
    FULL_SERVER_RESPONSE = 0x9
    AUDIO_ONLY_SERVER_RESPONSE = 0xB
    ERROR_RESPONSE = 0xF


class Event(IntEnum):
    """Lifecycle and data events used by bidirectional TTS."""

    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52
    START_SESSION = 100
    CANCEL_SESSION = 101
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_CANCELED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    TASK_REQUEST = 200
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352


class ProtocolError(ValueError):
    """Raised when a Volcengine binary frame is malformed or unsupported."""


@dataclass(frozen=True)
class ServerMessage:
    """Decoded Volcengine server frame."""

    message_type: MessageType
    event: Event | None
    payload: bytes
    session_id: str | None = None
    connection_id: str | None = None
    error_code: int | None = None

    def json_payload(self) -> dict[str, Any]:
        """Decode a JSON payload and require an object at the top level."""
        if not self.payload:
            return {}
        try:
            value = json.loads(self.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError("Volcengine response contains invalid JSON") from error
        if not isinstance(value, dict):
            raise ProtocolError("Volcengine JSON payload must be an object")
        return value


_HEADER = bytes([0x11, 0x14, 0x10, 0x00])
_CONNECTION_EVENTS = {
    Event.START_CONNECTION,
    Event.FINISH_CONNECTION,
    Event.CONNECTION_STARTED,
    Event.CONNECTION_FAILED,
    Event.CONNECTION_FINISHED,
}


def build_client_message(
    event: Event, payload: dict[str, Any] | None = None, *, session_id: str | None = None
) -> bytes:
    """Build a full-client request frame for a connection or session event."""
    payload_bytes = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    frame_parts = [_HEADER, struct.pack(">i", event)]
    if event not in {Event.START_CONNECTION, Event.FINISH_CONNECTION}:
        if not session_id:
            raise ProtocolError(f"session_id is required for event {event.name}")
        session_bytes = session_id.encode("utf-8")
        frame_parts.extend((struct.pack(">I", len(session_bytes)), session_bytes))
    frame_parts.extend((struct.pack(">I", len(payload_bytes)), payload_bytes))
    return b"".join(frame_parts)


def parse_server_message(frame: bytes) -> ServerMessage:
    """Parse a Volcengine V3 full-server, audio, or error frame."""
    if len(frame) < 4:
        raise ProtocolError("Volcengine frame is shorter than its header")
    if frame[0] >> 4 != 1:
        raise ProtocolError("Unsupported Volcengine protocol version")

    header_size = (frame[0] & 0x0F) * 4
    if header_size < 4 or len(frame) < header_size:
        raise ProtocolError("Invalid Volcengine header size")
    try:
        message_type = MessageType(frame[1] >> 4)
    except ValueError as error:
        raise ProtocolError("Unsupported Volcengine message type") from error

    offset = header_size
    if message_type == MessageType.ERROR_RESPONSE:
        error_code, offset = _read_uint32(frame, offset)
        payload, _ = _read_bytes(frame, offset)
        return ServerMessage(message_type, None, payload, error_code=error_code)

    if frame[1] & 0x0F != 0x4:
        raise ProtocolError("Volcengine server frame does not contain an event")
    event_number, offset = _read_int32(frame, offset)
    try:
        event = Event(event_number)
    except ValueError as error:
        raise ProtocolError(f"Unsupported Volcengine event {event_number}") from error

    identifier, offset = _read_text(frame, offset)
    payload, _ = _read_bytes(frame, offset)
    if event in _CONNECTION_EVENTS:
        return ServerMessage(message_type, event, payload, connection_id=identifier)
    return ServerMessage(message_type, event, payload, session_id=identifier)


def _read_uint32(frame: bytes, offset: int) -> tuple[int, int]:
    if len(frame) < offset + 4:
        raise ProtocolError("Volcengine frame is missing a uint32 field")
    return struct.unpack_from(">I", frame, offset)[0], offset + 4


def _read_int32(frame: bytes, offset: int) -> tuple[int, int]:
    if len(frame) < offset + 4:
        raise ProtocolError("Volcengine frame is missing an int32 field")
    return struct.unpack_from(">i", frame, offset)[0], offset + 4


def _read_bytes(frame: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _read_uint32(frame, offset)
    value = frame[offset : offset + size]
    if len(value) != size:
        raise ProtocolError("Volcengine frame payload is incomplete")
    return value, offset + size


def _read_text(frame: bytes, offset: int) -> tuple[str, int]:
    value, offset = _read_bytes(frame, offset)
    try:
        return value.decode("utf-8"), offset
    except UnicodeDecodeError as error:
        raise ProtocolError("Volcengine frame identifier is not UTF-8") from error
