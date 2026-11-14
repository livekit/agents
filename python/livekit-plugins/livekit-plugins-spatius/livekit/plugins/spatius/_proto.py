"""The ingress socket's protobuf messages: the client's are encoded, the server's decoded to
the fields the plugin reads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

# Message.type
_CONFIGURE_SESSION = 1
_CONFIRM_SESSION = 2
_AUDIO_INPUT = 3
_ERROR = 4
_INTERRUPT = 7
_EGRESS_LIVEKIT = 1


class ConfirmSession(TypedDict):
    type: Literal["confirm_session"]
    connection_id: str


class ServerError(TypedDict):
    type: Literal["error"]
    req_id: str
    code: int
    message: str


def _varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    out = bytearray()
    while value > 0x7F:
        out.append(value & 0x7F | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _int(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value) if value else b""


def _len(number: int, value: bytes, *, always: bool = False) -> bytes:
    if not value and not always:
        return b""
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _str(number: int, value: str) -> bytes:
    return _len(number, value.encode())


def _map(number: int, entries: Mapping[str, str]) -> bytes:
    return b"".join(
        _len(number, _str(1, key) + _str(2, value), always=True) for key, value in entries.items()
    )


def _envelope(kind: int, number: int, body: bytes) -> bytes:
    return _int(1, kind) + _len(number, body, always=True)


def configure_session(
    *,
    sample_rate: int,
    bitrate: int,
    audio_format: int,
    extra_params: Mapping[str, str],
    url: str,
    api_token: str,
    room_name: str,
    publisher_id: str,
    extra_attributes: Mapping[str, str],
    idle_timeout: int,
) -> bytes:
    """ClientConfigureSession with a LiveKit egress."""
    egress = (
        _str(1, url)
        + _str(4, room_name)
        + _str(5, publisher_id)
        + _map(6, extra_attributes)
        + _int(7, idle_timeout)
        + _str(8, api_token)
    )
    body = (
        _int(1, sample_rate)
        + _int(2, bitrate)
        + _int(3, audio_format)
        + _int(5, _EGRESS_LIVEKIT)
        + _len(6, egress, always=True)
        + _map(8, extra_params)
    )
    return _envelope(_CONFIGURE_SESSION, 2, body)


def audio_input(req_id: str, audio: bytes, *, end: bool) -> bytes:
    return _envelope(_AUDIO_INPUT, 4, _str(1, req_id) + _int(2, end) + _len(3, audio))


def interrupt(req_id: str) -> bytes:
    return _envelope(_INTERRUPT, 8, _str(1, req_id))


def _fields(data: bytes) -> dict[int, int | bytes]:
    """The last value of each field; ``ValueError`` for a truncated or unknown encoding."""
    fields: dict[int, int | bytes] = {}
    at = 0

    def varint() -> int:
        nonlocal at
        value = shift = 0
        while True:
            if at >= len(data) or shift > 63:
                raise ValueError("a truncated varint")
            byte = data[at]
            at += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    while at < len(data):
        key = varint()
        number, wire = key >> 3, key & 7
        if wire == 0:
            fields[number] = varint()
        elif wire == 2:
            length = varint()
            if at + length > len(data):
                raise ValueError("a truncated field")
            fields[number] = data[at : at + length]
            at += length
        elif wire in (1, 5):
            at += 8 if wire == 1 else 4
        else:
            raise ValueError(f"an unknown wire type {wire}")
    if at != len(data):
        raise ValueError("a truncated field")
    return fields


def _text(fields: dict[int, int | bytes], number: int) -> str:
    value = fields.get(number, b"")
    if not isinstance(value, bytes):
        raise ValueError(f"field {number} is not a string")
    return value.decode()


def decode(data: bytes) -> ConfirmSession | ServerError | None:
    """A server message the plugin reads, ``None`` for any other one."""
    message = _fields(data)
    kind = message.get(1, 0)
    if kind == _CONFIRM_SESSION:
        body = message.get(3, b"")
        if not isinstance(body, bytes):
            raise ValueError("server_confirm_session is not a message")
        return {"type": "confirm_session", "connection_id": _text(_fields(body), 1)}
    if kind == _ERROR:
        body = message.get(5, b"")
        if not isinstance(body, bytes):
            raise ValueError("server_error is not a message")
        error = _fields(body)
        code = error.get(3, 0)
        if not isinstance(code, int):
            raise ValueError("server_error.code is not an integer")
        return {
            "type": "error",
            "req_id": _text(error, 2),
            # int32: a negative code is sign-extended to 64 bits
            "code": code - (1 << 64) if code >= 1 << 63 else code,
            "message": _text(error, 4),
        }
    return None
