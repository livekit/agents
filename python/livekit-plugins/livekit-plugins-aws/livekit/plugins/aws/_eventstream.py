"""AWS event-stream framing: a prelude (total and header lengths, CRC32), typed headers, the
payload and a CRC32 of the whole message."""

from __future__ import annotations

import struct
import zlib

_PRELUDE = struct.Struct(">III")
_CRC = struct.Struct(">I")
_STRING = 7
# the value size of each fixed-width header type; 6 (bytes) and 7 (string) carry a length
_FIXED = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}


def encode(headers: dict[str, str], payload: bytes) -> bytes:
    """One message with string headers."""
    encoded = b"".join(
        bytes([len(name)])
        + name.encode()
        + bytes([_STRING])
        + struct.pack(">H", len(value))
        + value.encode()
        for name, value in headers.items()
    )
    total = _PRELUDE.size + len(encoded) + len(payload) + _CRC.size
    prelude = struct.pack(">II", total, len(encoded))
    message = prelude + _CRC.pack(zlib.crc32(prelude)) + encoded + payload
    return message + _CRC.pack(zlib.crc32(message))


def decode(frame: bytes) -> tuple[dict[str, str], bytes]:
    """A message's string headers and its payload; ``ValueError`` for a broken message."""
    if len(frame) < _PRELUDE.size + _CRC.size:
        raise ValueError(f"an event-stream message of {len(frame)} bytes")
    total, headers_length, prelude_crc = _PRELUDE.unpack_from(frame)
    if total != len(frame):
        raise ValueError(f"an event-stream message of {len(frame)} bytes declares {total}")
    if prelude_crc != zlib.crc32(frame[:8]):
        raise ValueError("an event-stream prelude fails its checksum")
    if _CRC.unpack_from(frame, total - _CRC.size)[0] != zlib.crc32(frame[: total - _CRC.size]):
        raise ValueError("an event-stream message fails its checksum")
    end = _PRELUDE.size + headers_length
    if end > total - _CRC.size:
        raise ValueError("event-stream headers overrun the message")
    headers: dict[str, str] = {}
    at = _PRELUDE.size
    try:
        while at < end:
            name_length = frame[at]
            name = frame[at + 1 : at + 1 + name_length].decode()
            kind = frame[at + 1 + name_length]
            at += 2 + name_length
            if kind in _FIXED:
                at += _FIXED[kind]
                continue
            if kind not in (6, _STRING):
                raise ValueError(f"an event-stream header of unknown type {kind}")
            (length,) = struct.unpack_from(">H", frame, at)
            value = frame[at + 2 : at + 2 + length]
            at += 2 + length
            if kind == _STRING:
                headers[name] = value.decode()
    except (IndexError, struct.error, UnicodeDecodeError) as e:
        raise ValueError(f"malformed event-stream headers: {e}") from e
    if at != end:
        raise ValueError("event-stream headers overrun their length")
    return headers, frame[end : total - _CRC.size]
