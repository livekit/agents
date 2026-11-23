"""G.711 mu-law <-> 16-bit PCM, vectorized (``audioop`` is gone from Python 3.13)."""

from __future__ import annotations

import numpy as np

_BIAS = 0x84


def _build_decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


_DECODE_TABLE = _build_decode_table()


def ulaw_to_pcm16(data: bytes) -> bytes:
    """Decode mu-law bytes to little-endian int16 PCM (one sample per byte)."""
    if not data:
        return b""
    return _DECODE_TABLE[np.frombuffer(data, dtype=np.uint8)].tobytes()


_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)


def pcm16_to_ulaw(data: bytes) -> bytes:
    """Encode little-endian int16 PCM to mu-law bytes (bit-exact with CCITT G.711 / audioop)."""
    if not data:
        return b""
    x14 = np.frombuffer(data, dtype=np.int16).astype(np.int32) >> 2
    negative = x14 < 0
    mag = np.minimum(np.where(negative, -x14, x14), 8159) + 0x21
    seg = np.searchsorted(_SEG_END, mag, side="left")
    uval = np.where(
        seg >= 8, 0x7F, (np.minimum(seg, 7) << 4) | ((mag >> (np.minimum(seg, 7) + 1)) & 0x0F)
    )
    return (uval ^ np.where(negative, 0x7F, 0xFF)).astype(np.uint8).tobytes()
