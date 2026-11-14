"""Mono 16-bit PCM to one Ogg Opus stream per request, sent page by page."""

from __future__ import annotations

import secrets
import struct

import opuslib_next as opuslib

SAMPLE_RATES = {8000, 12000, 16000, 24000, 48000}
FRAME_DURATIONS_MS = {10, 20, 40, 60}
APPLICATIONS = {"audio", "voip", "restricted_lowdelay"}
_PRE_SKIP = 312
_VENDOR = b"spatiussdk"


def _crc_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = (crc << 1 ^ 0x04C11DB7 if crc & 0x80000000 else crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return table


_CRC = _crc_table()


def _crc(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = (crc << 8 & 0xFFFFFFFF) ^ _CRC[(crc >> 24) ^ byte]
    return crc


class OggOpusEncoder:
    """One request's stream: ``encode`` returns the pages completed so far; ``end=True`` pads
    the pre-skip and closes the stream."""

    def __init__(
        self, *, sample_rate: int, bitrate: int, frame_duration_ms: int, application: str
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_size = sample_rate * frame_duration_ms // 1000
        self._scale = 48000 // sample_rate
        self._pcm = bytearray()
        self._pending: tuple[bytes, int] | None = None
        self._started = False
        self._sequence = 0
        self._serial = secrets.randbits(32)
        self._input_samples = 0
        self._encoded_samples = 0
        self._encoder = opuslib.Encoder(sample_rate, 1, application)
        if bitrate > 0:
            self._encoder.bitrate = bitrate
        ctl = opuslib.api.ctl
        for request, value in (
            (ctl.set_vbr, 1),
            (ctl.set_complexity, 10),
            (ctl.set_signal, opuslib.AUTO),
            (ctl.set_lsb_depth, 16),
            (ctl.set_dtx, 0),
            (ctl.set_inband_fec, 0),
        ):
            opuslib.api.encoder.encoder_ctl(self._encoder.encoder_state, request, value)

    def encode(self, pcm: bytes, *, end: bool) -> bytes:
        out = bytearray()
        self._pcm += pcm
        frame_bytes = self._frame_size * 2
        while len(self._pcm) >= frame_bytes:
            self._packet(out, bytes(self._pcm[:frame_bytes]), self._frame_size)
            del self._pcm[:frame_bytes]
        if end:
            if self._pcm:
                samples = len(self._pcm) // 2
                self._packet(out, bytes(self._pcm.ljust(frame_bytes, b"\0")), samples)
                self._pcm.clear()
            final = _PRE_SKIP + self._input_samples * self._scale
            if self._encoded_samples:
                # silence until the decoded stream covers the pre-skip
                while self._encoded_samples * self._scale < final:
                    self._packet(out, b"\0" * frame_bytes, 0)
            if self._pending is not None:
                self._page(out, self._pending[0], final, end_of_stream=True)
                self._pending = None
            elif self._started:
                self._page(out, b"", _PRE_SKIP, end_of_stream=True)
        return bytes(out)

    def _packet(self, out: bytearray, pcm: bytes, samples: int) -> None:
        if not self._started:
            self._started = True
            head = b"OpusHead" + struct.pack("<BBHIhB", 1, 1, _PRE_SKIP, self._sample_rate, 0, 0)
            tags = b"OpusTags" + struct.pack("<I", len(_VENDOR)) + _VENDOR + struct.pack("<I", 0)
            self._page(out, head, 0, begin_of_stream=True)
            self._page(out, tags, 0)
        packet = self._encoder.encode(pcm, self._frame_size)
        self._input_samples += samples
        self._encoded_samples += self._frame_size
        # a page is written once the next packet exists, so the last one can end the stream
        if self._pending is not None:
            self._page(out, *self._pending)
        self._pending = (packet, self._encoded_samples * self._scale)

    def _page(
        self,
        out: bytearray,
        packet: bytes,
        granule: int,
        *,
        begin_of_stream: bool = False,
        end_of_stream: bool = False,
    ) -> None:
        lacing = bytes([255] * (len(packet) // 255) + [len(packet) % 255]) if packet else b""
        flags = (0x02 if begin_of_stream else 0) | (0x04 if end_of_stream else 0)
        header = bytearray(b"OggS\0")
        header += struct.pack("<BQIII", flags, granule, self._serial, self._sequence, 0)
        header += bytes([len(lacing)]) + lacing
        header[22:26] = struct.pack("<I", _crc(bytes(header) + packet))
        self._sequence += 1
        out += header + packet
