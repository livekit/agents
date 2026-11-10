"""Framed audio streams: PCM and timed emotion cues in one byte stream.

For a ``persona-2.0-live`` persona the plugin opens ``lk.audio_stream`` with the stream
attribute ``kfl.framing: "1"`` and writes records instead of raw PCM::

    type (u8) | payload length (u32 LE) | payload

    0x01  audio: PCM16LE, 24 kHz, mono
    0x02  cue:   UTF-8 JSON, {"emotion": "happy", "at": 57600}; ``at`` is a sample
          offset from the start of the stream's audio

Records ride the ordered stream with the audio they describe, so a cue can neither
overtake nor trail it. Every ``write`` must hold whole records: the avatar uses chunk
boundaries to recover from a write that failed partway.

The avatar advertises what it understands as the participant attribute ``kfl.protocol``,
an integer to compare with ``>=``. Protocol 2 covers framed streams and the
``emotion_now:`` control message; earlier avatars take raw PCM and ``emotion:`` only.
"""

from __future__ import annotations

import json
import struct

FRAMING_ATTRIBUTE = "kfl.framing"
"""Byte stream attribute marking a framed ``lk.audio_stream``."""

FRAMING_VERSION = "1"

PROTOCOL_ATTRIBUTE = "kfl.protocol"
"""Participant attribute on which the avatar advertises its protocol version."""

EMOTION_NOW_PROTOCOL = 2
"""The first protocol version that understands the ``emotion_now:`` control message."""

RECORD_AUDIO = 0x01
RECORD_CUE = 0x02

_HEADER = struct.Struct("<BI")


def encode_record(kind: int, payload: bytes) -> bytes:
    return _HEADER.pack(kind, len(payload)) + payload


def encode_audio(pcm: bytes) -> bytes:
    """One audio record holding ``pcm`` (PCM16LE, 24 kHz, mono)."""
    return encode_record(RECORD_AUDIO, pcm)


def encode_cue(emotion: str, at: int) -> bytes:
    """One cue record switching to ``emotion`` at sample ``at`` of the stream."""
    body = {"emotion": emotion, "at": at}
    return encode_record(RECORD_CUE, json.dumps(body, separators=(",", ":")).encode())
