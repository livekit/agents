from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from livekit import rtc

_SAMPLE_RATE = 48000
_BLOCK = 4800

_ROOT_HZ = 174.61  # F3
_CHORD_SEMITONES = (0, 4, 7)  # F major triad

_BEAT_S = 0.28
_NOTE_DUR_S = 0.34
_TAG_DELAY_S = 0.08
_TAG_DUR_S = 0.18
_TAG_AMP = 0.45
_TAIL_S = 0.85

_ATTACK_FRAC = 0.55
_RELEASE_FRAC = 0.10

_WOBBLE_HZ = 22.0
_WOBBLE_DEPTH = 0.05
_DETUNE_CENTS = 2.0

_AMP = 2500.0


def _asr_envelope(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    attack_n = max(1, int(n * _ATTACK_FRAC))
    release_n = max(1, int(n * _RELEASE_FRAC))
    sustain_n = max(0, n - attack_n - release_n)
    env = np.empty(n, dtype=np.float64)
    env[:attack_n] = np.linspace(0.0, 1.0, attack_n)
    env[attack_n : attack_n + sustain_n] = 1.0
    env[attack_n + sustain_n :] = np.linspace(1.0, 0.0, release_n)
    if _WOBBLE_DEPTH > 0:
        t = np.arange(n, dtype=np.float64) / _SAMPLE_RATE
        env *= (
            1.0 - _WOBBLE_DEPTH + _WOBBLE_DEPTH * (0.5 + 0.5 * np.cos(2 * np.pi * _WOBBLE_HZ * t))
        )
    return env


def _note(freq: float, dur_s: float, amp: float) -> np.ndarray:
    n = int(dur_s * _SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / _SAMPLE_RATE
    det = 2.0 ** (_DETUNE_CENTS / 1200.0)
    voice = 0.5 * np.sin(2 * np.pi * freq * det * t)
    voice += 0.5 * np.sin(2 * np.pi * freq / det * t)
    return voice * _asr_envelope(n) * amp


def _semitone_freq(root_hz: float, semis: int) -> float:
    return root_hz * (2.0 ** (semis / 12.0))


def _build_hold_loop() -> np.ndarray:
    chord_notes = [_semitone_freq(_ROOT_HZ, s) for s in _CHORD_SEMITONES]
    tag_freq = chord_notes[-1]
    tag_onset = len(chord_notes) * _BEAT_S + _TAG_DELAY_S
    total_n = int((tag_onset + _TAG_DUR_S + _TAIL_S) * _SAMPLE_RATE)
    out = np.zeros(total_n, dtype=np.float64)

    for i, freq in enumerate(chord_notes):
        note = _note(freq, _NOTE_DUR_S, _AMP)
        start = int(i * _BEAT_S * _SAMPLE_RATE)
        end = min(total_n, start + note.shape[0])
        out[start:end] += note[: end - start]

    tag = _note(tag_freq, _TAG_DUR_S, _AMP * _TAG_AMP)
    start = int(tag_onset * _SAMPLE_RATE)
    end = min(total_n, start + tag.shape[0])
    out[start:end] += tag[: end - start]

    return np.clip(out, -32767.0, 32767.0).astype(np.int16)


_HOLD_LOOP: np.ndarray | None = None


def _get_hold_loop() -> np.ndarray:
    global _HOLD_LOOP
    if _HOLD_LOOP is None:
        _HOLD_LOOP = _build_hold_loop()
    return _HOLD_LOOP


async def hold_beats() -> AsyncIterator[rtc.AudioFrame]:
    loop = _get_hold_loop()
    loop_n = loop.shape[0]
    t = 0
    while True:
        idx = (np.arange(t, t + _BLOCK) % loop_n).astype(np.int64)
        chunk = loop[idx].tobytes()
        t += _BLOCK
        yield rtc.AudioFrame(chunk, _SAMPLE_RATE, 1, _BLOCK)
