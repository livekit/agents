"""bitHuman plugin: a picture is published when the audio rendered with it starts playing out.

AvatarRunner's audio source holds up to ~100 ms of audio before it plays, while each video
frame is published as soon as it is rendered, so without anchoring the mouth leads the voice
by whatever the audio source holds. With ``follow_audio`` the generator hands the audio over
first and releases the picture when that audio starts playing.
"""

from __future__ import annotations

import asyncio
import time
import types

import numpy as np
import pytest

from livekit import rtc
from livekit.plugins.bithuman.avatar import _PICTURE_LEAD_S, BithumanGenerator

pytestmark = [pytest.mark.unit, pytest.mark.plugin("bithuman")]

SR, SPT, FPS = 16000, 800, 20


class _Chunk:
    def __init__(self, level: int) -> None:
        self.array = np.full(SPT, level, dtype=np.int16)
        self.sample_rate = SR

    @property
    def bytes(self) -> bytes:
        return self.array.tobytes()


class _Runtime:
    """Duck-typed AsyncBithuman: one picture + its 50 ms of audio per tick, on a 20 fps clock."""

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks

    async def run(self):
        t0 = time.monotonic()
        for k in range(self.ticks):
            await asyncio.sleep(max(0.0, t0 + k / FPS - time.monotonic()))
            yield types.SimpleNamespace(
                bgr_image=np.full((4, 4, 3), k % 250, dtype=np.uint8),
                audio_chunk=_Chunk(k + 1),
                end_of_speech=(k == self.ticks - 1),
            )

    def interrupt(self) -> None:
        pass


class _AudioSource:
    """queued_duration as rtc.AudioSource reports it: pushed audio adds, real time drains."""

    def __init__(self, preload: float) -> None:
        self.q, self.t = preload, time.monotonic()

    @property
    def queued_duration(self) -> float:
        now = time.monotonic()
        self.q, self.t = max(0.0, self.q - (now - self.t)), now
        return self.q

    def push(self, seconds: float) -> float:
        before = self.queued_duration
        self.q += seconds
        return before


async def _drive(gen: BithumanGenerator, source: _AudioSource):
    audio, video = [], []
    async for item in gen:
        now = time.monotonic()
        if isinstance(item, rtc.AudioFrame):
            audio.append((now, source.push(item.samples_per_channel / item.sample_rate)))
        elif isinstance(item, rtc.VideoFrame):
            video.append(now)
    return audio, video


def test_picture_is_released_when_its_audio_starts_playing() -> None:
    source = _AudioSource(preload=0.1)  # the audio source already holds 100 ms
    gen = BithumanGenerator(_Runtime(40))  # type: ignore[arg-type]
    gen.follow_audio(lambda: source.queued_duration)
    audio, video = asyncio.run(_drive(gen, source))
    assert len(audio) == len(video) == 40
    errors_ms = [
        (v - (a_t + a_q - _PICTURE_LEAD_S)) * 1000
        for (a_t, a_q), v in zip(audio, video, strict=True)
    ]
    assert max(abs(e) for e in errors_ms) < 20, errors_ms


def test_without_follow_audio_the_picture_leads_its_audio_by_the_queue() -> None:
    source = _AudioSource(preload=0.1)
    gen = BithumanGenerator(_Runtime(40))  # type: ignore[arg-type]
    audio, video = asyncio.run(_drive(gen, source))
    assert len(audio) == len(video) == 40
    # each picture goes out before its own audio was even pushed, ~100 ms ahead of its playout
    lead_ms = np.mean([(a_t + a_q - v) * 1000 for (a_t, a_q), v in zip(audio, video, strict=True)])
    assert lead_ms > 80, lead_ms


def test_clear_buffer_drops_pictures_waiting_for_their_audio() -> None:
    source = _AudioSource(preload=0.1)
    gen = BithumanGenerator(_Runtime(40))  # type: ignore[arg-type]
    gen.follow_audio(lambda: source.queued_duration)

    async def run():
        audio, video = [], []
        async for item in gen:
            if isinstance(item, rtc.AudioFrame):
                audio.append(source.push(item.samples_per_channel / item.sample_rate))
                if len(audio) == 20:
                    gen.clear_buffer()  # a barge-in: the audio source is cleared too
                    source.q = 0.0
            elif isinstance(item, rtc.VideoFrame):
                video.append(item)
        return audio, video

    audio, video = asyncio.run(run())
    assert len(video) < len(audio)


def test_a_picture_that_arrives_without_audio_is_published_as_is() -> None:
    source = _AudioSource(preload=0.1)

    class _PictureOnly(_Runtime):
        async def run(self):
            async for f in super().run():
                yield types.SimpleNamespace(
                    bgr_image=f.bgr_image, audio_chunk=None, end_of_speech=False
                )

    gen = BithumanGenerator(_PictureOnly(10))  # type: ignore[arg-type]
    gen.follow_audio(lambda: source.queued_duration)
    audio, video = asyncio.run(_drive(gen, source))
    assert not audio and len(video) == 10
