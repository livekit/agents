"""Rime microphone buffering tests using the actual SDK and a local peer."""

# ruff: noqa: E402 -- optional SDK must be checked before importing the plugin
from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("rimelabs_sdk.realtime")

from livekit import rtc

from .test_plugin_rime_realtime import peer_model


@pytest.mark.parametrize("sample_rate,channels", [(44100, 1), (44100, 2), (44100, 4), (48000, 4)])
async def test_audio_converts_formats_outside_sdk_support(sample_rate, channels):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="test")
        frame = rtc.AudioFrame(
            data=np.full((sample_rate // 5, channels), 1000, dtype=np.int16).tobytes(),
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=sample_rate // 5,
        )
        session.push_audio(frame)
        assert session._failure is None
        event = await peer.next("input_audio_buffer.append")
        samples = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
        assert len(samples) == 640
        assert np.allclose(samples[100:], 1000, atol=5)
        await session.aclose()


async def test_audio_resampling_keeps_state_between_short_frames():
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="test")
        frame = rtc.AudioFrame(
            data=np.full(441, 1000, dtype=np.int16).tobytes(),
            sample_rate=44100,
            num_channels=1,
            samples_per_channel=441,
        )
        # Each frame is too short for a fresh resampler to produce output.
        for _ in range(20):
            session.push_audio(frame)
        event = await peer.next("input_audio_buffer.append")
        samples = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
        assert np.allclose(samples[100:], 1000, atol=5)
        await session.aclose()


@pytest.mark.parametrize("reset", ["clear", "format_change"])
async def test_audio_conversion_discards_buffered_samples_on_reset(reset):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="test")
        session.push_audio(
            rtc.AudioFrame(
                data=np.full(100, 30000, dtype=np.int16).tobytes(),
                sample_rate=44100,
                num_channels=1,
                samples_per_channel=100,
            )
        )
        if reset == "clear":
            session.clear_audio()
        else:
            session.push_audio(rtc.AudioFrame.create(16000, 1, 640))
            await peer.next("input_audio_buffer.append")
        session.push_audio(rtc.AudioFrame.create(44100, 1, 8820))
        event = await peer.next("input_audio_buffer.append")
        samples = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
        assert np.max(np.abs(samples)) <= 1
        await session.aclose()


@pytest.mark.parametrize("sample_rate", [8000, 16000, 24000, 48000])
@pytest.mark.parametrize("channels", [1, 2])
async def test_sdk_supported_audio_reaches_sdk_unchanged(sample_rate, channels, monkeypatch):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="test")
        sdk = session._require()
        send_audio = sdk.send_audio
        chunks = []

        async def observe(chunk):
            chunks.append(chunk)
            await send_audio(chunk)

        monkeypatch.setattr(sdk, "send_audio", observe)
        frame = rtc.AudioFrame.create(sample_rate, channels, sample_rate // 25)
        session.push_audio(frame)
        await peer.next("input_audio_buffer.append")
        assert len(chunks) == 1
        assert chunks[0].format.sample_rate == sample_rate
        assert chunks[0].format.channels == channels
        assert chunks[0].data == bytes(frame.data)
        await session.aclose()


async def test_clear_audio_follows_inflight_audio_and_discards_queued_audio(monkeypatch):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session._update_session(instructions="test")
        sdk = session._require()
        send = sdk._send
        clear = sdk.clear_audio
        append_started = asyncio.Event()
        append_release = asyncio.Event()
        clear_started = asyncio.Event()
        sent = []

        async def blocked_send(event, *args, **kwargs):
            if event["type"] == "input_audio_buffer.append" and not append_started.is_set():
                # The actual SDK already owns its audio lock at this point.
                append_started.set()
                await append_release.wait()
            sent.append(event["type"])
            await send(event, *args, **kwargs)

        async def observed_clear():
            clear_started.set()
            await clear()

        monkeypatch.setattr(sdk, "_send", blocked_send)
        monkeypatch.setattr(sdk, "clear_audio", observed_clear)

        def frame(sample):
            return rtc.AudioFrame(
                data=sample.to_bytes(2, "little") * 640,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=640,
            )

        first, queued, fresh = frame(1), frame(2), frame(3)
        session.push_audio(first)
        await asyncio.wait_for(append_started.wait(), 3)
        session.push_audio(queued)
        session.clear_audio()
        await asyncio.wait_for(clear_started.wait(), 3)
        session.push_audio(fresh)
        append_release.set()

        before_clear = await peer.next("input_audio_buffer.append")
        after_clear = await peer.next("input_audio_buffer.append")
        assert base64.b64decode(before_clear["audio"]) == bytes(first.data)
        assert base64.b64decode(after_clear["audio"]) == bytes(fresh.data)
        assert sent == [
            "input_audio_buffer.append",
            "input_audio_buffer.clear",
            "input_audio_buffer.append",
        ]
        await session.aclose()
