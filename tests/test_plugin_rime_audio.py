"""Rime microphone buffering tests using the actual SDK and a local peer."""

# ruff: noqa: E402 -- optional SDK must be checked before importing the plugin
from __future__ import annotations

import asyncio
import base64

import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("rimelabs_sdk.realtime")

from livekit import rtc

from .test_plugin_rime_realtime import peer_model


async def test_clear_audio_follows_inflight_audio_and_discards_queued_audio(monkeypatch):
    async with peer_model() as (peer, model, _):
        session = model.session()
        await session.initialize(instructions="test")
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
