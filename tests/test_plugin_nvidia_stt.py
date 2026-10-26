# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.nvidia import stt as nvidia_stt

pytestmark = pytest.mark.plugin("nvidia")


def _audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\0\0" * 160,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


@pytest.mark.parametrize("worker_error", [False, True])
async def test_worker_failure_reaches_retry_limit_with_open_input(monkeypatch, worker_error):
    attempts = []

    class Service:
        def streaming_response_generator(self, audio, config):
            attempts.append(None)
            if worker_error:
                raise RuntimeError("connection lost")
            return iter(())

    monkeypatch.setattr(nvidia_stt.auth, "create_riva_auth", lambda **kwargs: None)
    monkeypatch.setattr(nvidia_stt.riva.client, "ASRService", lambda auth: Service())
    provider = nvidia_stt.STT(api_key="test")
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=2, retry_interval=0))
    stream.push_frame(_audio_frame())
    try:
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._task, 1)
        assert len(attempts) == 3
    finally:
        await stream.aclose()
        await provider.aclose()


async def test_normal_input_completion_does_not_retry(monkeypatch):
    attempts = []

    class Service:
        def streaming_response_generator(self, audio, config):
            attempts.append(None)
            list(audio)
            return iter(())

    monkeypatch.setattr(nvidia_stt.auth, "create_riva_auth", lambda **kwargs: None)
    monkeypatch.setattr(nvidia_stt.riva.client, "ASRService", lambda auth: Service())
    provider = nvidia_stt.STT(api_key="test")
    stream = provider.stream()
    stream.push_frame(_audio_frame())
    stream.end_input()
    try:
        await asyncio.wait_for(stream._task, 1)
        assert [event async for event in stream] == []
        assert len(attempts) == 1
    finally:
        await stream.aclose()
        await provider.aclose()


def test_provider_result_resets_retries_without_transcript():
    stream = nvidia_stt.SpeechStream.__new__(nvidia_stt.SpeechStream)
    stream._on_provider_progress = None
    stream._num_retries = 2
    stream._handle_response(
        SimpleNamespace(results=[SimpleNamespace(alternatives=[SimpleNamespace(transcript="")])]),
        event_loop=SimpleNamespace(call_soon_threadsafe=lambda callback, *args: callback(*args)),
    )
    assert stream._num_retries == 0
