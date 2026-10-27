from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, stt

from .stt_harness.discovery import provider_by_name
from .stt_harness.faults import assert_control_write_failure, create_transport_fixture

pytestmark = pytest.mark.plugin("gladia")


async def test_update_options_stores_region_for_new_streams():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key", region="eu-west")
    stt.update_options(region="us-west")

    assert stt._opts.region == "us-west"


async def test_update_options_rejects_translation_without_target_languages():
    from livekit.plugins.gladia import STT

    stt = STT(api_key="test-key")
    with pytest.raises(ValueError, match="translation_target_languages is required"):
        stt.update_options(translation_enabled=True, region="us-west")
    assert stt._opts.translation_config.enabled is False
    assert stt._opts.region == "eu-west"

    stt = STT(api_key="test-key", translation_enabled=True, translation_target_languages=["fr"])
    with pytest.raises(ValueError, match="translation_target_languages is required"):
        stt.update_options(translation_target_languages=[])
    assert stt._opts.translation_config.target_languages == ["fr"]

    stt.update_options(translation_enabled=False, translation_target_languages=[])
    assert stt._opts.translation_config.enabled is False


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
@pytest.mark.parametrize("completion", ["close", "post_final_transcript", "end_session"])
async def test_gladia_drains_final_transcript_after_input_ends(monkeypatch, completion):
    _, provider, transport = create_transport_fixture(
        provider_by_name("gladia").transport_fixture, monkeypatch
    )
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))

    async def collect():
        return [event async for event in stream]

    try:
        stream.end_input()
        await transport.connected.wait()
        socket = transport.sockets[0]
        await _wait_for_stop(socket)
        await asyncio.sleep(0.01)
        socket.messages.put_nowait(
            aiohttp.WSMessage(
                aiohttp.WSMsgType.TEXT,
                json.dumps(
                    {
                        "type": "transcript",
                        "data": {"is_final": True, "utterance": {"text": "final words"}},
                    }
                ),
                "",
            )
        )
        if completion == "close":
            socket.server_close()
        else:
            socket.messages.put_nowait(
                aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps({"type": completion}), "")
            )
        events = await asyncio.wait_for(collect(), 1)
        assert [
            event.alternatives[0].text
            for event in events
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ] == ["final words"]
    finally:
        await stream.aclose()
        await provider.aclose()


async def _wait_for_stop(socket):
    async def wait():
        while not any(isinstance(data, str) and "stop_recording" in data for data in socket.writes):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 1)


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
async def test_stop_recording_write_failure_is_reported(monkeypatch):
    await assert_control_write_failure(
        provider_by_name("gladia").transport_fixture, monkeypatch, "stop_recording", "end"
    )


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
@pytest.mark.parametrize("stop", ["flush", "silence"])
@pytest.mark.parametrize("first_samples", [320, 1600])
async def test_segment_completion_preserves_queued_audio(monkeypatch, stop, first_samples):
    from livekit.plugins.gladia.stt import AudioEnergyFilter

    fixture = replace(
        provider_by_name("gladia").transport_fixture,
        completion_messages=(
            {
                "type": "transcript",
                "data": {
                    "is_final": True,
                    "utterance": {
                        "text": "words",
                        "start": 0,
                        "end": 0.02,
                        "words": [{"word": "words", "start": 0, "end": 0.02}],
                    },
                },
            },
            {"type": "end_session"},
        ),
        options={
            "energy_filter": AudioEnergyFilter(min_silence=0) if stop == "silence" else False,
        },
    )
    _, provider, transport = create_transport_fixture(fixture, monkeypatch)
    transport.auto_reply = True
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    frames = [
        rtc.AudioFrame(
            data=bytes([value, value]) * samples,
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=samples,
        )
        for value, samples in [(1, first_samples), (2, 1600)]
    ]
    try:
        stream.push_frame(frames[0])
        if stop == "flush":
            stream.flush()
        else:
            stream.push_frame(rtc.AudioFrame.create(16000, 1, 320))
        stream.push_frame(frames[1])
        # Close input before either connection runs. Its remaining queue must drain.
        stream.end_input()
        events = [event async for event in stream]
        assert [
            event.alternatives[0].text
            for event in events
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ] == ["words", "words"]
        finals = [
            event.alternatives[0]
            for event in events
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals[1].start_time - finals[0].start_time == pytest.approx(frames[0].duration)
        assert finals[1].words[0].start_time - finals[0].words[0].start_time == pytest.approx(
            frames[0].duration
        )
        assert len(transport.sockets) == 2
        for socket, frame in zip(transport.sockets, frames, strict=True):
            chunks = [
                base64.b64decode(json.loads(data)["data"]["chunk"])
                for data in socket.writes
                if json.loads(data).get("type") == "audio_chunk"
            ]
            assert b"".join(chunks) == frame.data.tobytes()
        assert stream._num_retries == 0
    finally:
        await stream.aclose()
        await provider.aclose()
