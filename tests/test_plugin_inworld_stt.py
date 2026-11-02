from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from livekit.agents import stt
from livekit.plugins.inworld import stt as inworld_stt

from .stt_harness.discovery import provider_by_name
from .stt_harness.fixtures import assert_control_write_failure

pytestmark = pytest.mark.plugin("inworld")


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
@pytest.mark.parametrize(
    ("message", "action"),
    [("audioChunk", "audio"), ("endTurn", "flush"), ("closeStream", "end")],
)
async def test_send_failure_is_not_treated_as_a_successful_stop(message, action, monkeypatch):
    await assert_control_write_failure(
        provider_by_name("inworld").transport_fixture, monkeypatch, message, action
    )


@pytest.mark.virtual_time
@pytest.mark.no_concurrent
async def test_shutdown_sends_close_stream_without_end_turn(monkeypatch):
    import asyncio
    import json

    from livekit import rtc
    from livekit.agents import APIConnectOptions

    from .stt_harness.fixtures import create_transport_fixture

    _, provider, transport = create_transport_fixture(
        provider_by_name("inworld").transport_fixture, monkeypatch
    )
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        stream.push_frame(rtc.AudioFrame.create(16000, 1, 1600))
        stream.end_input()
        await asyncio.sleep(0.1)
        socket = transport.sockets[0]
        messages = [json.loads(data) for data in socket.writes]
        assert "closeStream" in messages[-1]
        assert not any("endTurn" in message for message in messages)
        for event in transport.fixture.completion_messages:
            socket.emit(event)
        await asyncio.wait_for(stream._task, 1)
    finally:
        await stream.aclose()
        await provider.aclose()


@pytest.mark.parametrize("end_ms", [400, 0, None])
@pytest.mark.parametrize("text", ["hello", ""])
def test_endpoint_preserves_word_timing(end_ms, text):
    stream = object.__new__(inworld_stt.SpeechStream)
    stream._stt = inworld_stt.STT(api_key="test-key")
    stream._language = "en"
    stream._speaking = False
    stream._request_id = "test"
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    assert stream._build_transcribe_config()["includeWordTimestamps"] is True
    stream._process_stream_event({"result": {"speechStarted": {}}})
    transcript = {"transcript": text, "isFinal": True}
    if end_ms is not None:
        transcript["wordTimestamps"] = [{"word": "hello", "startTimeMs": 100, "endTimeMs": end_ms}]
    stream._process_stream_event({"result": {"transcription": transcript}})
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert not stream._speaking
    if end_ms:
        assert events[-1].alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert events[-1].alternatives == []
    if text:
        assert events[-2].type == stt.SpeechEventType.FINAL_TRANSCRIPT
        alternative = events[-2].alternatives[0]
        assert alternative.start_time == pytest.approx(7.1 if end_ms else 0.0)
        assert alternative.end_time == pytest.approx(7.4 if end_ms else 0.0)
    else:
        assert [event.type for event in events] == [
            stt.SpeechEventType.START_OF_SPEECH,
            stt.SpeechEventType.END_OF_SPEECH,
        ]


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.inworld.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream._reconnect_event.set())
        clock.sockets[-1].respond({"result": {"speechStarted": {}}})
        clock.sockets[-1].respond(
            {
                "result": {
                    "transcription": {
                        "transcript": "hello",
                        "isFinal": True,
                        "wordTimestamps": [{"word": "hello", "startTimeMs": 100, "endTimeMs": 400}],
                    }
                }
            }
        )
        async with asyncio.timeout(2):
            event = await stream.__anext__()
            while event.type != stt.SpeechEventType.FINAL_TRANSCRIPT:
                event = await stream.__anext__()
            assert event.alternatives[0].start_time == pytest.approx(17.1)
            assert event.alternatives[0].end_time == pytest.approx(17.4)
            event = await stream.__anext__()
            assert event.type == stt.SpeechEventType.END_OF_SPEECH
            assert event.alternatives[0].end_time == pytest.approx(17.4)
    finally:
        await stream.aclose()
        await provider.aclose()
