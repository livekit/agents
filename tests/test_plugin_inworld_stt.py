import pytest

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
