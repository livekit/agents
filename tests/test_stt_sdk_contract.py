"""The same completion, deadline, and abort rules across SDK-backed STTs."""

import asyncio
import sys

import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, stt, utils

from .stt_harness.discovery import provider_by_name, sdk_providers
from .stt_harness.drivers import collect_events

pytestmark = pytest.mark.unit


@pytest.fixture(
    params=[
        pytest.param(
            provider_config,
            id=provider_config.name,
            marks=pytest.mark.skipif(
                sys.version_info < provider_config.minimum_python,
                reason="provider SDK requires a newer Python",
            ),
        )
        for provider_config in sdk_providers()
    ]
)
async def connection(request, monkeypatch):
    provider, server = request.param.create_fixture(monkeypatch)
    stream = provider.stream(
        conn_options=APIConnectOptions(max_retry=0, timeout=0.2, final_response_timeout=0.2)
    )
    events = []
    collector = asyncio.create_task(collect_events(stream, events))
    try:
        await asyncio.wait_for(server.connected.wait(), 1)
        yield stream, server, collector, events
    finally:
        await stream.aclose()
        await utils.aio.gracefully_cancel(collector)
        await provider.aclose()
        await server.close()


async def test_deadline_does_not_apply_to_active_input(connection):
    stream, _, collector, _ = connection
    await asyncio.sleep(0.3)
    assert not collector.done()


async def test_final_response_is_drained_before_success(connection):
    stream, server, collector, events = connection
    first, tail = b"\x00\x10" * 3200, b"\x00\x20" * 317
    stream.push_frame(rtc.AudioFrame(first, 16000, 1, 3200))
    stream.push_frame(rtc.AudioFrame(tail, 16000, 1, 317))
    stream.end_input()
    await asyncio.wait_for(server.stopped.wait(), 1)
    await asyncio.sleep(0.05)
    assert not collector.done()
    assert b"".join(server.audio) == first + tail
    server.finish()

    await asyncio.wait_for(collector, 1)
    assert [
        e.alternatives[0].text for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ] == ["words"]


@pytest.mark.parametrize("failure", ["stop_write", "missing_ack"])
async def test_incomplete_shutdown_is_an_error(connection, failure):
    stream, server, collector, events = connection
    server.fail_stop = failure == "stop_write"
    stream.push_frame(rtc.AudioFrame.create(16000, 1, 3200))
    stream.end_input()
    with pytest.raises(APIError):
        await asyncio.wait_for(collector, 1)


@pytest.mark.parametrize("draining", [False, True])
async def test_cancel_aborts_without_waiting_for_completion(connection, draining):
    stream, server, collector, events = connection
    if draining:
        stream.end_input()
        await asyncio.wait_for(server.stopped.wait(), 1)
    await asyncio.wait_for(stream.aclose(), 1)
    assert collector.done()


async def test_remote_completion_before_input_end_is_an_error(connection):
    stream, server, collector, events = connection
    server.drop()
    with pytest.raises(APIError):
        await asyncio.wait_for(collector, 1)


@pytest.mark.parametrize("finish_old_session", [True, False])
async def test_azure_option_change_drains_the_old_recognizer(monkeypatch, finish_old_session):
    provider, server = provider_by_name("azure").create_fixture(monkeypatch)
    stream = provider.stream(
        conn_options=APIConnectOptions(max_retry=0, final_response_timeout=0.2)
    )
    events = []
    collector = asyncio.create_task(collect_events(stream, events))
    try:
        await asyncio.wait_for(server.connected.wait(), 1)
        old = server.recognizers[0]
        stream.push_frame(rtc.AudioFrame.create(16000, 1, 3200))
        stream.update_options(language="fr-FR")
        await asyncio.wait_for(server.stopped.wait(), 1)
        assert len(server.recognizers) == 1
        if not finish_old_session:
            with pytest.raises(APIError):
                await asyncio.wait_for(collector, 1)
            assert len(server.recognizers) == 1
            return
        server.connected.clear()
        server.finish()
        await asyncio.wait_for(server.connected.wait(), 1)
        assert len(server.recognizers) == 2
        assert [e.alternatives[0].text for e in events if e.alternatives] == ["words"]
        old.session_stopped.emit()
        await asyncio.sleep(0.01)
        assert not collector.done(), "an old recognizer stopped the new connection"
        server.stopped.clear()
        stream.end_input()
        await asyncio.wait_for(server.stopped.wait(), 1)
        server.finish()
        await asyncio.wait_for(collector, 1)
        assert [e.alternatives[0].text for e in events if e.alternatives] == ["words", "words"]
    finally:
        await stream.aclose()
        await utils.aio.gracefully_cancel(collector)
        await provider.aclose()
        await server.close()
