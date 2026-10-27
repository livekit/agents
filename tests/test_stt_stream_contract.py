"""Deterministic write faults and task races that need an injected transport.

Public lifecycle behavior is covered by tests/stt_harness/test_contract.py using
real WebSockets. Keep only faults that a TCP server cannot place precisely.
"""

import asyncio
from contextlib import asynccontextmanager

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, utils

from .stt_harness.discovery import fault_cases
from .stt_harness.drivers import collect_events
from .stt_harness.faults import make_provider

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.fixture(params=fault_cases(), ids=lambda case: case.name)
def provider(request, monkeypatch):
    return make_provider(request.param, monkeypatch)


@asynccontextmanager
async def running(
    provider, *, max_retry=0, liveness_timeout=0.3, timeout=10, continuous_audio=True
):
    case, instance, transport = provider
    stream = instance.stream(
        conn_options=APIConnectOptions(
            max_retry=max_retry,
            retry_interval=0,
            timeout=timeout,
            final_response_timeout=timeout,
            liveness_timeout=liveness_timeout,
        )
    )
    assert type(stream) is case.stream_type
    collector = asyncio.create_task(collect_events(stream, []))

    async def feed():
        # Keep supplying audio after a reconnect. Outbound audio must not count as
        # inbound activity or successful provider progress.
        while not collector.done():
            samples = 1600 if continuous_audio else 3200
            stream.push_frame(
                rtc.AudioFrame(
                    data=b"\x01\x01" * samples,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=samples,
                )
            )
            if not continuous_audio:
                return
            await asyncio.sleep(0.1)

    sender = asyncio.create_task(feed())
    try:
        await asyncio.wait_for(transport.connected.wait(), 1)
        yield stream, collector
    finally:
        await utils.aio.gracefully_cancel(sender)
        await stream.aclose()
        await utils.aio.gracefully_cancel(collector)
        await instance.aclose()


@pytest.mark.parametrize("failure", ["write", "close"])
@pytest.mark.parametrize("cause", ["timeout", "reset"])
async def test_failed_stop_is_an_error_even_if_the_peer_closes(provider, failure, cause):
    case, _, transport = provider
    if case.stop is None:
        pytest.skip("protocol has no stop-write fixture")
    transport.fail_stop = failure
    if cause == "reset":
        transport.error = aiohttp.ClientConnectionResetError("test stop write failed")
    async with running(provider, timeout=0.5, max_retry=2, continuous_audio=False) as (
        stream,
        collector,
    ):
        await asyncio.sleep(0.2)
        started = asyncio.get_running_loop().time()
        stream.end_input()
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, 2)
        if case.stop is not None and failure == "write":
            assert asyncio.get_running_loop().time() - started < 0.5, (
                "failed writes must not wait for the final-response deadline"
            )
        assert len(transport.sockets) <= 3


async def test_remote_close_cannot_discard_queued_audio(provider):
    _, _, transport = provider
    async with running(provider, timeout=0.5, continuous_audio=False) as (stream, collector):
        await asyncio.sleep(0.2)
        transport.block_writes = True
        stream.push_frame(rtc.AudioFrame.create(16000, 1, 3200))
        stream.push_frame(rtc.AudioFrame.create(16000, 1, 3200))
        stream.end_input()
        await asyncio.wait_for(transport.write_blocked.wait(), 1)
        transport.sockets[0].server_close()
        with pytest.raises(APIError):
            await asyncio.wait_for(collector, 1)
