"""Common lifecycle checks for every WebSocketMonitor-backed STT stream."""

import ast
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, stt, utils

from .fake_stt import FakeSTT
from .stt_contract_cases import CASES, make_provider

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.fixture(params=CASES, ids=lambda case: case.name)
def provider(request, monkeypatch):
    return make_provider(request.param, monkeypatch)


@asynccontextmanager
async def running(provider, *, max_retry=0, liveness_timeout=0.3):
    case, instance, transport = provider
    stream = instance.stream(
        conn_options=APIConnectOptions(
            max_retry=max_retry,
            retry_interval=0,
            timeout=10,
            liveness_timeout=liveness_timeout,
        )
    )
    assert type(stream) is case.stream_type
    stream._on_provider_progress = transport.progress.set

    async def feed():
        # Keep supplying audio after a reconnect. Outbound audio must not count as
        # inbound activity or successful provider progress.
        while not stream._task.done() and not stream._input_ch.closed:
            stream.push_frame(
                rtc.AudioFrame(
                    data=b"\x01\x01" * 1600,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=1600,
                )
            )
            await asyncio.sleep(0.1)

    sender = asyncio.create_task(feed())
    try:
        await asyncio.wait_for(transport.connected.wait(), 1)
        yield stream
    finally:
        await utils.aio.gracefully_cancel(sender)
        await stream.aclose()
        await instance.aclose()


async def test_quiet_connection_uses_ping_without_reconnecting(provider):
    _, _, transport = provider
    async with running(provider) as stream:
        await asyncio.sleep(0.7)
        if stream._task.done():
            await stream._task
        assert not stream._task.done()
        assert len(transport.sockets) == 1
        socket = transport.sockets[0]
        assert socket.kwargs.get("autoping") is False
        assert socket.pings >= 2


@pytest.mark.parametrize("failure", ["close", "error", "silent"])
async def test_connection_failure_has_a_bounded_retry_budget(provider, failure):
    case, _, transport = provider
    transport.failure = failure
    async with running(provider, max_retry=2) as stream:
        with pytest.raises(APIError) as info:
            await asyncio.wait_for(stream._task, 2)
        expected = 3 if case.retry_policy == "standard" else 1
        assert len(transport.sockets) == expected
        assert info.value.retryable is (case.retry_policy != "no_replay")
        if failure == "error":
            # Never spin through ERROR frames while waiting for a later CLOSED.
            assert all(socket.errors == 1 for socket in transport.sockets)
            if case.retry_policy != "no_replay":
                cause = info.value
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                assert cause is transport.error


@pytest.mark.parametrize("failure", ["close", "error", "silent"])
async def test_connection_failure_reaches_fallback(provider, failure):
    case, instance, transport = provider
    transport.failure = failure
    backup = FakeSTT(fake_transcript="backup")
    adapter = stt.FallbackAdapter([instance, backup], max_retry_per_stt=1, retry_interval=0)
    try:
        async with adapter.stream(
            conn_options=APIConnectOptions(
                max_retry=0,
                liveness_timeout=0.3,
            )
        ) as stream:

            async def feed():
                while True:
                    stream.push_frame(rtc.AudioFrame.create(16000, 1, 1600))
                    await asyncio.sleep(0.01)

            sender = asyncio.create_task(feed())
            try:

                async def final():
                    async for event in stream:
                        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                            return event.alternatives[0].text

                assert await asyncio.wait_for(final(), 2) == "backup"
            finally:
                await utils.aio.gracefully_cancel(sender)
        # A failed provider can also have one background recovery probe.
        attempts = 2 if case.retry_policy == "standard" else 1
        assert attempts <= len(transport.sockets) <= attempts + 1
    finally:
        await adapter.aclose()
        await instance.aclose()


def test_all_monitored_stt_streams_have_a_contract_case():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "livekit-agents/livekit/agents/inference/stt.py"]
    paths += list((root / "livekit-plugins").glob("livekit-plugins-*/livekit/plugins/**/*.py"))
    implementations = set()
    for path in paths:
        if "stt" not in path.stem and "recognize_stream" not in path.stem:
            continue
        source = path.read_text()
        if "WebSocketMonitor" not in source:
            continue
        tree = ast.parse(source)
        parts = path.relative_to(root).with_suffix("").parts
        module = ".".join(parts[parts.index("livekit") :])
        for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "WebSocketMonitor"
                for node in ast.walk(cls)
            ):
                implementations.add((module, cls.name))
    registered = {(case.module, case.stream_class) for case in CASES}
    assert implementations == registered
    assert len(registered) == len(CASES)


async def test_only_provider_progress_resets_retries(provider):
    case, _, transport = provider
    async with running(provider, max_retry=2) as stream:
        progress = transport.progress
        await asyncio.sleep(0.05)
        stream._num_retries = 2
        await asyncio.sleep(0.4)
        assert not progress.is_set(), "setup, outbound audio and PONG are not provider progress"
        assert stream._num_retries == 2
        transport.sockets[0].emit(case.progress)
        await asyncio.wait_for(progress.wait(), 1)
        assert stream._num_retries == 0


@pytest.mark.parametrize("failure", ["error", "silent"])
@pytest.mark.parametrize("liveness_timeout", [0.3, 6.0])
async def test_failure_while_waiting_for_final_result_is_reported(
    provider, failure, liveness_timeout
):
    _, _, transport = provider
    async with running(provider, liveness_timeout=liveness_timeout) as stream:
        await asyncio.sleep(0.05)
        transport.failure = failure
        stream.end_input()
        if failure == "error":
            transport.sockets[0].emit(
                aiohttp.WSMessage(aiohttp.WSMsgType.ERROR, transport.error, "")
            )
        with pytest.raises(APIError):
            await asyncio.wait_for(stream._task, liveness_timeout + 1)
        assert len(transport.sockets) == 1


async def test_cancellation_closes_socket_without_retry(provider):
    _, instance, transport = provider
    async with running(provider) as stream:
        await asyncio.sleep(0.05)
        await stream.aclose()
        await instance.aclose()
        assert len(transport.sockets) == 1
        assert transport.sockets[0].closed


async def test_close_after_input_end_finishes_without_retry(provider):
    case, _, transport = provider
    async with running(provider) as stream:
        await asyncio.sleep(0.05)
        stream.end_input()
        await asyncio.sleep(0.2)
        for message in case.completion:
            transport.sockets[0].emit(message)
        await asyncio.sleep(0.01)
        transport.sockets[0].server_close()
        # Some providers let a sleeping application keepalive task finish before
        # returning. Virtual time covers that wait without a wall-clock delay.
        await asyncio.wait_for(stream._task, 40)
        assert len(transport.sockets) == 1
