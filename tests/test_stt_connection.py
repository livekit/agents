"""Connection task coordination, independent of provider wire protocols."""

import asyncio

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APITimeoutError

from .fake_stt import FakeSTT

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]


@pytest.fixture
async def stream():
    provider = FakeSTT()
    stream = provider.stream(conn_options=APIConnectOptions(timeout=1))
    try:
        yield stream
    finally:
        await stream.aclose()
        await provider.aclose()


def futures():
    loop = asyncio.get_running_loop()
    return [loop.create_future() for _ in range(3)]


async def test_deadline_starts_only_after_sending_finishes(stream):
    sender, receiver, background = futures()
    task = asyncio.create_task(
        stream._run_connection_tasks(
            sender,
            receiver,
            background=[background],
        )
    )
    await asyncio.sleep(3)
    assert not task.done()
    sender.set_result(None)
    await asyncio.sleep(0.5)
    assert not task.done()
    receiver.set_result(None)
    assert await task is False
    assert background.cancelled()


async def test_completed_background_task_does_not_end_connection(stream):
    sender, receiver, background = futures()
    task = asyncio.create_task(
        stream._run_connection_tasks(sender, receiver, background=[background])
    )
    background.set_result(None)
    await asyncio.sleep(3)
    assert not task.done()
    receiver.set_result(None)
    assert await task is False
    assert sender.cancelled()


@pytest.mark.parametrize("input_ended", [False, True])
async def test_final_response_deadline_cancels_remaining_tasks(stream, input_ended):
    sender, receiver, background = futures()
    sender.set_result(None)
    if input_ended:
        stream.end_input()
        await asyncio.sleep(0)
    with pytest.raises(APITimeoutError) as info:
        await stream._run_connection_tasks(sender, receiver, background=[background])
    assert info.value.retryable is (not input_ended)
    assert receiver.cancelled()
    assert background.cancelled()


@pytest.mark.parametrize("role", [0, 1, 2], ids=["sender", "receiver", "background"])
async def test_task_failures_are_preserved(stream, role):
    tasks = futures()
    error = APIConnectionError("test transport failure")
    tasks[role].set_exception(error)
    with pytest.raises(APIConnectionError) as info:
        await stream._run_connection_tasks(tasks[0], tasks[1], background=[tasks[2]])
    assert info.value is error
    assert all(task.done() for task in tasks)


@pytest.mark.parametrize("draining", [False, True])
async def test_reconnect_cancels_connection_tasks(stream, draining):
    sender, receiver, background = futures()
    if draining:
        sender.set_result(None)
    reconnect = asyncio.Event()
    task = asyncio.create_task(
        stream._run_connection_tasks(
            sender,
            receiver,
            reconnect=reconnect,
            background=[background],
        )
    )
    await asyncio.sleep(0.1)
    reconnect.set()
    assert await task is True
    assert receiver.cancelled() and background.cancelled()
    assert sender.done()


@pytest.mark.parametrize("draining", [False, True])
async def test_cancellation_awaits_all_connection_tasks(stream, draining):
    sender, receiver, background = futures()
    if draining:
        sender.set_result(None)
    task = asyncio.create_task(
        stream._run_connection_tasks(
            sender,
            receiver,
            background=[background],
        )
    )
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(task.done() for task in (sender, receiver, background))


async def test_sender_can_include_the_protocol_acknowledgement(stream):
    sender, receiver, _ = futures()
    sender.set_result(None)
    assert await stream._run_connection_tasks(sender, receiver, sender_completes=True) is False
    assert receiver.cancelled()


async def test_provider_progress_does_not_extend_the_final_deadline(stream):
    sender, receiver, _ = futures()
    sender.set_result(None)

    async def report_progress():
        while True:
            stream._mark_provider_progress()
            await asyncio.sleep(0.1)

    progress = asyncio.create_task(report_progress())
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        with pytest.raises(APITimeoutError):
            await stream._run_connection_tasks(sender, receiver)
        assert loop.time() - start == pytest.approx(1)
    finally:
        progress.cancel()
        await asyncio.gather(progress, return_exceptions=True)
