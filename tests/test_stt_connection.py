"""Connection task coordination, independent of provider wire protocols."""

import asyncio

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APITimeoutError

from .fake_stt import FakeSTT

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]


@pytest.fixture
async def stream():
    provider = FakeSTT()
    stream = provider.stream(conn_options=APIConnectOptions(timeout=0.01, final_response_timeout=1))
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
    with pytest.raises(APIConnectionError, match="before audio sending finished"):
        await task
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
    assert info.value.retryable is True
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


@pytest.mark.parametrize("completion", ["sender", "receiver", "reconnect"])
async def test_failure_wins_over_simultaneous_completion(stream, completion):
    sender, receiver, background = futures()
    reconnect = asyncio.Event()
    if completion == "sender":
        sender.set_result(None)
    elif completion == "receiver":
        receiver.set_result(None)
    else:
        reconnect.set()
    error = APIConnectionError("background failed")
    background.set_exception(error)
    with pytest.raises(APIConnectionError) as info:
        await stream._run_connection_tasks(
            sender, receiver, background=[background], reconnect=reconnect
        )
    assert info.value is error
    assert all(task.done() for task in (sender, receiver, background))


@pytest.mark.parametrize("role", [0, 1, 2])
async def test_unexpected_child_cancellation_is_an_error(stream, role):
    sender, receiver, background = tasks = futures()
    tasks[role].cancel()
    with pytest.raises(APIConnectionError, match="cancelled unexpectedly"):
        await stream._run_connection_tasks(sender, receiver, background=[background])
    assert all(task.done() for task in tasks)


@pytest.mark.parametrize("error", [ConnectionResetError(), asyncio.TimeoutError()])
async def test_transport_exceptions_use_the_api_error_contract(stream, error):
    sender, receiver, _ = futures()
    sender.set_exception(error)
    expected = APITimeoutError if isinstance(error, asyncio.TimeoutError) else APIConnectionError
    with pytest.raises(expected) as info:
        await stream._run_connection_tasks(sender, receiver)
    assert info.value.__cause__ is error


@pytest.mark.parametrize("sender_ack", [False, True])
async def test_receiver_completion_during_drain_waits_for_the_sender(stream, sender_ack):
    sender, receiver, _ = futures()
    draining = asyncio.Event()
    draining.set()
    receiver.set_result(None)
    task = asyncio.create_task(
        stream._run_connection_tasks(
            sender, receiver, draining=draining, sender_completes=sender_ack
        )
    )
    await asyncio.sleep(0.1)
    assert not task.done()
    sender.set_result(None)
    assert await task is False


async def test_sender_failure_after_completion_is_not_discarded(stream):
    sender, receiver, _ = futures()
    draining = asyncio.Event()
    draining.set()
    receiver.set_result(None)
    task = asyncio.create_task(stream._run_connection_tasks(sender, receiver, draining=draining))
    await asyncio.sleep(0.1)
    sender.set_exception(ConnectionResetError("EOS write failed"))
    with pytest.raises(APIConnectionError) as info:
        await task
    assert isinstance(info.value.__cause__, ConnectionResetError)


async def test_drain_signal_bounds_an_sdk_sender_waiting_for_ack(stream):
    sender, receiver, _ = futures()
    draining = asyncio.Event()
    task = asyncio.create_task(
        stream._run_connection_tasks(sender, receiver, draining=draining, sender_completes=True)
    )
    await asyncio.sleep(3)
    assert not task.done()
    draining.set()
    with pytest.raises(APITimeoutError):
        await task


async def test_best_effort_drain_ends_without_an_error(stream):
    sender, receiver, _ = futures()
    sender.set_result(None)
    assert await stream._run_connection_tasks(sender, receiver, require_completion=False) is False
    assert receiver.cancelled()


async def test_best_effort_drain_cannot_hide_a_blocked_sender(stream):
    sender, receiver, _ = futures()
    draining = asyncio.Event()
    draining.set()
    with pytest.raises(APITimeoutError):
        await stream._run_connection_tasks(
            sender, receiver, draining=draining, require_completion=False
        )


@pytest.mark.parametrize(
    "error", [APIConnectionError("socket failed"), APITimeoutError("final timed out")]
)
async def test_drained_input_is_not_retried_even_for_a_retryable_error(error):
    provider = FakeSTT(fake_exception=error)
    errors = []
    provider.on("error", errors.append)
    try:
        async with provider.stream(conn_options=APIConnectOptions(max_retry=3)) as speech:
            speech.end_input()
            with pytest.raises(type(error)):
                async for _ in speech:
                    pass
            assert speech.attempt == 1
            assert len(errors) == 1
            assert not errors[0].recoverable
    finally:
        await provider.aclose()
