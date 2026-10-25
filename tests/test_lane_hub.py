"""The lane hub (``storage/hub.py``): a request in the holder's own process tells the holder at
once, hears at once when its task moved, and reads its task's frames as the holder writes them."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents.ipc import channel, proto
from livekit.agents.storage import hub

pytestmark = [pytest.mark.unit]


def holding(thread_id: str) -> hub.LaneHub:
    """A hub whose holder has a tool running in the background."""
    lane = hub.lane_hub(thread_id, "desk")
    lane.held = True
    lane.running["c1"] = ("AST_1", asyncio.ensure_future(asyncio.sleep(10)), None)  # type: ignore[arg-type]
    return lane


async def test_an_arrival_wakes_the_holder_at_once() -> None:
    lane = holding("AT_arrival")
    waiting = asyncio.ensure_future(lane.background(1.0))
    await asyncio.sleep(0.01)
    signaled = time.monotonic()
    hub.signal("AT_arrival", "desk")
    background = await waiting
    assert time.monotonic() - signaled < 0.05, "not at the next poll tick"
    assert background.arrived and background.running == 1
    lane.running.pop("c1")[1].cancel()


async def test_a_signal_that_came_during_a_check_is_not_lost() -> None:
    """A watcher reads the count, checks the boundary, then waits: a message that landed while
    it checked ends the wait at once, and the next wait waits for a newer one."""
    lane = hub.lane_hub("AT_since", "desk")
    seen = lane.signals
    hub.signal("AT_since", "desk")  # lands while the watcher reads its boundary
    started = time.monotonic()
    await lane.since(seen, 5.0)
    assert time.monotonic() - started < 0.05
    started = time.monotonic()
    await lane.since(lane.signals, 0.1)
    assert time.monotonic() - started >= 0.09
    hub.drop("AT_since", "desk")


async def test_a_request_hears_when_its_task_moved() -> None:
    waiting = asyncio.ensure_future(hub.settled("AST_2", 5.0))
    await asyncio.sleep(0.01)
    moved = time.monotonic()
    hub.task_moved("AST_2")
    assert await waiting is True
    assert time.monotonic() - moved < 0.05
    assert await hub.settled("AST_3", 0.01) is False, "nothing moved: the caller reads the row"


async def test_a_task_that_moved_just_before_the_wait_is_heard() -> None:
    """The request read its row, then the holder ended the task, then the request waits: it
    hears the end it missed at once, and the next wait waits again."""
    hub.task_moved("AST_6")
    started = time.monotonic()
    assert await hub.settled("AST_6", 5.0) is True
    assert time.monotonic() - started < 0.05
    assert await hub.settled("AST_6", 0.05) is False, "told once, not on every wait"


async def test_a_request_reads_its_frames_as_the_holder_writes_them() -> None:
    queue = hub.follow("AST_4")
    frame = pb.StreamResponse(status_update=pb.TaskStatusUpdateEvent(task_id="AST_4"))
    hub.publish("AST_4", frame)
    hub.publish("AST_5", pb.StreamResponse())  # another task's frames are not this request's
    assert await asyncio.wait_for(queue.get(), 1) == frame
    assert queue.empty()
    hub.unfollow("AST_4", queue)
    hub.publish("AST_4", frame)
    assert queue.empty()


async def test_the_hub_is_dropped_when_the_lane_idles() -> None:
    lane = holding("AT_idle")
    lane.running.pop("c1")[1].cancel()
    assert hub.held_here("AT_idle", "desk") is lane
    background = await lane.background(0.0)
    assert background.running == 0 and not background.ready
    assert hub.held_here("AT_idle", "desk") is None, "nothing runs: the lane is let go"


async def test_a_lane_held_elsewhere_is_reached_through_the_worker() -> None:
    sent: list[tuple[str, str, str]] = []
    hub.forward = lambda thread_id, lane, kind: sent.append((thread_id, lane, kind))
    try:
        hub.signal("AT_job", "desk", "cancel")
    finally:
        hub.forward = None
    assert sent == [("AT_job", "desk", "cancel")]


@pytest.mark.parametrize(
    "message",
    [
        proto.LaneSignal(thread_id="AT_1", lane="desk", kind="cancel"),
        proto.LaneHeld(thread_id="AT_1", lane="desk", holder="AW_1/p42/x", held=False),
        proto.TaskMoved(thread_id="AT_1", lane="desk", task_id="AST_1"),
    ],
)
def test_the_lane_messages_cross_the_ipc_channel(message: Any) -> None:
    assert channel._read_message(channel._write_message(message), proto.IPC_MESSAGES) == message


async def test_a_text_door_wakes_a_call_held_by_its_own_job() -> None:
    """The call holds the lane in a job with a loop of its own (a thread job, here): the
    worker's door finds no holder on its loop, and the worker reaches the job that announced
    the lane; the call's hub there wakes."""
    job_loop = asyncio.new_event_loop()
    ready = threading.Event()
    woke: asyncio.Future[float] = asyncio.get_running_loop().create_future()
    worker_loop = asyncio.get_running_loop()
    job_announced: list[tuple[str, str, str, bool]] = []

    async def call() -> None:
        hub.announce_with(lambda *args: job_announced.append(args))
        lane = hub.lane_hub("AT_call", "desk")
        lane.held = True
        hub.announce("AT_call", "desk", "AW_1/p1t2/h", True)
        ready.set()
        await lane.signaled(5.0)
        worker_loop.call_soon_threadsafe(woke.set_result, time.monotonic())
        hub.drop("AT_call", "desk")
        hub.announce_with(None)

    job = threading.Thread(target=job_loop.run_until_complete, args=(call(),))
    job.start()
    try:
        assert await asyncio.to_thread(ready.wait, 5)
        # the job's LaneHeld, as the worker's executor records it
        (announced,) = job_announced
        hub.job_holds(announced[0], announced[1], "AJ_1", announced[3])

        def forward(thread_id: str, lane: str, kind: str) -> None:
            # the executor's LaneSignal to the job, and the job's IPC loop waking its hub
            assert hub.job_holding(thread_id, lane) == "AJ_1"
            job_loop.call_soon_threadsafe(hub.wake, thread_id, lane)

        hub.forward = forward
        signaled = time.monotonic()
        hub.signal("AT_call", "desk")
        assert await asyncio.wait_for(woke, 5) - signaled < 0.05
    finally:
        hub.forward = None
        hub.job_holds("AT_call", "desk", "AJ_1", False)
        await asyncio.to_thread(job.join, 5)
        job_loop.close()


async def test_a_text_the_call_answered_wakes_its_request_through_the_worker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The call holds the lane in a job and answers a text: the job tells its worker the
    task ended, and the text door's request that waits on it hears it at once, with no poll
    of agent-db and no log that it polls."""
    from livekit.agents.storage.context import TurnHost
    from livekit.agents.storage.resolve import Thread

    job_loop = asyncio.new_event_loop()
    ready, answer = threading.Event(), threading.Event()
    worker_loop = asyncio.get_running_loop()

    async def call() -> None:
        # the job's IPC: TaskMoved, which the worker's executor turns into task_moved there
        hub.announce_with(
            lambda *args: None,
            moved=lambda thread_id, lane, task_id: worker_loop.call_soon_threadsafe(
                hub.task_moved, task_id
            ),
        )
        ready.set()
        await asyncio.to_thread(answer.wait, 5)
        hub.task_moved("AST_text", thread_id="AT_texted", lane="desk")
        hub.announce_with(None)

    hub.job_holds("AT_texted", "desk", "AJ_1", True)
    host: TurnHost[None] = TurnHost(
        lambda ctx: None,  # type: ignore[arg-type,return-value]
        thread=Thread(None, "AT_texted", agent_name="desk"),  # type: ignore[arg-type]
        state_type=None,
    )
    job = threading.Thread(target=job_loop.run_until_complete, args=(call(),))
    job.start()
    try:
        assert await asyncio.to_thread(ready.wait, 5)
        waiting = asyncio.ensure_future(host.settled("AST_text", 0.5))
        await asyncio.sleep(0.05)
        answered = time.monotonic()
        answer.set()
        assert await asyncio.wait_for(waiting, 5) is True
        assert time.monotonic() - answered < 0.1
    finally:
        hub.job_holds("AT_texted", "desk", "AJ_1", False)
        await asyncio.to_thread(job.join, 5)
        job_loop.close()
    assert "polls agent-db" not in caplog.text
