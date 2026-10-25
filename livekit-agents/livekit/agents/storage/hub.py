"""The lanes this process holds, and what their requests tell each other (docs/delegation.md).

A ``live`` lane's holder keeps the lane while its tools run in the background. Everything that
must reach it from a request in the same process goes through the lane's hub, not through
agent-db: a message that landed in the inbox, a cancel, a tool that ended. The request that waits
on its task hears from the hub when a turn of that task ended, and reads its frames from the
hub as the holder writes them; the durable stream stays the record. The holder's live state
object is kept here too, so a tool in the background writes the state the next turn commits.

A hub belongs to one event loop: a job run on a thread has its own, reached like a job process,
through the worker's IPC. A request that finds no holder in its process polls agent-db, the
backup path for a holder in another process or on another worker.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from a2a.types import a2a_pb2 as pb

from .._generated.storage_turn import Background
from ..llm import FunctionCall, ToolError
from ..log import logger
from ..voice.events import RunContext
from ..voice.generation import ToolExecutionOutput, make_tool_output

MOVED_MEMORY = 60.0
"""How long a task's last end is remembered for a waiter that starts just after it, in seconds."""

SignalHook = Callable[[str, str, str], None]
"""``(thread_id, lane, kind)``: a signal for a lane that a job of this worker may hold."""
AnnounceHook = Callable[[str, str, str, bool], None]
"""``(thread_id, lane, holder, held)``: a job tells its worker it holds a lane, or let it go."""
MovedHook = Callable[[str, str, str], None]
"""``(thread_id, lane, task_id)``: a job tells its worker a turn of its lane ended a task."""


@dataclass
class LaneHub:
    """One lane of one thread, on this loop: whether a turn here holds it, its tools in the
    background and the outputs that wait for their task's turn, and the signals between its
    requests and its holder."""

    thread_id: str
    lane: str
    held: bool = False
    """A turn of this loop holds the lane under ``live``."""
    state: Any = None
    """The holder's live state object, while it holds the lane."""
    running: dict[str, tuple[str, asyncio.Task[Any], RunContext[Any, Any]]] = field(
        default_factory=dict
    )
    ready: dict[str, list[ToolExecutionOutput]] = field(default_factory=dict)
    arrived: bool = False
    """A message landed in the lane's inbox, or a client asked to cancel a task, since the
    holder last looked."""
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    """Set by anything the holder or a tool watcher waits for."""
    signals: int = 0
    """How many signals came: a watcher reads it before it checks, and waits for a newer one."""

    def start(
        self,
        task_id: str,
        fnc_call: FunctionCall,
        work: asyncio.Task[Any],
        run_ctx: RunContext[Any, Any],
    ) -> None:
        """``work`` runs ``fnc_call`` for the task ``task_id``; its end waits for that task."""
        self.running[fnc_call.call_id] = (task_id, work, run_ctx)

        def done(t: asyncio.Task[Any]) -> None:
            self.running.pop(fnc_call.call_id, None)
            final = FunctionCall(
                call_id=f"{fnc_call.call_id}_final",
                name=fnc_call.name,
                arguments=fnc_call.arguments,
                extra=dict(fnc_call.extra),
            )
            if t.cancelled():
                out = make_tool_output(
                    fnc_call=final, output=None, exception=ToolError("the task was canceled")
                )
            elif (e := t.exception()) is not None:
                logger.warning(
                    "a tool in the background raised",
                    extra={"function": fnc_call.name, "taskId": task_id, "error": str(e)},
                )
                out = make_tool_output(fnc_call=final, output=None, exception=e)
            else:
                out = make_tool_output(fnc_call=final, output=t.result(), exception=None)
            self.ready.setdefault(task_id, []).append(out)
            self.wake.set()

        work.add_done_callback(done)

    def cancel(self, task_ids: list[str]) -> None:
        """Clients canceled these tasks: their tools stop at their next wait point outside the
        regions they protect, and the task's turn ends it ``canceled``."""
        for task_id, _, run_ctx in list(self.running.values()):
            if task_id in task_ids:
                run_ctx._cancel()

    def runs_for(self, task_id: str) -> bool:
        """Whether a tool of the task still runs."""
        return any(t == task_id for t, _, _ in self.running.values())

    def calls(self) -> list[RunContext[Any, Any]]:
        """The calls running in the background, each with its context."""
        return [run_ctx for _, _, run_ctx in self.running.values()]

    def take(self, task_id: str) -> list[ToolExecutionOutput]:
        """The finished outputs of a task, once."""
        return self.ready.pop(task_id, [])

    def signal(self) -> None:
        """A message landed in the inbox, or a client asked to cancel a task: the holder and
        the tool watchers look now."""
        self.arrived = True
        self.signals += 1
        self.wake.set()

    async def since(self, seen: int, wait: float) -> None:
        """Returns once a signal newer than the count ``seen`` came, or after ``wait`` seconds:
        one that came while the watcher checked is not lost."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while self.signals == seen and (left := deadline - loop.time()) > 0:
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), left)
            except asyncio.TimeoutError:
                return

    async def signaled(self, wait: float) -> None:
        """Returns on the next signal, or after ``wait`` seconds."""
        self.wake.clear()
        try:
            await asyncio.wait_for(self.wake.wait(), wait)
        except asyncio.TimeoutError:
            pass

    async def background(self, wait: float) -> Background:
        """What still runs, which tasks have output, and whether something arrived, after up
        to ``wait`` seconds when nothing has happened yet. With nothing left, the lane is let go
        and the hub is dropped."""
        if not self.ready and not self.arrived and self.running and wait > 0:
            await self.signaled(wait)
        arrived, self.arrived = self.arrived, False
        if not self.running and not self.ready:
            self.held = False
            self.state = None
            _hubs.pop((_loop_id(), self.thread_id, self.lane), None)
        return Background(running=len(self.running), ready=list(self.ready), arrived=arrived)


_hubs: dict[tuple[int, str, str], LaneHub] = {}
_moved: dict[tuple[int, str], asyncio.Event] = {}
_moved_at: dict[tuple[int, str], float] = {}
_seen: dict[tuple[int, str], float] = {}
_followers: dict[tuple[int, str], list[asyncio.Queue[pb.StreamResponse]]] = {}
_polling: set[tuple[str, str]] = set()
_announcers: dict[int, tuple[AnnounceHook, MovedHook | None]] = {}
_jobs: dict[tuple[str, str], str] = {}
forward: SignalHook | None = None
"""Set by the worker: reaches a lane that one of its jobs holds, through IPC."""


def _loop_id() -> int:
    return id(asyncio.get_running_loop())


def lane_hub(thread_id: str, lane: str) -> LaneHub:
    """The hub of one lane of one thread on this loop, made when missing."""
    key = (_loop_id(), thread_id, lane)
    hub = _hubs.get(key)
    if hub is None:
        hub = _hubs[key] = LaneHub(thread_id, lane)
    return hub


def held_here(thread_id: str, lane: str) -> LaneHub | None:
    """The lane's hub when a turn of this loop holds it."""
    hub = _hubs.get((_loop_id(), thread_id, lane))
    return hub if hub is not None and hub.held else None


def signal(thread_id: str, lane: str, kind: str = "arrival") -> None:
    """A message landed in the lane's inbox, or a client asked to cancel one of its tasks:
    the holder hears it at once, here or in a job of this worker."""
    if not wake(thread_id, lane) and forward is not None:
        forward(thread_id, lane, kind)


def wake(thread_id: str, lane: str) -> bool:
    """The lane's hub on this loop hears a signal; ``False`` when there is none."""
    hub = _hubs.get((_loop_id(), thread_id, lane))
    if hub is not None:
        hub.signal()
    return hub is not None


def drop(thread_id: str, lane: str) -> None:
    """The lane was let go here: its hub is dropped."""
    _hubs.pop((_loop_id(), thread_id, lane), None)


def announce_with(hook: AnnounceHook | None, *, moved: MovedHook | None = None) -> None:
    """A job's loop: how its lanes held for a whole call, and the tasks its turns ended, are
    told to its worker."""
    if hook is None:
        _announcers.pop(_loop_id(), None)
    else:
        _announcers[_loop_id()] = (hook, moved)


def announce(thread_id: str, lane: str, holder: str, held: bool) -> None:
    """A call on this loop holds a lane for the whole call, or let it go: its worker is told,
    so a text door of the worker signals it."""
    if (hooks := _announcers.get(_loop_id())) is not None:
        hooks[0](thread_id, lane, holder, held)


def job_holds(thread_id: str, lane: str, job_id: str, held: bool) -> None:
    """The worker's record of which of its jobs holds which lane."""
    if held:
        _jobs[(thread_id, lane)] = job_id
    elif _jobs.get((thread_id, lane)) == job_id:
        del _jobs[(thread_id, lane)]


def job_holding(thread_id: str, lane: str) -> str | None:
    return _jobs.get((thread_id, lane))


def task_moved(task_id: str, *, thread_id: str = "", lane: str = "") -> None:
    """A turn of the task ended in this loop: whoever waits on the task reads its row now. A
    job also tells its worker, whose text door may wait on the task (``thread_id`` and
    ``lane`` name it)."""
    if thread_id and (hooks := _announcers.get(_loop_id())) is not None and hooks[1]:
        hooks[1](thread_id, lane, task_id)
    key = (_loop_id(), task_id)
    now = time.monotonic()
    _moved_at[key] = now
    for memory in (_moved_at, _seen):
        for stale in [k for k, at in memory.items() if now - at > MOVED_MEMORY]:
            del memory[stale]
    if (event := _moved.pop(key, None)) is not None:
        event.set()


async def settled(task_id: str, wait: float) -> bool:
    """Waits up to ``wait`` seconds for a turn of the task to end in this loop. A turn that
    ended since the last wait on the task returned counts too: the caller read the row just
    before, and the end may have landed between that read and this wait."""
    key = (_loop_id(), task_id)
    moved, seen = _moved_at.get(key), _seen.get(key)
    if moved is not None and (seen is None or moved > seen):
        _seen[key] = time.monotonic()
        return True
    event = _moved.setdefault(key, asyncio.Event())
    try:
        await asyncio.wait_for(event.wait(), wait)
        return True
    except asyncio.TimeoutError:
        return False
    finally:
        _seen[key] = time.monotonic()
        if _moved.get(key) is event and not event.is_set():
            _moved.pop(key, None)


def polling(thread_id: str, lane: str) -> None:
    """A request waits on a lane whose holder is not in this process: agent-db is polled.
    Logged once per lane."""
    if (thread_id, lane) not in _polling:
        _polling.add((thread_id, lane))
        logger.info(
            "the lane's holder is not in this process; its request polls agent-db",
            extra={"threadId": thread_id, "lane": lane},
        )


def follow(stream_id: str) -> asyncio.Queue[pb.StreamResponse]:
    """The frames a holder of this loop writes to the stream, as it writes them."""
    queue: asyncio.Queue[pb.StreamResponse] = asyncio.Queue()
    _followers.setdefault((_loop_id(), stream_id), []).append(queue)
    return queue


def unfollow(stream_id: str, queue: asyncio.Queue[pb.StreamResponse]) -> None:
    key = (_loop_id(), stream_id)
    queues = _followers.get(key, [])
    if queue in queues:
        queues.remove(queue)
    if not queues:
        _followers.pop(key, None)


def publish(stream_id: str, frame: pb.StreamResponse) -> None:
    """A frame the holder appended to the stream, to the requests of this loop that follow it."""
    for queue in _followers.get((_loop_id(), stream_id), []):
        queue.put_nowait(frame)


__all__ = [
    "LaneHub",
    "announce",
    "announce_with",
    "drop",
    "job_holding",
    "job_holds",
    "wake",
    "follow",
    "held_here",
    "lane_hub",
    "polling",
    "publish",
    "settled",
    "signal",
    "task_moved",
    "unfollow",
]
