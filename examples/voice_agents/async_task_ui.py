"""Generic async-tool task tracker that mirrors tool progress to a custom frontend.

The agent's long-running tools report progress through ``ctx.update()`` and their
final return. This module turns *any* such tool call into a "task" the frontend can
render: a status (running/done/error/cancelled) plus a timeline of updates and the
result, each tagged with the lifecycle of the reply that voices it.

It does this without touching the tool bodies. ``AsyncTaskUI.instrument(agent)`` wraps
each function tool so that, per call, it:

- records the call as a task on entry (name + parsed args),
- transparently shadows ``ctx.update()`` to append timeline entries,
- records done/error/cancelled on exit.

The reply lifecycle (``pending -> generating -> spoken/interrupted/covered``) comes from
the framework's ``async_tool_reply`` session event, which links the buffered outputs to
the speech handle that voices them — closing the gap between an update landing in the
chat context and the agent actually speaking it.

State is pushed to the frontend over RPC: the agent sends full snapshots via
``client.task.sync``, and the frontend can pull the current snapshot on mount via
``agent.task.sync``. One JSON snapshot per change keeps the frontend a pure renderer.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from livekit import rtc
from livekit.agents import Agent, AgentSession, AsyncToolReplyEvent, RunContext
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.tool_executor import FINAL_CALL_ID_SUFFIX, update_call_id_suffix

logger = logging.getLogger("async-task-ui")

# RPC method names — keep in sync with the frontend.
RPC_PUSH = "client.task.sync"  # agent -> frontend: full snapshot on every change
RPC_SYNC = "agent.task.sync"  # frontend -> agent: pull the current snapshot on mount


@dataclass
class TimelineEntry:
    id: str  # synthetic call_id the framework assigns this update/result
    kind: str  # "update" | "result"
    text: str
    reply: str  # inline | pending | generating | spoken | interrupted | covered
    ts: float


@dataclass
class Task:
    id: str  # the tool call_id
    name: str
    args: dict[str, Any]
    status: str  # running | done | error | cancelled
    started_at: float
    timeline: list[TimelineEntry] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    ended_at: float | None = None


def _find_ctx(args: tuple[Any, ...], kwargs: dict[str, Any]) -> RunContext | None:
    for value in (*args, *kwargs.values()):
        if isinstance(value, RunContext):
            return value
    return None


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, ValueError):
        return {}


class AsyncTaskUI:
    """Tracks async tool calls and streams their state to a custom frontend over RPC."""

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._tasks: dict[str, Task] = {}  # call_id -> Task, in start order
        self._entry_owner: dict[str, str] = {}  # timeline entry id -> owning call_id
        room.local_participant.register_rpc_method(RPC_SYNC, self._on_sync)

    # -- wiring --

    def attach(self, session: AgentSession) -> None:
        """Subscribe to the reply lifecycle so timeline entries flip as replies are voiced."""
        session.on("async_tool_reply", self._on_async_tool_reply)

    def instrument(self, agent: Agent) -> None:
        """Wrap each of the agent's function tools so their calls become tracked tasks."""
        for tool in agent.tools:
            if not isinstance(tool, (FunctionTool, RawFunctionTool)):
                continue
            if tool.info.name.startswith("lk_agents_"):
                continue  # framework-internal cancel/list tools
            tool._func = self._wrap(tool._func)

    # -- tool wrapper --

    def _wrap(self, func: Any) -> Any:
        @functools.wraps(func)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            ctx = _find_ctx(args, kwargs)
            if ctx is None:
                return await func(*args, **kwargs)

            self._start(ctx)
            original_update = ctx.update

            async def tracked_update(message: Any, *, template: Any = None) -> None:
                # record the human-readable message before it's wrapped for the LLM
                self._update(ctx, message)
                await original_update(message, template=template)

            ctx.update = tracked_update  # instance attr shadows the bound method

            try:
                result = await func(*args, **kwargs)
            except asyncio.CancelledError:
                self._finish(ctx, status="cancelled")
                raise
            except Exception as e:
                self._finish(ctx, status="error", error=str(e))
                raise
            else:
                self._finish(ctx, status="done", result=result)
                return result

        return wrapped

    # -- task lifecycle --

    def _start(self, ctx: RunContext) -> None:
        fc = ctx.function_call
        self._tasks[fc.call_id] = Task(
            id=fc.call_id,
            name=fc.name,
            args=_parse_args(fc.arguments),
            status="running",
            started_at=time.time(),
        )
        self._push()

    def _update(self, ctx: RunContext, message: Any) -> None:
        task = self._tasks.get(ctx.function_call.call_id)
        if task is None:
            return
        # the step-th update; step 0 is voiced inline, later ones are deferred replies
        step = len(ctx._updates)
        entry_id = task.id + update_call_id_suffix(step)
        self._add_entry(
            task,
            entry_id,
            kind="update",
            text=str(message),
            reply="inline" if step == 0 else "pending",
        )
        self._push()

    def _finish(
        self, ctx: RunContext, *, status: str, result: Any = None, error: str | None = None
    ) -> None:
        task = self._tasks.get(ctx.function_call.call_id)
        if task is None:
            return
        task.status = status
        task.ended_at = time.time()
        if status == "error":
            task.error = error
        elif status == "done":
            text = "" if result is None else str(result)
            task.result = text
            # the final return is voiced as a deferred reply only when the tool also
            # emitted at least one ctx.update(); otherwise it's part of the normal turn
            if text and len(ctx._updates) > 0:
                entry_id = task.id + FINAL_CALL_ID_SUFFIX
                self._add_entry(task, entry_id, kind="result", text=text, reply="pending")
        self._push()

    def _add_entry(self, task: Task, entry_id: str, *, kind: str, text: str, reply: str) -> None:
        task.timeline.append(
            TimelineEntry(id=entry_id, kind=kind, text=text, reply=reply, ts=time.time())
        )
        self._entry_owner[entry_id] = task.id

    # -- reply lifecycle --

    def _on_async_tool_reply(self, ev: AsyncToolReplyEvent) -> None:
        affected = [entry for cid in ev.call_ids if (entry := self._entry_for(cid)) is not None]
        if not affected:
            return
        for entry in affected:
            entry.reply = "generating"
        self._push()

        def _on_speech_done(handle: SpeechHandle) -> None:
            for entry in affected:
                if handle.interrupted:
                    entry.reply = "interrupted"
                elif not handle.chat_items:
                    entry.reply = "covered"  # agent decided it was already covered
                else:
                    entry.reply = "spoken"
            self._push()

        ev.speech_handle.add_done_callback(_on_speech_done)

    def _entry_for(self, entry_id: str) -> TimelineEntry | None:
        task = self._tasks.get(self._entry_owner.get(entry_id, ""))
        if task is None:
            return None
        return next((e for e in task.timeline if e.id == entry_id), None)

    # -- transport --

    def _snapshot(self) -> str:
        return json.dumps({"tasks": [asdict(t) for t in self._tasks.values()]})

    def _push(self) -> None:
        payload = self._snapshot()
        for participant in list(self._room.remote_participants.values()):
            asyncio.create_task(self._push_to(participant.identity, payload))

    async def _push_to(self, identity: str, payload: str) -> None:
        try:
            await self._room.local_participant.perform_rpc(
                destination_identity=identity, method=RPC_PUSH, payload=payload
            )
        except Exception:
            logger.debug("task push to %s failed", identity, exc_info=True)

    async def _on_sync(self, data: rtc.RpcInvocationData) -> str:
        return self._snapshot()
