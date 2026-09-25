"""Answering requests with one ``AgentSession``, and knowing which answer belongs to which.

One session holds the whole context, and each request is one turn of it. What goes back
is attributed by lineage rather than guessed: a request owns the speech its turn produced,
the tool calls that speech made, the deferred replies to those calls, and anything said from
inside one of its tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Container, Coroutine
from types import TracebackType
from typing import TYPE_CHECKING, Any

from ..llm.chat_context import ChatItem, FunctionCall, FunctionCallOutput
from ..log import logger
from ..utils import aio
from ..voice.events import (
    TURN_ENDED_KEY,
    CloseEvent,
    SpeechCreatedEvent,
    ToolCallEnded,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
    ToolReplyUpdated,
)
from ..voice.served_request import ServedRequest
from ..voice.speech_handle import SpeechHandle
from ..voice.tool_executor import cancel_tool_call
from .types import TaskInput, TaskUpdate

if TYPE_CHECKING:
    from ..voice.agent_session import AgentSession

TASK_ID_KEY = "lk.task_id"
"""``extra`` key naming the A2A task an item belongs to: on the expert's side the task whose
request produced it, on the caller's the task that answered its delegate call."""

_RESULT_ENTRY = "_final"
"""What the executor suffixes a released tool's return entry with, to tell it from a report."""


class RequestRun:
    """One request in flight: the updates it produces, until it declares a terminal state.

    Read it under ``async with``; closing it early stops the work that can be stopped.
    """

    def __init__(self, runner: SessionRunner, task_input: TaskInput, task_id: str) -> None:
        self._runner = runner
        self._input = task_input
        self._task_id = task_id
        self._served = ServedRequest(
            metadata=dict(task_input.metadata), is_delegation=task_input.is_delegation
        )
        self._event_ch = aio.Chan[TaskUpdate]()
        self._finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._terminal: TaskUpdate | None = None
        self._ended = False

        self.speeches: dict[str, SpeechHandle] = {}
        self.open_calls: dict[str, str] = {}
        """call id -> tool name, for calls of this turn that started and have not ended."""
        self.awaiting_reply: set[str] = set()
        """result entries whose return reached the coalescer, or whose reply was scheduled,
        and not yet the other: the two events land in either order and cancel out."""
        self.pending_replies: set[str] = set()
        """speech ids of deferred replies scheduled and not yet done."""
        self.last_word = ""
        self.concluded_at = 0.0
        self.cancelled: list[str] = []
        self.cancelled_at = 0.0
        self._stopping = False

        self._task = asyncio.create_task(self._run(), name="RequestRun._run")
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def served(self) -> ServedRequest:
        return self._served

    def finished(self) -> bool:
        return self._finished.done()

    async def _run(self) -> None:
        runner = self._runner
        try:
            async with runner._setup:
                await runner._sync_chat_ctx(self)
                handle = runner.session.generate_reply(user_input=self._input.body)
                # whatever was said before this turn, such as on_enter, belongs to it
                for orphan in list(runner._orphans.values()):
                    self.claim(orphan)
                self.claim(handle)
        except Exception as exc:
            logger.exception("failed to start a request", extra={"task_id": self._task_id})
            self._push_update(TaskUpdate(state="failed", text=str(exc) or type(exc).__name__))
            return
        await self._finished

    async def cancel(self) -> TaskUpdate:
        """Stop what can be stopped and say how the request ended.

        Best-effort: work that finished first is reported as finished, and a tool that does
        not allow cancellation runs on unheard.
        """
        self.cancelled_at = time.monotonic()
        await self._stop()
        self.maybe_finish()
        return self._terminal or TaskUpdate(state="canceled", text=self.last_word)

    async def _stop(self) -> None:
        """Interrupt this request's speeches and stop the calls that allow it.

        Awaited from a caller rather than from inside a cancellation, since stopping a call
        is itself awaitable. A speech that disallows interruption plays out unheard.
        """
        self._stopping = True
        for handle in self.speeches.values():
            if not handle.done() and handle.allow_interruptions:
                handle.interrupt()
        for call_id in list(self.open_calls):
            await self._cancel_tool_call(call_id)

    async def _cancel_tool_call(self, call_id: str) -> None:
        with contextlib.suppress(Exception):
            await cancel_tool_call(self._runner.session, call_id)

    def _push_update(self, update: TaskUpdate) -> None:
        if self._finished.done():
            return
        if update.state == "completed" and self._served.directive is not None:
            update.directive = self._served.directive
        self._event_ch.send_nowait(update)
        if update.state != "working":
            self._terminal = update
            self._finished.set_result(None)

    # -- lineage

    def claim(self, handle: SpeechHandle) -> None:
        if handle.id in self.speeches:
            # generate_reply emits speech_created before it returns, so its handle is parked
            # as an orphan and then claimed again as the turn's own
            return
        self.speeches[handle.id] = handle
        handle.request = self._served
        self._runner._orphans.pop(handle.id, None)
        for item in handle.chat_items:
            # the request is one A2A task, whose id each item it produced carries; an item
            # recorded before the request existed is relayed stamped and stored as it was
            if item.type in ("function_call", "message"):
                item = item.model_copy(update={"extra": {TASK_ID_KEY: self._task_id, **item.extra}})
            self.on_item(item, handle)
        handle._add_item_added_callback(lambda item: self.on_item(item, handle))
        handle.add_done_callback(lambda _: self.maybe_finish())

    def on_item(self, item: ChatItem, handle: SpeechHandle) -> None:
        if item.type == "function_call":
            item.extra.setdefault(TASK_ID_KEY, self._task_id)
            self._runner._by_call[item.call_id] = self
            self.open_calls[item.call_id] = item.name
            # a call is structure with nothing to say, so it travels without relayed text
            self._push_update(TaskUpdate(item=item))
            if self._stopping:
                # a stop that already ran could not reach a call the turn had not made yet
                self._runner._spawn(self._cancel_tool_call(item.call_id))
            return

        if item.type != "message" or item.role != "assistant":
            return

        item.extra.setdefault(TASK_ID_KEY, self._task_id)
        text = item.text_content or ""
        said = self._runner._speech_sources.get(handle.id) == "say"
        if said or not item.extra.get(TURN_ENDED_KEY):
            # a line said outright, or one on the way to a tool call
            self._push_update(TaskUpdate(text=text, item=item, verbatim=said))
            return

        self.last_word, self.concluded_at = text, time.monotonic()
        # the speech this arrived on does not count against itself
        if self.has_open_work(besides=handle.id):
            # a conclusion with work still open announces that work
            self._push_update(TaskUpdate(text=text, item=item))
        else:
            self._push_update(TaskUpdate(state="completed", text=text, item=item))

    def on_tool_call_updated(self, update: ToolCallUpdated) -> None:
        """A tool's own report: always an item, and its words too where nobody else says them."""
        # the framework records the report as a call and an output but surfaces neither, so
        # the pair is rebuilt here in the shape the protocol names
        name = self.open_calls.get(update.call_id, "")
        call = FunctionCall(call_id=update.id, name=name, arguments="", update_of=update.call_id)
        call.extra[TASK_ID_KEY] = self._task_id
        # a line of this session's own is about to carry the report, or nobody is to hear it
        text = "" if update.reply_pending or update.silent else update.message
        self._push_update(TaskUpdate(text=text, item=call))

    def on_tool_call_ended(self, update: ToolCallEnded) -> None:
        name = self.open_calls.pop(update.call_id, "")
        if update.status == "cancelled":
            self.cancelled.append(name)
            self.cancelled_at = time.monotonic()
        elif update.id.endswith(_RESULT_ENTRY) and update.message is not None:
            # a released tool's return, or its error, goes to the coalescer for a reply; a
            # None return after an update files none
            self.awaiting_reply ^= {update.id}
            output = FunctionCallOutput(
                call_id=update.id, name=name, output=update.message, is_error=False
            )
            self._push_update(TaskUpdate(item=output))
        self.maybe_finish()

    def on_result_delivered(self, call_id: str, covered: Container[str]) -> None:
        """A reply covers this call's result, whoever ends up saying it.

        Keyed on the result entry, not the call: a reply to one of the same call's progress
        reports names the call too and is not a result.
        """
        entry = f"{call_id}{_RESULT_ENTRY}"
        if entry in covered:
            self.awaiting_reply ^= {entry}
            self.maybe_finish()

    def on_reply_scheduled(self, update: ToolReplyUpdated, handle: SpeechHandle | None) -> None:
        self.pending_replies.add(update.speech_id)
        if handle is not None:
            self.claim(handle)

    def on_reply_done(self, update: ToolReplyUpdated) -> None:
        self.pending_replies.discard(update.speech_id)
        self.maybe_finish()

    # -- completion

    def has_open_work(self, *, besides: str | None = None) -> bool:
        return bool(self.open_calls or self.awaiting_reply or self.pending_replies - {besides})

    def maybe_finish(self) -> None:
        if self._finished.done():
            return
        if any(not handle.done() for handle in self.speeches.values()) or self.has_open_work():
            return
        if self.cancelled_at and self.concluded_at < self.cancelled_at:
            # work of this turn was stopped and nothing was concluded after it
            stopped = ", ".join(self.cancelled) or "the work"
            what = self.last_word or f"{stopped} was cancelled before it finished"
            self._push_update(TaskUpdate(state="canceled", text=what))
        else:
            self._push_update(TaskUpdate(state="completed", text=self.last_word))

    # -- reading

    async def aclose(self) -> None:
        """The caller stopped listening: stop the work and drop the request."""
        if not self._finished.done():
            self._finished.set_result(None)
        await self._stop()
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()
        self._runner._forget(self)

    async def __anext__(self) -> TaskUpdate:
        if self._ended:
            raise StopAsyncIteration
        try:
            update = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc from None
            raise RuntimeError("the session ended without answering the request") from None
        if update.state != "working":
            self._ended = True
        return update

    def __aiter__(self) -> RequestRun:
        return self

    async def __aenter__(self) -> RequestRun:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class SessionRunner:
    """One context's session, and the requests fed through it as turns.

    The activity's scheduler serializes generation and playout, so requests are taken in
    arrival order with no queue here. Listening starts here, before the first request.
    """

    def __init__(self, session: AgentSession) -> None:
        self._session = session

        self._by_call: dict[str, RequestRun] = {}
        self._orphans: dict[str, SpeechHandle] = {}
        """speeches nothing has claimed: a deferred reply before its event, or on_enter."""
        self._speech_sources: dict[str, str] = {}
        """speech id -> what drew it, since say() is what makes a speech's text verbatim."""
        self._live: list[RequestRun] = []
        self._setup = asyncio.Lock()
        self._chores: set[asyncio.Task[None]] = set()

        self._listeners: list[tuple[str, Any]] = [
            ("speech_created", self._on_speech_created),
            ("tool_execution_updated", self._on_tool_execution_updated),
            ("close", self._on_close),
        ]
        self._listening = True
        for event, listener in self._listeners:
            session.on(event, listener)  # type: ignore[arg-type]

    @property
    def session(self) -> AgentSession:
        return self._session

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run cleanup that a synchronous callback discovered it needs."""
        task = asyncio.create_task(coro)
        self._chores.add(task)
        task.add_done_callback(self._chores.discard)

    def submit(self, task_input: TaskInput, *, task_id: str) -> RequestRun:
        run = RequestRun(self, task_input, task_id)
        self._live.append(run)
        return run

    async def _sync_chat_ctx(self, run: RequestRun) -> None:
        """Take into this session whatever the caller said that it has not seen.

        The caller sends its history whole and the merge takes the delta by item id,
        so what this session did itself stays as it recorded it. The caller's plumbing —
        its calls, its handoffs, its instructions — is not what was said and does not travel.
        """
        if not run._input.chat_ctx.items:
            return

        agent = self._session.current_agent
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.merge(
            run._input.chat_ctx,
            exclude_function_call=True,
            exclude_instructions=True,
            exclude_config_update=True,
        )
        await agent.update_chat_ctx(chat_ctx)

    def _forget(self, run: RequestRun) -> None:
        with contextlib.suppress(ValueError):
            self._live.remove(run)
        self._by_call = {k: v for k, v in self._by_call.items() if v is not run}
        for speech_id in run.speeches:
            self._speech_sources.pop(speech_id, None)

    # -- routing: every session event reaches the request it belongs to

    def _on_speech_created(self, ev: SpeechCreatedEvent) -> None:
        from ..voice.agent import _get_activity_task_info

        handle = ev.speech_handle
        self._speech_sources[handle.id] = ev.source
        if handle.request is not None:
            return
        # a line said, or a reply drawn, from inside a tool belongs to the request that
        # called the tool, wherever its turn has got to by then
        current = asyncio.current_task()
        info = _get_activity_task_info(current) if current is not None else None
        if info is not None and info.function_call is not None:
            if (owner := self._by_call.get(info.function_call.call_id)) is not None:
                owner.claim(handle)
                return
        # a deferred reply names its calls in the event that follows; anything else waits for
        # the next request and is that one's
        self._orphans[handle.id] = handle

    def _on_tool_execution_updated(self, ev: ToolExecutionUpdatedEvent) -> None:
        update = ev.update
        if update.type == "tool_call_started":
            # the call itself reached us through its speech; this says the executor has it
            # now, which is the first moment a stop asked for earlier can land
            call_id = update.function_call.call_id
            owner = self._by_call.get(call_id)
            if owner is not None and owner._stopping and call_id in owner.open_calls:
                self._spawn(owner._cancel_tool_call(call_id))
            return
        if update.type == "tool_reply_updated":
            owners = [(c, self._by_call[c]) for c in update.call_ids if c in self._by_call]
            if not owners:
                return
            # one reply says one thing about several results, so one request carries it. The
            # newest is the one still live — an earlier request whose work this covers is
            # being superseded, and ends with what it had already said
            newest = owners[-1][1]
            if update.status == "scheduled":
                # the reply is registered before the results are cleared, or the request
                # carrying it would see no work left and finish without it
                newest.on_reply_scheduled(update, self._orphans.get(update.speech_id))
                covered = set(update.update_ids)
                for call_id, reply_owner in owners:
                    reply_owner.on_result_delivered(call_id, covered)
            else:
                newest.on_reply_done(update)
            return
        if (owner := self._by_call.get(update.call_id)) is None:
            return
        if update.type == "tool_call_updated":
            owner.on_tool_call_updated(update)
        else:
            owner.on_tool_call_ended(update)

    def _on_close(self, ev: CloseEvent) -> None:
        """The session is gone, so nothing will answer what is still in flight.

        The session decides what an error means — it ignores a recoverable one and closes
        itself once the unrecoverable ones pass its limit — so this is the one signal that
        a request will never be answered, whether an error caused it or not.
        """
        for run in list(self._live):
            run._push_update(
                TaskUpdate(state="failed", text=str(ev.error) if ev.error else "the session closed")
            )

    async def aclose(self) -> None:
        if self._listening:
            self._listening = False
            for event, listener in self._listeners:
                self._session.off(event, listener)  # type: ignore[arg-type]
        for run in list(self._live):
            run._push_update(TaskUpdate(state="failed", text="the session was closed"))
            await run.aclose()
        if self._chores:
            await asyncio.gather(*self._chores, return_exceptions=True)


__all__ = ["TASK_ID_KEY", "RequestRun", "SessionRunner"]
