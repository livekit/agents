"""Answering requests with one ``AgentSession``, and knowing which answer belongs to which.

One session holds the whole conversation, and each request is one turn of it. What goes back
is attributed by lineage rather than guessed: a request owns the speech its turn produced,
the tool calls that speech made, the deferred replies to those calls, and anything said from
inside one of its tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Coroutine
from types import TracebackType
from typing import TYPE_CHECKING, Any

from ..llm.chat_context import ChatItem, FunctionCall, FunctionCallOutput
from ..log import logger
from ..utils import aio
from ..voice.events import (
    MESSAGE_SOURCE_KEY,
    ErrorEvent,
    SpeechCreatedEvent,
    ToolCallEnded,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
    ToolReplyUpdated,
)
from ..voice.served_request import ServedRequest
from ..voice.speech_handle import SpeechHandle
from ._types import TaskInput, TaskUpdate

if TYPE_CHECKING:
    from ..voice.agent_session import AgentSession

REQUEST_ID_KEY = "request_id"
"""``extra`` key on the items a request produced, so a stored history says which."""

_DELTA_PREAMBLE = "What the caller and the agent said since the last request:"


class RequestRun:
    """One request in flight: the updates it produces, until it declares a terminal state.

    Read it under ``async with``; closing it early stops the work where it can be stopped.
    """

    def __init__(self, runner: SessionRunner, task_input: TaskInput, request_id: str) -> None:
        self._runner = runner
        self._input = task_input
        self._request_id = request_id
        self._served = ServedRequest(metadata=dict(task_input.metadata))
        self._ch = aio.Chan[TaskUpdate]()
        self._finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._ended = False

        self.speeches: dict[str, SpeechHandle] = {}
        self.open_calls: dict[str, str] = {}
        """call id -> tool name, for calls of this turn that started and have not ended."""
        self.awaiting_reply: set[str] = set()
        """calls whose return reached the coalescer, or whose reply was scheduled, and not yet
        the other: the two events land in either order."""
        self.pending_replies: set[str] = set()
        """speech ids of deferred replies scheduled and not yet done."""
        self.last_word = ""
        self.concluded_at = 0.0
        self.cancelled: list[str] = []
        self.cancelled_at = 0.0
        self._abandoned = False

        self._task = asyncio.create_task(self._run(), name="RequestRun._run")
        self._task.add_done_callback(lambda _: self._ch.close())

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def served(self) -> ServedRequest:
        return self._served

    def finished(self) -> bool:
        return self._finished.done()

    async def _run(self) -> None:
        try:
            await self._runner._feed(self)
        except Exception as exc:
            logger.exception("failed to start a request", extra={"request_id": self._request_id})
            self._push(TaskUpdate(state="failed", text=str(exc) or type(exc).__name__))
            return
        await self._finished

    async def _abandon(self) -> None:
        """The caller stopped listening: stop the work where it can be stopped.

        Run before the reading task is cancelled, not from inside its cancellation: stopping
        a call is itself awaitable, and a cancelling task cannot wait for it.
        """
        self._abandoned = True
        if not self._finished.done():
            self._finished.set_result(None)
        for handle in self.speeches.values():
            if not handle.done():
                handle.interrupt()
        for call_id in list(self.open_calls):
            await self._stop_call(call_id)

    async def _stop_call(self, call_id: str) -> None:
        with contextlib.suppress(Exception):
            await self._runner.session.cancel_tool_call(call_id)

    def _push(self, update: TaskUpdate) -> None:
        if self._finished.done():
            return
        if update.state == "completed" and self._served.directive is not None:
            update.directive = self._served.directive
        self._ch.send_nowait(update)
        if update.state != "working":
            self._finished.set_result(None)

    # -- lineage

    def claim(self, handle: SpeechHandle) -> None:
        self.speeches[handle.id] = handle
        self._runner._by_speech[handle.id] = self
        self._runner._orphans.pop(handle.id, None)
        for item in handle.chat_items:
            self.on_item(item)
        handle.add_item_added_callback(self.on_item)
        handle.add_done_callback(lambda _: self.maybe_finish())

    def on_item(self, item: ChatItem) -> None:
        if item.type == "function_call":
            item.extra.setdefault(REQUEST_ID_KEY, self._request_id)
            self._runner._by_call[item.call_id] = self
            self.open_calls[item.call_id] = item.name
            if self._abandoned:
                # the turn recorded this call before the executor dispatched it, so the
                # abandon that already ran could not reach it
                self._runner._spawn(self._stop_call(item.call_id))
                return
            # a call is structure with nothing to say, so it travels without relayed text
            self._push(TaskUpdate(item=item))
            return

        if item.type != "message" or item.role != "assistant":
            return

        item.extra.setdefault(REQUEST_ID_KEY, self._request_id)
        text = item.text_content or ""
        source = item.extra.get(MESSAGE_SOURCE_KEY)
        if source != "turn_end":
            # a line from inside a tool, or one the expert said outright
            self._push(TaskUpdate(text=text, item=item, verbatim=source == "say"))
            return

        self.last_word, self.concluded_at = text, time.monotonic()
        if self.open_work(besides=None):
            # a conclusion with work still open announces that work
            self._push(TaskUpdate(text=text, item=item))
        else:
            self._push(TaskUpdate(state="completed", text=text, item=item))

    def on_tool_call_updated(self, update: ToolCallUpdated) -> None:
        """A tool's own report, relayed as it was written; the model never answers it."""
        if update.silent:
            return
        # the framework records the report as a call and an output but surfaces neither, so
        # the pair is rebuilt here in the shape the protocol names
        name = self.open_calls.get(update.call_id, "")
        call = FunctionCall(call_id=update.id, name=name, arguments="", update_of=update.call_id)
        call.extra[REQUEST_ID_KEY] = self._request_id
        self._push(TaskUpdate(text=update.message, item=call))

    def on_tool_call_ended(self, update: ToolCallEnded) -> None:
        name = self.open_calls.pop(update.call_id, "")
        if update.status == "cancelled":
            self.cancelled.append(name)
            self.cancelled_at = time.monotonic()
        elif update.id.endswith("_final") and update.message is not None:
            # a released tool's return, or its error, goes to the coalescer for a reply; a
            # None return after an update files none
            self.awaiting_reply ^= {update.call_id}
            output = FunctionCallOutput(
                call_id=update.id, name=name, output=update.message, is_error=False
            )
            self._push(TaskUpdate(item=output))
        self.maybe_finish()

    def on_reply_updated(self, update: ToolReplyUpdated, handle: SpeechHandle | None) -> None:
        if update.status == "scheduled":
            for call_id in update.call_ids:
                if self._runner._by_call.get(call_id) is self:
                    self.awaiting_reply ^= {call_id}
            self.pending_replies.add(update.speech_id)
            if handle is not None:
                self.claim(handle)
        else:
            self.pending_replies.discard(update.speech_id)
            self.maybe_finish()

    # -- completion

    def open_work(self, *, besides: str | None = None) -> bool:
        return bool(self.open_calls or self.awaiting_reply or self.pending_replies - {besides})

    def maybe_finish(self) -> None:
        if self._finished.done():
            return
        if any(not handle.done() for handle in self.speeches.values()) or self.open_work():
            return
        if self.cancelled and self.concluded_at < self.cancelled_at:
            # work of this turn was stopped and nothing was concluded after it
            what = self.last_word or f"{', '.join(self.cancelled)} was cancelled before it finished"
            self._push(TaskUpdate(state="canceled", text=what))
        else:
            self._push(TaskUpdate(state="completed", text=self.last_word))

    # -- reading

    async def aclose(self) -> None:
        await self._abandon()
        await aio.cancel_and_wait(self._task)
        self._ch.close()
        self._runner._forget(self)

    async def __anext__(self) -> TaskUpdate:
        if self._ended:
            raise StopAsyncIteration
        try:
            update = await self._ch.__anext__()
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
    """One conversation's session, and the requests fed through it.

    The session is started by whoever owns it and handed over here, so the handler decides
    what the agent is and this decides nothing about it. Requests are turns of that one
    session, taken in the order they arrive; the activity's scheduler serializes generation
    and playout, so there is no queue here.
    """

    def __init__(self, session: AgentSession) -> None:
        self._session = session
        self._by_call: dict[str, RequestRun] = {}
        self._by_speech: dict[str, RequestRun] = {}
        self._orphans: dict[str, SpeechHandle] = {}
        """speeches nothing has claimed: a deferred reply before its event, or on_enter."""
        self._seen_items: set[str] = set()
        """conversation items already shown, so each request carries only what is new."""
        self._live: list[RequestRun] = []
        self._setup = asyncio.Lock()
        self._attached = False
        self._chores: set[asyncio.Task[None]] = set()

    @property
    def session(self) -> AgentSession:
        return self._session

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run cleanup that a synchronous callback discovered it needs."""
        task = asyncio.create_task(coro)
        self._chores.add(task)
        task.add_done_callback(self._chores.discard)

    def attach(self) -> None:
        """Listen to the session. Called before the first request is fed."""
        if self._attached:
            return
        self._attached = True
        # the expert relays a tool's report as written; a model round would restate it
        self._session._reply_to_tool_updates = False
        self._session.on("speech_created", self._on_speech_created)
        self._session.on("tool_execution_updated", self._on_tool_execution_updated)
        self._session.on("error", self._on_error)

    def submit(self, task_input: TaskInput, *, request_id: str) -> RequestRun:
        run = RequestRun(self, task_input, request_id)
        self._live.append(run)
        self._session._served_request = run.served
        return run

    async def _feed(self, run: RequestRun) -> None:
        async with self._setup:
            await self._show_conversation(run)
            handle = self._session.generate_reply(user_input=run._input.body)
            # whatever was said before this turn, such as on_enter, belongs to it
            for orphan in list(self._orphans.values()):
                run.claim(orphan)
            run.claim(handle)

    async def _show_conversation(self, run: RequestRun) -> None:
        """Give the session what was said since its last request, ahead of this one."""
        lines: list[str] = []
        for item in run._input.chat_ctx.items:
            if item.type != "message" or item.id in self._seen_items:
                continue
            self._seen_items.add(item.id)
            if item.role in ("user", "assistant") and (text := item.text_content):
                lines.append(f"{'caller' if item.role == 'user' else 'agent'}: {text}")
        if not lines:
            return

        agent = self._session.current_agent
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.add_message(role="user", content=f"{_DELTA_PREAMBLE}\n" + "\n".join(lines))
        await agent.update_chat_ctx(chat_ctx)

    def _forget(self, run: RequestRun) -> None:
        with contextlib.suppress(ValueError):
            self._live.remove(run)
        self._by_call = {k: v for k, v in self._by_call.items() if v is not run}
        self._by_speech = {k: v for k, v in self._by_speech.items() if v is not run}
        self._session._served_request = self._live[-1].served if self._live else None

    # -- routing: every session event reaches the request it belongs to

    def _on_speech_created(self, ev: SpeechCreatedEvent) -> None:
        from ..voice.agent import _get_activity_task_info

        handle = ev.speech_handle
        if handle.id in self._by_speech:
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
            return  # the call reached us through its speech
        if update.type == "tool_reply_updated":
            handle = self._orphans.get(update.speech_id)
            for reply_owner in {self._by_call[c] for c in update.call_ids if c in self._by_call}:
                reply_owner.on_reply_updated(update, handle)
            return
        if (owner := self._by_call.get(update.call_id)) is None:
            return
        if update.type == "tool_call_updated":
            owner.on_tool_call_updated(update)
        else:
            owner.on_tool_call_ended(update)

    def _on_error(self, ev: ErrorEvent) -> None:
        for run in list(self._by_speech.values()):
            run._push(TaskUpdate(state="failed", text=str(ev.error)))

    async def aclose(self) -> None:
        for run in list(self._live):
            run._push(TaskUpdate(state="failed", text="the session was closed"))
            await run.aclose()
        if self._chores:
            await asyncio.gather(*self._chores, return_exceptions=True)
        self._session._served_request = None


__all__ = ["REQUEST_ID_KEY", "RequestRun", "SessionRunner"]
