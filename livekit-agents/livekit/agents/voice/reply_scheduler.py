from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from typing_extensions import TypedDict

from .. import utils
from ..llm.chat_context import ChatItem
from ..log import logger

if TYPE_CHECKING:
    from .agent import Agent
    from .agent_activity import AgentActivity
    from .agent_session import AgentSession
    from .speech_handle import SpeechHandle


REPLY_INSTRUCTIONS_AT_TAIL = """New results arrived from background tool calls (call_ids: {call_ids}).
Summarize the results naturally. Do NOT repeat information you have already told the user."""

REPLY_INSTRUCTIONS_MAYBE_COVERED = """New results arrived from background tool calls (call_ids: {call_ids}).
You may have already mentioned them in your most recent replies.
If you already told the user everything in these results, reply with an empty response (no text at all).
Otherwise, summarize only what you have not said yet, with a natural transition.
Never repeat information you have already told the user."""


ReplyStatus = Literal["completed", "interrupted", "skipped"]


class ReplyPromptArgs(TypedDict):
    """Arguments for a scheduled reply template."""

    call_ids: list[str]


ReplyTemplate = str | Callable[[ReplyPromptArgs], str]


class ReplyOptions(TypedDict):
    """Fully resolved templates used by the reply scheduler."""

    reply_at_tail_template: ReplyTemplate
    reply_maybe_covered_template: ReplyTemplate


class ReplyScheduledCallback(Protocol):
    def __call__(self, session: AgentSession, item_ids: list[str], speech_id: str, /) -> None: ...


class ReplyDoneCallback(Protocol):
    def __call__(
        self,
        session: AgentSession,
        item_ids: list[str],
        speech_id: str,
        status: ReplyStatus,
        /,
    ) -> None: ...


@dataclass
class _PendingReply:
    items: list[ChatItem]
    item_ids: list[str]
    source: object
    target: Agent


class _ReplyScheduler:
    """Coalesce already-created chat items and voice them when the session is idle."""

    def __init__(
        self,
        *,
        owning_activity: AgentActivity | None = None,
        reply_options: ReplyOptions | None = None,
        on_reply_scheduled: ReplyScheduledCallback | None = None,
        on_reply_done: ReplyDoneCallback | None = None,
    ) -> None:
        self._owning_activity = owning_activity
        self._reply_options: ReplyOptions = reply_options or {
            "reply_at_tail_template": REPLY_INSTRUCTIONS_AT_TAIL,
            "reply_maybe_covered_template": REPLY_INSTRUCTIONS_MAYBE_COVERED,
        }
        self._on_reply_scheduled = on_reply_scheduled
        self._on_reply_done = on_reply_done
        self._pending: list[_PendingReply] = []
        self._reply_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def reply_task(self) -> asyncio.Task[None] | None:
        return self._reply_task

    def set_owning_activity(self, activity: AgentActivity | None) -> None:
        self._owning_activity = activity

    def set_reply_options(self, options: ReplyOptions) -> None:
        self._reply_options = options

    def _mark_closed(self) -> None:
        self._closed = True

    async def enqueue(
        self,
        *,
        session: AgentSession,
        items: list[ChatItem],
        source: object,
        item_ids: list[str] | None = None,
    ) -> bool:
        async with self._lock:
            if self._closed:
                return False

            target = (
                self._owning_activity.agent
                if self._owning_activity is not None
                else session.current_agent
            )
            chat_ctx = target.chat_ctx.copy()
            chat_ctx.insert(items)
            await target.update_chat_ctx(chat_ctx)
            if self._closed:
                return False
            session.history.insert(items)

            if item_ids is None:
                item_ids = [item.call_id for item in items if item.type == "function_call_output"]
            self._pending.append(
                _PendingReply(items=items, item_ids=item_ids, source=source, target=target)
            )
            if self._reply_task is None or self._reply_task.done():
                self._reply_task = asyncio.create_task(
                    self._deliver_reply(session), name="reply_scheduler_deliver"
                )
                run_state = session._global_run_state
                if run_state is not None:
                    run_state._watch_handle(self._reply_task)
            return True

    async def aclose(self) -> None:
        self._mark_closed()
        reply_task = self._reply_task
        if reply_task is not None and not reply_task.done():
            await utils.aio.cancel_and_wait(reply_task)

        async with self._lock:
            self._pending.clear()
            self._reply_task = None

    async def _retarget_until_stable(
        self, session: AgentSession, pending: list[_PendingReply], target_agent: Agent
    ) -> Agent | None:
        """Copy pending items onto the current Agent, following handoffs until it is stable.

        ``session.generate_reply()`` targets whichever Agent is current when the reply runs,
        and a handoff can occur while ``update_chat_ctx()`` is awaited, so the current Agent
        is re-checked after every awaited insertion until it no longer changes. There is no
        ``await`` between the final stability check and returning, so the caller may safely
        call ``generate_reply()`` on the returned Agent.

        Each pending entry tracks the Agents it was already synced to; the per-Agent item-ID
        check is still required because ``ChatContext.insert()`` does not deduplicate by ID
        (an Agent's context may already contain an item this entry never delivered).

        Must be called while holding ``self._lock``. Returns the stable target Agent, or
        ``None`` if the scheduler closed while retargeting.
        """
        delivered_targets = [{id(entry.target)} for entry in pending]
        while True:
            target_id = id(target_agent)
            target_item_ids = {item.id for item in target_agent.chat_ctx.items}
            items_to_insert = [
                item
                for entry, delivered in zip(pending, delivered_targets, strict=True)
                if target_id not in delivered
                for item in entry.items
                if item.id not in target_item_ids
            ]
            if items_to_insert:
                logger.warning(
                    "agent handoff happened while waiting to deliver scheduled reply",
                    extra={"sources": [entry.source for entry in pending]},
                )
                chat_ctx = target_agent.chat_ctx.copy()
                chat_ctx.insert(items_to_insert)
                await target_agent.update_chat_ctx(chat_ctx)
            for delivered in delivered_targets:
                delivered.add(target_id)
            if self._closed:
                return None

            if self._owning_activity is not None:
                return target_agent

            current_agent = session.current_agent
            if current_agent is target_agent:
                return target_agent
            target_agent = current_agent

    async def _deliver_reply(self, session: AgentSession) -> None:
        # Module-level import cycles through agent_activity -> tool_executor -> this module.
        from .agent_activity import ActivityClosedError

        try:
            if self._owning_activity is not None:
                await self._owning_activity.wait_for_idle()
                target_agent = self._owning_activity.agent
            else:
                target_activity = await session.wait_for_idle()
                target_agent = target_activity.agent
        except ActivityClosedError:
            logger.debug("dropping scheduled reply — owning activity closed")
            self._pending.clear()
            return

        async with self._lock:
            pending = self._pending[:]
            pending_items = [item for entry in pending for item in entry.items]
            if not pending_items:
                return

            stable_target = await self._retarget_until_stable(session, pending, target_agent)
            if stable_target is None:
                self._pending.clear()
                return
            target_agent = stable_target

            self._pending.clear()
            at_tail = (items := target_agent.chat_ctx.items) and items[-1].id == pending_items[
                -1
            ].id
            template = (
                self._reply_options["reply_at_tail_template"]
                if at_tail
                else self._reply_options["reply_maybe_covered_template"]
            )
            item_ids = [item_id for entry in pending for item_id in entry.item_ids]
            instructions = (
                template({"call_ids": item_ids})
                if callable(template)
                else template.format(call_ids=item_ids)
            )
            speech = session.generate_reply(instructions=instructions, tool_choice="none")
            if self._on_reply_scheduled is not None:
                self._on_reply_scheduled(session, item_ids, speech.id)
            logger.debug(
                "generate scheduled reply",
                extra={"speech_id": speech.id, "updates_at_tail": at_tail},
            )

            def _on_speech_done(speech: SpeechHandle) -> None:
                status: ReplyStatus
                if speech.interrupted:
                    status = "interrupted"
                elif not speech.chat_items:
                    status = "skipped"
                else:
                    status = "completed"

                if not speech.chat_items:
                    logger.debug(
                        "scheduled reply was done without outputs",
                        extra={"speech_id": speech.id, "interrupted": speech.interrupted},
                    )

                if self._on_reply_done is not None:
                    self._on_reply_done(session, item_ids, speech.id, status)

            speech.add_done_callback(_on_speech_done)
