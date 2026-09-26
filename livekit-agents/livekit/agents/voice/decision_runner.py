from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .. import utils
from ..decisions import DecisionModel, DecisionsCompletedEvent
from ..llm import ChatContext, ChatMessage
from ..log import logger
from ..telemetry import tracer
from ..utils import aio
from .events import ConversationItemAddedEvent

if TYPE_CHECKING:
    from .agent_activity import AgentActivity


class _DecisionRunner:
    """Activity-scoped worker with one running request and one replaceable pending snapshot."""

    def __init__(self, activity: AgentActivity, model: DecisionModel) -> None:
        self._activity = activity
        self._session = activity._session
        self._model = model
        self._definitions = activity.agent.decisions
        self._options = self._session.options.decision_options
        self._activity_id = utils.shortuuid("decision_activity_")
        self._turn_count = 0
        self._pending: tuple[ChatContext, str] | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        self._session.on("conversation_item_added", self._on_item)

    def _active(self) -> bool:
        return (
            not self._closed
            and not self._session._closing
            and self._session._activity is self._activity
            and self._session._next_activity is None
            and not self._activity._new_turns_blocked
        )

    def _on_item(self, ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not self._active() or not isinstance(item, ChatMessage):
            return
        if item.role != "user" or not item.text_content:
            return
        self._turn_count += 1
        if self._turn_count % self._options["turn_interval"]:
            return
        self._pending = (self._snapshot(item.id), item.id)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="agent_decisions")

    def _snapshot(self, source_id: str) -> ChatContext:
        messages: list[ChatMessage] = []
        for item in self._session.history.items:
            if isinstance(item, ChatMessage) and item.role in ("user", "assistant"):
                if text := item.text_content:
                    # Create new messages, including new content lists. ChatContext.copy()
                    # alone does not isolate the contents from edits to session history.
                    messages.append(
                        ChatMessage(
                            id=item.id,
                            role=item.role,
                            content=[text],
                            created_at=item.created_at,
                            interrupted=item.interrupted,
                        )
                    )
            if item.id == source_id:
                break
        turns = 0
        start = 0
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                turns += 1
                start = index
                if turns == self._options["max_context_turns"]:
                    break
        return ChatContext([*messages[start:]])

    async def _run(self) -> None:
        while self._pending is not None and self._active():
            chat_ctx, source_id = self._pending
            self._pending = None
            try:
                with tracer.start_as_current_span(
                    "decisions",
                    context=self._session._root_span_context,
                    attributes={
                        "lk.source_message_id": source_id,
                        "lk.agent_id": self._activity.agent.id,
                    },
                ):
                    response = await asyncio.wait_for(
                        self._model.evaluate(chat_ctx=chat_ctx, decisions=self._definitions),
                        timeout=self._options["timeout"],
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                # A failed sidecar pass must not interrupt the conversational reply or
                # prevent a newer pending snapshot from being evaluated.
                logger.exception(
                    "background decision evaluation failed", extra={"source_message_id": source_id}
                )
                continue
            if self._active():
                self._session.emit(
                    "decisions_completed",
                    DecisionsCompletedEvent(
                        results=response.results,
                        source_message_id=source_id,
                        agent_id=self._activity.agent.id,
                        activity_id=self._activity_id,
                        request_id=response.request_id,
                    ),
                )

    async def aclose(self) -> None:
        self._closed = True
        self._pending = None
        self._session.off("conversation_item_added", self._on_item)
        if self._task is not None:
            await aio.cancel_and_wait(self._task)
