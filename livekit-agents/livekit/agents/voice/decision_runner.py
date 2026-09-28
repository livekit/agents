from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .. import utils
from ..decisions import Decision, DecisionModel, DecisionResponse, DecisionsCompletedEvent
from ..llm import ChatContext, ChatItem, ChatMessage
from ..log import logger
from ..telemetry import tracer
from ..types import APIConnectOptions
from ..utils import aio
from .events import ConversationItemAddedEvent

if TYPE_CHECKING:
    from .agent_activity import AgentActivity


class _SessionDecisionModel(DecisionModel):
    """Share provider requests while keeping each session's metrics separate."""

    def __init__(self, model: DecisionModel) -> None:
        if isinstance(model, _SessionDecisionModel):
            model = model._model
        super().__init__(capabilities=model.capabilities)
        self._model: DecisionModel = model
        self._label = model.label
        self.on("metrics_collected", lambda ev: model.emit("metrics_collected", ev))

    @property
    def model(self) -> str:
        return self._model.model

    @property
    def provider(self) -> str:
        return self._model.provider

    async def _evaluate_impl(
        self,
        *,
        chat_ctx: ChatContext,
        decisions: Mapping[str, Decision],
        include_context_events: bool,
        conn_options: APIConnectOptions,
    ) -> DecisionResponse:
        return await self._model._evaluate_impl(
            chat_ctx=chat_ctx,
            decisions=decisions,
            include_context_events=include_context_events,
            conn_options=conn_options,
        )

    async def aclose(self) -> None:
        await self._model.aclose()


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
        history = self._session.history
        source_index = history.index_by_id(source_id)
        if source_index is None:
            return ChatContext.empty()

        items: list[ChatItem] = []
        user_turns = 0
        for index in range(source_index, -1, -1):
            item = history.items[index]
            if not isinstance(item, ChatMessage):
                if self._options["include_context_events"] and item.type in (
                    "function_call",
                    "function_call_output",
                    "agent_handoff",
                ):
                    items.append(item.model_copy(deep=True))
                continue
            if item.role not in ("user", "assistant") or not (text := item.text_content):
                continue
            # ChatContext.copy() shares message content with the live history.
            items.append(
                ChatMessage(
                    id=item.id,
                    role=item.role,
                    content=[text],
                    created_at=item.created_at,
                    interrupted=item.interrupted,
                )
            )
            if item.role == "user":
                user_turns += 1
                if user_turns == self._options["max_context_turns"]:
                    break
        while items and not (isinstance(items[-1], ChatMessage) and items[-1].role == "user"):
            items.pop()
        return ChatContext([*reversed(items)])

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
                        self._model.evaluate(
                            chat_ctx=chat_ctx,
                            decisions=self._definitions,
                            allow_partial=self._options["allow_partial"],
                            include_context_events=self._options["include_context_events"],
                        ),
                        timeout=self._options["timeout"],
                    )
            except Exception:
                logger.exception(
                    "background decision evaluation failed", extra={"source_message_id": source_id}
                )
                continue
            if self._active():
                self._session.emit(
                    "decisions_completed",
                    DecisionsCompletedEvent(
                        results=response.results,
                        errors=response.errors,
                        model=response.model,
                        provider=response.provider,
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
            self._task = None
