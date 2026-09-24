"""An ``AgentSession`` bound to its rows in a conversation database.

Imported only when ``AgentSession.start`` is given a ``state``, so a session that persists
nothing never loads the store. The handler builds the agent and the session as it does on
every cold start; this restores data into what it was given, and rebuilds an agent from its
row only when its class can say how.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import pickle
from typing import TYPE_CHECKING, Any

from .. import llm
from ..log import logger
from ..store.session_state import (
    SESSION_OWNER,
    AgentRecord,
    LeaseLostError,
    SessionState,
    StoredSession,
    import_qualified,
    item_json,
    qualified_name,
)
from .agent import Agent, AgentTask
from .events import AgentStateChangedEvent, ConversationItemAddedEvent, ToolExecutionUpdatedEvent
from .tool_executor import _RunningTasks

if TYPE_CHECKING:
    from .agent_session import AgentSession

INTERRUPTED_OUTPUT = "the call was interrupted before it finished; its outcome is unknown"

_REHYDRATING = contextvars.ContextVar["SessionPersistence"]("agents_rehydrating")

# a class is checked once, the first time it starts under a persisted session: None means it
# rebuilds, a string says why it cannot
_REBUILD_CHECKS: dict[type[Agent], str | None] = {}


def _fingerprint(item: llm.ChatItem) -> int:
    return hash(item_json(item))


def lookup_rehydrated_agent(cls: type[Agent], agent_id: str) -> Agent:
    """Resolve an agent pickled inside userdata to the session's own instance."""
    persistence = _REHYDRATING.get(None)
    if persistence is None:
        raise RuntimeError("an Agent can only be unpickled while its session is rehydrated")
    agent = persistence._agents.get(agent_id)
    record = persistence._stored.agents.get(agent_id) if persistence._stored else None
    if agent is None and record is not None and record.state is not None:
        # an agent the handler did not build this time, such as one held in userdata
        agent = cls._from_state(record.state)
        agent._chat_ctx = llm.ChatContext(list(record.chat_items))
        persistence._agents[agent_id] = agent
    if agent is None or not isinstance(agent, cls):
        raise RuntimeError(f"no {cls.__name__} with id {agent_id} to rehydrate")
    return agent


class SessionPersistence:
    """Keeps one session's rows current: appends as items land, checkpoints when quiet."""

    def __init__(self, session: AgentSession, state: SessionState) -> None:
        self._session = session
        self._state = state
        self._stored: StoredSession | None = None
        self._agents: dict[str, Agent] = {}
        # per owner, the fingerprint of each item as last written: an item finalized in place,
        # such as an interrupted message, differs from it and is written again
        self._written: dict[str, dict[str, int]] = {}
        self._checkpoint_task: asyncio.Task[None] | None = None
        self._checkpoint_again = False
        self._lease_lost = False
        self._closed = False

    @property
    def state(self) -> SessionState:
        return self._state

    async def rehydrate(self, agent: Agent) -> Agent:
        """Claim the session and restore what it had; returns the agent to start."""
        stored = self._stored = await self._state.load()
        self._agents[agent.id] = agent
        if stored is None:
            self._check_rebuild(agent)
            self._listen()
            return agent

        session = self._session
        session._chat_ctx = llm.ChatContext(list(stored.history))
        self._written[SESSION_OWNER] = {item.id: _fingerprint(item) for item in stored.history}
        for stored_agent in stored.agents.values():
            self._written[stored_agent.agent_id] = {
                item.id: _fingerprint(item) for item in stored_agent.chat_items
            }

        current = agent
        reason: str | None = None
        if stored.current_agent_id not in (None, agent.id):
            # an AgentTask runs inside a tool call that did not survive, so its parent resumes
            record: AgentRecord | None = stored.agents.get(stored.current_agent_id or "")
            cls: Any = None
            while record is not None:
                try:
                    cls = import_qualified(record.cls)
                except ImportError:
                    cls = None
                    break
                if not (isinstance(cls, type) and issubclass(cls, AgentTask)):
                    break
                record = stored.agents.get(record.parent_agent_id or "")

            if record is None:
                reason = "its row is missing"
            elif record.agent_id == agent.id:
                pass
            elif not (isinstance(cls, type) and issubclass(cls, Agent)):
                reason = f"{record.cls} does not import as an Agent"
            elif record.state is None:
                reason = f"{record.cls} could not snapshot its state"
            else:
                try:
                    current = cls._from_state(record.state)
                except Exception as e:
                    reason = f"{record.cls}._from_state failed: {e}"
            if reason is not None:
                logger.warning(
                    "the stored agent could not be rebuilt, so the root agent resumes with the "
                    "whole conversation instead",
                    extra={
                        "session_id": self._state.session_id,
                        "stored_agent_id": stored.current_agent_id,
                        "agent_id": agent.id,
                        "reason": reason,
                    },
                )
        self._agents[current.id] = current
        # resolved the way the activity will resolve it, so the delegate names its context
        # before anything is sent
        delegation = session._opts.delegation | current._delegation
        if (delegate := delegation.get("delegate")) is not None:
            await self._state.resume_delegate(delegate)

        own = stored.agents.get(current.id)
        if reason is None and own is not None and own.chat_items:
            current._chat_ctx = llm.ChatContext(list(own.chat_items))
        elif stored.history:
            # a stand-in for the agent that was running, or one with nothing of its own stored,
            # starts from the whole conversation
            current._chat_ctx = session._chat_ctx.copy(
                exclude_handoff=True, exclude_config_update=True
            )

        for task in stored.interrupted:
            # the model is told the call has no known outcome, rather than left to assume one
            for ctx in (session._chat_ctx, current._chat_ctx):
                if not any(
                    item.type == "function_call" and item.call_id == task.call_id
                    for item in ctx.items
                ):
                    ctx.insert(
                        llm.FunctionCall(
                            call_id=task.call_id, name=task.name, arguments=task.arguments or "{}"
                        )
                    )
                if not any(
                    item.type == "function_call_output" and item.call_id == task.call_id
                    for item in ctx.items
                ):
                    ctx.insert(
                        llm.FunctionCallOutput(
                            call_id=task.call_id,
                            name=task.name,
                            output=INTERRUPTED_OUTPUT,
                            is_error=True,
                        )
                    )

        if stored.userdata_encoding is not None:
            userdata = stored.userdata
            if stored.userdata_encoding == "pickle":
                token = _REHYDRATING.set(self)
                try:
                    userdata = pickle.loads(userdata)
                finally:
                    _REHYDRATING.reset(token)
            session._userdata = userdata

        logger.info(
            "rehydrated a persisted session",
            extra={
                "session_id": self._state.session_id,
                "items": len(stored.history),
                "agent_id": current.id,
                "interrupted": len(stored.interrupted),
            },
        )
        self._check_rebuild(current)
        self._sync()
        # queued behind the items above: a crash before they land leaves the calls running,
        # so the next owner tells the model again
        for task in stored.interrupted:
            self._state.task_ended(
                task.call_id, status="interrupted", output=INTERRUPTED_OUTPUT, is_error=True
            )
        self._listen()
        return current

    def _listen(self) -> None:
        self._session.on("conversation_item_added", self._on_item_added)
        self._session.on("agent_state_changed", self._on_quiet_candidate)
        self._session.on("tool_execution_updated", self._on_quiet_candidate)

    def _check_rebuild(self, agent: Agent) -> str | None:
        """Why the agent's class cannot be rebuilt from its row, or None when it can."""
        cls = type(agent)
        if cls not in _REBUILD_CHECKS:
            try:
                json.dumps(agent._snapshot_state())
                _REBUILD_CHECKS[cls] = None
            except Exception as e:
                _REBUILD_CHECKS[cls] = str(e)
                logger.warning(
                    f"{cls.__name__} cannot be rebuilt on resume: {e}; define "
                    "_snapshot_state/_from_state to rebuild it, or it resumes as the root agent",
                    extra={"cls": qualified_name(cls)},
                )
        return _REBUILD_CHECKS[cls]

    def _chain(self) -> list[Agent]:
        """The current agent and the agents its ``AgentTask``s return to."""
        chain: list[Agent] = []
        agent = self._session._agent
        while agent is not None and agent not in chain:
            chain.append(agent)
            agent = agent._old_agent if isinstance(agent, AgentTask) else None
        return chain

    def _sync(self) -> None:
        """Queue whatever the history and the agents' contexts gained or lost since last time."""
        if self._closed:
            return
        self._sync_owner(SESSION_OWNER, self._session._chat_ctx.items, remove=False)
        for agent in self._chain():
            self._sync_owner(agent.id, agent._chat_ctx.items, remove=True)

    def _sync_owner(self, owner: str, items: list[llm.ChatItem], *, remove: bool) -> None:
        written = self._written.setdefault(owner, {})
        present = set()
        for item in items:
            present.add(item.id)
            fingerprint = _fingerprint(item)
            if written.get(item.id) != fingerprint:
                self._state.append(item, owner=owner)
                written[item.id] = fingerprint
        if remove:
            for item_id in written.keys() - present:
                self._state.remove(item_id, owner=owner)
                del written[item_id]

    async def checkpoint(self) -> None:
        self._sync()
        records: list[AgentRecord] = []
        for agent in self._chain():
            state: dict[str, Any] | None = None
            if self._check_rebuild(agent) is None:
                with contextlib.suppress(Exception):
                    state = agent._snapshot_state()
            parent = agent._old_agent if isinstance(agent, AgentTask) else None
            records.append(
                AgentRecord(
                    agent_id=agent.id,
                    cls=qualified_name(type(agent)),
                    parent_agent_id=parent.id if parent is not None else None,
                    state=state,
                    tools=[tool.id for tool in agent.tools],
                )
            )
        session = self._session
        await self._state.checkpoint(
            current_agent_id=session._agent.id if session._agent else None,
            userdata=session._userdata,
            agents=records,
            tools=[tool.id for tool in session.tools],
        )

    def _schedule_checkpoint(self) -> None:
        if self._closed or self._lease_lost:
            return
        if self._checkpoint_task is not None and not self._checkpoint_task.done():
            self._checkpoint_again = True
            return
        self._checkpoint_task = asyncio.create_task(
            self._run_checkpoints(), name="session_checkpoint"
        )

    async def _run_checkpoints(self) -> None:
        while True:
            self._checkpoint_again = False
            try:
                await self.checkpoint()
            except LeaseLostError:
                self._lease_lost = True
                logger.error(
                    "another worker took this session, so it stops checkpointing",
                    extra={"session_id": self._state.session_id},
                )
                return
            except Exception:
                logger.warning(
                    "could not checkpoint the session",
                    extra={"session_id": self._state.session_id},
                    exc_info=True,
                )
            if not self._checkpoint_again:
                return

    def _on_item_added(self, ev: ConversationItemAddedEvent) -> None:
        self._sync()
        if ev.item.type == "agent_handoff":
            self._schedule_checkpoint()

    def _on_quiet_candidate(self, ev: AgentStateChangedEvent | ToolExecutionUpdatedEvent) -> None:
        # a turn that ended with no tool still running is the point nothing is half-written
        self._sync()
        if self._session._agent_state == "listening" and not _RunningTasks.get(self._session):
            self._schedule_checkpoint()

    async def aclose(self) -> None:
        """Checkpoint once more and let the session go."""
        self._session.off("conversation_item_added", self._on_item_added)
        self._session.off("agent_state_changed", self._on_quiet_candidate)
        self._session.off("tool_execution_updated", self._on_quiet_candidate)
        if self._checkpoint_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(self._checkpoint_task)
        try:
            if not self._lease_lost:
                await self.checkpoint()
        except LeaseLostError:
            logger.error(
                "another worker took this session before it closed",
                extra={"session_id": self._state.session_id},
            )
        except Exception:
            logger.warning(
                "could not checkpoint the session on close",
                extra={"session_id": self._state.session_id},
                exc_info=True,
            )
        finally:
            self._closed = True
            # a fenced or failed checkpoint still lets the handle go, or the connection stays open
            with contextlib.suppress(Exception):
                await self._state.release()


__all__ = ["INTERRUPTED_OUTPUT", "SessionPersistence", "lookup_rehydrated_agent"]
