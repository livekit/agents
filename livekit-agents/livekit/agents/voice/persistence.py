"""An ``AgentSession`` bound to its rows in a session database.

Imported only when ``start()`` is given ``persist``; it restores data into the agent and
session the handler built, and rebuilds an agent from its row only when its class says how.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, TypeAdapter

from .. import llm
from ..log import logger
from ..store.session import (
    SESSION_OWNER,
    AgentRecord,
    LeaseLostError,
    PersistedSession,
    import_qualified,
    qualified_name,
)
from .agent import Agent, AgentTask
from .events import AgentStateChangedEvent, ConversationItemAddedEvent, ToolExecutionUpdatedEvent
from .tool_executor import _RunningTasks

if TYPE_CHECKING:
    from .agent_session import AgentSession

# per class, checked once: None when it rebuilds from its row, else why it cannot
_REBUILD_CHECKS: dict[type[Agent], str | None] = {}


def _userdata_json(userdata: Any) -> Any:
    """Userdata as the JSON its row stores; raises ``TypeError`` for anything else."""
    if userdata is None:
        return None
    cls = type(userdata)
    try:
        if not (dataclasses.is_dataclass(userdata) or isinstance(userdata, BaseModel)):
            json.dumps(userdata)
        adapter = TypeAdapter(cls)
        data = adapter.dump_python(userdata, mode="json")
        # JSON that reads back as something else, such as a tuple-keyed dict, is refused
        if adapter.validate_python(data) != userdata:
            raise ValueError("it does not read back from JSON as the same value")
    except Exception as e:
        raise TypeError(
            f"userdata of type {qualified_name(cls)} cannot be persisted ({e}); pass a "
            "dataclass, a pydantic model, or plain JSON data"
        ) from None
    return data


class SessionPersistence:
    """Keeps one session's rows current: appends as items land, checkpoints when quiet."""

    def __init__(self, session: AgentSession, persisted: PersistedSession) -> None:
        self._session = session
        self._persisted = persisted
        self._checkpoint_task: asyncio.Task[None] | None = None
        self._checkpoint_again = False
        self._lease_lost = False
        self._closed = False

    @property
    def persisted(self) -> PersistedSession:
        return self._persisted

    async def rehydrate(self, agent: Agent) -> Agent:
        """Claim the session and restore what it had; returns the agent to start."""
        session = self._session
        _userdata_json(session._userdata)
        stored = await self._persisted.load()
        if stored is None:
            self._check_rebuild(agent)
            self._listen()
            return agent

        session._chat_ctx = llm.ChatContext(list(stored.history))
        # the nearest agent up the stored chain that rebuilds, else the handler's
        current = agent
        own = stored.agents.get(stored.current_agent_id or agent.id)
        skipped: list[str] = []
        while own is not None and own.agent_id != agent.id:
            reason: str | None = None
            try:
                cls = import_qualified(own.cls)
            except ImportError:
                reason = f"{own.cls} does not import"
            else:
                if not (isinstance(cls, type) and issubclass(cls, Agent)):
                    reason = f"{own.cls} is not an Agent"
                elif issubclass(cls, AgentTask):
                    reason = "an AgentTask ends with the tool call that awaited it"
                elif own.state is None:
                    reason = f"{own.cls} could not snapshot its state"
                else:
                    try:
                        current = cls._from_state(own.state)
                        break
                    except Exception as e:
                        reason = f"{own.cls}._from_state failed: {e}"
            skipped.append(f"{own.agent_id}: {reason}")
            own = stored.agents.get(own.parent_agent_id or "")
        if skipped:
            logger.warning(
                "the stored agent could not be rebuilt, so the nearest agent that can resumes",
                extra={
                    "session_id": self._persisted.session_id,
                    "agent_id": current.id,
                    "skipped": skipped,
                },
            )

        delegation = session._opts.delegation | current._delegation
        delegate = delegation.get("delegate")
        if delegate is not None:
            # the delegate the activity will resolve, pointed back before anything is sent
            self._persisted.resume_delegate(delegate)
        for endpoint in stored.children:
            if delegate is None or delegate.endpoint != endpoint:
                logger.warning(
                    "the session had a child session on an endpoint it has no delegate for",
                    extra={"session_id": self._persisted.session_id, "endpoint": endpoint},
                )

        if own is not None and own.chat_items:
            current._chat_ctx = llm.ChatContext(list(own.chat_items))
        elif stored.history:
            # a stand-in, or an agent with nothing of its own stored, starts from the whole history
            current._chat_ctx = session._chat_ctx.copy(
                exclude_handoff=True, exclude_config_update=True
            )

        if stored.userdata is not None:
            if session._userdata is None:
                session._userdata = stored.userdata
            else:
                try:
                    adapter = TypeAdapter(type(session._userdata))
                    session._userdata = adapter.validate_python(stored.userdata)
                except Exception:
                    logger.warning(
                        "the stored userdata does not load into the handler's type, so the "
                        "handler's userdata is kept",
                        extra={"session_id": self._persisted.session_id},
                        exc_info=True,
                    )

        logger.info(
            "rehydrated a persisted session",
            extra={
                "session_id": self._persisted.session_id,
                "items": len(stored.history),
                "agent_id": current.id,
            },
        )
        self._check_rebuild(current)
        self._sync()
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
                    "_snapshot_state/_from_state to rebuild it, or the nearest agent above it "
                    "resumes instead",
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
        self._persisted.sync(self._session._chat_ctx.items, owner=SESSION_OWNER, prune=False)
        for agent in self._chain():
            self._persisted.sync(agent._chat_ctx.items, owner=agent.id, prune=True)

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
                )
            )
        session = self._session
        await self._persisted.checkpoint(
            current_agent_id=session._agent.id if session._agent else None,
            userdata=_userdata_json(session._userdata),
            agents=records,
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
                    extra={"session_id": self._persisted.session_id},
                )
                return
            except Exception:
                logger.warning(
                    "could not checkpoint the session",
                    extra={"session_id": self._persisted.session_id},
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
                extra={"session_id": self._persisted.session_id},
            )
        except Exception:
            logger.warning(
                "could not checkpoint the session on close",
                extra={"session_id": self._persisted.session_id},
                exc_info=True,
            )
        finally:
            self._closed = True
            # a fenced or failed checkpoint still lets the handle go, or the connection stays open
            with contextlib.suppress(Exception):
                await self._persisted.release()


__all__ = ["SessionPersistence"]
