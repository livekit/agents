"""An ``AgentSession`` bound to its rows in its conversation's database.

Imported only when ``start()`` is given ``persist``; it restores data into the agent and
session the handler built, and rebuilds an agent from its row only when its class says how.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import pickle
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, TypeAdapter

from .. import llm
from ..durable_scheduler import _REHYDRATING, DurableTask
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
from .agent_activity import AgentActivity
from .events import AgentStateChangedEvent, ConversationItemAddedEvent, ToolExecutionUpdatedEvent
from .generation import _DurableExecutionMetadata
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

        # the stored chain, newest first: the current agent, then those its tasks return to
        records: list[AgentRecord] = []
        record = stored.agents.get(stored.current_agent_id or agent.id)
        while record is not None and record not in records:
            records.append(record)
            record = stored.agents.get(record.parent_agent_id or "")

        # rebuilt oldest first: an agent that does not rebuild drops itself and all above it
        chain: list[tuple[Agent, AgentRecord | None]] = []
        for record in reversed(records):
            reason: str | None = None
            if record.agent_id == agent.id:
                chain.append((agent, record))
                continue
            try:
                cls = import_qualified(record.cls)
            except ImportError:
                reason = f"{record.cls} does not import"
            else:
                if not (isinstance(cls, type) and issubclass(cls, Agent)):
                    reason = f"{record.cls} is not an Agent"
                elif record.state is None:
                    reason = f"{record.cls} could not snapshot its state"
                else:
                    try:
                        chain.append((cls._from_state(record.state), record))
                        continue
                    except Exception as e:
                        reason = f"{record.cls}._from_state failed: {e}"
            logger.warning(
                "the stored agent could not be rebuilt, so the nearest agent that can resumes",
                extra={
                    "session_id": self._persisted.session_id,
                    "stored_agent_id": record.agent_id,
                    "reason": reason,
                },
            )
            break
        if not chain:
            chain = [(agent, stored.agents.get(agent.id))]

        for member, own in chain:
            if own is not None and own.chat_items:
                member._chat_ctx = llm.ChatContext(list(own.chat_items))

        # durable tools restored oldest first; a task above an agent resumes only when one of
        # its restored frames awaits it, since the tool that awaited it is what hands back
        kept = 0
        token = _REHYDRATING.set((session, {member.id: member for member, _ in chain}))
        try:
            for index, (member, own) in enumerate(chain):
                kept = index + 1
                newer = chain[index + 1][0] if index + 1 < len(chain) else None
                answered = {
                    item.call_id
                    for item in member._chat_ctx.items
                    if item.type == "function_call_output"
                }
                tasks: list[DurableTask] = []
                for snapshot in (
                    pickle.loads(own.durable_state) if own and own.durable_state else []
                ):
                    try:
                        restored: DurableTask = pickle.loads(snapshot)
                    except Exception:
                        logger.warning(
                            "a durable tool's state did not load, so it is lost",
                            extra={"session_id": self._persisted.session_id},
                            exc_info=True,
                        )
                        continue
                    metadata: _DurableExecutionMetadata = restored.metadata
                    call = llm.FunctionCall.model_validate_json(metadata.function_call)
                    # an answered call ended before its frame was cleared, so it does not rerun
                    if call.call_id not in answered:
                        tasks.append(restored)
                awaiting = next(
                    (
                        task
                        for task in tasks
                        if newer is not None
                        and task.next_value is not None
                        and task.next_value._c is newer
                    ),
                    None,
                )
                # a resumed task takes its activity too, so it goes on without running on_enter
                resumed_task = isinstance(member, AgentTask) and member._rehydrated
                failed = (
                    await AgentActivity(member, session)._rehydrate(tasks)
                    if tasks or resumed_task
                    else []
                )
                if not isinstance(newer, AgentTask):
                    break
                call_id = (
                    llm.FunctionCall.model_validate_json(awaiting.metadata.function_call).call_id
                    if awaiting is not None
                    else None
                )
                if call_id is not None and call_id not in failed:
                    logger.info(
                        "the AgentTask was awaited from a durable tool, so it resumes",
                        extra={"agent_id": newer.id, "call_id": call_id},
                    )
                    newer._rehydrated = True
                    newer._old_agent = member
                    continue
                logger.warning(
                    "the AgentTask was awaited from a tool that is not durable, so it is lost "
                    "with the tool and the agent that awaited it resumes",
                    extra={"agent_id": newer.id, "resumed_agent_id": member.id},
                )
                break
        finally:
            _REHYDRATING.reset(token)
        current, own = chain[kept - 1]
        if current.id != (stored.current_agent_id or agent.id) or not (own and own.chat_items):
            # a stand-in, or an agent with nothing of its own stored, starts from the history
            current._chat_ctx = session._chat_ctx.copy(
                exclude_handoff=True, exclude_config_update=True
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
            scheduler = agent._activity._durable_scheduler if agent._activity else None
            records.append(
                AgentRecord(
                    agent_id=agent.id,
                    cls=qualified_name(type(agent)),
                    parent_agent_id=parent.id if parent is not None else None,
                    state=state,
                    # a closed activity leaves the frames its tools stopped at
                    durable_state=scheduler.durable_state() if scheduler is not None else None,
                )
            )
        session = self._session
        await self._persisted.checkpoint(
            current_agent_id=session._agent.id if session._agent else None,
            userdata=_userdata_json(session._userdata),
            agents=records,
        )

    async def durable_boundary(self, agent: Agent) -> None:
        """Write the agent's durable tools as they stand at a boundary one of them reached."""
        scheduler = agent._activity._durable_scheduler if agent._activity else None
        if self._closed or self._lease_lost or scheduler is None:
            return
        self._sync()
        await self._persisted.write_durable_state(
            agent.id, cls=qualified_name(type(agent)), durable_state=scheduler.durable_state()
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
        # a turn that ended with no plain tool running and every durable one at a boundary is
        # the point nothing is half-written
        self._sync()
        if (
            self._session._agent_state == "listening"
            and not _RunningTasks.get(self._session)
            and all(
                agent._activity._durable_scheduler.at_boundary
                for agent in self._chain()
                if agent._activity is not None and agent._activity._durable_scheduler is not None
            )
        ):
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
