"""An ``AgentSession`` bound to its rows in its conversation's database.

Imported only when ``start()`` is given ``persist``; it restores data into the agent and
session the handler built, and rebuilds an agent from its row only when its class says how.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
import json
import pickle
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, TypeAdapter

from .. import llm
from ..durable_scheduler import _REHYDRATING, DurableScheduler, DurableTask, durable_chain
from ..log import logger
from ..store.session import AgentRecord, PersistedSession
from .agent import Agent, AgentTask
from .agent_activity import AgentActivity

if TYPE_CHECKING:
    from .agent_session import AgentSession

# per class, checked once: None when it rebuilds from its row, else why it cannot
_REBUILD_CHECKS: dict[type[Agent], str | None] = {}


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


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
            f"userdata of type {_qualified_name(cls)} cannot be persisted ({e}); pass a "
            "dataclass, a pydantic model, or plain JSON data"
        ) from None
    return data


class SessionPersistence:
    """Restores one session from its rows when it starts, and saves it to them."""

    def __init__(self, session: AgentSession, persisted: PersistedSession) -> None:
        self._session = session
        self._persisted = persisted
        # saves diff against the one before, so they land in turn, and none after the close's
        self._save_lock = asyncio.Lock()
        self._closed = False
        # the activities rehydrate restored, whose durable tools run once the session starts
        self._restored: list[AgentActivity] = []

    @property
    def persisted(self) -> PersistedSession:
        return self._persisted

    async def rehydrate(self, agent: Agent) -> tuple[Agent, bool]:
        """Load the session and restore what it had; returns the agent to start, and whether
        it is the one the session left off on."""
        session = self._session
        _userdata_json(session._userdata)
        stored = await self._persisted.load()
        if stored is None:
            self._check_rebuild(agent)
            return agent, False

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
                module_name, _, qualname = record.cls.partition(":")
                cls: Any = importlib.import_module(module_name)
                for part in qualname.split("."):
                    cls = getattr(cls, part)
            except (ImportError, AttributeError, ValueError):
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
                tasks: list[DurableTask] = []
                for snapshot in (
                    pickle.loads(own.durable_state) if own and own.durable_state else []
                ):
                    try:
                        tasks.append(pickle.loads(snapshot))
                    except Exception:
                        logger.warning(
                            "a durable tool's state did not load, so it is lost",
                            extra={"session_id": self._persisted.session_id},
                            exc_info=True,
                        )
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
                failed: list[str] = []
                if tasks or resumed_task:
                    activity = AgentActivity(member, session)
                    self._restored.append(activity)
                    failed = await activity._rehydrate(tasks)
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
        return current, current.id == stored.current_agent_id

    async def discard(self) -> None:
        """Close what the rehydrate restored and let the rows go, for a start that failed."""
        # nothing restored outlives the start, so no frame runs without a session
        for activity in self._restored:
            activity._restored_tools.clear()
            with contextlib.suppress(Exception):
                await activity.aclose()
            activity.agent._activity = None
            if isinstance(activity.agent, AgentTask):
                activity.agent._rehydrated = False
        self._restored.clear()
        with contextlib.suppress(Exception):
            await self._persisted.release()

    def resume_durable_tools(self) -> None:
        """Run the durable tools rehydrate restored, once the session has started."""
        for activity in self._restored:
            activity._resume_durable_tools()
        self._restored.clear()

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
                    extra={"cls": _qualified_name(cls)},
                )
        return _REBUILD_CHECKS[cls]

    async def save(self, chain: dict[Agent, DurableScheduler | None] | None = None) -> None:
        """Write what the session changed since the last save, with its durable tools held at
        a boundary while they are captured; ``chain`` is the one a close stopped."""
        async with self._save_lock:
            if self._closed:
                return
            if chain is None:
                chain = durable_chain(self._session._agent)
            schedulers = [scheduler for scheduler in chain.values() if scheduler is not None]
            try:
                for scheduler in schedulers:
                    await scheduler.pause()
                session = self._session
                history = list(session._chat_ctx.items)
                in_history = {item.id for item in history}
                records: list[AgentRecord] = []
                for agent, frames in chain.items():
                    items = list(agent._chat_ctx.items)
                    if agent._activity is not None:
                        # a step commits its finished tools once its last returns, which a
                        # tool awaiting a task holds past this save
                        known = {item.id for item in items}
                        for speech in agent._activity._background_speeches:
                            answered = {
                                item.call_id
                                for item in speech.chat_items
                                if item.type == "function_call_output"
                            }
                            for item in speech.chat_items:
                                if (
                                    item.type in ("function_call", "function_call_output")
                                    and item.call_id in answered
                                    and item.id not in known
                                ):
                                    known.add(item.id)
                                    items.append(item)
                                    if item.id not in in_history:
                                        in_history.add(item.id)
                                        history.append(item)
                    state: dict[str, Any] | None = None
                    if self._check_rebuild(agent) is None:
                        with contextlib.suppress(Exception):
                            state = agent._snapshot_state()
                    parent = agent._old_agent if isinstance(agent, AgentTask) else None
                    records.append(
                        AgentRecord(
                            agent_id=agent.id,
                            cls=_qualified_name(type(agent)),
                            parent_agent_id=parent.id if parent is not None else None,
                            state=state,
                            durable_state=frames.durable_state() if frames else b"",
                            chat_items=items,
                        )
                    )
                try:
                    userdata = _userdata_json(session._userdata)
                except TypeError as e:
                    # the rest still saves; the userdata keeps its last saved value
                    logger.warning(str(e), extra={"session_id": self._persisted.session_id})
                    userdata = None
                await self._persisted.save(
                    current_agent_id=next(iter(chain)).id if chain else None,
                    userdata=userdata,
                    history=history,
                    agents=records,
                )
            finally:
                for scheduler in schedulers:
                    scheduler.resume()

    async def aclose(self, chain: dict[Agent, DurableScheduler | None]) -> None:
        """Save the session as it closed and let its rows go."""
        try:
            await self.save(chain)
        except Exception:
            logger.warning(
                "could not save the session on close",
                extra={"session_id": self._persisted.session_id},
                exc_info=True,
            )
        finally:
            # a failed save still lets the handle go, or the connection stays open
            self._closed = True
            with contextlib.suppress(Exception):
                await self._persisted.release()


__all__ = ["SessionPersistence"]
