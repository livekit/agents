"""Restoring an ``AgentSession`` from its ``store.Session`` when it starts, and saving it.

Imported only when ``start()`` is given ``persist``; it restores data into the agent and
session the handler built, and rebuilds an agent from its row only when its class says how.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import json
import pickle
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, TypeAdapter

from .. import llm
from ..log import logger
from ..store.session import AgentRecord, Session
from .agent import Agent, AgentTask
from .agent_activity import AgentActivity
from .durable_tool import _REHYDRATING, DurableTask, durable_chain

if TYPE_CHECKING:
    from .agent_session import AgentSession
    from .tool_executor import _ToolExecutor

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


def _check_rebuild(agent: Agent) -> str | None:
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


async def _close_restored(activities: list[AgentActivity]) -> None:
    # nothing restored outlives a start that failed, so no frame runs without a session
    for activity in activities:
        activity._restored_tools.clear()
        with contextlib.suppress(Exception):
            await activity.aclose()
        activity.agent._activity = None
        if isinstance(activity.agent, AgentTask):
            activity.agent._rehydrated = False


async def rehydrate(session: AgentSession, persisted: Session, agent: Agent) -> tuple[Agent, bool]:
    """Load the session and restore what it had; returns the agent to start, and whether it
    is the one the session left off on."""
    _userdata_json(session._userdata)
    stored = await persisted.load()
    if stored is None:
        _check_rebuild(agent)
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
                "session_id": persisted.session_id,
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

    # durable tools restored oldest first; a task above an agent resumes only when one of its
    # restored frames awaits it, since the tool that awaited it is what hands back
    kept = 0
    restored: list[AgentActivity] = []
    token = _REHYDRATING.set((session, {member.id: member for member, _ in chain}))
    try:
        for index, (member, own) in enumerate(chain):
            kept = index + 1
            newer = chain[index + 1][0] if index + 1 < len(chain) else None
            tasks: list[DurableTask] = []
            for snapshot in pickle.loads(own.durable_state) if own and own.durable_state else []:
                try:
                    tasks.append(pickle.loads(snapshot))
                except Exception:
                    logger.warning(
                        "a durable tool's state did not load, so it is lost",
                        extra={"session_id": persisted.session_id},
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
                restored.append(activity)
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
    except BaseException:
        await _close_restored(restored)
        raise
    finally:
        _REHYDRATING.reset(token)
    current, own = chain[kept - 1]
    if current.id != (stored.current_agent_id or agent.id) or not (own and own.chat_items):
        # a stand-in, or an agent with nothing of its own stored, starts from the history
        current._chat_ctx = session._chat_ctx.copy(exclude_handoff=True, exclude_config_update=True)

    delegate = (session._opts.delegation | current._delegation).get("delegate")
    for endpoint in persisted.child_contexts:
        if delegate is None or delegate.endpoint != endpoint:
            logger.warning(
                "the session had a child session on an endpoint it has no delegate for",
                extra={"session_id": persisted.session_id, "endpoint": endpoint},
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
                    extra={"session_id": persisted.session_id},
                    exc_info=True,
                )

    logger.info(
        "rehydrated a persisted session",
        extra={
            "session_id": persisted.session_id,
            "items": len(stored.history),
            "agent_id": current.id,
        },
    )
    _check_rebuild(current)
    return current, current.id == stored.current_agent_id


async def discard(session: AgentSession, agent: Agent) -> None:
    """Close what the start restored or started from ``agent`` down, and let the session go,
    for a start that failed."""
    persisted, session._persisted = session._persisted, None
    await _close_restored(
        [member._activity for member in durable_chain(agent) if member._activity is not None]
    )
    if persisted is not None:
        with contextlib.suppress(Exception):
            await persisted.release()


async def save(
    session: AgentSession,
    chain: dict[Agent, _ToolExecutor | None] | None = None,
    *,
    release: bool = False,
) -> None:
    """Write what the session changed since the last save, its durable tools held at a boundary
    while captured; with ``release`` the session goes after, and a failed save is logged."""
    persisted = session._persisted
    if persisted is None:
        return
    if release:
        session._persisted = None
    # saves diff against the one before, so they land in turn; one queued behind the release
    # writes nothing
    async with persisted._lock:
        try:
            if chain is None:
                chain = durable_chain(session._agent)
            if not chain:
                # a session with no agent yet holds only what it loaded
                return
            executors = [executor for executor in chain.values() if executor is not None]
            try:
                for executor in executors:
                    await executor.pause()
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
                    if _check_rebuild(agent) is None:
                        with contextlib.suppress(Exception):
                            state = agent._snapshot_state()
                    parent = agent._old_agent if isinstance(agent, AgentTask) else None
                    snapshots = frames.durable_state() if frames else []
                    if agent._activity is not None:
                        # a restored frame waits for the start to finish before it runs
                        snapshots += [pickle.dumps(r.task) for r in agent._activity._restored_tools]
                    records.append(
                        AgentRecord(
                            agent_id=agent.id,
                            cls=_qualified_name(type(agent)),
                            parent_agent_id=parent.id if parent is not None else None,
                            state=state,
                            durable_state=pickle.dumps(snapshots) if snapshots else b"",
                            chat_items=items,
                        )
                    )
                try:
                    userdata = _userdata_json(session._userdata)
                except TypeError as e:
                    # the rest still saves; the userdata keeps its last saved value
                    logger.warning(str(e), extra={"session_id": persisted.session_id})
                    userdata = None
                await persisted.save(
                    current_agent_id=next(iter(chain)).id if chain else None,
                    userdata=userdata,
                    history=history,
                    agents=records,
                )
            finally:
                for executor in executors:
                    executor.resume()
        except Exception:
            if not release:
                raise
            logger.warning(
                "could not save the session on close",
                extra={"session_id": persisted.session_id},
                exc_info=True,
            )
        finally:
            if release:
                # a failed save still lets the session go, or the connection stays open
                with contextlib.suppress(Exception):
                    await persisted.release()


__all__ = ["discard", "rehydrate", "save"]
