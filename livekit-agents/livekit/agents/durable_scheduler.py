"""Durable tools: tool coroutines whose frame is pickled, so a loaded session resumes them.

A tool is captured at a boundary: after an ``EffectCall`` resolved and before the next is sent,
or while it awaits an ``AgentTask``. A save holds every tool at one, so no effect runs twice.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import pickle
import reprlib
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass, field
from types import coroutine
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from typing_extensions import Self

from livekit.durable.function import DurableCoroutine, durable

from .llm.tool_context import StopResponse, ToolError
from .log import logger
from .voice.agent import Agent, AgentTask

if TYPE_CHECKING:
    from .voice.agent_session import AgentSession

_REHYDRATING = contextvars.ContextVar[tuple["AgentSession", dict[str, Agent]]]("agents_rehydrating")
"""While frames are unpickled: the session, and its agents by id."""


def _lookup_rehydrated_agent(cls: type[Agent], agent_id: str) -> Agent:
    rehydrating = _REHYDRATING.get(None)
    agent = rehydrating[1].get(agent_id) if rehydrating is not None else None
    if not isinstance(agent, cls):
        raise RuntimeError(f"no {cls.__name__} with id {agent_id} to rehydrate")
    return agent


@coroutine
@durable
def yields(n: Any) -> Generator[Any, Any, Any]:
    return (yield n)


class EffectException(Exception):
    """A picklable stand-in for the exception an ``EffectCall``'s awaitable raised."""

    __slots__ = ("exc_type", "exc_message")

    def __init__(self, exc_type: str, exc_message: str) -> None:
        self.exc_type = exc_type
        self.exc_message = exc_message
        super().__init__(self.__str__())

    def __reduce__(self) -> tuple[type, tuple[str, str]]:
        # the snapshot after a failed effect holds one, and __init__ takes both fields back
        return (self.__class__, (self.exc_type, self.exc_message))

    def __str__(self) -> str:
        if self.exc_message:
            return f"{self.exc_type}: {self.exc_message}"
        return self.exc_type


TaskResult_T = TypeVar("TaskResult_T")


class EffectCall(Generic[TaskResult_T]):
    """Run an awaitable outside the durable tool's frame, which keeps only its outcome."""

    def __init__(self, aw: Awaitable[TaskResult_T] | AgentTask[TaskResult_T]) -> None:
        self._c: Awaitable[TaskResult_T] | AgentTask[TaskResult_T] | None = aw
        self._c_result: Any = None
        self._c_exc: Exception | None = None
        self._c_ctx: contextvars.Context | None = None
        self._done: bool = False

    def __await__(self) -> Generator[Any, Any, TaskResult_T]:
        self._c_ctx = contextvars.copy_context()
        return yields(self)  # type: ignore

    def _set_result(self, value: Any) -> None:
        self._c_result = value
        self._c_exc = None
        self._done = True

    def _set_exception(self, exc: BaseException) -> None:
        # the framework's own exceptions reach the frame as themselves, so the model sees them
        stored: Exception
        if isinstance(exc, ToolError):
            stored = ToolError(exc.message)
        elif isinstance(exc, StopResponse):
            stored = StopResponse()
        else:
            stored = EffectException(type(exc).__name__, str(exc))
        self._c_exc = stored.with_traceback(exc.__traceback__)
        self._done = True

    def __getstate__(self) -> dict[str, Any]:
        # a pending AgentTask pickles as a reference to the rebuilt agent, anything else cannot
        if not self._done and not isinstance(self._c, AgentTask):
            raise TypeError("Cannot pickle an unresolved EffectCall")

        return {
            "done": self._done,
            "c_result": self._c_result,
            "c_exc": self._c_exc,
            "c": self._c if not self._done else None,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._done = state["done"]
        self._c_result = state["c_result"]
        self._c_exc = state["c_exc"]
        self._c = state["c"]
        self._c_ctx = contextvars.copy_context()

    def __repr__(self) -> str:
        if not self._done:
            return f"EffectCall(status=pending, aw={self._c})"
        if self._c_exc is not None:
            return f"EffectCall(status=error, exception={self._c_exc!r})"
        return f"EffectCall(status=done, result={reprlib.repr(self._c_result)})"


class DurableInvalidStateError(RuntimeError):
    pass


@dataclass
class DurableTask:
    generator: Generator | bytes
    fnc_name: str
    next_value: EffectCall | None = None
    metadata: Any = None
    at_boundary: asyncio.Event = field(default_factory=asyncio.Event)
    snapshot: bytes = b""
    """The task pickled at its latest boundary."""

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        # the generator is pickled on its own, so the rest restores even when it cannot
        g = (
            pickle.dumps(self.generator)
            if not isinstance(self.generator, bytes)
            else self.generator
        )
        return (self.__class__, (g, self.fnc_name, self.next_value, self.metadata))

    def unpickle_generator(self) -> Self:
        if isinstance(self.generator, bytes):
            self.generator = pickle.loads(self.generator)
        return self


class DurableScheduler:
    """Runs one activity's durable tools, and snapshots each at its boundaries."""

    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._tasks: dict[asyncio.Task[Any], DurableTask] = {}
        # cleared to hold every tool at its next boundary
        self._running = asyncio.Event()
        self._running.set()
        # the frames of the tools a close stopped at a boundary
        self._stopped: list[bytes] = []

    def execute(
        self, fnc: Callable[[], DurableCoroutine] | DurableTask, *, metadata: Any | None = None
    ) -> asyncio.Task[Any]:
        from .voice.agent import _pass_through_activity_task_info

        if isinstance(fnc, DurableTask):
            task = fnc.unpickle_generator()
        else:
            try:
                if isinstance(fnc, functools.partial):
                    fnc_name = fnc.func.__qualname__
                else:
                    fnc_name = fnc.__qualname__
            except AttributeError:
                fnc_name = "<unknown function>"

            task = DurableTask(fnc().__await__(), fnc_name=fnc_name, metadata=metadata)

        exe_task = self._loop.create_task(self._execute(task), name=task.fnc_name)
        self._tasks[exe_task] = task
        exe_task.add_done_callback(lambda _: self._tasks.pop(exe_task, None))
        _pass_through_activity_task_info(exe_task)
        return exe_task

    async def pause(self) -> None:
        """Hold every tool at its next boundary, and return once each is at one."""
        self._running.clear()
        waiting = [task for task in self._tasks.values() if not task.at_boundary.is_set()]
        if waiting:
            logger.info(
                "waiting for durable tools to finish their effect in flight",
                extra={"functions": [task.fnc_name for task in waiting]},
            )
            await asyncio.gather(*(task.at_boundary.wait() for task in waiting))

    def resume(self) -> None:
        self._running.set()

    def durable_state(self) -> bytes:
        """Every tool as of its latest boundary, pickled, or empty when there is none."""
        snapshots = [task.snapshot for task in self._tasks.values() if task.snapshot]
        snapshots += self._stopped
        return pickle.dumps(snapshots) if snapshots else b""

    def close(self) -> None:
        """Stop every tool; one at a boundary keeps its frame, one mid-effect is lost."""
        for exe_task, task in self._tasks.items():
            if task.at_boundary.is_set() and task.snapshot:
                self._stopped.append(task.snapshot)
            else:
                logger.warning(
                    "a durable tool stopped before its effect returned, so it is lost",
                    extra={"function": task.fnc_name},
                )
            exe_task.cancel()
        self._tasks.clear()

    def _capture(self, task: DurableTask) -> None:
        task.at_boundary.set()
        try:
            task.snapshot = pickle.dumps(task)
        except Exception:
            # a tool that does not pickle here keeps running, and has no frame to save
            task.snapshot = b""
            logger.exception("could not snapshot a durable tool", extra={"function": task.fnc_name})

    async def _execute(self, task: DurableTask) -> Any:
        from .voice.agent import _pass_through_activity_task_info

        __tracebackhide__ = True

        g = task.generator
        assert not isinstance(g, bytes)
        nv: EffectCall | Any = task.next_value
        while True:
            try:
                if nv is None or (isinstance(nv, EffectCall) and nv._done):
                    self._capture(task)
                    await self._running.wait()
                    task.at_boundary.clear()
                    if nv is None:
                        nv = g.send(None)
                    else:
                        nv = g.throw(nv._c_exc) if nv._c_exc else g.send(nv._c_result)
                # else: restored while awaiting an AgentTask, which is awaited again

                if isinstance(nv, EffectCall):
                    task.next_value = nv
                    if isinstance(nv._c, AgentTask):
                        # a pending AgentTask pickles, so awaiting one is a boundary too
                        self._capture(task)
                    try:
                        if not nv._c or nv._c_ctx is None:
                            raise RuntimeError("invalid EffectCall state")
                        exe_task = nv._c_ctx.run(asyncio.ensure_future, nv._c, loop=self._loop)
                        _pass_through_activity_task_info(exe_task)
                        if isinstance(nv._c, AgentTask):
                            # a frame stopped here leaves the task to the session's close
                            exe_task.add_done_callback(lambda t: t.cancelled() or t.exception())
                            nv._set_result(await asyncio.shield(exe_task))
                        else:
                            nv._set_result(await exe_task)
                    except Exception as e:
                        if not isinstance(e, (ToolError, StopResponse)):
                            logger.exception("error executing step of durable function")
                        nv._set_exception(e)
                    task.at_boundary.clear()
                    assert nv._done
                else:
                    exc = DurableInvalidStateError(
                        f"Unsupported awaitable yielded: {nv!r}.\n"
                        "Durable functions may only await supported operations.\n"
                        "You awaited something that can't be saved and resumed.\n"
                        ">> Wrap it in EffectCall(...)."
                    )
                    nv = EffectCall(None)  # type: ignore[arg-type]
                    nv._set_exception(exc)
                    task.next_value = nv

            except StopIteration as e:
                return e.value


def durable_chain(agent: Agent | None) -> dict[Agent, DurableScheduler | None]:
    """The agent and the agents its tasks return to, each with its activity's durable tools."""
    chain: dict[Agent, DurableScheduler | None] = {}
    while agent is not None and agent not in chain:
        chain[agent] = agent._activity._durable_scheduler if agent._activity else None
        agent = agent._old_agent if isinstance(agent, AgentTask) else None
    return chain
