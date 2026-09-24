"""Durable tools: tool coroutines whose frame is pickled, so a restarted worker resumes them.

A durable tool awaits only ``EffectCall``s. The scheduler runs each effect outside the frame
and snapshots the frame at every boundary, the moment the tool can be captured: after an
effect resolved and before the next one is sent, or while it awaits an ``AgentTask``.
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

    @classmethod
    def from_exception(cls, exc: BaseException) -> EffectException:
        return cls(exc_type=type(exc).__name__, exc_message=str(exc)).with_traceback(
            exc.__traceback__
        )

    def __str__(self) -> str:
        if self.exc_message:
            return f"{self.exc_type}: {self.exc_message}"
        return self.exc_type


TaskResult_T = TypeVar("TaskResult_T")


class EffectCall(Generic[TaskResult_T]):
    """Run an awaitable outside the durable tool's frame, which keeps only its outcome.

    Awaiting it hands the awaitable to the scheduler, which resumes the tool with the result
    or the exception. The awaitable itself is never pickled, so an effect in flight at a
    crash runs again on resume; ``RunContext.idempotency_key`` names it stably across the two.
    """

    def __init__(self, aw: Awaitable[TaskResult_T] | AgentTask[TaskResult_T]) -> None:
        self._c: Awaitable[TaskResult_T] | AgentTask[TaskResult_T] | None = aw
        self._c_result: Any = None
        self._c_exc: EffectException | None = None
        self._c_ctx: contextvars.Context | None = None
        self._done: bool = False

    @classmethod
    def _from_exception(cls, exc: BaseException) -> EffectCall:
        ec = cls(None)  # type: ignore[arg-type]
        ec._set_exception(exc)
        return ec

    def __await__(self) -> Generator[Any, Any, TaskResult_T]:
        self._c_ctx = contextvars.copy_context()
        return yields(self)  # type: ignore

    def _set_result(self, value: Any) -> None:
        self._c_result = value
        self._c_exc = None
        self._done = True

    def _set_exception(self, exc: BaseException) -> None:
        self._c_exc = EffectException.from_exception(exc)
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
            return f"EffectCall(status=error, exception=EffectException({self._c_exc.exc_type}))"
        return f"EffectCall(status=done, result={reprlib.repr(self._c_result)})"


class DurableInvalidStateError(RuntimeError):
    pass


_CURRENT_TASK = contextvars.ContextVar["DurableTask"]("agents_durable_task")


@dataclass
class DurableTask:
    generator: Generator | bytes
    fnc_name: str
    next_value: EffectCall | None = None
    metadata: Any = None
    effects: int = 0
    """How many effects the tool has sent, which numbers the next one."""
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
        return (self.__class__, (g, self.fnc_name, self.next_value, self.metadata, self.effects))

    def unpickle_generator(self) -> Self:
        if isinstance(self.generator, bytes):
            self.generator = pickle.loads(self.generator)
        return self


class DurableScheduler:
    """Runs one activity's durable tools, and snapshots each at its boundaries."""

    def __init__(
        self,
        *,
        on_boundary: Callable[[], Awaitable[None]] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._tasks: dict[asyncio.Task[Any], DurableTask] = {}
        self._on_boundary = on_boundary

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
        exe_task.add_done_callback(lambda _: self._tasks.pop(exe_task))
        _pass_through_activity_task_info(exe_task)
        return exe_task

    @property
    def at_boundary(self) -> bool:
        """Whether every durable tool can be captured now."""
        return all(task.at_boundary.is_set() for task in self._tasks.values())

    def durable_state(self) -> bytes:
        """Each running tool as of its latest boundary, empty when none runs."""
        snapshots = [task.snapshot for task in self._tasks.values() if task.snapshot]
        return pickle.dumps(snapshots) if snapshots else b""

    def close(self) -> None:
        for task in self._tasks.keys():
            task.cancel()

    async def _boundary(self, task: DurableTask) -> None:
        # the frame is written before the next effect is sent, so a crash re-runs only that one
        task.at_boundary.set()
        try:
            task.snapshot = pickle.dumps(task)
        except Exception:
            # the tool keeps running; a crash before its next boundary resumes an older one
            logger.exception("could not snapshot a durable tool", extra={"function": task.fnc_name})
            return
        if self._on_boundary is not None:
            try:
                await self._on_boundary()
            except Exception:
                logger.exception(
                    "could not persist a durable tool", extra={"function": task.fnc_name}
                )

    async def _execute(self, task: DurableTask) -> Any:
        from .voice.agent import _pass_through_activity_task_info

        __tracebackhide__ = True

        async def _execute_step(ec: EffectCall) -> None:
            try:
                if not ec._c or ec._c_ctx is None:
                    raise RuntimeError("invalid EffectCall state")

                exe_task = ec._c_ctx.run(asyncio.ensure_future, ec._c, loop=self._loop)
                _pass_through_activity_task_info(exe_task)
                ec._set_result(await exe_task)
            except Exception as e:
                logger.exception("error executing step of durable function")
                ec._set_exception(e)

        g = task.generator
        assert not isinstance(g, bytes)
        nv: EffectCall | Any = task.next_value
        token = _CURRENT_TASK.set(task)
        try:
            while True:
                try:
                    if nv is None or (isinstance(nv, EffectCall) and nv._done):
                        await self._boundary(task)
                        task.at_boundary.clear()
                        if nv is None:
                            nv = g.send(None)
                        else:
                            nv = g.throw(nv._c_exc) if nv._c_exc else g.send(nv._c_result)
                    # else: restored while awaiting an AgentTask, which is awaited again

                    if isinstance(nv, EffectCall):
                        if task.next_value is not nv:
                            task.next_value = nv
                            task.effects += 1
                        if isinstance(nv._c, AgentTask):
                            # a pending AgentTask pickles, so awaiting one is a boundary too
                            await self._boundary(task)
                        await _execute_step(nv)
                        task.at_boundary.clear()
                        assert nv._done
                    else:
                        exc = DurableInvalidStateError(
                            f"Unsupported awaitable yielded: {nv!r}.\n"
                            "Durable functions may only await supported operations.\n"
                            "You awaited something that can't be checkpointed/replayed.\n"
                            ">> Wrap it in EffectCall(...)."
                        )
                        nv = EffectCall._from_exception(exc)
                        task.next_value = nv

                except StopIteration as e:
                    return e.value
        finally:
            _CURRENT_TASK.reset(token)
