from __future__ import annotations

import asyncio
import contextvars
import functools
import pickle
import reprlib
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass, field
from types import coroutine
from typing import Any, Generic, Self, TypeVar

from livekit.durable.function import DurableCoroutine, durable

from .log import logger
from .voice.agent import AgentTask


@coroutine
@durable
def yields(n: Any) -> Generator[Any, Any, Any]:
    return (yield n)


class EffectException(Exception):
    """
    Picklable exception representing a failure that occurred while
    executing the EffectCall's awaitable
    """

    __slots__ = ("exc_type", "exc_message")

    def __init__(self, exc_type: str, exc_message: str) -> None:
        self.exc_type = exc_type
        self.exc_message = exc_message
        super().__init__(self.__str__())

    @classmethod
    def from_exception(cls, exc: BaseException) -> EffectException:
        return cls(
            exc_type=type(exc).__name__,
            exc_message=str(exc),
        )

    def __str__(self) -> str:
        if self.exc_message:
            return f"{self.exc_type}: {self.exc_message}"

        return self.exc_type


TaskResult_T = TypeVar("TaskResult_T")


class EffectCall(Generic[TaskResult_T]):
    """
    An awaitable wrapper used to execute a coroutine outside of the current
    DurableFunction scheduler.

    Durable Functions require the entire execution stack to be serializable so
    that it can be safely checkpointed and replayed. Awaiting a coroutine
    directly would capture non-serializable state on the stack, including
    runtime objects (event loops, tasks, frames) as well as arbitrary
    user-defined state that cannot be reliably serialized.

    `EffectCall` solves this by acting as a lightweight, serializable boundary:
    when awaited, it yields control (and the effect instance) to an external
    runner that executes the wrapped coroutine outside of the DurableFunction
    scheduler. The runner must then resume execution by calling
    `set_result()` or `set_exception()`.

    The wrapped awaitable is runtime-only and is never serialized. Only the
    resolved result or error is retained, making completed `EffectCall`
    instances safe to pickle.
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


@dataclass
class DurableTask:
    generator: Generator | bytes
    fnc_name: str
    next_value: EffectCall | None = None
    metadata: Any | None = None
    at_checkpoint: asyncio.Event = field(default_factory=asyncio.Event)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        # pickle the generator separately, so if it failed to pickle, we can still restore the rest of the state
        # and it needs to be unpickled after the SpeechHandle is recreated
        g = (
            pickle.dumps(self.generator)
            if not isinstance(self.generator, bytes)
            else self.generator
        )

        # exclude the at_checkpoint event from the pickled state
        return (
            self.__class__,
            (g, self.fnc_name, self.next_value, self.metadata),
        )

    def unpickle_generator(self) -> Self:
        if isinstance(self.generator, bytes):
            self.generator = pickle.loads(self.generator)
        return self


class DurableScheduler:
    def __init__(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._tasks: dict[asyncio.Task[Any], DurableTask] = {}

        self._can_execute = asyncio.Event()
        self._can_execute.set()
        self._ckpt_lock = asyncio.Lock()

    def execute(
        self, fnc: Callable[[], DurableCoroutine] | DurableTask, *, metadata: Any | None = None
    ) -> asyncio.Task[Any]:
        from livekit.agents.voice.agent import _pass_through_activity_task_info

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

    async def checkpoint(self) -> bytes:
        async with self._ckpt_lock:
            self._can_execute.clear()

            try:
                # wait for all tasks to be ready to dump
                tasks = list(self._tasks.values())
                for task in tasks:
                    await task.at_checkpoint.wait()

                return pickle.dumps(tasks) if tasks else b""
            finally:
                self._can_execute.set()

    def checkpoint_no_wait(self) -> bytes:
        tasks = list(self._tasks.values())
        not_resolved = [task.fnc_name for task in tasks if not task.at_checkpoint.is_set()]
        if not_resolved:
            raise DurableInvalidStateError(
                "`checkpoint_no_wait` must be called when the executions are awaiting "
                f"an `AgentTask` or a resolved `EffectCall`. These executions are not ready: {not_resolved}"
            )
        return pickle.dumps(tasks) if tasks else b""

    def restore(self, states: bytes | list[DurableTask]) -> list[asyncio.Task[Any]]:
        if not states:
            return []

        tasks = pickle.loads(states) if isinstance(states, bytes) else states
        exe_tasks = []
        for task in tasks:
            exe_tasks.append(self.execute(task))
        return exe_tasks

    async def aclose(self) -> None:
        from livekit.agents.utils.aio import cancel_and_wait

        await cancel_and_wait(*self._tasks.keys())
        self._tasks.clear()

    async def _execute(self, task: DurableTask) -> Any:
        from livekit.agents import AgentTask
        from livekit.agents.voice.agent import _pass_through_activity_task_info

        __tracebackhide__ = True

        async def _execute_step(ec: EffectCall) -> None:
            try:
                if not ec._c or ec._c_ctx is None:
                    raise RuntimeError("invalid EffectCall state")

                exe_task = ec._c_ctx.run(asyncio.ensure_future, ec._c, loop=self._loop)
                _pass_through_activity_task_info(exe_task)

                val = await exe_task
                ec._set_result(val)
            except Exception as e:
                logger.exception("error executing step of durable function")
                ec._set_exception(e)

        g = task.generator
        assert not isinstance(g, bytes)
        nv: EffectCall | Any = task.next_value
        while True:
            try:
                # TODO: if throw, execute to the next yield point before checkpointing?
                task.at_checkpoint.set()
                await self._can_execute.wait()

                task.at_checkpoint.clear()
                if nv is None:
                    nv = g.send(None)
                elif isinstance(nv, EffectCall) and nv._done:
                    nv = g.throw(nv._c_exc) if nv._c_exc else g.send(nv._c_result)
                else:
                    # the next value is not yet resolved, e.g. restored from awaiting an AgentTask
                    pass

                if isinstance(nv, EffectCall):
                    task.next_value = nv

                    if isinstance(nv._c, AgentTask):
                        # allow pickling the AgentTask before it's resolved
                        task.at_checkpoint.set()

                    await _execute_step(nv)
                    assert nv._done
                else:
                    exc = DurableInvalidStateError(
                        f"Unsupported awaitable yielded: {nv!r}.\n"
                        "Durable functions may only await supported operations.\n"
                        "You awaited something that can’t be checkpointed/replayed.\n"
                        ">> Wrap it in EffectCall(...)."
                    )
                    nv = EffectCall._from_exception(exc)
                    task.next_value = nv

            except StopIteration as e:
                return e.value
