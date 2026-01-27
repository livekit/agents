from __future__ import annotations

import asyncio
import contextvars
import pickle
import reprlib
from collections.abc import Awaitable, Generator
from dataclasses import dataclass, field
from types import coroutine
from typing import Any, Callable, Literal

from livekit import durable
from livekit.agents import utils
from livekit.durable.function import DurableCoroutine, DurableGenerator


@coroutine
@durable.durable
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


class EffectCall:
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

    def __init__(self, aw: Awaitable) -> None:
        self._c: Awaitable = aw  # runtime-only, never pickled
        self._c_result: Any = None
        self._c_exc: EffectException | None = None
        self._c_ctx: contextvars.Context | None = None
        self._done: bool = False

    def __await__(self) -> Generator[Any, Any, Any]:
        self._c_ctx = contextvars.copy_context()
        return yields(self)

    def _set_result(self, value: Any) -> None:
        self._c_result = value
        self._c_exc = None
        self._done = True

    def _set_exception(self, exc: BaseException) -> None:
        self._c_exc = EffectException.from_exception(exc)
        self._done = True

    def __getstate__(self) -> dict[str, Any]:
        if not self._done:
            raise TypeError("Cannot pickle an unresolved EffectCall")

        return {
            "done": True,
            "c_result": self._c_result,
            "c_exc": self._c_exc,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._done = state["done"]
        self._c_result = state["c_result"]
        self._c_exc = state["c_exc"]

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
    generator: DurableGenerator
    next_value: Any = None
    next_action: Literal["send", "throw"] = "send"
    at_checkpoint: asyncio.Event = field(default_factory=asyncio.Event)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        # exclude the at_checkpoint event from the pickled state
        return (DurableTask, (self.generator, self.next_value, self.next_action))


class DurableScheduler:
    def __init__(self) -> None:
        self._tasks: dict[asyncio.Task[None], DurableTask] = {}

        self._can_execute = asyncio.Event()
        self._can_execute.set()
        self._ckpt_lock = asyncio.Lock()

    def execute(self, fnc: Callable[[], DurableCoroutine] | DurableTask) -> asyncio.Task[None]:
        if isinstance(fnc, DurableTask):
            task = fnc
        else:
            g = fnc().__await__()
            task = DurableTask(g)

        exe_task = asyncio.create_task(self._execute(task))
        self._tasks[exe_task] = task
        exe_task.add_done_callback(lambda _: self._tasks.pop(exe_task))

        return exe_task

    async def checkpoint(self) -> bytes:
        async with self._ckpt_lock:
            self._can_execute.clear()

            try:
                # wait for all tasks to be ready to dump
                tasks = list(self._tasks.values())
                for task in tasks:
                    await task.at_checkpoint.wait()

                return pickle.dumps(tasks)
            finally:
                self._can_execute.set()

    def restore(self, states: bytes) -> list[asyncio.Task[None]]:
        tasks = pickle.loads(states)
        exe_tasks = []
        for task in tasks:
            exe_tasks.append(self.execute(task))
        return exe_tasks

    async def aclose(self) -> None:
        await utils.aio.cancel_and_wait(*self._tasks.keys())
        self._tasks.clear()

    async def _execute(self, task: DurableTask) -> None:
        __tracebackhide__ = True

        async def _execute_step(ec: EffectCall) -> None:
            try:
                if not ec._c:
                    raise RuntimeError("invalid EffectCall state")

                exe_task = ec._c_ctx.run(asyncio.create_task, ec._c)
                val = await exe_task
                ec._set_result(val)
            except Exception as e:
                ec._set_exception(e)

        g = task.generator

        while True:
            try:
                # TODO: only dump when the action is send?
                task.at_checkpoint.set()
                await self._can_execute.wait()

                task.at_checkpoint.clear()
                nv = (
                    g.send(task.next_value)
                    if task.next_action == "send"
                    else g.throw(task.next_value)
                )

                if isinstance(nv, EffectCall):
                    await _execute_step(nv)
                    assert nv._done

                    print("nv", nv)
                    if nv._c_exc:
                        task.next_value = nv._c_exc
                        task.next_action = "throw"
                    else:
                        task.next_value = nv._c_result
                        task.next_action = "send"
                else:
                    exc = DurableInvalidStateError(
                        f"Unsupported awaitable yielded: {nv!r}.\n"
                        "Durable functions may only await supported operations.\n"
                        "You awaited something that can’t be checkpointed/replayed.\n"
                        ">> Wrap it in EffectCall(...)."
                    )
                    task.next_value = exc
                    task.next_action = "throw"

            except StopIteration:
                break
