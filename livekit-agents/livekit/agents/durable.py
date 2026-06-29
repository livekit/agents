import pickle
import contextvars
import reprlib
import asyncio
import aiohttp
from typing import Awaitable, Any, Optional, Callable
from types import coroutine
from livekit import durable
from dataclasses import dataclass


@coroutine
@durable.durable
def yields(n):
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
    def from_exception(cls, exc: BaseException) -> "EffectException":
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
        self._c_exc: Optional[EffectException] = None
        self._c_ctx: Optional[contextvars.Context] = None
        self._done: bool = False

    def __await__(self):
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



# async def _execute_aw(ctx: contextvars.Context, aw: Awaitable):
#     it = aw.__await__()
#     send_val = None
#     while True:
#         try:
#             yielded = ctx.run(it.send, send_val)
#         except StopIteration as e:
#             return e.value
#         except BaseException as e:
#             try:
#                 yielded = ctx.run(it.throw, e)
#             except StopIteration as e2:
#                 return e2.value

#         send_val = await yielded

class DurableScheduler:
    def __init__(self, external_loop: asyncio.AbstractEventLoop) -> None:
        pass

    async def execute(self, fnc: Callable) -> None:
        g = fnc().__await__()

        value = None
        while True:
            try:
                nv = g.send(value)
                print("nv", nv)

                if isinstance(nv, EffectCall):
                    if not nv._c:
                        g.throw(RuntimeError("invalid CallEffect state"))


                    # TODO(theomonnom): Copy contextvars
                    async def _execute_step(ec: EffectCall) -> None:
                        try:
                            task = ec._c_ctx.run(asyncio.create_task, ec._c)
                            val = await task
                            ec._set_result(val)
                        except Exception as e:
                            ec._set_exception(e)

                    await _execute_step(nv)
                    assert nv._done

                    if nv._c_exc:
                        g.throw(nv._c_exc)

                    value = nv._c_result


            except StopIteration:
                break


async def my_network_call() -> None:
    return 6

@durable.durable
async def my_function_tool() -> None:
    result = await EffectCall(my_network_call())
    print("a", result)

    e = await EffectCall(asyncio.sleep(5))
    print("b", e)
    #await MyAgentTask()



async def amain() -> None:
    loop = asyncio.get_event_loop()

    scheduler = DurableScheduler(external_loop=loop)
    await scheduler.execute(my_function_tool)

asyncio.run(amain())


# g = my_function_tool().__await__()

# pickle.dumps(g)

# my_effect = next(g)
# my_effect.set_exception(RuntimeError("test"))
# print(my_effect)
# assert isinstance(my_effect, EffectCall)



# pickle.dumps(g)





# import pickle
# from livekit import durable

# @durable.durable
# def my_generator():
#     for i in range(3):
#         yield i

# g = my_generator()
# print(next(g))  # 0

# b = pickle.dumps(g)
# g2 = pickle.loads(b)
# print(next(g2))  # 1
# print(next(g2))  # 2

# print(next(g))  # 1
# print(next(g))  # 2
