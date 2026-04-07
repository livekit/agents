# mypy: ignore-errors
# ruff: noqa
import asyncio
import io
import pickle
from types import coroutine

from livekit import durable


@coroutine
@durable.durable
def yields(n):
    return (yield n)


class EffectCall:
    def __init__(self, external_coro) -> None:
        self._c = external_coro

    def __await__(self):
        return yields(self)

    def __reduce_ex__(self, protocol):
        return (type(self), (None,))

    def __repr__(self) -> None:
        return "EffectCall"


async def my_network_call() -> None:
    # some request here
    pass


@durable.durable
async def my_function_tool() -> None:
    result = await EffectCall(my_network_call)

    await EffectCall(asyncio.sleep(5))
    await MyAgentTask()


g = my_function_tool().__await__()

pickle.dumps(g)

my_effect = next(g)
print(my_effect)
assert isinstance(my_effect, EffectCall)

pickle.dumps(g)


import pickle

from livekit import durable


class _RestrictedUnpickler(pickle.Unpickler):
    _ALLOWED_MODULE_PREFIXES = ("builtins", "copyreg", "livekit.", "livekit", "__main__")

    def find_class(self, module, name):
        if not any(module == p or module.startswith(p + ".") for p in self._ALLOWED_MODULE_PREFIXES):
            raise pickle.UnpicklingError(f"Disallowed class: {module}.{name}")
        return super().find_class(module, name)


def _restricted_loads(data):
    return _RestrictedUnpickler(io.BytesIO(data)).load()


@durable.durable
def my_generator():
    for i in range(3):
        yield i


g = my_generator()
print(next(g))  # 0

b = pickle.dumps(g)
g2 = _restricted_loads(b)
print(next(g2))  # 1
print(next(g2))  # 2

print(next(g))  # 1
print(next(g))  # 2
