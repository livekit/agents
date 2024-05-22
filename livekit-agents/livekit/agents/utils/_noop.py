from typing import Any


class _Awaitable:
    def __await__(self) -> Any:
        async def _await():
            pass

        return _await().__await__()


class Nop(object):
    def nop(*args, **kw) -> _Awaitable:
        return _Awaitable()

    def __getattr__(self, _):
        return self.nop
