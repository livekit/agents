from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Awaitable, Iterable, Union


class SelectLoop:
    @dataclass
    class Completed:
        selected: AsyncIterator[Any] | Awaitable[Any] = field(repr=False)
        index: int
        exc: BaseException | None
        value: Any

        def result(self) -> Any:
            if self.exc is not None:
                raise self.exc
            return self.value

    @dataclass
    class GenData:
        gen: AsyncIterator[Any]
        next_task: asyncio.Task[Any] | None = None

    @dataclass
    class CoroData:
        coro: Awaitable[Any]
        task: asyncio.Task[Any] | None = None

    def __init__(self, aw: Iterable[Union[Awaitable[Any], AsyncIterator[Any]]]) -> None:
        self._og: list[Any] = []
        self._pending_tasks: list[asyncio.Task[Any] | asyncio.Future[Any]] = []

        self._coros: list[SelectLoop.CoroData] = []
        self._gens: list[SelectLoop.GenData] = []

        for a in aw:
            if isinstance(a, AsyncIterator):
                self._gens.append(SelectLoop.GenData(gen=a))
            elif asyncio.isfuture(a):
                self._pending_tasks.append(a)
            else:
                t = asyncio.ensure_future(a)
                self._coros.append(SelectLoop.CoroData(coro=a, task=t))
                self._pending_tasks.append(t)

            self._og.append(a)

        self._q = deque[Any]()

    def __aiter__(self) -> "SelectLoop":
        return self

    def _done(self) -> bool:
        return not self._pending_tasks and not self._gens and not self._q

    def __call__(self) -> Awaitable[SelectLoop.Completed]:
        return self.__anext__()

    async def __anext__(self) -> Completed:
        if self._q:
            return self._q.popleft()

        if self._done():
            raise StopAsyncIteration

        for gen_data in self._gens:
            if gen_data.next_task is None:
                gen_data.next_task = asyncio.ensure_future(gen_data.gen.__anext__())
                self._pending_tasks.append(gen_data.next_task)

        done, pending = await asyncio.wait(
            self._pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )

        self._pending_tasks = list(pending)
        for t in done:
            g = None
            for g_data in self._gens:
                if g_data.next_task == t:
                    g = g_data
                    break

            c = None
            for c_data in self._coros:
                if c_data.task == t:
                    c = c_data
                    break

            if g is not None:
                ogi = self._og.index(g.gen)
                try:
                    v = t.result()
                    self._q.append(
                        SelectLoop.Completed(
                            selected=g.gen, index=ogi, exc=None, value=v
                        )
                    )
                except Exception as e:
                    if isinstance(e, StopAsyncIteration):
                        self._gens.pop(self._gens.index(g))
                        self._og.pop(ogi)

                    self._q.append(
                        SelectLoop.Completed(
                            selected=g.gen, index=ogi, exc=e, value=None
                        )
                    )
                finally:
                    g.next_task = None
            elif c is not None:
                ogi = self._og.index(c.coro)
                try:
                    v = t.result()
                    self._q.append(
                        SelectLoop.Completed(
                            selected=c.coro, index=ogi, exc=None, value=v
                        )
                    )
                except Exception as e:
                    self._q.append(
                        SelectLoop.Completed(
                            selected=c.coro, index=ogi, exc=e, value=None
                        )
                    )

                self._coros.pop(self._coros.index(c))
            else:
                ogi = self._og.index(t)
                try:
                    v = t.result()
                    self._q.append(
                        SelectLoop.Completed(selected=t, index=ogi, exc=None, value=v)
                    )
                except Exception as e:
                    self._q.append(
                        SelectLoop.Completed(selected=t, index=ogi, exc=e, value=None)
                    )

        return self._q.popleft()

    async def aclose(self) -> None:
        for t in self._pending_tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t


def select(aw: Iterable[Union[Awaitable[Any], AsyncIterator[Any]]]) -> SelectLoop:
    return SelectLoop(aw)
