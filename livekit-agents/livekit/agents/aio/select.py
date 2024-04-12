from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator
from typing import (
    Any,
    Awaitable,
    Iterable,
    Union,
)

from attr import define, field


class SelectLoop:
    @define(kw_only=True)
    class Completed:
        selected: Awaitable = field(repr=False)
        index: int
        exc: BaseException | None
        value: Any

        def result(self) -> Any:
            if self.exc is not None:
                raise self.exc
            return self.value

    @define(kw_only=True)
    class GenData:
        gen: AsyncIterator
        next_task: asyncio.Task | None = None

    @define(kw_only=True)
    class CoroData:
        coro: Awaitable
        task: asyncio.Task | None = None

    def __init__(self, aw: Iterable[Union[Awaitable, AsyncIterator]]) -> None:
        self._og = []
        self._pending_tasks = []

        self._coros = []
        self._gens = []

        for a in aw:
            if isinstance(a, AsyncIterator):
                self._gens.append(__class__.GenData(gen=a))
            elif asyncio.isfuture(a):
                self._pending_tasks.append(a)
            else:
                t = asyncio.ensure_future(a)
                self._coros.append(__class__.CoroData(coro=a, task=t))
                self._pending_tasks.append(t)

            self._og.append(a)

        self._q = deque()

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

        for g in self._gens:
            if g.next_task is None:
                g.next_task = asyncio.ensure_future(g.gen.__anext__())
                self._pending_tasks.append(g.next_task)

        done, pending = await asyncio.wait(
            self._pending_tasks, return_when=asyncio.FIRST_COMPLETED
        )

        self._pending_tasks = list(pending)
        for t in done:
            g = None
            for y in self._gens:
                if y.next_task == t:
                    g = y
                    break

            c = None
            for y in self._coros:
                if y.task == t:
                    c = y
                    break

            if g is not None:
                ogi = self._og.index(g.gen)
                try:
                    v = t.result()
                    self._q.append(
                        __class__.Completed(
                            selected=g.gen, index=ogi, exc=None, value=v
                        )
                    )
                except StopAsyncIteration:
                    self._gens.pop(self._gens.index(g))
                    self._og.pop(ogi)

                    if self._done():
                        raise StopAsyncIteration
                except Exception as e:
                    self._q.append(
                        __class__.Completed(
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
                        __class__.Completed(
                            selected=c.coro, index=ogi, exc=None, value=v
                        )
                    )
                except Exception as e:
                    self._q.append(
                        __class__.Completed(
                            selected=c.coro, index=ogi, exc=e, value=None
                        )
                    )

                self._coros.pop(self._coros.index(c))
            else:
                ogi = self._og.index(t)
                try:
                    v = t.result()
                    self._q.append(
                        __class__.Completed(selected=t, index=ogi, exc=None, value=v)
                    )
                except Exception as e:
                    self._q.append(
                        __class__.Completed(selected=t, index=ogi, exc=e, value=None)
                    )

        return self._q.popleft()

    async def aclose(self) -> None:
        for t in self._pending_tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t


def select(aw: Iterable[Union[Awaitable, AsyncIterator]]) -> SelectLoop:
    return SelectLoop(aw)
