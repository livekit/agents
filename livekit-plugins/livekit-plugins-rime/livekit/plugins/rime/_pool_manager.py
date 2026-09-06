# Copyright 2026 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

from livekit.agents.utils import ConnectionPool

from .log import logger

T = TypeVar("T")


class PoolManager(Generic[T]):
    """Keep replaced pools alive until their last stream finishes."""

    def __init__(self, pool: ConnectionPool[T]) -> None:
        self.current = pool
        self._retired_pools: set[ConnectionPool[T]] = set()
        self._pool_stream_counts: dict[ConnectionPool[T], int] = {}
        self._pool_close_tasks: set[asyncio.Task[None]] = set()

    def track_stream(self, pool: ConnectionPool[T], task: asyncio.Task[None]) -> None:
        self._pool_stream_counts[pool] = self._pool_stream_counts.get(pool, 0) + 1

        def _release_pool(_: asyncio.Task[None]) -> None:
            stream_count = self._pool_stream_counts[pool]
            if stream_count > 1:
                self._pool_stream_counts[pool] = stream_count - 1
                return

            del self._pool_stream_counts[pool]
            self._schedule_retired_pool_close(pool)

        task.add_done_callback(_release_pool)

    def replace(self, pool: ConnectionPool[T]) -> None:
        old_pool = self.current
        self.current = pool
        self._retired_pools.add(old_pool)
        if self._pool_stream_counts.get(old_pool, 0) == 0:
            self._schedule_retired_pool_close(old_pool)

    async def aclose(self) -> None:
        await self.current.aclose()
        for pool in list(self._retired_pools):
            await pool.aclose()
        self._retired_pools.clear()
        if self._pool_close_tasks:
            await asyncio.gather(*list(self._pool_close_tasks), return_exceptions=True)

    def _schedule_retired_pool_close(self, pool: ConnectionPool[T]) -> None:
        if pool not in self._retired_pools:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._retired_pools.remove(pool)
        task = loop.create_task(pool.aclose())
        self._pool_close_tasks.add(task)
        task.add_done_callback(self._on_retired_pool_closed)

    def _on_retired_pool_closed(self, task: asyncio.Task[None]) -> None:
        self._pool_close_tasks.discard(task)
        if task.cancelled():
            return
        if error := task.exception():
            logger.warning(
                "failed to close a retired Rime WebSocket pool",
                extra={"exception_type": type(error).__name__},
            )
