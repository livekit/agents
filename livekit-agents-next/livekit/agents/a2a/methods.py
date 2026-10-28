"""Host setup for the native A2A door. HTTP and application callbacks stay in Python."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable
from typing import TYPE_CHECKING

from .. import _core
from .._generated.a2a_card import EndpointDescriptor
from .._generated.a2a_door import DoorConfig
from ..storage.backends import Storage
from ..storage.context import TurnHost
from ..storage.resolve import Thread
from .auth import Credentials, Principal

if TYPE_CHECKING:
    from .endpoint import A2AEndpoint


class A2AService:
    def __init__(
        self,
        endpoint: A2AEndpoint,
        *,
        storage: Storage,
        agent_name: Callable[[], str],
        credentials: Credentials | None,
        lane: Callable[[], str] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._storage = storage
        self._agent_name = agent_name
        self._lane = lane
        self.credentials = credentials
        self._opening: asyncio.Task[_core._core_A2ADoor] | None = None

    @property
    def agent_name(self) -> str:
        return self._agent_name()

    @property
    def lane(self) -> str:
        return (self._lane() if self._lane is not None else None) or self.agent_name

    async def _open(self) -> _core._core_A2ADoor:
        endpoint = self._endpoint
        host = TurnHost(
            endpoint.handler,
            thread=Thread(self._storage, "", agent_name=self.lane),
        )
        return await self._storage._core._core_a2a_door(
            DoorConfig(
                lane=self.lane,
                holder=self._storage.holder,
                endpoint=EndpointDescriptor(
                    name=endpoint.name,
                    description=endpoint.description,
                    version=endpoint.version,
                    url=endpoint.url or f"/{endpoint.name}/v1",
                ),
                options=endpoint.turn_options,
            ),
            host,
        )

    async def native(self) -> _core._core_A2ADoor:
        opening = self._opening
        if opening is None:
            opening = self._opening = asyncio.create_task(self._open())
        try:
            return await asyncio.shield(opening)
        except Exception:
            if self._opening is opening:
                self._opening = None
            raise

    async def dispatch(
        self,
        body: bytes,
        principal: Principal,
        *,
        version: str | None = None,
        offset: str | None = None,
        body_size: int | None = None,
    ) -> _core._core_A2AResponse:
        return await (await self.native()).dispatch(
            body,
            principal,
            version=version,
            offset=offset,
            body_size=body_size,
            host_context=(asyncio.get_running_loop(), contextvars.copy_context()),
        )

    async def aclose(self) -> None:
        if self._opening is not None:
            await (await self._opening).aclose()
