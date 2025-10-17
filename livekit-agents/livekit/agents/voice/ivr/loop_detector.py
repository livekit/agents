from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from livekit.rtc import EventEmitter

if TYPE_CHECKING:
    from ..agent_session import AgentSession


EventTypes = Literal["loop_detected"]


class BaseLoopDetector(ABC, EventEmitter[EventTypes]):
    def __init__(self, session: AgentSession) -> None:
        self._session = session

    @abstractmethod
    async def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class TfidfLoopDetector(BaseLoopDetector):
    def __init__(self, session: AgentSession) -> None:
        pass
