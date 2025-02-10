from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_agent import PipelineAgent


class AgentContext:
    def __init__(self, agent: PipelineAgent) -> None:
        self._agent = agent

    @property
    def agent(self) -> PipelineAgent:
        return self._agent


@dataclass
class UserStartedSpeakingEvent:
    pass


@dataclass
class UserStoppedSpeakingEvent:
    pass
