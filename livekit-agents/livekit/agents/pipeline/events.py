from __future__ import annotations

from dataclasses import dataclass
from typing import List, TypeVar, TYPE_CHECKING, Generic, get_args
from ..llm import FunctionCall


if TYPE_CHECKING:
    from .pipeline_agent import PipelineAgent
    from .speech_handle import SpeechHandle


Userdata_T = TypeVar("Userdata_T")


class CallContext(Generic[Userdata_T]):
    # private ctor
    def __init__(
        self,
        *,
        agent: PipelineAgent,
        speech_handle: SpeechHandle,
        function_call: FunctionCall,
    ) -> None:
        self._agent = agent
        self._speech_handle = speech_handle
        self._function_call = function_call

    @property
    def agent(self) -> PipelineAgent[Userdata_T]:
        return self._agent

    @property
    def speech_handle(self) -> SpeechHandle:
        return self._speech_handle

    @property
    def function_call(self) -> FunctionCall:
        return self._function_call

    @property
    def userdata(self) -> Userdata_T:
        return self.agent.userdata


@dataclass
class UserStartedSpeakingEvent:
    pass


@dataclass
class UserStoppedSpeakingEvent:
    pass
