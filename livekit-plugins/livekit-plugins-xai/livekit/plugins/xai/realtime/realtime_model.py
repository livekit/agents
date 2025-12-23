import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import SessionUpdateEvent
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from livekit.agents import ProviderTool, llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai

from .types import GrokVoices

XAI_BASE_URL = "wss://api.x.ai/v1/realtime"

XAI_DEFAULT_TURN_DETECTION = ServerVad(
    type="server_vad",
    threshold=0.5,
    prefix_padding_ms=300,
    silence_duration_ms=200,
    create_response=True,
    interrupt_response=True,
)


class XAITool(ProviderTool):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class WebSearch(XAITool):
    """Enable web search tool for real-time internet searches."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "web_search"}


@dataclass(slots=True)
class XSearch(XAITool):
    """Enable X (Twitter) search tool for searching posts."""

    allowed_x_handles: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {"type": "x_search"}
        if self.allowed_x_handles:
            result["allowed_x_handles"] = self.allowed_x_handles
        return result


@dataclass(slots=True)
class FileSearch(XAITool):
    """Enable file search tool for searching uploaded document collections."""

    vector_store_ids: list[str] = field(default_factory=list)
    max_num_results: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "type": "file_search",
            "vector_store_ids": self.vector_store_ids,
        }
        if self.max_num_results is not None:
            result["max_num_results"] = self.max_num_results

        return result


class RealtimeModel(openai.realtime.RealtimeModel):
    def __init__(
        self,
        *,
        voice: NotGivenOr[GrokVoices | str | None] = "Ara",
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = XAI_DEFAULT_TURN_DETECTION,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        api_key = api_key or os.environ.get("XAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the XAI_API_KEY environment variable"
            )

        super().__init__(
            base_url=base_url if is_given(base_url) else XAI_BASE_URL,
            model="grok-4-1-fast-non-reasoning",
            voice=voice,
            api_key=api_key,
            modalities=["audio"],
            turn_detection=turn_detection,
            http_session=http_session if is_given(http_session) else None,
            max_session_duration=max_session_duration if is_given(max_session_duration) else None,
            conn_options=conn_options,
        )

    def session(self) -> "RealtimeSession":
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess


class RealtimeSession(openai.realtime.RealtimeSession):
    """xAI Realtime Session that supports xAI built-in tools."""

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._xai_model: RealtimeModel = realtime_model

    def _create_tools_update_event(
        self, tools: list[llm.FunctionTool | llm.RawFunctionTool | ProviderTool]
    ) -> SessionUpdateEvent | dict:
        event = super()._create_tools_update_event(tools)

        # inject supported Toolset
        xai_tools: list[dict] = []
        for tool in tools:
            if isinstance(tool, XAITool):
                xai_tools.append(tool.to_dict())

        event["session"]["tools"] += xai_tools
        return event
