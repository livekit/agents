import os
import time
from typing import Any

import aiohttp
from openai.types.beta.realtime.session import TurnDetection
from openai.types.realtime import RealtimeConversationItemFunctionCall
from openai.types.realtime.realtime_audio_input_turn_detection import ServerVad

from livekit.agents import llm
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins import openai

from ..log import logger
from ..tools import XAITool
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


class RealtimeModel(openai.realtime.RealtimeModel):
    def __init__(
        self,
        *,
        voice: NotGivenOr[GrokVoices | str | None] = "Ara",
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
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

        resolved_voice = voice if is_given(voice) else "Ara"
        super().__init__(
            base_url=base_url if is_given(base_url) else XAI_BASE_URL,
            model="grok-4-1-fast-non-reasoning",
            voice=resolved_voice,  # type: ignore[arg-type]
            api_key=api_key,
            modalities=["audio"],
            turn_detection=turn_detection
            if is_given(turn_detection)
            else XAI_DEFAULT_TURN_DETECTION,
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
        self._session_connected_at: float = 0.0

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        self._session_connected_at = time.time()
        await super()._run_ws(ws_conn)

    async def aclose(self) -> None:
        # emit session duration metrics before closing (for xAI's per-minute billing)
        if self._session_connected_at > 0:
            session_duration = time.time() - self._session_connected_at
            metrics = RealtimeModelMetrics(
                timestamp=time.time(),
                request_id="session_close",
                session_duration=session_duration,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=Metadata(
                    model_name=self._xai_model.model,
                    model_provider=self._xai_model.provider,
                ),
            )
            self.emit("metrics_collected", metrics)
        await super().aclose()

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        event = super()._create_tools_update_event(tools)

        # inject supported Toolset
        xai_tools: list[dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, XAITool):
                xai_tools.append(tool.to_dict())

        event["session"]["tools"] += xai_tools
        return event

    def _handle_function_call(self, item: RealtimeConversationItemFunctionCall) -> None:
        if not self._tools.get_function_tool(item.name):
            logger.warning(f"unknown function tool: {item.name}, ignoring")
            return

        super()._handle_function_call(item)
