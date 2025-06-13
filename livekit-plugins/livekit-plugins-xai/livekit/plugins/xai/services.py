from __future__ import annotations

import os

import aiohttp
import openai

from livekit.agents.llm import ToolChoice
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.plugins.openai import LLM as OpenAILLM
from livekit.plugins.openai.realtime import realtime_model as oai_realtime

from .log import logger
from .models import LLMModels


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str | LLMModels = "grok-3",
        api_key: str | None = None,
        base_url: str = "https://api.x.ai/v1",
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ):
        """
        Create a new instance of xAI LLM.

        ``api_key`` must be set to your xAI API key, either using the argument or by setting
        the ``XAI_API_KEY`` environmental variable.
        """

        api_key = api_key or os.environ.get("XAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the XAI_API_KEY environment variable"
            )

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            _strict_tool_call=False,
        )


DEFAULT_XAI_REALTIME_BASE_URL = "https://us-east-4.api.x.ai/voice-api/v1/chat/voice"

# TODO: set to actual limit
DEFAULT_MAX_SESSION_DURATION = None  # no limit


class RealtimeModel(oai_realtime.RealtimeModel):
    def __init__(
        self,
        *,
        model: str = "",
        voice: str = "",  # TODO: set model and voice
        api_key: str | None = None,
        base_url: str = DEFAULT_XAI_REALTIME_BASE_URL,
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
            model=model,
            voice=voice,
            base_url=base_url,
            api_key=api_key,
            max_session_duration=max_session_duration or DEFAULT_MAX_SESSION_DURATION,
            conn_options=conn_options,
        )

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess


class RealtimeSession(oai_realtime.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model, label="xAI Realtime API")
        self._realtime_model: RealtimeModel = realtime_model

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._realtime_model._opts.api_key}",
        }

        query_params = {
            "client": "oai",
            "enable_noise_suppression": "true",
            "stream_asr": "true",
            "use_time_based_playback_tracking": "true",
        }
        url = oai_realtime.process_base_url(
            self._realtime_model._opts.base_url,
            self._realtime_model._opts.model,
            is_azure=False,
            api_version=None,
            azure_deployment=None,
            extra_query_params=query_params,
        )

        if oai_realtime.lk_oai_debug:
            logger.debug(f"connecting to Realtime API: {url}")

        return await self._realtime_model._ensure_http_session().ws_connect(
            url=url,
            headers=headers,
            timeout=aiohttp.ClientWSTimeout(
                ws_close=self._realtime_model._opts.conn_options.timeout
            ),
        )
