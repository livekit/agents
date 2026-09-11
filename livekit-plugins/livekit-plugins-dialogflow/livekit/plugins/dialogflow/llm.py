# Copyright 2024 LiveKit, Inc.
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

import os
import uuid
from dataclasses import dataclass
from typing import Any

from google.api_core.exceptions import (
    GoogleAPIError,
    InvalidArgument,
    NotFound,
    PermissionDenied,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.cloud.dialogflowcx_v3 import SessionsAsyncClient, types as df_types

from livekit.agents import llm, utils
from livekit.agents._exceptions import APIConnectionError, APIStatusError
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .log import logger


@dataclass
class _LLMOptions:
    project_id: str
    location: str
    agent_id: str
    language_code: str
    environment_id: str | None
    session_ttl: int


class LLM(llm.LLM):
    """LiveKit Agents LLM plugin for Google Dialogflow CX.

    Dialogflow CX is an intent-based conversational AI engine with its own session
    management, flows, pages, and fulfillment logic. Unlike standard LLMs, it does
    not stream tokens — it returns the full response in one shot.

    Note: The ``instructions`` field on the Agent class has no effect when using
    Dialogflow. All conversational logic is configured in the Dialogflow CX console.
    """

    def __init__(
        self,
        *,
        project_id: str | None = None,
        location: str | None = None,
        agent_id: str,
        language_code: str = "en",
        environment_id: str | None = None,
        session_ttl: int = 3600,
    ) -> None:
        """Create a new Dialogflow CX LLM instance.

        Args:
            project_id: GCP project ID. Falls back to ``GOOGLE_CLOUD_PROJECT`` env var.
            location: Dialogflow CX location. Falls back to ``GOOGLE_CLOUD_LOCATION``
                env var, defaults to ``"global"``.
            agent_id: The Dialogflow CX agent ID.
            language_code: Language for detect intent requests.
            environment_id: Optional environment ID for versioned agents. If not set,
                uses the draft environment.
            session_ttl: How long to keep Dialogflow sessions alive (seconds).
        """
        super().__init__()

        resolved_project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not resolved_project_id:
            raise ValueError("project_id must be provided or set via GOOGLE_CLOUD_PROJECT env var")

        resolved_location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

        self._opts = _LLMOptions(
            project_id=resolved_project_id,
            location=resolved_location,
            agent_id=agent_id,
            language_code=language_code,
            environment_id=environment_id,
            session_ttl=session_ttl,
        )

        # Sessions map: cache_key -> dialogflow session ID
        self._sessions: dict[str, str] = {}

        # Determine the API endpoint based on location
        if resolved_location != "global":
            api_endpoint = f"{resolved_location}-dialogflow.googleapis.com"
        else:
            api_endpoint = "dialogflow.googleapis.com"

        self._client = SessionsAsyncClient(client_options={"api_endpoint": api_endpoint})

        logger.info(
            "Dialogflow CX LLM initialized (project=%s, location=%s, agent=%s)",
            resolved_project_id,
            resolved_location,
            agent_id,
        )

    @property
    def model(self) -> str:
        return f"dialogflow-cx/{self._opts.agent_id}"

    @property
    def provider(self) -> str:
        return "Google Dialogflow CX"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> DialogflowLLMStream:
        extra = dict(extra_kwargs) if isinstance(extra_kwargs, dict) else {}

        return DialogflowLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )

    async def aclose(self) -> None:
        pass


class DialogflowLLMStream(llm.LLMStream):
    """LLMStream implementation for Google Dialogflow CX.

    Since Dialogflow returns the full response in one shot (no streaming),
    the entire response is emitted as a single ChatChunk.
    """

    def __init__(
        self,
        llm_instance: LLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._llm: LLM = llm_instance
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        request_id = utils.shortuuid("dialogflow_")

        # Extract the latest user message from the chat context
        user_text = self._get_latest_user_text()
        if not user_text:
            raise APIStatusError(
                "dialogflow: no user message found in chat context",
                status_code=400,
                request_id=request_id,
                retryable=False,
            )

        # Build session path
        # TODO: In production, session IDs should be mapped to LiveKit participant/room
        # IDs for multi-turn conversation continuity. Currently a new session is created
        # per LLMStream instance unless the caller provides a session_id via extra_kwargs.
        session_id = self._extra_kwargs.get("session_id") or str(uuid.uuid4())
        session_path = self._build_session_path(session_id)

        # Build the DetectIntent request
        text_input = df_types.TextInput(text=user_text)
        query_input = df_types.QueryInput(
            text=text_input,
            language_code=self._llm._opts.language_code,
        )
        request = df_types.DetectIntentRequest(
            session=session_path,
            query_input=query_input,
        )

        try:
            response = await self._llm._client.detect_intent(request=request)
        except ResourceExhausted as e:
            raise APIStatusError(
                "dialogflow: rate limited",
                status_code=429,
                body=str(e),
                request_id=request_id,
                retryable=True,
            ) from e
        except ServiceUnavailable as e:
            raise APIStatusError(
                "dialogflow: service unavailable",
                status_code=503,
                body=str(e),
                request_id=request_id,
                retryable=True,
            ) from e
        except (InvalidArgument, NotFound, PermissionDenied) as e:
            raise APIStatusError(
                f"dialogflow: {type(e).__name__}",
                status_code=400,
                body=str(e),
                request_id=request_id,
                retryable=False,
            ) from e
        except GoogleAPIError as e:
            raise APIConnectionError(
                f"dialogflow: {e!s}",
                retryable=True,
            ) from e
        except Exception as e:
            raise APIConnectionError(
                f"dialogflow: unexpected error: {e!s}",
                retryable=False,
            ) from e

        # Extract response text from response messages
        query_result = response.query_result
        parts: list[str] = []
        custom_payloads: list[dict[str, Any]] = []

        for msg in query_result.response_messages:
            if msg.text and msg.text.text:
                parts.extend(msg.text.text)
            # Collect custom payloads (structured data like cards, suggestions, etc.)
            if msg.payload:
                custom_payloads.append(dict(msg.payload))

        response_text = " ".join(parts).strip()

        if not response_text:
            logger.warning(
                "dialogflow: empty response text (session=%s, request=%s)",
                session_id,
                request_id,
            )
            response_text = ""

        # Build extra metadata from Dialogflow-specific fields for observability
        extra: dict[str, Any] = {}
        if query_result.intent_detection_confidence:
            extra["intent_detection_confidence"] = query_result.intent_detection_confidence
        if query_result.match and query_result.match.match_type:
            extra["match_type"] = query_result.match.match_type.name
        if query_result.match and query_result.match.intent:
            extra["matched_intent"] = query_result.match.intent.display_name
        if query_result.current_page:
            extra["current_page"] = query_result.current_page.display_name
        if custom_payloads:
            extra["custom_payloads"] = custom_payloads

        # Emit the full response as a single ChatChunk (Dialogflow does not stream)
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=request_id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=response_text,
                    extra=extra if extra else None,
                ),
            )
        )

        # Emit usage metrics (Dialogflow does not provide token counts)
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=request_id,
                usage=llm.CompletionUsage(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                ),
            )
        )

    def _get_latest_user_text(self) -> str | None:
        """Extract the text content of the most recent user message."""
        for msg in reversed(self._chat_ctx.messages()):
            if msg.role == "user":
                text = msg.text_content
                if text:
                    return text
        return None

    def _build_session_path(self, session_id: str) -> str:
        """Build the Dialogflow CX session resource path."""
        opts = self._llm._opts
        base = f"projects/{opts.project_id}/locations/{opts.location}/agents/{opts.agent_id}"
        if opts.environment_id:
            base += f"/environments/{opts.environment_id}"
        return f"{base}/sessions/{session_id}"
