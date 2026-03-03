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

from aiobotocore.session import AioSession  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

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
    bot_id: str
    bot_alias_id: str
    locale_id: str
    region: str
    session_ttl: int


class LLM(llm.LLM):
    """LiveKit Agents LLM plugin for Amazon Lex V2.

    Amazon Lex V2 is an intent-based conversational AI engine with its own session
    management, intents, slots, and fulfillment logic. Unlike standard LLMs, it does
    not stream tokens — it returns the full response in one shot.

    Note: The ``instructions`` field on the Agent class has no effect when using
    Lex. All conversational logic is configured in the AWS Lex V2 console.
    """

    def __init__(
        self,
        *,
        bot_id: str,
        bot_alias_id: str,
        locale_id: str = "en_US",
        region: str | None = None,
        session_ttl: int = 3600,
    ) -> None:
        """Create a new Amazon Lex V2 LLM instance.

        Args:
            bot_id: The Lex bot ID.
            bot_alias_id: The Lex bot alias ID.
            locale_id: Locale for the bot (e.g. ``"en_US"``).
            region: AWS region. Falls back to ``AWS_REGION`` then
                ``AWS_DEFAULT_REGION`` env vars.
            session_ttl: How long to keep Lex sessions alive (seconds).
        """
        super().__init__()

        resolved_region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        )
        if not resolved_region:
            raise ValueError(
                "region must be provided or set via AWS_REGION / AWS_DEFAULT_REGION env var"
            )

        self._opts = _LLMOptions(
            bot_id=bot_id,
            bot_alias_id=bot_alias_id,
            locale_id=locale_id,
            region=resolved_region,
            session_ttl=session_ttl,
        )

        self._session = AioSession()

        logger.info(
            "Amazon Lex V2 LLM initialized (bot=%s, alias=%s, region=%s)",
            bot_id,
            bot_alias_id,
            resolved_region,
        )

    @property
    def model(self) -> str:
        return f"lex-v2/{self._opts.bot_id}"

    @property
    def provider(self) -> str:
        return "Amazon Lex V2"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LexLLMStream:
        extra = dict(extra_kwargs) if isinstance(extra_kwargs, dict) else {}

        return LexLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )

    async def aclose(self) -> None:
        pass


class LexLLMStream(llm.LLMStream):
    """LLMStream implementation for Amazon Lex V2.

    Since Lex returns the full response in one shot (no streaming),
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
        request_id = utils.shortuuid("lex_")

        user_text = self._get_latest_user_text()
        if not user_text:
            raise APIStatusError(
                "lex: no user message found in chat context",
                status_code=400,
                request_id=request_id,
                retryable=False,
            )

        session_id = self._extra_kwargs.get("session_id") or str(uuid.uuid4())
        opts = self._llm._opts

        try:
            async with self._llm._session.create_client(
                "lexv2-runtime",
                region_name=opts.region,
            ) as client:
                response = await client.recognize_text(
                    botId=opts.bot_id,
                    botAliasId=opts.bot_alias_id,
                    localeId=opts.locale_id,
                    sessionId=session_id,
                    text=user_text,
                )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            self._handle_client_error(e, error_code, request_id)
        except Exception as e:
            raise APIConnectionError(
                f"lex: unexpected error: {e!s}",
                retryable=False,
            ) from e

        # Extract response text from Lex messages
        parts: list[str] = []
        messages = response.get("messages", [])
        for msg in messages:
            content = msg.get("content")
            if content:
                parts.append(content)

        response_text = " ".join(parts).strip()

        if not response_text:
            logger.warning(
                "lex: empty response text (session=%s, request=%s)",
                session_id,
                request_id,
            )
            response_text = ""

        # Build extra metadata from Lex-specific fields for observability
        extra: dict[str, Any] = {}

        session_state = response.get("sessionState", {})
        if session_state:
            extra["session_state"] = session_state

        intent = session_state.get("intent", {})
        if intent.get("name"):
            extra["intent_name"] = intent["name"]

        slots = intent.get("slots")
        if slots:
            extra["slots"] = slots

        interpretations = response.get("interpretations", [])
        if interpretations:
            top = interpretations[0]
            nlu_confidence = top.get("nluConfidence", {})
            score = nlu_confidence.get("score")
            if score is not None:
                extra["interpretation_confidence"] = score

        # Emit the full response as a single ChatChunk (Lex does not stream)
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

        # Emit usage metrics (Lex does not provide token counts)
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

    @staticmethod
    def _handle_client_error(e: ClientError, error_code: str, request_id: str) -> None:
        """Map AWS Lex errors to appropriate API exceptions."""
        if error_code == "ThrottlingException":
            raise APIStatusError(
                "lex: rate limited",
                status_code=429,
                body=str(e),
                request_id=request_id,
                retryable=True,
            ) from e
        elif error_code == "ServiceUnavailableException":
            raise APIStatusError(
                "lex: service unavailable",
                status_code=503,
                body=str(e),
                request_id=request_id,
                retryable=True,
            ) from e
        elif error_code in (
            "ValidationException",
            "ResourceNotFoundException",
            "AccessDeniedException",
        ):
            raise APIStatusError(
                f"lex: {error_code}",
                status_code=400,
                body=str(e),
                request_id=request_id,
                retryable=False,
            ) from e
        else:
            raise APIConnectionError(
                f"lex: {e!s}",
                retryable=True,
            ) from e
