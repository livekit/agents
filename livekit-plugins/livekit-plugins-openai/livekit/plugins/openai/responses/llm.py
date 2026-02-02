from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union, cast

import httpx
from typing_extensions import Literal

import openai
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import (
    Tool,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types import Reasoning
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseInputParam,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ToolParam,
    response_create_params,
)
from openai.types.responses.response_stream_event import ResponseStreamEvent
from openai.types.shared_params import ResponsesModel

from ..models import _supports_reasoning_effort


@dataclass
class _LLMOptions:
    model: str | ResponsesModel
    user: NotGivenOr[str]
    temperature: NotGivenOr[float]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]]
    store: NotGivenOr[bool]
    reasoning: NotGivenOr[Reasoning]
    metadata: NotGivenOr[dict[str, str]]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str | ResponsesModel = "gpt-4o-mini",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        reasoning: NotGivenOr[Reasoning] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice | Literal["auto", "required", "none"]] = NOT_GIVEN,
        store: NotGivenOr[bool] = NOT_GIVEN,
        metadata: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of OpenAI Responses LLM.

        ``api_key`` must be set to your OpenAI API key, either using the argument or by setting the
        ``OPENAI_API_KEY`` environmental variable.
        """
        super().__init__()

        if not is_given(reasoning) and _supports_reasoning_effort(model):
            if model in ["gpt-5.1", "gpt-5.2"]:
                reasoning = Reasoning(effort="none")
            else:
                reasoning = Reasoning(effort="minimal")

        self._opts = _LLMOptions(
            model=model,
            user=user,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            reasoning=reasoning,
        )
        self._client = client or openai.AsyncClient(
            api_key=api_key if is_given(api_key) else None,
            base_url=base_url if is_given(base_url) else None,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=timeout
                if timeout
                else httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    @property
    def model(self) -> str:
        return self._opts.model

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra = {}

        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        if is_given(self._opts.metadata):
            extra["metadata"] = self._opts.metadata

        if is_given(self._opts.user):
            extra["user"] = self._opts.user

        if is_given(self._opts.reasoning):
            extra["reasoning"] = self._opts.reasoning

        parallel_tool_calls = (
            parallel_tool_calls if is_given(parallel_tool_calls) else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice  # type: ignore
        if is_given(tool_choice):
            oai_tool_choice: response_create_params.ToolChoice
            if isinstance(tool_choice, dict):
                oai_tool_choice = {
                    "type": "function",
                    "name": tool_choice["function"]["name"],
                }
                extra["tool_choice"] = oai_tool_choice
            elif tool_choice in ("auto", "required", "none"):
                oai_tool_choice = tool_choice  # type: ignore
                extra["tool_choice"] = oai_tool_choice

        return LLMStream(
            self,
            model=self._opts.model,
            strict_tool_schema=True,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        model: str | ResponsesModel,
        strict_tool_schema: bool,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._strict_tool_schema = strict_tool_schema
        self._client = client
        self._llm = llm
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        self._oai_stream: openai.AsyncStream[ResponseStreamEvent] | None = None
        retryable = True
        try:
            chat_ctx, _ = self._chat_ctx.to_provider_format(format="openai.responses")

            self._tool_ctx = llm.ToolContext(self.tools)
            tool_schemas = cast(
                list[ToolParam],
                self._tool_ctx.parse_function_tools(
                    "openai.responses", strict=self._strict_tool_schema
                ),
            )

            self._oai_stream = stream = await self._client.responses.create(
                model=self._model,
                tools=tool_schemas,
                input=cast(Union[str, ResponseInputParam, openai.NotGiven], chat_ctx),
                stream=True,
                timeout=httpx.Timeout(self._conn_options.timeout),
                **self._extra_kwargs,
            )

            async with stream:
                async for event in stream:
                    retryable = False
                    chunk = None

                    if isinstance(event, ResponseErrorEvent):
                        self._handle_error(event)

                    if isinstance(event, ResponseCreatedEvent):
                        self._handle_response_created(event)

                    if isinstance(event, ResponseOutputItemDoneEvent):
                        chunk = self._handle_output_items_done(event)

                    if isinstance(event, ResponseTextDeltaEvent):
                        chunk = self._handle_response_output_text_delta(event)

                    if isinstance(event, ResponseCompletedEvent):
                        chunk = self._handle_response_completed(event)

                    if chunk is not None:
                        self._event_ch.send_nowait(chunk)

        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable)  # noqa: B904
        except openai.APIStatusError as e:
            raise APIStatusError(  # noqa: B904
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            )
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _handle_error(self, event: ResponseErrorEvent) -> None:
        raise APIStatusError(event.message, status_code=-1, retryable=False)

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        self._response_id = event.response.id

    def _handle_response_completed(self, event: ResponseCompletedEvent) -> llm.ChatChunk | None:
        chunk = None
        if usage := event.response.usage:
            chunk = llm.ChatChunk(
                id=self._response_id,
                usage=llm.CompletionUsage(
                    completion_tokens=usage.output_tokens,
                    prompt_tokens=usage.input_tokens,
                    prompt_cached_tokens=usage.input_tokens_details.cached_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )
        return chunk

    def _handle_output_items_done(self, event: ResponseOutputItemDoneEvent) -> llm.ChatChunk | None:
        chunk = None
        if event.item.type == "function_call":
            chunk = llm.ChatChunk(
                id=self._response_id,
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        llm.FunctionToolCall(
                            arguments=event.item.arguments,
                            name=event.item.name,
                            call_id=event.item.id,
                        )
                    ],
                ),
            )
        return chunk

    def _handle_response_output_text_delta(
        self, event: ResponseTextDeltaEvent
    ) -> llm.ChatChunk | None:
        return llm.ChatChunk(
            id=self._response_id,
            delta=llm.ChoiceDelta(content=event.delta, role="assistant"),
        )
