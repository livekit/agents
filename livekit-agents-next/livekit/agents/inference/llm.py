from __future__ import annotations

from typing import Any, Literal, cast, get_args

import httpx2 as httpx
import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort

from .. import _core, llm
from .._exceptions import APIConnectionError, APIStatusError, APITimeoutError
from .._generated.inference_llm import (
    ChatCompletionOptions,
    DeepSeekModels,
    GoogleModels,
    InferenceClass,
    KimiModels,
    LLMModels,
    OpenAIModels,
    XAIModels,
    ZAIModels,
)
from ..llm import ToolChoice, utils as llm_utils
from ..llm.chat_context import ChatContext
from ..llm.tool_context import Tool
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from ._utils import (
    HEADER_INFERENCE_PROVIDER,
    Credentials,
    extract_quota_usage,
    get_inference_headers,
)

__all__ = [
    "LLM",
    "LLMStream",
    "LLMModels",
    "ChatCompletionOptions",
    "OpenAIModels",
    "GoogleModels",
    "KimiModels",
    "DeepSeekModels",
    "ZAIModels",
    "XAIModels",
    "drop_unsupported_params",
    "min_reasoning_effort",
    "llm_from_model_string",
    "RealtimeModels",
    "OpenAIRealtimeModels",
    "XAIRealtimeModels",
    "is_realtime_model",
]

OpenAIRealtimeModels = Literal[
    "openai/gpt-realtime",
    "openai/gpt-realtime-mini",
    "openai/gpt-realtime-1.5",
    "openai/gpt-realtime-2",
    "openai/gpt-realtime-2.1",
    "openai/gpt-realtime-2.1-mini",
]

XAIRealtimeModels = Literal[
    "xai/grok-voice",
    "xai/grok-voice-latest",
    "xai/grok-voice-think-fast-2.0",
]

RealtimeModels = OpenAIRealtimeModels | XAIRealtimeModels

_REALTIME_MODEL_IDS: frozenset[str] = frozenset(
    model for literal in get_args(RealtimeModels) for model in get_args(literal)
)


def is_realtime_model(model: str) -> bool:
    """Whether a LiveKit Inference model string names a realtime (speech-to-speech) model.

    Only the models listed in ``RealtimeModels`` count: a newly released realtime
    model has to be added there before ``llm="..."`` resolves it to a
    ``RealtimeModel``.
    """
    return model in _REALTIME_MODEL_IDS


# Reasoning models don't support sampling parameters.
# See: https://platform.openai.com/docs/guides/reasoning
_REASONING_UNSUPPORTED_PARAMS: set[str] = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "n",
}

# xAI reasoning models only restrict presence_penalty, frequency_penalty, stop.
# They still support temperature and top_p.
_XAI_REASONING_UNSUPPORTED_PARAMS: set[str] = {
    "presence_penalty",
    "frequency_penalty",
    "stop",
}

# Model prefix -> set of param names that should be dropped
_UNSUPPORTED_PARAMS: dict[str, set[str]] = {
    "o1": _REASONING_UNSUPPORTED_PARAMS,
    "o3": _REASONING_UNSUPPORTED_PARAMS,
    "o4": _REASONING_UNSUPPORTED_PARAMS,
    "gpt-5": _REASONING_UNSUPPORTED_PARAMS,
    "grok-4-1-fast-reasoning": _XAI_REASONING_UNSUPPORTED_PARAMS,
    "grok-4.20-0309-reasoning": _XAI_REASONING_UNSUPPORTED_PARAMS,
    "grok-4.20-multi-agent": _XAI_REASONING_UNSUPPORTED_PARAMS,
}

# temperature/top_p are accepted by gpt-5.1+ only at reasoning_effort
# "none"; see the note in drop_unsupported_params.
_SAMPLING_PARAMS: set[str] = {"temperature", "top_p"}

# models that don't support reasoning_effort when function tools are present
_REASONING_EFFORT_TOOL_INCOMPATIBLE_PREFIXES: set[str] = {"gpt-5.2", "gpt-5.4"}


def drop_unsupported_params(
    model: str, params: dict[str, Any], tools: list[Any] | None = None
) -> dict[str, Any]:
    """Remove parameters that are not supported by the given model.

    Strips any provider prefix (e.g. ``openai/o3-pro`` -> ``o3-pro``) before
    matching against known model prefixes.
    """
    model_name = model.split("/")[-1] if "/" in model else model
    if tools and any(
        model_name.startswith(p) for p in _REASONING_EFFORT_TOOL_INCOMPATIBLE_PREFIXES
    ):
        params = {k: v for k, v in params.items() if k != "reasoning_effort"}
    for prefix, unsupported in _UNSUPPORTED_PARAMS.items():
        if model_name.startswith(prefix):
            if (
                unsupported is _REASONING_UNSUPPORTED_PARAMS
                and min_reasoning_effort(model_name) == "none"
                and _reasoning_effort_is_none(params)
            ):
                # OpenAI accepts temperature/top_p on gpt-5.1+ models only at
                # effort "none" (the model's lowest supported effort). Verified
                # against the API on gpt-5.6-luna (2026-10): "none" +
                # temperature -> 200, "low" + temperature -> 400, on both the
                # chat completions and responses APIs. The reasoning guide no
                # longer documents this restriction. Applies only to models in
                # _MIN_REASONING_EFFORT — new gpt-5.1+ models must be added
                # there or temperature stays stripped.
                unsupported = unsupported - _SAMPLING_PARAMS
            params = {k: v for k, v in params.items() if k not in unsupported}
            break
    return params


def _reasoning_effort_is_none(params: dict[str, Any]) -> bool:
    # chat completions sends effort as params["reasoning_effort"]; the
    # responses plugin sends it as params["reasoning"], an openai Reasoning
    # object with an .effort attribute
    if params.get("reasoning_effort") == "none":
        return True
    return getattr(params.get("reasoning"), "effort", None) == "none"


# lowest supported reasoning effort per model; "none" requires gpt-5.1+
_MIN_REASONING_EFFORT: dict[str, ReasoningEffort] = {
    "gpt-5.1": "none",
    "gpt-5.2": "none",
    "gpt-5.4": "none",
    "gpt-5.4-mini": "none",
    "gpt-5.4-nano": "none",
    "gpt-5.5": "none",
    "gpt-5.6-luna": "none",
    "gpt-5.6-sol": "none",
    "gpt-5.6-terra": "none",
    "gpt-5": "minimal",
    "gpt-5-mini": "minimal",
    "gpt-5-nano": "minimal",
}


def min_reasoning_effort(model: str) -> ReasoningEffort | None:
    """Lowest reasoning effort the model supports, or None if the model has no
    reasoning-effort control.

    Strips any provider prefix (e.g. ``openai/gpt-5`` -> ``gpt-5``) before matching.
    """
    return _MIN_REASONING_EFFORT.get(model.split("/")[-1])


class LLM(llm.LLM):
    """LiveKit Inference LLM over the core's gateway client (``_core.InferenceLlm``)."""

    def __init__(
        self,
        model: LLMModels | str,
        *,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        inference_class: InferenceClass | None = None,
        extra_kwargs: ChatCompletionOptions | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        creds = Credentials.resolve(base_url, api_key, api_secret)
        self._inference_class = inference_class
        self._native = _core.InferenceLlm(
            model,
            base_url=creds.base_url,
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            provider=provider,
            inference_class=inference_class,
            extra_kwargs=dict(extra_kwargs or {}),
        )

    @classmethod
    def from_model_string(cls, model: str) -> LLM:
        return cls(model)

    def update_options(
        self,
        *,
        model: NotGivenOr[LLMModels | str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[ChatCompletionOptions | dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        """Read by the next ``chat()``. ``extra_kwargs`` replaces the persistent kwargs."""
        self._native.update_options(
            model=model if is_given(model) else None,
            extra_kwargs=dict(extra_kwargs) if is_given(extra_kwargs) else None,
        )

    @property
    def model(self) -> str:
        return self._native.model  # type: ignore[no-any-return]

    @property
    def provider(self) -> str:
        return "livekit"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[Any] = NOT_GIVEN,
        inference_class: NotGivenOr[InferenceClass] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        cls = inference_class if is_given(inference_class) else self._inference_class
        return _GatewayStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            request={
                "parallel_tool_calls": parallel_tool_calls
                if is_given(parallel_tool_calls)
                else None,
                "tool_choice": tool_choice if is_given(tool_choice) else None,
                "response_format": llm_utils.to_openai_response_format(response_format)  # type: ignore[arg-type]
                if is_given(response_format)
                else None,
                "extra_kwargs": extra_kwargs if is_given(extra_kwargs) else None,
                "headers": list(get_inference_headers(inference_class=cls).items()),
            },
        )


class _GatewayStream(llm.LLMStream):
    def __init__(self, llm: LLM, *, request: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(llm, **kwargs)
        self._native_llm = llm._native
        self._request = request

    async def _run(self) -> None:
        from .._core_view import to_core_items, tool_specs

        native = self._native_llm.chat(
            to_core_items(self._chat_ctx.items),
            tools=tool_specs(self._tools),
            timeout=self._conn_options.timeout,
            **self._request,
        )
        try:
            while (chunk := await native.recv()) is not None:
                self._event_ch.send_nowait(llm.ChatChunk.model_validate(chunk))
        finally:
            await native.aclose()


class LLMStream(llm.LLMStream):
    """A streamed OpenAI-compatible chat completion; the OpenAI and Google plugins run theirs
    on it with their own client."""

    def __init__(
        self,
        llm_v: llm.LLM,
        *,
        model: LLMModels | str,
        provider: str | None = None,
        inference_class: InferenceClass | None = None,
        strict_tool_schema: bool,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
        provider_fmt: str = "openai",
    ) -> None:
        super().__init__(llm_v, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._provider = provider
        self._inference_class = inference_class
        self._provider_fmt = provider_fmt
        self._strict_tool_schema = strict_tool_schema
        self._client = client
        self._extra_kwargs = drop_unsupported_params(model, extra_kwargs, tools=tools)
        self._tool_ctx = llm.ToolContext(tools)

    async def _run(self) -> None:
        retryable = True
        parser = _core.LlmChoiceParser(self._model)
        try:
            messages, _ = self._chat_ctx.to_provider_format(format=self._provider_fmt)
            schemas = self._tool_ctx.parse_function_tools("openai", strict=self._strict_tool_schema)
            _core.llm_normalize_tool_choice(self._extra_kwargs, bool(self._tools))
            headers = self._extra_kwargs.setdefault("extra_headers", {})
            headers.update(get_inference_headers(inference_class=self._inference_class))
            if self._provider:
                headers[HEADER_INFERENCE_PROVIDER] = self._provider

            stream = await self._client.chat.completions.create(
                messages=cast(list[ChatCompletionMessageParam], messages),
                tools=cast(list[ChatCompletionToolParam], schemas) or openai.omit,
                model=self._model,
                stream_options={"include_usage": True},
                stream=True,
                timeout=httpx.Timeout(self._conn_options.timeout),
                **self._extra_kwargs,
            )
            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        if (out := parser.parse(chunk.id, choice.model_dump())) is not None:
                            parsed = llm.ChatChunk.model_validate(out)
                            retryable = retryable and not parsed.has_response()
                            self._event_ch.send_nowait(parsed)
                    if chunk.usage is not None and (
                        usage := _core.llm_usage_chunk(chunk.model_dump())
                    ):
                        self._event_ch.send_nowait(llm.ChatChunk.model_validate(usage))
        except openai.APITimeoutError:
            raise APITimeoutError(retryable=retryable) from None
        except httpx.TimeoutException as e:
            raise APITimeoutError(retryable=retryable) from e
        except openai.APIStatusError as e:
            if e.status_code == 429:
                logger.warning(
                    "LLM request rate limited by inference gateway",
                    extra={
                        "model": self._model,
                        "request_id": e.request_id,
                        **extract_quota_usage(e.response.headers),
                    },
                )
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
                retryable=retryable,
            ) from None
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e


def llm_from_model_string(model: str) -> llm.LLM | llm.RealtimeModel:
    """The inference model a ``llm=`` string names: a realtime model or an LLM."""
    if is_realtime_model(model):
        from .realtime import RealtimeModel

        return RealtimeModel.from_model_string(model)
    return LLM.from_model_string(model)
