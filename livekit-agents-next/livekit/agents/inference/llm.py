from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast, get_args

from .. import _core, llm
from .._bindings import APIConnectionError, APIStatusError, APITimeoutError
from .._generated import facades_inference_llm as _facades, public_inference_llm as _public
from .._generated.public_inference_llm import *  # noqa: F403
from ..llm import ToolChoice, utils as llm_utils
from ..llm.chat_context import ChatContext
from ..llm.tool_context import Tool
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from . import (
    HEADER_INFERENCE_PROVIDER,
    InferenceClass,
    _resolve_credentials,
    extract_quota_usage,
    get_inference_headers,
)

if TYPE_CHECKING:
    import openai

    from ..llm import ToolChoice


__all__ = [
    "LLM",
    "LLMStream",
    "ProviderEndpoint",
]
__all__ += _public.__all__

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


def drop_unsupported_params(
    model: str, params: dict[str, Any], tools: list[Any] | None = None
) -> dict[str, Any]:
    """Remove parameters that are not supported by the given model.

    Strips any provider prefix (e.g. ``openai/o3-pro`` -> ``o3-pro``) before
    matching against known model prefixes.
    """
    effort = params.get("reasoning_effort")
    drop = _core.llm_unsupported_params(
        model,
        has_tools=bool(tools),
        reasoning_effort=effort if isinstance(effort, str) else None,
        reasoning=getattr(params.get("reasoning"), "effort", None),
    )
    return {k: v for k, v in params.items() if k not in drop}


def min_reasoning_effort(model: str) -> ReasoningEffort | None:
    """Lowest reasoning effort the model supports, or None if the model has no
    reasoning-effort control.

    Strips any provider prefix (e.g. ``openai/gpt-5`` -> ``gpt-5``) before matching.
    """
    return cast("ReasoningEffort | None", _core.min_reasoning_effort(model))


@dataclasses.dataclass(frozen=True)
class ProviderEndpoint:
    """An OpenAI-compatible provider the core's client calls directly with the provider's own
    key, in place of the LiveKit Inference gateway; no LiveKit header is sent."""

    base_url: str
    api_key: str
    strict_tool_schema: bool = True
    """Tool schemas in OpenAI's strict form; some OpenAI-compatible providers refuse it."""
    gzip_body: bool = False
    """Gzip-compressed request bodies, for a provider that accepts them."""
    msgpack_body: bool = False
    """Msgpack request bodies, for a provider that accepts them."""


class LLM(_facades.LLM, llm.LLM):
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
        _endpoint: ProviderEndpoint | None = None,
    ) -> None:
        """``_endpoint``: a provider plugin's own endpoint in place of the gateway."""
        super().__init__()
        self._model = model
        self._provider = provider
        self._inference_class = inference_class
        self._extra_kwargs: dict[str, object] = dict(extra_kwargs or {})
        # the client is built from these on first use, after a fork
        if _endpoint is None:
            credentials = _resolve_credentials(base_url, api_key, api_secret)
            self._base_url, self._api_key, self._api_secret = dataclasses.astuple(credentials)
            self._strict_tool_schema, self._gzip_body, self._msgpack_body = True, False, False
        else:
            self._base_url, self._api_key, self._api_secret = (
                _endpoint.base_url,
                _endpoint.api_key,
                None,
            )
            self._strict_tool_schema = _endpoint.strict_tool_schema
            self._gzip_body, self._msgpack_body = _endpoint.gzip_body, _endpoint.msgpack_body

    @classmethod
    def from_model_string(cls, model: str) -> LLM:
        return cls(model)

    def _core_native(self) -> _core.InferenceLlm:
        """The client as a session of the current job holds it: its requests carry that job."""
        headers = get_inference_headers(inference_class=self._inference_class)
        return self._native.identified(list(headers.items()), self)

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
                "response_format": llm_utils.to_openai_response_format(response_format)
                if is_given(response_format)
                else None,
                "extra_kwargs": extra_kwargs if is_given(extra_kwargs) else None,
                "headers": list(get_inference_headers(inference_class=cls).items()),
            },
        )


class _GatewayStream(llm.LLMStream):
    def __init__(
        self, llm: LLM, *, request: dict[str, Any], conn_options: APIConnectOptions, **kwargs: Any
    ) -> None:
        # The native request retries per `conn_options`; the base's loop runs `_run` once.
        super().__init__(llm, conn_options=dataclasses.replace(conn_options, max_retry=0), **kwargs)
        self._native_llm = llm._native
        self._request_conn_options = conn_options
        self._request = request

    async def _run(self) -> None:
        from .._bindings import tool_specs

        native = self._native_llm.chat(
            self._chat_ctx.items,
            tools=tool_specs(self._tools),
            conn_options=self._request_conn_options,
            **self._request,
        )
        try:
            while (chunk := await native.recv()) is not None:
                self.send(chunk)
        finally:
            await native.aclose()


class LLMStream(llm.LLMStream):
    """A streamed OpenAI-compatible chat completion; Azure OpenAI and the Google plugin run theirs
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
    ) -> None:
        super().__init__(llm_v, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._model = model
        self._provider = provider
        self._inference_class = inference_class
        self._strict_tool_schema = strict_tool_schema
        self._client = client
        self._extra_kwargs = drop_unsupported_params(model, extra_kwargs, tools=tools)
        self._tool_ctx = llm.ToolContext(tools)

    async def _run(self) -> None:
        # Only the Python provider plugins use this compatibility stream. Native
        # inference does not need the OpenAI SDK installed or imported.
        import httpx2 as httpx
        import openai
        from openai.types.chat import ChatCompletionToolParam

        options = self._conn_options
        timeout = httpx.Timeout(
            connect=options.connect_timeout,
            read=options.read_timeout,
            write=options.write_timeout,
            pool=options.pool_timeout,
        )
        client = self._client.with_options(max_retries=0)
        retryable = True
        parser = _core.LlmChoiceParser(self._model)
        try:
            messages, _ = await self._chat_ctx.to_provider_format(format="openai")
            schemas = self._tool_ctx.parse_function_tools("openai", strict=self._strict_tool_schema)
            extra = dict(self._extra_kwargs)
            if not self._tools:
                extra.pop("tool_choice", None)  # a tool choice without tools is refused
            headers = extra.setdefault("extra_headers", {})
            headers.update(get_inference_headers(inference_class=self._inference_class))
            if self._provider:
                headers[HEADER_INFERENCE_PROVIDER] = self._provider

            stream = await client.chat.completions.create(
                messages=messages,
                tools=cast(list[ChatCompletionToolParam], schemas) or openai.omit,
                model=self._model,
                stream_options={"include_usage": True},
                stream=True,
                timeout=timeout,
                **extra,
            )
            async with stream:
                async for chunk in stream:
                    for choice in chunk.choices:
                        if (parsed := parser.parse(chunk.id, choice.model_dump())) is not None:
                            retryable = retryable and not parsed.has_response()
                            self.send(parsed)
                    if chunk.usage is not None and (
                        usage := _core.llm_usage_chunk(chunk.model_dump())
                    ):
                        self.send(usage)
        except openai.APITimeoutError as e:
            raise APITimeoutError(retryable=retryable) from e
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
            ) from e
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e


def llm_from_model_string(model: str) -> llm.LLM | llm.RealtimeModel:
    """The inference model a ``llm=`` string names: a realtime model or an LLM."""
    if is_realtime_model(model):
        from .realtime import RealtimeModel

        return RealtimeModel.from_model_string(model)
    return LLM.from_model_string(model)
