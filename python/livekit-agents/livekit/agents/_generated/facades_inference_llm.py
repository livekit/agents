# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .. import _bindings, _core
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN
from ..utils import is_given
from .llm import LLM as BaseLLM, LLMStream

if TYPE_CHECKING:
    from livekit.agents.llm import ChatContext, Tool, ToolChoice

    from ..inference import llm
    from ..types import NotGivenOr
    from .connect import APIConnectOptions
    from .inference_llm import LLMModels
    from .inference_models import InferenceClass
    from .llm_provider_options import ChatCompletionOptions, ProviderEndpoint


class LLM(BaseLLM):
    """`inference.LLM`: the LiveKit Inference gateway's LLM, or an OpenAI-compatible provider's."""

    if TYPE_CHECKING:
        def _core_native(self) -> _core.InferenceLlm: ...

    def __init__(
        self,
        model: LLMModels | str,
        *,
        provider: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        inference_class: NotGivenOr[InferenceClass] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[ChatCompletionOptions | dict[str, Any]] = NOT_GIVEN,
        endpoint: NotGivenOr[ProviderEndpoint] = NOT_GIVEN,
    ) -> None:
        """A LiveKit Inference LLM.

        Args:
            model: The model, ``"provider/model"``.
            provider: The provider the gateway routes to; the gateway picks by default.
            base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
            api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
            inference_class: The scheduling class of its requests.
            extra_kwargs: Request options sent with every completion.
            endpoint: A provider plugin's own endpoint in place of the gateway.
        """
        super().__init__()
        self._native = _core.InferenceLlm(
            model,
            provider=provider if is_given(provider) else None,
            base_url=base_url if is_given(base_url) else None,
            api_key=api_key if is_given(api_key) else None,
            api_secret=api_secret if is_given(api_secret) else None,
            inference_class=inference_class if is_given(inference_class) else None,
            extra_kwargs=dict(extra_kwargs) if is_given(extra_kwargs) else None,
            endpoint=endpoint if is_given(endpoint) else None,
        )

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, LLM, "chat")
        return self if replaced else self._core_native()

    @property
    def model(self) -> str:
        return self._native.model

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
    ) -> LLMStream:
        return GatewayLLMStream(
            llm=cast("llm.LLM", self),
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls if is_given(parallel_tool_calls) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            response_format=response_format if is_given(response_format) else None,
            inference_class=inference_class if is_given(inference_class) else None,
            extra_kwargs=extra_kwargs if is_given(extra_kwargs) else None,
        )


class GatewayLLMStream(LLMStream):
    """`inference.LLM.chat()`: one gateway completion; the client retries it under its
    ``conn_options``.
    """

    _core_native_retries = True
    _llm: llm.LLM

    def __init__(
        self,
        *,
        llm: llm.LLM,
        chat_ctx: ChatContext,
        tools: list[Tool] | None,
        conn_options: APIConnectOptions,
        parallel_tool_calls: bool | None,
        tool_choice: ToolChoice | None,
        response_format: Any | None,
        inference_class: InferenceClass | None,
        extra_kwargs: dict[str, Any] | None,
    ) -> None:
        super().__init__(llm=llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._parallel_tool_calls: bool | None = parallel_tool_calls
        self._tool_choice: ToolChoice | None = tool_choice
        self._response_format: Any | None = response_format
        self._inference_class: InferenceClass | None = inference_class
        self._extra_kwargs: dict[str, Any] | None = extra_kwargs

    async def _core_open(self) -> _core.InferenceLlmChunks:
        return self._llm._core_native().chat(
            chat_ctx=self._chat_ctx,
            tools=self._tools,
            conn_options=self._conn_options,
            parallel_tool_calls=self._parallel_tool_calls,
            tool_choice=self._tool_choice,
            response_format=self._response_format,
            inference_class=self._inference_class,
            extra_kwargs=self._extra_kwargs,
        )

    _run = LLMStream._core_run_native


GatewayLLMStream.__module__ = "livekit.agents.inference.llm"
