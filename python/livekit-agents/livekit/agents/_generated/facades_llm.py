# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from .. import _bindings, _core
from .._bindings import core_legs
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN
from ..utils import is_given
from . import llm_availability
from .llm import LLM, LLMStream

if TYPE_CHECKING:
    from collections.abc import Sequence

    from livekit.agents.llm import ChatContext, Tool, ToolChoice

    from .. import llm
    from ..types import NotGivenOr
    from .chat_items import MetricsMetadata
    from .connect import APIConnectOptions
    from .llm_fallback import AvailabilityChanged


class FallbackAdapter(LLM[Literal["llm_availability_changed"]]):
    """`llm.FallbackAdapter`: several LLMs behind one; a failed one is skipped until it recovers."""

    if TYPE_CHECKING:
        _members: list[LLM[Any]]

        def _core_failed(self, error: BaseException, recoverable: bool) -> None: ...

    def __init__(
        self,
        llm: Sequence[LLM[Any]],
        *,
        attempt_timeout: float = 5.0,
        max_retry_per_llm: int = 0,
        retry_interval: float = 0.5,
        sticky: bool = False,
    ) -> None:
        """Several LLMs behind one: a failed one is skipped until it recovers.

        Args:
            llm: The LLMs, in order of preference.
            attempt_timeout: The timeout of one attempt.
            max_retry_per_llm: Retries of each before moving to the next.
            retry_interval: The wait between retries.
            sticky: Keep the LLM that last served until it fails, even once a preferred one
                recovers; on failure the rest are tried in order.
        """
        super().__init__()
        self._native = _core.FallbackLlm(
            core_legs(llm),
            attempt_timeout=attempt_timeout,
            max_retry_per_llm=max_retry_per_llm,
            retry_interval=retry_interval,
            sticky=sticky,
            on_availability=self,
        )
        self._members = list(llm)

    def _core_leg(self) -> Any:
        """This engine, unless a subclass replaces what it would bypass."""
        replaced = _bindings.overrides(self, FallbackAdapter, "chat")
        return self if replaced else self._native

    def _core_availability_changed(self, ev: AvailabilityChanged) -> None:
        self.emit(
            "llm_availability_changed",
            llm_availability.AvailabilityChangedEvent(self._members[ev.index], ev.available),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all LLMs are unavailable, retrying..")

    @property
    def _active(self) -> LLM[Any]:
        return self._members[self._native.active_index]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def provider(self) -> str:
        return self._active.provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._active.metrics_metadata

    async def aclose(self) -> None:
        await super().aclose()
        await self._native.aclose()

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> LLMStream:
        return FallbackLLMStream(
            llm=cast("llm.FallbackAdapter", self),
            chat_ctx=chat_ctx,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls if is_given(parallel_tool_calls) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            extra_kwargs=extra_kwargs if is_given(extra_kwargs) else None,
            conn_options=conn_options,
        )


class FallbackLLMStream(LLMStream):
    """`llm.FallbackAdapter.chat()`: the chain's completion, which retries and falls back."""

    _core_native_retries = True
    _llm: llm.FallbackAdapter

    def __init__(
        self,
        *,
        llm: llm.FallbackAdapter,
        chat_ctx: ChatContext,
        tools: list[Tool] | None,
        parallel_tool_calls: bool | None,
        tool_choice: ToolChoice | None,
        extra_kwargs: dict[str, Any] | None,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm=llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._parallel_tool_calls: bool | None = parallel_tool_calls
        self._tool_choice: ToolChoice | None = tool_choice
        self._extra_kwargs: dict[str, Any] | None = extra_kwargs

    async def _core_open(self) -> _core.FallbackLlmChunks:
        return await self._llm._native.chat(
            chat_ctx=self._chat_ctx,
            tools=self._tools,
            parallel_tool_calls=self._parallel_tool_calls,
            tool_choice=self._tool_choice,
            extra_kwargs=self._extra_kwargs,
            conn_options=self._conn_options,
        )

    _run = LLMStream._core_run_native


FallbackLLMStream.__module__ = "livekit.agents.llm"
