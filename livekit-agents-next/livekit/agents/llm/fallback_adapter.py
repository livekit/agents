from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Literal

from .. import _core
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from .chat_context import ChatContext, MetricsMetadata
from .llm import LLM, ChatChunk, LLMStream
from .tool_context import Tool, ToolChoice

__all__ = ["AvailabilityChangedEvent", "FallbackAdapter"]


@dataclass
class AvailabilityChangedEvent:
    llm: LLM
    available: bool


class FallbackAdapter(LLM[Literal["llm_availability_changed"]]):
    """Several LLMs behind one: a failed one is skipped until it recovers. The chain is the
    core's (``_core.FallbackLlm``)."""

    def __init__(
        self,
        llm: list[LLM],
        *,
        attempt_timeout: float = 5.0,
        max_retry_per_llm: int = 0,
        retry_interval: float = 0.5,
        retry_on_chunk_sent: bool = False,
        sticky: bool = False,
    ) -> None:
        """
        Args:
            llm: The LLMs, in order of preference.
            attempt_timeout: The timeout of one attempt.
            max_retry_per_llm: Retries of each before moving to the next.
            retry_interval: The wait between retries.
            retry_on_chunk_sent: Whether a failure after output was sent moves to the next LLM.
            sticky: Keep the LLM that last served until it fails, even once a preferred one
                recovers; on failure the rest are tried in order.
        """
        if not llm:
            raise ValueError("at least one LLM instance must be provided.")
        super().__init__()
        self._llm_instances = llm
        self._opts = {
            "attempt_timeout": attempt_timeout,
            "max_retry_per_llm": max_retry_per_llm,
            "retry_interval": retry_interval,
            "retry_on_chunk_sent": retry_on_chunk_sent,
            "sticky": sticky,
        }

    @functools.cached_property
    def _native(self) -> _core.FallbackLlm:
        # built on first use: the chain lives on the running loop
        return _core.FallbackLlm(self._llm_instances, on_availability=self, **self._opts)

    def _core_availability_changed(self, ev: dict[str, Any]) -> None:
        self.emit(
            "llm_availability_changed",
            AvailabilityChangedEvent(
                llm=self._llm_instances[ev["index"]], available=ev["available"]
            ),
        )

    def _core_all_unavailable(self) -> None:
        logger.error("all LLMs are unavailable, retrying..")

    @property
    def _active(self) -> LLM:
        return self._llm_instances[self._native.active_index if "_native" in vars(self) else 0]

    @property
    def model(self) -> str:
        return self._active.model

    @property
    def provider(self) -> str:
        return self._active.provider

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        return self._active.metrics_metadata

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
        return FallbackLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            request={
                "parallel_tool_calls": parallel_tool_calls if parallel_tool_calls else None,
                "tool_choice": tool_choice if tool_choice else None,
                "extra_kwargs": extra_kwargs if extra_kwargs else None,
            },
        )

    def prewarm(self, **kwargs: Any) -> None:
        self._llm_instances[0].prewarm(**kwargs)

    async def aclose(self) -> None:
        await super().aclose()
        if "_native" in vars(self):
            await self._native.aclose()


class FallbackLLMStream(LLMStream):
    _llm: FallbackAdapter

    def __init__(self, llm: FallbackAdapter, *, request: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(llm, **kwargs)
        self._request = request
        # a failure after output reaches the caller as final unless asked otherwise
        self._retry_on_chunk_sent = llm._opts["retry_on_chunk_sent"]

    async def _run(self) -> None:
        from .._core_view import tool_specs

        native = await self._llm._native.chat(
            self._chat_ctx, tools=tool_specs(self._tools), tools_host=self._tools, **self._request
        )
        try:
            while (chunk := await native.recv()) is not None:
                self._event_ch.send_nowait(ChatChunk.model_validate(chunk))
        finally:
            await native.aclose()
