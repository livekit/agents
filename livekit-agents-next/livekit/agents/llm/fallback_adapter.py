from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from .. import _bindings, _core
from .._generated.llm import LLM, ChatChunk, LLMStream
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from .chat_context import ChatContext, MetricsMetadata
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
        sticky: bool = False,
    ) -> None:
        """
        Args:
            llm: The LLMs, in order of preference.
            attempt_timeout: The timeout of one attempt.
            max_retry_per_llm: Retries of each before moving to the next.
            retry_interval: The wait between retries.
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
            "sticky": sticky,
        }

    @property
    def _native(self) -> _core.FallbackLlm:
        # one per running loop: its members are called on the loop the chain was built on
        return _bindings.on_loop(
            self,
            lambda: _core.FallbackLlm(
                [_bindings.core_leg(t) for t in self._llm_instances],
                on_availability=self,
                **self._opts,
            ),
        )

    def _core_leg(self) -> Any:
        return self if _bindings.overrides(self, FallbackAdapter, "chat") else self._native

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
        chains = _bindings.built_on_loop(self)
        return self._llm_instances[chains[0].active_index if chains else 0]

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
            # the core's chain retries and falls back: the stream does not retry on top of it
            conn_options=replace(conn_options, max_retry=0),
            request={
                "parallel_tool_calls": parallel_tool_calls
                if is_given(parallel_tool_calls)
                else None,
                "tool_choice": tool_choice if is_given(tool_choice) else None,
                "extra_kwargs": extra_kwargs if is_given(extra_kwargs) else None,
            },
        )

    def prewarm(self, **kwargs: Any) -> None:
        self._llm_instances[0].prewarm(**kwargs)

    async def aclose(self) -> None:
        await super().aclose()
        for chain in _bindings.built_on_loop(self):
            await chain.aclose()


class FallbackLLMStream(LLMStream):
    _llm: FallbackAdapter

    def __init__(self, llm: FallbackAdapter, *, request: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(llm, **kwargs)
        self._request = request

    async def _run(self) -> None:
        from .._core_view import tool_specs

        native = await self._llm._native.chat(
            self._chat_ctx,
            tools=tool_specs(self._tools),
            tools_host=self._tools,
            conn_options=self._conn_options,
            **self._request,
        )
        try:
            while (chunk := await native.recv()) is not None:
                self.send(ChatChunk.model_validate(chunk))
        finally:
            await native.aclose()
