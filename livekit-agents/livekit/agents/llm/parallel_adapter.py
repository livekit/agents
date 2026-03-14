from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any, ClassVar

from .._exceptions import APIConnectionError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio
from .chat_context import ChatContext
from .llm import LLM, ChatChunk, LLMStream
from .tool_context import Tool, ToolChoice


class ParallelAdapter(LLM):
    """Sends the same request to multiple LLMs in parallel and returns the first response.

    This is useful for hedging latency — by racing multiple providers (e.g. OpenAI and Azure),
    you get the response from whichever is fastest, reducing p99 latency.
    """

    def __init__(
        self,
        llm: list[LLM],
        *,
        attempt_timeout: float = 10.0,
    ) -> None:
        """Create a ParallelAdapter.

        Args:
            llm: List of LLM instances to race against each other.
            attempt_timeout: Timeout for each individual LLM attempt. Defaults to 10.0.

        Raises:
            ValueError: If fewer than 2 LLM instances are provided.
        """
        if len(llm) < 2:
            raise ValueError("at least two LLM instances must be provided for parallel inference.")

        super().__init__()

        self._llm_instances = llm
        self._attempt_timeout = attempt_timeout

        for llm_instance in self._llm_instances:
            llm_instance.on("metrics_collected", self._on_metrics_collected)

    @property
    def model(self) -> str:
        return "ParallelAdapter"

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
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return ParallelLLMStream(
            llm=self,
            conn_options=conn_options,
            chat_ctx=chat_ctx,
            tools=tools or [],
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )

    async def aclose(self) -> None:
        for llm_instance in self._llm_instances:
            llm_instance.off("metrics_collected", self._on_metrics_collected)

    def _on_metrics_collected(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)


class ParallelLLMStream(LLMStream):
    _llm_request_span_name: ClassVar[str] = "llm_parallel_adapter"

    def __init__(
        self,
        llm: ParallelAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._parallel_adapter = llm
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice
        self._extra_kwargs = extra_kwargs

        self._current_stream: LLMStream | None = None

    @property
    def chat_ctx(self) -> ChatContext:
        if self._current_stream is None:
            return self._chat_ctx
        return self._current_stream.chat_ctx

    @property
    def tools(self) -> list[Tool]:
        if self._current_stream is None:
            return self._tools
        return self._current_stream.tools

    async def _run(self) -> None:
        winner_index: int | None = None
        chunk_ch = aio.Chan[ChatChunk]()

        async def _race_llm(index: int, llm_instance: LLM) -> None:
            nonlocal winner_index
            try:
                async with llm_instance.chat(
                    chat_ctx=self._chat_ctx,
                    tools=self._tools,
                    parallel_tool_calls=self._parallel_tool_calls,
                    tool_choice=self._tool_choice,
                    extra_kwargs=self._extra_kwargs,
                    conn_options=APIConnectOptions(
                        max_retry=0,
                        timeout=self._parallel_adapter._attempt_timeout,
                    ),
                ) as stream:
                    async for chunk in stream:
                        if winner_index is None:
                            winner_index = index
                            self._current_stream = stream
                            logger.debug(
                                "llm.ParallelAdapter: %s won the race",
                                llm_instance.label,
                            )
                        if winner_index == index:
                            chunk_ch.send_nowait(chunk)
                        else:
                            # Another LLM won the race, stop
                            return
            except Exception as e:
                if winner_index is None:
                    logger.warning(
                        "llm.ParallelAdapter: %s failed",
                        llm_instance.label,
                        exc_info=e,
                    )

        tasks = [
            asyncio.create_task(
                _race_llm(i, llm_instance),
                name=f"ParallelAdapter._race_llm_{llm_instance.label}",
            )
            for i, llm_instance in enumerate(self._parallel_adapter._llm_instances)
        ]

        def _on_all_done(*_: Any) -> None:
            # Close the channel when all tasks are done (winner finished or all failed)
            if all(t.done() for t in tasks):
                chunk_ch.close()

        for t in tasks:
            t.add_done_callback(_on_all_done)

        try:
            async for chunk in chunk_ch:
                self._event_ch.send_nowait(chunk)

            if winner_index is None:
                raise APIConnectionError(
                    "all LLMs failed in parallel "
                    f"({[llm.label for llm in self._parallel_adapter._llm_instances]})"
                )
        finally:
            for i, task in enumerate(tasks):
                if i != winner_index and not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[ChatChunk]) -> None:
        return
