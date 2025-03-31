from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any, Literal

from livekit.agents._exceptions import APIConnectionError, APIError

from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from .chat_context import ChatContext
from .llm import LLM, ChatChunk, LLMStream, ToolChoice
from .tool_context import FunctionTool, ToolContext

DEFAULT_FALLBACK_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


@dataclass
class _LLMStatus:
    available: bool
    recovering_task: asyncio.Task | None


@dataclass
class AvailabilityChangedEvent:
    llm: LLM
    available: bool


class FallbackAdapter(
    LLM[Literal["llm_availability_changed"]],
):
    def __init__(
        self,
        llm: list[LLM],
        *,
        attempt_timeout: float = 10.0,
        max_retry_per_llm: int = 1,
        retry_interval: float = 5,
    ) -> None:
        if len(llm) < 1:
            raise ValueError("at least one LLM instance must be provided.")

        super().__init__()

        self._llm_instances = llm
        self._attempt_timeout = attempt_timeout
        self._max_retry_per_llm = max_retry_per_llm
        self._retry_interval = retry_interval

        self._status = [
            _LLMStatus(available=True, recovering_task=None) for _ in self._llm_instances
        ]

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_FALLBACK_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return FallbackLLMStream(
            llm=self,
            conn_options=conn_options,
            chat_ctx=chat_ctx,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )


class FallbackLLMStream(LLMStream):
    def __init__(
        self,
        llm: FallbackAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None,
        conn_options: APIConnectOptions,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._fallback_adapter = llm
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
    def tools(self) -> ToolContext | None:
        if self._current_stream is None:
            return self._tools
        return self._current_stream.tools

    async def _try_generate(
        self, *, llm: LLM, check_recovery: bool = False
    ) -> AsyncIterable[ChatChunk]:
        """
        Try to generate with the given LLM.

        Args:
            llm: The LLM instance to generate with
            check_recovery: When True, indicates this is a background recovery check and the
                          result will not be used. Recovery checks verify if a previously
                          failed LLM has become available again.
        """
        try:
            async with llm.chat(
                chat_ctx=self._chat_ctx,
                tools=self._tools,
                parallel_tool_calls=self._parallel_tool_calls,
                tool_choice=self._tool_choice,
                extra_kwargs=self._extra_kwargs,
                conn_options=dataclasses.replace(
                    self._conn_options,
                    max_retry=self._fallback_adapter._max_retry_per_llm,
                    timeout=self._fallback_adapter._attempt_timeout,
                    retry_interval=self._fallback_adapter._retry_interval,
                ),
            ) as stream:
                should_set_current = not check_recovery
                async for chunk in stream:
                    if should_set_current:
                        should_set_current = False
                        self._current_stream = stream
                    yield chunk

        except asyncio.TimeoutError:
            if check_recovery:
                logger.warning(f"{llm.label} recovery timed out")
                raise

            logger.warning(
                f"{llm.label} timed out, switching to next LLM",
            )

            raise
        except APIError as e:
            if check_recovery:
                logger.warning(
                    f"{llm.label} recovery failed",
                    exc_info=e,
                )
                raise

            logger.warning(
                f"{llm.label} failed, switching to next LLM",
                exc_info=e,
            )
            raise
        except Exception:
            if check_recovery:
                logger.exception(
                    f"{llm.label} recovery unexpected error",
                )
                raise

            logger.exception(
                f"{llm.label} unexpected error, switching to next LLM",
            )
            raise

    def _try_recovery(self, llm: LLM) -> None:
        llm_status = self._fallback_adapter._status[
            self._fallback_adapter._llm_instances.index(llm)
        ]
        if llm_status.recovering_task is None or llm_status.recovering_task.done():

            async def _recover_llm_task(llm: LLM) -> None:
                try:
                    async for _ in self._try_generate(llm=llm, check_recovery=True):
                        pass

                    llm_status.available = True
                    logger.info(f"llm.FallbackAdapter, {llm.label} recovered")
                    self._fallback_adapter.emit(
                        "llm_availability_changed",
                        AvailabilityChangedEvent(llm=llm, available=True),
                    )
                except Exception:
                    return

            llm_status.recovering_task = asyncio.create_task(_recover_llm_task(llm))

    async def _run(self) -> None:
        start_time = time.time()

        all_failed = all(not llm_status.available for llm_status in self._fallback_adapter._status)
        if all_failed:
            logger.error("all LLMs are unavailable, retrying..")

        for i, llm in enumerate(self._fallback_adapter._llm_instances):
            llm_status = self._fallback_adapter._status[i]
            if llm_status.available or all_failed:
                chunk_sent = False
                try:
                    async for result in self._try_generate(llm=llm, check_recovery=False):
                        chunk_sent = True
                        self._event_ch.send_nowait(result)

                    return
                except Exception:  # exceptions already logged inside _try_synthesize
                    if llm_status.available:
                        llm_status.available = False
                        self._fallback_adapter.emit(
                            "llm_availability_changed",
                            AvailabilityChangedEvent(llm=llm, available=False),
                        )

                    if chunk_sent:
                        raise

            self._try_recovery(llm)

        raise APIConnectionError(
            f"all LLMs failed ({[llm.label for llm in self._fallback_adapter._llm_instances]}) after {time.time() - start_time} seconds"  # noqa: E501
        )
