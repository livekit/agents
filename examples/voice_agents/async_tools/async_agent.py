# Copyright 2025 LiveKit, Inc.
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

import asyncio
import inspect
import json
import logging
import os
import sys
import typing
from typing import TYPE_CHECKING, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from async_function_tool import (
    AgentNotification,
    AsyncFunctionTool,
    ToolReplyMode,
    find_async_function_tools,
)

from livekit.agents import Agent, RunContext, llm, stt, tts, vad
from livekit.agents.llm.tool_context import FunctionTool, FunctionToolInfo, ToolFlag
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.voice.events import AgentStateChangedEvent

if TYPE_CHECKING:
    from livekit.agents.inference import LLMModels, STTModels, TTSModels
    from livekit.agents.llm import mcp
    from livekit.agents.voice.audio_recognition import TurnDetectionMode

logger = logging.getLogger(__name__)


class AsyncAgent(Agent):
    """``Agent`` subclass that supports ``@async_function_tool`` decorated async generators.

    ``AsyncAgent`` extends the standard ``Agent`` with support for long-running tool
    functions that can yield intermediate results. It works by converting
    ``@async_function_tool`` decorated methods into regular ``FunctionTool`` instances
    at construction time.

    How it works:

    1. The **first yield** from an async tool becomes the immediate tool return
       (sent to the LLM as a normal ``FunctionCallOutput``). This is **not** affected
       by ``reply_mode``.
    2. A background task is spawned for **subsequent yields** (second yield onward).
       These are notifications scheduled as speech based on the tool's ``reply_mode``:

       - ``"when_idle"``: Wait for the agent to finish speaking, then trigger a reply.
       - ``"interrupt"``: Interrupt the agent's speech and trigger an immediate reply.
       - ``"silent"``: Update the chat context without triggering a reply.

    3. Each notification includes the original ``tool_call_id`` so the LLM can
       correlate background updates back to the original tool call.

    Usage::

        from async_agent import AsyncAgent
        from async_function_tool import async_function_tool, notify

        class MyAgent(AsyncAgent):
            def __init__(self):
                super().__init__(instructions="You are helpful.")

            @async_function_tool(reply_mode="when_idle")
            async def search_web(self, query: str):
                yield f"Searching for '{query}'..."   # first yield = immediate tool return
                await asyncio.sleep(5)
                yield f"Found results for '{query}'"  # second yield = notification (scheduled when idle)
    """

    def __init__(
        self,
        *,
        instructions: str,
        id: str | None = None,
        chat_ctx: NotGivenOr[llm.ChatContext | None] = NOT_GIVEN,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | STTModels | str | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | LLMModels | str | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | TTSModels | str | None] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[mcp.MCPServer] | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
        use_tts_aligned_transcript: NotGivenOr[bool] = NOT_GIVEN,
        min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Initialize an ``AsyncAgent``.

        Accepts all the same arguments as ``Agent``. ``AsyncFunctionTool`` instances
        in the ``tools`` list and ``@async_function_tool`` decorated methods on the
        class are automatically discovered and converted into regular ``FunctionTool``
        instances.

        Args:
            instructions: System instructions for the agent.
            id: Optional agent identifier.
            chat_ctx: Initial ``ChatContext``.
            tools: List of tools. ``AsyncFunctionTool`` instances are automatically
                converted; regular tools are passed through unchanged.
            turn_detection: ``TurnDetectionMode`` configuration.
            stt: ``STT`` engine or model string.
            vad: ``VAD`` engine.
            llm: ``LLM`` engine, ``RealtimeModel``, or model string.
            tts: ``TTS`` engine or model string.
            mcp_servers: List of ``MCPServer`` instances for tool discovery.
            allow_interruptions: Whether user speech can interrupt the agent.
            min_consecutive_speech_delay: Min delay between consecutive speech segments.
            use_tts_aligned_transcript: Whether to use TTS-aligned transcripts.
            min_endpointing_delay: Minimum endpointing delay in seconds.
            max_endpointing_delay: Maximum endpointing delay in seconds.
        """
        tools = tools or []

        # Discover @async_function_tool methods on this instance
        async_tools = find_async_function_tools(self)

        # Separate async tools from regular tools in the tools list
        regular_tools: list[llm.Tool | llm.Toolset] = []
        for t in tools:
            if isinstance(t, AsyncFunctionTool):
                async_tools.append(t)
            else:
                regular_tools.append(t)

        # Convert each async tool into a regular FunctionTool
        wrapped_tools: list[FunctionTool] = []
        for at in async_tools:
            wrapped = self._wrap_async_tool(at)
            wrapped_tools.append(wrapped)

        super().__init__(
            instructions=instructions,
            id=id,
            chat_ctx=chat_ctx,
            tools=[*regular_tools, *wrapped_tools],
            turn_detection=turn_detection,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            mcp_servers=mcp_servers,
            allow_interruptions=allow_interruptions,
            min_consecutive_speech_delay=min_consecutive_speech_delay,
            use_tts_aligned_transcript=use_tts_aligned_transcript,
            min_endpointing_delay=min_endpointing_delay,
            max_endpointing_delay=max_endpointing_delay,
        )

        self._background_tasks: dict[str, asyncio.Task[None]] = {}

    def _wrap_async_tool(self, async_tool: AsyncFunctionTool) -> FunctionTool:
        """Convert an ``AsyncFunctionTool`` into a regular ``FunctionTool``.

        The wrapped tool:
        1. Starts the async generator.
        2. Awaits the first yield as the immediate tool return.
        3. Spawns a background task for remaining yields (second yield onward),
           scheduled as speech based on ``reply_mode``.
        4. Returns the first yield value.
        """
        info = async_tool.info
        reply_mode = info.reply_mode
        tool_name = info.name
        tool_description = info.description

        agent_ref = self

        # Find if/which param name the user typed for RunContext
        try:
            original_hints = typing.get_type_hints(async_tool._func, include_extras=True)
        except Exception:
            original_hints = {}
        original_hints.pop("self", None)
        original_hints.pop("return", None)

        # e.g. user wrote "ctx: RunContext" -> user_ctx_param_name = "ctx"
        user_ctx_param_name: str | None = None
        for param_name, param_type in original_hints.items():
            if param_type is RunContext:
                user_ctx_param_name = param_name
                break

        # Name we use in the wrapper's signature for RunContext injection
        # If user already has one, reuse their name; otherwise add our own
        ctx_param_name = user_ctx_param_name or "_async_tool_run_ctx"

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # RunContext is always injected by the framework because we ensure
            # our wrapper's __signature__ and __annotations__ include it (see below).
            # signature.bind() may put it in args (POSITIONAL_OR_KEYWORD) or kwargs
            # (KEYWORD_ONLY), so search both by type.
            run_ctx: RunContext | None = None
            ctx_idx: int | None = None  # index in args, if found there

            for i, v in enumerate(args):
                if isinstance(v, RunContext):
                    run_ctx = v
                    ctx_idx = i
                    break

            if run_ctx is None:
                for v in kwargs.values():
                    if isinstance(v, RunContext):
                        run_ctx = v
                        break

            if run_ctx is None:
                raise RuntimeError(
                    f"RunContext was not injected into async tool '{tool_name}'. "
                    "This is a bug in AsyncAgent._wrap_async_tool signature setup."
                )

            tool_call_id = run_ctx.function_call.call_id

            # If user's function does NOT expect RunContext, strip it before forwarding
            if user_ctx_param_name is None:
                if ctx_idx is not None:
                    args = args[:ctx_idx] + args[ctx_idx + 1 :]
                else:
                    kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, RunContext)}

            # Call the original async generator
            gen = async_tool(*args, **kwargs)

            # Get the first yield — this is the immediate tool return
            try:
                first_result = await gen.__anext__()
            except StopAsyncIteration:
                return None

            # Unwrap AgentNotification if the first yield is wrapped
            immediate_value = (
                first_result.value if isinstance(first_result, AgentNotification) else first_result
            )

            # Register background task for remaining generator yields
            task_id = f"{tool_name}_{tool_call_id}"
            agent_ref._register_background_task(
                task_id=task_id,
                generator=gen,
                reply_mode=reply_mode,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

            return immediate_value

        # Build wrapper's signature and annotations so the framework always injects RunContext.
        # We resolve type hints from the ORIGINAL function (using its __globals__) to get
        # actual type objects, avoiding cross-module string annotation resolution failures.
        wrapper_hints = dict(original_hints)
        if user_ctx_param_name is None:
            # User didn't declare RunContext — add it so the framework injects it for us
            wrapper_hints[ctx_param_name] = RunContext

        wrapper.__annotations__ = wrapper_hints
        wrapper.__name__ = async_tool._func.__name__
        wrapper.__qualname__ = async_tool._func.__qualname__
        wrapper.__module__ = async_tool._func.__module__
        wrapper.__doc__ = async_tool._func.__doc__

        # Build __signature__ with all params (including run_ctx).
        # inspect.signature checks __signature__ first, so prepare_function_arguments
        # will see the correct params.
        if hasattr(async_tool, "__signature__"):
            base_sig = async_tool.__signature__
        else:
            sig = inspect.signature(async_tool._func)
            params = list(sig.parameters.values())
            # Remove 'self' if present (class method)
            if params and params[0].name == "self":
                params = params[1:]
            base_sig = sig.replace(parameters=params)

        if user_ctx_param_name is None:
            # Append our RunContext parameter to the signature
            run_ctx_param = inspect.Parameter(
                ctx_param_name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=RunContext,
            )
            new_params = list(base_sig.parameters.values()) + [run_ctx_param]
            base_sig = base_sig.replace(parameters=new_params)

        wrapper.__signature__ = base_sig  # type: ignore[attr-defined]

        fnc_info = FunctionToolInfo(
            name=tool_name,
            description=tool_description,
            flags=ToolFlag.NONE,
        )
        return FunctionTool(wrapper, fnc_info)

    def _register_background_task(
        self,
        *,
        task_id: str,
        generator: Any,
        reply_mode: ToolReplyMode,
        tool_name: str,
        tool_call_id: str,
    ) -> None:
        async def _run_background() -> None:
            latest_output: Any = None
            try:
                async for value in generator:
                    if isinstance(value, AgentNotification):
                        await self._handle_notification(
                            tool_name, tool_call_id, value.value, value.mode
                        )
                        latest_output = value.value
                    else:
                        latest_output = value
            except Exception:
                logger.exception(
                    "background async tool '%s' failed",
                    tool_name,
                    extra={"tool_name": tool_name, "tool_call_id": tool_call_id},
                )
                error_mode: ToolReplyMode = "when_idle" if reply_mode != "silent" else "silent"
                await self._handle_notification(
                    tool_name, tool_call_id, "An error occurred in background task", error_mode
                )
                return
            finally:
                self._background_tasks.pop(task_id, None)

            # Handle final result with the tool's reply_mode
            if latest_output is not None:
                await self._handle_notification(tool_name, tool_call_id, latest_output, reply_mode)

        task = asyncio.create_task(_run_background(), name=f"async_tool_{task_id}")
        self._background_tasks[task_id] = task

    def render_notification(
        self,
        tool_name: str,
        tool_call_id: str,
        value: Any,
        mode: ToolReplyMode,
    ) -> str:
        """Render a tool notification into a message string for the chat context.

        Override this method to customize how async tool notifications appear in the
        conversation. The returned string is added as a user message to the chat context
        so the LLM can see it.

        Args:
            tool_name: Name of the async tool that produced this notification.
            tool_call_id: The original ``tool_call_id`` so the LLM can correlate
                this update back to the tool call.
            value: The notification payload (already the raw value from the yield).
            mode: The ``ToolReplyMode`` for this notification (``"when_idle"``,
                ``"interrupt"``, or ``"silent"``).

        Returns:
            The rendered message string to insert into the chat context.
        """
        str_value = value if isinstance(value, str) else json.dumps(value, default=str)
        return (
            f"[Async tool update] tool_name={tool_name} tool_call_id={tool_call_id}\n"
            f"Output: {str_value}"
        )

    async def _handle_notification(
        self,
        tool_name: str,
        tool_call_id: str,
        value: Any,
        mode: ToolReplyMode,
    ) -> None:
        message = self.render_notification(tool_name, tool_call_id, value, mode)

        logger.debug(
            "async tool notification",
            extra={
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "mode": mode,
                "value": message,
            },
        )

        # Update chat context with the notification
        chat_ctx = self._chat_ctx.copy()
        chat_ctx.add_message(role="user", content=message)
        await self.update_chat_ctx(chat_ctx)

        if mode == "silent":
            return

        session = self.session

        if mode == "when_idle":
            await self._wait_for_idle()
            session.generate_reply()
            return

        if mode == "interrupt":
            session.interrupt()
            session.generate_reply()
            return

    async def _wait_for_idle(self) -> None:
        session = self.session
        if session.agent_state == "listening":
            return

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[None] = loop.create_future()

        def on_state_changed(ev: AgentStateChangedEvent) -> None:
            if ev.new_state == "listening" and not fut.done():
                fut.set_result(None)

        session.on("agent_state_changed", on_state_changed)
        try:
            await fut
        finally:
            session.off("agent_state_changed", on_state_changed)

    async def on_exit(self) -> None:
        # Cancel all background generator tasks
        for task in self._background_tasks.values():
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks.values(), return_exceptions=True)

        self._background_tasks.clear()
        await super().on_exit()
