from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_origin, get_type_hints

from .. import utils
from ..llm.chat_context import ChatItem, FunctionCall
from ..llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    Tool,
    Toolset,
    function_tool,
)
from ..llm.utils import function_arguments_to_pydantic_model, prepare_function_arguments
from ..log import logger
from ..voice.agent import _pass_through_activity_task_info
from ..voice.events import RunContext, Userdata_T
from ..voice.generation import ToolExecutionOutput, make_tool_output

if TYPE_CHECKING:
    from ..voice.agent_session import AgentSession


UPDATE_TEMPLATE = """The tool `{function_name}` has updated, message: {message}
The task is still running, so DON'T make up or give information not included in the message above."""

DUPLICATE_REJECT = """Same tool `{function_name}` is already running:
{running_fnc_calls}
If you want to cancel the existing one, call `cancel_task` with call_id."""

DUPLICATE_CONFIRM = """Same tool `{function_name}` is already running:
{running_fnc_calls}
Re-call with confirm duplicate True to run a duplicate if needed,
or if you want to cancel the existing one, call `cancel_task` with call_id."""

REPLY_INSTRUCTIONS = """New results arrived from background tool calls (call_ids: {pending_call_ids}).
Summarize these results to the user naturally. Do NOT repeat information you have already told the user."""


class AsyncRunContext(RunContext[Userdata_T]):
    """Run context for async tool functions."""

    def __init__(self, *, run_ctx: RunContext[Userdata_T], toolset: AsyncToolset) -> None:
        super().__init__(
            session=run_ctx.session,
            speech_handle=run_ctx.speech_handle,
            function_call=run_ctx.function_call,
        )
        self._toolset = toolset
        self._pending_fut = asyncio.Future[Any]()
        self._step_idx: int = 0

    async def update(self, message: str | Any, *, _template: str = UPDATE_TEMPLATE) -> None:
        """Push an intermediate progress update into the conversation.

        Updates the chat context immediately and enqueues a speech delivery
        at the toolset level. Multiple updates from different tools are
        coalesced — only the latest pending update triggers a reply.

        Args:
            message: The update message for the LLM (e.g. "Found 3 flights, selecting the best option...").
        """
        if isinstance(message, str):
            message = _template.format(
                function_name=self.function_call.name,
                call_id=self.function_call.call_id,
                message=message,
            )

        if not self._pending_fut.done():
            # first update will mark the tool execution in AgentActivity done
            self._pending_fut.set_result(message)
            # make the speech handle awaitable in the rest of the function
            self._function_call.extra["__livekit_agents_tool_pending"] = True
            return

        self._step_idx += 1

        tool_output = self._make_tool_output(
            message, call_id=f"{self.function_call.call_id}/update_{self._step_idx}"
        )
        if tool_output.fnc_call_out is None:
            return

        tool_items: list[ChatItem] = [tool_output.fnc_call, tool_output.fnc_call_out]
        await self._toolset._enqueue_reply(self, tool_items)

    def _make_tool_output(
        self, output: Any | BaseException, call_id: str | None
    ) -> ToolExecutionOutput:
        exception: BaseException | None = None
        if isinstance(output, BaseException):
            exception = output
            output = None

        fnc_call = FunctionCall(
            call_id=call_id if call_id is not None else self.function_call.call_id,
            name=self.function_call.name,
            arguments=self.function_call.arguments,
            extra=self.function_call.extra,
        )
        return make_tool_output(fnc_call=fnc_call, output=output, exception=exception)


@dataclass
class _RunningTask:
    ctx: AsyncRunContext
    exe_task: asyncio.Task[Any]


@dataclass
class _PendingUpdate:
    ctx: AsyncRunContext
    items: list[ChatItem]


class AsyncToolset(Toolset):
    """A toolset for running long-running functions in the background.

    Tools with an :class:`AsyncRunContext` parameter are wrapped to run in the background.
    Each ``ctx.update()`` and the final ``return`` inject a tool output into the conversation;
    the agent then generates a natural-language reply to the user based on that output.

    Example::

        @function_tool
        async def book_flight(ctx: AsyncRunContext, origin: str, destination: str) -> dict:
            await ctx.update(f"Looking up flights from {origin} to {destination}...")
            # → agent says: "Sure, let me look up flights from NYC to Tokyo for you!"

            flights = await search_flights(origin, destination)
            await ctx.update(f"Found {len(flights)} flights, picking the best one...")
            # → agent says: "I found 3 flights — picking the best option for you now."

            booking = await book_best_flight(flights)
            return {"confirmation": booking.id}
            # → agent says: "All set! Your booking confirmation number is FL-847293."

        async_tools = AsyncToolset(id="booking", tools=[book_flight])
    """

    DuplicateMode = Literal["allow", "replace", "reject", "confirm"]

    def __init__(
        self,
        *,
        id: str,
        tools: list[Tool] | None = None,
        on_duplicate_call: DuplicateMode = "confirm",
    ) -> None:
        super().__init__(id=id, tools=tools)

        self._on_duplicate_call = on_duplicate_call
        self._tools = [
            self._wrap_tool(t) if isinstance(t, FunctionTool | RawFunctionTool) else t
            for t in self._tools
        ]

        self._running_tasks: dict[str, _RunningTask] = {}

        # speech delivery — shared across all tools in this toolset
        self._pending_updates: list[_PendingUpdate] = []
        self._reply_task: asyncio.Task[None] | None = None

    @function_tool
    async def get_running_tasks(self) -> list[dict]:
        """Get the list of running async tool calls."""
        return [task.ctx.function_call.model_dump() for task in self._running_tasks.values()]

    @function_tool
    async def cancel_task(self, call_id: str) -> str:
        """Cancel a running async tool call by call_id."""
        success = await self.cancel(call_id)
        if success:
            return f"Task {call_id} cancelled successfully."
        else:
            return f"Task {call_id} not found or already completed."

    async def cancel(self, call_id: str) -> bool:
        task = self._running_tasks.get(call_id)
        if task is not None:
            await utils.aio.cancel_and_wait(task.exe_task)
            return True
        return False

    async def aclose(self) -> None:
        """Cancel all tasks."""
        await super().aclose()
        tasks = [task.exe_task for task in self._running_tasks.values()]
        if self._reply_task is not None:
            tasks.append(self._reply_task)
        await utils.aio.cancel_and_wait(*tasks)
        self._running_tasks.clear()

    def _wrap_tool(self, tool: FunctionTool | RawFunctionTool) -> FunctionTool | RawFunctionTool:
        if not _has_async_context_param(tool):
            return tool

        raw_schema = _build_raw_schema(tool)

        # inject confirm_duplicate parameter for confirm mode
        confirm_duplicate_param = "_lk_agents_confirm_duplicate"
        if self._on_duplicate_call == "confirm":
            props = raw_schema["parameters"].setdefault("properties", {})
            props[confirm_duplicate_param] = {
                "type": ["boolean"],
                "description": (
                    "Set this to True to confirm you want to run a duplicate. "
                    "Only do this when user confirms the duplication is needed."
                ),
                "default": False,
            }

        @function_tool(raw_schema=raw_schema, flags=tool.info.flags)
        async def wrapper(ctx: RunContext, raw_arguments: dict[str, Any]) -> Any:
            call_id = ctx.function_call.call_id
            fnc_name = ctx.function_call.name

            # duplicate detection
            confirm_duplicate = raw_arguments.pop(confirm_duplicate_param, None)
            duplicate_result = await self._check_duplicate(fnc_name, confirm_duplicate)
            if duplicate_result is not None:
                logger.debug(
                    "duplicate tool call rejected", extra={"call_id": call_id, "function": fnc_name}
                )
                return duplicate_result

            if call_id in self._running_tasks:
                raise ValueError(f"Task already running for call_id: {call_id}")

            async_ctx = AsyncRunContext(run_ctx=ctx, toolset=self)

            async def _execute_tool() -> Any:
                try:
                    fnc_args, fnc_kwargs = prepare_function_arguments(
                        fnc=tool, json_arguments=raw_arguments, call_ctx=async_ctx
                    )
                    output = await tool(*fnc_args, **fnc_kwargs)
                except asyncio.CancelledError:
                    logger.debug(
                        "async tool cancelled", extra={"call_id": call_id, "function": fnc_name}
                    )
                    if not async_ctx._pending_fut.done():
                        async_ctx._pending_fut.set_result(None)
                    return
                except Exception as e:
                    output = e
                    logger.exception(
                        "error in async tool", extra={"call_id": call_id, "function": fnc_name}
                    )

                if not async_ctx._pending_fut.done():
                    # pending() was never called — return output directly
                    if isinstance(output, BaseException):
                        async_ctx._pending_fut.set_exception(output)
                    else:
                        async_ctx._pending_fut.set_result(output)
                    return

                if output is None:
                    return

                tool_output = async_ctx._make_tool_output(output, call_id=f"{call_id}/finished")
                if tool_output.fnc_call_out is None:
                    return

                await self._enqueue_reply(
                    async_ctx, [tool_output.fnc_call, tool_output.fnc_call_out]
                )

            exe_task = asyncio.create_task(_execute_tool(), name=f"async_tool_{fnc_name}")
            _pass_through_activity_task_info(exe_task)

            self._running_tasks[call_id] = _RunningTask(ctx=async_ctx, exe_task=exe_task)
            exe_task.add_done_callback(lambda _: self._running_tasks.pop(call_id, None))

            return await async_ctx._pending_fut

        return wrapper

    async def _enqueue_reply(self, ctx: AsyncRunContext, items: list[ChatItem]) -> None:
        """Enqueue a reply for delivery. Coalesces with pending replies."""
        agent = ctx.session.current_agent
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.insert(items)
        await agent.update_chat_ctx(chat_ctx)

        self._pending_updates.append(_PendingUpdate(ctx=ctx, items=items))

        if self._reply_task is None or self._reply_task.done():
            self._reply_task = asyncio.create_task(
                self._deliver_reply(ctx.session), name="async_toolset_deliver_reply"
            )

    async def _deliver_reply(self, session: AgentSession) -> None:
        """Wait for the agent to be idle, then generate a reply."""
        await session.wait_for_inactive()

        # no await after this line

        # snapshot and clear pending updates
        updates = self._pending_updates[:]
        self._pending_updates.clear()

        pending_items: list[ChatItem] = []
        for update in updates:
            pending_items.extend(update.items)

        if not pending_items:
            return

        # skip if the agent already spoke after our updates
        # (e.g. user asked something and got a reply)
        agent_chat_items = session.current_agent.chat_ctx.items
        if agent_chat_items and agent_chat_items[-1].created_at > pending_items[-1].created_at:
            logger.debug("skipping async toolset reply — agent already spoke after updates")
            # TODO: use a LLM to verify if another reply is needed?
            return

        pending_call_ids = [
            item.call_id for item in pending_items if item.type == "function_call_output"
        ]
        session.generate_reply(
            instructions=REPLY_INSTRUCTIONS.format(pending_call_ids=pending_call_ids),
            tool_choice="none",
        )

    async def _check_duplicate(self, fnc_name: str, confirm_duplicate: bool) -> str | None:
        """Check for duplicate running tasks. Returns a message if blocked, None otherwise."""
        if self._on_duplicate_call == "allow":
            return None

        running_fnc_calls = [
            t.ctx.function_call
            for t in self._running_tasks.values()
            if t.ctx.function_call.name == fnc_name
        ]
        if len(running_fnc_calls) == 0:
            return None

        if self._on_duplicate_call == "replace":
            await asyncio.gather(*[self.cancel(fnc_call.call_id) for fnc_call in running_fnc_calls])
            return None

        if self._on_duplicate_call == "reject":
            return DUPLICATE_REJECT.format(
                function_name=fnc_name,
                running_fnc_calls="\n".join(
                    [fnc_call.model_dump_json() for fnc_call in running_fnc_calls]
                ),
            )

        elif self._on_duplicate_call == "confirm" and not confirm_duplicate:
            return DUPLICATE_CONFIRM.format(
                function_name=fnc_name,
                running_fnc_calls="\n".join(
                    [fnc_call.model_dump_json() for fnc_call in running_fnc_calls]
                ),
            )

        return None


def _is_async_context_type(ty: type) -> bool:
    return ty is AsyncRunContext or get_origin(ty) is AsyncRunContext


def _has_async_context_param(tool: FunctionTool | RawFunctionTool) -> bool:
    """Check if the tool has an AsyncRunContext parameter."""
    try:
        type_hints = get_type_hints(tool)
    except Exception:
        return False
    return any(_is_async_context_type(h) for h in type_hints.values())


def _build_raw_schema(tool: FunctionTool | RawFunctionTool) -> dict[str, Any]:
    """Build a raw JSON schema dict from a tool, suitable for RawFunctionTool."""
    if isinstance(tool, RawFunctionTool):
        return copy.deepcopy(tool.info.raw_schema)

    model = function_arguments_to_pydantic_model(tool)
    return {
        "name": tool.info.name,
        "description": tool.info.description or "",
        "parameters": model.model_json_schema(),
    }
