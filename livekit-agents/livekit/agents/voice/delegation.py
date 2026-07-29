from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, TypeGuard

from typing_extensions import TypedDict

from ..llm.chat_context import FunctionCall, FunctionCallOutput, Instructions
from ..llm.tool_context import (
    DELEGATE_TOOL_NAME,
    FunctionTool,
    RawFunctionTool,
    Tool,
    ToolContext,
    ToolFlag,
    Toolset,
    function_tool,
)
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .events import RunContext

if TYPE_CHECKING:
    from ..llm import LLM
    from .agent_activity import AgentActivity


TOOL_DESCRIPTION = """Hand a request to the expert that handles reasoning, lookups and actions.

Default to using this. Delegate anything that is not small talk, not already answered earlier
in this conversation, and not covered by one of your other tools. Never guess, never answer
from memory, and never tell the user something is impossible before asking.

Delegate: account, order, billing and status questions; anything needing a lookup, a
calculation or a change; anything with a rule or policy behind it; anything you are unsure of.
Do not delegate: greetings, chit-chat, acknowledgements, or repeating what was already said.

State the request in full — the expert sees the conversation but not your intent."""

DELEGATION_DIRECTIVE = """You are answering on behalf of an agent that is talking to the user.
Return the facts it needs, not a phrased reply — it does the talking."""

# the conversation model answers this to acknowledge, and it stays in the context afterwards — hence the
# raw template: UPDATE_TEMPLATE's "still running, DON'T give information not included above"
# would read as current forever and argue against relaying the answer
DISPATCHED = (
    'Working on it. Acknowledge in a few natural words — "one moment", "sure, let me check", '
    '"hang on", "okay, looking now" — varying the wording, restating none of the request and '
    "promising nothing about the outcome."
)
DISPATCHED_TEMPLATE = "{message}"

EXHAUSTED = "The expert could not complete the request."


class DelegationOptions(TypedDict, total=False):
    """Configuration for the delegate tool and the delegation LLM."""

    instructions: NotGivenOr[str | Instructions]
    """The delegation LLM's prompt. Defaults to the agent's instructions."""
    tool_description: str
    """What the conversation model reads on the ``delegate`` tool."""
    announce: bool
    """Whether the conversation model acknowledges when it delegates, at the cost of one model turn.

    Defaults to True. On a model that answers tool outputs itself, False makes the call block
    for the whole delegation: releasing it means pushing the dispatch note, and that model
    speaks whatever is pushed."""


_DEFAULTS: DelegationOptions = {
    "instructions": NOT_GIVEN,
    "tool_description": TOOL_DESCRIPTION,
    "announce": True,
}


def _resolve_delegation_options(config: DelegationOptions | None = None) -> DelegationOptions:
    """Fill in the defaults for any key the caller left out."""
    return DelegationOptions(**{**_DEFAULTS, **(config or {})})


def _has_info(tool: Tool) -> TypeGuard[FunctionTool | RawFunctionTool]:
    """Whether we execute this tool, as opposed to the provider running it."""
    return isinstance(tool, (FunctionTool, RawFunctionTool))


def _no_delegate(tool: Tool) -> bool:
    return _has_info(tool) and ToolFlag.NO_DELEGATE in tool.info.flags


class _Delegation:
    """Delegation state for one activity: resolved config, the delegate tool, the tool split."""

    def __init__(self, activity: AgentActivity) -> None:
        self._activity = activity
        self._tool: FunctionTool | None = None

    @property
    def llm(self) -> LLM | None:
        agent_llm = self._activity._agent.delegation_llm
        return agent_llm if is_given(agent_llm) else self._activity._session.delegation_llm

    @property
    def enabled(self) -> bool:
        return self.llm is not None

    @property
    def options(self) -> DelegationOptions:
        """Agent options over session options over the defaults, key by key."""
        merged: DelegationOptions = {}
        merged.update(self._activity._session.options.delegation_options)
        if is_given(agent_opts := self._activity._agent._delegation_options):
            merged.update(agent_opts)
        return _resolve_delegation_options(merged)

    @property
    def tool(self) -> FunctionTool:
        """The one tool the conversation model gets. Built once so its identity is stable."""
        if self._tool is None:
            self._tool = _build_tool(self.options)
        return self._tool

    def _user_tools(self) -> list[Tool | Toolset]:
        activity = self._activity
        return [*activity._session.tools, *activity._agent.tools, *activity._mcp_tools]

    def split(self) -> tuple[list[Tool], list[Tool]]:
        """Split the user's tools into (conversation model's, delegation's), each complete.

        Everything delegates except the delegate tool, ``NO_DELEGATE`` tools, and provider
        tools. A cancellable tool cannot be ``NO_DELEGATE``, so only the delegation can hold
        one — which is why the management tools only ever go to its side.
        """
        from .tool_executor import cancel_task, get_running_tasks, has_cancellable_tool

        kept: list[Tool] = [self.tool]
        delegated: list[Tool] = []
        for tool in ToolContext(self._user_tools()).flatten():
            if not _has_info(tool) or _no_delegate(tool):
                kept.append(tool)
            else:
                delegated.append(tool)

        if has_cancellable_tool(delegated):
            delegated += [cancel_task, get_running_tasks]
        return kept, delegated

    def restrict(self, tool_ctx: ToolContext) -> ToolContext:
        """A view of ``tool_ctx`` holding only what the conversation model is offered."""
        offered = {t.info.name for t in self._activity.model_tools if _has_info(t)}
        restricted = tool_ctx.copy()
        restricted._exclude(
            [t for t in tool_ctx.flatten() if _has_info(t) and t.info.name not in offered]
        )
        return restricted


def _build_tool(options: DelegationOptions) -> FunctionTool:
    """The delegate tool. Not ``CANCELLABLE``: the delegation LLM manages its own work."""
    announce = options["announce"]

    async def delegate(ctx: RunContext, task: str) -> Any:
        from .agent import ModelSettings
        from .generation import update_instructions

        activity = ctx.session.current_agent._get_activity_or_raise()
        agent = activity.agent
        opts = activity.delegation_options

        # answering the dispatch note is the acknowledgement; a model that speaks whatever is
        # pushed can only honor `announce=False` by holding the turn instead of releasing
        rt_session = activity.realtime_llm_session
        released = (
            announce or rt_session is None or not rt_session.capabilities.auto_tool_reply_generation
        )
        if released:
            await ctx.update(DISPATCHED, template=DISPATCHED_TEMPLATE, silent=not announce)

        # tool traffic only for the delegation's own tools: the delegate pairs are the voice
        # model's, and a tool this LLM cannot call would read as one it can
        chat_ctx = agent.chat_ctx.copy(tools=activity.delegated_tools)

        opt_instructions = opts["instructions"]
        instructions = opt_instructions if is_given(opt_instructions) else agent.instructions
        # the delegation answers for the turn that delegated, so it renders for that modality
        modality = ctx.speech_handle.input_details.modality
        text = (
            instructions.render(modality=modality)
            if isinstance(instructions, Instructions)
            else instructions
        )
        try:
            update_instructions(
                chat_ctx,
                instructions=f"{text}\n{DELEGATION_DIRECTIVE}",
                add_if_missing=True,
                modality=modality,
            )
        except ValueError:
            logger.exception("failed to set the instructions of the delegation")

        chat_ctx.add_message(role="user", content=task)

        # the deferred reply is what finishes the delegation handle; an error still gets one,
        # but a call that was never released, or answered with None, does not
        reply_follows = released
        try:
            answer = agent.delegation_node(
                task, ctx, chat_ctx, activity.delegated_tools, ModelSettings()
            )
            if asyncio.iscoroutine(answer):
                answer = await answer
            reply_follows = released and answer is not None
            return answer
        finally:
            if not reply_follows and (handle := ctx._delegation_speech_handle) is not None:
                handle._mark_done()

    return function_tool(delegate, name=DELEGATE_TOOL_NAME, description=options["tool_description"])


async def execute_delegated_tools(
    activity: AgentActivity,
    function_stream: AsyncIterable[FunctionCall],
    *,
    tool_ctx: ToolContext,
    ctx: RunContext,
) -> list[FunctionCallOutput]:
    """Run the delegation LLM's tool calls on the session's delegation executor.

    Consumes ``function_stream`` as the calls arrive, so a tool starts while the LLM is still
    generating. They are the delegation's own work, so they land in neither conversation store
    and surface as ``function_tools_executed`` instead. ``ctx`` is the delegate call's
    context, which their updates are re-attributed to.
    """
    from .events import FunctionToolsExecutedEvent
    from .generation import perform_tool_executions
    from .speech_handle import SpeechHandle

    session = activity.session

    if ctx._delegation_speech_handle is None:
        # one per delegation, shared by every tool it calls
        ctx._delegation_speech_handle = SpeechHandle.create(allow_interruptions=True)

    exe_task, tool_output = perform_tool_executions(
        session=session,
        speech_handle=ctx._delegation_speech_handle,
        tool_ctx=tool_ctx,
        tool_choice=NOT_GIVEN,
        function_stream=function_stream,
        tool_execution_started_cb=lambda _: None,
        tool_execution_completed_cb=lambda _: None,
        default_executor=session._delegation_executor(),
        delegation_parent=ctx,
    )
    await exe_task

    fnc_executed_ev = FunctionToolsExecutedEvent(function_calls=[], function_call_outputs=[])
    outputs: list[FunctionCallOutput] = []
    for out in tool_output.output:
        fnc_executed_ev.function_calls.append(out.fnc_call)
        fnc_executed_ev.function_call_outputs.append(out.fnc_call_out)
        if out.fnc_call_out is not None:
            outputs.append(out.fnc_call_out)
        if out.agent_task is not None:
            # a handoff has to be sequenced against an answer nobody has spoken yet
            logger.warning(
                "a delegated tool returned an Agent; hand off with session.update_agent() "
                "inside ctx.foreground() from a delegation_node override instead",
                extra={"function": out.fnc_call.name},
            )
    session.emit("function_tools_executed", fnc_executed_ev)
    return outputs
