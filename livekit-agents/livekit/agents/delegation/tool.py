"""The one tool the conversation model gains when a delegate is in force."""

from __future__ import annotations

from ..a2a import TaskInput, TaskUpdate
from ..llm.tool_context import FunctionTool, ToolError, function_tool

# imported at runtime: the tool's signature is resolved with get_type_hints() when a call
# arrives, so RunContext has to be a real name by then
from ..voice.events import RunContext
from .a2a import A2ADelegate
from .delegate import DELEGATE_TOOL_NAME

TOOL_DESCRIPTION = """Hand a request to the expert that handles reasoning, lookups and actions.

Default to using this. Delegate anything that is not small talk, not already answered earlier
in this conversation, and not covered by one of your other tools. Never guess, never answer
from memory, and never tell the user something is impossible before asking.

Delegate: account, order, billing and status questions; anything needing a lookup, a
calculation or a change; anything with a rule or policy behind it; anything you are unsure of.
Do not delegate: greetings, chit-chat, acknowledgements, or repeating what was already said.

State the request in full — the expert sees the conversation but not your intent.
The expert is never mentioned to the user: no consulting anyone, no handing anything over, no
passing it on — as far as they are concerned, this is you doing the work."""

# with `announce` off nothing else acknowledges, so the line comes free in the same completion
ACK_DIRECTIVE = """
In the same turn as the call, say one short line so the user is not left in silence —
"one sec", "on it", "okay, looking now", etc. — varying the wording. Do not restate the
request and do not promise an outcome."""

TOOL_DESCRIPTION_WITH_ACK = TOOL_DESCRIPTION + "\n" + ACK_DIRECTIVE

# no reply is generated from this one: the model's own line alongside the call acknowledges
DISPATCHED_SILENT = "Started. The answer is a separate entry, not this one."

# with `announce` on, the model answers this instead of writing its own line
DISPATCHED = (
    'Acknowledge in a few natural words — "one moment", "sure, let me check", "okay, looking '
    'now" — varying the wording, restating none of the request and promising nothing about '
    "the outcome. The answer is a separate entry, not this one."
)


def build_delegate_tool(description: str | None = None, *, announce: bool = True) -> FunctionTool:
    """Build the tool that reaches whichever delegate is in force."""

    async def delegate(ctx: RunContext, task: str) -> str | None:
        session = ctx.session
        activity = session.current_agent._get_activity_or_raise()
        handler = activity._delegation["delegate"]
        if handler is None:
            raise RuntimeError("the delegate tool ran with no delegate configured")

        # releases the turn so the conversation model keeps talking while the expert works. a
        # model that speaks whatever is pushed to it cannot be released quietly, so with
        # `announce` off it holds the turn rather than acknowledging against instruction
        rt_session = activity.realtime_llm_session
        if announce or rt_session is None or not rt_session.capabilities.auto_tool_reply_generation:
            await ctx.update(DISPATCHED if announce else DISPATCHED_SILENT, silent=not announce)

        task_input = TaskInput(
            instruction=task,
            chat_ctx=session.current_agent.chat_ctx.copy(
                exclude_function_call=True,
                exclude_handoff=True,
                exclude_config_update=True,
                exclude_instructions=True,
            ),
            metadata=dict(activity._delegation["metadata"]),
        )

        call_id = ctx.function_call.call_id
        linked = False
        ended = "failed"
        if (state := session.state) is not None:
            # the expert joins this conversation's database, under this session
            task_input.conversation_id = state.conversation.database_id
            task_input.caller_session_id = state.session_id
            # a delegate an agent brings after a handoff is pointed back here, before it sends
            await state.resume_delegate(handler)

        # the terminal update leaves the delegation running, holding a session there or an
        # open HTTP stream here, until the stream is closed
        async with handler.submit(task_input) as stream:
            try:
                while True:
                    try:
                        update: TaskUpdate = await anext(stream)
                    except StopAsyncIteration:
                        # the stream ended without declaring a state, which is how a
                        # delegation that died mid-flight reaches the caller
                        raise ToolError("the delegation ended without an answer") from None
                    if (
                        state is not None
                        and not linked
                        and isinstance(handler, A2ADelegate)
                        and handler.context_id is not None
                        and stream.task_id
                    ):
                        # which expert task answered which call, for a dashboard to join
                        linked = True
                        state.delegation_started(
                            call_id,
                            endpoint=handler.endpoint,
                            child_session_id=handler.context_id,
                            task_id=stream.task_id,
                        )
                    if update.state == "working":
                        if not update.text:
                            continue
                        if update.verbatim:
                            # said as written, once, rather than handed to the model to phrase
                            session.say(update.text)
                        else:
                            await ctx.update(update.text)
                        continue
                    ended = update.state
                    if update.state == "failed":
                        raise ToolError(update.text or "the delegation failed")
                    if update.directive is not None:
                        from ..voice.events import DirectiveReceivedEvent

                        session.emit(
                            "directive_received",
                            DirectiveReceivedEvent(
                                kind=update.directive.kind,
                                reason=update.directive.reason,
                                call_id=ctx.function_call.call_id,
                            ),
                        )
                    if update.verbatim:
                        # said as written, then kept to the model: no return, so no reply
                        # repeats it
                        session.say(update.text)
                        await ctx.update(update.text, silent=True)
                        return None
                    # completed, canceled and input-required all answer: a cancelled
                    # delegation still says what happened, side effects included, and a
                    # question is what the conversation relays to the user
                    return update.text
            finally:
                if linked and state is not None:
                    state.delegation_ended(call_id, status=ended)

    # not CANCELLABLE, since the expert owns its work. duplicates are allowed because the
    # check keys on the function name, which would make every delegation a duplicate of every
    # other one; the in-flight placeholder for the pending call is what stops the model
    # re-delegating
    return function_tool(
        delegate,
        name=DELEGATE_TOOL_NAME,
        # exactly one of the two acknowledges: the dispatch note when `announce` is on, the
        # model's own line in the same completion when it is off
        description=description or (TOOL_DESCRIPTION if announce else TOOL_DESCRIPTION_WITH_ACK),
    )
