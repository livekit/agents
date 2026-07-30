"""Delegation: which brain is offered which tools, and how the delegate call behaves.

The fast/slow split is one builtin tool, so most of what needs covering is the tool routing
and that the delegate call releases the turn without speaking.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    FunctionCall,
    RunContext,
    function_tool,
    utils,
)
from livekit.agents.llm import DELEGATE_TOOL_NAME, ToolContext, ToolFlag, Toolset
from livekit.agents.voice import SpeechHandle
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.delegation import (
    DISPATCHED,
    _build_tool,
    _resolve_delegation_options,
)
from livekit.agents.voice.events import RunContext as _RunContext
from livekit.agents.voice.tool_executor import _RunningTasks, _ToolExecutor

pytestmark = pytest.mark.unit


@function_tool
async def lookup_order(ctx: RunContext, order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id} shipped Tuesday"


@function_tool(flags=ToolFlag.NO_DELEGATE)
async def send_dtmf(ctx: RunContext, digits: str) -> None:
    """Send DTMF digits."""


@function_tool(flags=ToolFlag.CANCELLABLE)
async def book_flight(ctx: RunContext, date: str) -> str:
    """Book a flight."""
    return f"booked for {date}"


def _names(tools: list[Any]) -> set[str]:
    return {t.info.name for t in tools if hasattr(t, "info")}


def _activity(*, tools: list[Any], delegating: bool = True, **session_kwargs: Any) -> AgentActivity:
    session = AgentSession(delegation_llm=MagicMock() if delegating else None, **session_kwargs)
    return AgentActivity(Agent(instructions="test", tools=tools), session)


@pytest.fixture
def _clear_running_tasks():
    yield
    _RunningTasks.clear()


async def _run_delegate(
    *,
    node: Any,
    history: ChatContext | None = None,
    instructions: Any = "test",
    delegated_tools: list[Any] | None = None,
    options: Any = None,
    modality: str = "audio",
    speaks_tool_outputs: bool = False,
) -> tuple[Any, _RunContext]:
    """Drive the delegate tool as the executor does; returns what it released, and its ctx."""
    from livekit.agents.llm import RealtimeCapabilities
    from livekit.agents.voice.delegation import _build_tool

    resolved = _resolve_delegation_options(options)
    session = _reply_session()
    agent = session.current_agent
    agent.chat_ctx = history if history is not None else ChatContext.empty()
    agent.instructions = instructions
    agent.delegation_node = node

    activity = MagicMock()
    activity.session = session
    activity.agent = agent
    activity.delegated_tools = delegated_tools or []
    activity.delegation_options = resolved
    activity.realtime_llm_session = (
        MagicMock(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=True,
                audio_output=True,
                manual_function_calls=True,
            )
        )
        if speaks_tool_outputs
        else None
    )
    agent._get_activity_or_raise = lambda: activity
    session._activity = activity
    activity._tool_executor = _ToolExecutor()

    ctx = _run_ctx(session, call_id="p1", name=DELEGATE_TOOL_NAME, modality=modality)
    released = await _ToolExecutor().execute(
        tool=_build_tool(resolved), run_ctx=ctx, raw_arguments={"task": "look it up"}
    )
    return released, ctx


class TestRouting:
    """Everything delegates except the delegate tool itself and NO_DELEGATE tools."""

    def test_the_voice_model_only_keeps_delegate_and_no_delegate_tools(self):
        activity = _activity(tools=[lookup_order, send_dtmf])

        assert _names(activity.model_tools) == {DELEGATE_TOOL_NAME, "send_dtmf"}
        assert _names(activity.delegated_tools) == {"lookup_order"}

    def test_the_executor_still_resolves_withheld_tools(self):
        # what lets a model that emits delegations natively call a tool it never saw
        activity = _activity(tools=[lookup_order])
        assert "lookup_order" in _names(ToolContext(activity.tools).flatten())

    def test_toolset_members_delegate(self):
        activity = _activity(tools=[Toolset(id="billing", tools=[lookup_order]), send_dtmf])

        assert _names(activity.model_tools) == {DELEGATE_TOOL_NAME, "send_dtmf"}
        assert _names(activity.delegated_tools) == {"lookup_order"}

    def test_without_a_delegation_llm_nothing_is_withheld(self):
        activity = _activity(tools=[lookup_order, send_dtmf], delegating=False)

        assert _names(activity.model_tools) == {"lookup_order", "send_dtmf"}
        assert activity.delegated_tools == []

    def test_the_delegate_tool_identity_is_stable(self):
        activity = _activity(tools=[])
        assert activity._delegation.tool is activity._delegation.tool


class TestManagementTools:
    """cancel_task / get_running_tasks only ever reach the delegation."""

    def test_a_cancellable_tool_cannot_be_kept_on_the_voice_model(self):
        # a cancellable tool is long-running, which is what delegation moves off the voice
        # model; keeping the combination out means level 1 never needs the management tools
        with pytest.raises(ValueError, match="CANCELLABLE and NO_DELEGATE"):

            @function_tool(flags=ToolFlag.CANCELLABLE | ToolFlag.NO_DELEGATE)
            async def dial(ctx: RunContext) -> None:
                """Dial out."""

    def test_they_go_to_the_delegation_only(self):
        activity = _activity(tools=[book_flight, send_dtmf])

        assert _names(activity.delegated_tools) == {
            "book_flight",
            "lk_agents_cancel_task",
            "lk_agents_get_running_tasks",
        }
        assert _names(activity.model_tools) == {DELEGATE_TOOL_NAME, "send_dtmf"}

    def test_absent_when_nothing_is_cancellable(self):
        activity = _activity(tools=[lookup_order])
        assert _names(activity.delegated_tools) == {"lookup_order"}


class TestOptions:
    def test_defaults_are_filled_in(self):
        opts = _resolve_delegation_options()
        assert opts["announce"] is True
        assert opts["tool_description"]

    def test_instructions_default_to_not_given(self):
        from livekit.agents.utils import is_given

        assert not is_given(_resolve_delegation_options()["instructions"])

    def test_agent_options_win_key_by_key(self):
        session = AgentSession(
            delegation_llm=MagicMock(),
            delegation_options={"announce": False, "tool_description": "session"},
        )
        agent = Agent(instructions="test", delegation_options={"tool_description": "agent"})
        opts = AgentActivity(agent, session).delegation_options

        assert opts["tool_description"] == "agent"  # the agent's key wins
        assert opts["announce"] is False  # the session's other key survives

    def test_agent_delegation_llm_wins(self):
        session_llm, agent_llm = MagicMock(), MagicMock()
        session = AgentSession(delegation_llm=session_llm)
        agent = Agent(instructions="test", delegation_llm=agent_llm)

        assert AgentActivity(agent, session).delegation_llm is agent_llm

    def test_the_tool_name_is_the_constant(self):
        assert _build_tool(_resolve_delegation_options()).info.name == DELEGATE_TOOL_NAME

    def test_the_delegate_tool_is_not_cancellable(self):
        # a delegation is the delegation LLM's to manage; the conversation model never cancels it
        assert ToolFlag.CANCELLABLE not in _build_tool(_resolve_delegation_options()).info.flags


def _reply_session() -> Any:
    session = MagicMock()
    agent = MagicMock()
    agent.chat_ctx = ChatContext.empty()
    agent.update_chat_ctx = AsyncMock()
    session.current_agent = agent
    session._global_run_state = None
    activity = MagicMock()
    activity.agent = agent
    session.wait_for_idle = AsyncMock(return_value=activity)

    # a real-enough reply speech: _deliver_reply builds a ToolReplyUpdated from its id
    reply = MagicMock(spec=SpeechHandle)
    reply.id = "speech_reply"
    reply.interrupted = False
    reply.chat_items = ["answer"]
    reply_callbacks: list[Any] = []
    reply.add_done_callback.side_effect = reply_callbacks.append
    reply.fire_done = lambda: [cb(reply) for cb in list(reply_callbacks)]
    session.generate_reply = MagicMock(return_value=reply)
    session.reply_speech = reply
    return session


def _one_call(call_id: str, name: str) -> Any:
    """A closed one-item stream of FunctionCall, as the node feeds the executor."""
    ch: Any = utils.aio.Chan[FunctionCall]()
    ch.send_nowait(FunctionCall(call_id=call_id, name=name, arguments="{}"))
    ch.close()
    return ch


def _run_ctx(session: Any, *, call_id: str, name: str, modality: str = "audio") -> _RunContext:
    from livekit.agents.voice.speech_handle import InputDetails

    speech_handle = MagicMock()
    speech_handle.num_steps = 1
    speech_handle.allow_interruptions = True
    speech_handle.input_details = InputDetails(modality=modality)  # type: ignore[arg-type]
    return _RunContext(
        session=session,
        speech_handle=speech_handle,
        function_call=FunctionCall(call_id=call_id, name=name, arguments="{}"),
    )


class TestDelegatedToolUpdates:
    """A delegated tool's update is re-attributed to the delegate call."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_forwarded_to_the_parent_without_releasing(self):
        ran_to_completion: list[str] = []

        @function_tool
        async def delegated(ctx: RunContext) -> str:
            """A tool the delegation LLM called."""
            await ctx.update("found 3 flights")
            ran_to_completion.append("after update")
            return "picked the cheapest"

        session = _reply_session()
        parent = _run_ctx(session, call_id="p1", name=DELEGATE_TOOL_NAME)
        parent.update = AsyncMock()  # type: ignore[method-assign]

        child = _run_ctx(session, call_id="c1", name="delegated")
        child._delegation_parent = parent

        result = await _ToolExecutor().execute(tool=delegated, run_ctx=child, raw_arguments={})

        parent.update.assert_awaited_once_with("found 3 flights")
        # the delegation LLM has nobody to talk to while it waits, so the tool runs through
        assert result == "picked the cheapest"
        assert ran_to_completion == ["after update"]


class TestDispatchedMessage:
    """It outlives the delegation, so it must not claim the work is still pending."""

    def test_it_asks_for_a_varied_content_free_acknowledgement(self):
        # restating the request would promise a capability the delegation may reject, and a
        # fixed phrase would repeat all call long
        lowered = DISPATCHED.lower()
        assert "acknowledge" in lowered
        assert "promising nothing" in lowered
        assert "varying the wording" in lowered
        # several examples, so the model samples a range instead of anchoring on one
        assert lowered.count('"') >= 8

    def test_it_never_claims_the_work_is_still_pending(self):
        lowered = DISPATCHED.lower()
        for stale in ("still running", "will arrive", "will follow", "wait for"):
            assert stale not in lowered, f"{stale!r} would read as current forever"

    def test_it_bypasses_the_async_tool_template(self):
        from livekit.agents.voice.delegation import DISPATCHED_TEMPLATE
        from livekit.agents.voice.tool_executor import UPDATE_TEMPLATE, _render

        rendered = _render(
            DISPATCHED_TEMPLATE,
            {"function_name": DELEGATE_TOOL_NAME, "call_id": "c1", "message": DISPATCHED},
        )
        assert rendered == DISPATCHED
        # the shared template asserts an ongoing state and argues against relaying anything
        # that arrived elsewhere — both wrong once the answer is in the context
        assert "still running" in UPDATE_TEMPLATE
        assert "still running" not in rendered


class TestDelegatedToolSpeechHandle:
    """A delegated tool gets its own handle, not the turn that dispatched the delegation."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_it_is_not_the_delegate_calls_turn(self):
        from livekit.agents.voice import delegation

        seen: list[Any] = []

        @function_tool
        async def delegated(ctx: RunContext) -> str:
            """A tool the delegation LLM called."""
            seen.append(ctx.speech_handle)
            return "done"

        session = _reply_session()
        session._delegation_executor = lambda: _ToolExecutor(owning_activity=None)
        parent = _run_ctx(session, call_id="p1", name=DELEGATE_TOOL_NAME)

        activity = MagicMock()
        activity.session = session
        session._activity = activity
        activity._tool_executor = _ToolExecutor()

        await delegation.execute_delegated_tools(
            activity,
            _one_call("c1", "delegated"),
            tool_ctx=ToolContext([delegated]),
            ctx=parent,
        )

        assert seen and seen[0] is not parent.speech_handle
        # interruptible, so lk_agents_cancel_task can still cancel a delegated call
        assert seen[0].allow_interruptions is True
        # not done yet: the answer has not been spoken, so anything sequenced off it waits
        assert seen[0].done() is False


class TestDelegationHandleCompletion:
    """The delegated handle stands for the turn that speaks the answer."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_it_completes_when_the_answer_speech_is_done(self):
        from livekit.agents.voice import SpeechHandle
        from livekit.agents.voice.tool_executor import _finish_delegations, _PendingUpdate

        parent = _run_ctx(_reply_session(), call_id="p1", name=DELEGATE_TOOL_NAME)
        handle = SpeechHandle.create()
        parent._delegation_speech_handle = handle

        fired: list[str] = []
        handle.add_done_callback(lambda _: fired.append("end_call would run here"))

        assert not handle.done() and not fired

        # what _deliver_reply does once the reply speaking the answer is finished
        _finish_delegations([_PendingUpdate(ctx=parent, items=[], target=MagicMock())])
        await asyncio.sleep(0)

        assert handle.done()
        assert fired == ["end_call would run here"]

    @pytest.mark.asyncio
    async def test_wait_for_playout_routes_to_the_dispatching_turn(self):
        # "the speech prior to running this tool" is the line the conversation model said when it
        # delegated, not the answer this tool's handle stands for
        session = _reply_session()
        parent = _run_ctx(session, call_id="p2", name=DELEGATE_TOOL_NAME)
        parent.wait_for_playout = AsyncMock()  # type: ignore[method-assign]

        child = _run_ctx(session, call_id="c2", name="delegated")
        child._delegation_parent = parent

        await child.wait_for_playout()
        parent.wait_for_playout.assert_awaited_once()


class TestSpeechHandleTargetsTheAnswer:
    """A delegated tool sequenced off ctx.speech_handle fires after the answer, not before."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_the_delegate_reply_speech_completes_it(self):
        # the shape end_call relies on: add_done_callback on ctx.speech_handle, which must
        # not run until the answer has been played out
        fired: list[str] = []

        @function_tool
        async def ends_the_call(ctx: RunContext) -> str:
            """Stands in for end_call."""
            ctx.speech_handle.add_done_callback(lambda _: fired.append("shutdown"))
            return "say goodbye"

        @function_tool
        async def delegate_like(ctx: RunContext) -> str:
            """Stands in for the delegate tool."""
            await ctx.update(DISPATCHED)
            await delegation.execute_delegated_tools(
                activity,
                _one_call("c1", "ends_the_call"),
                tool_ctx=ToolContext([ends_the_call]),
                ctx=ctx,
            )
            return "the call is wrapping up"

        from livekit.agents.voice import delegation

        session = _reply_session()
        session._delegation_executor = lambda: _ToolExecutor(owning_activity=None)
        activity = MagicMock()
        activity.session = session
        activity.llm = MagicMock()  # not a RealtimeModel, so a reply is generated
        session._activity = activity
        activity._tool_executor = _ToolExecutor()

        executor = _ToolExecutor()
        ctx = _run_ctx(session, call_id="p1", name=DELEGATE_TOOL_NAME)
        await executor.execute(tool=delegate_like, run_ctx=ctx, raw_arguments={})

        while executor.has_running_tasks:
            await asyncio.sleep(0)
        assert executor._reply_task is not None
        await executor._reply_task

        # flush any pending callbacks: the assertion below has to mean "not completed",
        # not merely "not dispatched yet"
        for _ in range(5):
            await asyncio.sleep(0)

        # the reply carrying the answer was scheduled but has not played yet
        assert fired == []

        session.reply_speech.fire_done()
        await asyncio.sleep(0)
        assert fired == ["shutdown"]


class TestCircularWaitGuard:
    """The existing guard against awaiting your own handle covers delegated tools too."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_awaiting_the_delegation_handle_from_inside_raises(self):
        from livekit.agents.voice import delegation

        caught: list[BaseException] = []

        @function_tool
        async def waits_on_itself(ctx: RunContext) -> str:
            """A delegated tool that awaits the handle it is running under."""
            try:
                await ctx.speech_handle.wait_for_playout()
            except RuntimeError as e:
                caught.append(e)
            return "done"

        session = _reply_session()
        session._delegation_executor = lambda: _ToolExecutor(owning_activity=None)
        activity = MagicMock()
        activity.session = session
        session._activity = activity
        activity._tool_executor = _ToolExecutor()

        parent = _run_ctx(session, call_id="p1", name=DELEGATE_TOOL_NAME)
        await delegation.execute_delegated_tools(
            activity,
            _one_call("c1", "waits_on_itself"),
            tool_ctx=ToolContext([waits_on_itself]),
            ctx=parent,
        )

        assert caught and "circular wait" in str(caught[0])


class TestDelegationChatCtx:
    """The delegation LLM only sees tool traffic for tools it has."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_foreign_tool_items_are_stripped(self):
        from livekit.agents.llm import FunctionCallOutput

        seen: list[ChatContext] = []

        async def _node(task, ctx, chat_ctx, tools, model_settings):
            seen.append(chat_ctx)
            return "answer"

        def _pair(call_id: str, name: str, output: str) -> list[Any]:
            return [
                FunctionCall(call_id=call_id, name=name, arguments="{}"),
                FunctionCallOutput(call_id=call_id, name=name, output=output, is_error=False),
            ]

        history = ChatContext.empty()
        history.add_message(role="user", content="where's my order")
        history.insert(_pair("d1", DELEGATE_TOOL_NAME, "shipped Tuesday"))  # conversation model's
        history.insert(_pair("n1", "send_dtmf", "sent"))  # NO_DELEGATE, conversation model's
        history.insert(_pair("o1", "lookup_order", "shipped"))  # the delegation's own
        history.add_message(role="assistant", content="it shipped Tuesday")

        _, _ctx = await _run_delegate(node=_node, history=history, delegated_tools=[lookup_order])

        items = seen[0].items
        names = {i.name for i in items if i.type in ("function_call", "function_call_output")}
        assert names == {"lookup_order"}
        # the spoken turns survive, and the delegated request is appended last
        assert [i.role for i in items if i.type == "message"] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert items[-1].text_content == "look it up"


class TestToolsOverlapGeneration:
    """A delegated tool starts while the delegation LLM is still generating."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_execution_starts_before_the_stream_ends(self):
        from livekit.agents.voice import delegation

        started = asyncio.Event()
        stream_finished = False

        @function_tool
        async def slow_lookup(ctx: RunContext) -> str:
            """A delegated tool."""
            started.set()
            return "found it"

        async def _calls() -> Any:
            nonlocal stream_finished
            yield FunctionCall(call_id="c1", name="slow_lookup", arguments="{}")
            # the LLM is still generating here; the tool must already be running
            await asyncio.wait_for(started.wait(), timeout=1)
            stream_finished = True

        session = _reply_session()
        session._delegation_executor = lambda: _ToolExecutor(owning_activity=None)
        activity = MagicMock()
        activity.session = session
        session._activity = activity
        activity._tool_executor = _ToolExecutor()

        parent = _run_ctx(session, call_id="p1", name=DELEGATE_TOOL_NAME)
        outputs = await delegation.execute_delegated_tools(
            activity, _calls(), tool_ctx=ToolContext([slow_lookup]), ctx=parent
        )

        assert stream_finished  # the tool ran before the stream was exhausted
        assert [o.output for o in outputs] == ["found it"]


class TestNodeOverride:
    """The node returns the answer; an override needs no streaming ceremony."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_a_plain_string_answer(self):
        async def _node(task, ctx, chat_ctx, tools, model_settings):
            return f"answered: {task}"

        released, _ = await _run_delegate(node=_node)
        assert released == DISPATCHED  # the answer follows as its own reply

    @pytest.mark.asyncio
    async def test_none_answers_the_call_without_speaking(self):
        async def _node(task, ctx, chat_ctx, tools, model_settings):
            return None

        released, ctx = await _run_delegate(node=_node, options={"announce": False})

        assert released == DISPATCHED
        # no deferred reply is scheduled, and the handle is finished so nothing waits on one
        assert ctx._executor is None or ctx._executor._reply_task is None
        assert ctx._delegation_speech_handle is None or ctx._delegation_speech_handle.done()


class TestNonTextAnswers:
    """The answer is serialized like any tool output, so a dict is fine."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_a_dict_answer_reaches_the_voice_model(self):
        answer = {"order_id": "1234", "status": "shipped"}

        @function_tool
        async def delegate_like(ctx: RunContext) -> Any:
            """Stands in for the delegate tool."""
            await ctx.update(DISPATCHED)
            return answer

        session = _reply_session()
        ctx = _run_ctx(session, call_id="p1", name=DELEGATE_TOOL_NAME)
        executor = _ToolExecutor()

        await executor.execute(tool=delegate_like, run_ctx=ctx, raw_arguments={})
        while executor.has_running_tasks:
            await asyncio.sleep(0)
        assert executor._reply_task is not None
        await executor._reply_task

        # the framework stringified it into the tool output the conversation model reads
        inserted = session.current_agent.update_chat_ctx.await_args[0][0]
        outputs = [i.output for i in inserted.items if i.type == "function_call_output"]
        assert str(answer) in outputs


class TestInstructionsModality:
    """The delegation renders instructions for the modality of the turn that delegated."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("modality", ["audio", "text"])
    async def test_it_follows_the_source_turn(self, modality: str):
        from livekit.agents.llm.chat_context import Instructions

        seen: list[ChatContext] = []

        async def _node(task, ctx, chat_ctx, tools, model_settings):
            seen.append(chat_ctx)
            return "answer"

        await _run_delegate(
            node=_node,
            instructions=Instructions(
                "You are a support agent.", audio="Keep it short.", text="Use markdown."
            ),
            modality=modality,
        )

        system = seen[0].items[0].text_content or ""
        assert ("Keep it short." in system) is (modality == "audio")
        assert ("Use markdown." in system) is (modality == "text")


class TestBothOutputsCoexist:
    """The dispatch note and the answer both stay in the conversation model's context."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_the_answer_lands_alongside_the_dispatch_note(self):
        @function_tool
        async def delegate_like(ctx: RunContext) -> str:
            """Stands in for the delegate tool."""
            await ctx.update(DISPATCHED)
            return "the store does not handle weather"

        session = _reply_session()
        ctx = _run_ctx(session, call_id="p1", name=DELEGATE_TOOL_NAME)
        executor = _ToolExecutor()

        await executor.execute(tool=delegate_like, run_ctx=ctx, raw_arguments={})
        while executor.has_running_tasks:
            await asyncio.sleep(0)
        assert executor._reply_task is not None
        await executor._reply_task

        inserted = session.current_agent.update_chat_ctx.await_args[0][0]
        outputs = [i.output for i in inserted.items if i.type == "function_call_output"]
        # the answer is there; the dispatch note is too, which is why it may not instruct
        assert "the store does not handle weather" in outputs


class TestAnnounceIsTheOneAckSwitch:
    """`announce` decides whether the dispatch is answered — the only ack mechanism."""

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    async def _dispatch(self, *, announce: bool, speaks_tool_outputs: bool = False) -> Any:
        async def _node(task, ctx, chat_ctx, tools, model_settings):
            return "shipped Tuesday"

        return await _run_delegate(
            node=_node,
            options={"announce": announce},
            speaks_tool_outputs=speaks_tool_outputs,
        )

    @pytest.mark.asyncio
    async def test_on_asks_for_a_reply(self):
        released, ctx = await self._dispatch(announce=True)
        # the note is the tool's first result, unwrapped, and a reply is generated from it
        assert released == DISPATCHED
        assert ctx._suppress_reply is False

    @pytest.mark.asyncio
    async def test_off_releases_without_speaking(self):
        released, ctx = await self._dispatch(announce=False)
        assert released == DISPATCHED
        assert ctx._suppress_reply is True

    @pytest.mark.asyncio
    async def test_off_blocks_on_a_model_that_speaks_tool_outputs(self):
        # silence and release are exclusive there, so the answer becomes the tool output
        released, ctx = await self._dispatch(announce=False, speaks_tool_outputs=True)
        assert released == "shipped Tuesday"
        assert ctx._updates == []

    @pytest.mark.asyncio
    async def test_on_still_releases_on_a_model_that_speaks_tool_outputs(self):
        released, _ = await self._dispatch(announce=True, speaks_tool_outputs=True)
        assert released == DISPATCHED

    def test_the_description_does_not_invite_a_second_ack(self):
        # asking for a line alongside the call would double up with the dispatch reply
        description = (_build_tool(_resolve_delegation_options()).info.description or "").lower()
        assert "same turn" not in description
        assert "same completion" not in description


class TestDelegatedToolTakesTheFloor:
    """A delegated tool can hold the line for an AgentTask, then keep reasoning with it.

    This is what lets the delegation drive a piece of the conversation itself — asking for
    an email and confirming the spelling — instead of answering "I need their email" and
    waiting to be delegated to again.
    """

    pytestmark = pytest.mark.usefixtures("_clear_running_tasks")

    @pytest.mark.asyncio
    async def test_foreground_and_agent_task_from_a_delegated_tool(self):
        from livekit.agents import AgentTask
        from livekit.agents.llm import FunctionToolCall

        from .fake_llm import FakeLLM, FakeLLMResponse

        seen: list[str] = []
        answered = asyncio.Event()

        class _EmailTask(AgentTask[str]):
            def __init__(self) -> None:
                super().__init__(instructions="collect the email")

            async def on_enter(self) -> None:
                seen.append("task_entered")
                self.complete("dana@example.com")

        @function_tool
        async def collect_email(ctx: RunContext) -> str:
            """Ask the caller for their email."""
            seen.append("tool_started")
            async with ctx.foreground():
                email = await _EmailTask()
            seen.append(f"tool_got:{email}")
            answered.set()
            return f"on file: {email}"

        def _reply(text: str, *, content: str = "", calls: Any = None) -> Any:
            return FakeLLMResponse(
                input=text, content=content, ttft=0.0, duration=0.0, tool_calls=calls or []
            )

        conversation = FakeLLM(
            fake_responses=[
                _reply(
                    "hi",
                    calls=[
                        FunctionToolCall(
                            call_id="c1",
                            name=DELEGATE_TOOL_NAME,
                            arguments='{"task": "identify the caller"}',
                        )
                    ],
                )
            ]
        )
        delegation = FakeLLM(
            fake_responses=[
                _reply(
                    "identify the caller",
                    calls=[FunctionToolCall(call_id="c2", name="collect_email", arguments="{}")],
                ),
                _reply("on file: dana@example.com", content="the caller is dana@example.com"),
            ]
        )

        session = AgentSession(llm=conversation, delegation_llm=delegation)
        await session.start(agent=Agent(instructions="test agent", tools=[collect_email]))
        try:
            await asyncio.wait_for(session.run(user_input="hi"), timeout=15.0)
            # the delegate call stays in flight past the run that dispatched it
            await asyncio.wait_for(answered.wait(), timeout=15.0)
        finally:
            await asyncio.wait_for(session.aclose(), timeout=15.0)

        assert seen == ["tool_started", "task_entered", "tool_got:dana@example.com"]
