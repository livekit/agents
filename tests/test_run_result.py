from __future__ import annotations

import asyncio
import contextvars
from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    AgentTask,
    APIConnectOptions,
    APIStatusError,
    RunContext,
    ToolError,
    UnexpectedModelBehavior,
    function_tool,
)
from livekit.agents.beta.workflows.dtmf_inputs import GetDtmfResult
from livekit.agents.llm import FunctionToolCall, ToolResult
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.remote_session import RemoteSession, SessionHost
from livekit.protocol.agent_pb import agent_session as agent_pb

from .fake_llm import FakeLLM, FakeLLMResponse
from .test_remote_session import PairedTransport

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

PARENT_LINES = ["Synthetic line one.", "Synthetic line two."]


async def _start_confirmation(
    *,
    depth: int,
    content: str = "",
    later_result: Any = NOT_GIVEN,
    task_result: Any = True,
    parent_speech: bool = True,
    complete_from_callback: bool = False,
) -> tuple[AgentSession, asyncio.Event, FakeLLM]:
    ready = asyncio.Event()
    continued = asyncio.Event()

    class Confirm(AgentTask[Any]):
        async def on_enter(self) -> None:
            await self.session.say("Please confirm the synthetic details.")
            ready.set()

        @function_tool
        async def confirm_details(self) -> ToolResult:
            """Confirm the synthetic details."""
            if complete_from_callback:
                asyncio.get_running_loop().call_soon(
                    self.complete, task_result, context=contextvars.Context()
                )
            else:
                self.complete(task_result)
            return ToolResult(None, reply_required=False)

    class Wrapper(AgentTask[Any]):
        def __init__(self, remaining: int) -> None:
            super().__init__(instructions="Await synthetic confirmation.")
            self.remaining = remaining

        async def on_enter(self) -> None:
            try:
                if self.remaining:
                    answer = await Wrapper(self.remaining - 1)
                else:
                    answer = await Confirm(instructions="Confirm synthetic details.")
            except (ValueError, ToolError) as error:
                self.complete(error)
            else:
                self.complete(answer)

    class CompleteLater(AgentTask[Any]):
        async def on_enter(self) -> None:
            self.complete(later_result)

    class Parent(Agent):
        async def on_enter(self) -> None:
            try:
                if depth:
                    answer = await Wrapper(depth - 1)
                else:
                    answer = await Confirm(instructions="Confirm synthetic details.")
                assert answer == task_result
                if later_result is not NOT_GIVEN:
                    await CompleteLater(instructions="Complete immediately.")
            except (ValueError, ToolError):
                pass
            if parent_speech:
                for line in PARENT_LINES:
                    await self.session.say(line)
            continued.set()

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="yes",
                content=content,
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="confirm_details", arguments="{}", call_id="confirm")
                ],
            ),
            FakeLLMResponse(input="next question", content="A separate reply.", ttft=0, duration=0),
        ]
    )
    session = AgentSession(llm=llm)
    await session.start(Parent(instructions="Synthetic conversation."), session_host=False)
    await asyncio.wait_for(ready.wait(), timeout=3)
    # task startup can still be handing control back when the first prompt finishes
    await asyncio.sleep(0.5)
    return session, continued, llm


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("content", ["", "Confirmed."], ids=["tool_only", "text_and_tool"])
async def test_nested_completion_records_parent_speech(depth: int, content: str) -> None:
    session, continued, llm = await _start_confirmation(depth=depth, content=content)
    try:
        with patch.object(llm, "chat", wraps=llm.chat) as requests:
            result = await asyncio.wait_for(session.run(user_input="yes"), timeout=3)
            assert requests.call_count == 1
        # the history also receives both lines when RunResult closes too early
        await asyncio.wait_for(continued.wait(), timeout=3)
        history = [item.text_content for item in session.history.items if item.type == "message"]
        assert history[-2:] == PARENT_LINES
        messages = [
            event.item.text_content
            for event in result.events
            if event.type == "message" and event.item.role == "assistant"
        ]
        assert messages == ([content] if content else []) + PARENT_LINES

        with patch.object(llm, "chat", wraps=llm.chat) as requests:
            followup = await asyncio.wait_for(session.run(user_input="next question"), timeout=3)
            assert requests.call_count == 1
        assert [
            event.item.text_content for event in followup.events if event.type == "message"
        ] == ["A separate reply."]
        with pytest.raises(RuntimeError, match="^no final output$"):
            _ = followup.final_output
    finally:
        await session.aclose()


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize(
    "task_result", [True, False, 0, ""], ids=["true", "false", "zero", "empty"]
)
async def test_task_output_survives_parent_speech(
    depth: int, task_result: bool | int | str
) -> None:
    session, continued, llm = await _start_confirmation(depth=depth, task_result=task_result)
    try:
        with patch.object(llm, "chat", wraps=llm.chat) as requests:
            result = await asyncio.wait_for(
                session.run(user_input="yes", output_type=type(task_result)), timeout=3
            )
            assert requests.call_count == 1
        assert type(result.final_output) is type(task_result)
        assert result.final_output == task_result
        assert continued.is_set()
        assert [event.item.text_content for event in result.events if event.type == "message"] == (
            PARENT_LINES
        )
    finally:
        await session.aclose()


@pytest.mark.parametrize("depth", [0, 1, 2])
async def test_awaited_callback_completion_supplies_current_run_output(depth: int) -> None:
    session, continued, _ = await _start_confirmation(depth=depth, complete_from_callback=True)
    try:
        await asyncio.wait_for(session.run(user_input="wait"), timeout=3)
        assert not continued.is_set()

        result = await asyncio.wait_for(
            session.run(user_input="yes", output_type=bool, output_options=None), timeout=3
        )
        assert result.final_output is True
        assert continued.is_set()
        assert [event.item.text_content for event in result.events if event.type == "message"] == (
            PARENT_LINES
        )
    finally:
        await session.aclose()


@pytest.mark.parametrize(
    "unrelated_result",
    [NOT_GIVEN, GetDtmfResult("1234"), None, ValueError("unrelated task failed")],
    ids=["control", "value", "explicit_none", "error"],
)
async def test_unrelated_completion_does_not_replace_run_output(unrelated_result: Any) -> None:
    ready = asyncio.Event()
    answer_ready = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    unrelated_task = AgentTask[Any](instructions="Reserved for unrelated work.")

    class Answer(AgentTask[str]):
        async def on_enter(self) -> None:
            self.complete("correct answer")

    class Parent(Agent):
        async def on_enter(self) -> None:
            async def finish_unrelated() -> None:
                await release.wait()
                if unrelated_result is not NOT_GIVEN:
                    unrelated_task.complete(unrelated_result)
                completed.set()

            self.callback = asyncio.create_task(finish_unrelated())
            ready.set()

        @function_tool
        async def answer(self) -> ToolResult:
            """Provide this run's typed answer."""
            answer = await Answer(instructions="Return the answer.")
            assert answer == "correct answer"
            answer_ready.set()
            await completed.wait()
            return ToolResult(None, reply_required=False)

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="answer",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="answer", arguments="{}", call_id="answer")],
            )
        ]
    )
    session = AgentSession(llm=llm)
    parent = Parent(instructions="Test callback completion ownership.")
    await session.start(parent, session_host=False)
    try:
        await asyncio.wait_for(ready.wait(), timeout=3)
        run = session.run(user_input="answer", output_type=str, output_options=None)
        await asyncio.wait_for(answer_ready.wait(), timeout=3)
        assert not run.done()
        release.set()

        result = await asyncio.wait_for(run, timeout=3)
        assert result.final_output == "correct answer"
        assert unrelated_task.done() == (unrelated_result is not NOT_GIVEN)
    finally:
        release.set()
        completed.set()
        await asyncio.wait_for(parent.callback, timeout=3)
        await session.aclose()


@pytest.mark.parametrize(
    "paused_result",
    [NOT_GIVEN, "wrong answer", True, None, ValueError("paused task failed")],
    ids=["control", "same_type", "wrong_type", "explicit_none", "error"],
)
async def test_paused_completion_does_not_replace_run_output(paused_result: Any) -> None:
    outer_ready = asyncio.Event()
    answer_ready = asyncio.Event()
    completed = asyncio.Event()
    parent_finished = asyncio.Event()

    class Answer(AgentTask[str]):
        async def on_enter(self) -> None:
            self.complete("correct answer")

    class Inner(AgentTask[None]):
        @function_tool
        async def answer(self) -> ToolResult:
            """Provide this run's typed answer."""
            assert await Answer(instructions="Return the answer.") == "correct answer"
            answer_ready.set()
            await completed.wait()
            return ToolResult(None, reply_required=False)

    inner = Inner(instructions="Handle the current turn.")

    class Outer(AgentTask[Any]):
        async def on_enter(self) -> None:
            outer_ready.set()

        @function_tool
        async def start_inner(self) -> ToolResult:
            """Start the inner task."""
            await inner
            return ToolResult(None, reply_required=False)

    outer = Outer(instructions="Wait for the inner task.")

    class Parent(Agent):
        async def on_enter(self) -> None:
            try:
                await outer
            except ValueError:
                pass
            parent_finished.set()

    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="start",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="start_inner", arguments="{}", call_id="start")
                    ],
                ),
                FakeLLMResponse(
                    input="answer",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="answer", arguments="{}", call_id="answer")],
                ),
            ]
        )
    )
    await session.start(Parent(instructions="Start the task chain."), session_host=False)
    try:
        await asyncio.wait_for(outer_ready.wait(), timeout=3)
        assert outer._activity is not None
        await asyncio.wait_for(session.run(user_input="start"), timeout=3)
        run = session.run(user_input="answer", output_type=str, output_options=None)
        await asyncio.wait_for(answer_ready.wait(), timeout=3)
        assert session.current_agent is inner
        assert outer._activity.scheduling_paused
        assert not run.done()
        if paused_result is not NOT_GIVEN:
            outer.complete(paused_result)
        completed.set()

        result = await asyncio.wait_for(run, timeout=3)
        assert result.final_output == "correct answer"
    finally:
        completed.set()
        inner.complete(None)
        await asyncio.wait_for(inner._wait_for_inactive(), timeout=3)
        if not outer.done():
            outer.complete(None)
        await asyncio.wait_for(parent_finished.wait(), timeout=3)
        # outer may already be complete when inner resumes it
        await session._update_activity(Agent(instructions="Finish the conversation."))
        await session.aclose()


async def test_current_task_completion_during_drain_supplies_run_output() -> None:
    ready = asyncio.Event()
    tool_started = asyncio.Event()
    release = asyncio.Event()
    draining = asyncio.Event()

    class Confirm(AgentTask[bool]):
        async def on_enter(self) -> None:
            ready.set()

        @function_tool
        async def confirm(self) -> ToolResult:
            """Confirm the answer."""
            tool_started.set()
            await release.wait()
            self.complete(True)
            return ToolResult(None, reply_required=False)

    confirm = Confirm(instructions="Confirm the answer.")

    class Parent(Agent):
        async def on_enter(self) -> None:
            assert await confirm is True

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="yes",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="confirm", arguments="{}", call_id="confirm")],
            )
        ]
    )
    async with AgentSession(llm=llm) as session:
        await session.start(Parent(instructions="Start confirmation."), session_host=False)
        await asyncio.wait_for(ready.wait(), timeout=3)
        run = session.run(user_input="yes", output_type=bool, output_options=None)
        try:
            await asyncio.wait_for(tool_started.wait(), timeout=3)
            activity = confirm._activity
            assert activity is not None
            clear_resumed = activity._scheduling_resumed.clear

            def notify_draining() -> None:
                clear_resumed()
                draining.set()

            with patch.object(activity._scheduling_resumed, "clear", side_effect=notify_draining):
                drain = asyncio.create_task(session.drain())
                try:
                    await asyncio.wait_for(draining.wait(), timeout=3)
                    assert activity.scheduling_paused
                    assert session.current_agent is confirm
                finally:
                    release.set()
                    await asyncio.wait_for(drain, timeout=3)
            result = await asyncio.wait_for(run, timeout=3)
            assert result.final_output is True
        finally:
            release.set()


async def test_later_explicit_none_replaces_task_output() -> None:
    session, _, _ = await _start_confirmation(depth=1, later_result=None)
    try:
        with pytest.raises(UnexpectedModelBehavior, match="got NoneType"):
            await asyncio.wait_for(
                session.run(user_input="yes", output_type=bool, output_options=None), timeout=3
            )
    finally:
        await session.aclose()


@pytest.mark.parametrize("task_result,actual_type", [(None, "NoneType"), ("wrong", "str")])
async def test_invalid_task_output_is_not_replaced_by_parent_speech(
    task_result: Any, actual_type: str
) -> None:
    session, _, _ = await _start_confirmation(depth=1, task_result=task_result)
    try:
        with pytest.raises(UnexpectedModelBehavior, match=f"got {actual_type}"):
            await asyncio.wait_for(
                session.run(user_input="yes", output_type=bool, output_options=None), timeout=3
            )
    finally:
        await session.aclose()


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("error_type", [ValueError, ToolError])
async def test_handled_task_error_allows_parent_speech(
    depth: int, error_type: type[Exception]
) -> None:
    session, continued, llm = await _start_confirmation(
        depth=depth, task_result=error_type("verification failed")
    )
    try:
        with patch.object(llm, "chat", wraps=llm.chat) as requests:
            result = await asyncio.wait_for(session.run(user_input="yes"), timeout=3)
            assert requests.call_count == 1
        assert continued.is_set()
        assert [event.item.text_content for event in result.events if event.type == "message"] == (
            PARENT_LINES
        )
    finally:
        await session.aclose()


@pytest.mark.parametrize("error_after_success", [False, True])
async def test_handled_task_error_does_not_supply_typed_output(error_after_success: bool) -> None:
    session, continued, _ = await _start_confirmation(
        depth=1,
        task_result=True if error_after_success else ValueError("verification failed"),
        later_result=ValueError("verification failed") if error_after_success else NOT_GIVEN,
    )
    try:
        with pytest.raises(UnexpectedModelBehavior, match="got NoneType"):
            await asyncio.wait_for(
                session.run(user_input="yes", output_type=bool, output_options=None), timeout=3
            )
        assert continued.is_set()
    finally:
        await session.aclose()


@pytest.mark.parametrize("speaks", [True, False], ids=["tool_says_a_line", "model_replies"])
async def test_handled_task_error_in_tool_allows_parent_speech(speaks: bool) -> None:
    class FailOnEnter(AgentTask[bool]):
        async def on_enter(self) -> None:
            self.complete(ValueError("verification failed"))

    class Parent(Agent):
        @function_tool
        async def verify_details(self) -> ToolResult:
            """Verify the synthetic details."""
            try:
                await FailOnEnter(instructions="Fail synthetic verification.")
            except ValueError:
                if speaks:
                    await self.session.say(PARENT_LINES[0])
                return ToolResult("verification failed", reply_required=not speaks)
            return ToolResult("verified", reply_required=True)

    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="yes",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="verify_details", arguments="{}", call_id="verify")
                ],
            ),
            FakeLLMResponse(
                input="verification failed", content=PARENT_LINES[1], ttft=0, duration=0
            ),
        ]
    )
    session = AgentSession(llm=llm)
    await session.start(Parent(instructions="Synthetic conversation."), session_host=False)
    try:
        await asyncio.sleep(0.5)
        result = await asyncio.wait_for(session.run(user_input="yes"), timeout=3)
        assert [event.item.text_content for event in result.events if event.type == "message"] == (
            [PARENT_LINES[0]] if speaks else [PARENT_LINES[1]]
        )
    finally:
        await session.aclose()


@pytest.mark.parametrize("error_type", [ValueError, ToolError])
async def test_task_error_on_final_speech_is_propagated(error_type: type[Exception]) -> None:
    session, _, _ = await _start_confirmation(
        depth=0, task_result=error_type("verification failed"), parent_speech=False
    )
    try:
        with pytest.raises(error_type, match="verification failed"):
            await asyncio.wait_for(session.run(user_input="yes"), timeout=3)
    finally:
        await session.aclose()


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize(
    "task_result",
    [True, ValueError("verification failed"), ToolError("verification failed")],
    ids=["success", "handled_value_error", "handled_tool_error"],
)
async def test_remote_run_records_parent_speech(depth: int, task_result: Any) -> None:
    session, continued, llm = await _start_confirmation(depth=depth, task_result=task_result)
    host_transport, client_transport = PairedTransport.create_pair()
    host = SessionHost(host_transport)
    host.register_session(session)
    client = RemoteSession(client_transport)
    await host.start()
    await client.start()
    try:
        with patch.object(llm, "chat", wraps=llm.chat) as requests:
            response = await client.run("yes", timeout=3)
            assert requests.call_count == 1
        await asyncio.wait_for(continued.wait(), timeout=3)
        assert [
            "".join(content.text for content in item.message.content)
            for item in response.items
            if item.HasField("message") and item.message.role == agent_pb.ASSISTANT
        ] == PARENT_LINES

        with patch.object(llm, "chat", wraps=llm.chat) as requests:
            response = await client.run("next question", timeout=3)
            assert requests.call_count == 1
        assert [
            "".join(content.text for content in item.message.content)
            for item in response.items
            if item.HasField("message") and item.message.role == agent_pb.ASSISTANT
        ] == ["A separate reply."]
    finally:
        await client.aclose()
        await host.aclose()
        await session.aclose()


# -- ctx.foreground() scopes --------------------------------------------------------------


def _tool_reply(input: str, tool: str, arguments: str = "{}") -> FakeLLMResponse:
    return FakeLLMResponse(
        input=input,
        content="",
        ttft=0.1,
        duration=0.1,
        tool_calls=[FunctionToolCall(name=tool, arguments=arguments, call_id=f"call_{tool}")],
    )


def _text_reply(input: str, content: str) -> FakeLLMResponse:
    return FakeLLMResponse(input=input, content=content, ttft=0.1, duration=0.1)


class Forwarded(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="forwarded agent")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="forwarded_greeting")


class Router(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="router agent")

    @function_tool
    async def forward(self, ctx: RunContext) -> None:
        """Hand off to the forwarded agent while holding the floor."""
        # the update goes non-blocking, so the handoff below runs after the turn ended
        await ctx.update("")
        forward_agent = Forwarded()
        async with ctx.foreground():
            ctx.session.update_agent(forward_agent)


class AskNameTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="ask name task")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="ask_name")

    @function_tool
    async def record_name(self, ctx: RunContext, name: str) -> str:
        """Called when the user provides their name."""
        self.complete(None)
        return "recorded"


class Surveyor(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="surveyor agent")

    @function_tool
    async def start_survey(self, ctx: RunContext) -> None:
        """Called when the user is ready to start the survey."""
        async with ctx.foreground():
            await AskNameTask()

    @function_tool
    async def ask_over_speech(self, ctx: RunContext) -> None:
        """Called when the user wants the survey read out first."""
        async with ctx.foreground():
            await self.session.generate_reply(instructions="ask_via_tool")

    @function_tool
    async def collect_name(self, ctx: RunContext) -> str:
        """Called when the name is needed."""
        await AskNameTask()
        return "collected"


async def test_handoff_inside_foreground_after_going_non_blocking() -> None:
    """The new agent's on_enter reply lands on the run of the tool that handed off."""
    llm = FakeLLM(
        fake_responses=[
            _tool_reply("go", "forward"),
            _text_reply("forwarded_greeting", "here is a joke"),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(Router())

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)

        result.expect.next_event().is_function_call(name="forward")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_agent_handoff(new_agent_type=Forwarded)
        result.expect.next_event().is_message(role="assistant")


async def test_background_tool_does_not_hold_the_run_open() -> None:
    """A tool that answers with ctx.update() and keeps working never delays the run."""

    class BackgroundWorker(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="background worker")
            self.cancelled = False

        @function_tool
        async def slow_job(self, ctx: RunContext) -> None:
            """Report progress, then keep working in the background."""
            await ctx.update("working on it")
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    llm = FakeLLM(
        fake_responses=[_tool_reply("go", "slow_job"), _text_reply("", "ok, working on it")]
    )
    agent = BackgroundWorker()
    async with AgentSession(llm=llm) as sess:
        await sess.start(agent)

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=60.0)
        elapsed = loop.time() - started_at

        assert "FunctionCallEvent" in [type(e).__name__ for e in result.events]
        assert elapsed < 1.0, f"the background tool held run() open for {elapsed:.2f}s"
        assert not agent.cancelled


async def test_guard_released_when_the_floor_never_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run still returns when wait_for_idle raises inside ctx.foreground()."""
    from livekit.agents.voice.agent_activity import ActivityClosedError

    async def _raise(self: AgentSession) -> None:
        raise ActivityClosedError("activity closed")

    llm = FakeLLM(fake_responses=[_tool_reply("go", "forward"), _text_reply("", "ok")])
    async with AgentSession(llm=llm) as sess:
        await sess.start(Router())
        monkeypatch.setattr(AgentSession, "wait_for_idle", _raise)

        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)


async def test_stale_foreground_scope_does_not_hold_a_later_run() -> None:
    """A tool taking the floor after its own run ended does not gate the next run."""
    hold = 5.0
    resume = asyncio.Event()

    class LateFloorWorker(Agent):
        @function_tool
        async def slow_job(self, ctx: RunContext) -> None:
            """Report progress, then take the floor much later."""
            await ctx.update("working on it")
            await resume.wait()
            async with ctx.foreground():
                await asyncio.sleep(hold)

    llm = FakeLLM(
        fake_responses=[
            _tool_reply("go", "slow_job"),
            _text_reply("", "ok, working on it"),
            _text_reply("second", "hello again"),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(LateFloorWorker(instructions="late floor worker"))

        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)

        second_run = sess.run(user_input="second")
        resume.set()

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        await asyncio.wait_for(second_run, timeout=60.0)
        elapsed = loop.time() - started_at

        assert elapsed < hold, f"the second run was held open for {elapsed:.2f}s"


async def test_agent_task_inside_foreground_does_not_deadlock() -> None:
    """A foregrounded AgentTask needing another user turn lets run() return first."""
    llm = FakeLLM(
        fake_responses=[
            _tool_reply("ready", "start_survey"),
            _text_reply("ask_name", "what is your name?"),
            _tool_reply("Bob", "record_name", '{"name": "Bob"}'),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(Surveyor())

        await asyncio.wait_for(sess.run(user_input="ready"), timeout=5.0)

        second_result = await asyncio.wait_for(sess.run(user_input="Bob"), timeout=5.0)
        second_result.expect.contains_function_call(name="record_name")


async def test_agent_task_under_a_foregrounded_reply_does_not_deadlock() -> None:
    """The same wait one level deeper: a tool of a foregrounded reply awaits the task."""
    llm = FakeLLM(
        fake_responses=[
            _tool_reply("ready", "ask_over_speech"),
            _tool_reply("ask_via_tool", "collect_name"),
            _text_reply("ask_name", "what is your name?"),
            _text_reply("collected", "all done"),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(Surveyor())

        await asyncio.wait_for(sess.run(user_input="ready"), timeout=5.0)


# -- start(capture_run=True) --------------------------------------------------------------


async def test_start_capture_run_records_every_on_enter_speech() -> None:
    class ChattyAgent(Agent):
        async def on_enter(self) -> None:
            await self.session.say("deterministic greeting")
            await self.session.generate_reply(instructions="generated_greeting")

    llm = FakeLLM(fake_responses=[_text_reply("generated_greeting", "generated greeting")])
    async with AgentSession(llm=llm) as sess:
        result = await asyncio.wait_for(
            sess.start(ChattyAgent(instructions="chatty agent"), capture_run=True), timeout=10.0
        )

        result.expect.next_event().is_agent_handoff(new_agent_type=ChattyAgent)
        first = result.expect.next_event().is_message(role="assistant")
        assert first.event().item.text_content == "deterministic greeting"
        second = result.expect.next_event().is_message(role="assistant")
        assert second.event().item.text_content == "generated greeting"
        result.expect.no_more_events()


async def test_start_capture_run_completes_with_a_silent_on_enter() -> None:
    async with AgentSession(llm=FakeLLM()) as sess:
        result = await asyncio.wait_for(
            sess.start(Agent(instructions="silent agent"), capture_run=True), timeout=5.0
        )

        result.expect.next_event().is_agent_handoff(new_agent_type=Agent)
        result.expect.no_more_events()


# -- errors and silent turns --------------------------------------------------------------


class FailingLLM(FakeLLM):
    def chat(self, **kwargs: Any) -> Any:
        raise APIStatusError("object cannot be found", status_code=401, retryable=True)


def _failing_session(max_retry: int = 0) -> AgentSession:
    return AgentSession(
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.01)
        )
    )


@pytest.mark.parametrize("max_retry", [0, 1])
async def test_run_propagates_llm_error(max_retry: int) -> None:
    session = _failing_session(max_retry)
    await session.start(agent=Agent(instructions="test agent", llm=FailingLLM()))
    try:
        with pytest.raises(APIStatusError):
            await asyncio.wait_for(session.run(user_input="hello"), timeout=10.0)
    finally:
        await session.aclose()


@pytest.mark.parametrize("failing", [True, False], ids=["llm_error", "silent_turn"])
async def test_remote_run_reports_errors_but_not_silence(failing: bool) -> None:
    """An LLM error reaches the remote caller; an empty completion is an empty item list."""
    session = _failing_session() if failing else AgentSession()
    llm = FailingLLM() if failing else FakeLLM()
    host_transport, client_transport = PairedTransport.create_pair()
    host = SessionHost(host_transport)
    host.register_session(session)
    await session.start(agent=Agent(instructions="test agent", llm=llm))
    await host.start()
    client = RemoteSession(client_transport)
    await client.start()
    try:
        if failing:
            with pytest.raises(RuntimeError, match="failed"):
                await client.run("order a big mac", timeout=10.0)
        else:
            resp = await client.run("okay, thank you", timeout=10.0)
            assert list(resp.items) == []
    finally:
        await client.aclose()
        await host.aclose()
        await session.aclose()
