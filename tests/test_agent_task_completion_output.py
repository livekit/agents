from __future__ import annotations

import asyncio
import contextvars
from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents import NOT_GIVEN, Agent, AgentSession, AgentTask, ToolError, function_tool
from livekit.agents.beta.workflows.dtmf_inputs import GetDtmfResult
from livekit.agents.llm import FunctionToolCall, ToolResult
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

    class Confirm(AgentTask[bool]):
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

    class Wrapper(AgentTask[bool]):
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
    # Task startup can still be handing control back when the first prompt finishes.
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
        # The history also receives both lines when RunResult closes too early.
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
    finally:
        await session.aclose()


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("disable_retries", [False, True], ids=["default_retries", "no_retries"])
async def test_task_output_survives_parent_speech(depth: int, disable_retries: bool) -> None:
    session, continued, llm = await _start_confirmation(depth=depth)
    try:
        with patch.object(llm, "chat", wraps=llm.chat) as requests:
            result = await asyncio.wait_for(
                session.run(
                    user_input="yes",
                    output_type=bool,
                    output_options=None if disable_retries else NOT_GIVEN,
                ),
                timeout=3,
            )
            assert requests.call_count == 1
        assert result.final_output is True
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


async def test_later_explicit_none_replaces_task_output() -> None:
    from livekit.agents import UnexpectedModelBehavior

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
    from livekit.agents import UnexpectedModelBehavior

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
    from livekit.agents import UnexpectedModelBehavior

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
