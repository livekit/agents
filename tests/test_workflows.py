from typing import Literal
from unittest.mock import patch

import pytest

from livekit.agents import AgentSession, beta, inference, llm
from livekit.agents.llm.tool_context import ToolError
from livekit.agents.voice.run_result import RunResult
from livekit.rtc import Room


def _llm_model() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1")


@pytest.mark.asyncio
@pytest.mark.parametrize("input_mode", ["text", "audio"])
async def test_collect_email(input_mode: Literal["text", "audio"]) -> None:
    async with _llm_model() as llm, AgentSession(llm=llm) as sess:
        await sess.start(beta.workflows.GetEmailTask())

        result = await sess.run(
            user_input="My email address is theo at livekit dot io?", input_mode=input_mode
        )

        if input_mode == "text":
            assert isinstance(result.final_output, beta.workflows.GetEmailResult)
        else:
            # confirmation is required for audio input
            result = await sess.run(
                user_input="Yes", output_type=beta.workflows.GetEmailResult, input_mode=input_mode
            )

        assert result.final_output.email_address == "theo@livekit.io"

    async with _llm_model() as llm, AgentSession(llm=llm) as sess:
        await sess.start(beta.workflows.GetEmailTask())

        with pytest.raises(ToolError):
            await sess.run(user_input="I don't want to give my email address")


class MockJobContext:
    def __init__(self) -> None:
        self.room = Room()


def get_mock_job_context() -> MockJobContext:
    return MockJobContext()


def get_dtmf_task(ask_for_confirmation: bool) -> beta.workflows.GetDtmfTask:
    return beta.workflows.GetDtmfTask(
        num_digits=10,
        ask_for_confirmation=ask_for_confirmation,
        extra_instructions="Let the caller know you'll record their 10-digit account number and that they can speak or dial it. ",
    )


@pytest.mark.asyncio
async def test_get_dtmf_sip_event_without_confirmation() -> None:
    with patch("livekit.agents.beta.workflows.dtmf_inputs.get_job_context", get_mock_job_context):
        async with _llm_model() as llm, AgentSession(llm=llm) as sess:
            await sess.start(get_dtmf_task(ask_for_confirmation=False))

            result = await sess.run(
                user_input="My account number is 1 2 3 4 5 6 7 8 9 0",
                output_type=beta.workflows.GetDtmfResult,
            )

            result.expect.contains_function_call(
                name="record_inputs",
                arguments={"inputs": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]},
            )

            assert result.final_output.user_input == "1 2 3 4 5 6 7 8 9 0"


@pytest.mark.asyncio
async def test_get_dtmf_sip_event_with_confirmation() -> None:
    with patch("livekit.agents.beta.workflows.dtmf_inputs.get_job_context", get_mock_job_context):
        async with _llm_model() as llm, AgentSession(llm=llm) as sess:
            await sess.start(get_dtmf_task(ask_for_confirmation=True))

            initial_result: RunResult[None] = await sess.run(
                user_input="My account number is 1 2 3 4 5 6 7 8 9 0",
                output_type=None,
            )

            await (
                initial_result.expect.next_event()
                .is_message(role="assistant")
                .judge(
                    llm,
                    intent="Asks user to confirm the entered digits 1 2 3 4 5 6 7 8 9 0 is correct",
                )
            )

            result = await sess.run(
                user_input="Yes, it's correct",
                output_type=beta.workflows.GetDtmfResult,
            )

            result.expect.contains_function_call(
                name="confirm_inputs",
                arguments={"inputs": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]},
            )

            assert result.final_output.user_input == "1 2 3 4 5 6 7 8 9 0"
