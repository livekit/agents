from typing import Literal
from unittest.mock import patch

import pytest

from livekit.agents import AgentSession, beta, inference, llm
from livekit.agents.llm.tool_context import ToolError
from livekit.agents.voice.run_result import RunResult
from livekit.rtc import Room

pytestmark = pytest.mark.evals


def _llm_model() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1")


@pytest.mark.asyncio
@pytest.mark.parametrize("input_modality", ["text", "audio"])
async def test_collect_email(input_modality: Literal["text", "audio"]) -> None:
    async with _llm_model() as llm, AgentSession(llm=llm) as sess:
        await sess.start(beta.workflows.GetEmailTask())

        result = await sess.run(
            user_input="My email address is theo at livekit dot io?", input_modality=input_modality
        )

        if input_modality == "text":
            assert isinstance(result.final_output, beta.workflows.GetEmailResult)
        else:
            # confirmation is required for audio input
            result = await sess.run(
                user_input="Yes",
                output_type=beta.workflows.GetEmailResult,
                input_modality=input_modality,
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


def _audio_run_context() -> object:
    # Minimal stand-in for RunContext: the update_* impls only read
    # ctx.speech_handle.input_details.modality (via _confirmation_required).
    from types import SimpleNamespace

    return SimpleNamespace(
        speech_handle=SimpleNamespace(input_details=SimpleNamespace(modality="audio"))
    )


@pytest.mark.asyncio
async def test_credit_card_update_redirects_to_confirm_when_pending() -> None:
    # A repeated identical value routed to update_* (instead of confirm_*)
    # must raise a ToolError pointing at the confirm tool - re-arming the
    # confirmation used to stall the task (AGT-3139).
    from livekit.agents.beta.workflows.credit_card import (
        GetCardNumberTask,
        GetExpirationDateTask,
        GetSecurityCodeTask,
    )

    ctx = _audio_run_context()

    number_task = GetCardNumberTask()
    assert await number_task._update_card_number_impl(ctx, "4242 4242 4242 4242")
    with pytest.raises(ToolError, match="confirm_card_number"):
        await number_task._update_card_number_impl(ctx, "4242424242424242")

    code_task = GetSecurityCodeTask()
    assert await code_task._update_security_code_impl(ctx, "123")
    with pytest.raises(ToolError, match="confirm_security_code"):
        await code_task._update_security_code_impl(ctx, "123")

    date_task = GetExpirationDateTask()
    assert await date_task._update_expiration_date_impl(ctx, 4, 99)
    with pytest.raises(ToolError, match="confirm_expiration_date"):
        await date_task._update_expiration_date_impl(ctx, 4, 99)

    # A *different* value is a correction, not a confirmation: it must
    # re-arm confirmation with the new value instead of raising.
    assert await code_task._update_security_code_impl(ctx, "456")


@pytest.mark.asyncio
async def test_collect_security_code_repeat_back_completes() -> None:
    # End-to-end: with audio-modality confirmation, repeating the code must
    # complete the task in that same turn, whichever tool the LLM routes
    # the repeat to (AGT-3139 stall regression).
    from livekit.agents.beta.workflows.credit_card import (
        GetSecurityCodeResult,
        GetSecurityCodeTask,
    )

    async with _llm_model() as llm, AgentSession(llm=llm) as sess:
        await sess.start(GetSecurityCodeTask())

        await sess.run(user_input="The security code is 123", input_modality="audio")

        result = await sess.run(
            user_input="1 2 3",
            output_type=GetSecurityCodeResult,
            input_modality="audio",
        )

        assert result.final_output.security_code == "123"
