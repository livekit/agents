import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents.beta.workflows.credit_card import (
    CardCollectionRestartError,
    GetCardNumberResult,
    GetCardNumberTask,
    GetCreditCardTask,
    GetExpirationDateTask,
    GetSecurityCodeTask,
)
from livekit.agents.llm.chat_context import ChatContext, Instructions
from livekit.agents.llm.tool_context import ToolError, ToolFlag

pytestmark = pytest.mark.unit


def _run_context(modality: str = "audio") -> Any:
    return SimpleNamespace(
        speech_handle=SimpleNamespace(input_details=SimpleNamespace(modality=modality))
    )


@pytest.mark.asyncio
async def test_credit_card_restart_uses_latest_chat_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_ctx = ChatContext.empty()
    initial_ctx.add_message(role="user", content="The previous card number was invalid.")
    task = GetCreditCardTask(chat_ctx=initial_ctx)
    group_contexts: list[ChatContext] = []

    class FakeTaskGroup:
        def __init__(self, *, chat_ctx: ChatContext, **_: Any) -> None:
            group_contexts.append(chat_ctx.copy())

        def add(self, *_: Any, **__: Any) -> "FakeTaskGroup":
            return self

        def __await__(self):
            async def run() -> Any:
                if len(group_contexts) == 1:
                    task._chat_ctx.add_message(
                        role="user", content="The first four digits are four two four two."
                    )
                    raise CardCollectionRestartError("start the card number again")

                return SimpleNamespace(
                    task_results={
                        "cardholder_name_task": SimpleNamespace(
                            first_name="Ada", last_name="Lovelace"
                        ),
                        "card_number_task": SimpleNamespace(
                            issuer="Visa", card_number="4242424242424242"
                        ),
                        "security_code_task": SimpleNamespace(security_code="123"),
                        "expiration_date_task": SimpleNamespace(date="07/32"),
                    }
                )

            return run().__await__()

    monkeypatch.setattr("livekit.agents.beta.workflows.credit_card.TaskGroup", FakeTaskGroup)

    await task.on_enter()

    assert len(group_contexts) == 2
    assert group_contexts[0].messages()[-1].text_content == "The previous card number was invalid."
    assert (
        group_contexts[1].messages()[-1].text_content
        == "The first four digits are four two four two."
    )


@pytest.mark.asyncio
async def test_card_number_incremental_suffix_correction_and_finish() -> None:
    task = GetCardNumberTask(require_confirmation=False)
    ctx = _run_context()

    first_output = await task._append_card_number_impl(ctx, "4242")
    await task._append_card_number_impl(ctx, "4243")
    await task._append_card_number_impl(ctx, "2", replace_last=1)
    await task._append_card_number_impl(ctx, "4242")

    with patch.object(task, "complete") as complete:
        await task._append_card_number_impl(ctx, "4242", finish=True)

    assert "4242" not in (first_output or "")
    result = complete.call_args.args[0]
    assert result == GetCardNumberResult(issuer="Visa", card_number="4242424242424242")


@pytest.mark.asyncio
async def test_card_number_invalid_updates_do_not_mutate_buffer() -> None:
    task = GetCardNumberTask(require_confirmation=False)
    ctx = _run_context()
    await task._append_card_number_impl(ctx, "4242")

    with pytest.raises(ToolError, match="negative"):
        await task._append_card_number_impl(ctx, "1", replace_last=-1)
    assert task._card_number_buffer == "4242"

    with pytest.raises(ToolError, match="replace_last"):
        await task._append_card_number_impl(ctx, "1", replace_last=5)
    assert task._card_number_buffer == "4242"

    with pytest.raises(ToolError, match="too long"):
        await task._append_card_number_impl(ctx, "1" * 16)
    assert task._card_number_buffer == "4242"

    with pytest.raises(ToolError, match="No card-number digits"):
        await task._append_card_number_impl(ctx, "not digits")
    assert task._card_number_buffer == "4242"


@pytest.mark.asyncio
async def test_card_number_equal_confirmation_auto_submits() -> None:
    task = GetCardNumberTask(require_confirmation=True)
    ctx = _run_context()

    transition = await task._append_card_number_impl(ctx, "4242424242424242", finish=True)
    assert transition is not None and "repeat" in transition
    assert task._card_number_buffer == ""

    await task._append_card_number_impl(ctx, "42424242")
    with patch.object(task, "complete") as complete:
        await task._append_card_number_impl(ctx, "42424242")

    result = complete.call_args.args[0]
    assert result == GetCardNumberResult(issuer="Visa", card_number="4242424242424242")


@pytest.mark.asyncio
async def test_card_number_valid_mismatch_becomes_corrected_original() -> None:
    task = GetCardNumberTask(require_confirmation=True)
    ctx = _run_context()
    await task._append_card_number_impl(ctx, "4242424242424242", finish=True)

    correction = await task._append_card_number_impl(ctx, "4000000000000002", finish=True)
    assert correction is not None and "correction" in correction
    assert task._card_number == "4000000000000002"
    assert task._card_number_buffer == ""

    with patch.object(task, "complete") as complete:
        await task._append_card_number_impl(ctx, "4000000000000002")

    result = complete.call_args.args[0]
    assert result == GetCardNumberResult(issuer="Visa", card_number="4000000000000002")


@pytest.mark.asyncio
async def test_invalid_confirmation_is_discarded() -> None:
    task = GetCardNumberTask(require_confirmation=True)
    ctx = _run_context()
    await task._append_card_number_impl(ctx, "4242424242424242", finish=True)

    with pytest.raises(ToolError, match="failed validation"):
        await task._append_card_number_impl(ctx, "4242424242424241", finish=True)

    assert task._card_number == "4242424242424242"
    assert task._card_number_buffer == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card_number, issuer",
    [
        ("378282246310005", "American Express"),
        ("5555555555554444", "Mastercard"),
        ("4000000000000000006", "Visa"),
    ],
)
async def test_card_number_conservative_auto_submit(card_number: str, issuer: str) -> None:
    task = GetCardNumberTask(require_confirmation=False)

    with patch.object(task, "complete") as complete:
        await task._append_card_number_impl(_run_context(), card_number)

    assert complete.call_args.args[0] == GetCardNumberResult(issuer=issuer, card_number=card_number)


@pytest.mark.asyncio
async def test_card_number_extendable_length_requires_submit() -> None:
    task = GetCardNumberTask(require_confirmation=False)
    ctx = _run_context()

    await task._append_card_number_impl(ctx, "4242424242424242")
    assert not task.done()

    with patch.object(task, "complete") as complete:
        await task._submit_card_number_impl(ctx)

    assert complete.call_args.args[0] == GetCardNumberResult(
        issuer="Visa", card_number="4242424242424242"
    )


@pytest.mark.asyncio
async def test_card_number_parallel_calls_preserve_order() -> None:
    task = GetCardNumberTask(require_confirmation=False)
    ctx = _run_context()

    with patch.object(task, "complete") as complete:
        await asyncio.gather(
            task._append_card_number_impl(ctx, "4242"),
            task._append_card_number_impl(ctx, "4242"),
            task._append_card_number_impl(ctx, "4242"),
            task._append_card_number_impl(ctx, "4242", finish=True),
        )

    assert complete.call_args.args[0] == GetCardNumberResult(
        issuer="Visa", card_number="4242424242424242"
    )


@pytest.mark.asyncio
async def test_sensitive_field_duplicate_updates_redirect_to_confirmation() -> None:
    ctx = _run_context()

    security_code_task = GetSecurityCodeTask(require_confirmation=True)
    assert await security_code_task._update_security_code_impl(ctx, "123")
    with pytest.raises(ToolError, match="confirm_security_code"):
        await security_code_task._update_security_code_impl(ctx, "123")
    assert await security_code_task._update_security_code_impl(ctx, "456")

    expiration_task = GetExpirationDateTask(require_confirmation=True)
    with patch.object(expiration_task, "_current_date", return_value=date(2026, 7, 21)):
        assert await expiration_task._update_expiration_date_impl(ctx, 4, 2099)
        assert expiration_task._expiration_date == "04/99"
        with pytest.raises(ToolError, match="confirm_expiration_date"):
            await expiration_task._update_expiration_date_impl(ctx, 4, 99)
        assert await expiration_task._update_expiration_date_impl(ctx, 5, 2099)

        for invalid_year in (1999, 2100):
            with pytest.raises(ToolError, match="last two digits or the full four-digit year"):
                await expiration_task._update_expiration_date_impl(ctx, 5, invalid_year)


@pytest.mark.asyncio
async def test_expiration_year_uses_the_current_century() -> None:
    task = GetExpirationDateTask(require_confirmation=True)

    with patch.object(task, "_current_date", return_value=date(3000, 1, 1)):
        assert await task._update_expiration_date_impl(_run_context(), 4, 3000)
        assert task._expiration_date == "04/00"

        with pytest.raises(ToolError, match="confirm_expiration_date"):
            await task._update_expiration_date_impl(_run_context(), 4, 0)

        for invalid_year in (2999, 3100):
            with pytest.raises(ToolError, match="last two digits or the full four-digit year"):
                await task._update_expiration_date_impl(_run_context(), 4, invalid_year)


def test_credit_card_instructions_prohibit_concrete_format_examples() -> None:
    card_number_task = GetCardNumberTask()
    tools_by_id = {tool.id: tool for tool in card_number_task.tools}
    tool_ids = set(tools_by_id)
    assert {"append_card_number", "submit_card_number"} <= tool_ids
    assert {"update_card_number", "confirm_card_number"}.isdisjoint(tool_ids)
    assert not (tools_by_id["append_card_number"].info.flags & ToolFlag.IGNORE_ON_ENTER)
    assert not (tools_by_id["submit_card_number"].info.flags & ToolFlag.IGNORE_ON_ENTER)
    for tool_id in ("append_card_number", "submit_card_number"):
        description = tools_by_id[tool_id].info.description or ""
        assert "confirmation" not in description.lower()

    assert isinstance(card_number_task.instructions, Instructions)
    rendered_card_number = card_number_task.instructions.render(modality="audio")
    assert "during confirmation" not in rendered_card_number.lower()
    assert "second complete reading" not in rendered_card_number.lower()

    explicit_ask_tools = {
        tool.id: tool for tool in GetCardNumberTask(require_explicit_ask=True).tools
    }
    assert explicit_ask_tools["append_card_number"].info.flags & ToolFlag.IGNORE_ON_ENTER
    assert explicit_ask_tools["submit_card_number"].info.flags & ToolFlag.IGNORE_ON_ENTER

    for task in (
        card_number_task,
        GetSecurityCodeTask(),
        GetExpirationDateTask(),
    ):
        assert isinstance(task.instructions, Instructions)
        rendered = task.instructions.render(modality="audio")
        assert "never provide invented or example" in rendered.lower()

    expiration_task = GetExpirationDateTask()
    expiration = expiration_task.instructions
    assert isinstance(expiration, Instructions)
    rendered_expiration = expiration.render(modality="audio")
    assert "last two digits or in full four digits" in rendered_expiration
    assert "never interpret any part as a day of the month" in rendered_expiration.lower()
    assert "strip any ordinal suffix" in rendered_expiration.lower()
    assert "never call `decline_card_capture` for it" in rendered_expiration.lower()

    update_expiration_tool = next(
        tool for tool in expiration_task.tools if tool.id == "update_expiration_date"
    )
    description = update_expiration_tool.info.description or ""
    assert "always interpreted as a year and never as a day" in description.lower()
