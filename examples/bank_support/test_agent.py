"""The agent's wiring: guarded tools keep their schemas, flows become tasks, caller turns are annotated."""

from __future__ import annotations

from agent import BankAgent, WriteTask, annotate_user_turn
from bank_db import FakeBank
from contract import CallState
from flows.bank import FLOWS

from livekit.agents.llm import ChatMessage


def _names(tools) -> set[str]:
    return {t.info.name for t in tools}


def test_the_parent_offers_reads_the_flows_a_transfer_and_the_gated_end_call():
    agent = BankAgent()
    names = _names(agent.tools)
    assert {
        "verify_identity",
        "get_accounts",
        "get_card",
        "get_transactions",
        "transfer_to_human",
        "end_call",
    } <= names
    assert {f"start_{flow}" for flow in FLOWS} <= names
    assert "block_card" not in names  # writes live in their tasks


def test_a_guarded_tool_keeps_its_description_and_parameters():
    agent = BankAgent()
    tool = next(t for t in agent.tools if t.info.name == "get_card")
    assert (
        "last_four" in str(tool.info.description)
        or "four digits" in str(tool.info.description).lower()
    )


def test_each_flow_becomes_a_task_with_only_its_tools():
    task = WriteTask(FLOWS["transfer"], CallState())
    assert _names(task.tools) == {
        "get_accounts",
        "transfer_check",
        "transfer_funds",
        "finish_request",
        "end_request",
    }
    assert task.instructions.startswith("You are handling exactly one request")


def test_spoken_digits_are_appended_beneath_the_callers_words():
    message = ChatMessage(
        role="user", content=["it's four one five, five five five, zero one two three"]
    )
    state = CallState()
    annotate_user_turn(state, message)
    assert (
        message.text_content
        == "it's four one five, five five five, zero one two three\n[heard: 4155550123]"
    )
    assert state.user_turns[-1] == "it's four one five, five five five, zero one two three"
    plain = ChatMessage(role="user", content=["I lost my card"])
    annotate_user_turn(state, plain)
    assert plain.text_content == "I lost my card"


def test_the_bank_seeds_from_scenario_userdata():
    import simulation

    bank = simulation.bank_from_userdata({"today": "2026-06-12", "balances": {"checking": 100.0}})
    assert bank.today.isoformat() == "2026-06-12"
    assert next(a for a in bank.accounts("c_1") if a.kind == "checking").balance == 100.0
    assert isinstance(simulation.bank_from_userdata({}), FakeBank)


async def test_a_task_does_not_finish_before_one_of_its_writes():
    state = CallState()
    task = WriteTask(FLOWS["block_card"], state)
    reply = await task.finish_request_reply("blocked the card")
    assert reply.startswith("NOT DONE") and "block_card" in reply
    assert not task.done()


async def test_a_task_stays_open_after_its_first_write():
    state = CallState()
    task = WriteTask(FLOWS["block_card"], state)
    state.record("block_card", {"card_id": "card_1", "reason": "lost"}, "{}")
    assert not task.done()
    assert await task.finish_request_reply("blocked the card") is None


def test_the_hand_off_is_recorded():
    agent = BankAgent()
    assert "transfer_to_human" in {t.info.name for t in agent.tools}
    from flows.bank import CONTRACT

    assert "transfer_to_human" in CONTRACT.rules["end_call"].after_any


async def test_a_task_can_undo_its_own_write():
    task = WriteTask(FLOWS["block_card"], CallState())
    assert "unblock_card" in _names(task.tools)


async def test_a_hand_off_says_goodbye_and_closes_the_session(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import agent as agent_module

    agent_module.SESSION_DRAIN_S = 0
    monkeypatch.setattr(agent_module, "get_job_context", MagicMock)
    ctx = MagicMock()
    ctx.session.say = AsyncMock()
    ctx.session.aclose = AsyncMock()
    bank_agent = BankAgent()
    await bank_agent._transfer_to_human_impl(ctx, "wants a higher limit", "Connecting you now.")
    await bank_agent._close_task
    ctx.session.say.assert_awaited_once_with("Connecting you now.", allow_interruptions=False)
    ctx.session.aclose.assert_awaited_once()


async def test_a_task_annotates_the_callers_digits():
    from agent import WriteTask as Task

    task = Task(FLOWS["block_card"], CallState())
    message = ChatMessage(role="user", content=["four four two one"])
    await task.on_user_turn_completed(None, message)
    assert message.text_content.endswith("[heard: 4421]")


async def test_ending_the_call_deletes_the_room_and_shuts_down_the_job(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import agent as agent_module

    agent_module.SESSION_DRAIN_S = 0
    job = MagicMock()
    monkeypatch.setattr(agent_module, "get_job_context", lambda: job)
    ctx = MagicMock()
    ctx.session.say = AsyncMock()
    ctx.session.aclose = AsyncMock()
    bank_agent = BankAgent()
    await bank_agent._say_and_close(ctx, "Goodbye.")
    await bank_agent._close_task
    job.shutdown.assert_called_once()
    # Run the callback the way JobContext does: a callback that takes an argument gets the
    # shutdown reason, so the room to delete must not come from that argument.
    (callback,), _ = job.add_shutdown_callback.call_args
    job.delete_room = AsyncMock()
    if callback.__code__.co_argcount >= 1:
        await callback("agent ended the call")
    else:
        await callback()
    job.delete_room.assert_awaited_once_with()


async def test_a_hand_off_log_carries_no_caller_request(caplog):
    import logging
    from unittest.mock import AsyncMock, MagicMock

    import agent as agent_module

    agent_module.SESSION_DRAIN_S = 0
    ctx = MagicMock()
    ctx.session.say = AsyncMock()
    ctx.session.aclose = AsyncMock()
    bank_agent = BankAgent()
    with caplog.at_level(logging.INFO, logger="bank-support"):
        await bank_agent._transfer_to_human_impl(
            ctx, "wants to move 9000 from savings", "One moment."
        )
    assert "9000" not in caplog.text
