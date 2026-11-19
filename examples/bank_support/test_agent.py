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
    assert _names(task.tools) == {"get_accounts", "transfer_check", "transfer_funds", "end_request"}
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
