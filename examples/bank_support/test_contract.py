"""Unit tests for the flow contract: one per rule kind, no model and no network."""

from __future__ import annotations

import json

from contract import CallState, Contract, Rule, digits_heard

CONTRACT = Contract(
    {
        "get_accounts": Rule(after=["verify_identity"]),
        "get_transactions": Rule(
            after=["verify_identity"], needs={"account_id": "get_accounts.accounts[].id"}
        ),
        "block_card": Rule(
            write=True,
            after=["verify_identity"],
            needs={"card_id": "get_card.card_id"},
            undo="unblock_card",
        ),
        "unblock_card": Rule(write=True, needs={"card_id": "get_card.card_id"}),
        "transfer_check": Rule(after=["verify_identity"]),
        "transfer_funds": Rule(write=True, needs={"amount": "transfer_check.amount"}),
        "open_dispute": Rule(
            write=True,
            after=["get_transactions"],
            needs={"transaction_ids": "get_transactions.transactions[].id"},
        ),
        "verify_identity": Rule(formats={"phone": (r"^\d{10}$", "ten digits")}),
        "end_call": Rule(
            after_any=["block_card", "open_dispute", "transfer_to_human", "declined"],
            requires_closing_question=True,
        ),
    }
)


def _verified() -> CallState:
    s = CallState()
    s.record(
        "verify_identity",
        {"phone": "4155550123", "date_of_birth": "1980-01-01"},
        json.dumps({"customer_id": "c_1", "name": "Dana Price"}),
    )
    return s


def test_a_write_before_its_prerequisite_is_refused_with_the_next_step():
    s = CallState()
    why = CONTRACT.check("block_card", {"card_id": "card_1", "reason": "lost"}, s, [])
    assert why.startswith("NOT DONE") and "verify_identity" in why


def test_an_argument_must_come_from_the_producers_reply():
    s = _verified()
    why = CONTRACT.check("block_card", {"card_id": "card_9", "reason": "lost"}, s, [])
    assert why.startswith("NOT DONE") and "get_card" in why and "card_id" in why
    s.record(
        "get_card",
        {"customer_id": "c_1", "last_four": "4421"},
        json.dumps({"card_id": "card_1", "last_four": "4421"}),
    )
    assert CONTRACT.check("block_card", {"card_id": "card_1", "reason": "lost"}, s, []) is None
    why = CONTRACT.check("block_card", {"card_id": "card_9", "reason": "lost"}, s, [])
    assert "card_9" in why and "card_1" in why


def test_list_arguments_are_checked_against_a_list_in_the_reply():
    s = _verified()
    s.record(
        "get_accounts",
        {"customer_id": "c_1"},
        json.dumps({"accounts": [{"id": "acc_1"}, {"id": "acc_2"}]}),
    )
    s.record(
        "get_transactions",
        {"account_id": "acc_1", "days": 30},
        json.dumps(
            {"transactions": [{"id": "tx_1", "amount": 20.0}, {"id": "tx_2", "amount": 35.5}]}
        ),
    )
    assert (
        CONTRACT.check(
            "open_dispute",
            {"card_id": "card_1", "transaction_ids": ["tx_1", "tx_2"], "reason": "unknown"},
            s,
            [],
        )
        is None
    )
    why = CONTRACT.check(
        "open_dispute",
        {"card_id": "card_1", "transaction_ids": ["tx_1", "tx_7"], "reason": "unknown"},
        s,
        [],
    )
    assert "tx_7" in why and "get_transactions" in why


def test_a_format_rule_refuses_a_wrong_shape_and_asks_for_a_read_back():
    why = CONTRACT.check(
        "verify_identity", {"phone": "415555012", "date_of_birth": "1980-01-01"}, CallState(), []
    )
    assert why.startswith("NOT DONE") and "ten digits" in why and "read" in why.lower()
    assert (
        CONTRACT.check(
            "verify_identity",
            {"phone": "4155550123", "date_of_birth": "1980-01-01"},
            CallState(),
            [],
        )
        is None
    )


def test_a_successful_write_is_not_repeated():
    s = _verified()
    s.record(
        "get_card", {"customer_id": "c_1", "last_four": "4421"}, json.dumps({"card_id": "card_1"})
    )
    s.record(
        "block_card", {"card_id": "card_1", "reason": "lost"}, json.dumps({"status": "blocked"})
    )
    why = CONTRACT.check("block_card", {"card_id": "card_1", "reason": "lost"}, s, [])
    assert why.startswith("ALREADY DONE")


def test_an_undo_reads_the_cards_current_state_not_this_calls_history():
    s = _verified()
    s.record("get_card", {"last_four": "4421"}, json.dumps({"card_id": "card_1"}))
    assert CONTRACT.check("unblock_card", {"card_id": "card_1"}, s, []) is None


def test_a_repeated_write_runs_again_after_a_fresh_reply_from_its_producer():
    s = _verified()
    s.record("transfer_check", {"amount": 50.0}, json.dumps({"amount": 50.0}))
    s.record("transfer_funds", {"amount": 50.0}, json.dumps({"transfer_id": "t1"}))
    assert CONTRACT.check("transfer_funds", {"amount": 50.0}, s, []).startswith("ALREADY DONE")
    s.record("transfer_check", {"amount": 50.0}, json.dumps({"amount": 50.0}))
    assert CONTRACT.check("transfer_funds", {"amount": 50.0}, s, []) is None


def test_a_read_reply_says_where_its_values_go():
    out = CONTRACT.narrate(
        "get_card", {"last_four": "4421"}, json.dumps({"card_id": "card_1", "last_four": "4421"})
    )
    assert (
        out.startswith('{"card_id"')
        and "card_1" in out.split("\n\n", 1)[1]
        and "block_card" in out
        and "card_id" in out
    )


def test_a_write_reply_names_its_undo():
    out = CONTRACT.narrate("block_card", {"card_id": "card_1"}, json.dumps({"status": "blocked"}))
    assert "unblock_card" in out


ASKED = ("assistant", "Your card is blocked. Can I help with anything else?")


def test_the_call_does_not_end_before_any_work():
    s = _verified()
    history = [ASKED, ("user", "No, that's all.")]
    why = CONTRACT.check("end_call", {"caller_reply": "No, that's all."}, s, history)
    assert why.startswith("NOT ENDED")


def test_the_call_ends_after_work_a_question_and_the_callers_quoted_answer():
    s = _verified()
    s.record("get_card", {"last_four": "4421"}, json.dumps({"card_id": "card_1"}))
    s.record("block_card", {"card_id": "card_1", "reason": "lost"}, json.dumps({}))
    history = [ASKED, ("user", "No, that's all, thanks.")]
    assert CONTRACT.check("end_call", {"caller_reply": "no, that's all"}, s, history) is None


def test_the_call_does_not_end_when_the_agent_did_not_ask_a_question():
    s = _verified()
    s.record("block_card", {"card_id": "card_1", "reason": "lost"}, json.dumps({}))
    history = [("assistant", "Your card is blocked."), ("user", "Okay.")]
    why = CONTRACT.check("end_call", {"caller_reply": "Okay."}, s, history)
    assert why.startswith("NOT ENDED") and "ask" in why.lower()


def test_the_caller_reply_must_quote_the_callers_last_turn():
    s = _verified()
    s.record("block_card", {"card_id": "card_1", "reason": "lost"}, json.dumps({}))
    history = [ASKED, ("user", "I don't know, maybe check my savings now")]
    why = CONTRACT.check("end_call", {"caller_reply": "No, that's all."}, s, history)
    assert why.startswith("NOT ENDED") and "caller_reply" in why


def test_a_hand_off_counts_as_finished_work():
    s = CallState()
    s.record(
        "transfer_to_human", {"summary": "wants a higher card limit"}, "A colleague will take it."
    )
    history = [
        ("assistant", "A colleague will take it from here. Anything else?"),
        ("user", "No, bye."),
    ]
    assert CONTRACT.check("end_call", {"caller_reply": "No, bye."}, s, history) is None


def test_digits_heard_from_spoken_numbers():
    assert digits_heard("my number is four one five five five five zero one two three") == [
        "4155550123"
    ]
    assert digits_heard("the last four are four four two one") == ["4421"]
    assert digits_heard("it's two zero two, five five five, oh one eight eight") == ["2025550188"]
    assert digits_heard("I'd like to block my card") == []


# --- through LiveKit's own argument binding ------------------------------------------------
# LiveKit binds a tool call's arguments positionally, so these tests call the guarded tool
# the way the framework does instead of calling Contract.check directly.

from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from livekit.agents import RunContext, function_tool  # noqa: E402
from livekit.agents.llm.utils import prepare_function_arguments  # noqa: E402

BOUND = Contract(
    {
        "lookup": Rule(needs={"account_id": "accounts.ids[]"}),
        "end_call": Rule(requires_closing_question=True),
    }
)


async def lookup(ctx: RunContext, account_id: str) -> str:
    return '{"ok": true}'


async def end_call(ctx: RunContext, caller_reply: str) -> str:
    return "Ending the call."


def _ctx(history: list[tuple[str, str]] = ()) -> RunContext:
    ctx = MagicMock(spec=RunContext)
    ctx.userdata = SimpleNamespace(state=CallState())
    ctx.session.history.items = [SimpleNamespace(role=r, text_content=t) for r, t in history]
    ctx.session.user_state = "listening"
    return ctx


async def _call(fn, ctx: RunContext, arguments: dict) -> str:
    tool = function_tool(BOUND.guarded(fn))
    args, kwargs = prepare_function_arguments(
        fnc=tool, json_arguments=json.dumps(arguments), call_ctx=ctx
    )
    return await tool(*args, **kwargs)


async def test_a_bound_argument_is_checked_against_the_producers_reply():
    ctx = _ctx()
    ctx.userdata.state.record("accounts", {}, json.dumps({"ids": ["acc_2", "acc_3"]}))
    reply = await _call(lookup, ctx, {"account_id": "unknown"})
    assert reply.startswith("NOT DONE") and "acc_2" in reply
    assert (await _call(lookup, ctx, {"account_id": "acc_2"})).startswith('{"ok": true}')


async def test_a_bound_caller_reply_ends_the_call():
    ctx = _ctx(
        [("assistant", "Anything else I can help with?"), ("user", "No, that’s all. Thank you.")]
    )
    assert await _call(end_call, ctx, {"caller_reply": "No, that’s all. Thank you."}) == (
        "Ending the call."
    )


async def test_the_same_refused_call_twice_tells_the_model_to_ask_the_caller():
    ctx = _ctx()
    ctx.userdata.state.record("accounts", {}, json.dumps({"ids": ["acc_2", "acc_3"]}))
    first = await _call(lookup, ctx, {"account_id": "unknown"})
    second = await _call(lookup, ctx, {"account_id": "unknown"})
    assert "ask the caller" not in first.lower()
    assert second.startswith("NOT DONE") and "ask the caller" in second.lower()
