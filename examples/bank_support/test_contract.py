"""Unit tests for the flow contract: one per rule kind, no model and no network."""

from __future__ import annotations

import json

from contract import CallState, Contract, Rule, closing_question_declined, digits_heard

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
        "unblock_card": Rule(write=True, after=["block_card"]),
        "open_dispute": Rule(
            write=True,
            after=["get_transactions"],
            needs={"transaction_ids": "get_transactions.transactions[].id"},
        ),
        "verify_identity": Rule(formats={"phone": (r"^\d{10}$", "ten digits")}),
        "end_call": Rule(
            after_any=["block_card", "open_dispute", "declined"], requires_closing_question=True
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


def test_an_undo_needs_the_action_it_undoes():
    s = _verified()
    assert "block_card" in CONTRACT.check("unblock_card", {"card_id": "card_1"}, s, [])


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


def test_the_call_ends_only_after_the_work_and_the_closing_question():
    s = _verified()
    history = []
    why = CONTRACT.check("end_call", {"closing_message": "Bye"}, s, history)
    assert why.startswith("NOT ENDED") and "anything else" in why.lower()
    s.record("get_card", {"last_four": "4421"}, json.dumps({"card_id": "card_1"}))
    s.record(
        "block_card", {"card_id": "card_1", "reason": "lost"}, json.dumps({"status": "blocked"})
    )
    history = [
        ("assistant", "Your card is blocked. Is there anything else I can help with?"),
        ("user", "No, that's all, thanks."),
    ]
    assert CONTRACT.check("end_call", {"closing_message": "Bye"}, s, history) is None
    history = [
        ("assistant", "Your card is blocked. Is there anything else I can help with?"),
        ("user", "Yes, one more thing."),
    ]
    assert CONTRACT.check("end_call", {"closing_message": "Bye"}, s, history).startswith(
        "NOT ENDED"
    )


def test_closing_question_detection():
    assert closing_question_declined(
        [("assistant", "Anything else today?"), ("user", "nope, that's it")]
    )
    assert not closing_question_declined(
        [("assistant", "Anything else today?"), ("user", "actually yes")]
    )
    assert not closing_question_declined([("user", "no")])


def test_digits_heard_from_spoken_numbers():
    assert digits_heard("my number is four one five five five five zero one two three") == [
        "4155550123"
    ]
    assert digits_heard("the last four are four four two one") == ["4421"]
    assert digits_heard("it's two zero two, five five five, oh one eight eight") == ["2025550188"]
    assert digits_heard("I'd like to block my card") == []
