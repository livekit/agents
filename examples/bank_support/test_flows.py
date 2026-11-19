"""The bank flow: the fake bank, the two advisors, and the contract table's shape."""

from __future__ import annotations

import json

import pytest
from bank_db import FakeBank
from contract import CallState
from flows.bank import CONTRACT, FLOWS, dispute_total, transfer_check


def test_the_seeded_bank_has_a_verifiable_customer():
    bank = FakeBank.seeded()
    customer = bank.verify("4155550123", "1980-04-02")
    assert customer is not None and customer.name == "Dana Price"
    assert bank.verify("4155550123", "1999-01-01") is None
    accounts = bank.accounts(customer.id)
    assert {a.kind for a in accounts} == {"checking", "savings"}
    card = bank.card(customer.id, "4421")
    assert card is not None and card.status == "active"


def test_block_and_unblock_are_a_pair():
    bank = FakeBank.seeded()
    card = bank.card("c_1", "4421")
    bank.block_card(card.id, "lost")
    assert bank.card("c_1", "4421").status == "blocked"
    bank.unblock_card(card.id)
    assert bank.card("c_1", "4421").status == "active"


def test_transfer_check_applies_the_balance_and_the_daily_limit():
    bank = FakeBank.seeded()
    checking = next(a for a in bank.accounts("c_1") if a.kind == "checking")
    ok = transfer_check(bank, checking.id, 500.0)
    assert ok["allowed"] is True and ok["amount"] == 500.0 and ok["fee"] == 0.0
    too_much = transfer_check(bank, checking.id, checking.balance + 1)
    assert too_much["allowed"] is False and "balance" in too_much["say"]
    over_limit = transfer_check(bank, checking.id, bank.DAILY_LIMIT + 1)
    assert over_limit["allowed"] is False and "daily limit" in over_limit["say"]


def test_dispute_total_sums_only_known_transactions():
    bank = FakeBank.seeded()
    checking = next(a for a in bank.accounts("c_1") if a.kind == "checking")
    txs = bank.transactions(checking.id, 30)
    out = dispute_total(bank, [txs[0].id, txs[1].id, "tx_nope"])
    assert out["total"] == pytest.approx(txs[0].amount + txs[1].amount)
    assert out["unknown"] == ["tx_nope"]


def test_the_contract_names_every_write_and_the_end_call():
    writes = {name for name, rule in CONTRACT.rules.items() if rule.write}
    assert writes == {
        "block_card",
        "unblock_card",
        "order_replacement",
        "open_dispute",
        "transfer_funds",
        "add_travel_notice",
    }
    assert CONTRACT.rules["end_call"].requires_closing_question
    assert CONTRACT.rules["transfer_funds"].after == ["get_accounts", "transfer_check"]
    assert all(flow.writes for flow in FLOWS.values())


def test_a_transfer_argument_must_match_the_advisors_figure():
    s = CallState()
    s.record(
        "verify_identity",
        {"phone": "4155550123", "date_of_birth": "1980-04-02"},
        json.dumps({"customer_id": "c_1"}),
    )
    s.record(
        "get_accounts",
        {"customer_id": "c_1"},
        json.dumps({"accounts": [{"id": "acc_1"}, {"id": "acc_2"}]}),
    )
    s.record(
        "transfer_check",
        {"from_account_id": "acc_1", "amount": 500.0},
        json.dumps({"allowed": True, "amount": 500.0}),
    )
    assert (
        CONTRACT.check(
            "transfer_funds",
            {"from_account_id": "acc_1", "to_account_id": "acc_2", "amount": 500.0},
            s,
            [],
        )
        is None
    )
    why = CONTRACT.check(
        "transfer_funds",
        {"from_account_id": "acc_1", "to_account_id": "acc_2", "amount": 600.0},
        s,
        [],
    )
    assert why.startswith("NOT DONE") and "transfer_check" in why


def test_a_replacement_needs_a_blocked_card_now_not_a_block_in_this_call():
    import asyncio
    from types import SimpleNamespace

    from flows.bank import order_replacement, unblock_card

    from livekit.agents import ToolError

    bank = FakeBank.seeded()
    card = bank.card("c_1", "4421")
    ctx = SimpleNamespace(userdata=SimpleNamespace(bank=bank, customer_id="c_1"))
    with pytest.raises(ToolError):
        asyncio.run(order_replacement(ctx, card.id))
    with pytest.raises(ToolError):
        asyncio.run(unblock_card(ctx, card.id))
    bank.block_card(card.id, "lost")  # blocked on an earlier call
    assert json.loads(asyncio.run(order_replacement(ctx, card.id)))["card_id"] == card.id
    assert CONTRACT.rules["order_replacement"].after == []
    assert CONTRACT.rules["unblock_card"].after == []


def test_reads_use_the_verified_customer_not_an_argument():
    import asyncio
    import inspect
    from types import SimpleNamespace

    from flows.bank import get_accounts, get_card, verify_identity

    from livekit.agents import ToolError

    assert "customer_id" not in inspect.signature(get_accounts).parameters
    assert "customer_id" not in inspect.signature(get_card).parameters
    ctx = SimpleNamespace(userdata=SimpleNamespace(bank=FakeBank.seeded(), customer_id=None))
    with pytest.raises(ToolError):
        asyncio.run(get_accounts(ctx))
    asyncio.run(verify_identity(ctx, "4155550123", "1980-04-02"))
    assert ctx.userdata.customer_id == "c_1"
    assert json.loads(asyncio.run(get_card(ctx, "4421")))["last_four"] == "4421"


def test_a_dispute_needs_only_the_transactions():
    import asyncio
    import inspect
    from types import SimpleNamespace

    from flows.bank import open_dispute

    assert "card_id" not in inspect.signature(open_dispute).parameters
    assert "card_id" not in CONTRACT.rules["open_dispute"].needs
    bank = FakeBank.seeded()
    checking = next(a for a in bank.accounts("c_1") if a.kind == "checking")
    ids = [t.id for t in bank.transactions(checking.id, 30)[:2]]
    ctx = SimpleNamespace(userdata=SimpleNamespace(bank=bank, customer_id="c_1"))
    asyncio.run(open_dispute(ctx, ids, "I do not recognise these"))
    assert len(bank.disputes) == 1 and bank.disputes[0]["transaction_ids"] == ids
