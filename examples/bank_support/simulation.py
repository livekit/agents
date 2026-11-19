"""Simulation glue: seed the fake bank from a scenario's userdata and grade the run on its
final state. The agent code stays production-shaped; everything scenario-specific is here."""

from __future__ import annotations

import datetime

from bank_db import FakeBank

SIMULATION_TODAY = "2026-06-12"


def bank_from_userdata(userdata: dict) -> FakeBank:
    """A seeded bank on the scenario's clock, with optional balance overrides
    (``balances: {checking: 100.0}``) and a ``transferred_today`` amount."""
    today = datetime.date.fromisoformat(userdata.get("today", SIMULATION_TODAY))
    bank = FakeBank.seeded(today)
    for kind, balance in (userdata.get("balances") or {}).items():
        for account in bank.accounts("c_1"):
            if account.kind == kind:
                account.balance = float(balance)
    if (sent := userdata.get("transferred_today")) is not None:
        checking = next(a for a in bank.accounts("c_1") if a.kind == "checking")
        bank.transfers.append(
            {
                "transfer_id": "trf_prior",
                "from_account_id": checking.id,
                "to_account_id": "",
                "amount": float(sent),
            }
        )
    return bank


def grade(bank: FakeBank, expected: dict) -> str | None:
    """None when the bank's final state matches ``expected``; otherwise the reason it does not.

    expected keys: card_status (active|blocked, for the card ending 4421), replacement (bool),
    transfers (list of amounts), disputes (count), travel_notices (count).
    """
    card = bank.card("c_1", "4421")
    if "card_status" in expected and card.status != expected["card_status"]:
        return f"card ending 4421 is {card.status}, expected {expected['card_status']}"
    if "replacement" in expected and bool(bank.replacements) != bool(expected["replacement"]):
        return f"replacement ordered: {bool(bank.replacements)}, expected {expected['replacement']}"
    if "transfers" in expected:
        amounts = [
            round(float(t["amount"]), 2) for t in bank.transfers if t["transfer_id"] != "trf_prior"
        ]
        if amounts != [round(float(a), 2) for a in expected["transfers"]]:
            return f"transfers made: {amounts}, expected {expected['transfers']}"
    if "disputes" in expected and len(bank.disputes) != expected["disputes"]:
        return f"disputes opened: {len(bank.disputes)}, expected {expected['disputes']}"
    if "travel_notices" in expected and len(card.travel_notices) != expected["travel_notices"]:
        return f"travel notices: {len(card.travel_notices)}, expected {expected['travel_notices']}"
    return None
