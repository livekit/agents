from datetime import date

from bank import Bank, Identity
from simulation import grade

TODAY = date(2026, 10, 26)


def caller(card: str) -> Identity:
    return Identity("Sam Ortiz", date(1984, 3, 3), card)


async def verified_bank() -> Bank:
    bank = Bank(lookup_seconds=0)
    await bank.authenticate(caller("1111"))
    return bank


async def test_a_lost_card_that_got_locked_passes():
    bank = await verified_bank()
    await bank.lock_card()
    assert grade(bank, {"verified": True, "card": "locked", "case": False}, TODAY) == []


async def test_saying_the_card_is_locked_without_locking_it_fails():
    assert grade(await verified_bank(), {"card": "locked"}, TODAY) == [
        "the card is active, expected locked"
    ]


async def test_a_lockout_needs_three_failed_checks():
    bank = Bank(lookup_seconds=0)
    for _ in range(2):
        await bank.authenticate(caller("0000"))
    assert grade(bank, {"failed_checks": 3, "locked_out": True}, TODAY) == [
        "2 failed checks, expected 3",
        "never locked the caller out",
    ]


async def test_a_travel_notice_is_graded_on_where_and_how_long():
    bank = await verified_bank()
    await bank.set_travel_notice("Japan", date(2026, 11, 6), TODAY)
    assert grade(bank, {"travel_to": "Japan", "back_in_days": [10, 11]}, TODAY) == []
    assert grade(bank, {"travel_to": "Japan", "back_in_days": 10}, TODAY) == [
        "the travel notice runs 11 days, expected 10"
    ]
