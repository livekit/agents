from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from mock_bank_service import (  # noqa: E402
    MockBankService,
    RewardsSummary,
    format_currency,
    format_transactions,
)


@pytest.fixture()
def service() -> MockBankService:
    return MockBankService()


def test_authentication(service: MockBankService) -> None:
    ids = service.list_customer_ids()
    assert "10000001" in ids

    assert service.customer_exists("10000001")
    assert not service.customer_exists("00000000")

    assert service.authenticate("10000001", "0000")
    assert not service.authenticate("10000001", "9999")


def test_deposit_accounts(service: MockBankService) -> None:
    accounts = service.list_deposit_accounts("10000001")
    assert len(accounts) == 2
    checking = service.find_deposit_account("10000001", "031890246")
    assert checking is not None
    assert checking.account_type == "Checking"

    total = service.calculate_total_deposits("10000001")
    assert total == pytest.approx(4821.37 + 18250.41)


def test_credit_cards(service: MockBankService) -> None:
    cards = service.list_credit_cards("20000002")
    assert len(cards) == 1
    card = cards[0]
    assert card.minimum_due == pytest.approx(25.00)

    total_balance = service.calculate_total_card_balance("20000002")
    assert total_balance == pytest.approx(412.09)


def test_loans(service: MockBankService) -> None:
    loans = service.list_loans("20000002")
    assert len(loans) == 2
    assert loans[0].loan_type == "Auto Loan"

    outstanding = service.calculate_total_loan_balance("20000002")
    assert outstanding == pytest.approx(18642.77 + 19880.43)

    payments = service.upcoming_payments("20000002")
    assert len(payments) == 2

    assert payments[0][0] == "AUTO-22901"
    assert payments[0][2] == "2025-10-10"
    assert payments[0][1] == pytest.approx(415.17)

    assert payments[1][0] == "STUDENT-00218"
    assert payments[1][2] == "2025-10-28"
    assert payments[1][1] == pytest.approx(290.10)


def test_rewards_summary(service: MockBankService) -> None:
    rewards = service.get_rewards("10000001")
    assert isinstance(rewards, RewardsSummary)
    assert rewards.tier == "Platinum"
    assert rewards.points_balance == 138940


def test_format_helpers(service: MockBankService) -> None:
    currency = format_currency(1234.5)
    assert currency == "$1,234.50"

    account = service.find_deposit_account("10000001", "031890246")
    assert account is not None
    formatted = format_transactions(account.recent_transactions)

    snapshot_path = Path(__file__).with_name("__snapshots__") / "format_transactions.snap"
    expected = snapshot_path.read_text(encoding="utf-8").rstrip("\n")
    assert formatted == expected
