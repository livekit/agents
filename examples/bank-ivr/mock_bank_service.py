"""Mock banking data structures for IVR demonstrations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional


@dataclass(frozen=True)
class Transaction:
    posted_at: str
    description: str
    amount: float


@dataclass(frozen=True)
class DepositAccount:
    account_number: str
    account_type: str
    balance: float
    available_balance: float
    interest_rate: float
    recent_transactions: tuple[Transaction, ...]


@dataclass(frozen=True)
class CreditCard:
    card_number: str
    product_name: str
    credit_limit: float
    statement_balance: float
    minimum_due: float
    payment_due_date: str
    rewards_earn_rate: str


@dataclass(frozen=True)
class LoanAccount:
    loan_id: str
    loan_type: str
    original_principal: float
    outstanding_balance: float
    interest_rate: float
    next_payment_due: str
    monthly_payment: float
    autopay_enabled: bool


@dataclass(frozen=True)
class RewardsSummary:
    tier: str
    points_balance: int
    expiring_next_statement: int
    cashback_available: float


@dataclass(frozen=True)
class CustomerProfile:
    customer_id: str
    pin: str
    full_name: str
    branch_name: str
    deposit_accounts: tuple[DepositAccount, ...]
    credit_cards: tuple[CreditCard, ...]
    loans: tuple[LoanAccount, ...]
    rewards: RewardsSummary


class MockBankService:
    """Read-only mock of a retail banking core used by the IVR example."""

    def __init__(self) -> None:
        self._customers: Mapping[str, CustomerProfile] = self._load_seed_data()

    def _load_seed_data(self) -> Mapping[str, CustomerProfile]:
        data_path = Path(__file__).with_name("data.json")
        try:
            with data_path.open(encoding="utf-8") as infile:
                data = json.load(infile)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Mock bank data file not found: {data_path}") from exc

        customers: dict[str, CustomerProfile] = {}
        for customer_payload in data.get("customers", []):
            profile = self._build_customer_profile(customer_payload)
            customers[profile.customer_id] = profile

        return MappingProxyType(customers)

    def _build_customer_profile(self, data: Mapping[str, Any]) -> CustomerProfile:
        deposit_accounts = tuple(
            self._build_deposit_account(account_data)
            for account_data in data.get("deposit_accounts", [])
        )

        credit_cards = tuple(
            self._build_credit_card(card_data) for card_data in data.get("credit_cards", [])
        )

        loans = tuple(self._build_loan_account(loan_data) for loan_data in data.get("loans", []))

        rewards_payload = data.get("rewards") or {}

        return CustomerProfile(
            customer_id=data["customer_id"],
            pin=data["pin"],
            full_name=data["full_name"],
            branch_name=data["branch_name"],
            deposit_accounts=deposit_accounts,
            credit_cards=credit_cards,
            loans=loans,
            rewards=RewardsSummary(
                tier=rewards_payload.get("tier", "Unknown"),
                points_balance=rewards_payload.get("points_balance", 0),
                expiring_next_statement=rewards_payload.get("expiring_next_statement", 0),
                cashback_available=rewards_payload.get("cashback_available", 0.0),
            ),
        )

    def _build_deposit_account(self, data: Mapping[str, Any]) -> DepositAccount:
        transactions = tuple(
            self._build_transaction(txn) for txn in data.get("recent_transactions", [])
        )

        return DepositAccount(
            account_number=data["account_number"],
            account_type=data["account_type"],
            balance=data["balance"],
            available_balance=data["available_balance"],
            interest_rate=data["interest_rate"],
            recent_transactions=transactions,
        )

    def _build_credit_card(self, data: Mapping[str, Any]) -> CreditCard:
        return CreditCard(
            card_number=data["card_number"],
            product_name=data["product_name"],
            credit_limit=data["credit_limit"],
            statement_balance=data["statement_balance"],
            minimum_due=data["minimum_due"],
            payment_due_date=data["payment_due_date"],
            rewards_earn_rate=data["rewards_earn_rate"],
        )

    def _build_loan_account(self, data: Mapping[str, Any]) -> LoanAccount:
        return LoanAccount(
            loan_id=data["loan_id"],
            loan_type=data["loan_type"],
            original_principal=data["original_principal"],
            outstanding_balance=data["outstanding_balance"],
            interest_rate=data["interest_rate"],
            next_payment_due=data["next_payment_due"],
            monthly_payment=data["monthly_payment"],
            autopay_enabled=data["autopay_enabled"],
        )

    def _build_transaction(self, data: Mapping[str, Any]) -> Transaction:
        return Transaction(
            posted_at=data["posted_at"],
            description=data["description"],
            amount=data["amount"],
        )

    # -- Authentication --------------------------------------------------------------

    def list_customer_ids(self) -> tuple[str, ...]:
        return tuple(self._customers.keys())

    def customer_exists(self, customer_id: str) -> bool:
        return customer_id in self._customers

    def authenticate(self, customer_id: str, pin: str) -> bool:
        profile = self._customers.get(customer_id)
        return bool(profile and profile.pin == pin)

    def get_profile(self, customer_id: str) -> CustomerProfile:
        try:
            return self._customers[customer_id]
        except KeyError as exc:
            raise KeyError(f"Unknown customer {customer_id}") from exc

    # -- Deposit accounts ------------------------------------------------------------

    def list_deposit_accounts(self, customer_id: str) -> tuple[DepositAccount, ...]:
        return self.get_profile(customer_id).deposit_accounts

    def find_deposit_account(
        self, customer_id: str, account_number: str
    ) -> Optional[DepositAccount]:  # noqa: UP007
        for acct in self.list_deposit_accounts(customer_id):
            if acct.account_number == account_number:
                return acct
        return None

    def calculate_total_deposits(self, customer_id: str) -> float:
        return sum(acct.balance for acct in self.list_deposit_accounts(customer_id))

    # -- Credit cards ----------------------------------------------------------------

    def list_credit_cards(self, customer_id: str) -> tuple[CreditCard, ...]:
        return self.get_profile(customer_id).credit_cards

    def calculate_total_card_balance(self, customer_id: str) -> float:
        return sum(card.statement_balance for card in self.list_credit_cards(customer_id))

    # -- Loans -----------------------------------------------------------------------

    def list_loans(self, customer_id: str) -> tuple[LoanAccount, ...]:
        return self.get_profile(customer_id).loans

    def calculate_total_loan_balance(self, customer_id: str) -> float:
        return sum(loan.outstanding_balance for loan in self.list_loans(customer_id))

    def upcoming_payments(self, customer_id: str) -> tuple[tuple[str, float, str], ...]:
        """Return tuples of (loan_id, amount, due_date)."""

        return tuple(
            (loan.loan_id, loan.monthly_payment, loan.next_payment_due)
            for loan in self.list_loans(customer_id)
        )

    # -- Rewards ---------------------------------------------------------------------

    def get_rewards(self, customer_id: str) -> RewardsSummary:
        return self.get_profile(customer_id).rewards


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_transactions(transactions: Iterable[Transaction]) -> str:
    lines: list[str] = []
    for txn in transactions:
        amount = format_currency(txn.amount)
        lines.append(f"{txn.posted_at} — {txn.description} ({amount})")
    return "\n".join(lines) if lines else "No recent activity."
