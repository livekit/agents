"""Mock banking data structures for IVR demonstrations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional


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
class SupportTicket:
    reference: str
    opened_at: str
    status: str
    topic: str


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
    support_tickets: tuple[SupportTicket, ...]


class MockBankService:
    """Read-only mock of a retail banking core used by the IVR example."""

    def __init__(self) -> None:
        self._customers: Mapping[str, CustomerProfile] = self._load_seed_data()

    def _load_seed_data(self) -> Mapping[str, CustomerProfile]:
        jordan_transactions = (
            Transaction("2025-10-08", "PAYROLL DEP - HORIZON TECH", 3250.00),
            Transaction("2025-10-05", "ZELLE TO ALEX R", -120.45),
            Transaction("2025-10-04", "COFFEE ROASTERS", -5.85),
        )

        jordan_savings_transactions = (
            Transaction("2025-09-30", "INT EARNED", 8.91),
            Transaction("2025-09-15", "TRANSFER TO CHECKING", -400.00),
        )

        riley_transactions = (
            Transaction("2025-10-06", "FREELANCE PAYMENT", 1500.00),
            Transaction("2025-10-03", "GROCERY MARKET", -212.12),
            Transaction("2025-09-29", "ELECTRIC UTILITY", -108.44),
        )

        customers = {
            "11111111": CustomerProfile(
                customer_id="53001422",
                pin="1111",
                full_name="Jordan Carter",
                branch_name="Downtown Austin",
                deposit_accounts=(
                    DepositAccount(
                        account_number="031890246",
                        account_type="Checking",
                        balance=4821.37,
                        available_balance=4615.92,
                        interest_rate=0.10,
                        recent_transactions=jordan_transactions,
                    ),
                    DepositAccount(
                        account_number="712450987",
                        account_type="High-Yield Savings",
                        balance=18250.41,
                        available_balance=18250.41,
                        interest_rate=3.30,
                        recent_transactions=jordan_savings_transactions,
                    ),
                ),
                credit_cards=(
                    CreditCard(
                        card_number="4485 1399 2211 0099",
                        product_name="Platinum Travel Rewards",
                        credit_limit=15000.00,
                        statement_balance=2150.76,
                        minimum_due=68.00,
                        payment_due_date="2025-10-18",
                        rewards_earn_rate="3x travel, 2x dining",
                    ),
                ),
                loans=(
                    LoanAccount(
                        loan_id="HOME-88421",
                        loan_type="30-Year Fixed Mortgage",
                        original_principal=420000.00,
                        outstanding_balance=367425.19,
                        interest_rate=3.45,
                        next_payment_due="2025-10-12",
                        monthly_payment=1954.32,
                        autopay_enabled=True,
                    ),
                ),
                rewards=RewardsSummary(
                    tier="Platinum",
                    points_balance=138940,
                    expiring_next_statement=4000,
                    cashback_available=182.55,
                ),
                support_tickets=(
                    SupportTicket(
                        reference="CS-44721",
                        opened_at="2025-10-02",
                        status="Awaiting Customer",
                        topic="Upload proof of homeowners insurance",
                    ),
                ),
            ),
            "28890317": CustomerProfile(
                customer_id="28890317",
                pin="6231",
                full_name="Riley Martinez",
                branch_name="North Loop",
                deposit_accounts=(
                    DepositAccount(
                        account_number="601244555",
                        account_type="Checking",
                        balance=2145.82,
                        available_balance=2012.68,
                        interest_rate=0.05,
                        recent_transactions=riley_transactions,
                    ),
                ),
                credit_cards=(
                    CreditCard(
                        card_number="5317 8810 4410 2256",
                        product_name="Cashback Everyday",
                        credit_limit=8000.00,
                        statement_balance=412.09,
                        minimum_due=25.00,
                        payment_due_date="2025-10-20",
                        rewards_earn_rate="1.5% unlimited cashback",
                    ),
                ),
                loans=(
                    LoanAccount(
                        loan_id="AUTO-22901",
                        loan_type="Auto Loan",
                        original_principal=28000.00,
                        outstanding_balance=18642.77,
                        interest_rate=4.99,
                        next_payment_due="2025-10-10",
                        monthly_payment=415.17,
                        autopay_enabled=False,
                    ),
                    LoanAccount(
                        loan_id="STUDENT-00218",
                        loan_type="Private Student Loan",
                        original_principal=42000.00,
                        outstanding_balance=19880.43,
                        interest_rate=5.25,
                        next_payment_due="2025-10-28",
                        monthly_payment=290.10,
                        autopay_enabled=True,
                    ),
                ),
                rewards=RewardsSummary(
                    tier="Gold",
                    points_balance=4820,
                    expiring_next_statement=0,
                    cashback_available=32.18,
                ),
                support_tickets=(),
            ),
        }

        return MappingProxyType(customers)

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

    # -- Support ---------------------------------------------------------------------

    def list_support_tickets(self, customer_id: str) -> tuple[SupportTicket, ...]:
        return self.get_profile(customer_id).support_tickets


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_transactions(transactions: Iterable[Transaction]) -> str:
    lines: list[str] = []
    for txn in transactions:
        amount = format_currency(txn.amount)
        lines.append(f"{txn.posted_at} — {txn.description} ({amount})")
    return "\n".join(lines) if lines else "No recent activity."
