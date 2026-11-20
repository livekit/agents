"""A fake bank: a few customers with accounts, cards and transactions, in memory.

The system of record for the example. Nothing here knows about the agent; the
simulation seeds it from a scenario's userdata and grades the run on its final
state, the same way the frontdesk example grades on its calendar.
"""

from __future__ import annotations

import datetime
import itertools
from dataclasses import dataclass, field


@dataclass
class Customer:
    id: str
    name: str
    phone: str
    date_of_birth: str
    address: str


@dataclass
class Account:
    id: str
    customer_id: str
    kind: str  # checking | savings
    number_last_four: str
    balance: float


@dataclass
class Card:
    id: str
    customer_id: str
    account_id: str
    last_four: str
    status: str = "active"  # active | blocked
    travel_notices: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Transaction:
    id: str
    account_id: str
    date: str
    merchant: str
    amount: float
    disputed: bool = False


class FakeBank:
    DAILY_LIMIT = 2500.0
    TRANSFERS_PER_DAY = 5

    def __init__(self, today: datetime.date | None = None) -> None:
        self.today = today or datetime.date(2026, 6, 12)
        self.customers: dict[str, Customer] = {}
        self._accounts: dict[str, Account] = {}
        self._cards: dict[str, Card] = {}
        self._transactions: dict[str, Transaction] = {}
        self.transfers: list[dict[str, object]] = []
        self.replacements: list[dict[str, str]] = []
        self.disputes: list[dict[str, object]] = []
        self._ids = itertools.count(1)

    # --- seeding ------------------------------------------------------------------------

    @classmethod
    def seeded(cls, today: datetime.date | None = None) -> FakeBank:
        bank = cls(today)
        dana = bank.add_customer(
            "Dana Price", "4155550123", "1980-04-02", "12 Harbor St, Oakland, CA 94607"
        )
        checking = bank.add_account(dana.id, "checking", "7781", 3240.55)
        savings = bank.add_account(dana.id, "savings", "0092", 12800.00)
        bank.add_card(dana.id, checking.id, "4421")
        for days_ago, merchant, amount in (
            (1, "Harbor Grocery", 84.20),
            (3, "Northline Transit", 12.50),
            (5, "Blue Fern Cafe", 23.75),
            (9, "Unknown merchant", 199.99),
            (20, "City Utilities", 143.10),
        ):
            bank.add_transaction(checking.id, days_ago, merchant, amount)
        bank.add_transaction(savings.id, 15, "Interest", -4.12)
        rob = bank.add_customer(
            "Robert Lane", "2025550188", "1990-03-15", "88 Pine Ave, Washington, DC 20001"
        )
        rob_checking = bank.add_account(rob.id, "checking", "3310", 410.00)
        bank.add_card(rob.id, rob_checking.id, "9087")
        bank.add_transaction(rob_checking.id, 2, "Pine Books", 31.00)
        return bank

    def add_customer(self, name: str, phone: str, date_of_birth: str, address: str) -> Customer:
        c = Customer(f"c_{next(self._ids)}", name, phone, date_of_birth, address)
        self.customers[c.id] = c
        return c

    def add_account(self, customer_id: str, kind: str, last_four: str, balance: float) -> Account:
        a = Account(f"acc_{next(self._ids)}", customer_id, kind, last_four, balance)
        self._accounts[a.id] = a
        return a

    def add_card(self, customer_id: str, account_id: str, last_four: str) -> Card:
        c = Card(f"card_{next(self._ids)}", customer_id, account_id, last_four)
        self._cards[c.id] = c
        return c

    def add_transaction(
        self, account_id: str, days_ago: int, merchant: str, amount: float
    ) -> Transaction:
        date = (self.today - datetime.timedelta(days=days_ago)).isoformat()
        t = Transaction(f"tx_{next(self._ids)}", account_id, date, merchant, amount)
        self._transactions[t.id] = t
        return t

    # --- reads --------------------------------------------------------------------------

    def verify(self, phone: str, date_of_birth: str) -> Customer | None:
        for c in self.customers.values():
            if c.phone == phone and c.date_of_birth == date_of_birth:
                return c
        return None

    def accounts(self, customer_id: str) -> list[Account]:
        return [a for a in self._accounts.values() if a.customer_id == customer_id]

    def account(self, account_id: str) -> Account | None:
        return self._accounts.get(account_id)

    def card(self, customer_id: str, last_four: str) -> Card | None:
        for c in self._cards.values():
            if c.customer_id == customer_id and c.last_four == last_four:
                return c
        return None

    def card_by_id(self, card_id: str) -> Card | None:
        return self._cards.get(card_id)

    def transactions(self, account_id: str, days: int) -> list[Transaction]:
        cutoff = (self.today - datetime.timedelta(days=days)).isoformat()
        return sorted(
            (
                t
                for t in self._transactions.values()
                if t.account_id == account_id and t.date >= cutoff
            ),
            key=lambda t: t.date,
            reverse=True,
        )

    def transaction(self, transaction_id: str) -> Transaction | None:
        return self._transactions.get(transaction_id)

    def transferred_today(self, account_id: str) -> float:
        return sum(float(t["amount"]) for t in self.transfers if t["from_account_id"] == account_id)

    # --- writes -------------------------------------------------------------------------

    def block_card(self, card_id: str, reason: str) -> Card:
        card = self._cards[card_id]
        card.status = "blocked"
        return card

    def unblock_card(self, card_id: str) -> Card:
        card = self._cards[card_id]
        card.status = "active"
        return card

    def order_replacement(self, card_id: str) -> dict[str, str]:
        card = self._cards[card_id]
        customer = self.customers[card.customer_id]
        order = {
            "order_id": f"rep_{next(self._ids)}",
            "card_id": card_id,
            "ship_to": customer.address,
            "eta_days": "5",
        }
        self.replacements.append(order)
        return order

    def open_dispute(self, transaction_ids: list[str], reason: str) -> dict[str, object]:
        for tid in transaction_ids:
            self._transactions[tid].disputed = True
        dispute = {
            "dispute_id": f"dsp_{next(self._ids)}",
            "transaction_ids": list(transaction_ids),
            "reason": reason,
        }
        self.disputes.append(dispute)
        return dispute

    def transfer(
        self, from_account_id: str, to_account_id: str, amount: float, fee: float = 0.0
    ) -> dict[str, object]:
        src, dst = self._accounts[from_account_id], self._accounts[to_account_id]
        src.balance = round(src.balance - amount - fee, 2)
        dst.balance = round(dst.balance + amount, 2)
        done = {
            "transfer_id": f"trf_{next(self._ids)}",
            "from_account_id": from_account_id,
            "to_account_id": to_account_id,
            "amount": amount,
            "fee": fee,
        }
        self.transfers.append(done)
        return done

    def add_travel_notice(
        self, card_id: str, start_date: str, end_date: str, countries: list[str]
    ) -> dict[str, str]:
        notice = {
            "notice_id": f"trv_{next(self._ids)}",
            "start_date": start_date,
            "end_date": end_date,
            "countries": ", ".join(countries),
        }
        self._cards[card_id].travel_notices.append(notice)
        return notice
