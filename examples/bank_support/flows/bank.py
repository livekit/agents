"""The customer's part: the tools as the bank exposes them, the contract table, and the
two pieces of policy arithmetic the model should not do in its head.

To adapt the example to another business, replace this file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from bank_db import FakeBank
from contract import Contract, Rule

from livekit.agents import RunContext, ToolError

ISO_DATE = (r"^\d{4}-\d{2}-\d{2}$", "a date as year-month-day, for example 2026-07-04")
TEN_DIGITS = (r"^\d{10}$", "ten digits")
FOUR_DIGITS = (r"^\d{4}$", "four digits")

# --- the contract table --------------------------------------------------------------

CONTRACT = Contract(
    {
        "verify_identity": Rule(formats={"phone": TEN_DIGITS, "date_of_birth": ISO_DATE}),
        "get_accounts": Rule(after=["verify_identity"]),
        "get_card": Rule(after=["verify_identity"], formats={"last_four": FOUR_DIGITS}),
        "get_transactions": Rule(
            after=["verify_identity"], needs={"account_id": "get_accounts.accounts[].id"}
        ),
        "transfer_check": Rule(
            after=["get_accounts"], needs={"from_account_id": "get_accounts.accounts[].id"}
        ),
        "dispute_total": Rule(
            after=["get_transactions"],
            needs={"transaction_ids": "get_transactions.transactions[].id"},
        ),
        "block_card": Rule(
            write=True,
            confirm=True,
            after=["verify_identity"],
            needs={"card_id": "get_card.card_id"},
            undo="unblock_card",
        ),
        # Card state, not call history, decides whether these can run: the tools refuse
        # an active card, so a card blocked on an earlier call can be unblocked or replaced.
        "unblock_card": Rule(write=True, confirm=True, needs={"card_id": "get_card.card_id"}),
        "order_replacement": Rule(write=True, confirm=True, needs={"card_id": "get_card.card_id"}),
        "open_dispute": Rule(
            write=True,
            confirm=True,
            after=["get_transactions"],
            needs={"transaction_ids": "get_transactions.transactions[].id"},
        ),
        "transfer_funds": Rule(
            write=True,
            confirm=True,
            after=["get_accounts", "transfer_check"],
            needs={
                "from_account_id": "get_accounts.accounts[].id",
                "to_account_id": "get_accounts.accounts[].id",
                "amount": "transfer_check.amount",
            },
        ),
        "add_travel_notice": Rule(
            write=True,
            confirm=True,
            after=["get_card"],
            needs={"card_id": "get_card.card_id"},
            formats={"start_date": ISO_DATE, "end_date": ISO_DATE},
        ),
        "transfer_to_human": Rule(),
        # Any handled request counts as work: a write, an answered question, a hand-off,
        # or a request the caller dropped.
        "end_call": Rule(
            after_any=[
                "block_card",
                "unblock_card",
                "order_replacement",
                "open_dispute",
                "transfer_funds",
                "add_travel_notice",
                "get_accounts",
                "get_card",
                "get_transactions",
                "transfer_to_human",
                "declined",
            ],
            requires_closing_question=True,
        ),
    }
)


# --- the arithmetic ---------------------------------------------------------------------

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + _words(n % 100) if n % 100 else "")
    if n < 1_000_000:
        return _words(n // 1000) + " thousand" + (" " + _words(n % 1000) if n % 1000 else "")
    return (
        _words(n // 1_000_000) + " million" + (" " + _words(n % 1_000_000) if n % 1_000_000 else "")
    )


def spoken_dollars(amount: float) -> str:
    """Money as a caller hears it: tool sentences are copied into speech, so they carry
    no decimals ("223.74" -> "two hundred twenty-three dollars and seventy-four cents")."""
    if amount < 0:
        return "minus " + spoken_dollars(-amount)
    cents_total = round(amount * 100)
    dollars, cents = divmod(cents_total, 100)
    parts = []
    if dollars or not cents:
        parts.append(f"{_words(dollars)} dollar{'s' if dollars != 1 else ''}")
    if cents:
        parts.append(f"{_words(cents)} cent{'s' if cents != 1 else ''}")
    return " and ".join(parts)


def transfer_check(bank: FakeBank, from_account_id: str, amount: float) -> dict[str, Any]:
    """Whether a transfer of ``amount`` can leave ``from_account_id`` today, and what it will cost."""
    if amount <= 0:
        return {
            "allowed": False,
            "requested_amount": amount,
            "fee": 0.0,
            "say": "A transfer amount must be more than zero.",
        }
    account = bank.account(from_account_id)
    if account is None:
        return {
            "allowed": False,
            "requested_amount": amount,
            "fee": 0.0,
            "say": "That account was not found; check the account id with get_accounts.",
        }
    fee = 0.0 if account.kind == "checking" else 2.0
    sent_today = bank.transferred_today(from_account_id)
    remaining_limit = round(bank.DAILY_LIMIT - sent_today, 2)
    total = round(amount + fee, 2)
    if total > account.balance:
        return {
            "allowed": False,
            "requested_amount": amount,
            "fee": fee,
            "available": account.balance,
            "say": f"The available balance on that account is {spoken_dollars(account.balance)}, which does not cover {spoken_dollars(total)}; offer a smaller amount or another account.",
        }
    if amount > remaining_limit:
        return {
            "allowed": False,
            "requested_amount": amount,
            "fee": fee,
            "limit_left_today": remaining_limit,
            "say": f"The daily limit leaves {spoken_dollars(remaining_limit)} today, which does not cover {spoken_dollars(amount)}. Offer that amount today. The caller can call again tomorrow for the rest; you cannot schedule a transfer.",
        }
    say = (
        f"A transfer of {spoken_dollars(amount)} is allowed"
        + (f" with a fee of {spoken_dollars(fee)} from savings" if fee else " with no fee")
        + f"; {spoken_dollars(remaining_limit - amount)} of today's limit would remain."
    )
    return {
        "allowed": True,
        "amount": amount,
        "fee": fee,
        "limit_left_after_transfer": round(remaining_limit - amount, 2),
        "say": say,
    }


def dispute_total(bank: FakeBank, transaction_ids: list[str]) -> dict[str, Any]:
    """The total of the disputed transactions, and the ids that are not on the account."""
    known = [bank.transaction(t) for t in transaction_ids]
    unknown = [t for t, tx in zip(transaction_ids, known, strict=False) if tx is None]
    total = round(sum(tx.amount for tx in known if tx is not None), 2)
    lines = [
        f"{tx.merchant} on {tx.date} for {spoken_dollars(tx.amount)}"
        for tx in known
        if tx is not None
    ]
    say = (
        f"{len(lines)} transaction(s) totalling {spoken_dollars(total)}: " + "; ".join(lines) + "."
    )
    if unknown:
        say += f" Not on this account: {', '.join(unknown)}."
    return {"total": total, "count": len(lines), "unknown": unknown, "say": say}


# --- the tools, exactly as the bank exposes them -------------------------------------------


def _bank(ctx: RunContext) -> FakeBank:
    return ctx.userdata.bank


def _customer_id(ctx: RunContext) -> str:
    """The verified caller. Identity is session state, never a model argument: a model
    asked to carry an id across a task boundary may invent one."""
    customer_id = ctx.userdata.customer_id
    if customer_id is None:
        raise ToolError("The caller is not verified yet. Call verify_identity first.")
    return customer_id


async def verify_identity(ctx: RunContext, phone: str, date_of_birth: str) -> str:
    """Verify the caller against the account holder on file. Call this before anything else.

    Args:
        phone: The caller's phone number on file, ten digits.
        date_of_birth: The caller's date of birth as year-month-day.
    """
    customer = _bank(ctx).verify(phone, date_of_birth)
    if customer is None:
        raise ToolError(
            f"No account holder matches phone {phone} and date of birth {date_of_birth}. Read both back to the caller and ask them to repeat."
        )
    ctx.userdata.customer_id = customer.id
    return json.dumps({"name": customer.name})


async def get_accounts(ctx: RunContext) -> str:
    """The verified caller's accounts with their type, last four digits and balance."""
    accounts = _bank(ctx).accounts(_customer_id(ctx))
    return json.dumps(
        {
            "accounts": [
                {"id": a.id, "type": a.kind, "last_four": a.number_last_four, "balance": a.balance}
                for a in accounts
            ]
        }
    )


async def get_card(ctx: RunContext, last_four: str) -> str:
    """The verified caller's card ending in the four digits they give.

    Args:
        last_four: The last four digits of the card number.
    """
    card = _bank(ctx).card(_customer_id(ctx), last_four)
    if card is None:
        raise ToolError(
            f"No card ending in {last_four} on this customer. Read the four digits back and ask again."
        )
    return json.dumps(
        {
            "card_id": card.id,
            "last_four": card.last_four,
            "status": card.status,
            "account_id": card.account_id,
        }
    )


async def get_transactions(ctx: RunContext, account_id: str, days: int = 30) -> str:
    """Recent transactions on an account, newest first.

    Args:
        account_id: An account id from get_accounts. If the caller has not said which
            account, ask them.
        days: How many days back to list, 30 by default.
    """
    txs = _bank(ctx).transactions(account_id, days)
    return json.dumps(
        {
            "transactions": [
                {
                    "id": t.id,
                    "date": t.date,
                    "merchant": t.merchant,
                    "amount": t.amount,
                    "disputed": t.disputed,
                }
                for t in txs
            ]
        }
    )


async def transfer_check_tool(ctx: RunContext, from_account_id: str, amount: float) -> str:
    """Whether a transfer is allowed today and what it costs: balance, daily limit and fee.
    Call this before transfer_funds and repeat its sentence to the caller.

    Args:
        from_account_id: The account the money leaves, from get_accounts.
        amount: The amount in dollars.
    """
    return json.dumps(transfer_check(_bank(ctx), from_account_id, amount))


transfer_check_tool.__name__ = "transfer_check"


async def dispute_total_tool(ctx: RunContext, transaction_ids: list[str]) -> str:
    """The total and the details of the transactions the caller wants to dispute.
    Call this before open_dispute and read its sentence back to the caller.

    Args:
        transaction_ids: Transaction ids from get_transactions.
    """
    return json.dumps(dispute_total(_bank(ctx), transaction_ids))


dispute_total_tool.__name__ = "dispute_total"


async def block_card(ctx: RunContext, card_id: str, reason: str) -> str:
    """Block a card so it cannot be used.

    Args:
        card_id: The card id from get_card.
        reason: Why, in the caller's words: lost, stolen, or suspected fraud.
    """
    card = _bank(ctx).block_card(card_id, reason)
    return json.dumps({"card_id": card.id, "status": card.status, "reason": reason})


async def unblock_card(ctx: RunContext, card_id: str) -> str:
    """Take a block back: the card works again.

    Args:
        card_id: The card id that was blocked.
    """
    bank = _bank(ctx)
    card = bank.card_by_id(card_id)
    if card is None or card.status != "blocked":
        raise ToolError("That card is not blocked, so there is nothing to take back.")
    card = bank.unblock_card(card_id)
    return json.dumps({"card_id": card.id, "status": card.status})


async def order_replacement(ctx: RunContext, card_id: str) -> str:
    """Order a replacement for a blocked card, shipped to the address on file.

    Args:
        card_id: The blocked card's id.
    """
    bank = _bank(ctx)
    card = bank.card_by_id(card_id)
    if card is None or card.status != "blocked":
        raise ToolError("That card is active. Block it first, after the caller agrees.")
    return json.dumps(bank.order_replacement(card_id))


async def open_dispute(ctx: RunContext, transaction_ids: list[str], reason: str) -> str:
    """Dispute one or more transactions. The bank knows which account and card each one
    is on, so the caller does not need to.

    Args:
        transaction_ids: The transaction ids from get_transactions, confirmed with the caller.
        reason: The caller's reason, in their words.
    """
    return json.dumps(_bank(ctx).open_dispute(transaction_ids, reason))


async def transfer_funds(
    ctx: RunContext, from_account_id: str, to_account_id: str, amount: float
) -> str:
    """Move money between the caller's own accounts.

    Args:
        from_account_id: The account the money leaves, from get_accounts.
        to_account_id: The account the money goes to, from get_accounts.
        amount: The amount in dollars, as confirmed with transfer_check.
    """
    # The write enforces its own policy: the contract checks where the amount came from,
    # not whether the check allowed it.
    bank = _bank(ctx)
    if from_account_id == to_account_id:
        raise ToolError("The two accounts are the same. Ask which account the money goes to.")
    check = transfer_check(bank, from_account_id, amount)
    if not check["allowed"]:
        raise ToolError(check["say"])
    return json.dumps(bank.transfer(from_account_id, to_account_id, amount, fee=check["fee"]))


async def add_travel_notice(
    ctx: RunContext, card_id: str, start_date: str, end_date: str, countries: list[str]
) -> str:
    """Tell the bank a card will be used abroad so it is not blocked for unusual activity.

    Args:
        card_id: The card id from get_card.
        start_date: First day of travel, year-month-day.
        end_date: Last day of travel, year-month-day.
        countries: The countries, by name.
    """
    return json.dumps(_bank(ctx).add_travel_notice(card_id, start_date, end_date, countries))


TOOLS = {
    f.__name__: f
    for f in (
        verify_identity,
        get_accounts,
        get_card,
        get_transactions,
        transfer_check_tool,
        dispute_total_tool,
        block_card,
        unblock_card,
        order_replacement,
        open_dispute,
        transfer_funds,
        add_travel_notice,
    )
}


# --- the flows: one scoped task per write --------------------------------------------------


@dataclass(frozen=True)
class Flow:
    """A unit of work the parent hands to a task: its instructions and its tools. The
    task finishes when the caller is done with the request, after at least one of its
    writes succeeded, or when the caller drops the request."""

    name: str
    instructions: str
    tools: list[str]
    description: str = field(default="")

    @property
    def writes(self) -> list[str]:
        return [t for t in self.tools if CONTRACT.rules.get(t, Rule()).write]


FLOWS = {
    "block_card": Flow(
        name="block_card",
        description="Block a lost, stolen or compromised card, and order a replacement if the caller wants one.",
        instructions=(
            "You are handling exactly one request: block the caller's card, and order a replacement only if they ask. "
            "Ask for the last four digits of the card, read them back digit by digit, then get_card, then block_card."
            " The change tool asks you to confirm with the caller when it is time. Say what was done and how to take it back, then offer a replacement."
        ),
        tools=["get_card", "block_card", "order_replacement"],
    ),
    "unblock_card": Flow(
        name="unblock_card",
        description="Take a block back when the caller found the card.",
        instructions=(
            "You are handling exactly one request: lift the block on a card. Get the card by its last four "
            "digits, then unblock_card. The change tool asks you to confirm with the caller when it is time."
        ),
        tools=["get_card", "unblock_card"],
    ),
    "dispute": Flow(
        name="dispute",
        description="Dispute transactions the caller does not recognise.",
        instructions=(
            "You are handling exactly one request: a dispute. Find the transactions the caller describes with "
            "get_transactions; if it is unclear which ones they mean, read the recent ones and let them pick. "
            "Total them with dispute_total, then open_dispute. The change tool asks you to confirm with the caller when it is time."
        ),
        tools=["get_accounts", "get_transactions", "dispute_total", "open_dispute"],
    ),
    "transfer": Flow(
        name="transfer",
        description="Move money between the caller's own accounts.",
        instructions=(
            "You are handling exactly one request: a transfer between the caller's accounts. Get the accounts, confirm "
            "which is the source and which the destination, run transfer_check with the amount and repeat its sentence, "
            "then transfer_funds with the exact amount transfer_check allowed. The change tool asks you to confirm with the caller when it is time."
        ),
        tools=["get_accounts", "transfer_check", "transfer_funds"],
    ),
    "travel_notice": Flow(
        name="travel_notice",
        description="Add a travel notice to a card.",
        instructions=(
            "You are handling exactly one request: a travel notice. Get the card by its last four digits, confirm the "
            "dates as year-month-day and the countries, then add_travel_notice. The change tool asks you to confirm with the caller when it is time."
        ),
        tools=["get_card", "add_travel_notice"],
    ),
}
