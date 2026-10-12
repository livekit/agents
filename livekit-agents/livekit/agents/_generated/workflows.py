# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/workflows.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time


@dataclass
class GetAddressResult:
    """GetAddressResult(address: 'str')"""

    address: str


@dataclass
class GetEmailResult:
    """GetEmailResult(email_address: 'str')"""

    email_address: str


@dataclass
class GetPhoneNumberResult:
    """GetPhoneNumberResult(phone_number: 'str')"""

    phone_number: str


@dataclass
class GetNameResult:
    """GetNameResult(first_name: 'str | None' = None, middle_name: 'str | None' = None, last_name: 'str | None' = None)"""

    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None


@dataclass
class GetDOBResult:
    """GetDOBResult(date_of_birth: 'date', time_of_birth: 'time | None' = None)"""

    date_of_birth: date
    time_of_birth: time | None = None


@dataclass
class GetCardNumberResult:
    """GetCardNumberResult(issuer: 'str', card_number: 'str')"""

    issuer: str
    card_number: str


@dataclass
class GetSecurityCodeResult:
    """GetSecurityCodeResult(security_code: 'str')"""

    security_code: str


@dataclass
class GetExpirationDateResult:
    """GetExpirationDateResult(date: 'str')"""

    date: str


@dataclass
class GetCreditCardResult:
    """GetCreditCardResult(cardholder_name: 'str', issuer: 'str', card_number: 'str', security_code: 'str', expiration_date: 'str')"""

    cardholder_name: str
    issuer: str
    card_number: str
    security_code: str
    expiration_date: str


@dataclass
class WarmTransferResult:
    human_agent_identity: str


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
GetAddressResult.__module__ = "livekit.agents.beta.workflows.address"
GetEmailResult.__module__ = "livekit.agents.beta.workflows.email_address"
GetPhoneNumberResult.__module__ = "livekit.agents.beta.workflows.phone_number"
GetNameResult.__module__ = "livekit.agents.beta.workflows.name"
GetDOBResult.__module__ = "livekit.agents.beta.workflows.dob"
GetCardNumberResult.__module__ = "livekit.agents.beta.workflows.credit_card"
GetSecurityCodeResult.__module__ = "livekit.agents.beta.workflows.credit_card"
GetExpirationDateResult.__module__ = "livekit.agents.beta.workflows.credit_card"
GetCreditCardResult.__module__ = "livekit.agents.beta.workflows.credit_card"
WarmTransferResult.__module__ = "livekit.agents.beta.workflows.warm_transfer"
