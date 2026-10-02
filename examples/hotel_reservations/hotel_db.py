from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import date
from typing import Any, Literal, get_args

import apsw

from livekit.agents.utils import shortuuid


@dataclass(frozen=True)
class Pricing:
    breakfast_per_night: int = 2500
    valet_per_night: int = 3500
    late_checkout: int = 4000
    pet_fee: int = 5000
    smoking_cleaning_fee: int = 25000
    tax_rate_pct: int = 12
    cancellation_window_hours: int = 48
    cancellation_forfeit_nights: int = 1
    minibar_auto_refund_threshold: int = 2000


PRICING = Pricing()


RoomExtra = Literal["breakfast", "valet", "late_checkout", "pets"]


ALLOWED_EXTRAS: frozenset[str] = frozenset(get_args(RoomExtra))


def extras_total(extras: Sequence[str], nights: int) -> int:
    total = 0
    if "breakfast" in extras:
        total += PRICING.breakfast_per_night * nights
    if "valet" in extras:
        total += PRICING.valet_per_night * nights
    if "late_checkout" in extras:
        total += PRICING.late_checkout
    if "pets" in extras:
        total += PRICING.pet_fee
    return total


def apply_tax(amount_cents: int) -> int:
    return (amount_cents * PRICING.tax_rate_pct) // 100


def format_usd(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100}.{cents % 100:02d}"


def speak_usd(cents: int) -> str:
    dollars, change = divmod(abs(cents), 100)
    if change == 0:
        return f"{dollars} dollars"
    return f"{dollars} dollars and {change} cents"


def describe_extras(nights: int) -> str:
    """Each extra and what it adds to a stay of this length.

    Priced through extras_total so there is no second pricing table to drift, and
    priced for the actual nights, since breakfast and valet are per-night while
    late checkout and the pet fee are one-off.
    """
    return "\n".join(
        f"- {extra.replace('_', ' ')}: adds {speak_usd(extras_total([extra], nights))}"
        for extra in sorted(ALLOWED_EXTRAS)
    )


DisputeCategory = Literal[
    "minibar",
    "room_service_restaurant",
    "damage_cleaning",
    "late_checkout_fee",
    "cancellation_fee",
    "no_show",
    "double_charge_billing_error",
    "other",
]


DisputeAction = Literal[
    "auto_refund_if_under_threshold",
    "verify_explain_then_offer_credit",
    "explain_no_refund",
    "explain_policy_offer_goodwill",
    "correct_immediately_or_open_ticket",
]


@dataclass(frozen=True)
class DisputePolicy:
    action: DisputeAction
    escalation: Literal["manager", "accounting", "none"]
    explanation: str


DISPUTE_POLICIES: dict[DisputeCategory, DisputePolicy] = {
    "minibar": DisputePolicy(
        action="auto_refund_if_under_threshold",
        escalation="manager",
        explanation=(
            "For small minibar charges I can waive them right away. "
            "If it's a larger amount I'll verify against the housekeeping note first."
        ),
    ),
    "room_service_restaurant": DisputePolicy(
        action="verify_explain_then_offer_credit",
        escalation="manager",
        explanation=(
            "I'll pull up the order. If something looks off I can apply a credit, "
            "or escalate to the food and beverage manager."
        ),
    ),
    "damage_cleaning": DisputePolicy(
        action="explain_no_refund",
        escalation="manager",
        explanation=(
            "Damage and cleaning fees are assessed by housekeeping. I can't waive them, "
            "but I can have the manager review and follow up by email."
        ),
    ),
    "late_checkout_fee": DisputePolicy(
        action="explain_policy_offer_goodwill",
        escalation="manager",
        explanation=(
            f"Late checkout past noon is {format_usd(PRICING.late_checkout)}. "
            "If this is your first time I can waive it as a one-time courtesy."
        ),
    ),
    "cancellation_fee": DisputePolicy(
        action="explain_policy_offer_goodwill",
        escalation="manager",
        explanation=(
            f"Our policy is free cancellation up to {PRICING.cancellation_window_hours} "
            f"hours before check-in. Inside that window it's one night. "
            "If you're a returning guest I can waive it once."
        ),
    ),
    "no_show": DisputePolicy(
        action="explain_no_refund",
        escalation="manager",
        explanation=(
            "This room was guaranteed to your card and there's no cancellation on record, "
            "so it was held for you and charged as a no-show under the guarantee policy. "
            "I can't reverse a guaranteed charge myself, but I can have the manager review "
            "it and follow up by email."
        ),
    ),
    "double_charge_billing_error": DisputePolicy(
        action="correct_immediately_or_open_ticket",
        escalation="accounting",
        explanation=(
            "If I can see the duplicate I'll refund it right now. "
            "Otherwise accounting will open a ticket and email you within two business days."
        ),
    ),
    "other": DisputePolicy(
        action="verify_explain_then_offer_credit",
        escalation="manager",
        explanation="Let me look into that and offer a fair resolution.",
    ),
}


# Set HOTEL_TODAY=YYYY-MM-DD before import for deterministic sim runs.
TODAY: date = (
    date.fromisoformat(os.environ["HOTEL_TODAY"]) if os.environ.get("HOTEL_TODAY") else date.today()
)


MAX_PARTY_SIZE = 6


RoomType = Literal["king", "queen_2beds", "suite", "penthouse"]


@dataclass
class RoomOption:
    """One bookable type+view pairing and what the cheapest room in it costs.

    The pairing is the unit a caller actually picks, not the type: rate varies with
    the view (a city king is 240/night, an ocean king 260), so a price quoted per
    type is a price that moves once the view is known. Splitting the row also means
    a view a type doesn't have has no row to be offered from.
    """

    type: RoomType
    view: str
    # cheapest free room in this pairing; rate varies room to room within it, and
    # _SQL_FREE_ROOM hands out the cheapest, so this is the rate that gets charged
    nightly_rate: int


def describe_room_options(options: Sequence[RoomOption]) -> str:
    """The bookable pairings as one line each, cheapest first.

    One line is one type with one view and that pairing's price. Rolling a type's
    views onto a single line lets a neighbouring line's view bind to the wrong type
    - a garden-view king, which has never existed - and hides that the view
    moves the price, so a figure spoken before the view is settled has to change.
    """
    return "\n".join(
        f"- {o.type.replace('_', ' ')}, {o.view} view: {speak_usd(o.nightly_rate)}/night"
        for o in options
    )


@dataclass
class RoomBooking:
    id: int
    code: str
    room_id: str
    room_type: RoomType
    smoking: bool
    nightly_rate: int
    first_name: str
    last_name: str
    email: str
    phone: str
    check_in: date
    check_out: date
    guests: int
    extras: list[RoomExtra]
    total: int
    card_last4: str
    status: Literal["confirmed", "cancelled"]
    late_arrival_note: str | None

    @property
    def nights(self) -> int:
        return (self.check_out - self.check_in).days


@dataclass
class LineItem:
    label: str
    amount_cents: int


@dataclass
class Invoice:
    id: int
    booking_code: str
    line_items: list[LineItem]
    subtotal: int
    taxes: int
    total: int
    paid: bool


FollowupKind = Literal[
    "sales_lead",  # group bookings, events, weddings, corporate rates
    "identity_change",  # caller wants name/email/phone/card on file updated
    "callback",  # caller asked to be called back later
    "verification_help",  # verification failed; route to a human
    "abandoned_booking",  # caller dropped mid-booking; a human can call back
    "other",
]


EmailKind = Literal[
    "booking_confirmation",  # the room booking details + code, re-sent to the address on file
    "folio",  # itemized bill / invoice, re-sent to the address on file
]


# Hotel departments a caller can be transferred to. NOT a guest room - the
# operator never connects a caller to a guest (see guest-privacy policy).
TransferDestination = Literal[
    "restaurant",
    "duty_manager",
    "housekeeping",
]


# the predominant room-share arrangement for a group block
GroupShareType = Literal["twin", "double", "single", "mixed"]


class Unavailable(Exception):
    pass


class NotFound(Exception):
    pass


def invoice_line_items(
    *, nights: int, room_subtotal: int, extras: Sequence[str], tax: int
) -> list[LineItem]:
    """The itemized invoice for a stay. Shared by book_room and the seed script
    so the breakdown can't drift between them."""
    items = [LineItem(f"Room ({nights} nights)", room_subtotal)]
    if "breakfast" in extras:
        items.append(LineItem(f"Breakfast ({nights} nights)", PRICING.breakfast_per_night * nights))
    if "valet" in extras:
        items.append(LineItem(f"Valet ({nights} nights)", PRICING.valet_per_night * nights))
    if "late_checkout" in extras:
        items.append(LineItem("Late checkout", PRICING.late_checkout))
    if "pets" in extras:
        items.append(LineItem("Pet fee", PRICING.pet_fee))
    items.append(LineItem(f"Tax ({PRICING.tax_rate_pct}%)", tax))
    return items


def compute_invoice(
    *, nightly_rate: int, nights: int, extras: Sequence[str]
) -> tuple[int, int, int, list[LineItem]]:
    """Single source of truth for booking math: returns (subtotal, taxes, total, line_items).
    Used by book_room, update_booking, and the seed script so no caller can drift."""
    room_subtotal = nightly_rate * nights
    subtotal = room_subtotal + extras_total(extras, nights)
    taxes = apply_tax(subtotal)
    items = invoice_line_items(nights=nights, room_subtotal=room_subtotal, extras=extras, tax=taxes)
    return subtotal, taxes, subtotal + taxes, items


class HotelDB:
    def __init__(self, conn: apsw.Connection) -> None:
        self._conn: apsw.Connection = conn

    @classmethod
    def empty(cls) -> HotelDB:
        conn = apsw.Connection(":memory:")
        _install_schema(conn)
        return cls(conn)

    @classmethod
    def from_bytes(cls, seed_bytes: bytes) -> HotelDB:
        conn = apsw.Connection(":memory:")
        conn.deserialize("main", seed_bytes)
        _install_schema(conn)
        return cls(conn)

    @property
    def connection(self) -> apsw.Connection:
        return self._conn

    def serialize(self) -> bytes:
        return bytes(self._conn.serialize("main"))

    def close(self) -> None:
        self._conn.close()

    async def aclose(self) -> None:
        self.close()

    async def list_room_options(
        self,
        *,
        check_in: date,
        check_out: date,
        guests: int,
        smoking: bool | None = None,
        exclude_booking_code: str | None = None,
    ) -> list[RoomOption]:
        rows = self.connection.execute(
            _SQL_AVAILABILITY,
            {
                "guests": guests,
                "smoking": int(smoking) if smoking is not None else None,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "exclude": exclude_booking_code,
            },
        )
        return [RoomOption(t, view, rate) for t, view, rate in rows]

    async def find_booking(
        self,
        *,
        last_name: str,
        confirmation_code: str | None = None,
        email: str | None = None,
        card_last4: str | None = None,
    ) -> RoomBooking | None:
        return _row_to_booking(
            self.connection.execute(
                _SQL_FIND_BOOKING,
                {
                    "last_name": last_name,
                    "code": confirmation_code.upper() if confirmation_code else None,
                    "email": email,
                    "card_last4": card_last4,
                },
            ).fetchone()
        )

    async def get_invoice(self, booking_code: str) -> Invoice:
        row = self.connection.execute(_SQL_GET_INVOICE, {"code": booking_code}).fetchone()
        if not row:
            raise NotFound(f"no invoice for {booking_code}")
        invoice_id, line_items_json, subtotal, taxes, total, paid = row
        return Invoice(
            id=invoice_id,
            booking_code=booking_code,
            line_items=[LineItem(**li) for li in json.loads(line_items_json)],
            subtotal=subtotal,
            taxes=taxes,
            total=total,
            paid=bool(paid),
        )

    async def book_room(
        self,
        *,
        room_type: RoomType,
        smoking: bool,
        guests: int,
        check_in: date,
        check_out: date,
        first_name: str,
        last_name: str,
        email: str,
        phone: str,
        card_last4: str,
        extras: list[RoomExtra],
        view: str | None = None,
    ) -> RoomBooking:
        clean_extras = sorted(e for e in extras if e in ALLOWED_EXTRAS)
        code = shortuuid("HTL-")
        conn = self.connection
        with conn:
            row = conn.execute(
                _SQL_FREE_ROOM,
                {
                    "room_type": room_type,
                    "smoking": int(smoking),
                    "guests": guests,
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "exclude": None,
                    "view": view,
                    "prefer": None,
                },
            ).fetchone()
            if not row:
                what = f"{view} {room_type}" if view else room_type
                raise Unavailable(f"sold out: {what}")
            room_id, nightly_rate = row
            nights = (check_out - check_in).days
            subtotal, taxes, total, items = compute_invoice(
                nightly_rate=nightly_rate, nights=nights, extras=clean_extras
            )
            booking_id = _insert(
                conn,
                "hotel_bookings",
                {
                    "code": code,
                    "room_id": room_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    # spoken emails are case-free; transcription capitalization is noise
                    "email": email.strip().lower(),
                    # digits only: a spoken number transcribes with unpredictable punctuation
                    "phone": "".join(c for c in phone if c.isdigit()),
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "guests": guests,
                    "extras": ",".join(clean_extras),
                    "total": total,
                    "card_last4": card_last4,
                },
            )
            _insert(
                conn,
                "hotel_invoices",
                {
                    "booking_code": code,
                    "line_items": json.dumps([li.__dict__ for li in items]),
                    "subtotal": subtotal,
                    "taxes": taxes,
                    "total": total,
                },
            )
        return RoomBooking(
            id=booking_id,
            code=code,
            room_id=room_id,
            room_type=room_type,
            smoking=smoking,
            nightly_rate=nightly_rate,
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            check_in=check_in,
            check_out=check_out,
            guests=guests,
            extras=clean_extras,
            total=total,
            card_last4=card_last4,
            status="confirmed",
            late_arrival_note=None,
        )

    async def update_booking(
        self,
        *,
        booking_code: str,
        room_type: RoomType,
        smoking: bool,
        guests: int,
        check_in: date,
        check_out: date,
        extras: list[RoomExtra],
        view: str | None = None,
    ) -> RoomBooking:
        # Re-pick a free room of the new (type, smoking) for the new dates,
        # ignoring the booking being modified itself (so same-room "extend
        # by one night" doesn't conflict with itself). The room the guest
        # already has wins when it still fits - a date change must never
        # quietly move someone out of their garden view. A requested `view`
        # filters to rooms with that view, which is how the guest gets moved
        # to a different room (e.g. a city-view room to a garden-view one).
        clean_extras = sorted(e for e in extras if e in ALLOWED_EXTRAS)
        conn = self.connection
        with conn:
            current = conn.execute(
                "SELECT room_id FROM hotel_bookings WHERE code = ? AND status = 'confirmed'",
                (booking_code,),
            ).fetchone()
            row = conn.execute(
                _SQL_FREE_ROOM,
                {
                    "room_type": room_type,
                    "smoking": int(smoking),
                    "guests": guests,
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "exclude": booking_code,
                    "view": view,
                    "prefer": current[0] if current else None,
                },
            ).fetchone()
            if not row:
                what = f"{view} {room_type}" if view else room_type
                raise Unavailable(f"sold out: {what}")
            room_id, nightly_rate = row
            nights = (check_out - check_in).days
            subtotal, taxes, total, items = compute_invoice(
                nightly_rate=nightly_rate, nights=nights, extras=clean_extras
            )
            changed = _update(
                conn,
                "hotel_bookings",
                {
                    "room_id": room_id,
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "guests": guests,
                    "extras": ",".join(clean_extras),
                    "total": total,
                },
                {"code": booking_code, "status": "confirmed"},
            )
            if changed == 0:
                raise NotFound(f"booking not found: {booking_code}")
            _update(
                conn,
                "hotel_invoices",
                {
                    "line_items": json.dumps([li.__dict__ for li in items]),
                    "subtotal": subtotal,
                    "taxes": taxes,
                    "total": total,
                },
                {"booking_code": booking_code},
            )
        updated = _row_to_booking(
            conn.execute(_SQL_BOOKING_BY_CODE, {"code": booking_code}).fetchone()
        )
        if updated is None:
            raise NotFound(f"booking vanished mid-update: {booking_code}")
        return updated

    async def cancel_room_booking(self, booking_code: str) -> None:
        conn = self.connection
        changed = _update(
            conn,
            "hotel_bookings",
            {"status": "cancelled"},
            {"code": booking_code, "status": "confirmed"},
        )
        if changed == 0:
            raise NotFound(f"booking not found: {booking_code}")

    async def lookup_guest_history(self, *, last_name: str) -> str | None:
        """Return a returning guest's remembered preferences from past stays, or None
        if there's no history on file for that name."""
        row = self.connection.execute(
            "SELECT preferences FROM guest_history WHERE LOWER(last_name) = LOWER(?)",
            (last_name,),
        ).fetchone()
        return row[0] if row else None

    async def add_to_waitlist(
        self,
        *,
        first_name: str,
        last_name: str,
        phone: str,
        check_in: date,
        check_out: date,
        guests: int,
    ) -> str:
        """Record a waitlist entry for dates the hotel is sold out on, and return a
        reference. No room is held - the desk calls back only if something frees up."""
        code = shortuuid("WL-")
        with self.connection as conn:
            _insert(
                conn,
                "waitlist",
                {
                    "code": code,
                    "first_name": first_name,
                    "last_name": last_name,
                    "phone": "".join(c for c in phone if c.isdigit()) or phone,
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "guests": guests,
                },
            )
        return code

    async def reinstate_booking(self, booking_code: str) -> None:
        """Reactivate a previously cancelled booking, but only if its original room is
        still free for its dates. Raises NotFound if the code is unknown, Unavailable if
        the room has since been taken. A no-op if the booking is already confirmed."""
        conn = self.connection
        row = conn.execute(
            "SELECT room_id, check_in, check_out, status FROM hotel_bookings WHERE code = ?",
            (booking_code,),
        ).fetchone()
        if row is None:
            raise NotFound(f"booking not found: {booking_code}")
        room_id, check_in, check_out, status = row
        if status == "confirmed":
            return
        clash = conn.execute(
            "SELECT 1 FROM hotel_bookings WHERE room_id = ? AND status = 'confirmed' "
            "AND code != ? AND NOT (check_out <= ? OR check_in >= ?) LIMIT 1",
            (room_id, booking_code, check_in, check_out),
        ).fetchone()
        if clash:
            raise Unavailable("that room is no longer free for those dates")
        with conn:
            _update(conn, "hotel_bookings", {"status": "confirmed"}, {"code": booking_code})

    async def flag_late_arrival(self, *, booking_code: str, note: str) -> None:
        conn = self.connection
        changed = _update(
            conn,
            "hotel_bookings",
            {"late_arrival_note": note},
            {"code": booking_code, "status": "confirmed"},
        )
        if changed == 0:
            raise NotFound(f"booking not found: {booking_code}")

    async def update_booking_card(self, *, booking_code: str, card_last4: str) -> None:
        conn = self.connection
        changed = _update(
            conn,
            "hotel_bookings",
            {"card_last4": card_last4},
            {"code": booking_code, "status": "confirmed"},
        )
        if changed == 0:
            raise NotFound(f"booking not found: {booking_code}")

    async def record_followup(
        self,
        *,
        kind: FollowupKind,
        caller_name: str,
        caller_phone: str,
        summary: str,
    ) -> str:
        code = shortuuid("FUP-")
        digits = "".join(c for c in caller_phone if c.isdigit())
        conn = self.connection
        with conn:
            _insert(
                conn,
                "hotel_followups",
                {
                    "code": code,
                    "kind": kind,
                    "caller_name": caller_name,
                    # "room 402" and "415-555-0173" both normalize to digits
                    "caller_phone": digits or caller_phone,
                    "summary": summary,
                },
            )
        return code

    async def send_email(self, *, recipient: str, kind: str) -> str:
        """Stub email send: records that a document of `kind` was sent to `recipient`
        and returns a reference. No real mail goes out; the row is the gradable signal
        that the agent actually sent rather than just claiming to."""
        if kind not in get_args(EmailKind):
            raise NotFound(
                f"unknown email kind: {kind} - options: {', '.join(get_args(EmailKind))}"
            )
        code = shortuuid("EML-")
        with self.connection as conn:
            _insert(
                conn,
                "emails_sent",
                {
                    "code": code,
                    "recipient": recipient.strip().lower(),
                    "kind": kind,
                },
            )
        return code

    async def transfer_call(self, *, destination: str, summary: str) -> str:
        """Stub call transfer: records that the caller was transferred to a hotel
        department with a one-line summary, and returns a reference."""
        if destination not in get_args(TransferDestination):
            raise NotFound(
                f"unknown destination: {destination} - options: {', '.join(get_args(TransferDestination))}"
            )
        code = shortuuid("XFR-")
        with self.connection as conn:
            _insert(
                conn,
                "transfer_calls",
                {
                    "code": code,
                    "destination": destination,
                    "summary": summary,
                },
            )
        return code

    async def peek_stay_total(
        self,
        *,
        room_type: str,
        smoking: bool,
        guests: int,
        check_in: date,
        check_out: date,
        view: str | None,
        extras: Sequence[str],
    ) -> int | None:
        """The exact total (with tax) for the room book_room would pick right now -
        so the agent can quote the real number in the read-back instead of doing
        per-night arithmetic itself (and forgetting tax)."""
        row = self.connection.execute(
            _SQL_FREE_ROOM,
            {
                "room_type": room_type,
                "smoking": int(smoking),
                "guests": guests,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "exclude": None,
                "view": view,
                "prefer": None,
            },
        ).fetchone()
        if not row:
            return None
        nights = (check_out - check_in).days
        _, _, total, _ = compute_invoice(nightly_rate=row[1], nights=nights, extras=list(extras))
        return total

    async def record_group_inquiry(
        self,
        *,
        company: str,
        contact_name: str,
        contact_phone: str,
        party_size: int,
        share_type: GroupShareType,
        check_in: date,
        nights: int,
    ) -> str:
        code = shortuuid("GRP-")
        conn = self.connection
        with conn:
            _insert(
                conn,
                "group_inquiries",
                {
                    "code": code,
                    "company": company,
                    "contact_name": contact_name,
                    # digits only: a spoken callback number transcribes with
                    # unpredictable punctuation, and nothing dials it back out
                    "contact_phone": "".join(c for c in contact_phone if c.isdigit()),
                    "party_size": party_size,
                    "share_type": share_type,
                    "check_in": check_in.isoformat(),
                    "nights": nights,
                },
            )
        return code

    async def file_dispute(
        self,
        *,
        booking_code: str,
        line_item: str,
        amount_cents: int,
        category: DisputeCategory,
        caller_note: str,
        outcome: str,
        refund_amount: int,
    ) -> str:
        case_number = shortuuid("DSP-")
        conn = self.connection
        with conn:
            _insert(
                conn,
                "hotel_disputes",
                {
                    "case_number": case_number,
                    "booking_code": booking_code,
                    "line_item": line_item,
                    "amount": amount_cents,
                    "category": category,
                    "caller_note": caller_note,
                    "outcome": outcome,
                    "refund_amount": refund_amount,
                },
            )
            if refund_amount > 0:
                conn.execute(
                    "UPDATE hotel_invoices SET total = total - :refund WHERE booking_code = :code",
                    {"refund": refund_amount, "code": booking_code},
                )
        return case_number


def _install_schema(conn: apsw.Connection) -> None:
    for _ in conn.execute(SCHEMA):
        pass


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS hotel_rooms (
    id            TEXT    PRIMARY KEY,  -- human room number, e.g. 'RM_201' (floor 2, room 01)
    type          TEXT    NOT NULL CHECK (type IN ('king','queen_2beds','suite','penthouse')),
    nightly_rate  INTEGER NOT NULL,
    max_occupancy INTEGER NOT NULL,
    smoking       BOOLEAN NOT NULL DEFAULT 0,
    pets_allowed  BOOLEAN NOT NULL DEFAULT 0,
    room_view     TEXT    NOT NULL CHECK (room_view IN ('city','ocean','garden','interior'))
);

CREATE TABLE IF NOT EXISTS hotel_bookings (
    id                 INTEGER PRIMARY KEY,
    code               TEXT    NOT NULL UNIQUE,
    room_id            TEXT    NOT NULL REFERENCES hotel_rooms(id),
    first_name         TEXT    NOT NULL,
    last_name          TEXT    NOT NULL,
    email              TEXT    NOT NULL,
    phone              TEXT    NOT NULL,
    check_in           DATE    NOT NULL,
    check_out          DATE    NOT NULL,
    guests             INTEGER NOT NULL CHECK (guests >= 1),
    extras             TEXT    NOT NULL DEFAULT '',
    total              INTEGER NOT NULL,
    card_last4         TEXT    NOT NULL,
    status             TEXT    NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed','cancelled')),
    late_arrival_note  TEXT,
    CHECK (check_out > check_in)
);

CREATE TABLE IF NOT EXISTS hotel_invoices (
    id           INTEGER PRIMARY KEY,
    booking_code TEXT    NOT NULL UNIQUE REFERENCES hotel_bookings(code),
    line_items   JSON    NOT NULL,
    subtotal     INTEGER NOT NULL,
    taxes        INTEGER NOT NULL,
    total        INTEGER NOT NULL,
    paid         BOOLEAN NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hotel_disputes (
    id            INTEGER PRIMARY KEY,
    case_number   TEXT    NOT NULL UNIQUE,
    booking_code  TEXT    NOT NULL REFERENCES hotel_bookings(code),
    line_item     TEXT    NOT NULL,
    amount        INTEGER NOT NULL,
    category      TEXT    NOT NULL CHECK (category IN ('minibar','room_service_restaurant','damage_cleaning','late_checkout_fee','cancellation_fee','no_show','double_charge_billing_error','other')),
    caller_note   TEXT    NOT NULL,
    outcome       TEXT    NOT NULL CHECK (outcome IN ('auto_refunded','credit_offered','explained_no_action','goodwill_waived','escalated_to_manager','accounting_ticket_opened','open')),
    refund_amount INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved','rejected'))
);

CREATE TABLE IF NOT EXISTS hotel_followups (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    kind         TEXT    NOT NULL CHECK (kind IN ('housekeeping','sales_lead','identity_change','callback','verification_help','early_checkout','abandoned_booking','lost_and_found','other')),
    caller_name  TEXT    NOT NULL,
    caller_phone TEXT    NOT NULL,
    summary      TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved'))
);

CREATE TABLE IF NOT EXISTS emails_sent (
    id         INTEGER PRIMARY KEY,
    code       TEXT    NOT NULL UNIQUE,
    recipient  TEXT    NOT NULL,
    kind       TEXT    NOT NULL CHECK (kind IN ('booking_confirmation','folio')),
    status     TEXT    NOT NULL DEFAULT 'sent'
);

CREATE TABLE IF NOT EXISTS transfer_calls (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    destination  TEXT    NOT NULL CHECK (destination IN ('restaurant','duty_manager','housekeeping')),
    summary      TEXT    NOT NULL DEFAULT '',
    status       TEXT    NOT NULL DEFAULT 'transferred'
);

CREATE TABLE IF NOT EXISTS waitlist (
    id          INTEGER PRIMARY KEY,
    code        TEXT    NOT NULL UNIQUE,
    first_name  TEXT    NOT NULL,
    last_name   TEXT    NOT NULL,
    phone       TEXT    NOT NULL,
    check_in    TEXT    NOT NULL,
    check_out   TEXT    NOT NULL,
    guests      INTEGER NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'waiting'
);

-- Reference data (read-only): preferences remembered from a returning guest's past
-- stays. The agent reads this to personalize; it never writes here.
CREATE TABLE IF NOT EXISTS guest_history (
    id          INTEGER PRIMARY KEY,
    last_name   TEXT    NOT NULL,
    preferences TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS group_inquiries (
    id            INTEGER PRIMARY KEY,
    code          TEXT    NOT NULL UNIQUE,
    company       TEXT    NOT NULL,
    contact_name  TEXT    NOT NULL,
    contact_phone TEXT    NOT NULL,
    party_size    INTEGER NOT NULL CHECK (party_size >= 15),
    share_type    TEXT    NOT NULL CHECK (share_type IN ('twin','double','single','mixed')),
    check_in      DATE    NOT NULL,
    nights        INTEGER NOT NULL CHECK (nights >= 1),
    status        TEXT    NOT NULL DEFAULT 'pending_credit_approval'
                          CHECK (status IN ('pending_credit_approval','approved','declined'))
);
"""


_BOOKING_COLS = "b.id, b.code, b.room_id, r.type AS room_type, r.smoking, r.nightly_rate, b.first_name, b.last_name, b.email, b.phone, b.check_in, b.check_out, b.guests, b.extras, b.total, b.card_last4, b.status, b.late_arrival_note"


# Names that _row_to_booking maps the SELECT columns onto - derived from the
# dataclass so adding a field in one place keeps them aligned.
_BOOKING_COL_NAMES: tuple[str, ...] = tuple(f.name for f in fields(RoomBooking))


_SQL_FREE_ROOM = """
SELECT id, nightly_rate FROM hotel_rooms
WHERE type = :room_type AND smoking = :smoking AND max_occupancy >= :guests
  AND (:view IS NULL OR room_view = :view)
  AND NOT EXISTS (
    SELECT 1 FROM hotel_bookings b
    WHERE b.room_id = hotel_rooms.id AND b.status = 'confirmed'
      AND (:exclude IS NULL OR b.code != :exclude)
      AND NOT (b.check_out <= :check_in OR b.check_in >= :check_out))
ORDER BY CASE WHEN id = :prefer THEN 0 ELSE 1 END, nightly_rate, id LIMIT 1
"""


_SQL_AVAILABILITY = """
SELECT r.type, r.room_view, MIN(r.nightly_rate)
FROM hotel_rooms r
WHERE r.max_occupancy >= :guests
  AND (:smoking IS NULL OR r.smoking = :smoking)
  AND NOT EXISTS (
    SELECT 1 FROM hotel_bookings b
    WHERE b.room_id = r.id AND b.status = 'confirmed'
      AND (:exclude IS NULL OR b.code != :exclude)
      AND NOT (b.check_out <= :check_in OR b.check_in >= :check_out))
GROUP BY r.type, r.room_view ORDER BY MIN(r.nightly_rate), r.type, r.room_view
"""


_SQL_FIND_BOOKING = f"""
SELECT {_BOOKING_COLS} FROM hotel_bookings b
JOIN hotel_rooms r ON r.id = b.room_id
WHERE LOWER(b.last_name) = LOWER(:last_name)
  AND (:code IS NULL OR REPLACE(b.code, '-', '') = REPLACE(:code, '-', ''))
  AND (:email IS NULL OR LOWER(b.email) = LOWER(:email))
  AND (:card_last4 IS NULL OR b.card_last4 = :card_last4)
LIMIT 1
"""


_SQL_BOOKING_BY_CODE = f"""
SELECT {_BOOKING_COLS} FROM hotel_bookings b
JOIN hotel_rooms r ON r.id = b.room_id
WHERE b.code = :code LIMIT 1
"""


_SQL_GET_INVOICE = "SELECT id, line_items, subtotal, taxes, total, paid FROM hotel_invoices WHERE booking_code = :code"


def _insert(conn: apsw.Connection, table: str, row: dict[str, Any]) -> int:
    keys = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT INTO {table} ({keys}) VALUES ({placeholders})", row)
    return conn.last_insert_rowid()


def _update(
    conn: apsw.Connection, table: str, set_fields: dict[str, Any], where: dict[str, Any]
) -> int:
    set_clause = ", ".join(f"{k} = :{k}" for k in set_fields)
    where_clause = " AND ".join(f"{k} = :w_{k}" for k in where)
    params = {**set_fields, **{f"w_{k}": v for k, v in where.items()}}
    conn.execute(f"UPDATE {table} SET {set_clause} WHERE {where_clause}", params)
    return conn.changes()


def _row_to_booking(row: tuple[Any, ...] | None) -> RoomBooking | None:
    if row is None:
        return None
    d = dict(zip(_BOOKING_COL_NAMES, row, strict=True))
    d["check_in"], d["check_out"] = (
        date.fromisoformat(d["check_in"]),
        date.fromisoformat(d["check_out"]),
    )
    d["extras"] = [e for e in d["extras"].split(",") if e]
    d["smoking"] = bool(d["smoking"])
    return RoomBooking(**d)
