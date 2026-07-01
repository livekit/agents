from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, fields
from datetime import date, time, timedelta
from itertools import groupby
from typing import Any, Literal, get_args

import apsw

from livekit.agents.utils import shortuuid

logger = logging.getLogger("hotel-receptionist.db")


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
    # Hand the TTS a plain currency string ("$100.10"); its parser reads it
    # correctly - we don't spell the amount out in words.
    dollars, change = divmod(abs(cents), 100)
    return f"${dollars:,}" if change == 0 else f"${dollars:,}.{change:02d}"


def speak_time(t: time) -> str:
    """A clock time as natural speech, e.g. '7 PM', '6:30 PM'."""
    hour = t.hour % 12 or 12
    suffix = "PM" if t.hour >= 12 else "AM"
    return f"{hour} {suffix}" if t.minute == 0 else f"{hour}:{t.minute:02d} {suffix}"


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

RoomType = Literal["king", "queen_2beds", "double_queen", "suite", "penthouse"]


@dataclass
class RoomTypeAvailability:
    type: RoomType
    nightly_rate: int
    # every distinct view available for this type on the dates,
    # e.g. ["city", "garden"]
    views: list[str]


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
class RestaurantTable:
    id: int
    label: str
    capacity: int
    location: str
    description: str = ""


@dataclass
class RestaurantReservation:
    id: int
    code: str
    table_id: int
    first_name: str
    last_name: str
    phone: str
    party_size: int
    date: date
    time: time
    notes: str | None
    status: Literal["confirmed", "cancelled"]


@dataclass
class TimeSlot:
    time: time
    available_table_ids: list[int]


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
    "housekeeping",  # in-house guest needs something brought or fixed (towels, amenities, maintenance)
    "sales_lead",  # group bookings, events, weddings, corporate rates
    "identity_change",  # caller wants name/email/phone/card on file updated
    "callback",  # caller asked to be called back later
    "verification_help",  # verification failed; route to a human
    "early_checkout",  # in-house guest wants to leave early; front desk handles
    "abandoned_booking",  # caller dropped mid-booking; a human can call back
    "lost_and_found",  # guest reports an item left behind; route to housekeeping/lost-and-found
    "other",
]

EmailKind = Literal[
    "booking_confirmation",  # the room booking details + code, re-sent to the address on file
    "folio",  # itemized bill / invoice, re-sent to the address on file
]

# Emergency classification (manual P1§13/P3§8): health -> ambulance, fire -> fire
# brigade, safety/theft/assault/threat -> police; every kind also alerts the duty
# manager and sends hotel staff to the room.
EmergencyKind = Literal["medical", "fire", "security"]

# Hotel departments a caller can be transferred to. NOT a guest room - the
# operator never connects a caller to a guest (see guest-privacy policy).
TransferDestination = Literal[
    "restaurant",
    "duty_manager",
    "housekeeping",
]


@dataclass
class Followup:
    id: int
    code: str
    kind: FollowupKind
    caller_name: str
    caller_phone: str
    summary: str
    status: Literal["open", "resolved"]


# the predominant room-share arrangement for a group block
GroupShareType = Literal["twin", "double", "single", "mixed"]


@dataclass(frozen=True)
class Tour:
    name: str
    pickup_time: time
    pickup_location: str
    price_per_person: int | None  # cents; None for flat-priced tours
    flat_price: int | None
    max_party: int
    description: str


# Bookable through the concierge desk. policies/tours.md describes the same
# catalog to the agent - keep the two in sync.
TOURS: dict[str, Tour] = {
    "half_day_city": Tour(
        name="Half-day city highlights",
        pickup_time=time(9, 0),
        pickup_location="hotel lobby",
        price_per_person=6500,
        flat_price=None,
        max_party=12,
        description="small group, English-speaking guide, about 4.5 hours, entry fees included",
    ),
    "full_day_city": Tour(
        name="Full-day city and bay",
        pickup_time=time(8, 30),
        pickup_location="hotel lobby",
        price_per_person=11000,
        flat_price=None,
        max_party=12,
        description="small group, English-speaking guide, lunch and entry fees included, back about 5 PM",
    ),
    "private_city": Tour(
        name="Private half-day tour",
        pickup_time=time(10, 0),
        pickup_location="hotel lobby",
        price_per_person=None,
        flat_price=29000,
        max_party=4,
        description="private car and English-speaking guide, flexible start, up to 4 guests",
    ),
}


@dataclass(frozen=True)
class SpaService:
    name: str
    price: int  # cents, per guest
    duration_min: int
    max_party: int
    description: str


# Bookable through the spa / health club desk. policies/spa.md describes the
# same catalog to the agent - keep the two in sync.
SPA_SERVICES: dict[str, SpaService] = {
    "deep_tissue_massage": SpaService(
        name="Deep-tissue massage",
        price=14000,
        duration_min=60,
        max_party=2,
        description="60-minute deep-tissue massage with a licensed therapist",
    ),
    "signature_facial": SpaService(
        name="Signature facial",
        price=12000,
        duration_min=50,
        max_party=2,
        description="50-minute signature facial, all skin types",
    ),
    "personal_training": SpaService(
        name="Personal training session",
        price=8000,
        duration_min=45,
        max_party=1,
        description="45-minute one-on-one session in the health club with a trainer",
    ),
    "group_yoga": SpaService(
        name="Group yoga class",
        price=4000,
        duration_min=60,
        max_party=8,
        description="60-minute group yoga class in the studio",
    ),
}


@dataclass(frozen=True)
class BusinessCenterService:
    name: str
    price_per_hour: int | None  # cents; None for flat-priced services
    flat_price: int | None
    max_hours: int
    description: str


# Bookable through the business centre. policies/business_center.md describes the
# same catalog to the agent - keep the two in sync.
BUSINESS_CENTER_SERVICES: dict[str, BusinessCenterService] = {
    "meeting_room": BusinessCenterService(
        name="Meeting room",
        price_per_hour=4000,
        flat_price=None,
        max_hours=8,
        description="seats up to 8, screen and whiteboard, booked by the hour",
    ),
    "secretarial": BusinessCenterService(
        name="Secretarial service",
        price_per_hour=3500,
        flat_price=None,
        max_hours=4,
        description="typing, dictation, and document prep, booked by the hour",
    ),
    "printing": BusinessCenterService(
        name="Printing and binding",
        price_per_hour=None,
        flat_price=2500,
        max_hours=1,
        description="flat-rate print, copy, and bind job, ready same day",
    ),
}


@dataclass(frozen=True)
class FloralArrangement:
    name: str
    price: int  # cents, flat per arrangement


# Ordered through the concierge desk via the hotel florist. policies/florist.md
# describes the same catalog to the agent - keep the two in sync.
FLORIST_ARRANGEMENTS: dict[str, FloralArrangement] = {
    "bouquet": FloralArrangement(name="Seasonal hand-tied bouquet", price=6500),
    "roses": FloralArrangement(name="Dozen long-stem roses", price=9500),
    "centerpiece": FloralArrangement(name="Table centerpiece arrangement", price=14000),
}

# The partner property used when a confirmed guest has to be walked.
WALK_PARTNER_HOTEL = "the Harbor House"


@dataclass
class ConflictResolution:
    """Result of resolve_room_conflict: exactly one of moved_to / walk is set."""

    moved_to: str | None = None  # new room id, stay unchanged
    moved_to_type: str = ""
    moved_to_view: str = ""
    upgraded: bool = False
    walk_partner: str | None = None
    walk_return_date: date | None = None


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


OnChange = Callable[[], Awaitable[None]]


class HotelDB:
    def __init__(self, conn: apsw.Connection, *, on_change: OnChange | None = None) -> None:
        self._conn: apsw.Connection = conn
        self.on_change = on_change

    @classmethod
    def empty(cls, *, on_change: OnChange | None = None) -> HotelDB:
        conn = apsw.Connection(":memory:")
        _install_schema(conn)
        return cls(conn, on_change=on_change)

    @classmethod
    def from_bytes(cls, seed_bytes: bytes, *, on_change: OnChange | None = None) -> HotelDB:
        conn = apsw.Connection(":memory:")
        conn.deserialize("main", seed_bytes)
        _install_schema(conn)
        return cls(conn, on_change=on_change)

    @classmethod
    def open_path(cls, db_path: str, *, on_change: OnChange | None = None) -> HotelDB:
        conn = apsw.Connection(db_path)
        _install_schema(conn)
        return cls(conn, on_change=on_change)

    @property
    def connection(self) -> apsw.Connection:
        return self._conn

    def serialize(self) -> bytes:
        return bytes(self._conn.serialize("main"))

    def close(self) -> None:
        self._conn.close()

    async def aclose(self) -> None:
        self.close()

    async def list_room_types_available(
        self,
        *,
        check_in: date,
        check_out: date,
        guests: int,
        smoking: bool | None = None,
        exclude_booking_code: str | None = None,
    ) -> list[RoomTypeAvailability]:
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
        return [
            RoomTypeAvailability(t, rate, views=sorted((concat or "").split(",")))
            for t, rate, concat in rows
        ]

    async def list_restaurant_availability(
        self, *, on_date: date, party_size: int
    ) -> list[TimeSlot]:
        rows = self.connection.execute(
            _SQL_DINING_AVAILABILITY, {"party_size": party_size, "date": on_date.isoformat()}
        )
        return [
            TimeSlot(time.fromisoformat(slot), [tid for _, tid in group if tid is not None])
            for slot, group in groupby(rows, key=lambda row: row[0])
        ]

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

    async def find_restaurant_reservation(
        self, *, last_name: str, confirmation_code: str | None = None, on_date: date | None = None
    ) -> RestaurantReservation | None:
        return _row_to_reservation(
            self.connection.execute(
                _SQL_FIND_RESERVATION,
                {
                    "last_name": last_name,
                    "code": confirmation_code.upper() if confirmation_code else None,
                    "date": on_date.isoformat() if on_date else None,
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
        if self.on_change:
            await self.on_change()
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
        if self.on_change:
            await self.on_change()
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
        if self.on_change:
            await self.on_change()

    async def lookup_guest_history(self, *, last_name: str) -> str | None:
        """Return a returning guest's remembered preferences from past stays, or None
        if there's no history on file for that name."""
        row = self.connection.execute(
            "SELECT preferences FROM guest_history WHERE LOWER(last_name) = LOWER(?)",
            (last_name,),
        ).fetchone()
        return row[0] if row else None

    async def set_do_not_disturb(self, *, room: str) -> str:
        """Record a Do-Not-Disturb hold on a room and return a reference. The switchboard
        holds the room's calls and messages until it's lifted; emergencies override it."""
        # Normalize to the canonical room id (and reject a mis-heard room) so storage
        # matches every other room-referencing table regardless of how it was spoken.
        room_id = self._require_room(room)
        code = shortuuid("DND-")
        with self.connection as conn:
            _insert(conn, "do_not_disturb", {"code": code, "room_id": room_id})
        if self.on_change:
            await self.on_change()
        return code

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
        if self.on_change:
            await self.on_change()
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
        if self.on_change:
            await self.on_change()

    async def book_restaurant(
        self,
        *,
        first_name: str,
        last_name: str,
        phone: str,
        party_size: int,
        on_date: date,
        at_time: time,
        notes: str | None = None,
    ) -> RestaurantReservation:
        conn = self.connection
        row = conn.execute(
            _SQL_FREE_TABLE,
            {"party_size": party_size, "date": on_date.isoformat(), "time": at_time.isoformat()},
        ).fetchone()
        if not row:
            raise Unavailable(f"restaurant full: {on_date} {at_time}")
        table_id = row[0]
        code = shortuuid("RES-")
        reservation_id = _insert(
            conn,
            "restaurant_reservations",
            {
                "code": code,
                "table_id": table_id,
                "first_name": first_name,
                "last_name": last_name,
                "phone": "".join(c for c in phone if c.isdigit()),
                "party_size": party_size,
                "date": on_date.isoformat(),
                "time": at_time.isoformat(),
                "notes": notes,
            },
        )
        if self.on_change:
            await self.on_change()
        return RestaurantReservation(
            id=reservation_id,
            code=code,
            table_id=table_id,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            party_size=party_size,
            date=on_date,
            time=at_time,
            notes=notes,
            status="confirmed",
        )

    async def cancel_restaurant_reservation(self, code: str) -> None:
        conn = self.connection
        changed = _update(
            conn,
            "restaurant_reservations",
            {"status": "cancelled"},
            {"code": code, "status": "confirmed"},
        )
        if changed == 0:
            raise NotFound(f"reservation not found: {code}")
        if self.on_change:
            await self.on_change()

    async def modify_restaurant_reservation(
        self,
        *,
        code: str,
        on_date: date,
        at_time: time,
        party_size: int | None = None,
    ) -> RestaurantReservation:
        """Change a confirmed reservation's date/time (and optionally party size).

        Prefers the reservation's current table when it's still free at the new
        slot (mirrors the prefer-current-room logic in update_booking), so a
        same-table time shift leaves table_location unchanged. Falls back to the
        next free table only if the current one is taken or too small.
        """
        if on_date < TODAY:
            raise Unavailable(f"{on_date.isoformat()} is in the past")
        conn = self.connection
        with conn:
            current = conn.execute(
                "SELECT table_id, party_size FROM restaurant_reservations "
                "WHERE code = :code AND status = 'confirmed'",
                {"code": code},
            ).fetchone()
            if not current:
                raise NotFound(f"reservation not found: {code}")
            current_table_id, current_party = current
            new_party = party_size if party_size is not None else current_party
            row = conn.execute(
                _SQL_FREE_TABLE_FOR_MODIFY,
                {
                    "party_size": new_party,
                    "date": on_date.isoformat(),
                    "time": at_time.isoformat(),
                    "code": code,
                    "current_table_id": current_table_id,
                },
            ).fetchone()
            if not row:
                raise Unavailable(f"restaurant full: {on_date} {at_time}")
            table_id = row[0]
            _update(
                conn,
                "restaurant_reservations",
                {
                    "table_id": table_id,
                    "party_size": new_party,
                    "date": on_date.isoformat(),
                    "time": at_time.isoformat(),
                },
                {"code": code, "status": "confirmed"},
            )
        if self.on_change:
            await self.on_change()
        updated = _row_to_reservation(
            conn.execute(_SQL_RESERVATION_BY_CODE, {"code": code}).fetchone()
        )
        if updated is None:
            raise NotFound(f"reservation vanished mid-update: {code}")
        return updated

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
        if self.on_change:
            await self.on_change()

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
        if self.on_change:
            await self.on_change()

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
        if self.on_change:
            await self.on_change()
        return code

    async def schedule_wakeup_call(
        self,
        *,
        room: str,
        guest_name: str,
        call_date: date,
        call_time: time,
    ) -> str:
        room_id = self._require_room(room)
        conn = self.connection
        if call_date < TODAY:
            raise Unavailable(f"{call_date.isoformat()} is in the past")
        code = shortuuid("WUC-")
        with conn:
            _insert(
                conn,
                "wakeup_calls",
                {
                    "code": code,
                    "room_id": room_id,
                    "guest_name": guest_name,
                    "date": call_date.isoformat(),
                    "time": call_time.isoformat(),
                },
            )
        if self.on_change:
            await self.on_change()
        return code

    async def book_tour(
        self,
        *,
        tour_id: str,
        guest_name: str,
        guest_phone: str,
        on_date: date,
        party_size: int,
    ) -> tuple[str, Tour, int]:
        tour = TOURS.get(tour_id)
        if tour is None:
            raise NotFound(f"no such tour: {tour_id} - options: {', '.join(TOURS)}")
        if on_date < TODAY:
            raise Unavailable(f"{on_date.isoformat()} is in the past")
        if party_size > tour.max_party:
            raise Unavailable(f"{tour.name} takes at most {tour.max_party} guests")
        total = tour.flat_price or (tour.price_per_person or 0) * party_size
        code = shortuuid("TUR-")
        with self.connection as conn:
            _insert(
                conn,
                "tour_bookings",
                {
                    "code": code,
                    "tour_id": tour_id,
                    "guest_name": guest_name,
                    "guest_phone": "".join(c for c in guest_phone if c.isdigit()),
                    "date": on_date.isoformat(),
                    "party_size": party_size,
                    "total": total,
                },
            )
        if self.on_change:
            await self.on_change()
        return code, tour, total

    async def book_spa_appointment(
        self,
        *,
        service_id: str,
        guest_name: str,
        guest_phone: str,
        on_date: date,
        at_time: time,
        party_size: int,
    ) -> tuple[str, SpaService, int]:
        service = SPA_SERVICES.get(service_id)
        if service is None:
            raise NotFound(
                f"no such spa service: {service_id} - options: {', '.join(SPA_SERVICES)}"
            )
        if on_date < TODAY:
            raise Unavailable(f"{on_date.isoformat()} is in the past")
        if party_size > service.max_party:
            raise Unavailable(f"{service.name} takes at most {service.max_party} guests")
        total = service.price * party_size
        code = shortuuid("SPA-")
        with self.connection as conn:
            _insert(
                conn,
                "spa_bookings",
                {
                    "code": code,
                    "service_id": service_id,
                    "guest_name": guest_name,
                    "guest_phone": "".join(c for c in guest_phone if c.isdigit()),
                    "date": on_date.isoformat(),
                    "time": at_time.isoformat(),
                    "party_size": party_size,
                    "total": total,
                },
            )
        if self.on_change:
            await self.on_change()
        return code, service, total

    async def book_business_center(
        self,
        *,
        service_id: str,
        guest_name: str,
        guest_phone: str,
        on_date: date,
        at_time: time,
        duration_hours: int,
    ) -> tuple[str, BusinessCenterService, int]:
        service = BUSINESS_CENTER_SERVICES.get(service_id)
        if service is None:
            raise NotFound(
                f"no such service: {service_id} - options: {', '.join(BUSINESS_CENTER_SERVICES)}"
            )
        if on_date < TODAY:
            raise Unavailable(f"{on_date.isoformat()} is in the past")
        if duration_hours > service.max_hours:
            raise Unavailable(f"{service.name} is booked for at most {service.max_hours} hours")
        total = service.flat_price or (service.price_per_hour or 0) * duration_hours
        code = shortuuid("BIZ-")
        with self.connection as conn:
            _insert(
                conn,
                "business_center_bookings",
                {
                    "code": code,
                    "service_id": service_id,
                    "guest_name": guest_name,
                    "guest_phone": "".join(c for c in guest_phone if c.isdigit()),
                    "date": on_date.isoformat(),
                    "time": at_time.isoformat(),
                    "duration_hours": duration_hours,
                    "total": total,
                },
            )
        if self.on_change:
            await self.on_change()
        return code, service, total

    async def order_flowers(
        self,
        *,
        arrangement_id: str,
        guest_name: str,
        guest_phone: str,
        deliver_to: str,
        on_date: date,
        card_message: str,
    ) -> tuple[str, FloralArrangement, int]:
        arrangement = FLORIST_ARRANGEMENTS.get(arrangement_id)
        if arrangement is None:
            raise NotFound(
                f"no such arrangement: {arrangement_id} - options: "
                f"{', '.join(FLORIST_ARRANGEMENTS)}"
            )
        if on_date < TODAY:
            raise Unavailable(f"{on_date.isoformat()} is in the past")
        total = arrangement.price
        code = shortuuid("FLR-")
        with self.connection as conn:
            _insert(
                conn,
                "florist_orders",
                {
                    "code": code,
                    "arrangement_id": arrangement_id,
                    "guest_name": guest_name,
                    "guest_phone": "".join(c for c in guest_phone if c.isdigit()),
                    "deliver_to": deliver_to,
                    "date": on_date.isoformat(),
                    "message": card_message,
                    "total": total,
                },
            )
        if self.on_change:
            await self.on_change()
        return code, arrangement, total

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
        if self.on_change:
            await self.on_change()
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
        if self.on_change:
            await self.on_change()
        return code

    async def request_flight_reconfirmation(
        self,
        *,
        room: str,
        airline: str,
        flight_number: str,
        flight_date: date,
        booking_reference: str,
        seat_check: bool,
    ) -> str:
        room_id = self._require_room(room)
        code = shortuuid("FLT-")
        with self.connection as conn:
            _insert(
                conn,
                "flight_reconfirmations",
                {
                    "code": code,
                    "room_id": room_id,
                    "airline": airline.strip().title(),
                    # spoken codes arrive with unpredictable spaces/dashes
                    "flight_number": "".join(c for c in flight_number if c.isalnum()).upper(),
                    "flight_date": flight_date.isoformat(),
                    "booking_reference": "".join(
                        c for c in booking_reference if c.isalnum()
                    ).upper(),
                    "seat_check": int(seat_check),
                },
            )
        if self.on_change:
            await self.on_change()
        return code

    async def book_airport_car(
        self,
        *,
        room: str,
        pickup_date: date,
        pickup_time: time,
        passengers: int,
    ) -> str:
        room_id = self._require_room(room)
        if pickup_date < TODAY:
            raise Unavailable(f"{pickup_date.isoformat()} is in the past")
        code = shortuuid("CAR-")
        with self.connection as conn:
            _insert(
                conn,
                "airport_cars",
                {
                    "code": code,
                    "room_id": room_id,
                    "pickup_date": pickup_date.isoformat(),
                    "pickup_time": pickup_time.isoformat(),
                    "passengers": passengers,
                },
            )
        if self.on_change:
            await self.on_change()
        return code

    async def dispatch_emergency(self, *, room: str, kind: str, situation: str) -> str:
        # A bad kind is an invalid argument, not a missing entity - keep it distinct
        # from the room's NotFound so the tool can't misreport it as a bad room number.
        if kind not in get_args(EmergencyKind):
            raise ValueError(
                f"unknown emergency kind: {kind} - options: {', '.join(get_args(EmergencyKind))}"
            )
        room_id = self._require_room(room)
        code = shortuuid("EMG-")
        with self.connection as conn:
            _insert(
                conn,
                "emergency_dispatches",
                {"code": code, "room_id": room_id, "kind": kind, "situation": situation},
            )
        if self.on_change:
            await self.on_change()
        return code

    async def room_conflict(self, *, booking_code: str) -> tuple[date, date] | None:
        """The overlap window if another confirmed booking holds this booking's room."""
        row = self.connection.execute(_SQL_ROOM_CONFLICT, {"code": booking_code}).fetchone()
        if not row:
            return None
        return date.fromisoformat(row[0]), date.fromisoformat(row[1])

    async def resolve_room_conflict(self, *, booking_code: str) -> ConflictResolution:
        """The house re-accommodation procedure, in fixed order: try to move the
        booking to a free room of the same or higher category for the whole
        (remaining) stay - an upgrade is free - and only when nothing in the
        house fits, arrange a walk: tonight at the partner hotel on us, back in
        the original room from the next day."""
        conn = self.connection
        booking = conn.execute(
            "SELECT room_id, check_in, check_out, guests FROM hotel_bookings"
            " WHERE code = :code AND status = 'confirmed'",
            {"code": booking_code},
        ).fetchone()
        if not booking:
            raise NotFound(f"booking not found: {booking_code}")
        if await self.room_conflict(booking_code=booking_code) is None:
            raise Unavailable("no room conflict on this booking - nothing to resolve")
        room_id, check_in, check_out, guests = booking
        original = conn.execute(
            "SELECT smoking, nightly_rate FROM hotel_rooms WHERE id = :id", {"id": room_id}
        ).fetchone()
        start = max(date.fromisoformat(check_in), TODAY)

        candidate = conn.execute(
            _SQL_FREE_BETTER_ROOM,
            {
                "exclude_room": room_id,
                "exclude_code": booking_code,
                "guests": guests,
                "smoking": original[0],
                "min_rate": original[1],
                "check_in": start.isoformat(),
                "check_out": check_out,
            },
        ).fetchone()

        if candidate:
            new_room, new_type, new_view, new_rate = candidate
            with conn:
                # the rate on the booking doesn't change - a forced move is never
                # the guest's cost, so an upgrade rides at the original total
                _update(conn, "hotel_bookings", {"room_id": new_room}, {"code": booking_code})
            if self.on_change:
                await self.on_change()
            return ConflictResolution(
                moved_to=new_room,
                moved_to_type=new_type,
                moved_to_view=new_view,
                upgraded=new_rate > original[1],
            )

        return_date = start + timedelta(days=1)
        with conn:
            _insert(
                conn,
                "walk_arrangements",
                {
                    "code": shortuuid("WLK-"),
                    "booking_code": booking_code,
                    "partner_hotel": WALK_PARTNER_HOTEL,
                    "return_date": return_date.isoformat(),
                },
            )
        if self.on_change:
            await self.on_change()
        return ConflictResolution(walk_partner=WALK_PARTNER_HOTEL, walk_return_date=return_date)

    def _require_room(self, room: str) -> str:
        """Normalize a spoken room number ("304") to its id and require it exists."""
        room_id = room.strip().upper()
        if not room_id.startswith("RM_"):
            room_id = f"RM_{room_id}"
        if not self.connection.execute(
            "SELECT 1 FROM hotel_rooms WHERE id = :id", {"id": room_id}
        ).fetchone():
            raise NotFound(f"no such room: {room}")
        return room_id

    async def take_guest_message(
        self,
        *,
        recipient: str,
        caller_name: str,
        caller_phone: str,
        message: str,
    ) -> str:
        """Record a message addressed to a (possibly) in-house guest. Whether the
        recipient actually has a stay here is resolved internally and never
        returned, so the agent taking the message cannot leak guest presence."""
        code = shortuuid("MSG-")
        conn = self.connection
        with conn:
            in_house = conn.execute(
                "SELECT first_name || ' ' || last_name FROM hotel_bookings"
                " WHERE status = 'confirmed'"
                " AND LOWER(first_name || ' ' || last_name) = LOWER(TRIM(:name))"
                " AND check_in <= :today AND check_out > :today",
                {"name": recipient, "today": TODAY.isoformat()},
            ).fetchone()
            _insert(
                conn,
                "guest_messages",
                {
                    "code": code,
                    # matched messages take the registered guest's casing so the
                    # stored name doesn't depend on how the caller's was heard
                    "recipient": in_house[0] if in_house else recipient,
                    "caller_name": caller_name,
                    "caller_phone": "".join(c for c in caller_phone if c.isdigit()),
                    "message": message,
                    "status": "delivered" if in_house else "undeliverable",
                },
            )
        if self.on_change:
            await self.on_change()
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
        if self.on_change:
            await self.on_change()
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
        if self.on_change:
            await self.on_change()
        return case_number


def _install_schema(conn: apsw.Connection) -> None:
    # Views DROP+CREATE so they pick up the current TODAY each time.
    for _ in conn.execute(SCHEMA):
        pass
    for _ in conn.execute(VIEWS):
        pass


_SQL_ROOM_CONFLICT = """
SELECT MAX(b.check_in, a.check_in), MIN(b.check_out, a.check_out)
FROM hotel_bookings a
JOIN hotel_bookings b
  ON b.room_id = a.room_id AND b.code != a.code AND b.status = 'confirmed'
 AND b.check_in < a.check_out AND b.check_out > a.check_in
WHERE a.code = :code AND a.status = 'confirmed'
LIMIT 1
"""

# A room that can absorb a conflicted booking for its whole (remaining) stay:
# fits the party, matches smoking, same or higher category (rate), cheapest first.
_SQL_FREE_BETTER_ROOM = """
SELECT r.id, r.type, r.room_view, r.nightly_rate
FROM hotel_rooms r
WHERE r.id != :exclude_room
  AND r.max_occupancy >= :guests
  AND r.smoking = :smoking
  AND r.nightly_rate >= :min_rate
  AND NOT EXISTS (
    SELECT 1 FROM hotel_bookings b
    WHERE b.room_id = r.id AND b.status = 'confirmed' AND b.code != :exclude_code
      AND b.check_in < :check_out AND b.check_out > :check_in)
ORDER BY r.nightly_rate, r.id
LIMIT 1
"""

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS hotel_rooms (
    id            TEXT    PRIMARY KEY,  -- human room number, e.g. 'RM_201' (floor 2, room 01)
    type          TEXT    NOT NULL CHECK (type IN ('king','queen_2beds','double_queen','suite','penthouse')),
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

CREATE TABLE IF NOT EXISTS restaurant_tables (
    id          INTEGER PRIMARY KEY,
    label       TEXT    NOT NULL UNIQUE,
    capacity    INTEGER NOT NULL CHECK (capacity >= 1),
    location    TEXT    NOT NULL CHECK (location IN ('indoor','terrace','bar')),
    description TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS restaurant_reservations (
    id         INTEGER PRIMARY KEY,
    code       TEXT    NOT NULL UNIQUE,
    table_id   INTEGER NOT NULL REFERENCES restaurant_tables(id),
    first_name TEXT    NOT NULL,
    last_name  TEXT    NOT NULL,
    phone      TEXT    NOT NULL,
    party_size INTEGER NOT NULL CHECK (party_size >= 1),
    date       DATE    NOT NULL,
    time       TIME    NOT NULL,
    notes      TEXT,
    status     TEXT    NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed','cancelled'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_restaurant_slot
    ON restaurant_reservations(date, time, table_id) WHERE status = 'confirmed';

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

CREATE TABLE IF NOT EXISTS tour_bookings (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    tour_id      TEXT    NOT NULL CHECK (tour_id IN ('half_day_city','full_day_city','private_city')),
    guest_name   TEXT    NOT NULL,
    guest_phone  TEXT    NOT NULL,
    date         DATE    NOT NULL,
    party_size   INTEGER NOT NULL CHECK (party_size >= 1),
    total        INTEGER NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed','cancelled'))
);

CREATE TABLE IF NOT EXISTS spa_bookings (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    service_id   TEXT    NOT NULL CHECK (service_id IN ('deep_tissue_massage','signature_facial','personal_training','group_yoga')),
    guest_name   TEXT    NOT NULL,
    guest_phone  TEXT    NOT NULL,
    date         DATE    NOT NULL,
    time         TIME    NOT NULL,
    party_size   INTEGER NOT NULL CHECK (party_size >= 1),
    total        INTEGER NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed','cancelled'))
);

CREATE TABLE IF NOT EXISTS business_center_bookings (
    id             INTEGER PRIMARY KEY,
    code           TEXT    NOT NULL UNIQUE,
    service_id     TEXT    NOT NULL CHECK (service_id IN ('meeting_room','secretarial','printing')),
    guest_name     TEXT    NOT NULL,
    guest_phone    TEXT    NOT NULL,
    date           DATE    NOT NULL,
    time           TIME    NOT NULL,
    duration_hours INTEGER NOT NULL CHECK (duration_hours >= 1),
    total          INTEGER NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed','cancelled'))
);

CREATE TABLE IF NOT EXISTS florist_orders (
    id             INTEGER PRIMARY KEY,
    code           TEXT    NOT NULL UNIQUE,
    arrangement_id TEXT    NOT NULL CHECK (arrangement_id IN ('bouquet','roses','centerpiece')),
    guest_name     TEXT    NOT NULL,
    guest_phone    TEXT    NOT NULL,
    deliver_to     TEXT    NOT NULL,
    date           DATE    NOT NULL,
    message        TEXT    NOT NULL DEFAULT '',
    total          INTEGER NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed','cancelled'))
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

CREATE TABLE IF NOT EXISTS do_not_disturb (
    id      INTEGER PRIMARY KEY,
    code    TEXT    NOT NULL UNIQUE,
    room_id TEXT    NOT NULL REFERENCES hotel_rooms(id),
    status  TEXT    NOT NULL DEFAULT 'active'
);

-- Reference data (read-only): preferences remembered from a returning guest's past
-- stays. The agent reads this to personalize; it never writes here.
CREATE TABLE IF NOT EXISTS guest_history (
    id          INTEGER PRIMARY KEY,
    last_name   TEXT    NOT NULL,
    preferences TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS flight_reconfirmations (
    id                INTEGER PRIMARY KEY,
    code              TEXT    NOT NULL UNIQUE,
    room_id           TEXT    NOT NULL REFERENCES hotel_rooms(id),
    airline           TEXT    NOT NULL,
    flight_number     TEXT    NOT NULL,
    flight_date       DATE    NOT NULL,
    booking_reference TEXT    NOT NULL,
    seat_check        BOOLEAN NOT NULL DEFAULT 0,
    status            TEXT    NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending','confirmed','problem'))
);

CREATE TABLE IF NOT EXISTS airport_cars (
    id          INTEGER PRIMARY KEY,
    code        TEXT    NOT NULL UNIQUE,
    room_id     TEXT    NOT NULL REFERENCES hotel_rooms(id),
    pickup_date DATE    NOT NULL,
    pickup_time TIME    NOT NULL,
    passengers  INTEGER NOT NULL CHECK (passengers >= 1),
    status      TEXT    NOT NULL DEFAULT 'booked' CHECK (status IN ('booked','cancelled'))
);

CREATE TABLE IF NOT EXISTS emergency_dispatches (
    id        INTEGER PRIMARY KEY,
    code      TEXT    NOT NULL UNIQUE,
    room_id   TEXT    NOT NULL REFERENCES hotel_rooms(id),
    kind      TEXT    NOT NULL DEFAULT 'medical' CHECK (kind IN ('medical','fire','security')),
    situation TEXT    NOT NULL,
    status    TEXT    NOT NULL DEFAULT 'dispatched' CHECK (status IN ('dispatched','resolved'))
);

CREATE TABLE IF NOT EXISTS walk_arrangements (
    id            INTEGER PRIMARY KEY,
    code          TEXT    NOT NULL UNIQUE,
    booking_code  TEXT    NOT NULL REFERENCES hotel_bookings(code),
    partner_hotel TEXT    NOT NULL,
    return_date   DATE    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'arranged' CHECK (status IN ('arranged','completed'))
);

CREATE TABLE IF NOT EXISTS wakeup_calls (
    id         INTEGER PRIMARY KEY,
    code       TEXT    NOT NULL UNIQUE,
    room_id    TEXT    NOT NULL REFERENCES hotel_rooms(id),
    guest_name TEXT    NOT NULL,
    date       DATE    NOT NULL,
    time       TIME    NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'scheduled'
                       CHECK (status IN ('scheduled','completed','cancelled'))
);

CREATE TABLE IF NOT EXISTS guest_messages (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    recipient    TEXT    NOT NULL,
    caller_name  TEXT    NOT NULL,
    caller_phone TEXT    NOT NULL,
    message      TEXT    NOT NULL,
    status       TEXT    NOT NULL CHECK (status IN ('delivered','undeliverable'))
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

CREATE TABLE IF NOT EXISTS lk_descriptions (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL
);
"""

VIEWS = f"""
DROP VIEW IF EXISTS hotel_room_status;
CREATE VIEW hotel_room_status AS
SELECT r.id AS room_number, r.type, r.room_view, r.max_occupancy, r.nightly_rate,
       CASE WHEN b.code IS NULL THEN 'available' ELSE 'occupied' END AS status,
       b.code AS current_booking, b.first_name, b.last_name, b.check_out AS free_after
FROM hotel_rooms r
LEFT JOIN hotel_bookings b
    ON b.room_id = r.id AND b.status = 'confirmed'
   AND b.check_in <= '{TODAY.isoformat()}' AND b.check_out > '{TODAY.isoformat()}'
ORDER BY r.id;

DROP VIEW IF EXISTS restaurant_table_status;
CREATE VIEW restaurant_table_status AS
WITH slots(t) AS (
    VALUES ('17:30:00'),('18:00:00'),('18:30:00'),('19:00:00'),
           ('19:30:00'),('20:00:00'),('20:30:00'),('21:00:00')
),
grid AS (
    SELECT rt.id, rt.label, rt.capacity, rt.location, s.t,
           CASE WHEN r.code IS NULL THEN '✓' ELSE '✗' END AS cell
    FROM restaurant_tables rt
    CROSS JOIN slots s
    LEFT JOIN restaurant_reservations r
        ON r.table_id = rt.id AND r.status = 'confirmed'
       AND r.date = '{TODAY.isoformat()}' AND r.time = s.t
)
SELECT label, capacity, location,
       MAX(CASE WHEN t='17:30:00' THEN cell END) AS "5:30 PM",
       MAX(CASE WHEN t='18:00:00' THEN cell END) AS "6:00 PM",
       MAX(CASE WHEN t='18:30:00' THEN cell END) AS "6:30 PM",
       MAX(CASE WHEN t='19:00:00' THEN cell END) AS "7:00 PM",
       MAX(CASE WHEN t='19:30:00' THEN cell END) AS "7:30 PM",
       MAX(CASE WHEN t='20:00:00' THEN cell END) AS "8:00 PM",
       MAX(CASE WHEN t='20:30:00' THEN cell END) AS "8:30 PM",
       MAX(CASE WHEN t='21:00:00' THEN cell END) AS "9:00 PM"
FROM grid GROUP BY id ORDER BY label;
"""

_BOOKING_COLS = "b.id, b.code, b.room_id, r.type AS room_type, r.smoking, r.nightly_rate, b.first_name, b.last_name, b.email, b.phone, b.check_in, b.check_out, b.guests, b.extras, b.total, b.card_last4, b.status, b.late_arrival_note"
# Names that _row_to_booking maps the SELECT columns onto - derived from the
# dataclass so adding a field in one place keeps them aligned.
_BOOKING_COL_NAMES: tuple[str, ...] = tuple(f.name for f in fields(RoomBooking))
_RESERVATION_COLS: tuple[str, ...] = tuple(f.name for f in fields(RestaurantReservation))


_SQL_FREE_ROOM = """
SELECT id, nightly_rate FROM hotel_rooms
WHERE type = :room_type AND smoking = :smoking AND max_occupancy >= :guests
  AND (:view IS NULL OR room_view = :view)
  AND NOT EXISTS (
    SELECT 1 FROM hotel_bookings b
    WHERE b.room_id = hotel_rooms.id AND b.status = 'confirmed'
      AND (:exclude IS NULL OR b.code != :exclude)
      AND NOT (b.check_out <= :check_in OR b.check_in >= :check_out))
ORDER BY CASE WHEN id = :prefer THEN 0 ELSE 1 END, id LIMIT 1
"""

_SQL_AVAILABILITY = """
SELECT r.type, r.nightly_rate, GROUP_CONCAT(DISTINCT r.room_view)
FROM hotel_rooms r
WHERE r.max_occupancy >= :guests
  AND (:smoking IS NULL OR r.smoking = :smoking)
  AND NOT EXISTS (
    SELECT 1 FROM hotel_bookings b
    WHERE b.room_id = r.id AND b.status = 'confirmed'
      AND (:exclude IS NULL OR b.code != :exclude)
      AND NOT (b.check_out <= :check_in OR b.check_in >= :check_out))
GROUP BY r.type ORDER BY r.nightly_rate
"""

_SQL_FREE_TABLE = """
SELECT t.id FROM restaurant_tables t
WHERE t.capacity >= :party_size
  AND NOT EXISTS (
    SELECT 1 FROM restaurant_reservations r
    WHERE r.table_id = t.id AND r.status = 'confirmed'
      AND r.date = :date AND r.time = :time)
ORDER BY t.capacity, t.id LIMIT 1
"""

# Pick a table for a modified reservation: prefer the reservation's CURRENT
# table when it's still big enough and free at the new slot (so a same-evening
# time shift keeps the same table_id and table_location), otherwise fall back to
# the next free table. Excludes the reservation being modified from the conflict
# check so shifting in place doesn't collide with itself.
_SQL_FREE_TABLE_FOR_MODIFY = """
SELECT t.id FROM restaurant_tables t
WHERE t.capacity >= :party_size
  AND NOT EXISTS (
    SELECT 1 FROM restaurant_reservations r
    WHERE r.table_id = t.id AND r.status = 'confirmed' AND r.code != :code
      AND r.date = :date AND r.time = :time)
ORDER BY (t.id = :current_table_id) DESC, t.capacity, t.id LIMIT 1
"""

_SQL_DINING_AVAILABILITY = """
WITH slots(slot) AS (
    VALUES ('17:30:00'),('18:00:00'),('18:30:00'),('19:00:00'),
           ('19:30:00'),('20:00:00'),('20:30:00'),('21:00:00')
)
SELECT slots.slot, rt.id
FROM slots
LEFT JOIN restaurant_tables rt
  ON rt.capacity >= :party_size
  AND NOT EXISTS (
    SELECT 1 FROM restaurant_reservations r
    WHERE r.table_id = rt.id AND r.status = 'confirmed'
      AND r.date = :date AND r.time = slots.slot)
ORDER BY slots.slot, rt.capacity, rt.id
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

_SQL_FIND_RESERVATION = f"""
SELECT {", ".join(_RESERVATION_COLS)} FROM restaurant_reservations
WHERE LOWER(last_name) = LOWER(:last_name)
  AND (:code IS NULL OR REPLACE(code, '-', '') = REPLACE(:code, '-', ''))
  AND (:date IS NULL OR date = :date)
LIMIT 1
"""

_SQL_RESERVATION_BY_CODE = f"""
SELECT {", ".join(_RESERVATION_COLS)} FROM restaurant_reservations
WHERE code = :code LIMIT 1
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


def _row_to_reservation(row: tuple[Any, ...] | None) -> RestaurantReservation | None:
    if row is None:
        return None
    d = dict(zip(_RESERVATION_COLS, row, strict=True))
    d["date"], d["time"] = date.fromisoformat(d["date"]), time.fromisoformat(d["time"])
    return RestaurantReservation(**d)
