from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, time
from typing import Literal, get_args


@dataclass(frozen=True)
class Pricing:
    breakfast_per_night: int = 2500
    valet_per_night: int = 3500
    late_checkout: int = 4000
    pet_fee: int = 5000
    smoking_cleaning_fee: int = 25000
    tax_rate_pct: int = 12
    cancellation_window_days: int = 2
    cancellation_forfeit_nights: int = 1
    minibar_auto_refund_threshold: int = 2000


PRICING = Pricing()

RoomExtra = Literal["breakfast", "valet", "late_checkout", "pets"]
ALLOWED_EXTRAS: frozenset[str] = frozenset(get_args(RoomExtra))


def cancellation_penalty_applies(*, check_in: date, today: date) -> bool:
    """Whether check-in is fewer than the required calendar days away."""
    return (check_in - today).days < PRICING.cancellation_window_days


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


def normalize_phone(phone: str) -> str:
    return "".join(c for c in phone if c.isdigit()) or phone


def normalize_email(email: str) -> str:
    return email.strip().lower()


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


def format_date(value: date) -> str:
    return f"{value:%A, %B} {value.day}"


def format_month_day(value: date) -> str:
    return f"{value:%B} {value.day}"


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
            f"Our policy is free cancellation at least {PRICING.cancellation_window_days} "
            "calendar days before check-in. Later cancellations forfeit one night. "
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

MAX_PARTY_SIZE = 6

RoomType = Literal["king", "queen_2beds", "double_queen", "suite", "penthouse"]


@dataclass(frozen=True)
class ViewRate:
    view: str
    minimum: int
    maximum: int


@dataclass(frozen=True)
class RoomTypeAvailability:
    type: RoomType
    view_rates: tuple[ViewRate, ...]

    @property
    def views(self) -> tuple[str, ...]:
        return tuple(rate.view for rate in self.view_rates)

    @property
    def lowest_nightly_rate(self) -> int:
        return min(rate.minimum for rate in self.view_rates)


def format_room_type_availability(availability: RoomTypeAvailability) -> str:
    """Format each available view with the nightly rate that actually applies."""
    parts: list[str] = []
    for rate in availability.view_rates:
        price = speak_usd(rate.minimum)
        if rate.maximum != rate.minimum:
            price = f"{price}–{speak_usd(rate.maximum)}"
        parts.append(f"{rate.view} {price}/night")
    return "; ".join(parts)


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
