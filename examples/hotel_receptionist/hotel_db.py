from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, time
from itertools import groupby
from pathlib import Path
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


def extras_total(extras: list[str], nights: int) -> int:
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

TODAY = date(2026, 1, 16)
MAX_PARTY_SIZE = 6

RoomType = Literal["king", "queen_2beds", "double_queen", "suite", "penthouse"]


@dataclass
class RoomTypeAvailability:
    type: RoomType
    nightly_rate: int
    available_count: int
    sample_view: str


@dataclass
class RoomBooking:
    id: int
    code: str
    room_id: int
    first_name: str
    last_name: str
    email: str
    phone: str
    check_in: date
    check_out: date
    guests: int
    extras: list[str]
    total: int
    card_last4: str
    status: Literal["confirmed", "cancelled"]

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


class Unavailable(Exception):
    pass


class NotFound(Exception):
    pass


def invoice_line_items(*, nights: int, room_subtotal: int, extras: list[str], tax: int) -> list[LineItem]:  # fmt: skip
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


OnChange = Callable[[], Awaitable[None]]


class HotelDB:
    def __init__(self, db_path: str | Path, *, on_change: OnChange | None = None) -> None:
        self.db_path = str(db_path)
        self.on_change = on_change
        self._conn: apsw.Connection | None = None

    @property
    def connection(self) -> apsw.Connection:
        if self._conn is None:
            raise RuntimeError("HotelDB.initialize() must be called first")
        return self._conn

    async def initialize(self) -> None:
        self._conn = apsw.Connection(self.db_path)
        for _ in self._conn.execute(_SCHEMA):  # apsw runs a script lazily; exhaust to run all
            pass
        for _ in self._conn.execute(_VIEWS):
            pass
        logger.info("hotel db initialized at %s", self.db_path)

    async def aclose(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def list_room_types_available(self, *, check_in: date, check_out: date, guests: int, smoking: bool | None = None) -> list[RoomTypeAvailability]:  # fmt: skip
        rows = self.connection.execute(_SQL_AVAILABILITY, {"guests": guests, "smoking": int(smoking) if smoking is not None else None, "check_in": check_in.isoformat(), "check_out": check_out.isoformat()})  # fmt: skip
        return [RoomTypeAvailability(*r) for r in rows]

    async def list_restaurant_availability(self, *, on_date: date, party_size: int) -> list[TimeSlot]:  # fmt: skip
        rows = self.connection.execute(_SQL_DINING_AVAILABILITY, {"party_size": party_size, "date": on_date.isoformat()})  # fmt: skip
        return [
            TimeSlot(time.fromisoformat(slot), [tid for _, tid in group if tid is not None])
            for slot, group in groupby(rows, key=lambda row: row[0])
        ]

    async def find_booking(self, *, last_name: str, confirmation_code: str | None = None, email: str | None = None, card_last4: str | None = None) -> RoomBooking | None:  # fmt: skip
        return _row_to_booking(self.connection.execute(_SQL_FIND_BOOKING, {"last_name": last_name, "code": confirmation_code.upper() if confirmation_code else None, "email": email, "card_last4": card_last4}).fetchone())  # fmt: skip

    async def find_restaurant_reservation(self, *, last_name: str, confirmation_code: str | None = None, on_date: date | None = None) -> RestaurantReservation | None:  # fmt: skip
        return _row_to_reservation(self.connection.execute(_SQL_FIND_RESERVATION, {"last_name": last_name, "code": confirmation_code.upper() if confirmation_code else None, "date": on_date.isoformat() if on_date else None}).fetchone())  # fmt: skip

    async def get_invoice(self, booking_code: str) -> Invoice:
        row = self.connection.execute(_SQL_GET_INVOICE, {"code": booking_code}).fetchone()
        if not row:
            raise NotFound(f"no invoice for {booking_code}")
        invoice_id, line_items_json, subtotal, taxes, total, paid = row
        return Invoice(
            id=invoice_id, booking_code=booking_code,
            line_items=[LineItem(**li) for li in json.loads(line_items_json)],
            subtotal=subtotal, taxes=taxes, total=total, paid=bool(paid),
        )  # fmt: skip

    async def book_room(self, *, room_type: RoomType, smoking: bool, guests: int, check_in: date, check_out: date, first_name: str, last_name: str, email: str, phone: str, card_last4: str, extras: list[str], subtotal: int, taxes: int, line_items: list[LineItem]) -> RoomBooking:  # fmt: skip
        params = {
            "code": shortuuid("HTL-"),
            "room_type": room_type,
            "smoking": int(smoking),
            "guests": guests,
            "check_in": check_in.isoformat(),
            "check_out": check_out.isoformat(),
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "card_last4": card_last4,
            "extras": ",".join(extras),
            "subtotal": subtotal,
            "taxes": taxes,
            "total": subtotal + taxes,
            "line_items": json.dumps([li.__dict__ for li in line_items]),
        }
        conn = self.connection
        with conn:
            row = conn.execute(_SQL_FREE_ROOM, params).fetchone()
            if not row:
                raise Unavailable(f"sold out: {room_type}")
            params["room_id"] = row[0]
            conn.execute(_SQL_INSERT_BOOKING, params)
            booking_id = conn.last_insert_rowid()
            conn.execute(_SQL_INSERT_INVOICE, params)
        if self.on_change:
            await self.on_change()
        return RoomBooking(id=booking_id, code=params["code"], room_id=params["room_id"], first_name=first_name, last_name=last_name, email=email, phone=phone, check_in=check_in, check_out=check_out, guests=guests, extras=extras, total=params["total"], card_last4=card_last4, status="confirmed")  # fmt: skip

    async def cancel_room_booking(self, booking_code: str) -> None:
        conn = self.connection
        conn.execute("UPDATE hotel_bookings SET status = 'cancelled' WHERE code = :code", {"code": booking_code})  # fmt: skip
        if conn.changes() == 0:
            raise NotFound(f"booking not found: {booking_code}")
        if self.on_change:
            await self.on_change()

    async def book_restaurant(self, *, first_name: str, last_name: str, phone: str, party_size: int, on_date: date, at_time: time, notes: str | None = None) -> RestaurantReservation:  # fmt: skip
        conn = self.connection
        row = conn.execute(_SQL_FREE_TABLE, {"party_size": party_size, "date": on_date.isoformat(), "time": at_time.isoformat()}).fetchone()  # fmt: skip
        if not row:
            raise Unavailable(f"restaurant full: {on_date} {at_time}")
        table_id = row[0]
        code = shortuuid("RES-")
        reservation_id = _insert(conn, "restaurant_reservations", {
            "code": code, "table_id": table_id,
            "first_name": first_name, "last_name": last_name, "phone": phone,
            "party_size": party_size,
            "date": on_date.isoformat(), "time": at_time.isoformat(), "notes": notes,
        })  # fmt: skip
        if self.on_change:
            await self.on_change()
        return RestaurantReservation(id=reservation_id, code=code, table_id=table_id, first_name=first_name, last_name=last_name, phone=phone, party_size=party_size, date=on_date, time=at_time, notes=notes, status="confirmed")  # fmt: skip

    async def cancel_restaurant_reservation(self, code: str) -> None:
        conn = self.connection
        conn.execute("UPDATE restaurant_reservations SET status = 'cancelled' WHERE code = :code AND status = 'confirmed'", {"code": code})  # fmt: skip
        if conn.changes() == 0:
            raise NotFound(f"reservation not found: {code}")
        if self.on_change:
            await self.on_change()

    async def file_dispute(self, *, booking_code: str, line_item: str, amount_cents: int, category: DisputeCategory, caller_note: str, outcome: str, refund_amount: int) -> str:  # fmt: skip
        case_number = shortuuid("DSP-")
        conn = self.connection
        with conn:
            _insert(conn, "hotel_disputes", {
                "case_number": case_number, "booking_code": booking_code, "line_item": line_item,
                "amount": amount_cents, "category": category, "caller_note": caller_note,
                "outcome": outcome, "refund_amount": refund_amount,
            })  # fmt: skip
            if refund_amount > 0:
                conn.execute("UPDATE hotel_invoices SET total = total - :refund WHERE booking_code = :code", {"refund": refund_amount, "code": booking_code})  # fmt: skip
        if self.on_change:
            await self.on_change()
        return case_number


_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS hotel_rooms (
    id            INTEGER PRIMARY KEY,
    room_number   TEXT    NOT NULL UNIQUE,
    type          TEXT    NOT NULL CHECK (type IN ('king','queen_2beds','double_queen','suite','penthouse')),
    nightly_rate  INTEGER NOT NULL,
    max_occupancy INTEGER NOT NULL,
    smoking       BOOLEAN NOT NULL DEFAULT 0,
    pets_allowed  BOOLEAN NOT NULL DEFAULT 0,
    room_view     TEXT    NOT NULL CHECK (room_view IN ('city','ocean','garden','interior'))
);

CREATE TABLE IF NOT EXISTS hotel_bookings (
    id         INTEGER PRIMARY KEY,
    code       TEXT    NOT NULL UNIQUE,
    room_id    INTEGER NOT NULL REFERENCES hotel_rooms(id),
    first_name TEXT    NOT NULL,
    last_name  TEXT    NOT NULL,
    email      TEXT    NOT NULL,
    phone      TEXT    NOT NULL,
    check_in   DATE    NOT NULL,
    check_out  DATE    NOT NULL,
    guests     INTEGER NOT NULL CHECK (guests >= 1),
    extras     TEXT    NOT NULL DEFAULT '',
    total      INTEGER NOT NULL,
    card_last4 TEXT    NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed','cancelled')),
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
    category      TEXT    NOT NULL CHECK (category IN ('minibar','room_service_restaurant','damage_cleaning','late_checkout_fee','cancellation_fee','double_charge_billing_error','other')),
    caller_note   TEXT    NOT NULL,
    outcome       TEXT    NOT NULL CHECK (outcome IN ('auto_refunded','credit_offered','explained_no_action','goodwill_waived','escalated_to_manager','accounting_ticket_opened','open')),
    refund_amount INTEGER NOT NULL DEFAULT 0,
    status        TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved','rejected'))
);

CREATE TABLE IF NOT EXISTS lk_descriptions (
    name        TEXT PRIMARY KEY,
    description TEXT NOT NULL
);
"""

_VIEWS = f"""
DROP VIEW IF EXISTS hotel_room_status;
CREATE VIEW hotel_room_status AS
SELECT r.room_number, r.type, r.room_view, r.max_occupancy, r.nightly_rate,
       CASE WHEN b.code IS NULL THEN 'available' ELSE 'occupied' END AS status,
       b.code AS current_booking, b.first_name, b.last_name, b.check_out AS free_after
FROM hotel_rooms r
LEFT JOIN hotel_bookings b
    ON b.room_id = r.id AND b.status = 'confirmed'
   AND b.check_in <= '{TODAY.isoformat()}' AND b.check_out > '{TODAY.isoformat()}'
ORDER BY r.room_number;

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

_BOOKING_COLS = "id, code, room_id, first_name, last_name, email, phone, check_in, check_out, guests, extras, total, card_last4, status"  # fmt: skip
_RESERVATION_COLS = "id, code, table_id, first_name, last_name, phone, party_size, date, time, notes, status"  # fmt: skip


_SQL_FREE_ROOM = """
SELECT id, nightly_rate FROM hotel_rooms
WHERE type = :room_type AND smoking = :smoking AND max_occupancy >= :guests
  AND NOT EXISTS (
    SELECT 1 FROM hotel_bookings b
    WHERE b.room_id = hotel_rooms.id AND b.status = 'confirmed'
      AND NOT (b.check_out <= :check_in OR b.check_in >= :check_out))
ORDER BY id LIMIT 1
"""

_SQL_INSERT_BOOKING = """
INSERT INTO hotel_bookings
  (code, room_id, first_name, last_name, email, phone,
   check_in, check_out, guests, extras, total, card_last4)
VALUES
  (:code, :room_id, :first_name, :last_name, :email, :phone,
   :check_in, :check_out, :guests, :extras, :total, :card_last4)
"""

_SQL_INSERT_INVOICE = """
INSERT INTO hotel_invoices (booking_code, line_items, subtotal, taxes, total, paid)
VALUES (:code, :line_items, :subtotal, :taxes, :total, 0)
"""

_SQL_AVAILABILITY = """
SELECT r.type, r.nightly_rate, COUNT(*), MIN(r.room_view)
FROM hotel_rooms r
WHERE r.max_occupancy >= :guests
  AND (:smoking IS NULL OR r.smoking = :smoking)
  AND NOT EXISTS (
    SELECT 1 FROM hotel_bookings b
    WHERE b.room_id = r.id AND b.status = 'confirmed'
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
SELECT {_BOOKING_COLS} FROM hotel_bookings
WHERE LOWER(last_name) = LOWER(:last_name)
  AND (:code IS NULL OR code = :code)
  AND (:email IS NULL OR LOWER(email) = LOWER(:email))
  AND (:card_last4 IS NULL OR card_last4 = :card_last4)
LIMIT 1
"""

_SQL_FIND_RESERVATION = f"""
SELECT {_RESERVATION_COLS} FROM restaurant_reservations
WHERE LOWER(last_name) = LOWER(:last_name)
  AND (:code IS NULL OR code = :code)
  AND (:date IS NULL OR date = :date)
LIMIT 1
"""

_SQL_GET_INVOICE = "SELECT id, line_items, subtotal, taxes, total, paid FROM hotel_invoices WHERE booking_code = :code"  # fmt: skip


def _insert(conn: apsw.Connection, table: str, row: dict[str, Any]) -> int:
    keys = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT INTO {table} ({keys}) VALUES ({placeholders})", row)
    return conn.last_insert_rowid()


def _row_to_booking(row: tuple[Any, ...] | None) -> RoomBooking | None:
    if row is None:
        return None
    d = dict(zip(_BOOKING_COLS.split(", "), row, strict=True))
    d["check_in"], d["check_out"] = date.fromisoformat(d["check_in"]), date.fromisoformat(d["check_out"])  # fmt: skip
    d["extras"] = [e for e in d["extras"].split(",") if e]
    return RoomBooking(**d)


def _row_to_reservation(row: tuple[Any, ...] | None) -> RestaurantReservation | None:
    if row is None:
        return None
    d = dict(zip(_RESERVATION_COLS.split(", "), row, strict=True))
    d["date"], d["time"] = date.fromisoformat(d["date"]), time.fromisoformat(d["time"])
    return RestaurantReservation(**d)
