from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import date, time, timedelta
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


def speak_time(t: time) -> str:
    """A clock time as natural speech, e.g. '7 PM', '6:30 PM'."""
    hour = t.hour % 12 or 12
    suffix = "PM" if t.hour >= 12 else "AM"
    return f"{hour} {suffix}" if t.minute == 0 else f"{hour}:{t.minute:02d} {suffix}"


# Set HOTEL_TODAY=YYYY-MM-DD before import for deterministic sim runs.
TODAY: date = (
    date.fromisoformat(os.environ["HOTEL_TODAY"]) if os.environ.get("HOTEL_TODAY") else date.today()
)


RoomType = Literal["king", "queen_2beds", "suite", "penthouse"]


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


FollowupKind = Literal[
    "housekeeping",  # in-house guest needs something brought or fixed (towels, amenities, maintenance)
    "callback",  # caller asked to be called back later
    "verification_help",  # verification failed; route to a human
    "early_checkout",  # in-house guest wants to leave early; front desk handles
    "lost_and_found",  # guest reports an item left behind; route to housekeeping/lost-and-found
    "other",
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

    async def set_do_not_disturb(self, *, room: str) -> str:
        """Record a Do-Not-Disturb hold on a room and return a reference. The switchboard
        holds the room's calls and messages until it's lifted; emergencies override it."""
        # Normalize to the canonical room id (and reject a mis-heard room) so storage
        # matches every other room-referencing table regardless of how it was spoken.
        room_id = self._require_room(room)
        code = shortuuid("DND-")
        with self.connection as conn:
            _insert(conn, "do_not_disturb", {"code": code, "room_id": room_id})
        return code

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
        return code


def _install_schema(conn: apsw.Connection) -> None:
    for _ in conn.execute(SCHEMA):
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

CREATE TABLE IF NOT EXISTS hotel_followups (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    kind         TEXT    NOT NULL CHECK (kind IN ('housekeeping','sales_lead','identity_change','callback','verification_help','early_checkout','abandoned_booking','lost_and_found','other')),
    caller_name  TEXT    NOT NULL,
    caller_phone TEXT    NOT NULL,
    summary      TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved'))
);

CREATE TABLE IF NOT EXISTS transfer_calls (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    destination  TEXT    NOT NULL CHECK (destination IN ('restaurant','duty_manager','housekeeping')),
    summary      TEXT    NOT NULL DEFAULT '',
    status       TEXT    NOT NULL DEFAULT 'transferred'
);

CREATE TABLE IF NOT EXISTS do_not_disturb (
    id      INTEGER PRIMARY KEY,
    code    TEXT    NOT NULL UNIQUE,
    room_id TEXT    NOT NULL REFERENCES hotel_rooms(id),
    status  TEXT    NOT NULL DEFAULT 'active'
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
"""


_BOOKING_COLS = "b.id, b.code, b.room_id, r.type AS room_type, r.smoking, r.nightly_rate, b.first_name, b.last_name, b.email, b.phone, b.check_in, b.check_out, b.guests, b.extras, b.total, b.card_last4, b.status, b.late_arrival_note"


# Names that _row_to_booking maps the SELECT columns onto - derived from the
# dataclass so adding a field in one place keeps them aligned.
_BOOKING_COL_NAMES: tuple[str, ...] = tuple(f.name for f in fields(RoomBooking))


_SQL_FIND_BOOKING = f"""
SELECT {_BOOKING_COLS} FROM hotel_bookings b
JOIN hotel_rooms r ON r.id = b.room_id
WHERE LOWER(b.last_name) = LOWER(:last_name)
  AND (:code IS NULL OR REPLACE(b.code, '-', '') = REPLACE(:code, '-', ''))
  AND (:email IS NULL OR LOWER(b.email) = LOWER(:email))
  AND (:card_last4 IS NULL OR b.card_last4 = :card_last4)
LIMIT 1
"""


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
