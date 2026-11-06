from __future__ import annotations

import os
from dataclasses import dataclass, fields
from datetime import date, time
from itertools import groupby
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


MAX_PARTY_SIZE = 6


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


FollowupKind = Literal[
    "sales_lead",  # group bookings, events, weddings, corporate rates
    "callback",  # caller asked to be called back later
    "other",
]


# Hotel departments a caller can be transferred to. NOT a guest room - the
# operator never connects a caller to a guest (see guest-privacy policy).
TransferDestination = Literal[
    "restaurant",
    "duty_manager",
    "housekeeping",
]


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


class Unavailable(Exception):
    pass


class NotFound(Exception):
    pass


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
        updated = _row_to_reservation(
            conn.execute(_SQL_RESERVATION_BY_CODE, {"code": code}).fetchone()
        )
        if updated is None:
            raise NotFound(f"reservation vanished mid-update: {code}")
        return updated

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
        return code, arrangement, total

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
        return code

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

CREATE TABLE IF NOT EXISTS transfer_calls (
    id           INTEGER PRIMARY KEY,
    code         TEXT    NOT NULL UNIQUE,
    destination  TEXT    NOT NULL CHECK (destination IN ('restaurant','duty_manager','housekeeping')),
    summary      TEXT    NOT NULL DEFAULT '',
    status       TEXT    NOT NULL DEFAULT 'transferred'
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
"""


_RESERVATION_COLS: tuple[str, ...] = tuple(f.name for f in fields(RestaurantReservation))


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


def _row_to_reservation(row: tuple[Any, ...] | None) -> RestaurantReservation | None:
    if row is None:
        return None
    d = dict(zip(_RESERVATION_COLS, row, strict=True))
    d["date"], d["time"] = date.fromisoformat(d["date"]), time.fromisoformat(d["time"])
    return RestaurantReservation(**d)
