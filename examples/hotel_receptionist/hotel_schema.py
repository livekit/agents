from __future__ import annotations

from dataclasses import fields
from datetime import date, time
from typing import Any

import apsw

from .hotel import RestaurantReservation, RoomBooking


def _install_schema(conn: apsw.Connection, today: date) -> None:
    # Views DROP+CREATE so they pick up the session's date each time.
    for _ in conn.execute(SCHEMA):
        pass
    for _ in conn.execute(_views_sql(today)):
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


def _views_sql(today: date) -> str:
    return f"""
DROP VIEW IF EXISTS hotel_room_status;
CREATE VIEW hotel_room_status AS
SELECT r.id AS room_number, r.type, r.room_view, r.max_occupancy, r.nightly_rate,
       CASE WHEN b.code IS NULL THEN 'available' ELSE 'occupied' END AS status,
       b.code AS current_booking, b.first_name, b.last_name, b.check_out AS free_after
FROM hotel_rooms r
LEFT JOIN hotel_bookings b
    ON b.room_id = r.id AND b.status = 'confirmed'
   AND b.check_in <= '{today.isoformat()}' AND b.check_out > '{today.isoformat()}'
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
       AND r.date = '{today.isoformat()}' AND r.time = s.t
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
ORDER BY CASE WHEN id = :prefer THEN 0 ELSE 1 END, nightly_rate, id LIMIT 1
"""

_SQL_AVAILABILITY = """
SELECT r.type, r.room_view, MIN(r.nightly_rate), MAX(r.nightly_rate)
FROM hotel_rooms r
WHERE r.max_occupancy >= :guests
  AND (:smoking IS NULL OR r.smoking = :smoking)
  AND NOT EXISTS (
    SELECT 1 FROM hotel_bookings b
    WHERE b.room_id = r.id AND b.status = 'confirmed'
      AND (:exclude IS NULL OR b.code != :exclude)
      AND NOT (b.check_out <= :check_in OR b.check_in >= :check_out))
GROUP BY r.type, r.room_view
ORDER BY r.type, r.room_view
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
