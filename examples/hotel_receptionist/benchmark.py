"""Grade a simulation on final DB state (tau-bench style).

A scenario's `userdata.expected_state` is SQL run against a fresh copy of the
seed to build the expected end state; we then diff it against the agent's DB.
The agent is graded on the resulting *state*, not on reproducing the SQL.

The diff is a *denylist*: every column of every transactional table is compared
except an explicit, justified set that genuinely can't match across two correct
runs. Foreign-key surrogates are resolved to their stable attribute (room_id ->
type, table_id -> location) so "which king" doesn't matter but the type does.
Comparison is an order-invariant multiset, so collateral damage (an extra,
missing, or altered row anywhere) still surfaces.
"""

from __future__ import annotations

import collections
from typing import Any

import apsw
from hotel_db import HotelDB

# Transactional tables the agent's tools mutate. Static reference data
# (hotel_rooms, restaurant_tables), the UI table (lk_descriptions), and
# hotel_invoices (fully derived from the booking) are not compared directly.
TRANSACTIONAL_TABLES: tuple[str, ...] = (
    "hotel_bookings",
    "restaurant_reservations",
    "hotel_followups",
    "hotel_disputes",
    "group_inquiries",
    "guest_messages",
    "wakeup_calls",
    "tour_bookings",
    "flight_reconfirmations",
    "airport_cars",
    "emergency_dispatches",
    "walk_arrangements",
)

# The only columns excluded from comparison, by reason:
DENY_COLUMNS = frozenset(
    {
        # surrogate / randomly-minted ids — vary per run and with action order
        "id",
        "code",
        "case_number",
        "booking_code",
        # booking math: tool-computed by compute_invoice from type + dates +
        # extras, which are all already compared — so asserting it would grade the
        # mock's arithmetic, not the agent. (It also varies with which room of a
        # type book_room picks, since seed rates differ within a type.)
        "total",
        "subtotal",
        "taxes",
        "line_items",
        # free text written by the agent / simulated user
        "summary",
        "caller_note",
        "notes",
        "late_arrival_note",
        "message",
        "situation",
    }
)

# Resolve FK surrogate -> stable attribute (correlated subquery, single table).
FK_RESOLVE: dict[tuple[str, str], str] = {
    (
        "hotel_bookings",
        "room_id",
    ): "(SELECT type || '/' || room_view FROM hotel_rooms WHERE id = room_id) AS room_type_view",
    (
        "restaurant_reservations",
        "table_id",
    ): "(SELECT location FROM restaurant_tables WHERE id = table_id) AS table_location",
}


def _select_sql(conn: apsw.Connection, table: str) -> str:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")]
    parts = [FK_RESOLVE.get((table, c), f'"{c}"') for c in cols if c not in DENY_COLUMNS]
    return f'SELECT {", ".join(parts)} FROM "{table}"'  # noqa: S608


def _rows(
    conn: apsw.Connection, sql: str
) -> tuple[list[str], collections.Counter[tuple[Any, ...]]]:
    cur = conn.execute(sql)
    cols: list[str] = []
    counter: collections.Counter[tuple[Any, ...]] = collections.Counter()
    for row in cur:
        if not cols:  # getdescription() is only valid while a row is in flight
            cols = [d[0] for d in cur.getdescription()]
        counter[tuple(row)] += 1
    return cols, counter


def diff_databases(
    expected: apsw.Connection,
    actual: apsw.Connection,
    *,
    tables: tuple[str, ...] = TRANSACTIONAL_TABLES,
) -> list[str]:
    """Order-invariant denylist diff of two hotel DBs. Empty list == states match."""
    diffs: list[str] = []
    for table in tables:
        sql = _select_sql(expected, table)
        ecols, exp = _rows(expected, sql)
        acols, act = _rows(actual, sql)
        cols = ecols or acols
        for row, n in (exp - act).items():
            diffs.append(f"{table}: missing {n}x {dict(zip(cols, row, strict=True))}")
        for row, n in (act - exp).items():
            diffs.append(f"{table}: unexpected {n}x {dict(zip(cols, row, strict=True))}")
    return diffs


async def build_expected(seed_bytes: bytes, expected_state: list[str]) -> HotelDB:
    """Construct the expected end state by applying `expected_state` SQL to a fresh
    seed. The agent's DB is compared against this by *state* (see diff_databases) —
    the agent does NOT have to reproduce these statements. The seed is pinned to a
    fixed date for simulations (HOTEL_TODAY), so dates are plain literals."""
    db = HotelDB.from_bytes(seed_bytes)
    for stmt in expected_state:
        db.connection.execute(stmt)
    return db
