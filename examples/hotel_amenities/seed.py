"""Seed data for the example hotel DB."""

from __future__ import annotations

from datetime import date, time, timedelta

from hotel_db import HotelDB

# fmt: off
# (id (room number, floor+number), type, nightly_rate_cents, max_occupancy, smoking, pets, view)
ROOMS = [
    ("RM_201", "king", 24000, 2, 0, 0, "city"),
    ("RM_202", "king", 26000, 2, 0, 1, "ocean"),
    ("RM_203", "king", 24000, 2, 1, 0, "city"),
    ("RM_204", "queen_2beds", 22000, 4, 0, 0, "city"),
    ("RM_205", "queen_2beds", 22000, 4, 0, 1, "garden"),
    ("RM_206", "queen_2beds", 26000, 4, 0, 0, "ocean"),
    ("RM_301", "king", 28000, 2, 0, 0, "ocean"),
    ("RM_302", "king", 28000, 2, 0, 0, "ocean"),
    ("RM_303", "queen_2beds", 24000, 4, 0, 0, "city"),
    ("RM_304", "queen_2beds", 28000, 4, 0, 1, "ocean"),
    ("RM_401", "suite", 48000, 4, 0, 1, "ocean"),
    ("RM_402", "suite", 52000, 4, 0, 0, "ocean"),
    ("RM_PH", "penthouse", 120000, 6, 0, 1, "ocean"),
]

# (label, capacity, location, description)
TABLES = [
    ("T-01", 2, "indoor", "Window two-top overlooking the harbor"),
    ("T-02", 2, "indoor", "Quiet corner booth, tucked beside the wine wall"),
    ("T-03", 4, "indoor", "Round table beneath the chandelier"),
    ("T-04", 4, "indoor", "Velvet banquette along the main dining wall"),
    ("T-05", 6, "indoor", "Chef's table facing the open kitchen"),
    ("P-01", 2, "terrace", "Intimate table for two at the terrace railing"),
    ("P-02", 4, "terrace", "Terrace table under the string lights"),
    ("P-03", 4, "terrace", "Shaded terrace table by the herb garden"),
    ("B-01", 2, "bar", "High-top at the end of the marble bar"),
    ("B-02", 2, "bar", "Counter seats facing the bartenders"),
]

# (first, last, phone, party, offset_days, hour, minute, code_suffix, table, notes, status)
RESERVATIONS = [
    # Tonight
    ("Marcus", "Bennett", "+1 415 555 0231", 4, 0, 19, 0, "JK90", "T-03", "Birthday", "confirmed"),
    ("Hannah", "Kowalski", "+1 415 555 0244", 2, 0, 20, 30, "LM12", "T-01", "Anniversary", "confirmed"),
    ("Sofía", "García", "+1 415 555 0107", 6, 0, 19, 30, "NP21", "T-05", "Family dinner", "confirmed"),
    ("Diego", "Herrera", "+1 415 555 0259", 2, 0, 18, 0, "QR34", "B-01", None, "confirmed"),
    # Tomorrow
    ("Yuki", "Sato", "+1 415 555 0277", 2, 1, 20, 0, "ST56", "P-01", None, "confirmed"),
    ("Olivia", "Brandt", "+1 415 555 0288", 4, 1, 18, 0, "UV78", "T-04", None, "confirmed"),
    # Day after tomorrow
    ("Tomás", "Silva", "+1 415 555 0290", 4, 2, 18, 30, "WX90", "T-04", None, "confirmed"),
    ("Naomi", "Adeyemi", "+1 415 555 0301", 4, 2, 19, 30, "YZ12", "T-04", "Window seat", "confirmed"),
    # Later this week
    ("Felix", "Wagner", "+1 415 555 0312", 4, 4, 20, 30, "AC34", "T-04", None, "confirmed"),
    ("Chiamaka", "Eze", "+1 415 555 0333", 2, 5, 19, 0, "BD45", "P-02", None, "confirmed"),
    # Cancelled (was for tomorrow, called this morning to cancel)
    ("Chen", "Wei", "+1 415 555 0344", 4, 1, 20, 0, "CW10", "T-04", None, "cancelled"),
    # Last night (already happened - for "I dined last night, can I leave feedback")
    ("Antonio", "Russo", "+1 415 555 0355", 2, -1, 19, 30, "AR22", "T-02", "Anniversary", "confirmed"),
]
# fmt: on


def populate(db: HotelDB, today: date) -> None:
    """Insert seed rows into `db`. Reservation dates are stored as offsets from `today`."""
    conn = db.connection
    conn.executemany(
        "INSERT INTO hotel_rooms (id, type, nightly_rate, max_occupancy, smoking, pets_allowed, room_view) VALUES (?,?,?,?,?,?,?)",
        ROOMS,
    )
    conn.executemany(
        "INSERT INTO restaurant_tables (label, capacity, location, description) VALUES (?,?,?,?)",
        TABLES,
    )

    for (
        first,
        last,
        phone,
        party,
        offset,
        hour,
        minute,
        suffix,
        label,
        notes,
        status,
    ) in RESERVATIONS:
        table_row = conn.execute(
            "SELECT id FROM restaurant_tables WHERE label = ?", (label,)
        ).fetchone()
        assert table_row is not None, f"seed fixture references unknown table {label}"
        table_id = table_row[0]
        conn.execute(
            "INSERT INTO restaurant_reservations (code, table_id, first_name, last_name, phone, party_size, date, time, notes, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                f"RES-{suffix}",
                table_id,
                first,
                last,
                phone,
                party,
                (today + timedelta(days=offset)).isoformat(),
                time(hour, minute).isoformat(),
                notes,
                status,
            ),
        )


def build_seed_bytes(today: date) -> bytes:
    db = HotelDB.empty()
    try:
        populate(db, today)
        return db.serialize()
    finally:
        db.close()
