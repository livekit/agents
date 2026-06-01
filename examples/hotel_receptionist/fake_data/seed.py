"""Standalone dev script - generates a fresh, fully-seeded hotel.db.

    python fake_data/seed.py [path/to/hotel.db]

Not imported by the agent. Run it once before `python agent.py console` to get a
database with rooms, tables, and a realistic spread of bookings/reservations/
disputes anchored to the example's fixed TODAY.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hotel_db import (  # noqa: E402
    PRICING,
    TODAY,
    HotelDB,
    apply_tax,
    extras_total,
    invoice_line_items,
)

logger = logging.getLogger("hotel-receptionist.seed")

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "hotel.db"
_LATE = PRICING.late_checkout

# fmt: off
# (room_number, type, nightly_rate_cents, max_occupancy, smoking, pets, view)
ROOMS = [
    ("201", "king", 24000, 2, 0, 0, "city"),
    ("202", "king", 26000, 2, 0, 1, "ocean"),
    ("203", "king", 24000, 2, 1, 0, "city"),
    ("204", "queen_2beds", 22000, 4, 0, 0, "city"),
    ("205", "queen_2beds", 22000, 4, 0, 1, "garden"),
    ("206", "double_queen", 26000, 4, 0, 0, "ocean"),
    ("301", "king", 28000, 2, 0, 0, "ocean"),
    ("302", "king", 28000, 2, 0, 0, "ocean"),
    ("303", "queen_2beds", 24000, 4, 0, 0, "city"),
    ("304", "double_queen", 28000, 4, 0, 1, "ocean"),
    ("401", "suite", 48000, 4, 0, 1, "ocean"),
    ("402", "suite", 52000, 4, 0, 0, "ocean"),
    ("PH", "penthouse", 120000, 6, 0, 1, "ocean"),
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

# (first, last, email, phone, code_suffix, room, offset_days, nights, guests, extras, card4)
#   offset < 0 and offset+nights > 0 -> in-house now;  offset > 0 -> upcoming;  else departed.
# Smith / García / Lee codes are referenced by the playground, so keep them stable.
BOOKINGS = [
    ("Eleanor", "Smith", "eleanor.smith@gmail.com", "+1 415 555 0142", "AB12", "203", 5, 2, 2, ["breakfast"], "4242"),
    ("Marcus", "Johnson", "m.johnson@outlook.com", "+1 628 555 0199", "CD34", "205", 9, 3, 4, ["breakfast", "valet"], "1881"),
    ("Sofía", "García", "sofia.garcia@proton.me", "+1 415 555 0107", "EF56", "401", -1, 4, 3, ["breakfast", "valet", "pets"], "0007"),
    ("Priya", "Nair", "priya.nair@gmail.com", "+1 510 555 0188", "KM21", "202", -2, 4, 2, ["breakfast"], "3310"),
    ("Kenji", "Tanaka", "kenji.tanaka@gmail.com", "+1 415 555 0164", "RT88", "301", 0, 3, 2, ["valet"], "7782"),
    ("Amara", "Okafor", "amara.okafor@gmail.com", "+1 650 555 0121", "WX53", "206", -1, 2, 4, ["breakfast", "pets"], "5550"),
    ("Lucas", "Meyer", "lucas.meyer@gmx.de", "+49 30 5550173", "ZP19", "402", -3, 5, 2, ["breakfast", "valet"], "9041"),
    ("Vivienne", "Laurent", "v.laurent@me.com", "+1 415 555 0193", "PH01", "PH", -2, 6, 2, ["breakfast", "valet", "pets"], "1206"),
    ("Daniel", "Lee", "daniel.lee@gmail.com", "+1 415 555 0104", "GH78", "302", -6, 2, 2, ["late_checkout"], "9999"),
]

# (first, last, phone, party, offset_days, hour, minute, code_suffix, table, notes)
RESERVATIONS = [
    ("Marcus", "Bennett", "+1 415 555 0231", 4, 0, 19, 0, "JK90", "T-03", "Birthday"),
    ("Hannah", "Kowalski", "+1 415 555 0244", 2, 0, 20, 30, "LM12", "T-01", "Anniversary"),
    ("Sofía", "García", "+1 415 555 0107", 6, 0, 19, 30, "NP21", "T-05", "Family dinner"),
    ("Diego", "Herrera", "+1 415 555 0259", 2, 0, 18, 0, "QR34", "B-01", None),
    ("Yuki", "Sato", "+1 415 555 0277", 2, 0, 20, 0, "ST56", "P-01", None),
    ("Olivia", "Brandt", "+1 415 555 0288", 4, 0, 18, 0, "UV78", "T-04", None),
    ("Tomás", "Silva", "+1 415 555 0290", 4, 0, 18, 30, "WX90", "T-04", None),
    ("Naomi", "Adeyemi", "+1 415 555 0301", 4, 0, 19, 30, "YZ12", "T-04", "Window seat"),
    ("Felix", "Wagner", "+1 415 555 0312", 4, 0, 20, 30, "AC34", "T-04", None),
]

# (case, booking_code, line_item, amount, category, note, outcome, refund, status)
DISPUTES = [
    ("DSP-4K7M", "HTL-GH78", "Late checkout", _LATE, "late_checkout_fee", "Front desk said a 1 PM checkout would be fine.", "goodwill_waived", _LATE, "resolved"),
    ("DSP-9X2C", "HTL-EF56", "Minibar", 1800, "minibar", "Says they never opened the minibar.", "auto_refunded", 1800, "resolved"),
    ("DSP-2H6T", "HTL-ZP19", "Room service", 8800, "room_service_restaurant", "Charged for a dinner they didn't order.", "escalated_to_manager", 0, "open"),
]
# fmt: on


async def generate(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = HotelDB(db_path)
    await db.initialize()
    conn = db.connection

    conn.executemany("INSERT INTO hotel_rooms (room_number, type, nightly_rate, max_occupancy, smoking, pets_allowed, room_view) VALUES (?,?,?,?,?,?,?)", ROOMS)  # fmt: skip
    conn.executemany("INSERT INTO restaurant_tables (label, capacity, location, description) VALUES (?,?,?,?)", TABLES)  # fmt: skip

    for first, last, email, phone, suffix, room_no, offset, nights, guests, extras, card4 in BOOKINGS:  # fmt: skip
        room_id, nightly = conn.execute("SELECT id, nightly_rate FROM hotel_rooms WHERE room_number = ?", (room_no,)).fetchone()  # fmt: skip
        check_in = TODAY + timedelta(days=offset)
        room_subtotal = nightly * nights
        pre_tax = room_subtotal + extras_total(extras, nights)
        tax = apply_tax(pre_tax)
        code = f"HTL-{suffix}"
        conn.execute(
            "INSERT INTO hotel_bookings (code, room_id, first_name, last_name, email, phone, check_in, check_out, guests, extras, total, card_last4) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (code, room_id, first, last, email, phone, check_in.isoformat(), (check_in + timedelta(days=nights)).isoformat(), guests, ",".join(sorted(extras)), pre_tax + tax, card4),
        )  # fmt: skip
        items = invoice_line_items(
            nights=nights, room_subtotal=room_subtotal, extras=extras, tax=tax
        )
        conn.execute(
            "INSERT INTO hotel_invoices (booking_code, line_items, subtotal, taxes, total, paid) VALUES (?,?,?,?,?,?)",
            (code, json.dumps([li.__dict__ for li in items]), pre_tax, tax, pre_tax + tax, 1 if offset <= 0 else 0),
        )  # fmt: skip

    for first, last, phone, party, offset, hour, minute, suffix, label, notes in RESERVATIONS:
        table_id = conn.execute("SELECT id FROM restaurant_tables WHERE label = ?", (label,)).fetchone()[0]  # fmt: skip
        conn.execute(
            "INSERT INTO restaurant_reservations (code, table_id, first_name, last_name, phone, party_size, date, time, notes) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"RES-{suffix}", table_id, first, last, phone, party, (TODAY + timedelta(days=offset)).isoformat(), time(hour, minute).isoformat(), notes),
        )  # fmt: skip

    for case, code, line_item, amount, category, note, outcome, refund, status in DISPUTES:
        conn.execute(
            "INSERT INTO hotel_disputes (case_number, booking_code, line_item, amount, category, caller_note, outcome, refund_amount, status) VALUES (?,?,?,?,?,?,?,?,?)",
            (case, code, line_item, amount, category, note, outcome, refund, status),
        )  # fmt: skip
        if refund > 0:  # mirrors file_dispute(): refund decrements the invoice total
            conn.execute("UPDATE hotel_invoices SET total = total - ? WHERE booking_code = ?", (refund, code))  # fmt: skip

    await db.aclose()
    print(
        f"seeded {db_path}: {len(ROOMS)} rooms, {len(TABLES)} tables, "
        f"{len(BOOKINGS)} bookings, {len(RESERVATIONS)} reservations, {len(DISPUTES)} disputes"
    )
    print("\nTry in the console:")
    for _f, last, _e, _p, suffix, *_ in BOOKINGS[:2]:
        print(f"  Cancel: last name '{last}', code 'HTL-{suffix}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB_PATH
    asyncio.run(generate(path))
