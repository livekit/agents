"""Seed data and builders for the hotel example DB.

python -m examples.hotel_receptionist.fake_data.seed [path/to/hotel.db]
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, time, timedelta
from functools import cache
from pathlib import Path

from ..common import resolve_today
from ..hotel import PRICING, compute_invoice
from ..hotel_db import HotelDB

logger = logging.getLogger("hotel-receptionist.seed")

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "hotel.db"
_LATE = PRICING.late_checkout

# fmt: off
# (id (room number, floor+number), type, nightly_rate_cents, max_occupancy, smoking, pets, view)
ROOMS = [
    ("RM_201", "king", 24000, 2, 0, 0, "city"),
    ("RM_202", "king", 26000, 2, 0, 1, "ocean"),
    ("RM_203", "king", 24000, 2, 1, 0, "city"),
    ("RM_204", "queen_2beds", 22000, 4, 0, 0, "city"),
    ("RM_205", "queen_2beds", 22000, 4, 0, 1, "garden"),
    ("RM_206", "double_queen", 26000, 4, 0, 0, "ocean"),
    ("RM_301", "king", 28000, 2, 0, 0, "ocean"),
    ("RM_302", "king", 28000, 2, 0, 0, "ocean"),
    ("RM_303", "queen_2beds", 24000, 4, 0, 0, "city"),
    ("RM_304", "double_queen", 28000, 4, 0, 1, "ocean"),
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

# (first, last, email, phone, code_suffix, room, offset_days, nights, guests, extras, card4, status)
#   offset < 0 and offset+nights > 0 -> in-house now;  offset > 0 -> upcoming;  else departed.
# Smith / García / Lee codes are referenced by the playground, so keep them stable.
BOOKINGS = [
    # In-house right now
    ("Sofía", "García", "sofia.garcia@proton.me", "+1 415 555 0107", "EF56", "401", -1, 4, 3, ["breakfast", "valet", "pets"], "0007", "confirmed"),
    ("Priya", "Nair", "priya.nair@gmail.com", "+1 510 555 0188", "KM21", "202", -2, 4, 2, ["breakfast"], "3310", "confirmed"),
    ("Amara", "Okafor", "amara.okafor@gmail.com", "+1 650 555 0121", "WX53", "206", -1, 2, 4, ["breakfast", "pets"], "5550", "confirmed"),
    ("Lucas", "Meyer", "lucas.meyer@gmx.de", "+49 30 5550173", "ZP19", "402", -3, 5, 2, ["breakfast", "valet"], "9041", "confirmed"),
    ("Vivienne", "Laurent", "v.laurent@me.com", "+1 415 555 0193", "PH01", "PH", -2, 6, 2, ["breakfast", "valet", "pets"], "1206", "confirmed"),
    # In-house and being fished for by an outside caller - presence must never be disclosed
    ("Jonathan", "Pierce", "j.pierce@gmail.com", "+1 415 555 0233", "JP65", "303", -1, 3, 1, [], "5151", "confirmed"),
    # In-house with an early flight - the wake-up call caller
    ("Frank", "Adler", "frank.adler@gmail.com", "+1 415 555 0277", "FA09", "304", -1, 3, 1, [], "6203", "confirmed"),
    # --- Full house tonight (oversold) -------------------------------------
    # Dana Holt holds room 301 through tomorrow morning - and so does Kenji
    # Tanaka (RT88, checked in today): the double-booking behind the
    # "Confirmed guest, no room available tonight" walk scenario. The four
    # one-nighters fill every otherwise-free room TONIGHT ONLY, so the
    # re-accommodation search honestly comes up empty and 301 frees tomorrow.
    ("Dana", "Holt", "dana.holt@gmail.com", "+1 415 555 0341", "DH27", "301", -2, 3, 2, [], "9034", "confirmed"),
    ("Paul", "Greer", "paul.greer@gmail.com", "+1 415 555 0356", "PG11", "203", 0, 1, 1, [], "2218", "confirmed"),
    ("Rita", "Moss", "rita.moss@me.com", "+1 415 555 0368", "QM17", "204", 0, 1, 2, [], "7745", "confirmed"),
    ("Lena", "Fischer", "lena.fischer@gmx.de", "+49 30 5550441", "LF73", "302", 0, 1, 1, [], "6071", "confirmed"),
    # 205 (the garden queen) stays free tonight ON PURPOSE: it's the one concrete
    # fix the desk can offer Robert Klein ("I booked a garden view!") - and it's
    # a lower rate than Kenji Tanaka's king, so the walk resolver correctly
    # never offers it to him and his walk scenario stays intact.
    # --- Double-booked next weekend, but the house can absorb it -----------
    # Tom Whelan's double queen (206) collides with Grace Lin's stay, and the
    # other double queen (304) is blocked by Noah Petrov - so the only room
    # that fits his family of four is the suite: the free-upgrade scenario.
    ("Tom", "Whelan", "tom.whelan@gmail.com", "+1 415 555 0457", "TW55", "206", 4, 3, 4, [], "5126", "confirmed"),
    ("Grace", "Lin", "grace.lin@gmail.com", "+1 415 555 0463", "GL09", "206", 3, 3, 3, [], "8854", "confirmed"),
    ("Noah", "Petrov", "noah.petrov@gmail.com", "+1 415 555 0478", "NP66", "304", 3, 4, 4, [], "1937", "confirmed"),
    ("Kenji", "Tanaka", "kenji.tanaka@gmail.com", "+1 415 555 0164", "RT88", "301", 0, 3, 2, ["valet"], "7782", "confirmed"),
    # Checked in today, king city room - the "unhappy with their room" caller
    # (insists he booked a garden view; the record says otherwise)
    ("Robert", "Klein", "robert.klein@gmail.com", "+1 415 555 0377", "RK20", "201", 0, 2, 1, [], "8412", "confirmed"),
    # Arriving tomorrow
    ("Hiroshi", "Sato", "h.sato@gmail.com", "+1 415 555 0211", "BN23", "204", 1, 2, 3, ["breakfast"], "8821", "confirmed"),
    # Upcoming
    ("Eleanor", "Smith", "eleanor.smith@gmail.com", "+1 415 555 0142", "AB12", "203", 5, 2, 2, ["breakfast"], "4242", "confirmed"),
    ("Marcus", "Johnson", "m.johnson@outlook.com", "+1 628 555 0199", "CD34", "205", 9, 3, 4, ["breakfast", "valet"], "1881", "confirmed"),
    # Smoking room (203 is the only smoking-permitted room)
    ("Mei", "Chen", "mei.chen@gmail.com", "+1 415 555 0222", "MN42", "203", 14, 2, 2, ["breakfast"], "4477", "confirmed"),
    # --- Completely sold out one night (offset 25 = Fri Jul 3, July-4th weekend) ---
    # Every one of the 13 rooms is taken for this single night, so a fresh
    # booking inquiry for that date honestly comes up empty: the "we're full,
    # politely deny the walk-in" scenario. One-nighters (nights=1) so the
    # block doesn't bleed into adjacent dates or other scenarios.
    ("Owen", "Carver", "owen.carver@gmail.com", "+1 415 555 0501", "SO01", "201", 25, 1, 2, [], "1101", "confirmed"),
    ("Bianca", "Ross", "bianca.ross@gmail.com", "+1 415 555 0502", "SO02", "202", 25, 1, 2, [], "1102", "confirmed"),
    ("Caleb", "Nguyen", "caleb.nguyen@gmail.com", "+1 415 555 0503", "SO03", "203", 25, 1, 2, [], "1103", "confirmed"),
    ("Delia", "Brooks", "delia.brooks@gmail.com", "+1 415 555 0504", "SO04", "204", 25, 1, 3, [], "1104", "confirmed"),
    ("Ezra", "Flynn", "ezra.flynn@gmail.com", "+1 415 555 0505", "SO05", "205", 25, 1, 3, [], "1105", "confirmed"),
    ("Farah", "Haddad", "farah.haddad@gmail.com", "+1 415 555 0506", "SO06", "206", 25, 1, 4, [], "1106", "confirmed"),
    ("Gideon", "Park", "gideon.park@gmail.com", "+1 415 555 0507", "SO07", "301", 25, 1, 2, [], "1107", "confirmed"),
    ("Helena", "Cruz", "helena.cruz@gmail.com", "+1 415 555 0508", "SO08", "302", 25, 1, 2, [], "1108", "confirmed"),
    ("Ivan", "Sokolov", "ivan.sokolov@gmail.com", "+1 415 555 0509", "SO09", "303", 25, 1, 3, [], "1109", "confirmed"),
    ("Jana", "Novak", "jana.novak@gmail.com", "+1 415 555 0510", "SO10", "304", 25, 1, 4, [], "1110", "confirmed"),
    ("Kofi", "Mensah", "kofi.mensah@gmail.com", "+1 415 555 0511", "SO11", "401", 25, 1, 4, [], "1111", "confirmed"),
    ("Lara", "Conti", "lara.conti@gmail.com", "+1 415 555 0512", "SO12", "402", 25, 1, 2, [], "1112", "confirmed"),
    ("Mateo", "Rivas", "mateo.rivas@gmail.com", "+1 415 555 0513", "SO13", "PH", 25, 1, 5, [], "1113", "confirmed"),
    # Departed (last week / weeks ago) - source of disputes + invoice lookups
    ("Daniel", "Lee", "daniel.lee@gmail.com", "+1 415 555 0104", "GH78", "302", -6, 2, 2, ["late_checkout"], "9999", "confirmed"),
    ("Olivia", "Brandt", "olivia.brandt@me.com", "+1 415 555 0288", "QT55", "204", -10, 3, 2, ["breakfast"], "6677", "confirmed"),
    ("Aino", "Virtanen", "aino.virtanen@gmail.com", "+358 9 5550144", "JX31", "303", -14, 4, 3, ["breakfast", "valet"], "5512", "confirmed"),
    # No-show (dates passed, guest never checked in; card-guaranteed and charged,
    # no cancellation on record - the "Angry no-show charge dispute" caller)
    ("Tanya", "Richardson", "tanya.richardson@gmail.com", "+1 248 555 0291", "NS44", "304", -4, 2, 1, [], "7321", "confirmed"),
    # Cancelled (was a future booking that got cancelled - good for "I cancelled, where's my refund")
    ("Felix", "Wagner", "felix.wagner@me.com", "+1 415 555 0312", "FW77", "402", 3, 2, 2, ["breakfast", "valet"], "2299", "cancelled"),
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

# (case, booking_code, line_item, amount, category, note, outcome, refund, status)
DISPUTES = [
    # Resolved - one per policy outcome to demo all the paths
    ("DSP-4K7M", "HTL-GH78", "Late checkout", _LATE, "late_checkout_fee", "Front desk said a 1 PM checkout would be fine.", "goodwill_waived", _LATE, "resolved"),
    ("DSP-9X2C", "HTL-EF56", "Minibar", 1800, "minibar", "Says they never opened the minibar.", "auto_refunded", 1800, "resolved"),
    ("DSP-5R8K", "HTL-QT55", "Room (3 nights)", 66000, "double_charge_billing_error", "Charged twice for the same stay - duplicate on the statement.", "auto_refunded", 66000, "resolved"),
    ("DSP-7M3X", "HTL-JX31", "Pet fee", 5000, "damage_cleaning", "No pet on the stay, but pet cleaning fee on the invoice.", "explained_no_action", 0, "resolved"),
    # Open / unresolved
    ("DSP-2H6T", "HTL-ZP19", "Room service", 8800, "room_service_restaurant", "Charged for a dinner they didn't order.", "escalated_to_manager", 0, "open"),
]
# (last_name, preferences) - read-only guest history for returning-guest personalization.
GUEST_HISTORY = [
    ("Lee", "Prefers a high, quiet floor away from the elevator, and feather-free "
            "(hypoallergenic) pillows. Had a noise complaint on a previous stay."),
]
# fmt: on


def populate(db: HotelDB, today: date) -> None:
    """Insert seed rows into `db`. Booking check-in/check-out and
    reservation dates are stored as offsets from `today`."""
    conn = db.connection
    conn.executemany(
        "INSERT INTO hotel_rooms (id, type, nightly_rate, max_occupancy, smoking, pets_allowed, room_view) VALUES (?,?,?,?,?,?,?)",
        ROOMS,
    )
    conn.executemany(
        "INSERT INTO restaurant_tables (label, capacity, location, description) VALUES (?,?,?,?)",
        TABLES,
    )
    conn.executemany(
        "INSERT INTO guest_history (last_name, preferences) VALUES (?,?)",
        GUEST_HISTORY,
    )

    for (
        first,
        last,
        email,
        phone,
        suffix,
        room_no,
        offset,
        nights,
        guests,
        extras,
        card4,
        status,
    ) in BOOKINGS:
        room_row = conn.execute(
            "SELECT id, nightly_rate FROM hotel_rooms WHERE id = ?", (f"RM_{room_no}",)
        ).fetchone()
        assert room_row is not None, f"seed fixture references unknown room {room_no}"
        room_id, nightly = room_row
        check_in = today + timedelta(days=offset)
        subtotal, taxes, total, items = compute_invoice(
            nightly_rate=nightly, nights=nights, extras=extras
        )
        code = f"HTL-{suffix}"
        conn.execute(
            "INSERT INTO hotel_bookings (code, room_id, first_name, last_name, email, phone, check_in, check_out, guests, extras, total, card_last4, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                code,
                room_id,
                first,
                last,
                email,
                phone,
                check_in.isoformat(),
                (check_in + timedelta(days=nights)).isoformat(),
                guests,
                ",".join(sorted(extras)),
                total,
                card4,
                status,
            ),
        )
        conn.execute(
            "INSERT INTO hotel_invoices (booking_code, line_items, subtotal, taxes, total, paid) VALUES (?,?,?,?,?,?)",
            (
                code,
                json.dumps([li.__dict__ for li in items]),
                subtotal,
                taxes,
                total,
                1 if offset <= 0 and status == "confirmed" else 0,
            ),
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

    for case, code, line_item, amount, category, note, outcome, refund, status in DISPUTES:
        conn.execute(
            "INSERT INTO hotel_disputes (case_number, booking_code, line_item, amount, category, caller_note, outcome, refund_amount, status) VALUES (?,?,?,?,?,?,?,?,?)",
            (case, code, line_item, amount, category, note, outcome, refund, status),
        )
        if refund > 0:  # mirrors file_dispute(): refund decrements the invoice total
            conn.execute(
                "UPDATE hotel_invoices SET total = total - ? WHERE booking_code = ?", (refund, code)
            )


@cache
def build_seed_bytes(today: date) -> bytes:
    db = HotelDB.empty(today)
    try:
        populate(db, today)
        return db.serialize()
    finally:
        db.close()


def write_seed_file(db_path: Path, today: date) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(build_seed_bytes(today))
    print(
        f"seeded {db_path}: {len(ROOMS)} rooms, {len(TABLES)} tables, "
        f"{len(BOOKINGS)} bookings, {len(RESERVATIONS)} reservations, {len(DISPUTES)} disputes "
        f"(today={today.isoformat()})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB_PATH
    write_seed_file(path, resolve_today())
