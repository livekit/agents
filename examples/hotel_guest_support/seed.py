"""Seed data for the example hotel DB."""

from __future__ import annotations

from datetime import date, timedelta

from hotel_db import HotelDB, compute_invoice

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

# (first, last, email, phone, code_suffix, room, offset_days, nights, guests, extras, card4, status)
#   offset < 0 and offset+nights > 0 -> in-house now;  offset > 0 -> upcoming;  else departed.
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
    # Tom Whelan's ocean queen (206) collides with Grace Lin's stay, and the
    # other ocean queen (304) is blocked by Noah Petrov. The resolver only
    # considers rooms at or above the rate already paid, so the city/garden
    # queens stay out of reach despite fitting four and being free, and the
    # cheapest room left to it is the suite: the free-upgrade scenario.
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
# fmt: on


def populate(db: HotelDB, today: date) -> None:
    """Insert seed rows into `db`. Booking check-in/check-out dates are stored as
    offsets from `today`."""
    conn = db.connection
    conn.executemany(
        "INSERT INTO hotel_rooms (id, type, nightly_rate, max_occupancy, smoking, pets_allowed, room_view) VALUES (?,?,?,?,?,?,?)",
        ROOMS,
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
        _, _, total, _ = compute_invoice(nightly_rate=nightly, nights=nights, extras=extras)
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


def build_seed_bytes(today: date) -> bytes:
    db = HotelDB.empty()
    try:
        populate(db, today)
        return db.serialize()
    finally:
        db.close()
