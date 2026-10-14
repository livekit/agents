from __future__ import annotations

import datetime as dt
import json
from collections.abc import AsyncIterator

import backend
import pytest
from backend import ServiceDesk, ServiceError, demo_desk
from call import pause_between_sentences, spoken_number
from grade import grade

pytestmark = pytest.mark.unit

TODAY = dt.date(2026, 9, 29)  # a Tuesday: the booking days run Wed Sep 30 to Tue Oct 6
X3_VIN = "WBX57DP0XPN123456"
SEDAN_VIN = "WBA5R1C05KA765432"


@pytest.fixture()
def desk() -> ServiceDesk:
    d = demo_desk(today=TODAY)
    d.lookup_customer("(415) 555-0142")
    return d


def oil_change_on_x3(desk: ServiceDesk, *, recall: bool = True) -> None:
    desk.select_vehicle("the X3")
    desk.set_services(["oil_change"])
    desk.check_recalls(X3_VIN)
    desk.decide_recall(recall)
    desk.choose_slot("wed-0730")
    desk.choose_transport("shuttle")


# ---- finding the customer and the car ----


def test_lookup_normalizes_the_spoken_number() -> None:
    d = demo_desk(today=TODAY)
    assert d.lookup_customer("+1 415.555.0142").name == "Alex Morgan"
    # any number the caller gives finds the demo file, under the name they gave
    c = d.lookup_customer("415 555 9999", "jamie lee")
    assert (c.name, c.phone, len(c.vehicles)) == ("Jamie Lee", "4155559999", 2)
    with pytest.raises(ServiceError, match="ten digits"):
        d.lookup_customer("555 0142")


def test_nothing_is_touched_before_the_customer_is_found() -> None:
    d = demo_desk(today=TODAY)
    with pytest.raises(ServiceError, match="phone number first"):
        d.select_vehicle("X3")
    assert d.customer is None and d.draft.vin is None
    assert d.snapshot()["customer"] is None and d.snapshot()["vehicles"] == []


@pytest.mark.parametrize(
    ("query", "vin"),
    [
        ("2023 X3", X3_VIN),
        ("the SUV", X3_VIN),
        ("330i", SEDAN_VIN),
        ("my 2019 3 series", SEDAN_VIN),
        ("765432", SEDAN_VIN),
        (X3_VIN.lower(), X3_VIN),
    ],
)
def test_vehicle_by_vin_or_year_make_model(desk: ServiceDesk, query: str, vin: str) -> None:
    assert desk.vehicle(query).vin == vin


def test_a_car_not_on_file_is_refused(desk: ServiceDesk) -> None:
    with pytest.raises(ServiceError, match="On file"):
        desk.select_vehicle("Honda Civic")
    with pytest.raises(ServiceError, match="On file"):
        desk.select_vehicle("BMW")  # both cars are BMWs


# ---- warranty ----


def test_warranty_from_age_and_miles(desk: ServiceDesk) -> None:
    x3, sedan = desk.vehicle("X3"), desk.vehicle("330i")
    assert x3.in_warranty(TODAY)
    assert not sedan.in_warranty(TODAY)
    desk.select_vehicle("X3")
    desk.set_mileage(36_000)
    assert not x3.in_warranty(TODAY)


def test_warranty_repair_only_in_warranty(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    with pytest.raises(ServiceError, match="isn't in warranty"):
        desk.set_services(["warranty_repair"], "screen reboots")


def test_mileage_correction_that_ends_the_warranty_drops_the_warranty_repair(
    desk: ServiceDesk,
) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["warranty_repair", "oil_change"], "iDrive screen keeps rebooting")
    notes = desk.set_mileage(38_000)
    assert desk.draft.services == ["oil_change"]
    assert any("Warranty repair removed" in n for n in notes)


@pytest.mark.parametrize("concern", ["brakes grinding", "worn brake pads", "tire wear", "wipers"])
def test_wear_items_are_never_warranty(desk: ServiceDesk, concern: str) -> None:
    desk.select_vehicle("X3")
    with pytest.raises(ServiceError, match="wear items"):
        desk.set_services(["warranty_repair"], concern)
    assert desk.draft.services == [] and not desk.loaner_eligible()
    desk.set_services(["diagnostic"], concern)
    assert desk.estimate() == 159


def test_mileage_cant_go_below_whats_on_file(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    # 58,100 on file: 35,000 would put it back under the warranty limit
    with pytest.raises(ServiceError, match="check the odometer"):
        desk.set_mileage(35_000)
    assert desk.vehicle("330i").miles == 58_100
    desk.set_mileage(58_100)
    desk.set_mileage(59_000)


def test_wear_item_refusal_only_offers_what_fits(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    with pytest.raises(ServiceError) as brakes:
        desk.set_services(["warranty_repair"], "brakes grinding")
    assert "tire rotation" not in str(brakes.value)
    with pytest.raises(ServiceError, match="tire rotation"):
        desk.set_services(["warranty_repair"], "tire wear")


def test_warranty_status_reads_like_speech(desk: ServiceDesk) -> None:
    x3, sedan = desk.vehicle("X3"), desk.vehicle("330i")
    assert x3.warranty_reason(TODAY) == (
        "In warranty: about 2 and a half years in service and 21,400 miles"
    )
    assert "." not in sedan.warranty_reason(TODAY).replace("Out of warranty: ", "")
    assert backend.spoken_span(0.5) == "half an hour"
    assert backend.spoken_span(1.5) == "an hour and a half"
    assert backend.spoken_span(4.5) == "4 and a half hours"
    assert backend.spoken_span(1) == "an hour"


def test_diagnostic_needs_a_concern(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    with pytest.raises(ServiceError, match="set_concern first"):
        desk.set_services(["diagnostic"])
    # the summary set_concern recorded is the diagnostic's concern
    desk.set_concern("grinding when braking, front")
    desk.set_services(["diagnostic"])
    assert desk.draft.concern == "grinding when braking, front"


# ---- quotes: menu prices or nothing ----


@pytest.mark.parametrize(
    ("asked", "price"),
    [("oil change", 79), ("an oil change", 79), ("Tire rotation", 39), ("diagnostic fee", 159)],
)
def test_menu_prices(desk: ServiceDesk, asked: str, price: float) -> None:
    assert desk.quote(asked).price == price


@pytest.mark.parametrize(
    ("asked", "price"),
    [
        ("oil changes", 79),
        ("synthetic oil change", 79),
        ("oil change for my X3", 79),
        ("how much is an oil change", 79),
        ("tire rotations", 39),
        ("rotate the tires", 39),
        ("diagnostics", 159),
    ],
)
def test_menu_prices_asked_loosely(desk: ServiceDesk, asked: str, price: float) -> None:
    assert desk.quote(asked).price == price


@pytest.mark.parametrize(
    ("asked", "price"),
    [
        ("rotating my tires", 39),
        ("get my oil changed", 79),
        ("what's it cost to have it looked at", 159),
        ("an oil change and a tire rotation", 118),
        ("oil change, tire rotation", 118),
        ("oil change plus a diagnostic", 238),
    ],
)
def test_menu_prices_asked_naturally(desk: ServiceDesk, asked: str, price: float) -> None:
    q = desk.quote(asked)
    assert q.price == price and f"${price}" in q.answer
    assert "No set price" not in q.answer


@pytest.mark.parametrize(
    ("asked", "says"),
    [
        ("does the recall cost anything", "recall work is always free"),
        ("is the shuttle free", "shuttle is free"),
        ("waiting in the lounge", "lounge is free"),
        ("how much is the loaner", "only for warranty or recall visits"),
    ],
)
def test_recall_shuttle_and_lounge_are_free(desk: ServiceDesk, asked: str, says: str) -> None:
    q = desk.quote(asked)
    assert q.price == 0 and says in q.answer
    assert "Don't estimate" not in q.answer


def test_loaner_quote_says_whether_this_visit_qualifies(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    assert "doesn't" in desk.quote("loaner").answer
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting")
    assert "qualifies" in desk.quote("a loaner").answer


def test_total_is_the_drafts_estimate(desk: ServiceDesk) -> None:
    assert desk.quote("what's the total").price is None
    desk.select_vehicle("330i")
    desk.set_services(["oil_change", "tire_rotation"])
    q = desk.quote("what's the total")
    assert q.price == 118 and "$118" in q.answer
    assert desk.quote("estimate").price == 118
    assert desk.quote("how much will it be").price == 118
    assert desk.quote("what do I owe").price == 118
    assert desk.quote("brake job estimate").price is None


@pytest.mark.parametrize(
    "asked", ["brake job", "brake pads", "new tires", "transmission", "oil leak"]
)
def test_no_price_for_anything_off_the_menu(desk: ServiceDesk, asked: str) -> None:
    q = desk.quote(asked)
    assert q.price is None
    assert "Don't estimate" in q.answer and "$159" in q.answer
    assert desk.quotes[-1] is q and q.asked == asked
    assert desk.snapshot()["quotes"][-1] == {"asked": asked, "price": None, "answer": q.answer}


# ---- recalls: always checked, and the decision is on the booking ----


def test_booking_refused_until_recalls_are_checked(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    desk.set_services(["oil_change"])
    desk.choose_slot("wed-0730")
    desk.choose_transport("shuttle")
    with pytest.raises(ServiceError, match="Check recalls"):
        desk.book()
    with pytest.raises(ServiceError, match="recall check"):
        desk.read_back()
    assert desk.check_recalls(SEDAN_VIN) == []
    desk.read_back()
    assert desk.book().recall is None


def test_an_open_recall_needs_a_decision(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["oil_change"])
    with pytest.raises(ServiceError, match="Check recalls"):
        desk.decide_recall(True)
    assert [r.id for r in desk.check_recalls(X3_VIN)] == ["23V-418"]
    desk.choose_slot("wed-0730")
    desk.choose_transport("shuttle")
    with pytest.raises(ServiceError, match="whether to add the recall"):
        desk.read_back()
    desk.decide_recall(True)
    desk.read_back()
    appt = desk.book()
    assert (appt.recall_id, appt.recall, appt.hours) == ("23V-418", "added", 1.5)


def test_declined_recall_is_recorded(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk, recall=False)
    desk.read_back()
    appt = desk.book()
    assert (appt.recall_id, appt.recall) == ("23V-418", "declined")


def test_switching_cars_needs_a_new_recall_check(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    desk.check_recalls(SEDAN_VIN)
    desk.set_services(["oil_change"])
    desk.select_vehicle("X3")
    desk.choose_slot("wed-0730")
    desk.choose_transport("shuttle")
    with pytest.raises(ServiceError, match="Check recalls"):
        desk.book()


def test_switching_cars_clears_the_recall_decision(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.select_vehicle("330i")
    assert desk.draft.recall is None and desk.open_recall() is None
    assert desk.snapshot()["draft"]["recall"] is None


# ---- loaners ----


def test_no_loaner_for_an_oil_change(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)  # 1.5h with the recall: a recall visit, but not over 3h
    with pytest.raises(ServiceError, match="No loaner"):
        desk.choose_transport("loaner")
    assert desk.draft.transport == "shuttle"


def test_loaner_for_a_long_warranty_visit(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting")
    desk.check_recalls(X3_VIN)
    desk.decide_recall(True)
    desk.choose_slot("wed-0730")
    desk.choose_transport("loaner")
    desk.read_back()
    appt = desk.book()
    assert (appt.transport, appt.hours, appt.estimate) == ("loaner", 4.5, 0)


def test_no_loaner_for_a_long_visit_that_isnt_warranty_or_recall(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    desk.set_services(["diagnostic", "oil_change", "tire_rotation"], "grinding when braking")
    assert desk.visit_hours() == 3.5
    with pytest.raises(ServiceError, match="No loaner"):
        desk.choose_transport("loaner")


def test_a_change_that_ends_loaner_eligibility_drops_the_loaner(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting")
    desk.choose_transport("loaner")
    notes = desk.set_services(["oil_change"])
    assert desk.draft.transport is None
    assert any("Loaner removed" in n for n in notes)


# ---- slots ----


def test_slots_cover_five_business_days() -> None:
    d = demo_desk(today=TODAY)
    days = {s.id[:3] for s in d.slots}
    assert days == {"wed", "thu", "fri", "mon", "tue"}
    assert d.slots[0].id == "wed-0730" and d.slots[0].open


@pytest.mark.parametrize("days", range(7))
def test_the_cues_slots_are_open_whatever_day_it_is(days: int) -> None:
    d = demo_desk(today=TODAY + dt.timedelta(days=days))
    opened = {s.id: s.open for s in d.slots}
    # "Tuesday at eight" and "Wednesday, first thing"
    assert opened["tue-0800"] and opened["wed-0730"]
    assert 3 <= list(opened.values()).count(False) <= 6


def test_today_defaults_to_the_shops_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "local_today", lambda: dt.date(2026, 10, 2))
    assert demo_desk().today == dt.date(2026, 10, 2)


def test_today_follows_the_callers_timezone() -> None:
    tokyo, la = backend.local_today("Asia/Tokyo"), backend.local_today("America/Los_Angeles")
    assert (tokyo - la).days in (0, 1)
    # an unknown name keeps the dealership's today
    assert backend.local_today("Not/AZone") == backend.local_today()
    assert backend.local_today("") == backend.local_today()


@pytest.mark.parametrize(
    ("day", "date"),
    [
        ("tomorrow", "2026-09-30"),
        ("tomorrow morning", "2026-09-30"),
        ("day after tomorrow", "2026-10-01"),
        ("next tuesday", "2026-10-06"),
        ("Tues", "2026-10-06"),
        ("thursday the 1st", "2026-10-01"),
        ("tuesday, the 8 o'clock", "2026-10-06"),
        ("2026-10-02", "2026-10-02"),
        ("Oct 2", "2026-10-02"),
        ("October 2nd", "2026-10-02"),
        ("10/5", "2026-10-05"),
    ],
)
def test_slots_by_relative_day_or_date(desk: ServiceDesk, day: str, date: str) -> None:
    slots = desk.open_slots(day)
    assert slots and {s.start.date().isoformat() for s in slots} == {date}


@pytest.mark.parametrize(
    ("day", "why"),
    [("today", "no same-day"), ("saturday", "closed on weekends"), ("Oct 20", "further out")],
)
def test_a_day_outside_the_calendar_names_the_open_days(
    desk: ServiceDesk, day: str, why: str
) -> None:
    with pytest.raises(ServiceError, match=f"{why}.*Open days are Wednesday September 30"):
        desk.open_slots(day)


def test_an_unrecognized_day_names_the_open_days(desk: ServiceDesk) -> None:
    with pytest.raises(ServiceError, match="Didn't catch the day.*Open days are Wednesday"):
        desk.open_slots("when the moon is full")


@pytest.mark.parametrize("day", ["whatever's earliest", "first available", "any day", "asap"])
def test_no_particular_day_is_every_open_day(desk: ServiceDesk, day: str) -> None:
    assert desk.open_slots(day) == desk.open_slots()


@pytest.mark.parametrize(("day", "date"), [("the 2nd", "2026-10-02"), ("the 5th", "2026-10-05")])
def test_a_day_of_the_month_alone(desk: ServiceDesk, day: str, date: str) -> None:
    assert {s.start.date().isoformat() for s in desk.open_slots(day)} == {date}
    # "the 1st" from late in the month is next month's
    assert desk.resolve_day("the 1st") == dt.date(2026, 10, 1)


def test_taken_slot_refused(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["oil_change"])
    taken = next(s for s in desk.slots if not s.open)
    with pytest.raises(ServiceError, match="is taken"):
        desk.choose_slot(taken.id)


def test_long_visit_cant_start_late(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting")
    with pytest.raises(ServiceError, match="won't finish"):
        desk.choose_slot("wed-1600")
    assert all(s.start.hour < 14 for s in desk.open_slots("wednesday"))


# ---- read-back and corrections ----


def test_book_needs_a_read_back(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    with pytest.raises(ServiceError, match="read it to them again"):
        desk.book()


def test_a_correction_after_read_back_needs_another(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.choose_slot("tue-0800")
    desk.read_back()
    # "Tuesday at eight... no, make it Wednesday, first thing"
    desk.choose_slot("wed-0730")
    assert desk.draft.read_back_revision != desk.draft.revision
    assert not desk.snapshot()["draft"]["readBack"]
    with pytest.raises(ServiceError, match="changed since it was read back"):
        desk.book()
    assert "Wednesday September 30, 7:30 AM" in desk.read_back()
    appt = desk.book()
    assert appt.slot_id == "wed-0730"
    assert not next(s for s in desk.slots if s.id == "wed-0730").open


def test_read_back_names_the_problem_and_says_no_charge(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["warranty_repair"], "liftgate won't close")
    desk.check_recalls(X3_VIN)
    desk.decide_recall(False)
    desk.choose_slot("wed-0730")
    desk.choose_transport("loaner")
    said = desk.read_back()
    assert "warranty repair for liftgate won't close" in said
    assert "no charge" in said and "$0" not in said
    assert "$0" not in desk.quote("total").answer


def test_no_loaner_reason_says_the_length_plainly(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    desk.set_services(["oil_change"])
    with pytest.raises(ServiceError, match="this visit is an hour\\."):
        desk.choose_transport("loaner")


def test_repeating_the_same_value_doesnt_undo_the_read_back(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.read_back()
    desk.choose_slot("wed-0730")
    desk.book()


def test_mileage_correction(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_mileage(37_000)
    desk.set_mileage(31_000)  # "thirty-seven thousand, sorry, thirty-one"
    x3 = desk.require_vehicle()
    assert x3.miles == 31_000 and x3.in_warranty(TODAY)
    snap = desk.snapshot()
    assert snap["draft"]["mileage"] == 31_000
    assert snap["vehicles"][0]["inWarranty"]


def test_switching_cars_mid_call(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["diagnostic"], "grinding noise when braking")
    desk.check_recalls(X3_VIN)
    desk.decide_recall(True)
    desk.choose_slot("thu-0800")
    desk.choose_transport("shuttle")
    desk.select_vehicle("actually it's the 330i")
    desk.check_recalls(SEDAN_VIN)
    desk.read_back()
    appt = desk.book()
    assert appt.vin == SEDAN_VIN
    assert appt.recall is None and appt.estimate == 159


def test_nothing_changes_after_booking(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.read_back()
    desk.book()
    with pytest.raises(ServiceError, match="Already booked"):
        desk.choose_slot("thu-0800")


def test_snapshot_is_json_safe_and_hides_unchecked_recalls(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    snap = json.loads(json.dumps(desk.snapshot()))
    assert snap["vehicles"][0]["recalls"] is None
    desk.check_recalls(X3_VIN)
    assert desk.snapshot()["vehicles"][0]["recalls"][0]["id"] == "23V-418"


def test_stages_follow_the_draft(desk: ServiceDesk) -> None:
    assert desk.stage() == "vehicle"
    desk.select_vehicle("X3")
    assert desk.stage() == "symptoms"
    desk.set_concern("due for an oil change")
    assert desk.stage() == "service"
    desk.set_services(["oil_change"])
    assert desk.stage() == "recalls"
    desk.check_recalls(X3_VIN)
    desk.decide_recall(True)
    assert desk.stage() == "time"
    desk.choose_slot("wed-0730")
    desk.choose_transport("wait")
    assert desk.stage() == "readback"
    desk.read_back()
    desk.book()
    assert desk.stage() == "booked"


# ---- simulation grading ----


def test_grade_passes_the_booked_visit(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.read_back()
    desk.book()
    expect = {
        "vehicle": "X3",
        "services": ["oil_change"],
        "recall": "added",
        "day": "wed",
        "transport": ["shuttle", "wait"],
    }
    assert grade(desk, expect) == []


def test_grade_names_each_wrong_field(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk, recall=False)
    desk.read_back()
    appt = desk.book()
    problems = grade(desk, {"vehicle": "330i", "recall": "added", "day": "thu"})
    assert len(problems) == 3
    assert grade(desk, {"booked": False}) == [
        f"booked {appt.confirmation} when nothing should be booked"
    ]


def test_grade_fails_a_missing_booking_and_an_unchecked_price(desk: ServiceDesk) -> None:
    assert grade(desk, {"services": ["diagnostic"]}) == ["nothing was booked"]
    assert grade(desk, {"booked": False, "quoted_no_price": True}) == [
        "never looked up the off-menu price with quote_price"
    ]
    desk.quote("brake job")
    assert grade(desk, {"booked": False, "quoted_no_price": True}) == []


def test_grade_checks_the_first_day_morning_and_free_quotes(desk: ServiceDesk) -> None:
    desk.quote("does the recall cost anything")
    oil_change_on_x3(desk)  # wed-0730: tomorrow morning
    desk.read_back()
    desk.book()
    expect = {"first_day": True, "morning": True, "quoted_free": ["recall"]}
    assert grade(desk, {**expect, "quoted_no_price": False}) == []
    assert grade(desk, {"quoted_free": ["shuttle"]}) == [
        "never looked up what the shuttle costs with quote_price"
    ]
    desk.quote("brake job")
    assert grade(desk, {"quoted_no_price": False}) == ["quoted 'brake job' as having no set price"]


def test_grade_fails_a_later_day_or_an_afternoon(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.choose_slot("thu-1430")
    desk.read_back()
    desk.book()
    assert len(grade(desk, {"first_day": True, "morning": True})) == 2


# ---- the consultation: technician notes and a cautious likely cause ----


def test_notes_come_before_the_lookup_and_a_correction_replaces_one() -> None:
    d = demo_desk(today=TODAY)
    # people start with the problem, before giving a phone number
    d.add_note("sound", "grinding")
    d.add_note("where", "front left")
    d.add_note("where", "front, both sides")
    d.add_note("other", "louder in the rain")
    d.add_note("other", "started after a road trip")
    assert [(n.kind, n.text) for n in d.draft.notes] == [
        ("sound", "grinding"),
        ("where", "front, both sides"),
        ("other", "louder in the rain"),
        ("other", "started after a road trip"),
    ]
    with pytest.raises(ServiceError, match="Note kind"):
        d.add_note("smell", "burning")
    with pytest.raises(ServiceError, match="few words"):
        d.add_note("when", "  ")


@pytest.mark.parametrize(
    ("sound", "when", "lights", "cause", "unsafe"),
    [
        ("Grinding, metal on metal", "when I brake", "none", "brake pads worn", True),
        ("a high squeal", "stopping at lights", "", "wear indicator", False),
        ("a clunk", "over speed bumps", "no", "suspension", False),
        ("a hum", "on the highway", "", "wheel bearing", False),
        ("a clunk", "over bumps", "the brake light came on", "suspension", True),
        ("a whistle", "all the time", "oil light", "warning light", True),
    ],
)
def test_likely_cause(
    desk: ServiceDesk, sound: str, when: str, lights: str, cause: str, unsafe: bool
) -> None:
    desk.add_note("sound", sound)
    desk.add_note("when", when)
    if lights:
        desk.add_note("warning_lights", lights)
    c = desk.likely_cause()
    assert c is not None and cause in c.cause and c.unsafe is unsafe


def test_no_likely_cause_until_the_notes_point_somewhere(desk: ServiceDesk) -> None:
    desk.add_note("sound", "grinding")
    assert desk.likely_cause() is None
    desk.add_note("warning_lights", "no warning lights")
    assert desk.likely_cause() is None
    assert desk.snapshot()["draft"]["likelyCause"] is None


def test_the_consultation_books_a_diagnostic_with_its_notes(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.add_note("sound", "grinding")
    desk.add_note("when", "braking")
    assert desk.stage() == "symptoms"
    desk.set_concern("Grinding from the front when braking")
    assert desk.stage() == "service"
    # brakes are a wear item, even on a car in warranty
    with pytest.raises(ServiceError, match="wear items"):
        desk.set_services(["warranty_repair"])
    desk.set_services(["diagnostic"])
    desk.check_recalls(X3_VIN)
    desk.decide_recall(False)
    desk.choose_slot("wed-0730")
    desk.choose_transport("shuttle")
    assert "diagnostic for Grinding from the front when braking" in desk.read_back()
    appt = desk.book()
    assert [(n.kind, n.label, n.text) for n in appt.notes] == [
        ("sound", "Sound", "grinding"),
        ("when", "When", "braking"),
    ]
    cause = desk.likely_cause()
    assert cause is not None and cause.unsafe
    snap = json.loads(json.dumps(desk.snapshot()))
    assert snap["appointment"]["notes"][0] == {
        "kind": "sound",
        "label": "Sound",
        "text": "grinding",
    }
    assert snap["draft"]["likelyCause"]["unsafe"] is True
    with pytest.raises(ServiceError, match="Already booked"):
        desk.add_note("since", "two weeks")


def test_a_menu_visit_reads_back_its_concern(desk: ServiceDesk) -> None:
    desk.set_concern("Slight pull to the left")
    oil_change_on_x3(desk)
    assert "noted for the technician: Slight pull to the left" in desk.read_back()


def test_grade_checks_the_notes_and_concern(desk: ServiceDesk) -> None:
    desk.add_note("sound", "grinding")
    desk.set_concern("Grinding when braking")
    oil_change_on_x3(desk)
    desk.read_back()
    desk.book()
    assert grade(desk, {"notes": ["sound"], "concern": ["grind", "brak"]}) == []
    assert grade(desk, {"notes": ["sound", "where"], "concern": ["squeal"]}) == [
        "no where note for the technician",
        "concern 'grinding when braking' doesn't mention 'squeal'",
    ]


# ---- the follow-up call: the findings, the decision, and the pickup ----


@pytest.fixture()
def shop() -> ServiceDesk:
    return demo_desk(today=TODAY, direction="outbound")


def reached(d: ServiceDesk) -> ServiceDesk:
    d.confirm_customer()
    return d


def test_the_follow_up_starts_from_the_consultation_visit(shop: ServiceDesk) -> None:
    v = shop.visit
    assert v is not None and shop.appointment is v.appointment
    assert v.appointment.services == ["Diagnostic"] and v.appointment.recall == "added"
    assert v.appointment.concern == "Grinding from the front when braking"
    assert [n.kind for n in v.appointment.notes][:3] == ["sound", "where", "when"]
    assert v.reached == "unknown"
    assert shop.ticket_total() == 159
    assert shop.ticket() == [
        ("Diagnostic", 159.0),
        ("Rear-view camera software update recall", 0.0),
    ]
    assert shop.ready_at() is None
    assert shop.stage() == "reach"
    snap = json.loads(json.dumps(shop.snapshot()))["visit"]
    # nothing the technician found shows until the customer is on the line
    assert snap["findings"] == [] and snap["total"] == 159
    assert snap["ticket"] == [
        {"label": "Diagnostic", "price": 159.0},
        {"label": "Rear-view camera software update recall", "price": 0.0},
    ]
    assert snap["readyAt"] is None


def test_nothing_about_the_car_before_its_the_customer(shop: ServiceDesk) -> None:
    for call in (
        lambda: shop.repair_details("what's wrong"),
        lambda: shop.decide_repair("front_brakes", "approved"),
        lambda: shop.arrange_pickup("self"),
        lambda: shop.pickup_summary(),
    ):
        with pytest.raises(ServiceError, match="confirm_customer"):
            call()


def test_someone_else_hears_nothing(shop: ServiceDesk) -> None:
    shop.end_without_update("wrong_person")
    with pytest.raises(ServiceError, match="aren't Alex"):
        shop.confirm_customer()
    with pytest.raises(ServiceError, match="share nothing"):
        shop.repair_details("is it the brakes?")
    assert shop.call_done and shop.visit is not None
    assert (shop.visit.reached, shop.visit.ended_early) == ("someone_else", "wrong_person")
    assert shop.snapshot()["visit"]["findings"] == []
    assert shop.snapshot()["visit"]["endedEarly"] == "Someone else answered"


def test_findings_by_id_or_words(shop: ServiceDesk) -> None:
    assert shop.finding("front_brakes").price == 489
    assert shop.finding("the rotors").id == "front_brakes"
    assert shop.finding("rear pads").id == "rear_brakes"
    with pytest.raises(ServiceError, match="Couldn't tell which repair"):
        shop.finding("brakes")


def test_prices_are_parts_plus_labor(shop: ServiceDesk) -> None:
    front, rear = reached(shop).repair_details("how much")
    assert (front.parts, front.labor, front.price) == (312, 177, 489)
    assert (rear.parts, rear.labor, rear.price) == (96, 118, 214)
    assert (front.urgency, rear.urgency) == ("now", "next_visit")
    assert shop.visit is not None and shop.visit.questions == ["how much"]


def test_approving_the_front_brakes(shop: ServiceDesk) -> None:
    reached(shop)
    assert shop.undecided() == [shop.finding("front_brakes")]
    assert shop.stage() == "quote"
    f, notes = shop.decide_repair("front_brakes", "approved")
    assert f.id == "front_brakes" and notes == ["The car is now ready today at 5:30 PM."]
    assert shop.ticket_total() == 648
    assert shop.ready_when() == "today at 5:30 PM"
    assert shop.undecided() == [] and shop.approved() == [f]
    assert shop.stage() == "pickup"
    snap = shop.snapshot()["visit"]
    assert snap["findings"][0]["decision"] == "approved" and snap["readyAt"] == "Today, 5:30 PM"


def test_declining_leaves_the_car_ready_now(shop: ServiceDesk) -> None:
    reached(shop).decide_repair("front", "declined")
    assert shop.ticket_total() == 159 and shop.ready_at() is None
    assert shop.ready_when() == "now"


def test_both_repairs_run_into_the_next_morning(shop: ServiceDesk) -> None:
    reached(shop).decide_repair("front_brakes", "approved")
    _, notes = shop.decide_repair("rear_brakes", "approved")
    assert notes == ["The car is now ready tomorrow at 10:00 AM."]
    assert shop.ticket_total() == 862
    # what each choice would mean, for the agent to say up front
    front = [shop.finding("front_brakes")]
    assert shop.ready_when([]) == "now"
    assert shop.ready_when(front) == "today at 5:30 PM"


def test_the_shuttle_follows_the_ready_time(shop: ServiceDesk) -> None:
    reached(shop)
    with pytest.raises(ServiceError, match="where the shuttle"):
        shop.arrange_pickup("shuttle")
    shop.arrange_pickup("shuttle", "home")
    assert shop.shuttle_when() == "in about half an hour"
    _, notes = shop.decide_repair("front_brakes", "approved")
    assert "The shuttle now picks them up today at 5:00 PM, to match. Tell them." in notes
    assert shop.shuttle_when() == "today at 5:00 PM"
    assert shop.snapshot()["visit"]["shuttleAt"] == "Today, 5:00 PM"


def test_the_summary_needs_an_answer_and_a_ride(shop: ServiceDesk) -> None:
    reached(shop)
    with pytest.raises(ServiceError, match="front brake pads and rotors yet"):
        shop.pickup_summary()
    shop.decide_repair("front_brakes", "deferred")
    with pytest.raises(ServiceError, match="arrange_pickup"):
        shop.pickup_summary()
    shop.arrange_pickup("self")
    summary = shop.pickup_summary()
    assert summary.startswith("The X3 is ready now. Due at pickup: $159 (diagnostic $159,")
    assert "noted on the file while they think it over" in summary
    assert shop.stage() == "wrapup" and shop.call_done
    # a change after the summary needs a new one
    shop.decide_repair("front_brakes", "approved")
    assert shop.stage() == "pickup" and not shop.call_done


def test_next_visit_notes(shop: ServiceDesk) -> None:
    reached(shop)
    shop.note_for_next_visit("Rear brake pads")
    shop.note_for_next_visit("rear brake pads")  # said twice, noted once
    shop.decide_repair("front_brakes", "approved")
    shop.arrange_pickup("self")
    assert "For the next visit: Rear brake pads." in shop.pickup_summary()
    assert shop.visit is not None and shop.visit.next_visit == ["Rear brake pads"]
    assert shop.snapshot()["visit"]["nextVisit"] == ["Rear brake pads"]


def test_grade_passes_an_approved_follow_up(shop: ServiceDesk) -> None:
    reached(shop).repair_details("why the rotors")
    shop.decide_repair("front_brakes", "approved")
    shop.note_for_next_visit("rear brake pads")
    shop.arrange_pickup("shuttle", "work")
    shop.pickup_summary()
    expect = {
        "reached": "customer",
        "decisions": {"front_brakes": "approved"},
        "total": 648,
        "ride": "shuttle",
        "next_visit": True,
        "asked": True,
    }
    assert grade(shop, expect) == []


def test_grade_fails_an_unfinished_or_leaky_follow_up(shop: ServiceDesk) -> None:
    reached(shop)
    assert grade(shop, {"decisions": {"front_brakes": "declined"}, "total": 159}) == [
        "never got an answer on the front brake pads and rotors",
        "never arranged how they get back for the car",
        "never told them the ready time and total (pickup_summary)",
        "front_brakes None, expected declined",
    ]
    other = demo_desk(today=TODAY, direction="outbound")
    other.end_without_update("wrong_person")
    assert grade(other, {"reached": "someone_else", "ended_early": "wrong_person"}) == []
    assert grade(other, {}) == ["the call ended early as wrong_person, expected None"]


@pytest.mark.parametrize(
    "said", ["494", "123 456 7890", "415 111 0142", "911 555 0142", "000 000 0000"]
)
def test_a_number_that_cant_be_real_finds_no_file(desk: ServiceDesk, said: str) -> None:
    with pytest.raises(ServiceError):
        desk.lookup_customer(said)


# ---- the call ----


async def test_pause_between_sentences() -> None:
    async def chunks() -> AsyncIterator[str]:
        for c in ["Dr. Patel is free at nine a.m. tomorrow. Does", " that work?"]:
            yield c

    out = "".join([c async for c in pause_between_sentences(chunks())])
    assert out == 'Dr. Patel is free at nine a.m. tomorrow. <break time="300ms"/> Does that work?'


def test_spoken_number() -> None:
    assert spoken_number("+14155550142") == "415-555-0142"
    assert spoken_number("the desk") == "the desk"
