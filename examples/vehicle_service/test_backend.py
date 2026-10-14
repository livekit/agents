from __future__ import annotations

import datetime as dt
import json

import backend
import pytest
from backend import ServiceDesk, ServiceError, demo_desk
from call import spoken_number
from grade import grade

pytestmark = pytest.mark.unit

TODAY = dt.date(2026, 9, 29)  # a Tuesday: the booking days run Wed Sep 30 to Tue Oct 6
X3_VIN = "WBX57DP0XPN123456"
SEDAN_VIN = "WBA5R1C05KA765432"


@pytest.fixture()
def desk() -> ServiceDesk:
    d = demo_desk(today=TODAY)
    d.lookup_customer("(415) 555-0142", "Alex Morgan")
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
    c = d.lookup_customer("+1 415.555.0142", "Alex Morgan")
    assert (c.name, c.phone, len(c.vehicles), c.new) == ("Alex Morgan", "4155550142", 2, False)
    assert d.lookup_customer("415 555 0142", "alex") is c
    with pytest.raises(ServiceError, match="ten digits"):
        d.lookup_customer("555 0142", "Alex Morgan")
    # a website visitor's file is under the name they gave the page
    assert demo_desk("jamie lee", TODAY).lookup_customer("4155550142", "Jamie").name == "jamie lee"


def test_a_name_and_number_with_no_file_get_a_new_one() -> None:
    d = demo_desk(today=TODAY)
    c = d.lookup_customer("437-465-7465", " DZ ")
    assert (c.name, c.phone, c.vehicles, c.new) == ("DZ", "4374657465", [], True)
    assert d.customer is c and d.on_file is None and d.stage() == "vehicle"
    snap = json.loads(json.dumps(d.snapshot()))
    assert snap["customer"] == {"name": "DZ", "phone": "(437) 465-7465", "isNew": True}
    assert snap["vehicles"] == [] and snap["existingAppointment"] is None
    with pytest.raises(ServiceError, match="no cars on their file yet.*add_vehicle"):
        d.select_vehicle("X5")
    # the same name and number again is the same new file
    assert d.lookup_customer("4374657465", "dz") is c
    # Alex's name on another number is someone new too
    other = d.lookup_customer("510 555 0199", "Alex Morgan")
    assert other.new and other.vehicles == [] and other is not c


@pytest.mark.parametrize("name", ["Jordan Morgan", "Jordan", "Alex Kim", "Alexander Morgan"])
def test_another_name_on_alexs_number_gets_a_new_file_and_nothing_of_his(name: str) -> None:
    d = demo_desk(today=TODAY)
    c = d.lookup_customer("415 555 0142", name)
    assert c.new and c.name == name and c.phone == "4155550142" and c.vehicles == []
    snap = json.loads(json.dumps(d.snapshot()))
    assert snap["customer"]["isNew"] is True and snap["customer"]["name"] == name
    assert snap["vehicles"] == [] and snap["existingAppointment"] is None
    # nothing of Alex's anywhere but the caller's own name and the price menu
    text = json.dumps({k: v for k, v in snap.items() if k not in ("customer", "menu")})
    for secret in ("Alex Morgan", "X3", "330i", X3_VIN, SEDAN_VIN, "21400", "23V-418", "Tire"):
        assert secret not in text, secret
    with pytest.raises(ServiceError, match="no cars on their file"):
        d.select_vehicle(X3_VIN)
    with pytest.raises(ServiceError, match="There's no other appointment"):
        d.booked("existing")
    with pytest.raises(ServiceError):
        d.check_recalls(X3_VIN)
    # a car they add is theirs alone, and the recall comes from the catalog, not Alex's file
    v = d.add_vehicle(2023, "X3", 12_000)
    assert v.vin != X3_VIN and v.miles == 12_000 and [r.id for r in v.recalls] == ["23V-418"]
    assert d.on_file is None and d.duplicates() == []


def test_switching_files_drops_the_other_files_car(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    c = desk.lookup_customer("4374657465", "DZ")
    assert desk.draft.vin is None and desk.draft_vehicle() is None
    assert desk.snapshot()["vehicles"] == [] and c.vehicles == []


def test_the_lookup_needs_a_name() -> None:
    with pytest.raises(ServiceError, match="Ask for their name first"):
        demo_desk(today=TODAY).lookup_customer("415 555 0142")


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
    with pytest.raises(ServiceError, match="On file.*add_vehicle"):
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
        desk.set_services(["warranty_repair"], "screen reboots", warranty_agreed=True)


def test_mileage_correction_that_ends_the_warranty_drops_the_warranty_repair(
    desk: ServiceDesk,
) -> None:
    desk.select_vehicle("X3")
    desk.set_services(
        ["warranty_repair", "oil_change"], "iDrive screen keeps rebooting", warranty_agreed=True
    )
    notes = desk.set_mileage(38_000)
    assert desk.draft.services == ["oil_change"]
    assert any("Warranty repair removed" in n for n in notes)


@pytest.mark.parametrize("concern", ["brakes grinding", "worn brake pads", "tire wear", "wipers"])
def test_wear_items_are_never_warranty(desk: ServiceDesk, concern: str) -> None:
    desk.select_vehicle("X3")
    with pytest.raises(ServiceError, match="wear items"):
        desk.set_services(["warranty_repair"], concern, warranty_agreed=True)
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
        desk.set_services(["warranty_repair"], "brakes grinding", warranty_agreed=True)
    assert "tire rotation" not in str(brakes.value)
    with pytest.raises(ServiceError, match="tire rotation"):
        desk.set_services(["warranty_repair"], "tire wear", warranty_agreed=True)


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
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting", warranty_agreed=True)
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
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting", warranty_agreed=True)
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
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting", warranty_agreed=True)
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
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting", warranty_agreed=True)
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
    desk.set_services(["warranty_repair"], "liftgate won't close", warranty_agreed=True)
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
        desk.set_services(["warranty_repair"], warranty_agreed=True)
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


# ---- what's already booked: the tire rotation on file, and changes after booking ----


def test_the_file_has_a_tire_rotation_on_the_330i_next_week(desk: ServiceDesk) -> None:
    e = desk.on_file
    assert e is not None and e.status == "booked" and e.vin == SEDAN_VIN
    assert (e.services, e.slot_id, e.estimate) == (["Tire rotation"], "tue-1430", 39)
    assert e.what == "tire rotation on the 330i" and e.when == "Tuesday October 6, 2:30 PM"
    assert not next(s for s in desk.slots if s.id == "tue-1430").open
    snap = json.loads(json.dumps(desk.snapshot()))
    assert snap["existingAppointment"]["status"] == "booked"
    assert snap["existingAppointment"]["movedFrom"] is None
    assert snap["pendingChange"] is None
    # nothing about the caller shows until they're looked up
    assert demo_desk(today=TODAY).snapshot()["existingAppointment"] is None


@pytest.mark.parametrize("days", range(7))
def test_the_appointment_on_file_is_always_next_week(days: int) -> None:
    today = TODAY + dt.timedelta(days=days)
    d = demo_desk(today=today)
    d.lookup_customer("4155550142", "Alex Morgan")
    assert d.on_file is not None
    slot = next(s for s in d.slots if s.id == d.on_file.slot_id)
    monday = today - dt.timedelta(days=today.weekday())
    assert 7 <= (slot.start.date() - monday).days < 14
    assert slot.start.time() == dt.time(14, 30)


def test_the_follow_up_has_no_appointment_on_file(shop: ServiceDesk) -> None:
    assert shop.on_file is None and shop.snapshot()["existingAppointment"] is None


def test_booking_the_x3_leaves_the_appointment_on_file_alone(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.read_back()
    appt = desk.book()
    assert appt is desk.appointment and desk.on_file is not None
    assert desk.on_file.status == "booked" and desk.on_file.moved_from is None
    assert appt.confirmation != desk.on_file.confirmation
    assert grade(desk, {"vehicle": "X3", "existing": "kept"}) == []


def test_a_second_tire_rotation_on_the_330i_is_never_booked(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    notes = desk.set_services(["oil_change", "tire_rotation"])
    assert any("already have the tire rotation on the 330i booked" in n for n in notes)
    desk.check_recalls(SEDAN_VIN)
    desk.choose_slot("wed-0730")
    desk.choose_transport("wait")
    desk.read_back()
    with pytest.raises(ServiceError, match="Never book it twice.*move_appointment"):
        desk.book()
    # the oil change alone is fine
    desk.set_services(["oil_change"])
    desk.read_back()
    desk.book()
    assert grade(desk, {"services": ["oil_change"]}) == []


def test_moving_the_appointment_on_file_is_read_back_first(desk: ServiceDesk) -> None:
    with pytest.raises(ServiceError, match="Read the move back first"):
        desk.move("existing", "thu-0900")
    said = desk.propose_move("existing", "thu-0900")
    assert said.startswith("Move the tire rotation on the 330i from Tuesday October 6, 2:30 PM")
    assert desk.on_file is not None and desk.on_file.slot_id == "tue-1430"
    assert desk.snapshot()["pendingChange"] == {
        "action": "move",
        "appointment": "existing",
        "confirmation": desk.on_file.confirmation,
        "to": "Thursday October 1, 9:00 AM",
    }
    # a yes to a different time than the one read back isn't a yes
    with pytest.raises(ServiceError, match="Read the move back first"):
        desk.move("existing", "thu-1030")
    e = desk.move("existing", "thu-0900")
    assert (e.slot_id, e.moved_from) == ("thu-0900", "Tuesday October 6, 2:30 PM")
    opened = {s.id: s.open for s in desk.slots}
    assert opened["tue-1430"] and not opened["thu-0900"]
    snap = desk.snapshot()
    assert snap["existingAppointment"]["movedFrom"] == "Tuesday October 6, 2:30 PM"
    assert snap["existingAppointment"]["when"] == "Thursday October 1, 9:00 AM"
    assert snap["pendingChange"] is None
    assert grade(desk, {"booked": False, "existing": "moved", "existing_day": "thu"}) == []
    assert grade(desk, {"booked": False}) == ["the appointment on file was moved, expected kept"]


def test_a_move_only_lands_on_an_open_time_that_fits(desk: ServiceDesk) -> None:
    taken = next(s for s in desk.slots if not s.open and s.id != "tue-1430")
    with pytest.raises(ServiceError, match="is taken"):
        desk.propose_move("existing", taken.id)
    with pytest.raises(ServiceError, match="already booked for"):
        desk.propose_move("existing", "tue-1430")
    with pytest.raises(ServiceError, match="Nothing's been booked on this call"):
        desk.propose_move("new", "thu-0900")
    # a half-hour visit fits any time, whatever is on the draft
    desk.select_vehicle("X3")
    desk.set_services(["warranty_repair"], "iDrive screen keeps rebooting", warranty_agreed=True)
    assert "wed-1600" in {s.id for s in desk.open_slots("wednesday", hours=0.5)}
    assert "wed-1600" not in {s.id for s in desk.open_slots("wednesday")}


def test_cancelling_the_appointment_on_file_needs_a_yes(desk: ServiceDesk) -> None:
    with pytest.raises(ServiceError, match="Read the cancellation back first"):
        desk.cancel("existing")
    assert desk.propose_cancel("existing") == (
        "Cancel the tire rotation on the 330i on Tuesday October 6, 2:30 PM? Nothing is "
        "charged for cancelling."
    )
    e = desk.cancel("existing")
    assert e.status == "cancelled"
    assert next(s for s in desk.slots if s.id == "tue-1430").open
    assert desk.snapshot()["existingAppointment"]["status"] == "cancelled"
    with pytest.raises(ServiceError, match="already cancelled"):
        desk.propose_cancel("existing")
    assert grade(desk, {"booked": False, "existing": "cancelled"}) == []
    # with it cancelled, a tire rotation on the 330i is no longer a duplicate
    desk.select_vehicle("330i")
    assert desk.set_services(["tire_rotation"]) == []


def test_the_visit_booked_on_this_call_can_be_moved(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.read_back()
    booked = desk.book()
    desk.propose_move("new", "thu-0900")
    appt = desk.move("new", "thu-0900")
    assert appt is booked and appt.confirmation == booked.confirmation
    assert (appt.slot_id, appt.moved_from) == ("thu-0900", "Wednesday September 30, 7:30 AM")
    assert next(s for s in desk.slots if s.id == "wed-0730").open
    assert desk.draft.slot_id == "thu-0900" and desk.stage() == "booked"
    assert grade(desk, {"day": "thu"}) == []


def test_the_visit_booked_on_this_call_can_be_cancelled(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.read_back()
    first = desk.book()
    desk.propose_cancel("new")
    desk.cancel("new")
    assert first.status == "cancelled" and next(s for s in desk.slots if s.id == "wed-0730").open
    assert desk.snapshot()["appointment"]["status"] == "cancelled"
    assert grade(desk, {"booked": False, "cancelled": True}) == []
    assert grade(desk, {"vehicle": "X3"}) == ["nothing was booked"]
    # booking it again is a new booking: read back again, under a new number
    with pytest.raises(ServiceError, match="read it to them again"):
        desk.book()
    desk.read_back()
    again = desk.book()
    assert again.confirmation != first.confirmation and again.status == "booked"


def test_a_service_added_after_booking_keeps_the_number(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.read_back()
    booked = desk.book()
    with pytest.raises(ServiceError, match="call change_booking first"):
        desk.set_services(["oil_change", "tire_rotation"])
    desk.change_booking()
    assert desk.snapshot()["draft"]["changingBooking"] and desk.stage() != "booked"
    with pytest.raises(ServiceError, match="open for changes"):
        desk.propose_move("new", "thu-0900")
    desk.set_services(["oil_change", "tire_rotation"])
    # its own time is still its own while it's open for changes
    desk.choose_slot("wed-0730")
    with pytest.raises(ServiceError, match="read it to them again"):
        desk.book()
    desk.read_back()
    appt = desk.book()
    assert appt.confirmation == booked.confirmation and appt.moved_from is None
    assert appt.services == ["Oil change", "Tire rotation"] and appt.estimate == 118
    assert not desk.changing and desk.stage() == "booked"
    assert [s.id for s in desk.slots if s.id == "wed-0730" and not s.open] == ["wed-0730"]


def test_nobody_is_offered_a_handoff(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    desk.read_back()
    desk.book()
    refusals = []
    for call in (lambda: desk.choose_slot("thu-0800"), lambda: desk.add_note("since", "a week")):
        with pytest.raises(ServiceError) as e:
            call()
        refusals.append(str(e.value))
    refusals.append(desk.quote("brake job").answer)
    refusals.append(backend._STOP_ADVICE)
    for said in refusals:
        assert "connect" not in said and "advisor" not in said, said
    assert "towed in" in backend._STOP_ADVICE and "can't arrange a tow" in backend._STOP_ADVICE


def test_payment_is_at_the_service_desk_on_pickup(shop: ServiceDesk) -> None:
    assert "in person at the service desk when they pick the car up" in backend.PAYMENT
    assert "never ask for or take card details" in backend.PAYMENT
    reached(shop).decide_repair("front_brakes", "approved")
    shop.arrange_pickup("self")
    assert (
        "Due at pickup: $648 (diagnostic $159, rear-view camera software update recall no "
        "charge, front brake pads and rotors $489), paid in person at the service desk when "
        "they collect the car; nothing is paid over the phone."
    ) in shop.pickup_summary()


@pytest.mark.parametrize(
    "said", ["494", "123 456 7890", "415 111 0142", "911 555 0142", "000 000 0000"]
)
def test_a_number_that_cant_be_real_finds_no_file(desk: ServiceDesk, said: str) -> None:
    with pytest.raises(ServiceError):
        desk.lookup_customer(said)


# ---- the call ----


def test_spoken_number() -> None:
    assert spoken_number("+14155550142") == "415-555-0142"
    assert spoken_number("the desk") == "the desk"


# ---- what the edge-case simulations found ----


@pytest.mark.parametrize(
    ("kind", "said", "why"),
    [
        ("other", "my car won't stop safely", "brakes"),
        ("other", "the brakes failed on the freeway", "brakes"),
        ("other", "pedal goes to the floor", "brakes"),
        ("other", "smoke coming from under the hood", "smoke"),
        ("other", "a burning smell", "smoke"),
        ("other", "I lost the steering for a second", "steering"),
        ("warning_lights", "the engine light is flashing", "warning light"),
        ("warning_lights", "a red light on the dash", "warning light"),
        ("other", "it's overheating", "overheating"),
        ("warning_lights", "the oil light", "warning light"),
    ],
)
def test_what_means_dont_drive_it(desk: ServiceDesk, kind: str, said: str, why: str) -> None:
    desk.add_note(kind, said)
    stop = desk.stop_driving()
    assert stop is not None and why in stop


@pytest.mark.parametrize(
    ("kind", "said"),
    [
        ("sound", "grinding, metal on metal"),
        ("other", "the noise won't stop"),
        ("warning_lights", "check engine light, steady"),
        ("warning_lights", "no warning lights"),
        ("other", "no smoke or anything"),
        ("when", "squeals when I stop at lights"),
    ],
)
def test_what_doesnt(desk: ServiceDesk, kind: str, said: str) -> None:
    desk.add_note(kind, said)
    assert desk.stop_driving() is None


def test_an_unsafe_car_is_read_back_and_booked_with_the_warning(desk: ServiceDesk) -> None:
    desk.set_concern("smoke from under the hood")
    desk.select_vehicle("330i")
    desk.set_services(["diagnostic"])
    desk.check_recalls(SEDAN_VIN)
    desk.choose_slot("wed-0730")
    desk.choose_transport("wait")
    said = desk.read_back()
    assert "It isn't safe to drive (smoke or burning): have it towed in, not driven" in said
    appt = desk.book()
    assert appt.stop_driving == "smoke or burning"
    snap = desk.snapshot()
    assert snap["appointment"]["stopDriving"] == snap["draft"]["stopDriving"]
    cause = desk.likely_cause()
    assert cause is not None and cause.unsafe and "towed in" in cause.advice


@pytest.mark.parametrize(
    ("concern", "says"),
    [
        ("grinding and recall", "never for 'grinding'"),
        ("oil change plus recall", "never for 'oil change'"),
        ("due for maintenance", "never for 'maintenance'"),
    ],
)
def test_a_warranty_repair_is_only_for_a_defect(desk: ServiceDesk, concern: str, says: str) -> None:
    desk.select_vehicle("X3")
    with pytest.raises(ServiceError, match=f"{says}.*Don't try warranty_repair again"):
        desk.set_services(["warranty_repair"], concern, warranty_agreed=True)
    assert desk.draft.services == []


def test_the_notes_count_for_warranty_not_just_the_concern(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.add_note("sound", "grinding")
    with pytest.raises(ServiceError, match="never for 'grinding'"):
        desk.set_services(["warranty_repair"], "a noise from the front", warranty_agreed=True)


def test_a_warranty_repair_needs_the_callers_say_so(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    with pytest.raises(ServiceError, match="warranty_agreed true"):
        desk.set_services(["warranty_repair"], "oil leak under the engine")
    desk.set_services(["warranty_repair"], "oil leak under the engine", warranty_agreed=True)
    # already on the visit: changing the rest doesn't ask again
    desk.set_services(["warranty_repair", "oil_change"])
    assert desk.draft.services == ["warranty_repair", "oil_change"]


def test_grade_fails_a_warranty_repair_for_a_grinding(desk: ServiceDesk) -> None:
    desk.select_vehicle("X3")
    desk.set_services(["warranty_repair"], "liftgate won't close", warranty_agreed=True)
    desk.check_recalls(X3_VIN)
    desk.decide_recall(False)
    desk.choose_slot("wed-0730")
    desk.choose_transport("shuttle")
    desk.read_back()
    appt = desk.book()
    assert grade(desk, {}) == []
    appt.concern = "grinding when braking"
    assert grade(desk, {}) == [
        "warranty repair booked for 'grinding when braking', which isn't a defect"
    ]


def test_a_weekday_is_today_only_when_today_has_openings() -> None:
    wed = dt.date(2026, 10, 14)  # a Wednesday: the calendar starts tomorrow
    d = demo_desk(today=wed)
    assert d.resolve_day("wednesday") == dt.date(2026, 10, 21)
    assert d.resolve_day("next wednesday") == dt.date(2026, 10, 21)
    assert {s.start.date() for s in d.open_slots("wednesday")} == {dt.date(2026, 10, 21)}
    assert d.day_word(dt.date(2026, 10, 15)) == "tomorrow"
    assert d.day_word(dt.date(2026, 10, 16)) == "this Friday"
    assert d.day_word(dt.date(2026, 10, 21)) == "next Wednesday"
    # a calendar with a same-day opening counts today
    next(s for s in d.slots if s.open).start = dt.datetime.combine(wed, dt.time(13, 0))
    assert d.resolve_day("wednesday") == wed
    assert d.resolve_day("next wednesday") == dt.date(2026, 10, 21)


def test_the_read_back_says_which_day_it_is(desk: ServiceDesk) -> None:
    oil_change_on_x3(desk)
    assert "Wednesday September 30, 7:30 AM (tomorrow) with" in desk.read_back()


def test_an_approval_limit_is_on_the_booking(desk: ServiceDesk) -> None:
    desk.select_vehicle("330i")
    desk.set_concern("check engine light")
    desk.set_services(["diagnostic"])
    desk.check_recalls(SEDAN_VIN)
    desk.choose_slot("wed-0730")
    desk.choose_transport("wait")
    desk.read_back()
    with pytest.raises(ServiceError, match="dollar amount"):
        desk.set_approval_limit(0)
    desk.set_approval_limit(500)
    with pytest.raises(ServiceError, match="read it to them again"):
        desk.book()
    assert "we'll call before doing any work over $500" in desk.read_back()
    appt = desk.book()
    assert appt.approval_limit == 500
    assert desk.snapshot()["appointment"]["approvalLimit"] == 500


def test_a_second_concern_goes_in_the_summary(desk: ServiceDesk) -> None:
    desk.add_note("sound", "grinding")
    desk.add_note("warning_lights", "check engine light came on")
    desk.set_concern("grinding when braking")
    oil_change_on_x3(desk)
    with pytest.raises(ServiceError, match="leaves out the warning light"):
        desk.read_back()
    desk.set_concern("grinding when braking; check engine light on")
    desk.set_services(["diagnostic"])
    assert "check engine light on" in desk.read_back()


@pytest.mark.parametrize("said", ["my 3", "the three", "3"])
def test_the_3_could_be_either_car(desk: ServiceDesk, said: str) -> None:
    with pytest.raises(ServiceError, match="could be the X3 or the 330i"):
        desk.vehicle(said)
    assert desk.vehicle("my 2019 3 series").vin == SEDAN_VIN


def test_a_friend_can_collect_the_car(shop: ServiceDesk) -> None:
    reached(shop).decide_repair("front_brakes", "approved")
    shop.arrange_pickup("self", collected_by="their friend Sam")
    said = shop.pickup_summary()
    assert "their friend Sam is collecting it: they give Alex's name or the repair order" in said
    assert shop.snapshot()["visit"]["collectedBy"] == "their friend Sam"
    assert "authorized" not in backend.PICKUP_BY_SOMEONE_ELSE


def test_no_early_end_once_a_repair_is_decided(shop: ServiceDesk) -> None:
    reached(shop).decide_repair("front_brakes", "approved")
    with pytest.raises(ServiceError, match="pickup_summary"):
        shop.end_without_update("call_back_later")
    assert shop.visit is not None and shop.visit.ended_early is None
    shop.visit.ended_early = "call_back_later"
    assert "ended without the pickup summary after a repair was decided" in grade(shop, {})


def test_a_new_noise_is_noted_neutrally(shop: ServiceDesk) -> None:
    reached(shop)
    shop.note_concern("new rattle since the repair")
    shop.note_concern("New rattle since the repair")
    shop.decide_repair("front_brakes", "declined")
    shop.arrange_pickup("self")
    said = shop.pickup_summary()
    assert "Noted for the technician to check when they come in: new rattle" in said
    assert "free" not in said and shop.snapshot()["visit"]["concerns"] == [
        "new rattle since the repair"
    ]


def test_grade_checks_the_new_expectations(desk: ServiceDesk) -> None:
    desk.set_concern("smoke from under the hood")
    desk.select_vehicle("330i")
    desk.set_services(["diagnostic"])
    desk.check_recalls(SEDAN_VIN)
    desk.choose_slot("wed-0730")
    desk.choose_transport("wait")
    desk.set_approval_limit(500)
    desk.read_back()
    desk.book()
    assert grade(desk, {"stop_driving": True, "approval_limit": 500}) == []
    assert grade(desk, {"approval_limit": 300}) == ["approval limit 500.0, expected 300"]
    assert grade(desk, {"new_customer": True}) == ["never set up a new file for the caller"]


def test_grade_checks_the_follow_ups_pickup_and_concerns(shop: ServiceDesk) -> None:
    reached(shop).decide_repair("front_brakes", "approved")
    shop.arrange_pickup("self", collected_by="Sam")
    shop.pickup_summary()
    assert grade(shop, {"collected_by": True, "concerns": True}) == [
        "never noted the new concern on the visit"
    ]


# ---- a car the caller adds to their file ----


def test_add_vehicle_on_a_new_file_selects_it_and_works_out_the_warranty() -> None:
    d = demo_desk(today=TODAY)
    d.lookup_customer("4374657465", "DZ")
    x5 = d.add_vehicle(2021, "x five", 40_000)
    assert (x5.label, x5.body, x5.miles, x5.added, x5.recalls) == (
        "2021 BMW X5",
        "suv",
        40_000,
        True,
        [],
    )
    assert d.draft.vin == x5.vin and len(x5.vin) == 17 and d.stage() == "symptoms"
    # in service July 1 of the model year: past 3 years, so out of warranty
    assert x5.in_service == dt.date(2021, 7, 1) and not x5.in_warranty(TODAY)
    assert x5.warranty_reason(TODAY).startswith("Out of warranty: about 5 years")
    i4 = d.add_vehicle(2025, "i4", 9_000, vin="WBA 73AW0 5SFR 12345")
    assert i4.vin == "WBA73AW05SFR12345" and i4.in_warranty(TODAY) and d.draft.vin == i4.vin
    # a new car's July 1 is never later than today
    assert d.add_vehicle(2027, "X1", 12).in_service == TODAY
    with pytest.raises(ServiceError, match="mileage doesn't look right"):
        d.add_vehicle(2024, "X2", -5)
    with pytest.raises(ServiceError, match="year doesn't look right"):
        d.add_vehicle(1890, "X2", 5)
    vehicles = d.snapshot()["vehicles"]
    assert [(v["label"], v["added"], v["selected"]) for v in vehicles] == [
        ("2021 BMW X5", True, False),
        ("2025 BMW i4", True, False),
        ("2027 BMW X1", True, True),
    ]
    assert vehicles[0]["inWarranty"] is False and vehicles[1]["inWarranty"] is True


def test_an_added_2023_x3_gets_the_catalog_recall_and_other_cars_none() -> None:
    d = demo_desk(today=TODAY)
    d.lookup_customer("4374657465", "DZ")
    x3 = d.add_vehicle(2023, "X3 M40i", 30_000)
    assert x3.name == "X3 M40i"
    assert [(r.id, r.title, r.hours) for r in d.check_recalls(x3.vin)] == [
        ("23V-418", "Rear-view camera software update", 0.5)
    ]
    for year, model in ((2022, "X3"), (2024, "X3"), (2023, "X5"), (2023, "330i")):
        assert backend.recalls_for(backend.bmw_model(model)[1], year) == []  # type: ignore[index]
    # each car gets its own copy of the recall
    x3.recalls[0].parts_in_stock = False
    assert backend.recalls_for("X3", 2023)[0].parts_in_stock
    # Alex's seeded cars keep theirs
    alex = demo_desk(today=TODAY)
    alex.lookup_customer("4155550142", "Alex Morgan")
    assert [r.id for r in alex.check_recalls(X3_VIN)] == ["23V-418"]
    assert alex.check_recalls(SEDAN_VIN) == []


def test_everything_else_works_on_an_added_car() -> None:
    d = demo_desk(today=TODAY)
    d.lookup_customer("4374657465", "DZ")
    v = d.add_vehicle(2021, "X5", 40_000)
    d.add_note("sound", "clunk")
    d.add_note("when", "over bumps")
    d.set_concern("clunk over bumps")
    with pytest.raises(ServiceError, match="isn't in warranty"):
        d.set_services(["warranty_repair"], "liftgate won't close", warranty_agreed=True)
    d.set_services(["diagnostic"])
    assert d.check_recalls(v.vin) == []
    d.choose_slot("thu-0800")
    d.choose_transport("shuttle")
    d.set_mileage(40_250)
    assert d.read_back().startswith("2021 BMW X5; diagnostic for clunk over bumps; Thursday")
    appt = d.book()
    assert (appt.vehicle, appt.services, appt.recall_id) == ("2021 BMW X5", ["Diagnostic"], None)
    assert appt.what == "diagnostic on the X5"
    assert grade(d, {"vehicle": "X5", "services": ["diagnostic"], "new_customer": True}) == []
    snap = d.snapshot()
    assert snap["appointment"]["vehicle"] == "2021 BMW X5" and snap["draft"]["mileage"] == 40_250


def test_a_car_on_alexs_file_or_not_and_a_3_series_says_its_name() -> None:
    d = demo_desk(today=TODAY)
    d.lookup_customer("4155550142", "Alex Morgan")
    with pytest.raises(ServiceError, match="2023 BMW X3 xDrive30i is already on their file"):
        d.add_vehicle(2023, "X3", 22_000)
    v = d.add_vehicle(2018, "5 series", 81_000)
    assert v.label == "2018 BMW 5 Series" and len(d.customer.vehicles) == 3  # type: ignore[union-attr]
    assert d.vehicle("the 5 series") is v and d.draft.vin == v.vin
    d.set_services(["oil_change"])
    d.check_recalls(v.vin)
    d.choose_slot("wed-0730")
    d.choose_transport("wait")
    d.read_back()
    assert d.book().what == "oil change on the 5 Series"


@pytest.mark.parametrize(
    ("said", "name"),
    [
        ("X5", "X5"),
        ("BMW X7", "X7"),
        ("x 1", "X1"),
        ("330i", "330i"),
        ("M340i xDrive", "M340i xDrive"),
        ("3 Series", "3 Series"),
        ("two series", "2 Series"),
        ("8 series", "8 Series"),
        ("i4 M50", "i4 M50"),
        ("iX", "iX"),
        ("X3 xDrive30i", "X3 xDrive30i"),
        ("M3 Competition", "M3 Competition"),
    ],
)
def test_bmw_models_the_desk_services(said: str, name: str) -> None:
    found = backend.bmw_model(said)
    assert found is not None and found[0] == name


@pytest.mark.parametrize(
    "said", ["Honda Civic", "Toyota Camry", "Tesla Model 3", "Lexus RX 350", "Mini Cooper", "BMW"]
)
def test_a_car_that_isnt_a_bmw_is_refused(said: str) -> None:
    d = demo_desk(today=TODAY)
    d.lookup_customer("4374657465", "DZ")
    with pytest.raises(ServiceError, match="only services BMWs"):
        d.add_vehicle(2020, said, 30_000)
    assert d.customer is not None and d.customer.vehicles == [] and d.draft.vin is None
