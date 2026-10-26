from datetime import date

import pytest

from service_desk import ServiceDesk, ServiceDeskError

MONDAY = date(2026, 10, 19)
ALEX = ("Alex Morgan", "415-555-0142")


@pytest.fixture
def desk() -> ServiceDesk:
    return ServiceDesk(MONDAY)


@pytest.fixture
async def alex(desk: ServiceDesk) -> ServiceDesk:
    await desk.find_customer(*ALEX)
    return desk


async def visit_ready_to_book(desk: ServiceDesk) -> None:
    await desk.choose_vehicle("V1")
    await desk.describe_problem(summary="grinding from the front when braking")
    await desk.add_service("diagnostic")
    await desk.choose_time("2026-10-21T07:30")
    await desk.choose_ride("shuttle")


async def test_a_file_opens_only_for_its_exact_name_and_number(desk: ServiceDesk):
    assert await desk.find_customer("alex  morgan", "+1 (415) 555-0142") is not None
    other = ServiceDesk(MONDAY)
    assert await other.find_customer("Alex Morgen", "4155550142") is None
    assert await other.find_customer("Alex", "4155550142") is None
    assert other.customer is None


async def test_a_partial_phone_number_is_refused(desk: ServiceDesk):
    with pytest.raises(ServiceDeskError, match="ten digits"):
        await desk.find_customer("Alex Morgan", "555-0142")


async def test_a_new_file_starts_empty_and_never_duplicates_one(desk: ServiceDesk):
    customer = await desk.open_new_file("Sam Lee", "4155550188")
    assert customer.new and customer.vehicles == []
    with pytest.raises(ServiceDeskError, match="already a file"):
        await ServiceDesk(MONDAY).open_new_file(*ALEX)


async def test_choosing_the_x3_surfaces_its_open_recall(alex: ServiceDesk):
    await alex.choose_vehicle("V1")
    car = alex.visit.vehicle
    assert car is not None and car.recall is not None
    assert car.in_warranty(MONDAY)


async def test_switching_cars_takes_off_the_other_cars_recall(alex: ServiceDesk):
    await alex.choose_vehicle("V1")
    await alex.add_service("recall")
    changed = await alex.choose_vehicle("V2")
    assert "recall" not in alex.visit.services
    assert changed


async def test_mileage_only_goes_up(alex: ServiceDesk):
    with pytest.raises(ServiceDeskError, match="only goes up"):
        await alex.choose_vehicle("V1", mileage=1_000)
    await alex.choose_vehicle("V1", mileage=22_000)
    assert alex.visit.vehicle is not None and alex.visit.vehicle.miles == 22_000


async def test_a_car_already_on_file_is_not_added_twice(alex: ServiceDesk):
    with pytest.raises(ServiceDeskError, match="already on their file"):
        await alex.add_vehicle(2023, "X3", 21_000)


async def test_an_added_2023_x3_gets_the_same_recall(desk: ServiceDesk):
    await desk.open_new_file("Sam Lee", "4155550188")
    car = await desk.add_vehicle(2023, "X3", 12_000)
    assert car.recall is not None and desk.visit.vehicle is car


async def test_a_warranty_repair_needs_a_failed_part_a_problem_and_warranty(
    alex: ServiceDesk,
):
    await alex.choose_vehicle("V1")
    with pytest.raises(ServiceDeskError, match="Record the problem"):
        await alex.add_service("warranty_repair", "liftgate_or_trunk")
    await alex.describe_problem(summary="liftgate won't close")
    with pytest.raises(ServiceDeskError, match="failed_part"):
        await alex.add_service("warranty_repair")
    await alex.add_service("warranty_repair", "liftgate_or_trunk")
    assert alex.visit.services == ["warranty_repair"]

    await alex.choose_time("2026-10-21T07:30")
    await alex.choose_ride("wait")
    await alex.choose_vehicle("V2")
    with pytest.raises(ServiceDeskError, match="out of warranty"):
        alex.read_back()


async def test_a_diagnostic_needs_the_problem_first(alex: ServiceDesk):
    await alex.choose_vehicle("V1")
    with pytest.raises(ServiceDeskError, match="Record the problem"):
        await alex.add_service("diagnostic")


async def test_a_loaner_is_only_for_long_warranty_or_recall_visits(alex: ServiceDesk):
    await alex.choose_vehicle("V1")
    await alex.add_service("oil_change")
    with pytest.raises(ServiceDeskError, match="loaners are only for"):
        await alex.choose_ride("loaner")
    await alex.describe_problem(summary="liftgate won't close")
    await alex.add_service("warranty_repair", "liftgate_or_trunk")
    await alex.choose_ride("loaner")
    assert alex.visit.ride == "loaner"


async def test_describing_the_problem_keeps_what_was_left_out(alex: ServiceDesk):
    await alex.describe_problem(sound="grinding", where="front")
    await alex.describe_problem(when="braking")
    p = alex.visit.problem
    assert (p.sound, p.where, p.when) == ("grinding", "front", "braking")


async def test_openings_are_only_on_bookable_days_and_fit_the_visit(alex: ServiceDesk):
    with pytest.raises(ServiceDeskError, match="isn't bookable"):
        alex.openings("2026-10-19")
    with pytest.raises(ServiceDeskError, match="YYYY-MM-DD"):
        alex.openings("next tuesday")
    await alex.choose_vehicle("V1")
    await alex.describe_problem(summary="liftgate won't close")
    await alex.add_service("warranty_repair", "liftgate_or_trunk")
    afternoon = alex.openings("2026-10-21", "afternoon")
    assert all(s.starts_at.hour < 14 for s in afternoon)
    assert not any(s.taken for s in alex.openings())


async def test_nothing_reads_back_until_the_visit_is_complete(alex: ServiceDesk):
    await alex.choose_vehicle("V1")
    with pytest.raises(
        ServiceDeskError, match="a service, a time, how they'll get around"
    ):
        alex.read_back()


async def test_a_service_already_booked_on_file_is_not_booked_again(alex: ServiceDesk):
    await alex.choose_vehicle("V2")
    await alex.add_service("tire_rotation")
    await alex.choose_time("2026-10-21T07:30")
    await alex.choose_ride("wait")
    with pytest.raises(ServiceDeskError, match="already booked"):
        alex.read_back()


async def test_a_change_after_booking_updates_the_same_booking(alex: ServiceDesk):
    await visit_ready_to_book(alex)
    first = await alex.book()
    assert not alex.changed_since_booked()
    await alex.add_service("recall")
    assert alex.changed_since_booked()
    second = await alex.book()
    assert second is first and second.services == ["diagnostic", "recall"]
    assert sum(a.status == "booked" for a in alex.customer_appointments()) == 2


async def test_moving_and_cancelling_free_the_old_time(alex: ServiceDesk):
    on_file = alex.appointment("RO-48194")
    old = on_file.slot
    alex.move_read_back("RO-48194", "2026-10-22T08:00")
    await alex.move("RO-48194", "2026-10-22T08:00")
    assert not old.taken and on_file.slot.taken and on_file.moved_from is not None
    await alex.cancel("RO-48194")
    assert on_file.status == "cancelled" and not on_file.slot.taken
    with pytest.raises(ServiceDeskError, match="already cancelled"):
        alex.appointment("RO-48194")


def test_a_price_comes_only_from_the_menu(desk: ServiceDesk):
    assert desk.quote("oil_change")["price"] == 79
    assert desk.quote("recall")["price"] == 0
    assert desk.quote("something_else")["price"] is None


async def test_the_callback_total_and_ready_time_follow_the_decisions():
    desk = ServiceDesk.for_callback(MONDAY)
    assert desk.total_due() == 159
    await desk.decide("front_brakes", "approved")
    assert desk.total_due() == 159 + 489
    assert desk.pickup_summary()["readyAt"] == "today at 5:30 PM"
    await desk.decide("rear_brakes", "approved")
    assert desk.pickup_summary()["readyAt"] == "tomorrow at 10:00 AM"
    with pytest.raises(ServiceDeskError, match="already decided"):
        await desk.end_early("call_back_later")


async def test_the_shuttle_needs_somewhere_to_pick_them_up():
    desk = ServiceDesk.for_callback(MONDAY)
    with pytest.raises(ServiceDeskError, match="Ask where"):
        await desk.arrange_pickup("shuttle")
    assert desk.shop is not None and desk.shop.ride is None
    await desk.arrange_pickup("shuttle", "home")
    assert desk.pickup_summary()["shuttleFrom"] == "home"
