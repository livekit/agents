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


async def test_nothing_reads_back_until_the_visit_is_complete(alex: ServiceDesk):
    await alex.choose_vehicle("V1")
    with pytest.raises(
        ServiceDeskError, match="a service, a time, how they'll get around"
    ):
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
