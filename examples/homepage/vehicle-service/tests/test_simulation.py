from datetime import date

from service_desk import ServiceDesk
from simulation import grade

MONDAY = date(2026, 10, 19)
BOOKED_VISIT = {
    "booked": True,
    "vehicle": "X3",
    "services": ["diagnostic"],
    "day": "2026-10-21",
    "ride": "shuttle",
}


async def booked_desk() -> ServiceDesk:
    desk = ServiceDesk(MONDAY)
    await desk.find_customer("Alex Morgan", "4155550142")
    await desk.choose_vehicle("V1")
    await desk.describe_problem(summary="grinding from the front when braking")
    await desk.add_service("diagnostic")
    await desk.choose_time("2026-10-21T07:30")
    await desk.choose_ride("shuttle")
    await desk.book()
    return desk


async def test_the_visit_that_was_asked_for_passes():
    assert grade(await booked_desk(), BOOKED_VISIT) == []


async def test_a_wrong_booking_fails_the_run():
    problems = grade(
        await booked_desk(), {**BOOKED_VISIT, "day": "2026-10-22", "ride": "wait"}
    )
    assert len(problems) == 2


async def test_moving_the_appointment_on_file_when_it_should_stay_fails():
    desk = ServiceDesk(MONDAY)
    await desk.find_customer("Alex Morgan", "4155550142")
    await desk.move("RO-48194", "2026-10-21T08:00")
    assert grade(desk, {"booked": False}) == [
        "the appointment on file was moved, expected kept"
    ]
    assert grade(desk, {"booked": False, "on_file": "moved"}) == []


async def test_a_callback_is_graded_on_its_decisions():
    desk = ServiceDesk.for_callback(MONDAY)
    await desk.decide("front_brakes", "approved")
    await desk.arrange_pickup("shuttle", "home")
    assert (
        grade(desk, {"decisions": {"front_brakes": "approved"}, "ride": "shuttle"})
        == []
    )
    assert grade(desk, {"decisions": {"front_brakes": "declined"}}) == [
        "front_brakes approved, expected declined"
    ]
