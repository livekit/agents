from datetime import date

from clinic import Clinic
from simulation import grade

MONDAY = date(2026, 9, 28)


async def first_opening(clinic: Clinic, part_of_day: str):
    days = clinic.clinic_days()
    return (await clinic.check_availability(days[0], days[-1], part_of_day))[0]


async def test_a_new_patient_booked_in_the_afternoon_passes():
    clinic = Clinic(today=MONDAY)
    await clinic.register("Casey Morgan", date(1995, 6, 1))
    await clinic.book(
        "Casey Morgan", await first_opening(clinic, "afternoon"), "Sore throat"
    )
    expected = {"new_patient": "Casey Morgan", "booked": "afternoon", "moved": False}
    assert grade(clinic, expected) == []
    assert grade(clinic, {"booked": "morning"}) == [
        "booked one for the afternoon, expected the morning"
    ]


async def test_moving_an_appointment_is_told_apart_from_booking_one():
    clinic = Clinic(today=MONDAY)
    await clinic.reschedule(
        "Taylor Smith", "A-101", await first_opening(clinic, "morning")
    )
    expected = {"new_patient": False, "booked": False, "moved": "morning"}
    assert grade(clinic, expected) == []


async def test_nothing_changing_fails_a_call_that_should_book():
    assert grade(
        Clinic(today=MONDAY), {"new_patient": "Sam Rivera", "booked": "morning"}
    ) == [
        "registered nobody, expected Sam Rivera",
        "booked 0 appointments, expected one in the morning",
    ]
