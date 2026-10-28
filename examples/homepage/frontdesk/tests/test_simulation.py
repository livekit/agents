from datetime import date

from clinic import Clinic
from receptionist import Caller, CallState
from simulation import grade

MONDAY = date(2026, 9, 28)
CASEY = Caller("Casey Morgan", date(1995, 6, 1))
CASEY_BOOKS = {
    "patient": "Casey Morgan",
    "born": "1995-06-01",
    "new_patient": True,
    "booked": "afternoon",
    "moved": False,
}


async def first_opening(clinic: Clinic, part_of_day: str):
    days = clinic.clinic_days()
    return (await clinic.check_availability(days[0], days[-1], part_of_day))[0]


async def booked_for(patient: str, registered: Caller = CASEY) -> CallState:
    clinic = Clinic(today=MONDAY)
    await clinic.register(registered.name, registered.date_of_birth)
    opening = await first_opening(clinic, "afternoon")
    await clinic.book(patient, opening, "Sore throat")
    return CallState(caller=registered, clinic=clinic)


async def test_a_new_patient_booked_in_the_afternoon_passes():
    state = await booked_for("Casey Morgan")
    assert grade(state, CASEY_BOOKS) == []
    assert grade(state, {"booked": "morning"}) == [
        "booked one for the afternoon, expected the morning"
    ]


async def test_booking_someone_else_fails():
    assert grade(await booked_for("Taylor Smith"), CASEY_BOOKS) == [
        "booked one for Taylor Smith, expected Casey Morgan"
    ]


async def test_a_chart_with_the_wrong_birthday_fails():
    wrong = Caller("Casey Morgan", date(1995, 6, 2))
    assert grade(await booked_for("Casey Morgan", wrong), CASEY_BOOKS) == [
        "Casey Morgan's chart has birthday 1995-06-02, expected 1995-06-01"
    ]


async def test_moving_an_appointment_is_told_apart_from_booking_one():
    clinic = Clinic(today=MONDAY)
    await clinic.reschedule(
        "Taylor Smith", "A-101", await first_opening(clinic, "morning")
    )
    state = CallState(caller=Caller("Taylor Smith", date(1988, 3, 14)), clinic=clinic)
    expected = {
        "patient": "Taylor Smith",
        "born": "1988-03-14",
        "new_patient": False,
        "booked": False,
        "moved": "morning",
    }
    assert grade(state, expected) == []


async def test_a_lookup_without_opening_the_chart_fails():
    expected = {"patient": "Jordan Lee", "new_patient": False, "booked": False}
    assert grade(CallState(clinic=Clinic(today=MONDAY)), expected) == [
        "verified nobody, expected Jordan Lee"
    ]
