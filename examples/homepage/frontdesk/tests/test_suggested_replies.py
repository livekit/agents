from datetime import date

from clinic import Clinic
from receptionist import Caller, CallState, caller_script

MONDAY = date(2026, 9, 28)


def test_script_moves_the_seeded_appointment_to_the_fourth_clinic_day():
    script = caller_script(Clinic(today=MONDAY), CallState())

    assert "Taylor Smith" in script and "March 14, 1988" in script
    assert "Knee pain follow-up" in script
    assert "move their appointment to Friday morning" in script
    assert "Alex Morgan" in script and "A L E X, M O R G A N" in script


async def test_script_follows_the_call_from_verified_to_moved():
    clinic = Clinic(today=MONDAY)
    state = CallState(caller=Caller("Taylor Smith", date(1988, 3, 14)))

    assert "take the first one" in caller_script(clinic, state)

    friday = clinic.clinic_days()[3]
    opening = (await clinic.check_availability(friday, friday, "morning"))[0]
    await clinic.reschedule("Taylor Smith", clinic.appointments[0].id, opening)

    assert "goodbye" in caller_script(clinic, state)
