from datetime import date, datetime, time

import pytest

from clinic import CLINIC_TZ, Clinic, ClinicError

MONDAY = date(2026, 9, 28)
TAYLOR_DOB = date(1988, 3, 14)


@pytest.fixture
def published() -> dict:
    return {}


@pytest.fixture
def clinic(published: dict) -> Clinic:
    async def publish(key, value):
        published[key] = value

    return Clinic(today=MONDAY, publish=publish)


def at(day: date, t: time) -> datetime:
    return datetime.combine(day, t, tzinfo=CLINIC_TZ)


async def test_patient_on_file_verifies_with_name_and_birth_date(clinic, published):
    assert await clinic.verify("Taylor Smith", TAYLOR_DOB)
    [appointment] = clinic.upcoming("Taylor Smith")
    assert appointment.starts_at == at(date(2026, 9, 30), time(14, 30))
    assert published["caller"] == {"name": "Taylor Smith", "new": False}


async def test_wrong_birth_date_does_not_verify(clinic, published):
    assert not await clinic.verify("Taylor Smith", date(1988, 3, 15))
    assert "caller" not in published


async def test_name_must_match_exactly(clinic):
    assert not await clinic.verify("taylor smith", TAYLOR_DOB)


async def test_every_clinic_day_has_morning_and_afternoon_openings(clinic):
    for day in clinic.clinic_days():
        assert await clinic.check_availability(day, day, "morning"), day
        assert await clinic.check_availability(day, day, "afternoon"), day


async def test_availability_skips_weekends_and_is_published(clinic, published):
    openings = await clinic.check_availability(date(2026, 10, 2), date(2026, 10, 6))
    assert {o.date() for o in openings} <= {
        date(2026, 10, 2),
        date(2026, 10, 5),
        date(2026, 10, 6),
    }
    assert published["search"]["openings"] == [o.isoformat() for o in openings]


async def test_new_patient_books_an_open_slot(clinic, published):
    await clinic.register("Casey Morgan", date(1995, 6, 1))
    [slot, *_] = await clinic.check_availability(date(2026, 10, 5), date(2026, 10, 5))

    appointment = await clinic.book("Casey Morgan", slot, "Sore throat")

    assert appointment.patient == "Casey Morgan"
    assert published["caller"]["new"] is True
    assert published["appointments"][-1]["startsAt"] == slot.isoformat()
    assert not clinic.is_open(slot)


async def test_reschedule_moves_the_appointment_and_frees_the_old_time(clinic):
    [appointment] = clinic.upcoming("Taylor Smith")
    old = appointment.starts_at
    [slot, *_] = await clinic.check_availability(
        date(2026, 10, 5), date(2026, 10, 5), "morning"
    )

    await clinic.reschedule("Taylor Smith", appointment.id, slot)

    assert clinic.upcoming("Taylor Smith")[0].starts_at == slot
    assert all(a.starts_at != old for a in clinic.appointments)


async def test_cannot_book_a_taken_slot(clinic):
    jordan = clinic.upcoming("Jordan Lee")[0].starts_at
    with pytest.raises(ClinicError, match="isn't open"):
        await clinic.book("Taylor Smith", jordan, "Checkup")


async def test_cannot_reschedule_someone_elses_appointment(clinic):
    jordan = clinic.upcoming("Jordan Lee")[0]
    [slot, *_] = await clinic.check_availability(date(2026, 10, 5), date(2026, 10, 5))
    with pytest.raises(ClinicError, match="no appointment"):
        await clinic.reschedule("Taylor Smith", jordan.id, slot)


async def test_nothing_is_offered_past_the_fifth_clinic_day(clinic):
    assert clinic.clinic_days()[-1] == date(2026, 10, 5)
    assert await clinic.check_availability(date(2026, 10, 6), date(2026, 10, 9)) == []
