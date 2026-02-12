import random
from datetime import date, datetime, time, timedelta


class FakeDatabase:
    def __init__(self):
        self._patient_records = [
            {
                "name": "Mary Jane",
                "date_of_birth": date(2001, 6, 10),
                "phone_number": "18005882300",
                "insurance": "Anthem",
                "outstanding_balance": round(random.uniform(20, 3000), 2),
            },
            {
                "name": "Peter Parker",
                "date_of_birth": date(2001, 8, 10),
                "phone_number": "18005882300",
                "insurance": "Aetna",
                "outstanding_balance": round(random.uniform(20, 3000), 2),
            }
        ]
        today = date.today()
        self._doctor_records = [
            {
                "name": "Dr. Henry Jekyll",
                "accepted_insurances": ["Anthem", "HealthFirst"],
                "availability": [
                    {"date": today + timedelta(days=2), "time": time(9, 30)},
                    {"date": today + timedelta(days=4), "time": time(14, 30)},
                    {"date": today + timedelta(days=7), "time": time(11, 0)},
                ],
            },
            {
                "name": "Dr. Edward Hyde",
                "accepted_insurances": ["Anthem", "Aetna", "EmblemHealth"],
                "availability": [
                    {"date": today + timedelta(days=1), "time": time(10, 0)},
                    {"date": today + timedelta(days=3), "time": time(14, 30)},
                    {"date": today + timedelta(days=5), "time": time(15, 45)},
                ],
            },
        ]

    @property
    def patient_records(self) -> list:
        return self._patient_records

    @property
    def doctor_records(self) -> list:
        return self._doctor_records

    def get_patient_by_name(self, name: str) -> dict | None:
        return next(
            (record for record in self._patient_records if record["name"] == name),
            None,
        )

    def get_doctor_by_name(self, name: str) -> dict | None:
        return next(
            (record for record in self._doctor_records if record["name"] == name),
            None,
        )

    def get_compatible_doctors(self, insurance: str) -> list:
        return [
            doctor for doctor in self._doctor_records if insurance in doctor["accepted_insurances"]
        ]

    def update_patient_record(self, name: str, **fields) -> bool:
        record = self.get_patient_by_name(name)
        if record is None:
            return False
        record.update(fields)
        return True

    def add_appointment(self, name: str, appointment: dict) -> bool:
        record = self.get_patient_by_name(name)
        if record is None:
            return False
        record.setdefault("appointments", []).append(appointment)
        appt_time = appointment["appointment_time"]
        if isinstance(appt_time, str):
            appt_time = datetime.fromisoformat(appt_time)
        self.remove_doctor_availability(
            appointment["doctor_name"],
            {
                "date": appt_time.date(),
                "time": appt_time.time(),
            },
        )
        return True

    def cancel_appointment(self, name: str, appointment: dict) -> bool:
        record = self.get_patient_by_name(name)
        if record is None or "appointments" not in record:
            return False
        try:
            record["appointments"].remove(appointment)
        except ValueError:
            return False
        doctor = self.get_doctor_by_name(appointment["doctor_name"])
        if doctor is not None:
            appt_time = appointment["appointment_time"]
            if isinstance(appt_time, str):
                appt_time = datetime.fromisoformat(appt_time)
            doctor["availability"].append(
                {
                    "date": appt_time.date(),
                    "time": appt_time.time(),
                }
            )
        return True

    def add_patient_record(self, info: dict) -> None:
        info.setdefault("outstanding_balance", round(random.uniform(20, 3000), 2))
        self._patient_records.append(info)

    def get_outstanding_balance(self, name: str) -> float | None:
        record = self.get_patient_by_name(name)
        if record is None:
            return None
        return record.get("outstanding_balance", 0.0)

    def apply_payment(self, name: str, amount: float) -> float | None:
        record = self.get_patient_by_name(name)
        if record is None:
            return None
        record["outstanding_balance"] = round(record.get("outstanding_balance", 0.0) - amount, 2)
        return record["outstanding_balance"]

    def remove_doctor_availability(self, doctor_name: str, appointment_time: dict) -> None:
        for doctor in self._doctor_records:
            if doctor["name"] == doctor_name:
                doctor["availability"] = [
                    slot
                    for slot in doctor["availability"]
                    if not (
                        slot["date"] == appointment_time["date"]
                        and slot["time"] == appointment_time["time"]
                    )
                ]
