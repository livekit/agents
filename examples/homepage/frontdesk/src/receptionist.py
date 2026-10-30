from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time

from livekit.agents import Agent, RunContext, function_tool, inference
from livekit.agents.llm import ToolError

from clinic import (
    CLINIC_NAME,
    CLINIC_TZ,
    CLINIC_TZ_NAME,
    DOCTOR,
    SCHEDULE,
    Appointment,
    Clinic,
    ClinicError,
    PartOfDay,
)

GREETING = (
    f"Thanks for calling {CLINIC_NAME}, this is Riley. What can I help you with today?"
)


VOICE_MODEL = "gpt-live-1"
BACKEND_MODEL = "gpt-5.6-luna"

NEW_PATIENT = ("Alex Morgan", date(1992, 3, 14))


@dataclass(frozen=True)
class Caller:
    name: str
    date_of_birth: date


@dataclass
class CallState:
    caller: Caller | None = None
    birthday: date | None = None
    charts: list[str] = field(default_factory=list)
    clinic: Clinic | None = None


def verified_caller(context: RunContext[CallState]) -> Caller:
    caller = context.userdata.caller
    if caller is None:
        raise ToolError(
            "The caller isn't verified. Verify their name and date of birth first."
        )
    return caller


class FrontDesk(Agent):
    """What every call at the front desk can do with the chart: find open times and move a visit."""

    def __init__(
        self, clinic: Clinic, *, instructions: str, backend_instructions: str
    ) -> None:
        self._clinic = clinic
        super().__init__(
            instructions=instructions,
            llm=inference.DuplexModel(
                model=f"openai/{VOICE_MODEL}",
                voice="ripple",
                extra_kwargs={
                    "responses_options": {
                        "model": BACKEND_MODEL,
                        "instructions": backend_instructions,
                    },
                },
            ),
        )

    @function_tool
    async def check_availability(
        self,
        context: RunContext[CallState],
        earliest_date: str,
        latest_date: str,
        part_of_day: PartOfDay = "any",
    ) -> str:
        """See which appointment times are open on the calendar.

        Args:
            earliest_date: First day to check, YYYY-MM-DD.
            latest_date: Last day to check, YYYY-MM-DD.
            part_of_day: "morning", "afternoon", or "any".
        """
        verified_caller(context)
        try:
            openings = await self._clinic.check_availability(
                _date(earliest_date), _date(latest_date), part_of_day
            )
        except ClinicError as e:
            raise ToolError(str(e)) from e
        return json.dumps(
            [
                {"starts_at": f"{o:%Y-%m-%dT%H:%M}", "spoken": _spoken(o)}
                for o in openings
            ]
        )

    @function_tool
    async def reschedule_appointment(
        self, context: RunContext[CallState], appointment_id: str, starts_at: str
    ) -> str:
        """Move one of the caller's existing appointments to an open time.

        Args:
            appointment_id: The appointment to move.
            starts_at: The opening's starts_at from check_availability.
        """
        try:
            appointment = await self._clinic.reschedule(
                verified_caller(context).name, appointment_id, _datetime(starts_at)
            )
        except ClinicError as e:
            raise ToolError(str(e)) from e
        return f"Moved {appointment.reason} to {_spoken(appointment.starts_at)}."


class Receptionist(FrontDesk):
    """Riley answering the phone: existing patients, new ones, booking and moving visits."""

    def __init__(self, clinic: Clinic) -> None:
        super().__init__(
            clinic,
            instructions=f"""
You are Riley, the front desk at {CLINIC_NAME}, {DOCTOR}'s practice, answering the phone. {clock()}
Keep the call brisk and warm. One or two short spoken sentences at a time, one question at a time.
Say times the way people do, like "nine thirty Tuesday morning", and dates too, like "March fourteenth, nineteen eighty-eight".

Delegate anything that needs the clinic's records: looking up a caller, registering a new patient, checking open times, booking, or moving an appointment. While that runs, say a few words like "one sec" and wait for the result.
Never state an appointment, a time, or a booking that the delegated work didn't return.
Before anything can be looked up or booked, find the caller's chart the way a front desk does: date of birth first, then name. Ask for whichever of the two they haven't given, each only once. As soon as you have both, delegate looking them up by that date of birth and the name as you heard it. While that runs you don't know yet whether they have a chart, so never say you're pulling them up or setting them up; "Okay ... one sec" is enough.
- If it matched their name to a chart, spell that name back the way the delegated work gave it, as a question, like "Okay, Taylor Smith, that's T A Y L O R, S M I T H, right?", and once they say yes, delegate opening their chart. Then tell them the upcoming appointment it returned, what it's for and the day and time, before anything else.
- If no chart matched but they've said they've been in before, read their whole date of birth back once to check it, and if they correct it, delegate looking them up again with the new date. If that finds nothing either, get them set up.
- Otherwise, say you'll get them set up, without saying you couldn't find them, then spell their name back the way you'd write it and ask if that's right. If they correct a letter, delegate looking them up again with their spelling; if that finds their chart, go on as if it had matched. Only after they say yes and nothing has matched, delegate registering them with that spelling.
If you didn't catch all of the date of birth, ask for the whole date again, never just part of it like the last two digits of the year.
If they ask you to say or spell their name, just do it.
When you confirm a booking, say the day, date, and time exactly as the delegated work returned them.
While you're waiting for the delegation to return, keep talking the way a person at a front desk does while they look something up. They say things like 'Okay ... checking Tuesday ... okay, there's one', not 'Let me check that for you'. Never talk about typing or a keyboard.

When they say they're done, say a short goodbye and delegate ending the call.
Saying goodbye and delegating ending the call always go together, in the same turn: never say goodbye without delegating the end of the call, and don't say goodbye until they've had a chance to say they're done.
""".strip(),
            backend_instructions=f"""
You do the record-keeping for {CLINIC_NAME}, {DOCTOR}'s practice, on behalf of the receptionist on the phone.
{calendar_context(clinic)}

Find every caller's chart before anything else, the way a front desk does: call look_up_birthday with their date of birth and the name they gave. It lists only the chart with that date of birth and that name, spelled as given, so call it again with any corrected spelling the receptionist passes on.
- If it lists a chart, report that chart's letters for the receptionist to spell back, and nothing about any other chart. Once the caller says the spelling is right, call open_chart with that chart's name and report their upcoming appointment.
- If it lists none, report only that no chart matched that spelling.
No other record can be read or changed until their chart is open or they're registered.
Call register_new_patient only once the caller has confirmed the spelling and look_up_birthday has found no chart for it, with the letters they confirmed.
Never reveal a chart that doesn't match the caller's name, or any date of birth on file.
To find a time, call check_availability for what the caller asked, then report the one or two openings closest to it.
When the caller has picked a time, book it with book_appointment, or reschedule_appointment for an existing appointment, without asking again.
When asked to end the call, call end_call.
Hand back short, plain results the receptionist can say out loud. Only report times and appointments that a tool returned.
""".strip(),
        )

    @function_tool
    async def look_up_birthday(
        self, context: RunContext[CallState], date_of_birth: str, name: str
    ) -> str:
        """List the chart with the caller's date of birth and name, spelled as given. The first step for every caller, and again with any corrected spelling.

        Args:
            date_of_birth: The caller's date of birth, YYYY-MM-DD.
            name: The caller's full name as they said it, or as they spelled it.
        """
        born = _date(date_of_birth)
        context.userdata.birthday = born
        context.userdata.charts = [
            chart
            for chart in self._clinic.born_on(born)
            if _letters(chart) == _letters(name)
        ]
        return json.dumps(
            {
                "charts": [
                    {"name": chart, "letters": _read_back(chart)}
                    for chart in context.userdata.charts
                ]
            }
        )

    @function_tool
    async def open_chart(self, context: RunContext[CallState], name: str) -> str:
        """Open the chart that matched the caller's name, once they've said its spelling is right. Required before any other record is read or changed.

        Args:
            name: The chart's name exactly as look_up_birthday listed it.
        """
        born = context.userdata.birthday
        if born is None:
            raise ToolError("Look up the caller's date of birth first.")
        if name not in context.userdata.charts or not await self._clinic.verify(
            name, born
        ):
            raise ToolError(
                f"No chart named {name!r} matched that date of birth. Use a name "
                "look_up_birthday listed."
            )
        context.userdata.caller = Caller(name, born)
        return json.dumps(
            {"upcoming": [_describe(a) for a in self._clinic.upcoming(name)]}
        )

    @function_tool
    async def register_new_patient(
        self, context: RunContext[CallState], first_name: str, last_name: str
    ) -> str:
        """Create a chart for a caller who isn't in the records, under the spelling they confirmed and the date of birth look_up_birthday checked.

        Args:
            first_name: The letters of the first name the caller confirmed, separated by spaces, like "S H A Y N E".
            last_name: The letters of the last name the caller confirmed, separated by spaces. A hyphen or apostrophe that is part of the name stays in place, like "O ' B R I E N".
        """
        born = context.userdata.birthday
        if born is None:
            raise ToolError(
                "Look up the caller's date of birth before registering them."
            )
        name = f"{_from_letters(first_name)} {_from_letters(last_name)}".strip()
        if " " not in name:
            raise ToolError(
                "Both a first and a last name are needed, letter by letter."
            )
        try:
            await self._clinic.register(name, born)
        except ClinicError as e:
            raise ToolError(str(e)) from e
        context.userdata.caller = Caller(name, born)
        return f"Registered and verified {name}."

    @function_tool
    async def book_appointment(
        self, context: RunContext[CallState], starts_at: str, reason: str
    ) -> str:
        """Book a new appointment for the caller at an open time.

        Args:
            starts_at: The opening's starts_at from check_availability.
            reason: What the visit is for, in a few words.
        """
        try:
            appointment = await self._clinic.book(
                verified_caller(context).name, _datetime(starts_at), reason
            )
        except ClinicError as e:
            raise ToolError(str(e)) from e
        return f"Booked {appointment.reason} for {_spoken(appointment.starts_at)}."


class ConfirmationCall(FrontDesk):
    """Riley calling a patient to confirm their upcoming visit and take anything for the doctor."""

    def __init__(self, clinic: Clinic, appointment: Appointment) -> None:
        self._appointment = appointment
        name = appointment.patient
        first = name.split()[0]
        super().__init__(
            clinic,
            instructions=f"""
You are Riley, from the front desk at {CLINIC_NAME}, {DOCTOR}'s practice. You placed this call to {name}, a patient, to confirm their upcoming appointment. {clock()}
Keep the call brisk and warm. One or two short spoken sentences at a time, one question at a time.
Say times the way people do, like "two thirty Tuesday afternoon".

You've already asked whether you're speaking with {first}. If it's someone else, or not a good time, say you'll call back another time, say goodbye, and delegate ending the call. Never mention the appointment or anything medical to anyone but the patient.
Before anything else, verify them. This is a LiveKit demo, so ask for their date of birth with a quick demo note, something like: "Okay great! Usually this is where I'd ask for your date of birth to verify it's you, but since this is a demo, any date will do." Delegate checking the date they give. If you didn't catch all of it, ask for the whole date again, never just part of it.
Once verified, tell them the appointment as the check returned it, what it's for and the day and time, and ask if they can still make it.
- If they can, delegate confirming it.
- If they can't, ask what day and time would work better, delegate finding openings, and offer one or two. Once they pick, delegate moving it and confirming it.
Then ask if there's anything they'd like {DOCTOR} to know before they come in. If they mention something, delegate adding it to the appointment as a note for the doctor, in their words, and say briefly that you've passed it along. If there's nothing, that's fine.
While you're waiting for the delegation to return, keep talking the way a person at a front desk does while they look something up: 'Okay ... just pulling up your chart ... there it is'. Never talk about typing or a keyboard. Never say something is done unless the delegated work returned it.
Never state an appointment or a time that the delegated work didn't return.
When they're all set, remind them of the day and time and ask if there's anything else. Once they say that's all, say a short goodbye and delegate ending the call.
Saying goodbye and delegating ending the call always go together, in the same turn: never say goodbye without delegating the end of the call, and don't say goodbye until they've had a chance to say they're done.
""".strip(),
            backend_instructions=f"""
You do the record-keeping for {CLINIC_NAME}, {DOCTOR}'s practice, on behalf of the receptionist, who called the patient {name} to confirm their appointment {appointment.id}.
{calendar_context(clinic)}

Verify the patient with verify_patient and the date of birth they gave, before anything else. Report only whether it matched, and their appointment once it does. Never reveal a date of birth on file.
Call confirm_appointment once the patient says they can make it, or right after moving it.
To move it, call check_availability for what they asked and report the one or two openings closest to it; once they pick, call reschedule_appointment, then confirm_appointment.
Call add_note_for_doctor with anything the patient wants the doctor to know before the visit, in their words.
When asked to end the call, call end_call.
Hand back short, plain results the receptionist can say out loud. Only report times and appointments that a tool returned.
""".strip(),
        )

    @function_tool
    async def verify_patient(
        self, context: RunContext[CallState], date_of_birth: str
    ) -> str:
        """Verify the patient. Any date of birth passes in the demo. Required before any other record is read or changed.

        Args:
            date_of_birth: The date of birth the patient gave, YYYY-MM-DD.
        """
        name = self._appointment.patient
        context.userdata.caller = Caller(name, _date(date_of_birth))
        return json.dumps(
            {
                "verified": True,
                "upcoming": [_describe(a) for a in self._clinic.upcoming(name)],
            }
        )

    @function_tool
    async def confirm_appointment(
        self, context: RunContext[CallState], appointment_id: str
    ) -> str:
        """Mark the patient's appointment as confirmed: they've said they'll be there.

        Args:
            appointment_id: The appointment to confirm.
        """
        try:
            appointment = await self._clinic.confirm(
                verified_caller(context).name, appointment_id
            )
        except ClinicError as e:
            raise ToolError(str(e)) from e
        return f"Confirmed {appointment.reason} for {_spoken(appointment.starts_at)}."

    @function_tool
    async def add_note_for_doctor(
        self, context: RunContext[CallState], appointment_id: str, note: str
    ) -> str:
        """Add something the patient wants the doctor to know before the visit to their appointment.

        Args:
            appointment_id: The appointment the note is for.
            note: What the patient said, in their own words, as one or two short sentences.
        """
        try:
            await self._clinic.add_note(
                verified_caller(context).name, appointment_id, note
            )
        except ClinicError as e:
            raise ToolError(str(e)) from e
        return f"Added for {DOCTOR}: {note.strip()}"


def clock() -> str:
    now = datetime.now(CLINIC_TZ)
    return f"It's {now:%A, %B} {now.day}, {now:%-I:%M %p} {CLINIC_TZ_NAME}."


def calendar_context(clinic: Clinic) -> str:
    days = ", ".join(
        f"{d:%a %b} {d.day} = {d.isoformat()}" for d in clinic.clinic_days()
    )
    return (
        f"{clock()} Clinic days that can be booked: {days}. Nothing can be booked after "
        "the last one; if the caller wants later, check the latest days instead."
    )


def outbound_greeting(appointment: Appointment) -> str:
    return (
        f"Hi, this is Riley from {CLINIC_NAME}, calling for {appointment.patient} about an "
        f"upcoming appointment with {DOCTOR}. Is this {appointment.patient.split()[0]}?"
    )


def caller_script(clinic: Clinic, state: CallState) -> str:
    """What the caller knows and wants right now, for the suggested replies."""
    days = clinic.clinic_days()
    you = clinic.appointments[0]
    born = clinic.patients[you.patient].strftime("%B %-d, %Y")
    target = days[3]
    seeded = SCHEDULE["appointments"][0]
    was_at = datetime.combine(
        days[seeded["day"]], time.fromisoformat(seeded["time"]), tzinfo=CLINIC_TZ
    )
    new_name, new_dob = NEW_PATIENT
    on_file = f"""
The caller is {you.patient}, an existing patient at {CLINIC_NAME}, trying out Riley's front desk. Their details on file, the only ones to use:
- Full name: {you.patient}, spelled "{_read_back(you.patient)}"
- Date of birth: {born}
- Their appointment: {you.reason}, {_spoken(you.starts_at)}
They want to move that appointment to {target:%A} morning.
If the caller has instead said they're a new patient, they are {new_name}, born {new_dob:%B %-d, %Y}, spelled "{_read_back(new_name)}", and they want a check-up {days[2]:%A} afternoon.
""".strip()
    if you.starts_at != was_at:
        stage = "The appointment is moved. Say thanks, that's everything, and goodbye."
    elif state.caller is None:
        stage = (
            "They haven't been verified yet. If Riley asks what they need, say who "
            "they are, their date of birth, and that they'd like to move their "
            f"appointment to {target:%A} morning, all in one line. If Riley asks for "
            "the date of birth or the name, give just that. If Riley spells the name "
            "back right, say yes; if not, spell it."
        )
    else:
        stage = (
            "They're verified. When Riley offers times, take the first one, in "
            "Riley's words. When Riley confirms the move, say thanks and goodbye."
        )
    return f"{on_file}\n{stage}"


DOCTOR_NOTE = (
    "my knee has been swelling after I run, and I've been taking ibuprofen for it"
)


def outbound_caller_script(
    clinic: Clinic, state: CallState, appointment: Appointment
) -> str:
    """What the patient Riley called knows and wants right now, for the suggested replies."""
    born = clinic.patients[appointment.patient].strftime("%B %-d, %Y")
    on_file = f"""
Riley from {CLINIC_NAME} called the caller, {appointment.patient}, to confirm their appointment. Their details, the only ones to use:
- Full name: {appointment.patient}
- Date of birth: {born}
- Their appointment: {appointment.reason}, {_spoken(appointment.starts_at)}
They can make it, and want {DOCTOR} to know: "{DOCTOR_NOTE}".
""".strip()
    if appointment.notes:
        stage = (
            "Everything's sorted. When Riley asks if there's anything else, "
            "say that's all, thanks, and goodbye."
        )
    elif appointment.confirmed:
        stage = f"When Riley asks if there's anything for {DOCTOR}, say: {DOCTOR_NOTE}."
    elif state.caller is not None:
        stage = "When Riley asks if they can still make it, say yes."
    else:
        stage = (
            f"They haven't been verified yet. Confirm they're "
            f"{appointment.patient.split()[0]}, and give the date of birth when Riley asks."
        )
    return f"{on_file}\n{stage}"


def _describe(appointment: Appointment) -> dict[str, str]:
    return {
        "appointment_id": appointment.id,
        "when": _spoken(appointment.starts_at),
        "reason": appointment.reason,
    }


def _spoken(starts_at: datetime) -> str:
    return f"{starts_at:%A %B} {starts_at.day}, {starts_at:%-I:%M %p}"


def _from_letters(spelled: str) -> str:
    out: list[str] = []
    capitalize = True
    for ch in "".join(spelled.split()):
        out.append(ch.upper() if capitalize else ch.lower())
        capitalize = ch in "-'"
    return "".join(out)


def _letters(name: str) -> str:
    return "".join(c for c in name if c.isalpha()).casefold()


def _read_back(name: str) -> str:
    spoken = {"-": "hyphen", "'": "apostrophe"}
    return ", ".join(
        " ".join(spoken.get(ch, ch.upper()) for ch in part) for part in name.split()
    )


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise ToolError(f"Use YYYY-MM-DD, got {value!r}.") from e


def _datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=CLINIC_TZ)
    except ValueError as e:
        raise ToolError(
            f"Use the starts_at value from check_availability, got {value!r}."
        ) from e
