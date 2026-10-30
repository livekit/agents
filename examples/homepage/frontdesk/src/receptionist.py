from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time

from livekit.agents import Agent, RunContext, function_tool
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
from conversation import caller_spoke_last, heard

GREETING = (
    f"Thanks for calling {CLINIC_NAME}, this is Riley. What can I help you with today?"
)


VOICE_RULES = """\
You're on a phone call, so talk like a person, not a document:
- One or two short sentences at a time, and one question at a time. No lists, markdown, or symbols.
- Say dates and times the way people do, like "nine thirty Tuesday morning" or "March fourteenth, nineteen eighty-eight". Never read out ids.
- Never say you've done something, or that you're about to, without calling its tool in that same reply.
- Only state an appointment or a time that a tool gave you, with the day and time just as it gave them.
- When a tool refuses, fix what it says with your tools. Tell the caller only what changes for them.
- When they're done, say a short goodbye and call end_call in that same reply. Never say goodbye before they've had a chance to say they're done."""

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

    def __init__(self, clinic: Clinic, *, instructions: str) -> None:
        self._clinic = clinic
        self._must_hear: dict[str, str] = {}
        super().__init__(instructions=instructions)

    def _must_tell(self, context: RunContext[CallState], what: str) -> None:
        self._must_hear[what] = context.function_call.call_id

    @function_tool
    async def end_call(self, context: RunContext[CallState]) -> None:
        """Hang up. Only once the caller has nothing else, and after a short goodbye in this
        same reply."""
        await context.wait_for_playout()
        items = context.session.history.items
        if unheard := [
            w for w, token in self._must_hear.items() if not heard(items, token)
        ]:
            self._must_hear[unheard[0]] = context.function_call.call_id
            raise ToolError(f"Don't hang up yet: first tell them {unheard[0]}.")
        if caller_spoke_last(items):
            raise ToolError(
                "Not hung up yet. Answer what they just said with a short goodbye, and call "
                "end_call again in that same reply."
            )
        context.speech_handle.add_done_callback(lambda _: context.session.shutdown())

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
        self._must_tell(context, "the day and time it's moved to")
        return f"Moved {appointment.reason} to {_spoken(appointment.starts_at)}."


class Receptionist(FrontDesk):
    """Riley answering the phone: existing patients, new ones, booking and moving visits."""

    def __init__(self, clinic: Clinic) -> None:
        super().__init__(
            clinic,
            instructions=f"""
You are Riley, the front desk at {CLINIC_NAME}, {DOCTOR}'s practice, answering the phone. Brisk and warm.
{calendar_context(clinic)}

{VOICE_RULES}

Find the caller's chart before anything else, the way a front desk does: date of birth first, then name. Ask for whichever they haven't given, each only once. If you didn't catch all of the date of birth, ask for the whole date again, never just part of it like the last two digits of the year. As soon as you have both, call look_up_birthday with the date and the name as you heard it. Until it returns you don't know whether they have a chart, so never say you're pulling them up or setting them up.
- It lists their chart: spell that chart's name back from its letters, as a question, like "Taylor Smith, that's T A Y L O R, S M I T H, right?". Once they say yes, call open_chart, then tell them their upcoming appointment, what it's for and the day and time, before anything else.
- It lists none, and they've said they've been in before: read their whole date of birth back once to check it. If they correct it, call look_up_birthday again with the new date. If that finds nothing either, get them set up.
- It lists none otherwise: say you'll get them set up, without saying you couldn't find them, and spell their name back the way you'd write it, as a question. If they correct a letter, call look_up_birthday again with their spelling, and if that lists their chart, go on as if it had matched. Once they say the spelling is right and nothing matched, call register_new_patient with those letters.
Never mention a chart other than theirs. If they ask you to say or spell their name, just do it.

To find a time, call check_availability for what they asked, and offer the one or two openings closest to it. If it has nothing then, check the next clinic days for the same part of the day and offer those, before asking them to pick another time. Once they pick one, call book_appointment, or reschedule_appointment for an appointment they already have, without asking again.
""".strip(),
        )

    async def on_enter(self) -> None:
        self.session.say(GREETING)

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
        self._must_tell(context, "the day and time it's booked for")
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
You are Riley, from the front desk at {CLINIC_NAME}, {DOCTOR}'s practice. You placed this call to {name}, a patient, to confirm their upcoming appointment. Brisk and warm.
{calendar_context(clinic)}

{VOICE_RULES}

You've already asked whether you're speaking with {first}. If it's someone else, or not a good time, say you'll call back another time, then say goodbye and call end_call. Never mention the appointment or anything medical to anyone but the patient.
Once they say it's them, verify them before anything else. This is a LiveKit demo, so ask for their date of birth with a quick demo note, something like: "Okay great! Usually this is where I'd ask for your date of birth to verify it's you, but since this is a demo, any date will do." Then call verify_patient with the date they give. If you didn't catch all of it, ask for the whole date again, never just part of it.
Then tell them the appointment it returned, what it's for and the day and time, and ask if they can still make it.
- If they can, call confirm_appointment.
- If they can't, ask what day and time would work better, call check_availability, and offer one or two openings. Once they pick one, call reschedule_appointment, then confirm_appointment.
Then ask if there's anything they'd like {DOCTOR} to know before they come in. If they mention something, call add_note_for_doctor with it in their words, and say briefly that you've passed it along. If there's nothing, that's fine.
When they're all set, remind them of the day and time and ask if there's anything else.
""".strip(),
        )

    async def on_enter(self) -> None:
        self.session.say(outbound_greeting(self._appointment))

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
        await self._clinic.publish_caller(name)
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
        self._must_tell(context, "that it's confirmed")
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
