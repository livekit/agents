import os
from dataclasses import dataclass
import logging
import datetime
import sys
from typing import Literal
from zoneinfo import ZoneInfo


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from livekit.agents import (
    Agent,
    workflows,
    function_tool,
    JobContext,
    cli,
    WorkerOptions,
    RunContext,
    AgentSession,
    ToolError,
    workflows,
)


from dotenv import load_dotenv
from calendar_api import CalComCalendar, Calendar, FakeCalendar, AvailableSlot
from livekit.plugins import deepgram, openai, silero, cartesia
from livekit.plugins.turn_detector.multilingual import MultilingualModel


load_dotenv()


TIMEZONE = "UTC"


@dataclass
class Userdata:
    cal: Calendar


logger = logging.getLogger("front-desk")


class FrontDeskAgent(Agent):
    def __init__(self, *, timezone: str) -> None:
        self.tz = ZoneInfo(timezone)
        today = datetime.datetime.now(self.tz).strftime("%A, %B %d, %Y")

        super().__init__(
            instructions=(
                f"You are Front-Desk, a helpful and efficient voice assistant. "
                f"Today is {today}. Your main goal is to schedule an appointment for the user. "
                "This is a voice conversation — speak naturally, clearly, and concisely. "
                "When the user says hello or greets you, don’t just respond with a greeting — use it as an opportunity to move things forward. "
                "For example, follow up with a helpful question like: 'Would you like to book a time?' "
                "When asked for availability, call list_available_slots and offer a few clear, simple options. "
                "Say things like 'Monday at 2 PM' — avoid timezones, timestamps, and avoid saying 'AM' or 'PM'. "
                "Use natural phrases like 'in the morning' or 'in the evening', and don’t mention the year unless it’s different from the current one. "
                "Offer a few options at a time, pause for a response, then guide the user to confirm. "
                "If the time is no longer available, let them know gently and offer the next options. "
                "Always keep the conversation flowing — be proactive, human, and focused on helping the user schedule with ease."
            )
        )

        self._slots_map: dict[str, AvailableSlot] = {}

    @function_tool
    async def schedule_appointment(
        self,
        ctx: RunContext[Userdata],
        slot_id: str,
    ) -> str:
        """
        Schedule an appointment at the given slot.

        Args:
            slot_id: The identifier for the selected time slot (as shown in the list of available slots).
        """
        if not (slot := self._slots_map.get(slot_id)):
            raise ToolError(f"error: slot {slot_id} was not found")

        email_result = await workflows.GetEmailAgent(chat_ctx=self.chat_ctx)

        await ctx.userdata.cal.schedule_appointment(
            start_time=slot.start_time, attendee_email=email_result.email_address
        )

        local = slot.start_time.astimezone(self.tz)
        return f"The appointment was successfully scheduled for {local.strftime('%A, %B %d, %Y at %H:%M %Z')}."

    @function_tool
    async def list_available_slots(
        self,
        ctx: RunContext[Userdata],
        range: Literal["+2week", "+1month", "+3month", "default"],
    ) -> str:
        """
        Return a plain-text list of available slots, one per line.

        <slot_id> – <Weekday>, <Month> <Day>, <Year> at <HH:MM> <TZ> (<relative time>)

        You must infer the appropriate ``range`` implicitly from the
        conversational context and **must not** prompt the user to pick a value
        explicitly.

        Args:
            range: Determines how far ahead to search for free time slots.
        """
        now = datetime.datetime.now(self.tz)
        lines: list[str] = []

        if range == "+2week" or range == "default":
            range_days = 14
        elif range == "+1month":
            range_days = 30
        elif range == "+3month":
            range_days = 90

        for slot in await ctx.userdata.cal.list_available_slots(
            start_time=now, end_time=now + datetime.timedelta(days=range_days)
        ):
            local = slot.start_time.astimezone(self.tz)
            delta = local - now
            days = delta.days
            seconds = delta.seconds

            if days == 0 and seconds < 3600:
                rel = "in less than an hour"
            elif days == 0:
                rel = "later today"
            elif days == 1:
                rel = "in 1 day"
            elif days < 7:
                rel = f"in {days} days"
            elif days < 14:
                rel = "in 1 week"
            else:
                rel = f"in {days // 7} weeks"

            lines.append(
                f"{slot.unique_hash} – {local.strftime('%A, %B %d, %Y')} at "
                f"{local:%H:%M} {local.tzname()} ({rel})"
            )
            self._slots_map[slot.unique_hash] = slot

        return "\n".join(lines) or "No slots available at the moment."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    if cal_api_key := os.getenv("CAL_API_KEY", None):
        logger.info("CAL_API_KEY detected, using cal.com calendar")
        cal = CalComCalendar(api_key=cal_api_key, timezone=TIMEZONE)
    else:
        logger.warning(
            "CAL_API_KEY is not set. Falling back to FakeCalendar; set CAL_API_KEY to enable Cal.com integration."
        )
        cal = FakeCalendar(timezone=TIMEZONE)

    await cal.initialize()

    session = AgentSession[Userdata](
        userdata=Userdata(cal=cal),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o", parallel_tool_calls=False, temperature=0.45),
        tts=cartesia.TTS(voice="39b376fc-488e-4d0c-8b37-e00b72059fdd", speed="fast"),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        max_tool_steps=1,
    )

    await session.start(agent=FrontDeskAgent(timezone=TIMEZONE), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
