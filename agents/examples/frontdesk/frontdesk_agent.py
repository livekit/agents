from __future__ import annotations

import datetime
import logging
import os
import sys
from dataclasses import dataclass
from typing import Literal
from zoneinfo import ZoneInfo
from agents.extensions.interrupt_handler import InterruptHandler

# ensure relative imports resolve correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from calendar_api import AvailableSlot, CalComCalendar, Calendar, FakeCalendar, SlotUnavailableError
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    ToolError,
    WorkerOptions,
    beta,
    cli,
    function_tool,
)
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


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
        self, ctx: RunContext[Userdata], slot_id: str,
    ) -> str | None:
        """Schedule an appointment at the given slot."""
        if not (slot := self._slots_map.get(slot_id)):
            raise ToolError(f"error: slot {slot_id} was not found")

        email_result = await beta.workflows.GetEmailTask(chat_ctx=self.chat_ctx)

        if ctx.speech_handle.interrupted:
            return

        ctx.disallow_interruptions()

        try:
            await ctx.userdata.cal.schedule_appointment(
                start_time=slot.start_time, attendee_email=email_result.email_address
            )
        except SlotUnavailableError:
            raise ToolError("This slot isn't available anymore") from None

        local = slot.start_time.astimezone(self.tz)
        return f"The appointment was successfully scheduled for {local.strftime('%A, %B %d, %Y at %H:%M %Z')}."

    @function_tool
    async def list_available_slots(
        self, ctx: RunContext[Userdata], range: Literal["+2week", "+1month", "+3month", "default"],
    ) -> str:
        """Return a plain-text list of available slots, one per line."""
        now = datetime.datetime.now(self.tz)
        lines: list[str] = []

        if range == "+2week" or range == "default":
            range_days = 14
        elif range == "+1month":
            range_days = 30
        elif range == "+3month":
            range_days = 90
        else:
            range_days = 14

        for slot in await ctx.userdata.cal.list_available_slots(
            start_time=now, end_time=now + datetime.timedelta(days=range_days)
        ):
            local = slot.start_time.astimezone(self.tz)
            delta = local - now
            days = delta.days
            seconds = delta.seconds

            if local.date() == now.date():
                if seconds < 3600:
                    rel = "in less than an hour"
                else:
                    rel = "later today"
            elif local.date() == (now.date() + datetime.timedelta(days=1)):
                rel = "tomorrow"
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

    # --- Initialize interrupt handler ---
    interrupt_handler = InterruptHandler()

    timezone = "utc"

    # --- Calendar setup ---
    if cal_api_key := os.getenv("CAL_API_KEY", None):
        logger.info("CAL_API_KEY detected, using Cal.com calendar")
        cal = CalComCalendar(api_key=cal_api_key, timezone=timezone)
    else:
        logger.warning(
            "CAL_API_KEY is not set. Falling back to FakeCalendar; set CAL_API_KEY to enable Cal.com integration."
        )
        cal = FakeCalendar(timezone=timezone)

    await cal.initialize()

    # --- Create Agent Session ---
    session = AgentSession[Userdata](
        userdata=Userdata(cal=cal),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o", parallel_tool_calls=False, temperature=0.45),
        tts=cartesia.TTS(
            voice="39b376fc-488e-4d0c-8b37-e00b72059fdd",
            speed="fast",
            api_key=os.getenv("CARTESIA_API_KEY"),
        ),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        max_tool_steps=1,
    )

    await session.start(agent=FrontDeskAgent(timezone=timezone), room=ctx.room)

    # --- Integrate interrupt handler with session events ---
    async for event in session.events():
        if event.type == "playback_started":
            interrupt_handler.set_agent_state(True)

        elif event.type == "playback_finished":
            interrupt_handler.set_agent_state(False)

        elif event.type == "transcription":
            text = getattr(event, "text", "").strip()
            confidence = getattr(event, "confidence", 0.9)

            if not text:
                continue

            result = await interrupt_handler.handle_transcript(text, confidence)
            if result:
                print(f"🛑 Interruption detected: '{result}'")
                await session.stop_playback()  # Immediately stop TTS


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
