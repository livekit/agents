from __future__ import annotations

import datetime
import logging
import os
import sys
from dataclasses import dataclass
from typing import Literal, List
from zoneinfo import ZoneInfo
import asyncio

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
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

# ---------------------- NEW INTERRUPT HANDLER ----------------------

class InterruptHandler:
    """
    Handles filler-word-based interruption filtering during voice interaction.
    """
    def __init__(self, ignored_words: List[str] | None = None):
        env_list = os.getenv("IGNORED_FILLERS", "uh,umm,hmm,haan").split(",")
        self.ignored_words = set(w.strip().lower() for w in (ignored_words or env_list))
        self.agent_speaking = False
        self.logger = logging.getLogger("interrupt-handler")

    def set_agent_state(self, is_speaking: bool):
        """Update the agent speaking status."""
        self.agent_speaking = is_speaking

    async def handle_transcription(self, text: str, confidence: float = 0.9) -> bool:
        """
        Returns True if interruption should be passed to agent.
        Returns False if it's a filler-only utterance while agent is speaking.
        """
        cleaned = text.strip().lower()

        # Ignore very low confidence transcriptions
        if confidence < 0.6:
            self.logger.debug(f"Ignored low-confidence input: {cleaned}")
            return False

        # If agent not speaking, treat all speech normally
        if not self.agent_speaking:
            return True

        # Agent is speaking — check if filler or real interruption
        tokens = cleaned.split()
        if all(word in self.ignored_words for word in tokens):
            self.logger.info(f"Ignored filler interruption: '{cleaned}'")
            return False

        # Mixed or valid interruption (contains command words)
        self.logger.info(f"Valid interruption detected: '{cleaned}'")
        return True

# ------------------------------------------------------------------

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
                "When the user greets you, use that as an opportunity to move things forward. "
                "For example: 'Would you like to book a time?' "
                "When asked for availability, call list_available_slots and offer clear options. "
                "Avoid AM/PM and timestamps; use 'morning' or 'evening'. "
                "If a time is unavailable, inform the user and suggest another. "
                "Always keep the conversation flowing — be proactive, human, and helpful."
            )
        )

        self._slots_map: dict[str, AvailableSlot] = {}

    @function_tool
    async def schedule_appointment(
        self,
        ctx: RunContext[Userdata],
        slot_id: str,
    ) -> str | None:
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
        self, ctx: RunContext[Userdata], range: Literal["+2week", "+1month", "+3month", "default"]
    ) -> str:
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

            if local.date() == now.date():
                rel = "in less than an hour" if seconds < 3600 else "later today"
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

    timezone = "utc"
    interrupt_handler = InterruptHandler()

    if cal_api_key := os.getenv("CAL_API_KEY", None):
        logger.info("CAL_API_KEY detected, using cal.com calendar")
        cal = CalComCalendar(api_key=cal_api_key, timezone=timezone)
    else:
        logger.warning("CAL_API_KEY not set. Using FakeCalendar fallback.")
        cal = FakeCalendar(timezone=timezone)

    await cal.initialize()

    session = AgentSession[Userdata](
        userdata=Userdata(cal=cal),
        stt="assemblyai/universal-streaming:en",
        llm="google/gemini-2.0-flash",
        tts=cartesia.TTS(voice="39b376fc-488e-4d0c-8b37-e00b72059fdd", speed="fast"),
        # turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        max_tool_steps=1,
    )

    agent = FrontDeskAgent(timezone=timezone)

    async def event_loop():
        async for event in session.events():
            if event.type == "playback_started":
                interrupt_handler.set_agent_state(True)
            elif event.type == "playback_finished":
                interrupt_handler.set_agent_state(False)
            elif event.type == "transcription":
                text = getattr(event, "text", "").strip()
                confidence = getattr(event, "confidence", 0.9)
                allow_interrupt = await interrupt_handler.handle_transcription(text, confidence)
                if not allow_interrupt:
                    continue  # Skip filler interruptions

            await session.handle_event(event)

    await asyncio.gather(event_loop(), session.start(agent=agent, room=ctx.room))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
