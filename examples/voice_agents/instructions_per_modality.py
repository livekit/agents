import logging
from datetime import datetime

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    function_tool,
    inference,
)
from livekit.agents.beta import Instructions
from livekit.plugins import silero

logger = logging.getLogger("instructions-per-modality")

load_dotenv()

BASE_INSTRUCTIONS = """\\
You are a scheduling assistant named Alex that helps users book appointments.
{modality_specific}
Call `book_appointment` to finalise the booking.
Never invent or assume details the user did not provide — ask for them instead.
The current date is {current_date}.
"""

# Voice users speak in approximate, self-correcting natural language.
# The LLM needs guidance on how to parse what was said, not how to say things back.
AUDIO_SPECIFIC = """
The user is speaking — their input arrives as voice transcription and may be imperfect.
When interpreting what the user said:
- Resolve relative spoken expressions to a concrete date/time: 'next Tuesday', 'tomorrow afternoon', 'the week after next around 3'.
- Spoken numbers may be ambiguous: 'three thirty' could mean 3:30 PM or the 30th of March — ask for clarification when context does not make it obvious.
- Honor verbal self-corrections: if the user says 'wait, I meant Thursday not Tuesday', update your understanding to Thursday and discard Tuesday.
- Ignore filler words and hesitations ('um', 'uh', 'like', 'I guess').
- Always confirm the resolved date and time out loud before booking, since spoken input is inherently ambiguous.
"""

# Text users type precise values — no need to normalise spoken patterns.
TEXT_SPECIFIC = """
The user is typing — take their input literally.
When interpreting what the user wrote:
- Accept exact dates and times in any common format (ISO, natural language, 12-hour or 24-hour clock).
- If the user provides a complete and unambiguous date and time, you may book immediately without asking for confirmation.
- Only ask follow-up questions for genuinely missing information.
"""


class SchedulingAgent(Agent):
    def __init__(self) -> None:
        current_date = datetime.now().strftime("%Y-%m-%d %A")
        super().__init__(
            instructions=Instructions(
                audio=BASE_INSTRUCTIONS.format(
                    modality_specific=AUDIO_SPECIFIC, current_date=current_date
                ),
                text=BASE_INSTRUCTIONS.format(
                    modality_specific=TEXT_SPECIFIC, current_date=current_date
                ),
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()

    @function_tool
    async def book_appointment(self, date: str, time: str) -> None:
        """Book an appointment.

        Args:
            date: The date of the appointment in the format YYYY-MM-DD
            time: The time of the appointment in the format HH:MM
        """
        logger.info(f"booking appointment for {date} at {time}")
        return f"Appointment booked for {date} at {time}"


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=ctx.proc.userdata["vad"],
    )

    await session.start(agent=SchedulingAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
