import os
from collections.abc import AsyncIterator

from livekit.agents import Agent, AgentHandler, JobContext, LiveSession, inference
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.plugins import openai

# the back office on the project's own host, which the worker connects to
BACKEND = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/dentist/default/backend"

PERSONA = (
    "You are the front desk of Pearl Dental, a small dental practice. You speak English. Warm "
    "and brief, never a script. Anything about a visit, a cleaning, a check-up, a toothache, "
    "openings, insurance or a booking goes to the back office, in full with what the patient said, "
    "dates, times and email included; it holds the calendar and asks for what it still needs. "
    "Tell the patient what it answered, in your own words, and nothing is booked until it "
    "says so. Never guess what it will say or "
    "confirm anything it did not return; dates and times are its, repeated as given. Only "
    "pleasantries, the hours (weekdays, 9 to 5) and the conversation itself are yours."
)

# the back office as the front desk's one tool, described from its card
back_office = A2ADelegate(BACKEND)

chat = Agent(
    instructions=PERSONA + " By text: one or two plain sentences, no lists.",
    llm="google/gemma-4-31b-it",
    tools=[back_office],
)

voice = Agent(
    instructions=PERSONA + " On the phone: one short sentence at a time, one question per "
    "turn, plain speech. Say times the way a person does, Monday at two, no AM or PM. While "
    "the back office works, keep the patient company in one line.",
    tools=[back_office],
)


front_desk = AgentHandler()  # the worker's own lane: the phone and the text door share it


@front_desk.rtc_session()
async def phone(ctx: JobContext) -> None:
    # a local VAD, so the back office's progress and answers wait for the patient to stop talking
    # (with the model's own turn detection, the default VAD is left out)
    session = LiveSession(
        llm=openai.realtime.RealtimeModel(), vad=inference.VAD(model="silero"), thread=ctx.thread
    )
    await session.start(agent=voice, room=ctx.room)
    if not session.history.items:
        await session.generate_reply(instructions="Greet the patient and offer to book a visit.")


@front_desk.a2a_session(endpoint="frontend", description="Pearl Dental front desk.", inbox="live")
async def text(ctx: A2ASessionContext) -> AsyncIterator[str]:
    async for delta in chat.stream(ctx.message, message_history=ctx.message_history):
        yield delta
