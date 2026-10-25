import os
from collections.abc import AsyncIterator

from livekit.agents import Agent, AgentHandler, JobContext, LiveSession
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.plugins import openai

# the back office on the project's own host, which the worker connects to
BACKEND = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/dentist/default/backend"

PERSONA = (
    "You are the front desk of Pearl Dental, a small dental practice. You speak English. Warm "
    "and brief, never a script. You can check openings and book a visit; say so to a greeting, "
    "a question about the practice, or a patient unsure what to ask. Anything about openings "
    "or a booking, even only 'I need one' or 'I don't want that slot', goes at once to the "
    "back office with what the patient said. "
    "Never ask first for a day, a time, an email or what the visit is for: every visit is the "
    "same half-hour slot, and the back office asks for what it needs. State only what it gave, "
    "as given, and no opening hours of your own; nothing is booked until it says so."
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
    session = LiveSession(llm=openai.realtime.RealtimeModel(), thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)
    if not session.history.items:
        await session.generate_reply(
            instructions="Greet the patient and say that you can check openings and book a visit."
        )


@front_desk.a2a_session(endpoint="frontend", description="Pearl Dental front desk.")
async def text(ctx: A2ASessionContext) -> AsyncIterator[str]:
    async for delta in chat.stream(ctx.message, message_history=ctx.message_history):
        yield delta
