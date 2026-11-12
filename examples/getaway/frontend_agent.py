import os
from collections.abc import AsyncIterator

from livekit.agents import Agent, AgentHandler, JobContext, LiveSession
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate

DESKS = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/getaway/default/"

PERSONA = (
    "You are the concierge of Getaway, a travel agency. You speak English. Warm and brief, never "
    "a script. A flight the traveler wants goes to the flight desk, a hotel to the hotel desk, at "
    "once, with all they said, cities, dates and numbers; a question or a change goes to the desk "
    "it is about, even while it works, never resending what it still works on. Each desk holds "
    "its own fares or rooms and asks for what it still needs, never you first. Say what each "
    'answers, asks or is doing as your own, "I\'m checking", naming no desk; nothing is booked or '
    "charged until a desk says so. Never guess: no answer, fare, price or time a desk did not "
    "give. Only pleasantries and the conversation itself are yours."
)

# two experts, each its own tool: both can work, and ask, at once; the answer detector picks
# which open question the traveler's reply answers
flight_desk = A2ADelegate(DESKS + "flights", name="flights", answer_detector=True)
hotel_desk = A2ADelegate(DESKS + "hotels", name="hotels", answer_detector=True)

chat = Agent(
    instructions=PERSONA + " By text: one or two plain sentences, no lists.",
    llm="google/gemma-4-31b-it",
    tools=[flight_desk, hotel_desk],
)

voice = Agent(
    instructions=PERSONA + " On the phone: one short sentence at a time, one question per "
    "turn, plain speech. Say times the way a person does, Friday at two, no AM or PM. While a "
    "desk works, keep the traveler company in one line.",
    tools=[flight_desk, hotel_desk],
)


concierge = AgentHandler()  # the worker's own lane: the phone and the text door share it


@concierge.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = LiveSession(
        stt="deepgram/nova-3",
        llm="google/gemma-4-31b-it",
        tts="cartesia/sonic-3",
        thread=ctx.thread,
    )
    await session.start(agent=voice, room=ctx.room)
    await session.generate_reply(instructions="Greet the traveler and offer to plan a trip.")


@concierge.a2a_session(endpoint="frontend", description="Getaway's concierge.", inbox="live")
async def text(ctx: A2ASessionContext) -> AsyncIterator[str]:
    async for delta in chat.stream(ctx.message, message_history=ctx.message_history):
        yield delta
