import os

from livekit.agents import Agent, AgentHandler, JobContext, LiveSession
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.plugins import openai

# the back office on the project's own host, which the worker connects to
BACKEND = os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/hotel/default/backend"


PERSONA = (
    "You are the front desk of The LiveKit Hotel, a small boutique hotel with a restaurant. You "
    "speak English. Natural, warm, never a script, and never padding. You know nothing about "
    "the hotel yourself: everything about it, a room, a table, a booking, a bill, a request, "
    "a policy, a price or the area goes to the back office, stated in full with what the "
    "guest said, codes, names and numbers included; it does the work and asks for what it "
    "still needs. Tell the guest what it answered, in your own words; its questions reach the "
    "guest as they are. Never guess what it will say or confirm anything it did not return; "
    "dates and times are its, repeated as given. Only pleasantries and the conversation itself "
    "are yours."
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
    "turn, plain speech the way a receptionist talks. Say money, dates and codes the way a "
    "person does, a code letter by letter. While the back office works, keep the guest "
    "company in one line.",
    tools=[back_office],
)


front_desk = AgentHandler()  # the worker's own lane: the phone and the text door share it


@front_desk.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = LiveSession(llm=openai.realtime.RealtimeModel(), thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)
    if not session.history.items:
        await session.generate_reply(instructions="Greet the guest in one sentence.")


@front_desk.a2a_session(endpoint="frontend", description="The LiveKit Hotel front desk.")
async def text(ctx: A2ASessionContext) -> str:
    result = await chat.run(ctx.message, message_history=ctx.message_history)
    return result.input_required or str(result.output)
