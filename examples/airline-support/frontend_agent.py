import os

from livekit.agents import Agent, AgentHandler, JobContext, RunResult, VoiceSession
from livekit.agents.a2a import A2ASessionContext, RemoteAgent

# the back office on the project's own host, which the worker connects to
BACKEND = (
    os.environ["LIVEKIT_URL"].replace("ws", "http", 1) + "/agents/skyline-support/default/backend"
)
LLM = "google/gemma-4-31b-it"

PERSONA = (
    "You are Skyline Air's customer care. You speak English. Warm and brief, never a script. "
    "A late or cancelled flight, a bag, a refund or a change goes to the back office, stated "
    "in full: what the customer wrote, word for word, and the address they write from; ask "
    "for that address first if they have not given it. It holds the bookings and the policy "
    "and asks for what it still needs. A reply it drafted for the customer reaches them as "
    "written; anything else it answered, tell them in your own words. Never decide, compute "
    "or promise anything yourself: amounts and references are its, repeated as given. Only "
    "pleasantries and the conversation itself are yours."
)

# the back office as the front desk's one tool, described from its card; with no token, each
# call shares its thread with it under a grant minted for that thread
back_office = RemoteAgent(BACKEND)

chat = Agent(
    instructions=PERSONA + " By text: one or two plain sentences, no lists.",
    llm=LLM,
    tools=[back_office.as_tool()],
)

voice = Agent(
    instructions=PERSONA + " On the phone: one short sentence at a time, one question per "
    "turn, plain speech. Say references letter by letter and money the way a person does. "
    "While the back office works, keep the customer company in one line.",
    tools=[back_office.as_tool()],
)


front_desk = AgentHandler()  # the worker's own lane: the phone and the text door share it


@front_desk.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = VoiceSession(llm=LLM, thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)
    if not session.history.items:
        await session.generate_reply(instructions="Greet the customer in one sentence.")


@front_desk.a2a_session(endpoint="frontend", description="Skyline Air customer care.")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    return await chat.run(ctx.message, message_history=ctx.message_history)
