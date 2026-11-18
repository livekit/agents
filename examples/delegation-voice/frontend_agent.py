import backend_agent

from livekit.agents import (
    Agent,
    AgentHandler,
    DelegationOptions,
    JobContext,
    VoiceSession,
    inference,
)
from livekit.agents.a2a import RemoteAgent

MODEL = "google/gemma-4-31b-it"

frontend = Agent(
    instructions=(
        "You are a voice assistant. One short sentence at a time, plain speech. Jobs and "
        "questions about documents go to the backend, with everything the user said. While it "
        "works, keep talking with the user and answer small talk yourself. Relay its questions "
        "and results in your own words, and never confirm anything it did not return."
    ),
    llm=inference.LLM(model=MODEL),
    # The backend as a tool: each call starts a remote task, and the frontend does not wait.
    tools=[RemoteAgent(backend_agent.text).as_tool()],
)

front_desk = AgentHandler()


@front_desk.rtc_session()
async def call(ctx: JobContext) -> None:
    session = VoiceSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        tts=inference.TTS(model="fishaudio/s2.1-pro", voice="51b44863613e405a896f7f4294c6e6d0"),
        # A separate model decides which open question each thing the user says answers.
        delegation=DelegationOptions(routing_model=inference.LLM(model=MODEL)),
        context=ctx.store.bind(),
    )
    await session.start(agent=frontend, room=ctx.room)
    await session.generate_reply(instructions="Greet the user and offer to help.")
