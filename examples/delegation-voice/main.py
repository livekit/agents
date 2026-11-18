from dotenv import load_dotenv

# before the agents import: they build their models from these keys
load_dotenv()

import jobs  # noqa: E402
import lookup  # noqa: E402

from livekit.agents import (  # noqa: E402
    Agent,
    AgentHandler,
    AgentServer,
    DelegationOptions,
    JobContext,
    VoiceSession,
    cli,
    inference,
)
from livekit.agents.a2a import RemoteAgent  # noqa: E402
from livekit.agents.storage import AgentDBStorage  # noqa: E402

MODEL = "google/gemma-4-31b-it"

assistant = Agent(
    instructions=(
        "You are a voice assistant. One short sentence at a time, plain speech. Jobs go to the "
        "jobs backend and questions about documents to the lookup backend, each with "
        "everything the user said; both can work at the same time. While they work, keep "
        "talking with the user. Relay their questions and results in your own words, and "
        "never confirm anything they did not return."
    ),
    llm=inference.LLM(model=MODEL),
    # Two remote agents as tools. Each call starts a remote task; the assistant does not wait.
    tools=[
        RemoteAgent(jobs.endpoint).as_tool(),
        RemoteAgent(lookup.endpoint).as_tool(),
    ],
)

frontend = AgentHandler()


@frontend.rtc_session()
async def call(ctx: JobContext) -> None:
    session = VoiceSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        tts=inference.TTS(model="fishaudio/s2.1-pro", voice="51b44863613e405a896f7f4294c6e6d0"),
        # A separate model decides which open question each thing the user says answers.
        delegation=DelegationOptions(routing_model=inference.LLM(model=MODEL)),
        context=ctx.store.bind(),
    )
    await session.start(agent=assistant, room=ctx.room)
    await session.generate_reply(instructions="Greet the user and offer to help.")


server = AgentServer(storage=AgentDBStorage(), agent_name="delegation-voice")
server.add(frontend, jobs.jobs, lookup.lookup)

if __name__ == "__main__":
    cli.run_app(server)
