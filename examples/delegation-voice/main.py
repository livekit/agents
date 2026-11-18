from dotenv import load_dotenv

# before the agents import: they build their models from these keys
load_dotenv()

import pastry  # noqa: E402
import reservations  # noqa: E402

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

host = Agent(
    instructions=(
        "You are the host of Bistro Lumen, on the phone. One short sentence at a time, plain "
        "speech. Tables go to the reservations desk and cakes to the pastry kitchen, each with "
        "everything the guest said; both can work at the same time. While they work, keep the "
        "guest company and answer what you can yourself: the bistro is open 6 to 11 pm, on Rue "
        "des Lilas, and the menu is seasonal French. Relay their questions and answers in your "
        "own words, and never confirm anything they did not return."
    ),
    llm=inference.LLM(model=MODEL),
    # Two slow brains as tools. Each call is a remote task; the host does not wait on it.
    tools=[
        RemoteAgent(reservations.desk).as_tool(),
        RemoteAgent(pastry.kitchen).as_tool(),
    ],
)

front = AgentHandler()


@front.rtc_session()
async def call(ctx: JobContext) -> None:
    session = VoiceSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        tts=inference.TTS(model="fishaudio/s2.1-pro", voice="51b44863613e405a896f7f4294c6e6d0"),
        # A separate model decides which open question each thing the guest says answers.
        delegation=DelegationOptions(routing_model=inference.LLM(model=MODEL)),
        context=ctx.store.bind(),
    )
    await session.start(agent=host, room=ctx.room)
    await session.generate_reply(instructions="Greet the guest and offer to book a table.")


server = AgentServer(storage=AgentDBStorage(), agent_name="delegation-voice")
server.add(front, reservations.reservations, pastry.pastry)

if __name__ == "__main__":
    cli.run_app(server)
