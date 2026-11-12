from pathlib import Path

import behaviors
import knowledge
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    JobContext,
    VoiceSession,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

load_dotenv()

AGENT_NAME = "homepage_agent_v3"
LLM_MODEL = "google/gemma-4-31b-it"
STT_MODEL = "deepgram/nova-3"
TTS_MODEL = "fishaudio/s2.1-pro"
TTS_VOICE = "51b44863613e405a896f7f4294c6e6d0"
TTS_VOICE_LABEL = "Marley"

GREETING = (
    "Greet the user warmly. Introduce yourself as an assistant who can help with LiveKit Agents "
    "and other LiveKit products. Ask them how familiar they are with LiveKit so you can tailor "
    "your answers."
)

assistant = Agent(
    instructions=Path(__file__).with_name("instructions.md").read_text(encoding="utf-8"),
    llm=inference.LLM(model=LLM_MODEL),
    tools=[knowledge.lookup_product],
)

server = AgentServer(agent_name=AGENT_NAME)


@server.rtc_session()
async def homepage_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = VoiceSession(
        stt=inference.STT(model=STT_MODEL, language="multi"),
        tts=inference.TTS(model=TTS_MODEL, voice=TTS_VOICE),
        turn_handling={"preemptive_generation": {"enabled": True}},
        expressive=True,
    )
    behaviors.pronounce_livekit(session)
    behaviors.check_in_when_user_away(session)
    behaviors.disconnect_scripted_callers(session)
    behaviors.publish_tts_voice(TTS_VOICE_LABEL)

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )
    session.generate_reply(instructions=GREETING, allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
