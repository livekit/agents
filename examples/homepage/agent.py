import logging
from collections.abc import AsyncIterable
from dataclasses import dataclass

from behaviors.frontend_attributes import publish_frontend_attributes
from behaviors.user_away import check_in_when_user_away
from dotenv import load_dotenv
from filters.pronunciation import pronounce_livekit
from knowledge_base import KnowledgeBase
from prompts import prompt

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    ModelSettings,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

logger = logging.getLogger("agent")

load_dotenv()


@dataclass(frozen=True, slots=True)
class AgentConfig:
    name: str = "homepage_agent_v3"
    llm_model: str = "google/gemma-4-31b-it"
    stt_model: str = "deepgram/nova-3"
    stt_language: str = "multi"
    tts_model: str = "inworld/inworld-tts-2"
    tts_voice: str = "Nate"


CONFIG = AgentConfig()
KNOWLEDGE_BASE = KnowledgeBase()

INSTRUCTIONS = prompt("agents_sdks")
GREETING = prompt("greeting")


class Assistant(Agent):
    def __init__(
        self,
        config: AgentConfig = CONFIG,
        knowledge_base: KnowledgeBase = KNOWLEDGE_BASE,
    ) -> None:
        super().__init__(
            llm=inference.LLM(model=config.llm_model),
            instructions=INSTRUCTIONS,
            tools=[knowledge_base.lookup_tool()],
        )

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        async for frame in Agent.default.tts_node(self, pronounce_livekit(text), model_settings):
            yield frame

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=GREETING,
            allow_interruptions=True,
        )


server = AgentServer()


@server.rtc_session(agent_name=CONFIG.name)
async def homepage_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=inference.STT(model=CONFIG.stt_model, language=CONFIG.stt_language),
        tts=inference.TTS(model=CONFIG.tts_model, voice=CONFIG.tts_voice),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
        preemptive_generation=True,
    )

    check_in_when_user_away(session)
    publish_frontend_attributes(tts_voice=CONFIG.tts_voice)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
