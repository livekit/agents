import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.transcription.filters import filter_markdown
from livekit.plugins import cartesia, deepgram, openai, silero
from livekit.rtc.audio_frame import AudioFrame

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your are a helpful assistant.",
        )

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[AudioFrame]:
        filtered_text = filter_markdown(text)
        return super().tts_node(filtered_text, model_settings)

    # async def transcription_node(
    #     self, text: AsyncIterable[str], model_settings: ModelSettings
    # ) -> AsyncIterable[str]:
    #     filtered_text = filter_markdown(text)
    #     return super().transcription_node(filtered_text, model_settings)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=cartesia.TTS(),
    )

    await session.start(agent=MyAgent(), room=ctx.room)

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
