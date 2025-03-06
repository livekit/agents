import logging

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask, VoiceAgent
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


class MyTask(AgentTask):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant that can answer questions and help with tasks.",
        )

    @ai_function()
    async def open_door(self):
        await self.agent.say("Opening the door...")

        print("Door opened")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = VoiceAgent(
        instructions="You are a helpful assistant that can answer questions and help with tasks.",
        # llm=openai.realtime.RealtimeModel(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )
    await agent.start(room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
