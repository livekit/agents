import logging

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

# from livekit.plugins import noise_cancellation

logger = logging.getLogger("roomio-example")
logger.setLevel(logging.INFO)

load_dotenv()


class EchoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo.",
            # llm=openai.realtime.RealtimeModel(voice="echo"),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
            vad=silero.VAD.load()
        )

    async def on_enter(self):
        self.session.generate_reply()


    @function_tool
    async def talk_to_alloy(self, context: RunContext):
        """Called when want to talk to Alloy."""
        return AlloyAgent(), "Transferring you to Alloy."


class AlloyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def talk_to_echo(self, context: RunContext):
        """Called when want to talk to Echo."""
        return EchoAgent(), "Transferring you to Echo."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await session.start(
        agent=AlloyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True)
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
