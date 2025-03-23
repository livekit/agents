import logging

from dotenv import load_dotenv

from livekit.agents import JobContext, JobProcess, WorkerOptions, cli, llm
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.io import PlaybackFinishedEvent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector import EOUModel

logger = logging.getLogger("vad-realtime-example")
logger.setLevel(logging.INFO)

load_dotenv()


class AlloyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy.",
            llm=openai.realtime.RealtimeModel(
                voice="alloy", turn_detection=None, input_audio_transcription=None
            ),
        )

    async def on_enter(self):
        self.session.generate_reply()

    @llm.function_tool
    async def fetch_weather_today(self) -> str:
        """Called when the user asks for the weather today"""
        return "The weather today is sunny and 70 degrees."

    @llm.function_tool
    async def fetch_weather_tomorrow(self) -> str:
        """Called when the user asks for the weather tomorrow"""
        return "The weather tomorrow is rainy and 60 degrees."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        turn_detection=EOUModel(),
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        allow_interruptions=True,
    )
    await session.start(agent=AlloyAgent(), room=ctx.room)

    @session.output.audio.on("playback_finished")
    def on_playback_finished(event: PlaybackFinishedEvent):
        logger.info(
            "playback finished",
            extra={"duration": event.playback_position, "interrupted": event.interrupted},
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
