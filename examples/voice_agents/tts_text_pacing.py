import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, function_tool, tts
from livekit.plugins import cartesia, deepgram, openai, silero  # noqa: F401

logger = logging.getLogger("tts-text-pacing")

load_dotenv()

# Example using TTS stream pacer to control text flow to TTS.
# Reduces interruption waste and improves speech quality by batching text with more context.
# Works with non-streaming TTS (StreamAdapter) or Cartesia TTS in streaming mode.
#
# NOTE: The default transcription sync relies on full audio length, so sync quality suffers when audio
# generation is incomplete. Enable `use_tts_aligned_transcript` to improve sync quality if possible.


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
        )

    @function_tool
    async def lookup_weather(self, location: str):
        """Called when the user asks for weather related information.

        Args:
            location: The location they are asking for
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=tts.StreamAdapter(
            tts=openai.TTS(),
            text_pacing=True,  # use the default pacer configuration
        ),
        # tts=cartesia.TTS(
        #     text_pacing=tts.SentenceStreamPacer(  # or use a custom pacer to specify the params
        #         min_remaining_audio=3.0,
        #         max_text_length=300,
        #     )
        # ),
        use_tts_aligned_transcript=True,
    )

    await session.start(agent=MyAgent(), room=ctx.room)
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
