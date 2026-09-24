import logging
import os

from azure.ai.voicelive.models import AudioInputTranscriptionOptions
from dotenv import load_dotenv

from livekit.agents import NOT_GIVEN, Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.llm import function_tool
from livekit.plugins import azure

logger = logging.getLogger("azure-realtime-agent")
# the Azure SDK logs every websocket message, including base64 audio, at debug level
logging.getLogger("azure").setLevel(logging.WARNING)

# the plugin reads AZURE_VOICE_LIVE_ENDPOINT, AZURE_VOICE_LIVE_API_KEY and AZURE_VOICE_LIVE_MODEL
# from the environment, see examples/.env
load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a friendly voice assistant powered by Azure Voice Live. "
            "The user is talking to you over voice, so keep your responses short and "
            "conversational, without emojis, markdown, or other special characters. "
            "Always reply in the language the user speaks.",
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and offer your help")

    @function_tool
    async def lookup_weather(self, location: str) -> str:
        """Called when the user asks for weather related information.

        Args:
            location: The city or region to look up the weather for
        """
        logger.info(f"looking up weather for {location}")
        return "sunny, 22 degrees Celsius"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    transcription_model = os.getenv("AZURE_VOICE_LIVE_TRANSCRIPTION_MODEL")
    session: AgentSession[None] = AgentSession(
        llm=azure.realtime.RealtimeModel(
            voice=os.getenv("AZURE_VOICE_LIVE_VOICE") or "en-US-AvaMultilingualNeural",
            # optional: the plugin defaults to azure-speech for text models such as gpt-4.1 and
            # phi4-mm-realtime, and to whisper-1 for the other realtime models
            input_audio_transcription=AudioInputTranscriptionOptions(model=transcription_model)
            if transcription_model
            else NOT_GIVEN,
        ),
    )

    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
