import asyncio
import logging
import data_url

from dotenv import load_dotenv
from livekit import ByteStreamReader
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.voice import AgentTask, CallContext, VoiceAgent
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai

# from livekit.plugins import noise_cancellation

logger = logging.getLogger("open-gpt-example")
logger.setLevel(logging.INFO)

load_dotenv()


class OpenGptTask(AgentTask):
    def __init__(self) -> None:
        self.llm = openai.LLM(model="gpt-4o-mini")
        super().__init__(
            instructions="You are OpenGPT. Format all responses in markdown.",
            stt=deepgram.STT(),
            llm=self.llm,
            tts=cartesia.TTS(),
        )

    def upload_user_file(self, stream: ByteStreamReader, participant_identity: str):
        self.llm.upload_file(stream, participant_identity)


# received_images = dict[str, llm.ImageContent]()


async def handle_user_file(agent: VoiceAgent, stream: ByteStreamReader):
    if stream.info.mime_type.startswith("image/"):
        data = await stream.read_all()
        url = data_url.construct_data_url(stream.info.mime_type, base64_encoded=True, data=data)
        image = llm.ImageContent(image=url)
        # received_images[stream.info.stream_id] = image
        agent.chat_ctx.add_message(llm.ChatMessage(role="user", content=image))
    else:
        logger.error(f"Unsupported file type: {stream.info.mime_type}")


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = VoiceAgent(
        task=OpenGptTask(),
    )

    ctx.room.register_byte_stream_handler(
        "user_files",
        lambda reader: asyncio.create_task(handle_user_file(agent, reader)),
    )

    await agent.start(
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
