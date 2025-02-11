import logging

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.llm import ai_function
from livekit.agents.pipeline import AgentContext, AgentTask, ChatCLI, PipelineAgent
from livekit.plugins import openai, deepgram, cartesia, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


class EchoTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo, always speak in English even if the user speaks in another language or wants to use another language.",
            # llm=openai.realtime.RealtimeModel(voice="echo"),
        )

    @ai_function
    async def talk_to_alloy(self, context: AgentContext):
        return AlloyTask(), "Transfering you to Alloy."


class AlloyTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy, always speak in English even if the user speaks in another language or wants to use another language.",
            # llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    @ai_function
    async def talk_to_echo(self, context: AgentContext):
        return EchoTask(), "Transfering you to Echo."


async def entrypoint(ctx: JobContext):
    agent = PipelineAgent(
        task=AlloyTask(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load()
    )

    await agent.start()

    chat_cli = ChatCLI(agent)
    await chat_cli.run()

    # start a chat inside the CLI


if __name__ == "__main__":
    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
