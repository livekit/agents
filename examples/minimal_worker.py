import logging

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.pipeline import ChatCLI, PipelineAgent, AgentTask
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    agent = PipelineAgent(
        task=AgentTask(
            instructions="Talk to me!",
            llm=openai.realtime.RealtimeModel(),
        )
    )
    agent.start()

    # start a chat inside the CLI
    chat_cli = ChatCLI(agent)
    await chat_cli.run()


if __name__ == "__main__":
    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
