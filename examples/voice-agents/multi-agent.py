import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.job import get_current_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero

# uncomment to enable Krisp BVC noise cancellation, currently supported on Linux and MacOS
# from livekit.plugins import noise_cancellation

## The storyteller agent is a multi-agent that can handoff the session to another agent.
## This example demonstrates more complex workflows with multiple agents.
## Each agent could have its own instructions, as well as different STT, LLM, TTS, or realtime models.

logger = logging.getLogger("multi-agent")

load_dotenv()

common_instructions = (
    "Your name is Echo. You are a story teller that interacts with the user via voice."
)
"You are curious and friendly, with a sense of humor."


@dataclass
class StoryData:
    """Shared data that's used by the storyteller agent. This structure is passed between agents in function calls' RunContext."""

    name: Optional[str] = None
    location: Optional[str] = None
    fun_activity: Optional[str] = None


class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{common_instructions} Your goal is to gather a few pieces of information from the user to make the story personalized and engaging."
            "You should ask the user for their name, where they are from, and what they'd like to do for fun."
            "Start the conversation with a short introduction, and then proceed to gather information.",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    @function_tool
    async def information_gathered(
        self,
        context: RunContext[StoryData],
        name: str,
        location: str,
        fun_activity: str,
    ):
        """Called when the user has provided the information needed to make the story personalized and engaging.

        Args:
            name: The name of the user
            location: The location of the user
            fun_activity: The activity the user would like to do for fun
        """

        context.userdata.name = name
        context.userdata.location = location
        context.userdata.fun_activity = fun_activity

        story_agent = StoryAgent(data=context.userdata)
        # by default, StoryAgent will start with a new context, to carry through the current
        # chat history, pass in the chat_ctx
        # story_agent = StoryAgent(data=context.userdata, chat_ctx=context.chat_ctx)

        return story_agent, "Let's start the story!"


class StoryAgent(Agent):
    def __init__(self, data: StoryData, chat_ctx: Optional[ChatContext] = None) -> None:
        super().__init__(
            instructions=f"{common_instructions}. You should use the user's information in order to make the story personalized."
            "create the entire story, weaving in elements of their information, and make it interactive, occasionally interating with the user."
            "do not end on a statement, where the user is not expected to respond."
            f"The user's name is {data.name}, they are from {data.location}, and they like to {data.fun_activity} for fun.",
            # each agent could override any of the model services, including mixing realtime and non-realtime models
            llm=openai.realtime.RealtimeModel(voice="echo"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        # when the agent is added to the session, we'll initiate the conversation by
        # using the LLM to generate a reply
        self.session.generate_reply()

    @function_tool
    async def story_finished(self, context: RunContext[StoryData]):
        """When you are fininshed telling the story (and the user confirms they don't want anymore), call this function to end the conversation."""
        # interrupt any existing generation
        self.session.interrupt()

        # generate a goodbye message and hang up
        # awaiting it will ensure the message is played out before returning
        await self.session.generate_reply(
            instructions=f"say goodbye to {context.userdata.name}", allow_interruptions=False
        )

        job_ctx = get_current_job_context()
        lkapi = job_ctx.api
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    userdata = StoryData()
    session = AgentSession[StoryData](
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(model="nova-3"),
        tts=openai.TTS(voice="echo"),
        userdata=userdata,
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=IntroAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
