import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    TurnHandlingOptions,
    cli,
    inference,
    metrics,
    room_io,
    text_transforms,
)
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import ChatMessage, function_tool
from livekit.plugins import assemblyai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("assemblyai-agent-context")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly, built by LiveKit. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
            tools=[EndCallTool()],
        )

    async def on_enter(self) -> None:
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply(instructions="greet the user and introduce yourself")

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ) -> str:
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=0.3)


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session: AgentSession = AgentSession(
        # Speech-to-text (STT) is your agent's ears. Here we use AssemblyAI's
        # universal-streaming model directly (requires ASSEMBLYAI_API_KEY).
        # u3-rt-pro is used because it supports the prompt/agent_context biasing
        # demonstrated below.
        stt=assemblyai.STT(
            model="u3-rt-pro",
            vad_threshold=0.3,
        ),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM("openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
            # See more at https://docs.livekit.io/agents/build/turns
            turn_detection=MultilingualModel(),
            interruption={
                # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
                # when it's detected, you may resume the agent's speech
                "resume_false_interruption": True,
            },
        ),
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        # blocks interruptions for a few seconds after the agent starts speaking to allow client to calibrate AEC
        aec_warmup_duration=3.0,
        tts_text_transforms=[
            "filter_emoji",
            "filter_markdown",
            text_transforms.replace({"LiveKit": "<<ˈ|l|aɪ|v>> <<ˈ|k|ɪ|t>>"}),
        ],
    )

    # After each agent turn, push what the agent said into AssemblyAI's
    # `agent_context` so it has conversational context for transcribing the
    # user's reply. This is the one line that enables the feature.
    assemblyai.enable_agent_context(session)

    # For this demo only: log the text being fed as agent_context so you can
    # watch it in the console. The actual push to AssemblyAI is done by
    # enable_agent_context above — this handler is purely for visibility.
    @session.on("conversation_item_added")
    def _log_agent_context(ev) -> None:
        item = ev.item
        if isinstance(item, ChatMessage) and item.role == "assistant" and item.text_content:
            logger.info(f"agent_context → AssemblyAI: {item.text_content!r}")

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {session.usage}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
