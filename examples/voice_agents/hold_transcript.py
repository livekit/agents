import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    JobContext,
    JobProcess,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
        )

    async def on_enter(self) -> None:
        self.session.generate_reply()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    tasks = set()

    async def track_audio_input_status(session: AgentSession) -> None:
        while session.input.audio is None:
            await asyncio.sleep(0.01)
        audio_recognition = session._activity._audio_recognition if session._activity else None
        if audio_recognition is None:
            return
        audio_recognition.enable_barge_in()

        def no_op() -> None:
            pass

        # disable interruption to allow stt to capture transcript but not interrupt the agent
        # it will be properly parameterized and conditioned later when we integrate the model
        session._activity._interrupt_by_audio_activity = no_op  # type: ignore[union-attr, method-assign]
        logger.debug("Barge-in monitoring enabled")

    session: AgentSession = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="assemblyai/universal-streaming:en",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        # still disabled internally, but this allows STT to work
        allow_interruptions=True,
    )
    task = asyncio.create_task(track_audio_input_status(session))
    tasks.add(task)
    task.add_done_callback(tasks.discard)

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
        logger.info(f"Agent state changed: {ev.old_state} -> {ev.new_state}")
        audio_recognition = session._activity._audio_recognition if session._activity else None
        if audio_recognition is None:
            return
        if ev.new_state == "speaking":
            audio_recognition.start_barge_in_monitoring()
        elif ev.old_state == "speaking":
            audio_recognition.end_barge_in_monitoring(ev.created_at)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
