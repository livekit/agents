import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.utils import aio
from livekit.agents.voice.recorder_io.stereo_audio_recorder import StereoAudioRecorder
from livekit.plugins import silero
from livekit.plugins.cartesia import TTS
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

    async def on_enter(self):
        self.session.generate_reply()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


tasks = set()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="assemblyai/universal-streaming:en",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        # tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        tts=TTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
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
        # Only for recording purposes
        allow_interruptions=False,
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    async def record_stereo_audio(recorder: StereoAudioRecorder):
        while session.input.audio is None or session.output.audio is None:
            await asyncio.sleep(0.05)
        logger.info("Input and output audio are ready")
        try:
            session.input.audio = recorder.record_input(session.input.audio)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"RecorderIO record_input failed: {e}")
        try:
            session.output.audio = recorder.record_output(session.output.audio)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"RecorderIO record_output failed: {e}")

        try:
            await recorder.start(output_path="output.wav")
            logger.info("Started RecorderIO")
            try:
                ctx.add_shutdown_callback(recorder.aclose)
            except RuntimeError:
                pass
        except Exception as e:
            logger.error(f"Failed to start RecorderIO: {e}")

    recorder = StereoAudioRecorder(agent_session=session)

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent):
        nonlocal recorder
        if ev.new_state == "speaking":
            logger.info(f"Agent is speaking {recorder.started=}")
            recorder.start_agent_speech(ev.created_at)
        elif ev.old_state == "speaking":
            recorder.end_agent_speech()

    task = asyncio.create_task(record_stereo_audio(recorder))
    task.add_done_callback(tasks.discard)
    tasks.add(task)
    ctx.add_shutdown_callback(lambda: aio.cancel_and_wait(*tasks))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
