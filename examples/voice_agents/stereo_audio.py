import asyncio
import logging

import numpy as np
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
from livekit.agents.utils import aio
from livekit.agents.voice.recorder_io.stereo_audio_recorder import (
    StereoAudioRecorder,
)
from livekit.plugins import silero
from livekit.plugins.cartesia import TTS
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self, recording_started: asyncio.Future[None]) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
        )
        self.recording_started_fut = recording_started

    async def on_enter(self) -> None:
        # This is needed to ensure all audio is recorded before the agent starts speaking
        # as detaching and attaching the audio might take place after the on_enter call
        await self.recording_started_fut
        self.session.generate_reply()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


tasks = set()


async def entrypoint(ctx: JobContext) -> None:
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session: AgentSession = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="assemblyai/universal-streaming:en",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        # tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
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
        # Disabling it for recording demo purposes
        allow_interruptions=False,
    )

    recording_started: asyncio.Future[None] = asyncio.Future[None]()
    output_path: str = "output.wav"

    async def record_stereo_audio(recorder: StereoAudioRecorder) -> None:
        while session.input.audio is None or session.output.audio is None:
            await asyncio.sleep(0.05)

        try:
            session.input.audio = recorder.record_input(session.input.audio)  # type: ignore[arg-type]
        except Exception as e:
            raise RuntimeError(f"StereoAudioRecorder record_input failed: {e}") from e

        try:
            session.output.audio = recorder.record_output(session.output.audio)  # type: ignore[arg-type]
        except Exception as e:
            raise RuntimeError(f"StereoAudioRecorder record_output failed: {e}") from e

        logger.debug(f"Recorded audio: {session.input.audio=} {session.output.audio=}")

        try:
            await recorder.start(output_path=output_path)
            recording_started.set_result(None)
            try:
                ctx.add_shutdown_callback(recorder.aclose)
            except RuntimeError:
                pass
        except Exception as e:
            logger.error(f"Failed to start StereoAudioRecorder: {e}")

    # TODO: implement a proper callback for inference.
    # This is a placeholder for now.
    def inference_callback(wav: np.ndarray) -> None:
        import soundfile as sf

        sf.write(output_path, wav.T, 16000)

    recorder = StereoAudioRecorder(agent_session=session, inference_callback=inference_callback)
    task = asyncio.create_task(record_stereo_audio(recorder))
    tasks.add(task)
    task.add_done_callback(tasks.discard)

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
        nonlocal recorder
        if ev.new_state == "speaking":
            recorder.start_agent_speech(ev.created_at)
        elif ev.old_state == "speaking":
            recorder.end_agent_speech()

    await session.start(
        agent=MyAgent(recording_started=recording_started),
        room=ctx.room,
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    ctx.add_shutdown_callback(lambda: aio.cancel_and_wait(*tasks))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
