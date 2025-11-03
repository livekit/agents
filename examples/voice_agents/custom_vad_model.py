from pathlib import Path

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import (
    deepgram,
    elevenlabs,
    openai,
    silero,
)
from livekit.plugins.turn_detector.english import EnglishModel

load_dotenv()


async def entrypoint(ctx: agents.JobContext):
    elevenlabs_tts = elevenlabs.TTS()
    # Use the silero vad model from the livekit-plugins-silero package itself.
    # We can also use a prev version of silero vad model by specifying the onnx_file_path.
    onnx_file_path = (
        Path(__file__).parent.parent.parent
        / "livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/resources/silero_vad.onnx"
    )
    session = AgentSession(
        stt=deepgram.STT(model="nova-2-phonecall"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs_tts,
        vad=silero.VAD.load(onnx_file_path=onnx_file_path),
        turn_detection=EnglishModel(),
        min_interruption_duration=0.5,
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You are a helpful assistant."),
    )

    session.on(
        "user_state_changed",
        lambda ev: print(f"User state changed: {ev.old_state} -> {ev.new_state}"),
    )

    await session.generate_reply(instructions="Greet the user and offer your assistance.")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
