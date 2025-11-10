from __future__ import annotations

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    inference,
)
from livekit.agents.voice.interrupt_filter import InterruptionClassifier

# Load environment after imports to satisfy linter (E402)
load_dotenv()
load_dotenv('config', override=True)


@function_tool
async def update_filter(
    context: RunContext,
    fillers: list[str] | None = None,
    stop_keywords: list[str] | None = None,
    language: str | None = None,
    min_confidence: float | None = None,
) -> str:
    """Hot-reload interruption filter settings during the session."""
    # Works if your session exposes update_interruption_filter (it does in this repo)
    context.session.update_interruption_filter(
        fillers=fillers,
        stop_keywords=stop_keywords,
        language=language,
        min_confidence=min_confidence,
    )
    return "interruption filter updated"


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # Ensure required providers are configured via env vars
    # DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVEN_API_KEY must be set for this demo

    # Show active interruption filter config (from env/config)
    clf = InterruptionClassifier.from_env()
    def _fmt_set(s):
        return ", ".join(sorted(s)) if s else "(empty)"
    print("\n=== Interruption Filter Config ===")
    print(f"Default fillers: [{_fmt_set(clf._fillers_default)}]")  # type: ignore[attr-defined]
    print(f"Default stops:   [{_fmt_set(clf._stop_default)}]")    # type: ignore[attr-defined]
    print(f"Min confidence:  {clf._min_conf}")                     # type: ignore[attr-defined]
    if getattr(clf, "_fillers_by_lang", None):
        for lang, s in clf._fillers_by_lang.items():
            print(f"Fillers[{lang}]:  [{_fmt_set(s)}]")
    if getattr(clf, "_stop_by_lang", None):
        for lang, s in clf._stop_by_lang.items():
            print(f"Stops[{lang}]:    [{_fmt_set(s)}]")
    print("===================================\n")

    agent = Agent(
        instructions=(
            "You are a concise, helpful voice assistant. "
            "Speak naturally in short sentences. "
            "While you are speaking, ignore filler utterances (e.g., 'um', 'umm', 'hmm', 'haan'). "
            "If the user gives a real instruction or a stop keyword (e.g., 'stop', 'wait', 'ruk'), "
            "immediately stop speaking and listen. "
            "When interrupted, briefly acknowledge and address the most recent user request first. "
            "If unsure, ask a single clarifying question. "
            "Avoid special formatting or symbols. "
            "Support English and Hindi; reply in the user's language."
        ),
        tools=[update_filter],
    )

    session = AgentSession(
        # Use VAD for responsive interruption handling
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
    )

    await session.start(agent=agent, room=ctx.room)

    # Kick off a greeting so you can test interruptions while the agent is speaking
    await session.generate_reply(instructions="Greet the user and ask how you can help today.")


if __name__ == "__main__":
    # Run with:  python examples/voice_agents/filler_interrupt_demo.py console
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
