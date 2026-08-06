"""A LiveKit voice agent powered by Cartesia speech-to-text and text-to-speech.

Requires ``CARTESIA_API_KEY`` from https://play.cartesia.ai/keys
and one of:

- ``LIVEKIT_INFERENCE_API_KEY`` + ``LIVEKIT_INFERENCE_API_SECRET``
- or ``ANTHROPIC_API_KEY``
- or ``GOOGLE_API_KEY``
- or ``OPENAI_API_KEY``

Run with:

    uv run examples/other/cartesia.py
"""

import logging
import os
from collections.abc import Callable

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    cli,
    inference,
    metrics,
    room_io,
)
from livekit.agents.beta.tools.end_call import EndCallTool
from livekit.agents.llm import LLM
from livekit.plugins import anthropic, cartesia, google, openai


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="your name is Katie, built by Cartesia."
            " you would interact with users via voice."
            " with that in mind, keep your responses concise and to the point."
            " do not use emojis, asterisks, markdown, or other special characters in your responses."
            " you are curious and friendly, and have a sense of humor."
            " you will speak english to the user.",
            tools=[EndCallTool()],
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and introduce yourself")


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("CARTESIA_API_KEY")

    llm_factories: list[Callable[[], LLM]] = [
        lambda: inference.LLM("google/gemini-3-flash"),
        lambda: anthropic.LLM(model="claude-haiku-4-5"),
        lambda: google.LLM(model="gemini-3.5-flash"),
        lambda: openai.LLM(model="gpt-5.4-mini"),
    ]

    llm: LLM | None = None
    for factory in llm_factories:
        try:
            llm = factory()
            break
        except ValueError:
            continue

    if not api_key or llm is None:
        parts: list[str] = []
        if not api_key:
            parts.append("CARTESIA_API_KEY is required")
        if llm is None:
            parts.append(
                "No LLM keys were provided (e.g. LIVEKIT_INFERENCE_API_KEY + LIVEKIT_INFERENCE_API_SECRET,"
                " ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY)"
            )
        raise ValueError(". ".join(parts))

    logger = logging.getLogger("cartesia-demo-agent")
    server = AgentServer()

    @server.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        ctx.log_context_fields = {
            "room": ctx.room.name,
        }
        session: AgentSession = AgentSession(
            stt=cartesia.STT(
                model="ink-2",
                api_key=api_key,
            ),
            llm=llm,
            tts=cartesia.TTS(
                model="sonic-3.5",
                api_key=api_key,
            ),
            turn_handling={
                # ink-2 does a great job without VAD
                # you may use ink-2 with VAD if desired
                "turn_detection": "stt",
            },
        )

        @session.on("metrics_collected")
        def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
            metrics.log_metrics(ev.metrics)

        async def log_usage():
            logger.info(f"Usage: {session.usage}")

        ctx.add_shutdown_callback(log_usage)

        await session.start(
            agent=MyAgent(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(),
            ),
        )

    cli.run_app(server)


if __name__ == "__main__":
    main()
