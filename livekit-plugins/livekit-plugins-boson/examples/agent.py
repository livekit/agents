"""A voice agent backed by the Boson (Higgs) realtime model.

Get an API key at https://www.boson.ai/workspace/api-key and export it:

    export BOSON_API_KEY=...

Then talk to it through your own microphone and speakers, with no LiveKit
server and no LIVEKIT_* credentials involved:

    python agent.py console

`console` needs PortAudio (`apt-get install libportaudio2` on Debian/Ubuntu).
To run it against a LiveKit room instead, use `dev` or `start` and set the
LIVEKIT_* variables as usual.

Try asking what the weather is somewhere, or what time it is, to see a function
call go out and come back. Interrupting mid-answer should stop it immediately.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import boson

logger = logging.getLogger("boson-agent")


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant speaking to the user over voice. "
                "Keep answers to one or two sentences. Do not use emoji, markdown, "
                "or any other characters that do not read aloud."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Greet the user and offer your help.")

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Look up the current weather for a location.

        Args:
            location: The city or region the user asked about.
        """
        logger.info("looking up weather for %s", location)
        # A real integration would call a weather API here.
        return f"It is sunny in {location}, 22 degrees celsius."

    @function_tool
    async def get_current_time(self, context: RunContext) -> str:
        """Get the current UTC date and time."""
        return datetime.now(timezone.utc).strftime("%A %d %B %Y, %H:%M UTC")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        # url defaults to the hosted endpoint and api_key falls back to
        # BOSON_API_KEY, so a bare RealtimeModel() is enough to connect.
        # input_audio_transcription_model is what turns user transcripts on;
        # without it the server still runs ASR for the model's own use but
        # sends nothing back to display.
        llm=boson.realtime.RealtimeModel(
            input_audio_transcription_model="higgs-stt-3.1",
            input_audio_transcription_language="english",
        ),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
