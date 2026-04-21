# Inworld plugin for LiveKit Agents

Support for voice synthesis and speech-to-text with [Inworld TTS](https://docs.inworld.ai/tts/tts) and [Inworld STT](https://docs.inworld.ai/stt/overview).

See [Inworld TTS](https://docs.livekit.io/agents/integrations/tts/inworld/) and [Inworld STT](https://docs.livekit.io/agents/models/stt/inworld/) for more information.

## Installation

```bash
pip install livekit-plugins-inworld
```

## Authentication

Set `INWORLD_API_KEY` in your `.env` file ([get one here](https://platform.inworld.ai/login)).

## Usage

### TTS

Use Inworld TTS within an `AgentSession` or as a standalone speech generator.

```python
from livekit.plugins import inworld

tts = inworld.TTS()
```

Or with options:

```python
from livekit.plugins import inworld

tts = inworld.TTS(
    voice="Hades",                 # voice ID (default or custom cloned voice)
    model="inworld-tts-1",         # or "inworld-tts-1-max"
    encoding="OGG_OPUS",           # LINEAR16, MP3, OGG_OPUS, ALAW, MULAW, FLAC
    sample_rate=48000,             # 8000-48000 Hz
    bit_rate=64000,                # bits per second (for compressed formats)
    speaking_rate=1.0,             # 0.5-1.5
    temperature=1.1,               # 0-2
    timestamp_type="WORD",         # WORD, CHARACTER, or TIMESTAMP_TYPE_UNSPECIFIED
    text_normalization="OFF",      # ON, OFF, or APPLY_TEXT_NORMALIZATION_UNSPECIFIED
)
```

### TTS Streaming

Inworld TTS supports WebSocket streaming for lower latency real-time synthesis. Use the
`stream()` method for streaming text as it's generated:

```python
from livekit.plugins import inworld

tts = inworld.TTS(
    voice="Hades",
    model="inworld-tts-1",
    buffer_char_threshold=100,     # chars before triggering synthesis (default: 100)
    max_buffer_delay_ms=3000,      # max buffer time in ms (default: 3000)
)

# Create a stream for real-time synthesis
stream = tts.stream()

# Push text incrementally
stream.push_text("Hello, ")
stream.push_text("how are you today?")
stream.flush()  # Flush any remaining buffered text
stream.end_input()  # Signal end of input

# Consume audio as it's generated
async for audio in stream:
    # Process audio frames
    pass
```

### STT

Use Inworld STT for streaming speech-to-text. Multiple models are supported.

```python
from livekit.plugins import inworld

session = AgentSession(
   stt=inworld.STT()
   # ... llm, tts, etc.
)
```

With a specific model and voice profile detection:

```python
from livekit.plugins import inworld

session = AgentSession(
   stt=inworld.STT(
       model="inworld/inworld-stt-1",
       enable_voice_profile=True,
   )
   # ... llm, tts, etc.
)
```

### Example

A full voice agent using Inworld for both STT and TTS:

```python
"""Inworld STT + TTS voice agent example.

Demonstrates using Inworld for both speech-to-text and text-to-speech
in a LiveKit voice agent. Save this as ``inworld_agent.py`` and run:

    uv run inworld_agent.py console   # local console mode
    uv run inworld_agent.py dev       # LiveKit Cloud (requires LIVEKIT_URL,
                                      # LIVEKIT_API_KEY, LIVEKIT_API_SECRET)

Then connect via https://agents-playground.livekit.io
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    metrics,
    room_io,
)
from livekit.plugins import inworld, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("inworld-agent")

load_dotenv()


class InworldAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Nova. You interact with users via voice. "
                "Keep your responses concise and to the point. "
                "Do not use emojis, asterisks, markdown, or other special characters. "
                "You are helpful, curious, and friendly."
            ),
        )

    async def on_enter(self):
        self.session.generate_reply()


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=inworld.STT(model="inworld/inworld-stt-1"),
        llm="openai/gpt-4.1-mini",
        tts=inworld.TTS(voice="Clive"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=InworldAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
```

### Combined TTS + STT

```python
from livekit.plugins import inworld

session = AgentSession(
   tts=inworld.TTS(voice="Hades"),
   stt=inworld.STT(),
   # ... llm, etc.
)
```
