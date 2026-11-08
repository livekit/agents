# LiveKit Ultravox Plugin

LiveKit plugin for Ultravox's real-time speech-to-speech AI models, providing seamless integration with the LiveKit Agents framework.



## Installation

```bash
pip install livekit-plugins-ultravox
```

## Prerequisites

You'll need an API key from Ultravox. Set it as an environment variable:

```bash
export ULTRAVOX_API_KEY="your_api_key_here"
```

Optional: enable debug logs for the plugin (disabled by default):

```bash
export LK_ULTRAVOX_DEBUG=true
```

## Basic Usage

### Simple Voice Assistant

```python
from livekit.agents import Agent, AgentServer, JobContext, VoiceSession, cli, inference
from livekit.plugins.ultravox.realtime import RealtimeModel

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = VoiceSession(
        vad=inference.VAD(),
        llm=RealtimeModel(
            model="fixie-ai/ultravox",
            voice="Mark",
        ),
    )
    
    await session.start(
        agent=Agent(
            instructions="You are a helpful voice assistant.",
        ),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(server)
```

### Voice Assistant with Tools

```python
from livekit.agents import (
    Agent,
    AgentServer,
    JobContext,
    VoiceSession,
    cli,
    function_tool,
    inference,
)
from livekit.plugins.ultravox.realtime import RealtimeModel

@function_tool
async def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"The weather in {location} is sunny and 72°F"

@function_tool
async def book_appointment(date: str, time: str) -> str:
    """Book an appointment."""
    return f"Appointment booked for {date} at {time}"

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = VoiceSession(
        vad=inference.VAD(),
        llm=RealtimeModel(model="fixie-ai/ultravox"),
    )
    
    await session.start(
        agent=Agent(
            instructions="You are a helpful assistant with access to weather and scheduling tools.",
            tools=[get_weather, book_appointment],
        ),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(server)
```


## Configuration Options

### Ultravox API (/api/calls) Parameters

```python
RealtimeModel(
    model="fixie-ai/ultravox",            # Model to use (warn + pass-through if unknown)
    voice="Mark",                         # Voice to use (warn + pass-through if unknown)
    api_key=None,                          # API key (defaults to env var)
    base_url=None,                         # API base URL (defaults to Ultravox API)
    system_prompt="You are helpful.",      # System prompt
    input_sample_rate=16000,               # Input audio sample rate
    output_sample_rate=24000,              # Output audio sample rate
    http_session=None,                     # Custom HTTP session
)

Notes:
- Unknown models/voices: the plugin logs a warning and sends them as-is; the server validates.
- Metrics: the plugin emits a single `metrics_collected` event per generation.
```



