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
import asyncio
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import silero
from livekit.plugins.ultravox.realtime import RealtimeModel

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session: AgentSession[None] = AgentSession(
        allow_interruptions=True,
        vad=ctx.proc.userdata["vad"],
        llm=RealtimeModel(
            model_id="fixie-ai/ultravox",
            voice="Mark",
        ),
    )
    
    await session.start(
        agent=Agent(
            instructions="You are a helpful voice assistant.",
        ),
        room=ctx.room,
    )

def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
```

### Voice Assistant with Tools

```python
from livekit.agents import function_tool, Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import silero
from livekit.plugins.ultravox.realtime import RealtimeModel

@function_tool
async def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"The weather in {location} is sunny and 72°F"

@function_tool
async def book_appointment(date: str, time: str) -> str:
    """Book an appointment."""
    return f"Appointment booked for {date} at {time}"

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    session: AgentSession[None] = AgentSession(
        allow_interruptions=True,
        vad=ctx.proc.userdata["vad"],
        llm=RealtimeModel(model_id="fixie-ai/ultravox"),
    )
    
    await session.start(
        agent=Agent(
            instructions="You are a helpful assistant with access to weather and scheduling tools.",
            tools=[get_weather, book_appointment],
        ),
        room=ctx.room,
    )

def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
```


## Configuration Options

### Ultravox API (/api/calls) Parameters

```python
RealtimeModel(
    model_id="fixie-ai/ultravox",         # Model to use (warn + pass-through if unknown)
    voice="Mark",                         # Voice to use (warn + pass-through if unknown)
    api_key=None,                          # API key (defaults to env var)
    base_url=None,                         # API base URL (defaults to Ultravox API)
    system_prompt="You are helpful.",      # System prompt
    input_sample_rate=16000,               # Input audio sample rate
    output_sample_rate=24000,              # Output audio sample rate
    client_buffer_size_ms=60,              # Audio buffer size (min 200ms used on WS)
    http_session=None,                     # Custom HTTP session
)

Notes:
- Unknown models/voices: the plugin logs a warning and sends them as-is; the server validates.
- Metrics: the plugin emits a single `metrics_collected` event per generation. To log them,
  add a listener in your app and call the helper:

```python
from livekit.agents import metrics

@session.on("metrics_collected")
def on_metrics_collected(ev):
    metrics.log_metrics(ev.metrics)
```
```



