# LiveKit Plugins PersonaPlex

Agent Framework plugin for NVIDIA PersonaPlex full-duplex conversational AI.

## Installation

```bash
pip install livekit-plugins-personaplex
```

## Prerequisites

You need a running PersonaPlex server on a GPU machine. See https://github.com/NVIDIA/personaplex for setup.

Set the server URL as an environment variable:

```bash
export PERSONAPLEX_URL="gpu-server:8998"
```

## Usage

```python
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import personaplex, silero

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=personaplex.RealtimeModel(
            voice="NATF2",
            text_prompt="You are a friendly assistant named Aria.",
        ),
    )

    await session.start(
        agent=Agent(instructions="You are a friendly assistant."),
        room=ctx.room,
    )

def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
```

## Configuration

```python
personaplex.RealtimeModel(
    base_url=None,                  # Server URL (defaults to PERSONAPLEX_URL env var, then localhost:8998)
    voice="NATF2",                  # Voice prompt (NATF0-3, NATM0-3, VARF0-4, VARM0-4)
    text_prompt="You are helpful.", # System prompt / persona description
    seed=None,                      # Optional seed for reproducibility
    silence_threshold_ms=500,       # Silence duration before finalizing a generation
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PERSONAPLEX_URL` | PersonaPlex server address (host:port) | `localhost:8998` |
