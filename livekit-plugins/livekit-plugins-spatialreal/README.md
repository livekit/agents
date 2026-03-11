# LiveKit Agents Plugin for SpatialReal Avatar

This plugin provides integration with [SpatialReal](https://www.spatialreal.ai/)'s avatar service for lip-synced avatar rendering in LiveKit voice agents.

## Installation

```bash
pip install livekit-plugins-spatialreal
```

## Configuration

Set the following environment variables:

```bash
SPATIALREAL_API_KEY=your-api-key
SPATIALREAL_APP_ID=your-app-id
SPATIALREAL_AVATAR_ID=your-avatar-id

LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

## Usage

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import spatialreal

class VoiceAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant."
        )

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Configure your pipeline components (STT, LLM, TTS)
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
    )

    # Initialize and start the avatar session
    avatar = spatialreal.AvatarSession()
    await avatar.start(session, room=ctx.room)

    # Start the agent session
    await session.start(
        agent=VoiceAssistant(),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

For production agents, catch `SpatialRealException` so you can decide whether to fail the job or continue without avatar output:

```python
try:
    await avatar.start(session, room=ctx.room)
except spatialreal.SpatialRealException as err:
    logger.error("Avatar startup failed: %s", err)
    raise
```

## API Reference

### `AvatarSession`

Main class for integrating SpatialReal avatars with LiveKit agents.

#### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | `str` | SpatialReal API key (or set `SPATIALREAL_API_KEY`) |
| `app_id` | `str` | SpatialReal application ID (or set `SPATIALREAL_APP_ID`) |
| `avatar_id` | `str` | Avatar ID to use (or set `SPATIALREAL_AVATAR_ID`) |
| `console_endpoint_url` | `str` | Custom console endpoint URL |
| `ingress_endpoint_url` | `str` | Custom ingress endpoint URL |
| `avatar_participant_identity` | `str` | LiveKit identity for avatar participant |
| `idle_timeout_seconds` | `int` | LiveKit egress idle timeout in seconds (`0` uses server defaults) |
| `sample_rate` | `int \| None` | Optional avatar audio sample rate override |

#### Methods

- `start(agent_session, room, *, livekit_url, livekit_api_key, livekit_api_secret)`: Start the avatar session and hook into the agent's audio output. Raises `SpatialRealException` with actionable context if startup fails.
- `aclose()`: Clean up avatar session resources.

When starting, the plugin automatically sets `lk.publish_on_behalf` to the
agent participant identity for avatar worker association in LiveKit frontends.

### `SpatialRealException`

Exception raised for SpatialReal-related errors.

## License

MIT
