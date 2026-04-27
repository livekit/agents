# livekit-plugins-runway

[LiveKit Agents](https://docs.livekit.io/agents/) plugin for [Runway Characters](https://dev.runwayml.com/) avatar integration.

Your LiveKit agent owns the full conversational AI pipeline (STT, LLM, TTS). Runway provides the visual layer — audio in, avatar video out.

## Installation

```bash
pip install livekit-plugins-runway
```

## Usage

```python
from livekit.agents import AgentSession, Agent, RoomOutputOptions
from livekit.plugins import runway

async def entrypoint(ctx):
    session = AgentSession()

    avatar = runway.AvatarSession(
        avatar_id="your-custom-avatar-id",
        # api_key defaults to RUNWAYML_API_SECRET env var
    )
    await avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
        room_output_options=RoomOutputOptions(audio_enabled=False),
    )
```

### Using a preset avatar

```python
avatar = runway.AvatarSession(
    preset_id="runway-preset-slug",
)
```

### With a session duration limit

```python
avatar = runway.AvatarSession(
    avatar_id="your-custom-avatar-id",
    max_duration=300,
)
```

## Configuration

| Parameter | Env var | Description |
|-----------|---------|-------------|
| `api_key` | `RUNWAYML_API_SECRET` | Runway API key |
| `api_url` | `RUNWAYML_BASE_URL` | API base URL (default: `https://api.dev.runwayml.com`) |
| `avatar_id` | — | Custom avatar ID (mutually exclusive with `preset_id`) |
| `preset_id` | — | Preset avatar slug (mutually exclusive with `avatar_id`) |
| `max_duration` | — | Maximum session duration in seconds |

LiveKit credentials (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`) are read from environment variables or can be passed to `avatar.start()`.
