# livekit-plugins-keyframe

Agent Framework plugin for [Keyframe Labs](https://keyframelabs.com) avatars.

## Installation

```bash
pip install livekit-plugins-keyframe
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import keyframe

session = AgentSession(stt=..., llm=..., tts=...)

avatar = keyframe.AvatarSession(
    persona_id="ab85a2a0-0555-428d-87b2-ff3019a58b93",  # or persona_slug="public:cosmo_persona-1.5-live"
    api_key="keyframe_sk_live_...",  # or set KEYFRAME_API_KEY env var
)

await avatar.start(session, room=ctx.room)
await session.start(room=ctx.room, agent=my_agent)
```

## Authentication

Set the following environment variables:

- `KEYFRAME_API_KEY` - Your Keyframe API key
- `LIVEKIT_URL` - LiveKit server URL
- `LIVEKIT_API_KEY` - LiveKit API key
- `LIVEKIT_API_SECRET` - LiveKit API secret
