# livekit-plugins-kfl

Agent Framework plugin for [Keyframe Labs](https://keyframelabs.com) avatars.

## Installation

```bash
pip install livekit-plugins-kfl
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import kfl

session = AgentSession(stt=..., llm=..., tts=...)

avatar = kfl.AvatarSession(
    persona_id="ab85a2a0-0555-428d-87b2-ff3019a58b93",  # or persona_slug="public:cosmo_persona-1.5-live"
    api_key="kfl_sk_live_...",  # or set KFL_API_KEY env var
)

await avatar.start(session, room=ctx.room)
await session.start(room=ctx.room, agent=my_agent)
```

## Authentication

Set the following environment variables:

- `KFL_API_KEY` - Your KFL API key
- `LIVEKIT_URL` - LiveKit server URL
- `LIVEKIT_API_KEY` - LiveKit API key
- `LIVEKIT_API_SECRET` - LiveKit API secret
