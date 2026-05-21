# LiveKit Plugins Ojin

Agent Framework plugin for [Ojin](https://ojin.ai) avatars.

## Installation

```bash
pip install livekit-plugins-ojin
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import ojin

avatar = ojin.AvatarSession(
    api_key="your-api-key",
    config_id="your-config-id",
)

# In your agent entrypoint function:
await avatar.start(agent_session, room)
```

See the [examples](../../examples/avatar_agents/ojin/) for a complete working implementation.

## Environment Variables

- `OJIN_API_KEY`: Your Ojin API key
- `OJIN_CONFIG_ID`: Your Ojin configuration ID
- `OJIN_WS_URL`: WebSocket URL (optional, defaults to `wss://models.ojin.ai/realtime`)
