# livekit-plugins-facemarket

`livekit-plugins-facemarket` is a lightweight Avatar Plugin for LiveKit Agents.

It is designed for the narrow integration model confirmed for this project:

- LiveKit Agents users only
- caller owns `STT / LLM / TTS`
- plugin is responsible for avatar session orchestration and signaling
- FaceMarket backend is responsible for starting renderer/coordinator participants

## Install

```bash
pip install livekit-plugins-facemarket
```

## Quick Start

```python
from livekit.agents import RoomOutputOptions
from livekit.plugins.facemarket import AvatarSession

avatar = AvatarSession(
    avatar_id="2",
    platform_api_key="your-app-key",
    livekit_url="wss://your-livekit-host",
    livekit_api_key="your-livekit-api-key",
    livekit_api_secret="your-livekit-api-secret",
)

@avatar.on("session_ready")
async def on_session_ready() -> None:
    print("avatar session is ready")

await session.start(
    agent=agent,
    room=ctx.room,
    room_output_options=RoomOutputOptions(audio_enabled=False),
)

await avatar.start(agent_session=session, room=ctx.room)

await avatar.stop()
```

## Get More

See the [FaceMarket LiveAvatar Integration Docs](https://github.com/newportAI-lab/liveavatar-integration-guide) for more information.