# LiveKit Plugins Spatius

Agent Framework plugin for [Spatius](https://www.spatius.ai) avatars.

See the [Spatius documentation](https://docs.spatius.ai) for Spatius account setup and
avatar configuration.

## Client-side rendering

Spatius avatars are rendered on the client instead of being sent as conventional
server-rendered video. The avatar's LiveKit video track carries motion data in
otherwise black frames, so a standard LiveKit video renderer will display a black
screen. Your frontend must use the Spatius client SDK and LiveKit adapter to decode
the track and render the avatar.

See the [client integration guide](https://docs.spatius.ai/livekit-agents/client) and
the [reference frontend](https://github.com/spatius-ai/spatius-avatar-demo/tree/main/platform-integrations/livekit-agents-demo/livekit-agents-reference-demo/frontend)
for a working implementation.

## Installation

```bash
pip install livekit-plugins-spatius
```

## Usage

```python
from livekit.plugins import spatius

avatar = spatius.AvatarSession()
await avatar.start(session, room=ctx.room)
```

The plugin reads `SPATIUS_API_KEY`, `SPATIUS_APP_ID`, and `SPATIUS_AVATAR_ID` from the
environment when constructor arguments are omitted. It defaults to the `us-west` Spatius
region and composes the production endpoint URLs automatically.
