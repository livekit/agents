# LiveKit Plugins Spatius

Agent Framework plugin for [Spatius](https://www.spatius.ai) avatars.

See the [Spatius documentation](https://docs.spatius.ai) for Spatius account setup and
avatar configuration.

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
