# OpenSpline virtual avatars

OpenSpline adds a live avatar to any LiveKit voice agent, using Cloud or your own GPU.

```python
from livekit.plugins import openspline

avatar = openspline.AvatarSession(avatar_id="YOUR_AVATAR_ID")
await avatar.start(session, room=ctx.room)
```

Set `OPENSPLINE_API_KEY` and the standard `LIVEKIT_*` environment variables. Start the avatar before the agent session. For self-hosting, unset the Cloud key and use `portrait="portrait.jpg"`.

Requires OpenSpline 0.1.2 and Python 3.10–3.12. [Setup, examples and videos](https://docs.openspline.dev/connectors/livekit).
