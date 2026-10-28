# Humanlike plugin for LiveKit Agents

Attach a [Humanlike](https://humanlike.co/) realtime avatar to a LiveKit voice agent. The avatar joins the room
and lip-syncs the agent's speech.

See the [Humanlike integration docs](https://docs.livekit.io/agents/models/avatar/plugins/humanlike/) for more information.

## Installation

```bash
pip install livekit-plugins-humanlike
```

## Pre-requisites

You'll need a secret API key from Humanlike. It can be set as an environment variable: `HUMANLIKE_API_KEY`

## Usage

```python
from livekit.plugins import humanlike

avatar = humanlike.AvatarSession(model="humanlike/homo-1")  # HUMANLIKE_API_KEY from env
await avatar.start(session, room=ctx.room)  # before session.start
await session.start(agent=Agent(...), room=ctx.room)
```

`start()` mints a token for the avatar's participant in this room from `LIVEKIT_URL`, `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET` (or the `livekit_*` arguments of `start()`). Humanlike receives only that token, never the
API secret.

## Models

| Model | Rendering | Notes |
| --- | --- | --- |
| `humanlike/homo-1` (default) | GPU, published as a video track | Works with any LiveKit frontend. Sessions are capped at 570 s; the plugin starts a successor before the cap, with the same identity, so the avatar stays in the call. |
| `humanlike/homo-1-pro` | GPU, published as a video track | HOMO 1 PRO, with the same session cap and renewal. |
| `humanlike/homo-0` | In the viewer's browser or app | No avatar participant joins and the agent keeps its own audio track. The agent publishes what to draw in the `humanlike.avatar.render` participant attribute (`humanlike.ATTRIBUTE_RENDER`), so the frontend needs the Humanlike client renderer. |

The same key also starts the other vendors' avatars Humanlike resells (for example `anam/cara-4`), billed to one
Humanlike account. A model id the API does not know raises `APIStatusError` with the API's message.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `model` | `"humanlike/homo-1"` | The model's public id. |
| `avatar_id` | | The avatar (face or persona) to use, for models that take one. |
| `image_url` | | A photo to animate, for models that take one. |
| `options` | | Model-specific options, checked by the API against the model's schema. |
| `fallback` | | Model ids to try, in order, if `model` cannot start. `avatar.model` tells which one started. |
| `max_session_seconds` | | End the avatar after this long. |
| `avatar_participant_identity` | `"humanlike-avatar-agent"` | The identity the avatar joins under. Must be unique per concurrent avatar in a room. |
| `avatar_participant_name` | `"Humanlike Avatar"` | The display name the avatar joins under. |
| `start_timeout` | `180` | Seconds the start request may take, a cold GPU boot included. |
| `join_timeout` | `60` | Seconds to wait for the avatar's video once the API answered. |

`avatar.aclose()` ends the avatar and its billing; inside a job it runs at shutdown. If the avatar leaves the room
mid-call, the plugin logs a warning and ends the session right away rather than at its cap.
