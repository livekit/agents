# Synthesia plugin for LiveKit Agents

Attach a Synthesia interactive avatar to a LiveKit voice agent. The avatar joins
the room and lip-syncs the agent's speech in real time.

See the [Synthesia integration docs](https://docs.synthesia.io) for more information.

## Installation

```bash
pip install livekit-plugins-synthesia
```

## Pre-requisites

You'll need an API key from Synthesia. It can be set as an environment variable:
`SYNTHESIA_API_KEY`

## Usage

```python
from livekit.plugins import synthesia

avatar = synthesia.AvatarSession(
    synthesia.AvatarConfig(
        avatar_ids=["03cee7ec-ac90-45ec-8c20-74a399cf3dc4"]
    ),  # SYNTHESIA_API_KEY from env
)
await avatar.start(session, room=ctx.room)  # before session.start
await session.start(agent=Agent(...), room=ctx.room)
```

Set your Synthesia workspace API key in `SYNTHESIA_API_KEY`, or pass `api_key=`.
`avatar_ids` takes one to five gallery ids of avatars available to your
workspace. The first is the active avatar; the rest are precomputed by the
worker so `swap_avatar()` can switch to them mid-session. An id your workspace
cannot access raises `synthesia.SynthesiaError` with `type=synthesia.ErrorType.UNKNOWN_AVATAR`.

```python
await avatar.swap_avatar("<another-id-from-avatar-ids>")  # switch mid-session
await avatar.swap_avatar("default")  # back to the first id
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `avatar_participant_identity` | `"synthesia-avatar-agent"` | The LiveKit identity the avatar joins under. Must be unique per concurrent avatar in a room: LiveKit evicts an existing participant when a second joins with the same identity, so give each avatar its own identity to run several in one room. |
| `avatar_participant_name` | `"Synthesia avatar"` | The LiveKit display name the avatar joins under. |

```python
avatar = synthesia.AvatarSession(
    synthesia.AvatarConfig(avatar_ids=["03cee7ec-ac90-45ec-8c20-74a399cf3dc4"]),
    avatar_participant_identity="avatar-host",
)
```
