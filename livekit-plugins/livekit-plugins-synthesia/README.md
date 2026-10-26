# Synthesia plugin for LiveKit Agents

Attach a [Synthesia](https://www.synthesia.io/) interactive avatar to a LiveKit voice agent. The avatar joins
the room and lip-syncs the agent's speech in real time.

See the [Synthesia integration docs](https://docs.livekit.io/agents/models/avatar/plugins/synthesia/) for more information.

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

## Faster first words

By default `start()` returns once the avatar has joined the room, so the agent's
greeting is only generated after the session request completes. Pass
`wait_for_avatar=False` to return straight away and launch the avatar in the
background. The greeting is then generated while the avatar starts, held until
the avatar's video is up, and spoken as soon as it is.

```python
await avatar.start(session, room=ctx.room, wait_for_avatar=False)
await session.start(agent=Agent(...), room=ctx.room)
session.generate_reply(instructions="Greet the user.")

await avatar.wait_until_ready()  # optional: raises synthesia.SynthesiaError if the avatar failed to start
```

With `wait_for_avatar=False`, a failed launch no longer raises from `start()`.
It closes the avatar session, is logged, and is raised by `wait_until_ready()`.
The agent's audio stays routed to the avatar until the session ends, so its
speech won't be heard: end or restart the session when `wait_until_ready()`
raises.

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
