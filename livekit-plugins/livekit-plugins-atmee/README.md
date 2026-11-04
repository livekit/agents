# Atmee plugin for LiveKit Agents

Give a LiveKit voice agent a face with an [Atmee](https://www.atmee.ai/) avatar. The avatar joins
the room as its own participant and speaks the agent's audio with lips in sync. Your agent keeps its
own speech recognition, model and voice.

Any Atmee avatar renders by its id: a **v1 avatar** is a talking head generated from a single
portrait (`create_avatar` below); a **v2 avatar** moves as a whole and is created in the
[Atmee studio](https://www.atmee.ai/studio).

See the [Atmee integration docs](https://docs.livekit.io/agents/models/avatar/plugins/atmee/) for more information.

## Installation

```bash
pip install livekit-plugins-atmee
```

## Pre-requisites

You'll need an API key from Atmee, created at [atmee.ai/studio/api-keys](https://www.atmee.ai/studio/api-keys).
It can be set as an environment variable: `ATMEE_API_KEY`

The plugin mints the avatar participant's join token with your own LiveKit credentials
(`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`); only that token is sent to Atmee.

## Usage

```python
from livekit.plugins import atmee

# once: create an avatar from a portrait (a local path, raw bytes, or an https URL)
avatar_id = await atmee.AtmeeAPI().create_avatar(name="Val", image="portrait.jpg")

avatar = atmee.AvatarSession(avatar_id=avatar_id)  # ATMEE_API_KEY from env
await avatar.start(session, room=ctx.room)  # before session.start
await session.start(agent=Agent(...), room=ctx.room)
```

`start()` returns once Atmee's rendering worker has acknowledged the session; the avatar's video
and audio tracks appear in the room a few seconds later. The first v2 session after a quiet period
can take a few minutes to start, and `start()` waits for it (up to 5 minutes, 7.5 with
`wait_for="avatar_joined"`). The session ends when your agent leaves the room, when `aclose()` runs
(registered as a job shutdown callback), or at `max_duration_seconds`.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `avatar_id` | required | The Atmee avatar to render. |
| `api_key` | `ATMEE_API_KEY` | Your Atmee API key. |
| `api_url` | `ATMEE_API_URL` or `https://api.atmanity.us` | Atmee API base URL. |
| `avatar_participant_identity` | `"atmee-avatar-agent"` | The LiveKit identity the avatar joins under. Must be unique per concurrent avatar in a room. |
| `avatar_participant_name` | `"atmee-avatar-agent"` | The LiveKit display name the avatar joins under. |
| `max_duration_seconds` | `3600` | Hard ceiling of the session. |
| `wait_for` | `"initializing"` | `"avatar_joined"` makes `start()` block until the avatar is in the room. |
| `metadata` | `None` | Free-form JSON stored with the session on the Atmee side. |

API failures raise `atmee.AtmeeException` with `status_code`, `code` and `message`;
`atmee.AtmeeNoCapacityError` (check `retry_after`) and `atmee.AtmeeAvatarNotReadyError` are
subclasses worth handling. `atmee.AtmeeAvatarWarmingError` means a v2 avatar was still getting
ready when the start's wait ran out; nothing was billed, and you can try again after
`retry_after`.
