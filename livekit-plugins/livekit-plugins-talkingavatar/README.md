# Talking Avatar plugin for LiveKit Agents

Support for [Talking Avatar](https://talking-avatar.dev) realistic avatars: a photorealistic avatar that joins the
room and speaks your agent's voice, with lip sync, from any LLM, TTS or realtime model.

See the [Talking Avatar LiveKit docs](https://docs.talking-avatar.dev/realistic-avatars/livekit) for more information.

## Installation

```bash
pip install livekit-plugins-talkingavatar
```

## Pre-requisites

You'll need an API key from the [Talking Avatar dashboard](https://app.talking-avatar.dev). It can be set as an
environment variable: `TALKING_AVATAR_API_KEY`. The plugin also reads `LIVEKIT_URL`, `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET` to mint the avatar's token.

## Usage

```python
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import talkingavatar


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        # stt=..., llm=..., tts=... or a realtime model
    )

    avatar = talkingavatar.AvatarSession(avatar_id="ava4")  # any avatar ID from your dashboard
    await avatar.start(session, room=ctx.room)

    await session.start(agent=Agent(instructions="You are a friendly assistant."), room=ctx.room)
```

The avatar joins the room as its own participant and publishes, on behalf of your agent:

| Track | Name | Content |
|---|---|---|
| Video | `avatar` | The avatar, 832 × 468 at 25 fps |
| Audio | `avatar-audio` | Your agent's voice, in sync with the lips |

Your agent no longer publishes its own audio. Interruptions stop the avatar mid-sentence, and the agent's
transcript is cut at what the user actually heard.

Billing is $0.02 per minute, per second, from the moment the avatar is ready until the session ends.
