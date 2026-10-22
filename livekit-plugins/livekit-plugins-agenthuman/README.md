# AgentHuman plugin for LiveKit Agents

Support for [AgentHuman](https://agenthuman.com/) real-time avatars: the agent's speech is turned
into a lip-synced video avatar that joins the room and publishes on behalf of the agent.

See the [AgentHuman LiveKit docs](https://docs.agenthuman.com/documentation/livekit/overview) for
more information.

## Installation

```bash
pip install livekit-plugins-agenthuman
```

## Pre-requisites

You'll need an API key from the [AgentHuman dashboard](https://app.agenthuman.com/settings/apikeys). Set it, together with your
LiveKit credentials, as environment variables:

```bash
export AGENTHUMAN_API_KEY="..."
export LIVEKIT_URL="wss://..."
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
```

## Usage

Start the avatar before the agent session:

```python
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.plugins import agenthuman

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemini-2.5-flash"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_handling=TurnHandlingOptions(
            interruption={"mode": "vad", "resume_false_interruption": False},
        ),
    )

    avatar = agenthuman.AvatarSession(avatar_id="avat_...", aspect_ratio="3:4")
    await avatar.start(session, room=ctx.room)

    await session.start(agent=Agent(instructions="Talk to me!"), room=ctx.room)
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
```

Pick the avatar with exactly one of:

- `avatar_id`: an avatar from your AgentHuman account (`avat_...`), or `AGENTHUMAN_AVATAR_ID`.
- `avatar_image_url`: a publicly accessible image to animate, or `AGENTHUMAN_IMAGE_URL`.

## Interruptions

The avatar stops speaking as soon as the agent is interrupted, and reports how much of the reply
was actually heard, so the agent keeps only that part of the transcript. Two interruption settings
are recommended (see the usage example above):

- `"mode": "vad"`: interrupt on the user's voice activity. With the adaptive detector, speech
  that overlaps the avatar can be classified as a backchannel and ignored, so the avatar keeps
  talking through an interruption.
- `"resume_false_interruption": False`: the avatar can't pause and resume a reply.
