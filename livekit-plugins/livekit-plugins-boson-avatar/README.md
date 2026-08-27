# Boson Higgs Avatar plugin for LiveKit Agents

Use Boson's Higgs Audio-Driven Avatar as the video output for a LiveKit voice
agent. This is an Avatar plugin: it composes with your existing voice/LLM
plugin and does not replace or fork it.

## Installation

```shell
pip install livekit-plugins-boson-avatar
```

Set the credentials used by your LiveKit Agent Worker:

```shell
export BOSON_API_KEY="..."
export LIVEKIT_URL="wss://..."
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
```

## Usage

Create the voice `AgentSession` as usual, then start the Avatar before starting
the agent session:

```python
from livekit.agents import Agent, AgentSession, JobContext, inference
from livekit.plugins import boson_avatar


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
    )
    avatar = boson_avatar.AvatarSession(
        avatar_id="your-avatar-id",
    )

    await avatar.start(session, room=ctx.room)
    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room,
    )
```

`BOSON_AVATAR_ID` can supply `avatar_id` instead. `AvatarSession` also accepts
optional `width`, `height`, `max_duration_seconds`,
`avatar_participant_identity`, and `APIConnectOptions`.

The `avatar_id` is the value behind the Avatar selected in your application or
Boson dashboard; end users do not need to type or remember it.

The plugin handles the provider API call, LiveKit participant token, PCM data
stream routing, interruption buffer clears, and provider-session cleanup. A
developer does not need to call Boson's Avatar REST API or combine voice and
Avatar responses in an application server. The application server only needs
its normal responsibility: create a LiveKit room and dispatch the Agent Worker.

When started inside a LiveKit job, cleanup is registered automatically. If you
start an Avatar outside a job context, call `await avatar.aclose()` yourself.
