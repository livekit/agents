# Boson Higgs Avatar plugin for LiveKit Agents

Use Boson's Higgs Audio-Driven Avatar as the video output for a LiveKit voice
agent. This is an Avatar plugin: it composes with your existing voice/LLM
plugin and does not replace or fork it.

The plugin has no dependency on Boson Voice, Boson Audio, or a particular TTS
provider. Its input is the standard audio output of a LiveKit `AgentSession`,
so it works with any TTS plugin, realtime model, or custom source that produces
LiveKit audio frames. Only Avatar rendering and Avatar session lifecycle are
Boson-specific.

## Installation

```shell
pip install livekit-plugins-boson-avatar
```

Set the credentials used by your LiveKit Agent Worker:

```shell
export BOSON_API_KEY="..."
export BOSON_AVATAR_API_URL="https://your-avatar-session-service.example/v1"
export LIVEKIT_URL="wss://..."
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
```

The plugin deliberately has no hard-coded provider endpoint. Your application
or deployment environment supplies the base URL exposed by its Boson Avatar
deployment/operator. The plugin appends `POST /sessions` when starting an
Avatar and `DELETE /sessions/{id}` during cleanup, so do not include
`/sessions` itself in `BOSON_AVATAR_API_URL`.

## Usage

Create the voice `AgentSession` with the audio provider of your choice, then
start the Avatar before starting the agent session:

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
`avatar_participant_identity`, `idempotency_key`, and `APIConnectOptions`.
`max_duration_seconds` must be an integer from 1 through 14400.

Inside a LiveKit Agent job, provider-session creation automatically derives a
stable UUID idempotency key from the LiveKit job ID and Avatar session binding.
If LiveKit redelivers that job after a worker crash, the plugin recovers the
existing provider session instead of allocating another one. The standard
model is one Avatar lifecycle per LiveKit job. If one job intentionally starts
another Avatar after closing the first, pass a new explicit UUID
`idempotency_key` for that lifecycle.

The `avatar_id` is the value behind the Avatar selected in your application or
Boson dashboard; end users do not need to type or remember it.

The plugin handles the provider API call, LiveKit participant token, PCM data
stream routing, interruption buffer clears, and provider-session cleanup. A
developer does not need to call Boson's Avatar REST API or combine voice and
Avatar responses in an application server. The application server only needs
its normal responsibility: create a LiveKit room and dispatch the Agent Worker.

When started inside a LiveKit job, cleanup is registered automatically. When
using the plugin in a standalone script or test, open LiveKit's HTTP context and
close the Avatar explicitly:

```python
from livekit.agents import utils


async with utils.http_context.open():
    await avatar.start(session, room)
    try:
        # Run the standalone session.
        ...
    finally:
        await avatar.aclose()
```
