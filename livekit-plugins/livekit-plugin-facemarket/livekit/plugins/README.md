# livekit-plugins-facemarket

`livekit-plugins-facemarket` is a lightweight Avatar Plugin for LiveKit Agents.

It is designed for the narrow integration model confirmed for this project:

- LiveKit Agents users only
- caller owns `STT / LLM / TTS`
- plugin is responsible for avatar session orchestration and signaling
- FaceMarket backend is responsible for starting renderer/coordinator participants

## Install

```bash
pip install livekit-plugins-facemarket
```

## Quick Start

```python
from livekit.agents import RoomOutputOptions
from livekit.plugins.facemarket import AvatarSession

avatar = AvatarSession(
    avatar_id="2",
    platform_api_key="your-app-key",
    livekit_url="wss://your-livekit-host",
    livekit_api_key="your-livekit-api-key",
    livekit_api_secret="your-livekit-api-secret",
)

@avatar.on("session_ready")
async def on_session_ready() -> None:
    print("avatar session is ready")

await session.start(
    agent=agent,
    room=ctx.room,
    room_output_options=RoomOutputOptions(audio_enabled=False),
)

await avatar.start(agent_session=session, room=ctx.room)

await avatar.interrupt()
await avatar.stop()
```

## Expected Agent Setup

This plugin assumes your agent already owns the conversation pipeline.

- user audio -> your `STT`
- text -> your `LLM`
- response -> your `TTS`
- avatar plugin only starts and coordinates FaceMarket renderer participants

When using a renderer-based avatar, your agent should not also publish the final spoken audio back to the room for end users. Follow your LiveKit Agents setup so avatar output is the visible user-facing stream.

In practice, the simplest way is to disable the default room audio output from the agent session and let the FaceMarket renderer publish the user-facing media.

## Public API

### `AvatarSession(...)`

Constructor arguments:

- `avatar_id`: FaceMarket avatar ID
- `platform_api_key`: FaceMarket app key used to exchange a bearer token
- `livekit_url`: LiveKit server URL
- `livekit_api_key`: LiveKit API key used to mint renderer/coordinator tokens
- `livekit_api_secret`: LiveKit API secret used to mint renderer/coordinator tokens

### `await avatar.start(agent_session, room, ready_timeout=30.0)`

Behavior:

- exchanges FaceMarket bearer token
- mints renderer and coordinator LiveKit tokens locally
- calls `POST /dispatcher/v1/session/start`
- waits for `session.state=connected` from the coordinator data channel before returning

### `await avatar.interrupt()`

Publishes a `control.interrupt` message over LiveKit data channel.

### `await avatar.stop()`

Calls `POST /dispatcher/v1/session/stop` with the active `sessionId`.

## Events

Decorator-based callbacks are supported:

```python
@avatar.on("session_ready")
async def on_ready() -> None:
    ...

@avatar.on("idle_trigger")
async def on_idle() -> None:
    ...

@avatar.on("session_closing")
async def on_closing() -> None:
    ...

@avatar.on("error")
async def on_error(payload: dict) -> None:
    ...
```

Supported events:

- `session_ready`
- `idle_trigger`
- `session_closing`
- `error`

## Default FaceMarket API Paths

- `GET /dispatcher/auth/get_auth_token?appKey=...`
- `POST /dispatcher/v1/session/start`
- `POST /dispatcher/v1/session/stop`

Base URL is fixed to:

```text
https://pre.facemarket.ai/vih
```

## Smoke Demo

A local self-test is included and does not require network credentials:

```bash
python -m livekit.plugins.facemarket.demo
```

The self-test validates:

- dispatcher start/stop client injection
- coordinator `session.state=connected` readiness handling
- AgentSession event mapping into Data Channel messages
- `idle_trigger`, `session_closing`, and `error` callbacks

A minimal real room/session smoke demo is also included:

```bash
python -m livekit.plugins.facemarket.demo \
  --avatar-id 2 \
  --platform-api-key your-app-key \
  --livekit-url wss://your-livekit-host \
  --livekit-api-key your-livekit-api-key \
  --livekit-api-secret your-livekit-api-secret \
  --room-name your-room
```

Or use the installed script:

```bash
facemarket-avatar-demo \
  --avatar-id 2 \
  --platform-api-key your-app-key \
  --livekit-url wss://your-livekit-host \
  --livekit-api-key your-livekit-api-key \
  --livekit-api-secret your-livekit-api-secret \
  --room-name your-room
```

This demo is only for validating:

- token minting
- room join
- backend session startup
- coordinator readiness

It is not a full conversational agent sample.

## Real Voice Agent Demo

To test real interaction, install the AI plugins:

```bash
python -m pip install livekit-plugins-openai livekit-plugins-silero
```

Then run:

```bash
python -m livekit.plugins.facemarket.voice_agent_demo \
  --avatar-id avatar_01kpymd6csx995qen61hqn7m76 \
  --platform-api-key your-facemarket-key \
  --livekit-url wss://your-livekit-host \
  --livekit-api-key your-livekit-api-key \
  --livekit-api-secret your-livekit-api-secret \
  --room-name test_room \
  --stt-api-key your-openai-key \
  --llm-api-key your-openai-key \
  --tts-api-key your-openai-key
```

This demo runs a real `VAD -> STT -> LLM -> TTS` chain and leaves room audio
output enabled so the FaceMarket renderer can subscribe to the agent TTS track.
The agent participant is hidden by default to avoid showing a black tile in LiveKit Meet.

If you use DeepSeek for LLM, keep a real OpenAI-compatible STT/TTS key for speech:

```bash
python -m livekit.plugins.facemarket.voice_agent_demo \
  --avatar-id avatar_01kpymd6csx995qen61hqn7m76 \
  --platform-api-key your-facemarket-key \
  --livekit-url wss://your-livekit-host \
  --livekit-api-key your-livekit-api-key \
  --livekit-api-secret your-livekit-api-secret \
  --room-name test_room \
  --stt-api-key your-openai-key \
  --tts-api-key your-openai-key \
  --llm-api-key your-deepseek-key \
  --llm-model deepseek-chat \
  --llm-base-url https://api.deepseek.com/v1
```

DeepSeek only covers the LLM part. Voice interaction still requires working STT and TTS providers.
