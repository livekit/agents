# bitHuman plugin for LiveKit Agents

Support for [bitHuman](https://www.bithuman.ai/) virtual avatars in LiveKit Agents. The avatar renders either locally, inside your agent's process from a bitHuman `.imx` model file (no GPU required), or in the bitHuman cloud, where it joins the room as a participant.

See the [bitHuman integration docs](https://docs.livekit.io/agents/models/avatar/plugins/bithuman/) for more information, and bitHuman's [LiveKit guide](https://docs.bithuman.ai/platforms/livekit) for a complete agent.

## Installation

```bash
pip install livekit-plugins-bithuman
```

To render Expression 2 models locally, also install the SDK's Expression 2 extra:

```bash
pip install "bithuman[expression-2]"
```

## Pre-requisites

You'll need an API secret from bitHuman ([create one](https://www.bithuman.ai/developer/api-keys)). Store it under a name the plugin doesn't read, such as `BITHUMAN_MASTER_SECRET`, and pass it as `api_secret=`. If `api_secret` isn't passed, the plugin falls back to `BITHUMAN_API_SECRET`.

In cloud mode the plugin puts `api_secret` in the avatar participant's attributes, which everyone in the room can read. Pass a one-hour token minted for one agent and one room (`POST https://api.bithuman.ai/v1/runtime-tokens/mint`, [how](https://docs.bithuman.ai/platforms/livekit#authenticate)) instead of the secret, and keep `BITHUMAN_API_SECRET` unset. Cloud mode also uses `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.

## Usage

Local mode: the avatar renders in this process.

```python
import os

from livekit.plugins import bithuman

avatar = bithuman.AvatarSession(
    model_path="./agent.imx",  # or set BITHUMAN_MODEL_PATH
    api_secret=os.environ["BITHUMAN_MASTER_SECRET"],
)
await avatar.start(session, room=ctx.room)
```

Cloud mode: the avatar renders on bitHuman's servers.

```python
avatar = bithuman.AvatarSession(
    avatar_id="A23WJF0199",  # your bitHuman agent code (A23WJF0199 is the wise-pup sample)
    api_secret=minted_token,  # from POST /v1/runtime-tokens/mint
)
await avatar.start(session, room=ctx.room)
```

In both modes the avatar publishes the agent's audio, so start the session with `room_options=room_io.RoomOptions(audio_output=False)`.

## Models

- **Essence 2** (`essence-2`): a photoreal person from one portrait.
- **Expression 2** (`expression-2`): any character, including people, animals and cartoons, from one portrait.
- `essence` / `expression`: the first-generation Essence 1 and Expression 1. Expression 1 runs in the cloud only.

In local mode the `.imx` file decides the model. In cloud mode, leave `model` unset to use the model the agent was created with, or set it to pin one; bitHuman refuses a model the avatar wasn't prepared for. A photo passed as `avatar_image` (no `avatar_id`) works only with Expression 1. Pricing: [docs.bithuman.ai/pricing](https://docs.bithuman.ai/pricing).
