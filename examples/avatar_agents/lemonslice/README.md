# LiveKit LemonSlice Avatar Agent

This example demonstrates how to create an animated avatar using [LemonSlice](https://www.lemonslice.com/).

## Usage

* Update the environment:

```bash
# LemonSlice Config
export LEMONSLICE_API_KEY="..."
export LEMONSLICE_IMAGE_URL="..." # Publicly accessible image url for the avatar.

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/lemonslice/agent_worker.py dev
```
