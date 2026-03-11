# LiveKit SpatialReal Avatar Agent

This example demonstrates how to create a lip-synced avatar agent using [SpatialReal](https://www.spatialreal.ai/).

Create your SpatialReal app and avatar [here](https://app.spatialreal.ai/).

## Usage

* Update the environment:

```bash
# SpatialReal config
export SPATIALREAL_API_KEY="..."
export SPATIALREAL_APP_ID="..."
export SPATIALREAL_AVATAR_ID="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/spatialreal/agent_worker.py dev
```
