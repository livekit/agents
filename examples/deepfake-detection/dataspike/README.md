# LiveKit Dataspike Deepfake Detection Example

This example demonstrates how to use the [Dataspike](https://dataspike.io/) deepfake detection plugin with a LiveKit Agent.

## Usage

* Update the environment:

```bash
# Dataspike Config
export DATASPIKE_API_KEY="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/dataspike/agent_worker.py dev
```
