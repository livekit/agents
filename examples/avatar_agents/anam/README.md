# LiveKit Anam Avatar Agent

This example demonstrates how to create a animated avatar using [Anam](https://lab.anam.ai/).

Create your avatar [here](https://lab.anam.dev/avatars)

## Usage

* Update the environment:

```bash
# Anam Config
export ANAM_API_KEY="..."
export ANAM_AVATAR_ID="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/anam/agent_worker.py dev
```
