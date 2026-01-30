# LiveKit Avatario Avatar Agent

This example demonstrates how to create an animated avatar using [Avatario](https://avatario.ai/).

## Usage

* Update the environment:

```bash
# Avatario Config
export AVATARIO_API_KEY="..."
export AVATARIO_AVATAR_ID="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/avatario/agent_worker.py dev
```
