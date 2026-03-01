# LiveKit Simli Avatar Agent

This example demonstrates how to create a animated avatar using [Simli](https://app.simli.com/).

## Usage

* Update the environment:

```bash
# Simli Config
export SIMLI_API_KEY="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/simli/agent_worker.py dev
```
