# LiveKit Tavus Avatar Agent

This example demonstrates how to create a animated avatar using [Tavus](https://platform.tavus.io/).

## Usage

* Update the environment:

```bash
# Tavus Config
export TAVUS_API_KEY="..."
export TAVUS_REPLICA_ID="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/tavus/agent_worker.py dev
```
