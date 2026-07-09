# LiveKit Avaluma Avatar Agent

This example demonstrates how to create an animated avatar using [Avaluma](https://avaluma.ai/).

## Usage

* Update the environment:

```bash
# Avaluma Config
export AVALUMA_LICENSE_KEY="..."
export AVALUMA_AVATAR_ID="..."
# optional: only when self-hosting the avatar server (defaults to Avaluma's hosted service)
export AVALUMA_AVATAR_SERVER_URL="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/avaluma/agent_worker.py dev
```
