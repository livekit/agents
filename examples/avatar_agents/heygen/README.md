# LiveKit HeyGen LiveAvatar Avatar Agent

This example demonstrates how to create a animated avatar using [Heygen LiveAvatar](https://www.liveavatar.com/).

## Usage

* Update the environment:

```bash
# HeyGen Config
export HEYGEN_API_KEY="..."
export HEYGEN_AVATAR_ID="..."

# STT + LLM + TTS config
export OPENAI_API_KEY="..."
export DEEPGRAM_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/heygen/agent_worker.py dev
```
