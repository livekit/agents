# LiveKit LiveAvatar Avatar Agent

This example demonstrates how to create a animated avatar using [LiveAvatar by HeyGen](https://www.liveavatar.com/).

## Usage

* Update the environment:

```bash
# LiveAvatar Config
export LIVEAVATAR_API_KEY="..."
export LIVEAVATAR_AVATAR_ID="..."

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
python examples/avatar_agents/liveavatar/agent_worker.py dev
```
