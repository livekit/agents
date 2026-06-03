# LiveKit D-ID Avatar Agent

This example demonstrates how to create an animated avatar using [D-ID](https://www.d-id.com/).

## Usage

* Update the environment:

```bash
# D-ID Config
export DID_API_KEY="..."
export DID_AGENT_ID="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/did/agent_worker.py dev
```
