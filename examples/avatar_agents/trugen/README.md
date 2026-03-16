# LiveKit TruGen.AI Realtime Avatar

This example demonstrates how to create a trugen realtime avatar session for your Livekit Voice Agents using [TruGen Developer Studio](https://app.trugen.ai/).

Select your avatar [list](https://docs.trugen.ai/docs/avatars/overview)

## Usage

* Update the environment:

```bash
# TruGen Config
export TRUGEN_API_KEY="..."

# Google config (or other models, tts, stt)
export GOOGLE_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/trugen/agent_worker.py dev
```
