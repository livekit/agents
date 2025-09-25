# LiveKit AvatarTalk Avatar Agent

This example demonstrates how to create a animated avatar using [AvatarTalk](https://avatartalk.ai/).

## Usage

* Update the environment:

```bash
# AvatarTalk Config
export AVATARTALK_API_KEY="..."
export AVATARTALK_API_URL="..."
export AVATARTALK_AVATAR="..."
export AVATARTALK_EMOTION="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/avatartalk/agent_worker.py dev
```
