# LiveKit Akool Avatar Agent

This example demonstrates how to create a animated avatar using [Akool](https://akool.com/).

create your avatar [here](https://akool.com/apps/upload/avatar?from=%2Fapps%2Fstreaming-avatar%2Fedit)

## Usage

* Update the environment:

```bash
# Akool Config
export AKOOL_CLIENT_ID="..."
export AKOOL_CLIENT_SECRET="..."

# OpenAI config (or other models, tts, stt)
export OPENAI_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/akool/agent_worker.py dev
```
