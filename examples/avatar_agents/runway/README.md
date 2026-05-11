# LiveKit Runway Avatar Agent

This example demonstrates how to create an animated avatar using [Runway Characters](https://dev.runwayml.com/).

## Usage

- Update the environment:

```bash
# Runway Config
export RUNWAYML_API_SECRET="..."
export RUNWAY_AVATAR_PRESET_ID="..."  # or RUNWAY_AVATAR_ID for a custom avatar

# Google config (or other models, tts, stt)
export GOOGLE_API_KEY="..."

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

- Start the agent worker:

```bash
python examples/avatar_agents/runway/agent_worker.py dev
```

