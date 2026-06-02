# LiveKit Spatius Avatar Agent

This example demonstrates how to create an animated avatar using
[Spatius](https://www.spatius.ai/).

## Prerequisites

Set up your LiveKit credentials and Spatius avatar credentials:

```bash
export LIVEKIT_URL="wss://your-livekit-host"
export LIVEKIT_API_KEY="your-livekit-api-key"
export LIVEKIT_API_SECRET="your-livekit-api-secret"

export SPATIUS_API_KEY="your-spatius-api-key"
export SPATIUS_APP_ID="your-spatius-app-id"
export SPATIUS_AVATAR_ID="your-spatius-avatar-id"
```

Optional region configuration:

```bash
export SPATIUS_REGION="us-west"
```

## Run

```bash
python examples/avatar_agents/spatius/agent_worker.py dev
```
