# LiveKit Spatius Avatar Agent

This example demonstrates how to create an animated avatar using
[Spatius](https://www.spatius.ai/).

## Client-side rendering

Spatius renders the avatar in the frontend. A standard LiveKit video renderer will
show black frames because the avatar video track carries motion data rather than
server-rendered video. Use the [Spatius client SDK and LiveKit adapter](https://docs.spatius.ai/livekit-agents/client)
to render the avatar. The [reference frontend](https://github.com/spatius-ai/spatius-avatar-demo/tree/main/platform-integrations/livekit-agents-demo/livekit-agents-reference-demo/frontend)
contains a working integration.

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
