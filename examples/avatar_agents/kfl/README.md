# LiveKit Keyframe Labs (KFL) Avatar Agent

This example demonstrates how to create an avatar using [Keyframe Labs](https://keyframelabs.com/).

See available personas [here](https://platform.keyframelabs.com).

## Usage

* Update the environment:

```bash
# KFL Config (use either PERSONA_ID or PERSONA_SLUG, not both)
export KFL_API_KEY="..."
export KFL_PERSONA_ID="..."
# or: export KFL_PERSONA_SLUG="public:luna"

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/kfl/agent_worker.py dev
```
