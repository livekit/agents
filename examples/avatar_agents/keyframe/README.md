# LiveKit Keyframe Labs Avatar Agent

This example demonstrates how to create an avatar using [Keyframe Labs](https://keyframelabs.com/).

See available personas [here](https://platform.keyframelabs.com).

## Usage

* Update the environment:

```bash
# Keyframe Config (use either PERSONA_ID or PERSONA_SLUG, not both)
export KEYFRAME_API_KEY="..."
export KEYFRAME_PERSONA_ID="..."
# or: export KEYFRAME_PERSONA_SLUG="public:luna"

# LiveKit config
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
export LIVEKIT_URL="..."
```

* Start the agent worker:

```bash
python examples/avatar_agents/keyframe/agent_worker.py dev
```

## Emotion Control

You can change the avatar's facial expression at runtime using `set_emotion()`:

```python
await avatar.set_emotion("happy")  # "neutral", "happy", "sad", "angry"
```

To let the LLM control the avatar's expression, register it as a tool:

```python
@function_tool()
async def set_avatar_emotion(emotion: keyframe.Emotion):
    """Set the avatar's facial expression and demeanor."""
    await avatar.set_emotion(emotion)
```
