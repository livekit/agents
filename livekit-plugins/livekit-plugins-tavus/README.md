# Tavus plugin for LiveKit Agents

Support for the [Tavus](https://tavus.io/) virtual avatar.

See the [Tavus integration docs](https://docs.livekit.io/agents/models/avatar/plugins/tavus/) for more information.

## Installation

```bash
pip install livekit-plugins-tavus
```

## Pre-requisites

You'll need an API key from Tavus. It can be set as an environment variable: `TAVUS_API_KEY`

## Avatar emotion

Once the avatar has joined the room, set its facial emotion with `set_emotion`. The avatar keeps it until the next call.

`start()` returns before the avatar joins the room, and an emotion sent before then is dropped, so wait for the avatar with `wait_for_join()` first.

```python
avatar = tavus.AvatarSession(face_id="...")
await avatar.start(session, room=ctx.room)
await avatar.wait_for_join()

await avatar.set_emotion("happiness")
```

Supported emotions: `neutral`, `happiness`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `contempt`.

To let the LLM choose the emotion, wrap it in a function tool:

```python
@function_tool
async def set_emotion(context: RunContext, emotion: tavus.Emotion) -> None:
    """Set the facial emotion of your avatar."""
    await avatar.set_emotion(emotion)
```
