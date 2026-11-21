# MeshLite plugin for LiveKit Agents

Support for the [MeshLite](https://meshlite.io) avatar: a lip-synced face rendered in the visitor's own browser.

Unlike the other avatar plugins, MeshLite adds no participant and no video track to the room. The agent keeps its own STT, LLM and TTS (or realtime model). The plugin sends the agent's voice to the visitor's page over the avatar data-stream protocol (`lk.audio_stream`, `lk.clear_buffer`, `lk.playback_started`, `lk.playback_finished`), and the MeshLite web SDK on that page plays it in sync with the face. Interruptions and the played-back position work as with the other avatar plugins.

## Installation

```bash
pip install livekit-plugins-meshlite
```

## Pre-requisites

The agent side needs no MeshLite key. The visitor's page needs the MeshLite web SDK and a MeshLite session token that your server creates with your MeshLite secret key. See the [MeshLite LiveKit guide](https://meshlite.io/docs#livekit).

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import meshlite

session = AgentSession(stt=..., llm=..., tts=...)

avatar = meshlite.AvatarSession()  # the first standard participant, or visitor_identity="..."
await avatar.start(session, room=ctx.room)  # before session.start()

await session.start(agent=..., room=ctx.room)
```

On the page, connect with `livekit-client` and hand the room to the web SDK (`controller.attachLiveKit(room)`). Do not subscribe to an agent audio track: the voice arrives as a data stream.

## Notes

- `avatar_identity` is the visitor's identity, since the avatar lives in the visitor's page. `aclose()` never removes a participant from the room.
- Room recordings (egress) do not contain the agent's voice, because no audio track is published.
- A full example agent with its page: [meshlite-avatar/livekit-plugins-meshlite](https://github.com/meshlite-avatar/livekit-plugins-meshlite/tree/main/examples/browser-avatar).
