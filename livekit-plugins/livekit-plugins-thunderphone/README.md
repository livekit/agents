# ThunderPhone plugin for LiveKit Agents

Run a [ThunderPhone](https://thunderphone.com) voice agent as the realtime
(speech-to-speech) model of a LiveKit Agents session. ThunderPhone handles
speech recognition, the language model, the voice, turn-taking, 47 languages
and tools; LiveKit handles the room, the transport and the telephony.

See the docs at https://docs.livekit.io/agents/models/realtime/thunderphone/ for
setup and examples.

## Installation

```bash
pip install livekit-plugins-thunderphone
```

## Pre-requisites

A ThunderPhone secret API key (`sk_live_...`), set as `THUNDERPHONE_API_KEY`.

## Usage

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import thunderphone

# a saved ThunderPhone agent: prompt, voice, engine, languages and tools live on ThunderPhone
session = AgentSession(llm=thunderphone.RealtimeModel(agent_id=12))

# ...or configure the session inline: instructions and tools come from the LiveKit Agent
session = AgentSession(llm=thunderphone.RealtimeModel(model="bolt", voice="olivia"))
await session.start(agent=Agent(instructions="You are Acme Dental's receptionist."), room=ctx.room)
```

An inline call fixes its instructions and tools when it starts. A handoff to an `Agent` with
different instructions or tools therefore ends that ThunderPhone call and starts a new one with
the new configuration. With a saved agent, handoffs keep the call.

Every reply uses the session's instructions and tools (the saved agent's, or the inline
`Agent`'s). ThunderPhone has no per-reply overrides, so `generate_reply(instructions=...)`
replies normally and the plugin logs a warning. Framework features that steer a single reply
with instructions, such as the beta workflows, do not steer it on ThunderPhone.
