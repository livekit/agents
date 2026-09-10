# ThunderPhone plugin for LiveKit Agents

Run a [ThunderPhone](https://thunderphone.com) voice agent as the realtime
(speech-to-speech) model of a LiveKit Agents session. ThunderPhone handles
speech recognition, the language model, the voice, turn-taking, 47 languages
and tools; LiveKit handles the room, the transport and the telephony.

See the docs at https://thunderphone.com/docs/guides/use-with-livekit for
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
session = AgentSession(llm=thunderphone.RealtimeModel(product="bolt", voice="olivia"))
await session.start(agent=Agent(instructions="You are Acme Dental's receptionist."), room=ctx.room)
```
