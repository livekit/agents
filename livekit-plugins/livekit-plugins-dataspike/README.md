# Dataspike Deepfake Detection Plugin for LiveKit Agents

This plugin integrates [Dataspike](https://dataspike.io/) with LiveKit Agents to provide **real-time deepfake detection**.  
It enables detection of synthetic or manipulated media during live or recorded video streams.

## Installation

```bash
pip install livekit-plugins-dataspike
```

## Prerequisites

You’ll need a **Dataspike API key**. Set it as an environment variable before running your agent:

```bash
export DATASPIKE_API_KEY="your_api_key_here"
```

## Usage Example

```python
from livekit.plugins import dataspike
from livekit.agents import AgentSession, Agent

async def entrypoint(ctx):
    await ctx.connect()
    session = AgentSession(...)
    detector = dataspike.DataspikeDetector()
    await detector.start(session, room=ctx.room)
    await session.start(agent=Agent(instructions="Talk to me!"), room=ctx.room)
```
 
## Links

- [Dataspike API](https://docs.dataspike.io/api)
- [LiveKit Agents SDK](https://github.com/livekit/agents)
