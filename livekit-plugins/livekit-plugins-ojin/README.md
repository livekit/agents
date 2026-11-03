# Ojin plugin for LiveKit Agents

Add Ojin avatars to LiveKit Agents. The plugin uses the agent's speech to
animate the avatar and publishes its audio and video on the agent's participant.

For a complete browser-based example, see the
[Ojin LiveKit demo integration guide](https://github.com/ojinai/livekit-demo/blob/main/docs/livekit-integration-guide.md).

## Installation

```bash
pip install livekit-plugins-ojin
```

## Prerequisites

Set these environment variables, replacing the placeholders with your values:

```shell
export OJIN_API_KEY="your-ojin-api-key"
export OJIN_CONFIG_ID="your-persona-config-id"
```

See the [Ojin docs](https://docs.ojin.ai/) for information about Ojin accounts
and personas.

## Usage

In your agent's Python entrypoint (usually `agent.py`), add the Ojin session
inside your LiveKit room handler. Keep your existing STT, LLM, and TTS setup in
`AgentSession`, and start the avatar before the agent session because the
avatar supplies its audio output.

```python
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession
from livekit.plugins import ojin

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext) -> None:
    session = AgentSession(
        # Configure STT, LLM, and TTS here.
    )
    avatar = ojin.AvatarSession()

    await avatar.start(session, room=ctx.room)
    await session.start(
        agent=Agent(instructions="You are a helpful voice assistant."),
        room=ctx.room,
    )
```

The video track is published on the agent participant, not on a separate avatar
participant. Set `OJIN_WS_URL` only when you need to override the Ojin endpoint;
otherwise the SDK's default endpoint is used.
