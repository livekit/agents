# LiveKit Plugins: Phonic

Realtime voice AI integration for [Phonic](https://phonic.co/) with LiveKit Agents.

## Installation

```bash
uv add livekit-plugins-phonic
```

## Usage

```python
import asyncio
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
)
from livekit.plugins.phonic.realtime import RealtimeModel

logger = logging.getLogger("phonic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant named Sabrina.",
            llm=RealtimeModel(
                voice="sabrina",
                audio_speed=1.2,
            ),
        )

    @function_tool(
        description="Toggle a light on or off. Available lights are A05, A06, A07, and A08."
    )
    async def toggle_light(self, light_id: str, state: str) -> str:
        """Called when the user asks to toggle a light on or off.

        Args:
            light_id: The ID of the light to toggle
            state: Whether to turn the light on or off, e.g., 'on', 'off'
        """
        logger.info(f"Turning {state} light {light_id}")
        await asyncio.sleep(1.0)
        return f"Light {light_id} turned {state}"


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession()
    await session.start(agent=MyAgent(), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user, asking about their day.",
    )


if __name__ == "__main__":
    cli.run_app(server)
```

```bash
cd examples
uv run voice_agents/phonic_realtime_agent.py dev
```

## Configuration

Set the `PHONIC_API_KEY` environment variable, or pass `api_key` directly to `RealtimeModel`. All other options are optional.

| Option | Type | Description |
| --- | --- | --- |
| `api_key` | `str` | Phonic API key. Falls back to `PHONIC_API_KEY` environment variable |
| `phonic_agent` | `str` | Phonic agent name. Options set explicitly here override agent settings |
| `voice` | `str` | Voice ID — `sabrina`, `grant`, `virginia`, `landon`, `eleanor`, `shelby`, `nolan` |
| `welcome_message` | `str` | Message the agent says when the conversation starts. Ignored when `generate_welcome_message` is True |
| `generate_welcome_message` | `bool` | Auto-generate the welcome message (ignores `welcome_message`) |
| `project` | `str` | Project name (default: `main`) |
| `languages` | `list[str]` | ISO 639-1 language codes the agent should recognize and speak |
| `audio_speed` | `float` | Audio playback speed |
| `phonic_tools` | `list[str]` | [Phonic Webhook tool](https://docs.phonic.co/docs/using-tools/tools_overview#webhook-tools) names available to the assistant |
| `boosted_keywords` | `list[str]` | Keywords to boost in speech recognition |
| `generate_no_input_poke_text` | `bool` | Auto-generate poke text when user is silent |
| `no_input_poke_sec` | `float` | Seconds of silence before sending poke message |
| `no_input_poke_text` | `str` | Poke message text (ignored when `generate_no_input_poke_text` is True) |
| `no_input_end_conversation_sec` | `float` | Seconds of silence before ending conversation |

If you already have an agent set up on the Phonic platform, you can use the `phonic_agent` option to specify the agent name. As a note, configuration options you set in the LiveKit Agents SDK will override the agent settings set on the Phonic platform. This means the system prompt you have set on the Phonic platform will be ignored in favor of the `instructions` field set on the LiveKit `Agent`. Likewise, options explicitly set in the `RealtimeModel` constructor will override the Phonic agent's settings.

If you have Webhook tools set up on the Phonic platform, you can use `phonic_tools` to make them available to your agent. Only [Phonic Webhook tools](https://docs.phonic.co/docs/using-tools/tools_overview#webhook-tools) are supported with LiveKit Agents.
