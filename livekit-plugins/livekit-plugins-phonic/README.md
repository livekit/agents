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
| `default_language` | `str` | ISO 639-1 default language for recognition and speech |
| `additional_languages` | `list[str]` | Further ISO 639-1 codes (must not repeat `default_language`) |
| `multilingual_mode` | `"auto"` \| `"request"` | Per-utterance language detection vs. change on user request (recommended: `request`) |
| `audio_speed` | `float` | Audio playback speed |
| `phonic_tools` | `list[str]` | [Phonic Webhook tool](https://docs.phonic.co/docs/using-tools/tools_overview#webhook-tools) names available to the assistant |
| `boosted_keywords` | `list[str]` | Keywords to boost in speech recognition |
| `min_words_to_interrupt` | `int` | Minimum number of user words required to interrupt the assistant |
| `generate_no_input_poke_text` | `bool` | Auto-generate poke text when user is silent |
| `no_input_poke_sec` | `float` | Seconds of silence before sending poke message |
| `no_input_poke_text` | `str` | Poke message text (ignored when `generate_no_input_poke_text` is True) |
| `no_input_end_conversation_sec` | `float` | Seconds of silence before ending conversation |
| `websocket_timeout_sec` | `int` | Seconds of inactivity before the Phonic websocket is closed |
| `intelligence_level` | `"standard"` \| `"high"` | LLM intelligence level |
| `is_welcome_message_interruptible` | `bool` | When False, the welcome message cannot be interrupted |
| `vad_prebuffer_duration_ms` | `int` | Voice-activity-detection prebuffer duration (ms) |
| `vad_min_speech_duration_ms` | `int` | Minimum speech duration for VAD (ms) |
| `vad_min_silence_duration_ms` | `int` | Minimum silence duration for VAD (ms) |
| `vad_threshold` | `float` | Voice-activity-detection threshold |
| `enable_assistant_backchannel` | `bool` | When True, the assistant backchannels (e.g. "mm-hmm") while the user speaks |
| `assistant_backchannel_aggressiveness` | `float` | How aggressively the assistant backchannels (needs `enable_assistant_backchannel`) |
| `pronunciation_dictionary` | `list[PronunciationEntry]` | `{ word, pronunciation }` entries; words must be unique |
| `template_variables` | `dict[str, str]` | Variables substituted into the system prompt and welcome message |
| `enable_redaction` | `bool` | Redact PII/PHI from transcripts and bleep it from audio after the conversation |
| `mcp_servers` | `list[str]` | Names of pre-configured MCP servers to make available (must be unique) |
| `observability_integrations` | `list["braintrust"]` | Observability integrations to forward traces to |
| `configuration_endpoint` | `ConfigurationEndpoint` \| `None` | Endpoint the agent calls to fetch per-conversation configuration |
| `additional_params` | `dict[str, Any]` | Additional runtime parameters forwarded to Phonic |
| `configs_for_tools` | `list[PhonicToolConfig]` | Per-tool behavior overrides (see [Per-tool configuration](#per-tool-configuration)) |

### Per-tool configuration

`configs_for_tools` takes one entry per tool you want to customize. Each entry is keyed by the tool `name`; every other field is optional and falls back to the plugin default when omitted. Tools with no entry keep the defaults.

```python
RealtimeModel(
    configs_for_tools=[
        {"name": "transfer_call", "forbid_speech_after_tool_call": True},
        {"name": "submit_form", "forbid_tool_call_after_speech": True},
    ],
)
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | — | Tool this config applies to (required) |
| `require_speech_before_tool_call` | `bool` | `False` | Require the agent to speak before the tool can be called |
| `forbid_speech_after_tool_call` | `bool` | `False` | Suppress the auto-generated spoken reply after the tool. Use for tools that always hand off to another agent (a non-handoff tool set here would leave the agent silent) |
| `forbid_tool_call_after_speech` | `bool` | `False` | Drop the tool call if the agent already spoke this turn |

The plugin always sends tool calls with `wait_for_speech_before_tool_call` on and `allow_tool_chaining` off; these are not configurable per tool.

> **Deprecated:** the top-level `forbid_speech_after_tool_call: list[str]` option still works but is deprecated — it now folds each listed tool into `configs_for_tools` as `forbid_speech_after_tool_call=True` (an explicit `configs_for_tools` entry wins) and logs a warning. Prefer `configs_for_tools`.

If you already have an agent set up on the Phonic platform, you can use the `phonic_agent` option to specify the agent name. As a note, configuration options you set in the LiveKit Agents SDK will override the agent settings set on the Phonic platform. This means the system prompt you have set on the Phonic platform will be ignored in favor of the `instructions` field set on the LiveKit `Agent`. Likewise, options explicitly set in the `RealtimeModel` constructor will override the Phonic agent's settings.

If you have Webhook tools set up on the Phonic platform, you can use `phonic_tools` to make them available to your agent. Only [Phonic Webhook tools](https://docs.phonic.co/docs/using-tools/tools_overview#webhook-tools) are supported with LiveKit Agents.
