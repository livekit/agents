# LiveKit Plugins Alibaba

Agent Framework plugin for Alibaba Cloud DashScope Realtime Audio API (`qwen-audio-3.1-realtime-plus`).

## Installation

```bash
pip install livekit-plugins-alibaba
```

## Pre-requisites

Set `DASHSCOPE_API_KEY` in your environment:

```bash
export DASHSCOPE_API_KEY=your-api-key
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import alibaba

session = AgentSession(
    llm=alibaba.realtime.RealtimeModel(),
)
```

See the [runnable voice agent example](../../examples/voice_agents/alibaba_realtime.py).

## Configuration

- Model: `qwen-audio-3.1-realtime-plus`; voice: `longanqian`.
  Voice can be changed before audio starts; later changes are rejected locally.
- Region: `region="cn"` (Beijing) or `region="intl"` (Singapore). If omitted,
  `DASHSCOPE_REGION` is used, falling back to `cn`. Credentials and model access
  must match the selected endpoint.
- Endpoint precedence: explicit `base_url`, then `workspace_id` (or
  `DASHSCOPE_WORKSPACE_ID`) for a dedicated MaaS endpoint, then the region endpoint.
- Audio: mono PCM16, resampled to 16 kHz on input; 24 kHz on output.
- Turn detection: server VAD by default. Pass `turn_detection=None` for client
  turn-taking with local VAD. Both modes support tools, greetings and explicit
  `generate_reply()`. A proactive greeting needs an initial user message, e.g.
  `session.generate_reply(user_input="Please greet me.")`.
  Choose VAD settings at construction. Runtime VAD updates are rejected by the
  plugin because DashScope cannot apply them after audio begins.
  Custom server VAD supports `threshold`, `prefix_padding_ms`, and
  `silence_duration_ms`. Semantic VAD, idle timeouts, and disabling automatic
  response creation/interruption via VAD flags are rejected; use `None` for
  client turn-taking.
- Input transcription: `gummy-realtime-v1` by default; pass
  `input_audio_transcription=None` to disable it.
  During client turn-taking, an input buffer already started before a runtime
  update can still produce a transcript; verify the next complete turn.
- Tool registration, tool outputs, chat history, interruption and metrics use the
  shared OpenAI realtime implementation. Session-level `tool_choice` is supported.
- Non-null OpenAI-specific `speed`, `reasoning`, tracing, truncation, noise
  reduction and output-token-limit options raise `ValueError`.

Qwen Audio 3.1 does not echo metadata in our live probes and permits only one
generating response. The adapter uses a single pending slot (like the Google
plugin): the next generation after a request is sent satisfies its waiter. This
is not a guarantee of causal request correlation when VAD competes with a request.
No new create is transmitted while a request or response is active. Interrupt stops
local streams, but the remote slot stays occupied until `response.done`. Missing
creation/cancellation acknowledgement triggers reconnection after ten seconds.
Old requests are never replayed; conversation item IDs are never rewritten.
One replacement call may wait for an interrupted response's terminal event;
ordinary overlapping creates are rejected.

For optional local turn-taking, install `livekit-plugins-silero` and use:

```python
from livekit.plugins import silero

session = AgentSession(
    llm=alibaba.realtime.RealtimeModel(turn_detection=None),
    vad=silero.VAD.load(),
)
```

## Development verification

```bash
uv run pytest tests/test_plugin_alibaba_realtime.py --unit
uv run mypy -p livekit.plugins.alibaba
```
