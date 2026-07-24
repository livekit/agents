# Boson plugin for LiveKit Agents

This plugin lets LiveKit Agents use the Boson realtime WebSocket API as a
LiveKit `RealtimeModel`.

LiveKit owns the room connection, participant audio, interruption plumbing,
transcript IO, and worker lifecycle. This plugin adapts LiveKit Agents realtime
model calls to Boson's OpenAI-compatible realtime protocol subset.

## Installation

```bash
pip install livekit-plugins-boson
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import boson


session = AgentSession(
    llm=boson.realtime.RealtimeModel(
        url="wss://api.example.com/v1/realtime/",
        api_key="...",
        model="Qwen2.5-72B-Instruct",
        voice="default",
    )
)
```

## Model options

`boson.realtime.RealtimeModel` accepts these options:

| Option | Description |
| --- | --- |
| `url` | Realtime WebSocket endpoint. `http`/`https` URLs are normalized to `ws`/`wss`. |
| `api_key` | Sent as an `Authorization: Bearer ...` header. |
| `model` | Sent as `session.model`. |
| `voice` | Sent as `session.audio.output.voice`. |
| `instructions` | Sent as `session.instructions`. |
| `modalities` | `["audio"]` by default. `["text"]` is also supported for text-only responses. |
| `temperature` | Sent as `session.temperature`. |
| `max_output_tokens` | Sent as `session.max_output_tokens`. |
| `tool_choice` | Sent as `session.tool_choice` and per-response `tool_choice`. |
| `speed` | Sent as `session.audio.output.speed`. |
| `turn_detection` | Sent as `session.audio.input.turn_detection`. |
| `input_audio_transcription` | Sent as `session.audio.input.transcription`. |
| `input_audio_noise_reduction` | Sent as `session.audio.input.noise_reduction`. |
| `query_params` | Extra query parameters added to the WebSocket URL. |

If `turn_detection` is omitted, the plugin sends a default server VAD config:

```python
{
    "type": "server_vad",
    "create_response": True,
    "interrupt_response": True,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "threshold": 0.55,
}
```

Pass `turn_detection=None` to disable server-side turn detection from the plugin.

`input_audio_transcription` can be passed as a full dict, or built with:

- `input_audio_transcription_model`
- `input_audio_transcription_language`
- `input_audio_transcription_prompt`

User transcription is enabled only when the transcription config contains a
non-empty `model`.

## Protocol compatibility

The plugin translates LiveKit realtime model operations to the following client
events:

- `session.update`
- `input_audio_buffer.append`
- `input_audio_buffer.commit`
- `input_audio_buffer.clear`
- `conversation.item.create`
- `conversation.item.delete`
- `conversation.item.truncate`
- `response.create`
- `response.cancel`

The plugin handles Boson response audio, response text, audio transcript, input
transcription, function call, interruption, and error events and maps them back
to LiveKit realtime model streams.

Function tools registered on the LiveKit `Agent` are sent in `session.update`.
When Boson returns a `function_call` item, LiveKit Agents executes the Python
tool locally and the plugin sends the result back as a
`function_call_output` conversation item.

## Limitations

- The plugin expects the current realtime event names, such as
  `response.output_audio.delta` and `response.output_text.delta`.
- Video input is not supported yet.
- Mixed `["text", "audio"]` output modalities are not supported; choose either
  `["audio"]` or `["text"]`.
