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

Set `BOSON_API_KEY` in your environment. Get a key at
[boson.ai/workspace/api-key](https://www.boson.ai/workspace/api-key).

```python
from livekit.agents import AgentSession
from livekit.plugins import boson


session = AgentSession(llm=boson.realtime.RealtimeModel())
```

The defaults connect to the hosted API (`wss://api.boson.ai/v1/realtime`) with
the `higgs-realtime` model and the `default` voice. To point at another
deployment, pass `url=` — and `api_key=None` if it does not authenticate.

## Model options

`boson.realtime.RealtimeModel` accepts these options:

| Option | Description |
| --- | --- |
| `url` | Realtime WebSocket endpoint. Defaults to `wss://api.boson.ai/v1/realtime`. `http`/`https` URLs are normalized to `ws`/`wss`; the path is passed through unchanged. |
| `api_key` | Sent as an `Authorization: Bearer ...` header. Omit it to read `BOSON_API_KEY` from the environment; omitting both raises `ValueError`. Pass `None` explicitly to send no header, for a local dev server without auth. |
| `model` | Sent as `session.model`. Defaults to `"higgs-realtime"`. |
| `voice` | Sent as `session.audio.output.voice`. |
| `instructions` | Sent as `session.instructions`. |
| `output_modalities` | `["audio"]` by default. `["text"]` is also supported for text-only responses. |
| `temperature` | Sent as `session.temperature`. |
| `max_output_tokens` | Sent as `session.max_output_tokens`. |
| `tool_choice` | Sent as `session.tool_choice`. |
| `speed` | Sent as `session.audio.output.speed`. Not currently supported for output audio; kept for wire compatibility. |
| `turn_detection` | Sent as `session.audio.input.turn_detection`. |
| `input_audio_transcription` | Sent as `session.audio.input.transcription`. |
| `input_audio_noise_reduction` | Sent as `session.audio.input.noise_reduction`. |
| `truncation` | `"auto"` (default) or `"disabled"`. Sent as `session.truncation`. `"auto"` enables background context summarization where the model supports it. Do not pass `None` — it is rejected and ends the session. |
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

`create_response` and `interrupt_response` are accepted for OpenAI wire
compatibility but are not configurable here: with server VAD enabled, a
response is always created after user speech and the active response is always
cancelled when new speech is detected. The plugin's own duplicate
`response.cancel` suppression follows that behavior (keyed on whether server
VAD is enabled at all), not on what these fields are set to.

The dict is sent to the server as given, so `{"type": "semantic_vad", ...}` works
too; the plugin does not filter or rewrite its keys.

Passing `turn_detection=None`/`False` disables server-side turn detection and
switches to client-driven `input_audio_buffer.commit`. This is exposed for
interface completeness but isn't a heavily-exercised path yet — treat it as
experimental rather than a recommended default.

`input_audio_transcription` can be passed as a full dict, or built with:

- `input_audio_transcription_model` (e.g. `"higgs-stt-3.1"`)
- `input_audio_transcription_language`

User transcription is enabled only when the transcription config contains a
non-empty `model`. A transcription `prompt` is not supported; the plugin drops
it even if present in a raw `input_audio_transcription` dict.

### `generate_reply()` overrides

`generate_reply(instructions=...)` sends the instructions in `response.create`.
The server applies them to that response alone: they replace the session
instructions for the turn, which still answers from the conversation so far.
Because it is a replacement rather than an addition, the event carries the
session instructions and the per-response ones together.

`tools` and `tool_choice` are **not** applied per response — the server accepts
and ignores them. `generate_reply()` therefore ignores them too (with a warning)
rather than sending values that would appear to take effect. The framework
scopes them at the session level around the call instead, since the plugin
reports `per_response_tool_choice = False`; to scope them from your own code,
use `update_tools()` and `update_options(tool_choice=...)` directly.

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

Only function tools have a shape on the wire. A tool of any other kind is left
out of `session.update` — the model never sees it — and the plugin logs a
warning once per session saying how many were dropped.

The plugin does not retry a WebSocket close that a reconnect cannot fix: an
invalid API key (close code 3000) or a billing entitlement refusal (close code
4429, e.g. `insufficient_quota`/`monthly_cap_reached`) both end the session
instead of retrying. Server errors that reflect an expected client/server race
rather than a real failure (`response_not_active`, `response_id_mismatch`,
`voice_output_task_ongoing`, `invalid_previous_item_id`) are logged but not
surfaced as recoverable `error` events.

The server has no insert-at-head primitive for conversation items
(`previous_item_id: null` always means append-at-tail). If `update_chat_ctx()`
needs to insert a new item ahead of turns the server already has (e.g.
prepending a context summary), the plugin deletes and recreates the entire
remote conversation in the target order instead of silently misordering it by
appending the new item at the tail.

That rebuild can only put back what it can express as text. A user turn whose
transcript has not arrived yet has none, and no audio is kept client-side, so it
is deleted along with the rest and not recreated — the server answers without it
until a later `update_chat_ctx()` recreates it (at the tail) once its transcript
exists. The plugin logs a warning naming the affected item ids whenever this
happens.

## Limitations

- The plugin expects the current realtime event names, such as
  `response.output_audio.delta` and `response.output_text.delta`.
- Video input is not supported yet.
- Mixed `["text", "audio"]` output modalities are not supported; choose either
  `["audio"]` or `["text"]`.
- This plugin sends and receives 24 kHz PCM audio. That's the format this
  integration uses, not the full range the Higgs Realtime API supports.
- `system`/`developer`-role chat items are not supported (the server's
  conversation store only accepts `assistant`/`user` items); they're silently
  dropped when syncing `update_chat_ctx()`. Use `instructions` for persistent
  directives instead.
