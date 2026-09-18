# 60db Agents Plugin

A [LiveKit Agents](https://github.com/livekit/agents) plugin for the [60db.ai](https://60db.ai) STT, TTS, and LLM APIs.

## Features

- **STT** — Streaming speech-to-text via WebSocket with interim results
- **TTS** — Streaming and chunked text-to-speech via WebSocket
- **LLM** — Chat completions with tool-call support via HTTP/SSE

## Installation

Requires Python >=3.10.

```bash
pip install livekit-plugins-60db
```

## Configuration

Pass your API key via `_60dbClient`:

```python
from livekit.plugins._60db import _60dbClient

client = _60dbClient("your-api-key")  # sets global default
```

Alternatively, set the `SIXTY_DB_API_KEY` environment variable (or a `.env.local` file).

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SIXTY_DB_API_KEY` | — | API key for all 60db services |
| `SIXTY_DB_STT_URL` | `wss://api.60db.ai/ws/stt` | STT WebSocket endpoint |
| `SIXTY_DB_TTS_URL` | `wss://api.60db.ai/ws/tts` | TTS WebSocket endpoint |
| `SIXTY_DB_LLM_URL` | `https://api.60db.ai/v1/chat/completions` | LLM HTTP endpoint |

Each service also accepts a direct `ws_url` (or `api_url`) constructor argument, which takes precedence over env vars.

## STT

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | global client / env var | API key |
| `ws_url` | `str \| None` | env var / `wss://api.60db.ai/ws/stt` | WebSocket endpoint URL |
| `languages` | `list[str] \| None` | `["en"]` | Language codes for recognition |
| `encoding` | `str` | `"mulaw"` | Audio encoding format |
| `sample_rate` | `int` | `8000` | Audio sample rate in Hz |
| `continuous_mode` | `bool` | `True` | Keep session open for continuous recognition |

### Example

```python
from livekit.plugins._60db import _60dbClient, STT

client = _60dbClient("your-api-key")  # sets global default
stt = STT()  # picks up api_key from client

async with stt.stream() as stream:
    # push audio frames into the stream
    stream.push_frame(audio_frame)

    async for event in stream:
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            print(event.alternatives[0].text)
```

### WebSocket Protocol

The STT service uses a bidirectional WebSocket connection for real-time streaming transcription.

#### Connection

```
wss://api.60db.ai/ws/stt?apiKey={API_KEY}
```

Connect with WebSocket keepalive enabled (`ping_interval=30`, `ping_timeout=10`, `max_size=10 MB`).

#### Handshake Sequence

1. **Client connects** to the WebSocket URL.
2. **Server sends** `connection_established`:

```json
{"connection_established": true}
```

3. **Client sends** `start` command:

```json
{
  "type": "start",
  "languages": ["en"],
  "config": {
    "encoding": "mulaw",
    "sample_rate": 8000,
    "continuous_mode": true
  }
}
```

4. **Server sends** `connected` acknowledgment:

```json
{"type": "connected"}
```

#### Audio Streaming

After the handshake, the client sends raw audio as **binary WebSocket frames**. Audio must match the `encoding` and `sample_rate` declared in the `start` message. The plugin automatically converts input PCM to the target format:

- Stereo-to-mono downmix (`audioop.tomono`)
- Sample rate resampling (`audioop.ratecv`)
- LINEAR16-to-mulaw encoding (`audioop.lin2ulaw`)

#### Transcription Messages

The server sends JSON text frames with transcription results:

```json
{
  "type": "transcription",
  "text": "Hello, world",
  "is_final": false,
  "language": "en"
}
```

| Field | Type | Description |
|---|---|---|
| `text` | `string` | Recognized text |
| `is_final` | `bool` | `true` for final results, `false` for interim/partial |
| `language` | `string` | Detected or configured language code |

#### Stop / Teardown

When the input stream ends, the client sends:

```json
{"type": "stop"}
```

The server responds with:

```json
{
  "type": "session_stopped",
  "billing_summary": {
    "total_cost": "..."
  }
}
```

#### Error Messages

```json
{"type": "error", "error": "description of the error"}
```

## TTS

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | global client / env var | API key |
| `ws_url` | `str \| None` | env var / `wss://api.60db.ai/ws/tts` | WebSocket endpoint URL |
| `voice_id` | `str` | `"fbb75ed2-975a-40c7-9e06-38e30524a9a1"` | Voice identifier |
| `encoding` | `str` | `"LINEAR16"` | Output audio encoding |
| `sample_rate` | `int` | `16000` | Output sample rate in Hz |

### Example

```python
from livekit.plugins._60db import _60dbClient, TTS

client = _60dbClient("your-api-key")  # sets global default
tts = TTS()  # picks up api_key from client

# Chunked synthesis
async for chunk in tts.synthesize("Hello, world!"):
    print(len(chunk.data))

# Streaming synthesis
async with tts.stream() as stream:
    stream.push_text("Hello, ")
    stream.push_text("world!")
    stream.end_input()
    async for chunk in stream:
        print(len(chunk.data))
```

### WebSocket Protocol

The TTS service uses a context-based WebSocket protocol. A single connection can host multiple contexts, but the plugin creates one context per synthesis request.

#### Connection

```
wss://api.60db.ai/ws/tts?apiKey={API_KEY}
```

Connect with WebSocket keepalive enabled (`ping_interval=30`, `ping_timeout=10`, `max_size=10 MB`).

#### Handshake and Context Creation

1. **Client connects** to the WebSocket URL.
2. **Server sends** `connection_established`:

```json
{"connection_established": true}
```

3. **Client sends** `create_context`:

```json
{
  "create_context": {
    "context_id": "unique-context-id",
    "voice_id": "fbb75ed2-975a-40c7-9e06-38e30524a9a1",
    "audio_config": {
      "audio_encoding": "LINEAR16",
      "sample_rate_hertz": 16000
    }
  }
}
```

4. **Server sends** `context_created`:

```json
{"context_created": true}
```

#### Sending Text

```json
{
  "send_text": {
    "context_id": "unique-context-id",
    "text": "Hello, world!"
  }
}
```

#### Flushing

To trigger audio generation and signal that no more text is coming for this batch:

```json
{
  "flush_context": {
    "context_id": "unique-context-id"
  }
}
```

#### Audio Response

The server returns audio as base64-encoded chunks:

```json
{
  "audio_chunk": {
    "audioContent": "<base64-encoded PCM audio>"
  }
}
```

Multiple `audio_chunk` messages may arrive for a single flush. After all audio is delivered:

```json
{"flush_completed": true}
```

#### Closing a Context

```json
{
  "close_context": {
    "context_id": "unique-context-id"
  }
}
```

Server responds:

```json
{"context_closed": true}
```

#### Chunked vs Streaming Mode

- **Chunked** (`synthesize`): Creates a context, sends all text, flushes once, collects all audio chunks, then closes the context. Suitable for single-shot synthesis.
- **Streaming** (`stream`): Creates a context, then runs send and receive tasks in parallel. Text segments are pushed incrementally; each flush triggers a segment boundary in the output. The context is closed when the input channel ends.

## LLM

### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | global client / env var | API key |
| `ws_url` | `str \| None` | env var / `https://api.60db.ai/v1/chat/completions` | Chat completions endpoint URL |
| `model` | `str` | `"qcall/slm-3b-int4"` | Model identifier |
| `top_k` | `int \| None` | `None` | Top-K sampling parameter |
| `chat_template_kwargs` | `dict \| None` | `None` | Extra template kwargs passed to the model |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `top_p` | `float \| None` | `None` | Nucleus sampling probability |
| `max_tokens` | `int \| None` | `None` | Maximum tokens in the response |

### Example

```python
from livekit.plugins._60db import _60dbClient, LLM
from livekit.agents import llm

client = _60dbClient("your-api-key")  # sets global default
model = LLM()  # picks up api_key from client

chat_ctx = llm.ChatContext()
chat_ctx.append(role="system", text="You are a helpful assistant.")
chat_ctx.append(role="user", text="What is the capital of France?")

async for chunk in model.chat(chat_ctx=chat_ctx):
    print(chunk.choices[0].delta.content, end="")
```

### HTTP API Protocol

The LLM service uses a standard OpenAI-compatible chat completions API with Server-Sent Events (SSE) streaming.

#### Endpoint

```
POST https://api.60db.ai/v1/chat/completions
```

#### Authentication

```
Authorization: Bearer {API_KEY}
Content-Type: application/json
```

#### Request Body

```json
{
  "model": "qcall/slm-3b-int4",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "tools": [],
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "max_tokens": 1024,
  "chat_template_kwargs": {}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | `string` | yes | Model identifier |
| `messages` | `array` | yes | OpenAI-format message list (`role`, `content`) |
| `tools` | `array` | no | OpenAI-format function tool schemas |
| `temperature` | `float` | no | Sampling temperature |
| `top_p` | `float` | no | Nucleus sampling probability |
| `top_k` | `int` | no | Top-K sampling parameter |
| `max_tokens` | `int` | no | Maximum tokens in response |
| `chat_template_kwargs` | `object` | no | Extra template kwargs for the model |

Only `model` and `messages` are required. All other fields are optional and included only when set on the `LLM` constructor.

#### SSE Response Format

The response is streamed as SSE. Each line is prefixed with `data: `:

```
data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" capital"},"finish_reason":null}]}
data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" of France is Paris."},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":8,"total_tokens":23}}
data: [DONE]
```

- Each chunk contains a `choices` array with `delta` objects.
- `delta.content` carries incremental text.
- `delta.tool_calls` may appear when the model invokes a tool (see below).
- `finish_reason` is `null` during streaming and `"stop"` (or `"tool_calls"`) on the final chunk.
- The stream terminates with `data: [DONE]`.
- The final data chunk may include a `usage` object with `prompt_tokens`, `completion_tokens`, and `total_tokens`.

#### Tool Call Handling

When the model invokes a tool, the response includes `tool_calls` in the delta. Tool calls are accumulated across multiple chunks:

**First chunk** (tool call start):
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "id": "call_abc123",
        "function": {"name": "get_weather", "arguments": ""}
      }]
    }
  }]
}
```

**Subsequent chunks** (argument fragments):
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "function": {"arguments": "{\"loc"}
      }]
    }
  }]
}
```

The plugin accumulates `function.name` and `function.arguments` across chunks. When a `finish_reason` appears or `[DONE]` is reached, the complete `FunctionToolCall` is emitted with the full concatenated `arguments` string.

#### Error Responses

Standard HTTP error codes are used:

| Status | Meaning |
|---|---|
| `401` | Invalid or missing API key |
| `429` | Rate limit exceeded |
| `500` | Server error |
| `503` | Service unavailable |

Error responses raise `APIConnectionError` with the HTTP status code and response body.

## Agent Example

A minimal `VoicePipelineAgent` wiring all three services together:

```python
import asyncio
from livekit.agents import VoicePipelineAgent
from livekit.plugins._60db import _60dbClient, STT, TTS, LLM

client = _60dbClient("your-api-key")  # sets global default


async def entrypoint(ctx):
    await ctx.connect()

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=STT(),
        llm=LLM(),
        tts=TTS(),
    )

    agent.start(ctx.room)


if __name__ == "__main__":
    from livekit.agents import cli
    from livekit.plugins import silero

    cli.run_app(worker_type=cli.WorkerType.ROOM, entrypoint_fnc=entrypoint)
```

## Audio Format Notes

### STT (Speech-to-Text)

| Parameter | Default | Supported Values |
|---|---|---|
| Encoding | `mulaw` | `mulaw`, `LINEAR16` |
| Sample rate | `8000 Hz` | Any rate (auto-resampled) |

The STT plugin automatically converts incoming audio to the configured format:
1. **Downmix** — stereo input is converted to mono via `audioop.tomono`.
2. **Resample** — input at any sample rate is resampled to the target rate via `audioop.ratecv`.
3. **Encode** — when `encoding="mulaw"`, LINEAR16 PCM is converted to mulaw via `audioop.lin2ulaw`.

### TTS (Text-to-Speech)

| Parameter | Default | Supported Values |
|---|---|---|
| Encoding | `LINEAR16` | `LINEAR16` |
| Sample rate | `16000 Hz` | Configurable via `sample_rate` parameter |

Audio is returned as base64-encoded PCM in `audio_chunk` messages.

## Error Handling

All three services raise the same set of exceptions from the LiveKit Agents framework:

| Exception | Cause |
|---|---|
| `APIConnectionError` | WebSocket connection/handshake failure, HTTP error response, or unexpected protocol message |
| `APITimeoutError` | Operation exceeds the configured timeout (defaults to `DEFAULT_API_CONNECT_OPTIONS`) |
| `ValueError` | Missing API key or service URL at construction time |

### Timeout Configuration

Timeouts are controlled via `APIConnectOptions` passed to each service method:

```python
from livekit.agents import APIConnectOptions

opts = APIConnectOptions(timeout=30.0)

# STT
async with stt.stream(conn_options=opts) as s:
    ...

# TTS
async for chunk in tts.synthesize("hello", conn_options=opts):
    ...

# LLM
async for chunk in model.chat(chat_ctx=ctx, conn_options=opts):
    ...
```

### Retry Guidance

- **STT WebSocket**: The server may send a `{"type": "connecting"}` status before `connection_established`. If an `error` type arrives during handshake, reconnect with a short delay.
- **TTS WebSocket**: Connection failures should be retried with exponential backoff.
- **LLM HTTP**: `httpx.TimeoutException` maps to `APITimeoutError`; `httpx.HTTPStatusError` maps to `APIConnectionError` with the status code.

## License

[Add license here]
