# Models and connection policy

LLM, STT, TTS and realtime providers accept the same connection policy.
Pass `conn_options=APIConnectOptions(...)` to a model operation or a provider
that accepts a default policy. An explicitly supplied policy replaces that
default in full.

## Model strings

`Agent(llm=)` and `VoiceSession(stt=, llm=, tts=)` take a model instance or a
LiveKit Inference model string: `"provider/model"`, with an optional
`:language` for STT and `:voice` for TTS. A string is
`inference.LLM(model)`, `inference.STT(model)` or `inference.TTS(model)` from
`livekit.agents.inference`; an `llm=` string naming a realtime model is its
realtime model. Construct the class directly for its other options. Provider
plugins (`livekit.plugins.<provider>`) supply the same interfaces.

```python
agent = Agent(instructions="...", llm="google/gemma-4-31b-it")
session = VoiceSession(stt="deepgram/nova-3:en", tts="cartesia/sonic-3:<voice>")
```

## Connection options

`APIConnectOptions` is immutable and exported from `livekit.agents`.
Durations are seconds. Each timeout accepts a finite nonnegative number or
`None` to disable that limit; invalid values raise `ValueError` before I/O.

| Field | Default | Bounds |
| --- | --- | --- |
| `pool_timeout` | `10.0` | Waiting for transport capacity or a pooled connection |
| `connect_timeout` | `10.0` | Establishing a usable connection, including DNS and TLS |
| `write_timeout` | `10.0` | Writing pending bytes without progress |
| `read_timeout` | `10.0` | Waiting without progress when a remote response is expected |
| `total_timeout` | `None` | The whole operation, including admission, retries and fallback |
| `max_retry` | `3` | Additional attempts after the first; a nonnegative integer |
| `retry_interval` | `2.0` | Delay between retries; the first retry waits 0.1 seconds |

There is no `timeout` field. Read and write limits restart on transport
progress. Waiting for application input, passive listening and downstream
backpressure do not count as remote inactivity. The total deadline starts
when the operation is accepted and does not restart on progress or retry.
Unsupported applicable timeout settings fail before I/O.

A fallback adapter's `attempt_timeout` caps each phase of a backend attempt;
it does not limit the duration of a healthy stream. The operation's
`total_timeout` still bounds retries and fallback together.

## Errors and streams

`APIError`, `APIStatusError`, `APIConnectionError`, `APITimeoutError` and
`StreamStateError` are exported from `livekit.agents`. Status errors retain
`status_code`, `request_id` and the diagnostic `body`. A timeout is a
connection error; its `phase` is `"pool"`, `"connect"`, `"write"`, `"read"`
or `"total"`, or `None` when the failing layer cannot identify it. Provider
errors keep their class and diagnostic cause through an agent run.

`retryable` permits retry consideration; it does not establish safe replay.
Delivered output is not replayed on failure without a deduplication protocol.
A TTS stream (`tts.stream()`) takes text through `push_text()`, `flush()` and
`end_input()`, and speaks every flushed segment once; `aclose()` cancels it
without ending its input. Provider cancellation ends
the request without retry or a provider-health penalty.

A concurrent pull of a provider stream raises `StreamStateError` without
disturbing its accepted pull. `aclose()` requests cancellation and waits for
bounded cleanup. A cleanup deadline leaves the framework responsible for
late resources and emits a diagnostic observation.

## Plugin authors

A provider plugin implements the model interfaces and registers itself with
`livekit.agents.Plugin`. These names are its contract; applications use the
plugin's classes, not these bases.

| Module | Names |
| --- | --- |
| `livekit.agents.llm` | `LLM`, `LLMStream`, `ChatChunk`, `ChoiceDelta`, `CompletionUsage`, `FunctionToolCall`, `LLMError`; realtime: `RealtimeModel`, `RealtimeSession`, `RealtimeCapabilities`, `RealtimeError`, `RealtimeModelError` and its events; duplex: `DuplexModel`, `DuplexSession`, `DuplexCapabilities`; tools as providers read them: `Tool`, `ToolContext`, `ToolChoice`, `FunctionTool`, `RawFunctionTool`, `ProviderTool`, `is_function_tool`, `is_raw_function_tool` |
| `livekit.agents.stt`, `livekit.agents.tts` | `STT`, `RecognizeStream`, `SpeechEvent`, `STTCapabilities`; `TTS`, `ChunkedStream`, `SynthesizeStream`, `AudioEmitter`, `TTSCapabilities`; each module's `FallbackAdapter` |
| `livekit.agents` | `Plugin`, `VAD`, `VADStream`, `VADEvent`, `VADEventType`, `LanguageCode`, `get_job_context`, `NOT_GIVEN`/`NotGivenOr`, `DEFAULT_API_CONNECT_OPTIONS`, `create_api_error_from_http` |
| `livekit.agents.utils` | `ConnectionPool`, `AudioBuffer`, `http_context`, `aio` |
| `livekit.agents.voice.avatar` | `AvatarSession`, `DataStreamAudioOutput`; an avatar replaces the call's audio with `session.output.replace_audio_tail(sink)` |

In a plugin's own retry loop, `async with conn_options.total_deadline():`
bounds the operation by `total_timeout`, and `conn_options.interval_for_retry(n)`
is the delay before retry `n`.

### Connection pools

`livekit.agents.utils.ConnectionPool` takes `conn_options=` for prewarming.
Its `connect_cb` receives the resolved `APIConnectOptions` and applies the
appropriate transport phases. Call `get(conn_options=...)` or use
`async with pool.connection(conn_options=...)` for each operation. The
context manager returns a healthy connection to the pool and retires it on
failure. `await pool.aclose()` releases the pool's connections.

## Native realtime context updates

Native realtime sessions retain superseded conversation snapshots until admitted cleanup
workers release them. Preparation pauses when cleanup needs capacity. Interruption, close,
incoming protocol events and already written response deadlines continue to run.

Additional `update_chat_ctx()` calls can exhaust the retained-context limit while cleanup
is waiting, including replacements of a pending target during reconnect. The session then
emits a nonrecoverable `RealtimeModelError` with message
`realtime context retirement capacity exceeded`, fails pending reply requests and closes.
Cleanup ownership remains with the framework. Coalesce successive history changes before
submitting a new full snapshot. Submitting an update does not await cleanup admission.

The limit bounds retained ownership, so the number of updates accepted depends on the
conversation and outstanding work. The runtime does not promise an unlimited burst of
context replacements. Ordinary replay of an unchanged conversation uses bounded
continuations; external item or context growth can still reach the overload limit.
