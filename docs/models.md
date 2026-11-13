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

The classes take `extra_kwargs=`, the provider's own options; `inference.STT` and
`inference.TTS` also take `fallback=`, models the gateway falls back to (`FallbackModels`: a
`FallbackModel`, a model string, or a list of them, each a `FallbackModelType`). Options change through the provider's `update_options()`,
which also reaches its open streams (an STT stream gets them as a mid-stream update); streams
have no `update_options` of their own. Their typed forms:

| Module | Model strings | Provider options |
| --- | --- | --- |
| `inference.llm` | `LLMModels`: `OpenAIModels`, `GoogleModels`, `KimiModels`, `DeepSeekModels`, `ZAIModels`, `XAIModels`; realtime `RealtimeModels`: `OpenAIRealtimeModels`, `XAIRealtimeModels` | `ChatCompletionOptions` (its `reasoning_effort` a `ReasoningEffort`), `ProviderEndpoint` |
| `inference.stt` | `STTModels`: `AssemblyAIModels`, `CartesiaModels`, `DeepgramModels`, `DeepgramFluxModels`, `GoogleModels`, `InworldModels`, `OpenAIModels`, `SpeechmaticsModels`, `XaiModels`, `AutoModels`; `STTLanguages`, `STTEncoding` | `AssemblyaiOptions`, `CartesiaOptions`, `DeepgramOptions`, `DeepgramFluxOptions`, `GoogleOptions`, `InworldOptions`, `OpenAIOptions`, `SpeechmaticsOptions`, `XaiOptions` |
| `inference.tts` | `TTSModels`: `AssemblyAIModels`, `CartesiaModels`, `DeepgramModels`, `FishAudioModels`, `InworldModels`, `RimeModels`, `XaiModels`; `TTSEncoding` | `AssemblyaiOptions`, `CartesiaOptions`, `DeepgramOptions`, `FishAudioOptions`, `InworldOptions`, `RimeOptions`, `XaiOptions` |
| `inference` | `VADModels`; `TurnDetectorModels`, `TurnDetectorVersions` | `LemonSliceOptions`, for `AvatarSession` |

`inference.LLM(model, provider=, inference_class=, endpoint=)` takes the provider the
gateway routes to (the gateway picks by default), the `InferenceClass` of its requests, and a
`ProviderEndpoint(base_url=, api_key=)`: an OpenAI-compatible provider called directly with
its own key in place of the gateway. Its stream is a `GatewayLLMStream`.
`inference.STT(model=, vad=)` takes an optional model: left out or `"auto"`
(`AutoModels`), the gateway picks. `vad=` is a VAD whose end of speech finalizes the transcript,
for Speechmatics and OpenAI. Its stream is a `SpeechStream`.

`inference.TurnDetector()` is an end-of-turn model for
`VoiceSession(turn_handling={"turn_detection": ...})`.
On a call with adaptive interruption, the session runs an
`inference.AdaptiveInterruptionDetector`, which reports `voice.events.OverlappingSpeechEvent`s and fails
with `InterruptionDetectionError`. A realtime model's `inference_class` is an
`InferenceClass`: `"priority"`, `"standard"` or `"low"`.

`llm.FallbackAdapter`, `stt.FallbackAdapter`, `tts.FallbackAdapter` and
`llm.RealtimeModelFallbackAdapter` try a list of models in order; each emits an
`AvailabilityChangedEvent` (`llm.realtime.RealtimeAvailabilityChangedEvent` for the realtime
adapter) when a model goes down or comes back. `tts.StreamAdapter` and
`stt.StreamAdapter` give a non-streaming TTS or STT a streaming interface. An adapter retries on
its own (a chain its members, a stream adapter each sentence or utterance), so a request over it
runs once and hands it the whole `conn_options`. Each member attempt the chain moves past is a
recoverable `error` event on that member; the attempt it gives up after is that member's
non-recoverable one, and the adapter emits its own once.

### Sentences and pacing

`tts.synthesize(text)` and `tts.stream()` are the framework's for every TTS: the text goes through
the TTS's stage (see [sentences for the voice](lexer.md#sentences-for-the-voice)) and each attempt
is retried within `conn_options`; each retried attempt is a recoverable `error` event and the error
the request raises is its one final event. `tts.options` is a read-only `TTSOptions`, changed with
`tts.update_options(sentence_tokenizer=, pacing=)`, which every TTS takes beside its provider's
own options. A change reaches the TTS's next synthesis wherever the TTS was handed (a session, a
`StreamAdapter`, a `FallbackAdapter`); one already open keeps what it started with. The options
set how:

- `sentence_tokenizer`: what splits the text, `tokenize.SentenceTokenizer()` by default. The
  built-in runs natively with its settings; a subclass that overrides `tokenize` or `stream` runs
  through Python, its sentences joined until none ends inside a tag and cut at the provider's
  limit. A provider that takes words is sent words whatever this is.
- `pacing`: a `SentenceStreamPacer`, or `None` to send each chunk as soon as it is split. A
  TTS without streaming (sent through `StreamAdapter`, one request per batch) is paced by default,
  for more context per request; a streaming one is not. With a pacer the first chunk goes at once; each later batch waits until
  the synthesized audio stopped growing and at most `min_remaining_audio` seconds of it are left
  to play, then whole chunks are joined with a space up to `max_text_length` characters, never
  past the provider's request limit. It synthesizes less that an interruption throws away; on a
  streaming TTS the provider sits idle between batches, and one that ends an idle stream
  (Cartesia through LiveKit Inference) fails a long reply.

`tokenize.SentenceTokenizer(min_words=4, clause_words=None, retain_format=False,
max_token_len=None)` batches sentences until a chunk holds `min_words` spoken
words (a Chinese or Japanese character is a word, Thai about every three characters), never past
`max_token_len` characters; a longer sentence is cut at its last clause mark or word boundary
before it, and with no cap set at 1000 characters, so no text waits on a sentence that never ends.
Its stream emits a chunk as soon as no text to come could change where it ends, and emits exactly
the chunks `tokenize` returns for the whole text. With `clause_words`, a sentence running past that
many words breaks at its next clause mark (`,` `;` `:` and their fullwidth forms, never inside a
number such as `1,000`).

`tts.StreamAdapter(tts=)` gives a non-streaming TTS a streaming interface: each chunk is
synthesized on its own with the wrapped TTS, retried within `conn_options` while it has delivered
no audio, as one segment whose last frame is final. Its `options` are the wrapped TTS's, and its
`update_options` changes them. A `tts.FallbackAdapter`'s members each prepare their text by their
own options: its `options` are the serving member's, and its `update_options` sets every member's.

A plugin implements one provider attempt: `_synthesize(chunk, conn_options=)` for a whole text,
and `_stream(conn_options=)` when the service synthesizes text as it arrives. Its stream's `input`
yields a `tts.SpeechChunk` per chunk, `text` in the provider's syntax and `said` the words it
says, and a `Flush` at each segment's end. A provider that buffers text itself passes
`text_unit="words"` in its `TTSCapabilities`; one with a request size limit passes `max_text_len=`.

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

A fallback adapter's `attempt_timeout` caps each limit of a backend attempt;
it does not limit the duration of a healthy stream. The operation's
`total_timeout` still bounds retries and fallback together.

## Errors and streams

`APIError`, `APIStatusError`, `APIConnectionError`, `APITimeoutError` and
`StreamStateError` are exported from `livekit.agents`. Status errors retain
`status_code`, `request_id` and the diagnostic `body`. A timeout is a
connection error; its `timeout` names the limit that expired: `"pool"`, `"connect"`,
`"write"`, `"read"` or `"total"`, or `None` when the failing layer cannot identify it. Provider
errors keep their class and diagnostic cause through an agent run.

`retryable` permits retry consideration; it does not establish safe replay.
Delivered output is not replayed on failure without a deduplication protocol.
A TTS stream (`tts.stream()`) takes text through `push_text()`, `flush()` and
`end_input()`, and speaks every flushed segment once; `aclose()` cancels it
without ending its input. Provider cancellation ends
the request without retry or a provider-health penalty.

After its input ends, an inference STT stream completes when the provider
acknowledges the end, or after a quiet period: 3 seconds after a final
transcript, 30 seconds after the input ended or after a later interim
transcript. A transcript that arrives after the quiet period is dropped.

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
| `livekit.agents.llm` | `LLM`, `LLMStream`, `ChatChunk` (its `usage` is a `TokenUsage`), `ChoiceDelta`, `FunctionToolCall`, `LLMError`; realtime: `RealtimeModel`, `RealtimeSession`, `RealtimeCapabilities`, `RealtimeError`, `RealtimeModelError` and its events (`llm.realtime.EventTypes`: `InputSpeechStartedEvent`, `InputSpeechStoppedEvent`, `InputTranscriptionCompleted`, `GenerationCreatedEvent` with its `MessageGeneration`s, `RemoteItemAddedEvent`, `RealtimeSessionReconnectedEvent`); duplex (`llm.duplex`): `DuplexModel`, `DuplexSession`, `DuplexCapabilities`, `DuplexRealtimeAdapter` (a duplex model as a realtime model), `DuplexAudioFrame`, `DuplexOutputTranscriptDelta`, `DuplexInputGap`/`DuplexInputMute`, `AudioGate`/`FixedGate`; tools as providers read them: `Tool`, `ToolContext`, `ToolChoice`, `FunctionTool`, `RawFunctionTool`, `ProviderTool`, `is_function_tool`, `is_raw_function_tool` |
| `livekit.agents.stt`, `livekit.agents.tts` | `STT`, `RecognizeStream`, `SpeechEvent` (`SpeechEventType`, `SpeechData` alternatives, `RecognitionUsage`), `STTCapabilities`, `STTError`; `TTS`, `ChunkedStream`, `SynthesizeStream`, `SynthesizedAudio`, `AudioEmitter`, `TTSCapabilities`, `TTSError`, `SentenceStreamPacer`, `split_expr_markup`; each module's `FallbackAdapter` and `StreamAdapter` |
| `livekit.agents` | `Plugin`, `VAD`, `VADStream`, `VADEvent`, `VADEventType`, `VADCapabilities`, `LanguageCode`, `get_job_context`, `NOT_GIVEN`/`NotGivenOr`/`NotGiven`, `DEFAULT_API_CONNECT_OPTIONS`, `create_api_error_from_http`, `redacted_cause`, `ProviderAttempts` (a stream's `_core_attempts`, see [provider attempts](#provider-attempts)) |
| `livekit.agents.tokenize` | `SentenceTokenizer` and `WordTokenizer` (the built-in splitters; subclass one and override `tokenize` for a custom one), `SentenceStream`, `WordStream`, `TokenData`, `TokenSpan`, `BufferedSentenceStream`, `BufferedWordStream`, `split_words`, `replace_words` |
| `livekit.agents.utils` | `ConnectionPool`, `shortuuid`, `is_given`, `log_exceptions`, `EventEmitter`; `aio`: `Chan`, `ChanClosed`, `ChanEmpty`, `Interval`, `interval`, `cancel_and_wait`; `media`: `AudioByteStream`, `AudioStreamDecoder`, `encode` with `EncodeOptions`/`ResizeOptions` |
| `livekit.agents.http` | the client: `HttpSession`, `HttpResponse`, `FormData`, `WebSocket`, `WebSocketMessage`, `WebSocketMessageType`, `http_session`, `open`; see [HTTP and WebSockets](#http-and-websockets) |
| `livekit.agents.voice.avatar` | `AvatarSession` (its join wait, metrics and close: [Avatars](voice-session.md#avatars)), `AvatarError` (the recoverable `"error"` of an avatar that did not join in time), `DataStreamAudioOutput` (native, with the whole `io.AudioOutput` surface, so a Python stage can wrap it); an avatar replaces the call's audio with `session.output.replace_audio_tail(sink)`. The avatar worker's side: `AvatarRunner`, `AvatarOptions`, `VideoGenerator`, `AudioReceiver`/`DataStreamAudioReceiver`, `AudioSegmentEnd`, `QueueAudioOutput`; it publishes on the agent's behalf (`livekit.agents.types.ATTRIBUTE_PUBLISH_ON_BEHALF`) |
| `livekit.agents.types` | the TTS audio frame `userdata` keys: `USERDATA_TIMED_TRANSCRIPT` (the frame's `TranscriptChunk` words) and `USERDATA_TTS_STARTED_TIME` |

An OpenAI-compatible chat-completions provider (Groq, Cerebras, Baseten and the like)
subclasses `livekit.plugins.openai.LLM` with its `base_url` and key: its requests run on
the core's client with the provider's own key and no LiveKit header, not on the
`openai` SDK. Its instance options are the request's kwargs; a call's
`tool_choice`, `parallel_tool_calls` and `response_format` override them, so the
framework can forbid tools on a turn.

An STT or TTS declares every fact fixed at construction in its `STTCapabilities` or
`TTSCapabilities`, passed to the base constructor (`super().__init__(capabilities=...)`);
`update_options` never changes one. An LLM has no such facts: everything it does is chosen
per request.

Usage follows one rule: a figure the provider reports wins over one the framework
measures. An LLM's tokens come only from the provider, on `ChatChunk.usage`.

A TTS plugin whose provider reports what it billed passes it on with
`output_emitter.report_usage(usage, characters_count=...)`: tokens as a
`TokenUsage`, billed characters as a count, each report one provider request's
share of the open segment. It rides on the segment's final `SynthesizedAudio`
(`usage`, `characters_count`) into `TTSMetrics` and the usage totals, the billed
characters replacing the ones the framework counts; see
[usage](usage.md#text-to-speech). A provider that reports nothing needs no call.

A streaming STT's usage is measured by the framework, whether the stream was
opened by a session or directly: it reports the audio each attempt takes from
`input` as `RECOGNITION_USAGE` every 5 seconds of audio and when the attempt
ends, failed or not. Since it cannot take back what it reported, a
plugin whose provider reports what it billed says so up front: it passes
`reports_usage=True` in its `STTCapabilities` and emits its own `RECOGNITION_USAGE`.

A realtime session declares the audio its `push_audio` takes,
`super().__init__(model, input_sample_rate=24000, input_num_channels=1)`, and
the framework converts what it pushes. Elsewhere `utils.media.AudioConverter(rate,
channels)` does the same: `push(frame)` and `flush()` return frames in that format.
A TTS plugin passes G.711 bytes as `audio/pcmu` or `audio/pcma`.

`ChoiceDelta.extra` and `FunctionToolCall.extra` carry provider data (such as
Google thought signatures) as a JSON object: a value that is not JSON fails the
stream when the framework reads the chunk.

A plugin keeps the cause of the errors it raises, redacted:
`raise APIConnectionError(...) from redacted_cause(error)` chains the original
error's type and traceback without its message, which can carry credentials or
provider data. The provider's own message goes on the error's `detail=`: the
framework logs it once under `lk.pii.provider_detail` and never shows it in the
error's text, so a plugin never logs it before raising.

### HTTP and WebSockets

A plugin talks to its provider through the client in `livekit.agents.http`, never
through an HTTP library directly. The same package holds the server side, the routes an
application serves (`AgentRoutes`, see [Server](server.md)); neither side imports the
other. Every call runs under the `conn_options` it is given
(`DEFAULT_API_CONNECT_OPTIONS` when omitted): each limit applies to the wait it
names, and failures are framework errors, never the transport's exceptions.

| Name | Contract |
| --- | --- |
| `http_session()` | The job's shared `HttpSession`, created on first use and closed with the job; `RuntimeError` outside a job |
| `open()` | `async with http.open() as session:` binds one outside a job (tests, scripts); inside a job it yields the job's |
| `HttpSession(proxy=None)` | A session of its own; `await session.aclose()` closes it. A plugin's `http_session=` argument takes one. The HTTP library underneath is internal: nothing accepts or returns its objects. `session.with_request_id(header)` shares its connections and puts the response's `header` on `APIStatusError.request_id` |
| `session.request(method, url, *, conn_options=, headers=, params=, json=, data=)`, `get`, `post`, `put`, `patch`, `delete` | `async with session.post(...) as response:`; `data` is `bytes`, `str`, a `FormData` or an async iterable of `bytes` |
| `HttpResponse` | `status`, `ok`, `reason`, `headers`, `content_type`; `await read()`, `text()`, `json()` (a `JsonValue`); `async for chunk in response.iter_bytes()`, `async for line in response.iter_lines()`; `await raise_for_status()` raises `APIStatusError` carrying the status, body and `retry_after` (the `Retry-After` seconds) |
| `FormData` | `add_field(name, value, *, filename=, content_type=)`, a multipart body |
| `await session.ws_connect(url, *, conn_options=, headers=, params=, heartbeat=, protocols=)` | A `WebSocket`; a rejected handshake raises `APIStatusError` |
| `WebSocket` | `headers` (the upgrade response's), `closed`, `close_code`; `await send_str()`, `send_bytes()`, `send_json()`, `close(code=1000)`; `await receive(timeout=None)` returns a `WebSocketMessage`; `async for message in ws` yields until the socket closes; `raise ws.close_error(message)` for a socket the provider closed: final when its code refuses the request or the client (policy, auth, invalid data), else retryable, the close reason as its `detail` |
| `await retry_request(request, *, conn_options=)` | Runs `request()` (one request and the parsing of its answer, such as starting an avatar session) under `conn_options`' retries: 5xx, 429 and connection failures are retried after at least `Retry-After`; a request whose answer timed out is never sent again |
| `WebSocketMessage` | `type` (`WebSocketMessageType.TEXT`, `BINARY` or `CLOSE`), `data` (`str` text, `bytes` binary, the close reason), `json()` |

A transport failure raises `APIConnectionError`; a transport error on a socket
raises it from `receive()`. A limit that expires raises `APITimeoutError` naming
it in `timeout`: `pool` while waiting for a connection, `connect` for DNS, TCP and
TLS, `write` while a request body or frame waits for the transport, `read` while
a response or handshake is expected and when `receive(timeout=)` expires. A
non-2xx response is returned, so the plugin can read its body.

Every client in the process trusts the same CA certificates: `run_app` exports
the system's paths, or certifi's bundle when there are none, as
`SSL_CERT_FILE`/`SSL_CERT_DIR`, which Python, the native clients and
`livekit-rtc` read.

### Provider attempts

A plugin's `LLMStream`, `ChunkedStream`, `SynthesizeStream` and
`RecognizeStream` implement `_run`: **one attempt** at the request. The
framework runs it; the plugin never retries, sleeps between retries or
enforces `conn_options` itself.

- **When `_run` is called.** Once when the stream starts, then once per retry,
  each time in a fresh task. Nothing from a failed attempt is kept: an LLM
  attempt sends its chunks again, a TTS attempt gets a fresh `AudioEmitter`, a
  `SynthesizeStream` retry reads `input` again from its first token, and a
  `RecognizeStream` retry reads the audio the failed attempt had not taken, with
  `start_time`/`start_time_offset` moved on so timestamps stay linear.
- **What is retried.** An `APIError` with `retryable=True` (the default for
  connection errors and retryable statuses), up to `max_retry` more attempts
  (the first after 0.1 seconds, then `retry_interval` apart), all within
  `total_timeout`. Any other exception, or a
  non-retryable `APIError`, ends the stream with that same exception. For LLM and TTS a
  status 499 stops the request quietly. Once retries are exhausted the stream raises
  `APIConnectionError` whose `__cause__` is the last attempt's error.
- **Delivered output is final.** After an LLM chunk with content or a tool
  call reached the reader, or TTS audio was pushed, a failure is not retried: the caller already
  has partial output. A recognition retries regardless; transcripts already
  sent stay delivered.
- **Errors as events.** Each failed attempt that is retried is emitted as the
  provider's `error` event with `recoverable=True`; the error that ends the
  stream is emitted with `recoverable=False`: the very exception the stream
  raises, or, once retries are exhausted, the last attempt's error (the raised
  `APIConnectionError`'s `__cause__`). Inside a session the session reports
  what ends the stream.
- **Deadlines.** `read_timeout` runs only while a response is owed and
  restarts on progress. Report it through `self._core_attempts` (a
  `livekit.agents.ProviderAttempts`): `expect_response(final=False)` when the
  plugin sent something that owes an answer (`final=True` for its last input),
  `progress()` when response data arrived (a heartbeat is not progress),
  `audio_received()` for synthesized audio, `complete_response()` when nothing
  is owed any more. STT streams report through `send()` and the input channel
  automatically. When a deadline passes, the attempt fails like a retryable
  failure; one still running 50 ms later is cancelled (an attempt already
  finishing on its own is left to finish its cleanup).
- **Cancellation.** `_run` is cancelled when a deadline passes or the stream is
  closed (`aclose()`). Clean up (close sockets, release connections) and let
  `CancelledError` propagate; the stream counts as closed only after that
  cleanup, waited for at most a bounded teardown time, after which it is
  reported as stalled and keeps running.
- **Inside a session.** The session owns retries and fallback: it opens the
  stream again itself, so there `_run` runs exactly once per stream, still
  under its deadlines. Exactly one layer retries a request, and each retry is
  logged once as a warning
  (`"<provider>: attempt N failed, retrying in Xs: <error>"`, keyed `lk.retry.subject`,
  `lk.retry.attempt` and `lk.retry.delay`).

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

## Realtime options over the OpenAI protocol

`openai.realtime.RealtimeModel` and the providers speaking its protocol (`with_azure()`,
`xai.realtime.RealtimeModel`, `azure.realtime.RealtimeModel` for Voice Live and
`inference.RealtimeModel`) take their session options as records from
`livekit.agents.llm._realtime`, re-exported by each plugin's `realtime` module; plain dicts
and OpenAI SDK objects are refused. An option not given is the provider's default; `None`
turns it off. Every entry point takes every option its provider supports; an option a
provider does not offer is absent from its model rather than ignored. `update_options()` takes
the same records and reaches every live session; it omits only what cannot change
mid-session: `modalities`, which fixes the session's audio output, and `max_session_duration`,
both set at construction.

| Option | Records | Not offered by |
| --- | --- | --- |
| `voice` | a voice name, `CustomVoice(id=)`, Voice Live's `AzureVoice(name=, locale=)` | |
| `modalities` | `["text"]` or `["audio"]`; constructor only | xAI (always audio and text) |
| `tool_choice` | a `ToolChoice` | xAI |
| `input_audio_transcription` | `AudioTranscription(model=, language=, prompt=)` | |
| `input_audio_noise_reduction` | `"near_field"`, `"far_field"`, Voice Live's `AzureDeepNoiseSuppression()` | xAI |
| `turn_detection` | `ServerVad(...)`, `SemanticVad(eagerness=)`, Voice Live's `AzureSemanticVad(...)` | |
| `speed` | a playback speed multiplier | |
| `tracing` | `Tracing(workflow_name=, group_id=, metadata=)`; nothing set is the server's defaults | xAI |
| `truncation` | `"auto"`, `"disabled"`, `RetentionRatio(retention_ratio=, post_instructions_token_limit=)` | xAI, Voice Live |
| `reasoning` | `Reasoning(effort=)` | Voice Live |
| `max_response_output_tokens` | a count; `None` is unlimited; Voice Live's `max_output_tokens` | xAI |
| `max_session_duration` | a `timedelta` after which the connection is recycled; `None` never; constructor only | |

```python
from livekit.plugins import openai

model = openai.realtime.RealtimeModel(
    voice="marin",
    input_audio_transcription=openai.realtime.AudioTranscription(model="gpt-4o-mini-transcribe"),
    turn_detection=openai.realtime.SemanticVad(eagerness="low"),
    reasoning=openai.realtime.Reasoning(effort="low"),
)
```

Selecting Azure (`azure_deployment=` or `entra_token=`) reads `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_ENDPOINT` and defaults to `whisper-1` transcription and server VAD.

## Realtime context updates over the OpenAI protocol

Realtime sessions over the OpenAI realtime protocol (OpenAI, Azure, xAI and LiveKit Inference)
retain superseded conversation snapshots until admitted cleanup workers release them. Preparation pauses when cleanup needs capacity. Interruption, close,
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
