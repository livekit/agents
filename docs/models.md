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
`ProviderEndpoint(base_url=, api_key=, provider=)`: an OpenAI-compatible provider called directly
with its own key in place of the gateway, reported under the vendor name `provider=` gives. Its stream is a `GatewayLLMStream`.
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

A provider failure names what failed with its `kind`, and the kind alone decides
`retryable`: `"rate_limit"`, `"server"`, `"connection"` and `"timeout"` are retried;
`"auth"`, `"quota"`, `"invalid_request"` and `"protocol"` are final. `"protocol"` means the
provider broke its own contract: an answer or frame whose fields the plugin reads are missing
or have the wrong type, or an answer that does not parse; the offending payload is its
`detail`. A plugin fails only on what it reads: unknown message types and extra fields are
ignored. `APIError(message, kind=...)`
requires one. `APIStatusError(message, status_code=...)` takes its status's kind (401 and
403 `auth`, 402 `quota`, 408 `timeout`, 429 `rate_limit`, other 3xx and 4xx
`invalid_request`, anything else `server`) unless `kind=` names the provider's own meaning.
`APIConnectionError` is `"connection"` unless a close code says otherwise, and
`APITimeoutError` is `"timeout"`. `ErrorKind` is exported from `livekit.agents.types`.

`retryable` permits retry consideration; it does not establish safe replay.
Delivered output is not replayed on failure without a deduplication protocol.
A TTS stream (`tts.stream()`) takes text through `push_text()`, `flush()` and
`end_input()`, and speaks every flushed segment once; `aclose()` cancels it
without ending its input. Provider cancellation ends
the request without retry or a provider-health penalty.

After its input ends, an inference STT stream drains for at most 30 seconds.
After a final transcript, 3 seconds without another transcript completes the drain early;
empty interims do not restart that quiet period. Finalize acknowledgements do not complete
the drain: they are uncorrelated and some providers acknowledge before sending their finals.
A transcript arriving after the drain ends is dropped.

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
| `livekit.agents.stt`, `livekit.agents.tts` | `STT`, `RecognizeStream`, `SpeechEvent` (`SpeechEventType`, `SpeechData` alternatives, `RecognitionUsage`), `STTCapabilities`, `STTError`; `TTS`, `ChunkedStream`, `SynthesizeStream`, `SynthesizedAudio`, `AudioEmitter`, `TTSCapabilities`, `TTSError`, `SentenceStreamPacer`, `split_expr_markup`; each module's `FallbackAdapter` and `StreamAdapter`; a socket provider's `WebSocketStream` and its names (see [Provider sockets](#provider-sockets)) |
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

A model's `provider` is its vendor's name, declared by its class or constructor, and never the
endpoint's host: one vendor has one name whichever plugin reaches it
(`openai.LLM.with_x_ai()` and `livekit.plugins.xai` both report `xAI`), and Azure OpenAI is
`azure_openai`. An OpenAI or Anthropic model given a `base_url` or a client without a vendor
reports `"unknown"`. Telemetry spells a vendor the OpenTelemetry GenAI registry names as the
registry does (`x_ai`, `azure.ai.openai`).

An STT or TTS declares every fact fixed at construction in its `STTCapabilities` or
`TTSCapabilities`, passed to the base constructor (`super().__init__(capabilities=...)`);
`update_options` never changes one. An LLM has no such facts: everything it does is chosen
per request.

Usage follows one rule: a figure the provider reports wins over one the framework
measures. An LLM's tokens come only from the provider, on `ChatChunk.usage`.

A TTS plugin whose provider reports what it billed passes it on with
`output_emitter.report_usage(usage, characters_count=...)` (a WebSocket stream reports
`tts.SegmentUsage(segment_id, usage=, characters_count=)` from `on_message` instead); a provider's
request id reaches the audio with `output_emitter.provider_request_id(request_id)`: tokens as a
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

### Provider sockets

The bases speak WebSocket only; a provider over gRPC, HTTP/2 or its own SDK writes its stream
directly. A provider reached over a WebSocket subclasses its kind's socket base, sets a few class
attributes and writes three methods: `connect` (a `WebSocketTarget`, called again at each
connection), `encode` (one frame per input item: `bytes`, or an `Outbound` TypedDict sent as
JSON; `None` only where an item sends nothing) and `on_message` (each received message, decoded
once from the `Inbound` TypedDicts the class declares, reported through the kind's `report`; it
returns nothing). `decode` overrides the decoding for a binary codec; `protocol_error` is the
error for a frame that breaks the protocol. Plugins branch on items with `match` and
`assert_never`, and on tagged messages with `if message["type"] == ...` and `assert_never`.

The end of the input is the kind's end item, which `encode` turns into the provider's goodbye
(`case stt.EndOfAudio(): return _Out(type="CloseStream")`). The base then waits for the
provider's reply only when the connection sent input the kind says is owed an answer (STT: its
transcripts; TTS: a segment still open), bounded by the read limit. A protocol declares
`completion="session_finished"` when its documentation promises an explicit terminal response,
or `completion="peer_closed"` when the server flushes results before closing the socket.
An earlier final transcript cannot replace either required completion. The default
`completion="settled"` uses bounded transcript settlement where no terminal guarantee is known.
An STT protocol
that tracks outstanding item IDs overrides `nothing_pending()` so a completed item cannot
stand in for other pending transcripts. The framework runs the socket
in the core: the job's proxy, pings, the `keepalive=Keepalive(every=, frame=)` frame, the `setup`
frame and `awaits_ready` (nothing else goes until `on_message` reports the kind's `Ready`, within
the connect limit), close codes, `reconnect(reason)`, retries and the bounded close.
`max_session_duration` replaces the connection at the first moment the stream has nothing in
progress (STT: no pending recognition; TTS: no open segment; realtime: no reply being generated). A
provider that ends its session (the kind's `SessionFinished`, a segment's `SegmentDone` with
`connection_per_segment`, a clean close after the stream asked for it, or a clean close while
nothing it was sent is unanswered) gets a new connection when more input comes; a clean close
with something still owed fails the attempt. `WebSocketTarget.max_message_size` raises the
largest message the provider may send (bytes). A keepalive is a covariant `Keepalive`, so a
frame of one of the protocol's messages fits its `Outbound` union. A stream's input-idle timer
(`text_idle`, `audio_idle`) runs on a multiplexed stream as on its own socket. Tests advance the
host clock with `_core.skip_host_time`; when one timer arms another, await the first timer's
observable result before advancing through the next deadline.
`send_now(message, first=False)` sends a control message on the open
connection, ahead of input not yet sent; one sent while no connection is open is dropped unless
sent from that connection's `connect`. A command queued after `reconnect()` is requested waits
for the replacement connection. The provider's request id, read from the handshake
(`WebSocketTarget.request_id_header`) or reported, is the stream's `request_id`, and every error
the stream raises carries it. `pacing=WebSocketPacing(min_interval=, real_time=)` spaces the
frames (real time by the duration the kind computes); one instance on several streams paces them
together. Each kind's module exports the shared names its plugins write: `WebSocketTarget`,
`Keepalive`, `protocol_error`, and `WebSocketMultiplexer` and `WebSocketPacing` for STT and TTS.

| Base | `encode` takes | `report` takes | Its own class attributes |
| --- | --- | --- | --- |
| `stt.WebSocketStream` | `rtc.AudioFrame` (re-chunked to `frame_ms`), `stt.Flush`, `stt.EndOfAudio` | `Transcript`, `SpeechStarted`, `SpeechEnded`, `stt.Ready`, `stt.SessionFinished` | `frame_ms`; `provider_vad`; `new_session_per_flush`: the provider ends its session at each flush, the audio after it waits for that, then goes on a new connection; `audio_idle` with `on_audio_idle()`: a frame sent once the audio stayed silent. A transcript naming no request id carries the stream's |
| `tts.WebSocketStream` | `SegmentStart`, `SegmentText`, `SegmentEnd`, `tts.EndOfText` | `SegmentAudio`, `SegmentWords`, `SegmentUsage` (what the provider billed for the segment), `SegmentDone` (also from `encode`, for a provider that answers each text on its own), `tts.Ready`, `tts.ProviderRequestId` | `connection_per_segment`: each segment its own session, ended by its `SegmentDone` or the provider's close; `text_idle` with `on_text_idle()`; `audio_idle` with `on_audio_idle()`; `mime_type` |
| `llm.RealtimeWebSocketSession` | `rtc.AudioFrame` from `push_audio` (no end item) | `llm.realtime.Ready` (its events it emits itself) | `frame_ms`. A due replacement waits until no reply it requested or the provider started is still generating |
| `llm.duplex.DuplexWebSocketSession` | `rtc.AudioFrame`, `DuplexInputGap`, `DuplexInputMute` | `llm.duplex.Ready`, `llm.duplex.SessionFinished`; the plugin emits its session events | `frame_ms`; `goodbye()` asks the provider to finish before replacement or close |
| `voice.avatar.AvatarWebSocket` | `bytes` or the protocol's outbound message, queued with `send()` | The plugin handles received messages and calls `ready()` | `awaits_ready`; `open()` waits for readiness, `run()` propagates failure; no retry |

A TTS's `audio_idle` timer waits until response data has arrived, then measures silence from
the later of the latest input and response. Received audio, words and completion reports restart
it; control messages and unknown frames do not. It fires once per silence, including on a pooled
socket. It does not replace the read timeout for a provider that never sends audio.

SmallestAI continuations use this timer because the protocol supplies no final-context marker.
Each segment uses its own socket because responses omit the context id. Completion inferred
from silence remains a heuristic: an audio gap longer than the interval can end a segment early.

`stt.WebSocketMultiplexer` and `tts.WebSocketMultiplexer` (one class) share sockets across streams:
`connect`, `route` (a received message's context), `ready` (whether a message says the socket's
session is ready, with `awaits_ready`), `setup`, `open_context`, `close_context` (the contexts a
stream left early), `max_contexts`, `max_session_duration`, `prewarm()` and `invalidate()`. A
message `route` decoded reaches its stream as decoded. A pool (`routed = False`) carries one
stream a socket, hands it every frame, and gives the socket to the next stream only after one that
ended cleanly; `max_sockets` caps the sockets (a stream waits for room within `pool_timeout`),
`idle_timeout` closes a socket no stream holds. Nothing keeps an open context alive while the
LLM is slow between sentences: a `SegmentDone` before the segment's text ended is the provider
ending its context early, so the segment is done with what it said, one warning is logged, and
the rest of its text opens a new segment (a new context). A pool whose multiplexer reads `context_ended(message)` (the context
a message ends) cancels a stream left mid-answer with `close_context` and gives the socket to the
next stream once the provider ended every cancelled context, within a second, else closes it; a
stream finding no room waits for that rather than opening another. `reuses_sockets = False` gives
each stream (each session, with `connection_per_segment`) a fresh socket, closed after it.
`keep_warm` keeps a socket with room open as a standby: one opens when a stream takes the last,
and an idle one the provider closed opens again, so a direct provider that wants `prewarm()` or a
warm standby for its next segment declares a single-use warm pool. `goodbye()` is the frame sent
on a socket the multiplexer closes (idle, replaced, spent or at `aclose()`), ahead of the close:
the provider session's own end. `Keepalive(..., only_while_unused=True)` goes only while no
stream holds the socket, keeping a parked one off the provider's idle timeout.

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
| `WebSocket` | `headers` (the upgrade response's), `closed`, `close_code`; `await send_str()`, `send_bytes()`, `send_json()`, `close(code=1000)`; `await receive(timeout=None)` returns a `WebSocketMessage`; `async for message in ws` yields until the socket closes; `raise ws.close_error(message, codes=None)` for a socket the provider closed: an `APIConnectionError` of the kind its close code names (policy and auth codes `auth`, refused data `invalid_request`, a server error `server`, a private 44xx code its HTTP status's kind, anything else `connection`), the provider's own close codes in `codes` (`{4001: "auth"}`) read first, the close reason as its `detail`; `http.close_kind(code, codes=None)` is the same kind for a socket a provider SDK owns |
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
- **What is retried.** An `APIError` whose kind is retryable, up to `max_retry` more attempts
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

An integration that owns its operation loop rather than using these attempt-based
stream classes applies `async with conn_options.total_deadline():` to bound the
operation and `conn_options.interval_for_retry(n)` for retry delays. It must
remain the request's sole retry owner; an attempt-based `_run` never adds that loop.

### Connection pools

`livekit.agents.utils.ConnectionPool` takes `conn_options=` for prewarming.
Its `connect_cb` receives the resolved `APIConnectOptions` and applies the
appropriate transport phases. Call `get(conn_options=...)` or use
`async with pool.connection(conn_options=...)` for each operation. The
context manager returns a healthy connection to the pool and retires it on
failure. `await pool.aclose()` releases the pool's connections.

## Realtime options over the OpenAI protocol

`openai.realtime.RealtimeModel` (`OpenAIRealtimeModel` in `livekit.agents.llm.openai_protocol`)
and the providers speaking its protocol (each a subclass of `OpenAIProtocolModel`) (`with_azure()`,
`xai.realtime.RealtimeModel`, `azure.realtime.RealtimeModel` for Voice Live and
`inference.RealtimeModel`) take their session options as records from
`livekit.agents.llm.openai_protocol`, re-exported by each plugin's `realtime` module; plain dicts
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
