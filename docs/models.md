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
`inference.TTS` also take `fallback=`, models the gateway falls back to (a `FallbackModel` or a
model string, `FallbackModelType`). Their typed forms:

| Module | Model strings | Provider options |
| --- | --- | --- |
| `inference.llm` | `LLMModels`: `OpenAIModels`, `GoogleModels`, `KimiModels`, `DeepSeekModels`, `ZAIModels`, `XAIModels`; realtime `RealtimeModels`: `OpenAIRealtimeModels`, `XAIRealtimeModels` | `ChatCompletionOptions` |
| `inference.stt` | `STTModels`: `AssemblyAIModels`, `CartesiaModels`, `DeepgramModels`, `DeepgramFluxModels`, `GoogleModels`, `InworldModels`, `SpeechmaticsModels`, `XaiModels`; `STTLanguages`, `STTEncoding` | `AssemblyaiOptions`, `CartesiaOptions`, `DeepgramOptions`, `DeepgramFluxOptions`, `GoogleOptions`, `InworldOptions`, `SpeechmaticsOptions`, `XaiOptions` |
| `inference.tts` | `TTSModels`: `CartesiaModels`, `DeepgramModels`, `FishAudioModels`, `InworldModels`, `RimeModels`, `XaiModels`; `TTSEncoding` | `CartesiaOptions`, `DeepgramOptions`, `FishAudioOptions`, `InworldOptions`, `RimeOptions`, `XaiOptions` |
| `inference` | `VADModels`; `TurnDetectorModels`, `TurnDetectorVersions` | `LemonSliceOptions`, for `AvatarSession` |

`inference.TurnDetector()` is an end-of-turn model for `VoiceSession(turn_detection=)`.
On a call with adaptive interruption, the session runs an
`inference.AdaptiveInterruptionDetector`, which reports `OverlappingSpeechEvent`s and fails
with `InterruptionDetectionError`. A realtime model's `inference_class` is an
`InferenceClass`: `"priority"`, `"standard"` or `"low"`.

`llm.FallbackAdapter`, `stt.FallbackAdapter`, `tts.FallbackAdapter` and
`llm.RealtimeModelFallbackAdapter` try a list of models in order; each emits an
`AvailabilityChangedEvent` when a model goes down or comes back. `tts.StreamAdapter` and
`stt.StreamAdapter` give a non-streaming TTS or STT a streaming interface.

### Sentences and pacing

`tts.StreamAdapter(tts=, sentence_tokenizer=, text_pacing=)` splits the text pushed into
sentences and synthesizes each with the wrapped TTS's `synthesize()`, one after the other, as
one segment whose last frame is final. The split is blingfire's (`retain_format`, XML-aware on
an expressive stream) unless `sentence_tokenizer` is given: the framework then feeds the text
to its `stream()` and synthesizes what that stream yields, and closes it with the synthesis.
On an expressive stream the wrapped TTS's markup is normalized before the split and lowered in
each sentence. Each sentence is retried on its own within `conn_options` while it has
delivered no audio. Time to first audio is measured from the stream's first text.

`text_pacing=True` (or a `SentenceStreamPacer`) holds sentences back: the first goes at once;
each later batch waits until the synthesized audio stopped growing and at most
`min_remaining_audio` seconds of it are left to play (from the first audio), then whole
sentences are joined with a space up to `max_text_length` characters. A streaming plugin paces
its own tokenizer stream with `pacer.wrap(sentence_stream, output_emitter)`: a
`SentenceStream` whose text goes to `sentence_stream` and whose sentences are paced by the
audio `output_emitter` pushed; closing it closes `sentence_stream` and ends a pending read.

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
| `livekit.agents.llm` | `LLM`, `LLMStream`, `ChatChunk` (its `usage` is a `TokenUsage`), `ChoiceDelta`, `FunctionToolCall`, `LLMError`; realtime: `RealtimeModel`, `RealtimeSession`, `RealtimeCapabilities`, `RealtimeError`, `RealtimeModelError` and its events (`EventTypes`: `InputSpeechStartedEvent`, `InputSpeechStoppedEvent`, `InputTranscriptionCompleted`, `GenerationCreatedEvent` with its `MessageGeneration`s, `RemoteItemAddedEvent`, `RealtimeSessionReconnectedEvent`); duplex (`llm.duplex`): `DuplexModel`, `DuplexSession`, `DuplexCapabilities`, `DuplexRealtimeAdapter` (a duplex model as a realtime model), `DuplexAudioFrame`, `DuplexOutputTranscriptDelta`, `AudioGate`/`FixedGate`; tools as providers read them: `Tool`, `ToolContext`, `ToolChoice`, `FunctionTool`, `RawFunctionTool`, `ProviderTool`, `is_function_tool`, `is_raw_function_tool` |
| `livekit.agents.stt`, `livekit.agents.tts` | `STT`, `RecognizeStream`, `SpeechEvent` (`SpeechEventType`, `SpeechData` alternatives, `RecognitionUsage`), `STTCapabilities`, `STTError`, `SpeakerContext`; `TTS`, `ChunkedStream`, `SynthesizeStream`, `SynthesizedAudio`, `AudioEmitter`, `TTSCapabilities`, `MarkupInfo`, `TTSError`, `SentenceStreamPacer`; each module's `FallbackAdapter` and `StreamAdapter` |
| `livekit.agents` | `Plugin`, `VAD`, `VADStream`, `VADEvent`, `VADEventType`, `VADCapabilities`, `LanguageCode`, `get_job_context`, `NOT_GIVEN`/`NotGivenOr`/`NotGiven`, `DEFAULT_API_CONNECT_OPTIONS`, `create_api_error_from_http`, `redacted_cause`, `ProviderAttempts` (a stream's `_core_attempts`, see [provider attempts](#provider-attempts)) |
| `livekit.agents.tokenize` | `SentenceTokenizer`, `SentenceStream`, `WordTokenizer`, `WordStream`, `TokenData`, `BufferedSentenceStream`, `BufferedWordStream`, the `basic` and `blingfire` tokenizers, `replace_words` |
| `livekit.agents.utils` | `ConnectionPool`, `http_context`, `shortuuid`, `is_given`, `log_exceptions`, `EventEmitter`; `aio`: `Chan`, `ChanClosed`, `ChanEmpty`, `Interval`, `cancel_and_wait`; `media`: `AudioStreamDecoder`, `encode` |
| `livekit.agents.voice.avatar` | `AvatarSession`, `DataStreamAudioOutput`; an avatar replaces the call's audio with `session.output.replace_audio_tail(sink)`. The avatar worker's side: `AvatarRunner`, `AvatarOptions`, `VideoGenerator`, `AudioReceiver`/`DataStreamAudioReceiver`, `AudioSegmentEnd`, `QueueAudioOutput`; it publishes on the agent's behalf (`livekit.agents.types.ATTRIBUTE_PUBLISH_ON_BEHALF`) |
| `livekit.agents.types` | the TTS audio frame `userdata` keys: `USERDATA_TIMED_TRANSCRIPT` (the frame's `TranscriptChunk` words) and `USERDATA_TTS_STARTED_TIME` |

A TTS plugin whose provider reports what it billed passes it on with
`output_emitter.report_usage(usage, characters_count=...)`: tokens as a
`TokenUsage`, billed characters as a count, each report one provider request's
share of the open segment. It rides on the segment's final `SynthesizedAudio`
(`usage`, `characters_count`) into `TTSMetrics` and the usage totals; see
[usage](usage.md#text-to-speech). A provider that reports nothing needs no call.

`ChoiceDelta.extra` and `FunctionToolCall.extra` carry provider data (such as
Google thought signatures) as a JSON object: a value that is not JSON fails the
stream when the framework reads the chunk.

A plugin keeps the cause of the errors it raises, redacted:
`raise APIConnectionError(...) from redacted_cause(error)` chains the original
error's type and traceback without its message, which can carry credentials or
provider data.

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
  under its deadlines.

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
