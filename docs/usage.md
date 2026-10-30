# Usage and metrics

Usage answers what an operation consumed. Metrics describe how it performed:
latency, throughput, connection acquisition, interruption, and playback.
Every text run, text-session turn, and voice session exposes the same immutable
`Usage` snapshot. Applications read it directly; collection is automatic.

This page owns usage types, accounting semantics, and measurement availability.
[Agents](agents.md) owns run results and streaming;
[VoiceSession](voice-session.md) owns call lifecycle;
[telemetry](telemetry.md) owns export configuration.

## Reading usage

```python
result = await assistant.run("Hello", context=app)
print(result.usage.input_tokens)
print(result.usage.output_tokens)
print(result.usage.total_tokens)

for model in result.usage.models:
    print(model.type, model.provider, model.model, model.tokens)
```

Both completed and input-required results carry usage. An input-required result
accounts for the work done before asking the question; the next invocation
accounts for its own work.

| Accessor | Scope | Availability |
| --- | --- | --- |
| `result.usage` | One direct agent invocation | When its result is returned |
| `stream.usage` | The operation owned by the stream | From context entry, throughout execution, and after context exit |
| `turn.result.usage` | One `TextSession.run()` turn, across its local agent frames | When the turn's result is returned |
| `session.usage` | One `VoiceSession`, across agent changes and voice tasks | Before start, while running, and after shutdown |
| `ctx.usage` | The owning invocation, text-session turn, or voice session | During a tool or callback |

Before execution starts, a snapshot is empty. Reading an accessor returns the
latest snapshot without advancing execution, consuming stream events, or changing
accounting. A saved snapshot never changes underneath its caller.

### Streaming and cancellation

```python
stream = assistant.run_stream("Hello", context=app)
try:
    async with stream:
        async for delta in stream.stream_text(delta=True):
            await send_text(delta)
        result = await stream.result()
finally:
    # Usage remains accessible after completion, failure, or cancellation.
    usage = stream.usage
```

Usage updates when a provider supplies a measurement. It is not estimated from
text chunks, audio frames, or elapsed time. Some providers report only at the
end of a response, so usage can remain unchanged while output is streaming.

Iterator exhaustion and `result()` follow the normal run-settlement contract.
The last visible token is not the usage-finalization boundary. A stream retains
its usage after failure or cancellation even when `result()` raises. Context exit
waits for the normal cleanup boundary; usage inspection never suppresses an error
or manufactures a successful result.

If entering the context fails before execution starts, the stream retains its
empty snapshot. Cleanup that remains outstanding is reflected by `settled=False`.

### Voice sessions

```python
session = VoiceSession(llm=model, context=app)
await session.start(agent=assistant, room=room)

# Latest measurements received so far.
current = session.usage

await session.aclose()
# Retained after cleanup, including partial usage on failure.
final = session.usage
```

Read `session.usage` when the application needs it: while refreshing a dashboard,
after a conversational boundary, or after awaiting shutdown. A conversation event
does not guarantee that the provider has already reported that turn's usage.
Awaiting `aclose()` is the final local accounting boundary. A close notification
alone is not a substitute for awaiting cleanup.

There is no public `session_usage_updated` event or usage event in `RunStream`.
Provider reporting cadence is an internal concern. Internal notifications still
drive accounting and telemetry; applications do not have to subscribe or collect
provider metrics to obtain usage.

## Public records

The public records are exported from `livekit.agents`. Records, nested records,
and collections are immutable. Python collections use tuples. Counts are
nonnegative integers, represented as `u64` in the Rust ABI. Durations are finite,
nonnegative seconds.

### `TokenUsage`

One token vocabulary is used by provider responses, request metrics, run results,
session reports, storage, and telemetry.

| Field | Meaning |
| --- | --- |
| `input_tokens` | All input tokens, including cache reads and writes and every modality |
| `output_tokens` | All output tokens, including hidden reasoning and every modality |
| `cache_read_tokens` | Input tokens read from cache |
| `cache_write_tokens` | Input tokens written to cache |
| `output_reasoning_tokens` | Reasoning tokens included in output |
| `input_text_tokens`, `input_audio_tokens`, `input_image_tokens` | Provider-reported input modality breakdown |
| `output_text_tokens`, `output_audio_tokens`, `output_image_tokens` | Provider-reported output modality breakdown |
| `cache_read_text_tokens`, `cache_read_audio_tokens`, `cache_read_image_tokens` | Provider-reported cached-input modality breakdown |
| `total_tokens` | Computed as `input_tokens + output_tokens` |

Totals are inclusive. Cache and modality breakdowns overlap: cached audio tokens
are included in input tokens, input audio tokens, and cache-read tokens. Reasoning
tokens are included in output tokens. Never add these subsets to their totals.

For example, 100 input tokens, including 60 cache reads and 20 cache writes,
plus 30 output tokens, including 10 reasoning tokens, means 130 total tokens.
The cache and reasoning breakdowns do not increase that total.

Providers that report cache usage outside their input total are normalized at
the provider boundary. A provider's raw total does not override the normalized
contract. Missing modality breakdowns are not inferred from the model's name or
the absence of audio output.

Input/output totals start at zero. Optional breakdowns use `None` when the
provider did not report them; a reported zero stays zero. When aggregating, a
breakdown is available only if every contributing measured request supplies it
or the adapter can establish that the quantity is zero. Partial breakdowns are
not presented as complete totals.

### `ModelUsage`

`ModelUsage` is a discriminated union with explicit variants for `llm`, `stt`,
`tts`, `eot`, and `interruption`, each exported from `livekit.agents`. Every
variant carries `type`, `provider`, `model`, `requests`, `unmeasured_requests`,
and `tokens: TokenUsage`.

| Variant | Additional billing measurements |
| --- | --- |
| `LLMUsage`, including realtime | `connection_duration`: metered session-connection seconds |
| `STTUsage` | `audio_duration`: processed input audio seconds |
| `TTSUsage` | `audio_duration`: synthesized output audio seconds; `characters_count`: billed characters as the provider reported them, else the characters synthesized ([text to speech](#text-to-speech)) |
| `EOTUsage` | Request count |
| `InterruptionUsage` | Request count |

Optional billing measurements distinguish an unavailable measurement from a
measured zero. Connection duration, processed audio duration, and synthesized
audio duration are different quantities and are never combined into one generic
audio-seconds counter. A request-priced detector can have complete usage without
reporting tokens.

The aggregation key is `(type, provider, model)`. Text and realtime LLM usage
share the LLM variant when their identities match. Explicit breakdowns remain
optional when one source lacks them. Missing identities use `None`; empty strings
are normalized to `None`. Unidentified usage remains accounted for, but its bucket
does not claim that the underlying requests used the same model.

Common billing quantities have typed fields. The normalized public record does
not accept arbitrary dynamic attributes. Raw provider payloads belong in provider
diagnostics, not in the accounting schema. Adding a billable quantity requires
defining its units, inclusion rules, and aggregation behavior.

### `Usage`

| Member | Meaning |
| --- | --- |
| `models` | Per-model usage entries, in deterministic first-observed order |
| `tokens` | Aggregate `TokenUsage` derived from those entries |
| `input_tokens`, `output_tokens`, `total_tokens` | Convenience properties over `tokens` |
| `requests` | Sum of the per-model request counts |
| `tool_calls` | Tool-body executions started in this accounting scope |
| `unmeasured_requests` | Requests whose expected usage measurement is unavailable |
| `settled` | The operation has finished local execution and accounting cleanup |
| `complete` | The operation is settled and no expected usage measurements are missing |

Aggregate tokens cover every included model kind. Use `models` to separate LLM,
STT, and TTS consumption or calculate model-specific charges. Summing token totals
across models is useful for volume; it is not a cost calculation.

Usage is consumption, not an invoice. Pricing, currency conversion, taxes, and
discounts are separate concerns. A missing measurement is never treated as a
known zero charge.

### Text to speech

A TTS request's usage is what its provider reported billing, when it reports it,
and the framework's own measure otherwise. The plugin reports each provider
request with `AudioEmitter.report_usage(usage, characters_count=...)`. Reports
for one segment add up and ride on its final `SynthesizedAudio` (`usage`,
`characters_count`), from which `TTSMetrics` and `TTSUsage` take them. A
forwarding stream (a fallback or sentence adapter) reports what it forwards
into its own emitter, so each request counts once.

| Measure | Reported by the provider | Otherwise |
| --- | --- | --- |
| `tokens` | Token-billed models (OpenAI `gpt-4o-mini-tts`, Mistral): input and output tokens, with the input's text/audio split when given | Zero |
| `characters_count` | The billed characters (AWS Polly, Deepgram, ElevenLabs HTTP, Inworld, MiniMax, Speechify, Vakyam), which can differ from the text sent (markup, normalization) | The characters of the text synthesized |
| `audio_duration` | Not taken from providers | The audio the framework received |

The providers above report usage on the endpoints their plugins use; every
other TTS plugin's provider reports none there, and its usage is the
framework's measure. Provider credits and billed seconds have no typed field:
none of these providers reports them on the synthesis response.

## Counting and missing measurements

A request counts when a framework-visible provider attempt starts, including an
attempt that later fails or is cancelled. Fallback attempts count separately.
Retries hidden inside an SDK or transport are not separately observable and are
not represented as additional requests. Request counts are not conversation-turn
counts. For a streaming connection, adapters define the metered operation boundary;
individual usage reports do not create new requests.

`tool_calls` counts actual tool-body starts, whether they succeed, fail, or are
cancelled. Invalid arguments, approval deferrals, and structured-output candidates
do not execute a tool body and do not increment it. Provider-native tools are not
host tool executions; their consumption requires an explicit provider measurement.

A request awaiting its expected measurement contributes to `unmeasured_requests`.
Receiving its complete measurement resolves that missing entry. A failed or
cancelled request without a final measurement remains missing after settlement.
Partially reported tokens remain included as known consumption, with the request
still marked unmeasured until its measurement is complete.

Thus `settled=True, complete=False` is a valid final snapshot. Local cleanup cannot
recover usage a provider never reported. Counts and tokens describe known
consumption, not a guarantee of agreement with the provider's eventual invoice.
`complete=True` does not promise every optional modality breakdown is available;
it means the expected usage measurements for the operation have arrived.

## Scope and delegation

A direct `agent.run()` result describes that invocation's own work. A nested
direct invocation has its own usage. Calling a backend through an agent tool does
not make its tokens part of the frontend model's usage.

A `TextSession` turn owns the local invocations it coordinates, so
`turn.result.usage` includes work across those frames exactly once. Its history
accessors retain their frame-specific contract. The next reconstructed turn starts
a new accounting scope; saved routing state does not carry old consumption into
the new result.

A voice session includes its locally owned LLM/realtime, STT, TTS, EOT, and
interruption models, across agent changes and temporary voice tasks. A voice task
does not start another independently additive session total. VAD timings, avatar
playback latency, and turn latency are performance metrics, not model usage.

`ctx.usage` reads the latest immutable snapshot of the owning scope. It never
serves as a mutable accumulator to pass to another run. Reads in a tool obey the
same measurement-availability rules as reads outside that tool.

Applications aggregate independent invocations from their returned snapshots.
Do not add an enclosing session/turn snapshot to the child snapshots it already
includes, or repeatedly add cumulative snapshots from the same live operation.
There is no mutable `usage=` parameter on `run()` or `run_stream()`.

Remote backend usage stays attributed to the backend operation. Transporting a
backend usage report requires an explicit wire contract with operation identity
and duplicate handling. A frontend cannot infer it from tool text, and remote
consumption is never silently included in local usage totals.

## Performance metrics

Keep specialized request measurements for LLM, realtime, STT, TTS, VAD, turn
detection, interruption, and avatars. Request metrics embed normalized usage
rather than redeclaring token fields. Timing and connection fields remain specific
to the operation being measured.

The records are exported from `livekit.agents.metrics`; `AgentMetrics` is their
union. Each carries `metadata`, the model and provider that produced it when known.

| Record | Measures | Emitted as `metrics_collected` by |
| --- | --- | --- |
| `LLMMetrics` | One LLM request: time to first token, duration, tokens, tokens per second | the LLM |
| `RealtimeModelMetrics` | One realtime response: time to first audio, duration, connection duration, tokens | the realtime model |
| `STTMetrics` | One recognition request or a stream's audio: duration, connection acquisition | the STT |
| `TTSMetrics` | One synthesis: time to first audio, audio duration, characters, tokens, connection acquisition | the TTS |
| `VADMetrics` | Inference count, total inference time and idle time since the last record | the VAD |
| `EOUMetrics` | One user turn's end: end-of-utterance, transcription and turn-callback delays | the session's turn telemetry |
| `EOTInferenceMetrics` | One end-of-turn model prediction: request timing and server inference time | the turn detector |
| `InterruptionMetrics` | Adaptive interruption detection: inference timing and detection counts | the session's telemetry |
| `AvatarMetrics` | Avatar startup: playback latency and join times | the avatar |

`log_metrics(metrics, logger=None)` logs a record at info level with its fields as
`extra`; VAD and end-of-turn inference records are not logged.

Unavailable timings use `None`, including time to first token/audio when none
arrived. A measured zero is zero. Timestamps specify their clock domain and unit;
duration calculations use the injected monotonic clock. A cancelled or failed
request retains available measurements and its outcome.

`ChatMessage.metrics` remains the application-facing view of per-turn performance.
Provider metrics and OpenTelemetry expose request-level diagnostics. Applications
do not reconstruct billing from either latency fields or generated text length.

## Implementation boundaries

`lk-agents-abi` declares the canonical records once. Bindings, wire encoding, and
storage conversions preserve the same fields and integer ranges. Storage does not
define another independently evolving `RunUsage` schema.

Provider adapters normalize raw measurements into request usage. Each stream
contract specifies whether an update replaces a cumulative measurement or adds a
delta. The collector converts it to a contribution exactly once, using request
identity; repeated final reports and cumulative reports must not double count.
Metric export never feeds consumption back into the collector.

Core owns deterministic accounting and snapshot construction. Runtime owns
provider I/O, cleanup, and internal notifications. Reading a snapshot does not
invoke the host, emit telemetry, or alter usage. Export failures cannot lose the
authoritative accounting state.

Run results, session reports, persisted usage, and telemetry derive from the same
normalized contributions. Telemetry can export a selected breakdown, but names,
units, and inclusion rules must agree. Per-request spans describe their own work;
an inclusive session total is not another independent billable contribution.

## Removing duplicate contracts

The public API uses `Usage`, `TokenUsage`, and the specialized `ModelUsage` union.
It has no separate `RunUsage` or `AgentSessionUsage` shape, public
`session_usage_updated` event, or mutable caller-supplied usage accumulator.
An application can read a snapshot without configuring telemetry or collecting
provider events.

Normalized token names are `input_tokens`, `output_tokens`, `cache_read_tokens`,
and `cache_write_tokens`. Remove `prompt_tokens`/`completion_tokens`, the duplicate
`prompt_cached_tokens`/`cache_read_tokens` representation, and cache-creation
aliases from the normalized API. Provider SDK field names remain confined to
adapters. Deprecated provider-specific output-image fields are replaced by the
optional normalized output-image breakdown when a provider actually reports it.

Totals are computed, absent measurements stay explicit, and empty compatibility
base classes do not form part of the public usage API. Legacy persisted records
are decoded at the storage boundary; compatibility decoding does not reintroduce
old field names or types into newly returned records.
