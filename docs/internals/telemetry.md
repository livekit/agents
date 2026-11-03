# Telemetry internals

How every telemetry signal is produced and where it travels. The public API is
in [telemetry.md](../telemetry.md); crate placement follows
[rust-core.md](rust-core.md#telemetry). What is emitted, in full, and what each
consumer reads is in the [telemetry reference](telemetry-reference.md).

Rust owns telemetry end to end: the framework's spans and metrics, the session
report's records, the session report upload, and OTLP export. Python is an
adapter. It keeps the OpenTelemetry API working for application code and hands
everything to Rust.

## Rules

1. **One owner.** Every span, metric and report record is recorded and exported
   by Rust. Python code, the framework's included, creates spans only through
   the OpenTelemetry API, whose provider is Rust. No Python SDK pipeline exists.
2. **Names are declared once.** Span, attribute, event, metric and record names
   come from typed declarations in `lk-agents-abi`. Code never writes a key as a
   string literal.
3. **Consumers see no accidental change.** LiveKit Cloud reads exact names,
   types, units, parents and record shapes. A change to any of them shows up as a
   generated schema diff and a golden OTLP diff in review.
4. **Instrumentation stays at choke points.** A span is opened at the one place
   that owns its work. Business code does not mention spans.
5. **Standards first.** Model requests, tool calls and agent invocations follow
   the OpenTelemetry GenAI semantic conventions ([reference](telemetry-reference.md#standard-conventions)). LiveKit keys extend them only
   where no convention exists.
6. **Mistakes fail early.** Where the type system can reject a mistake it does;
   otherwise `codegen --check` or the DST does. Nothing depends on remembering a
   convention.

## Data flows

```text
  application Python                      Rust (lk-agents-*)                         destinations
  ──────────────────                      ──────────────────                         ────────────
  OTel API spans ─────► Rust-backed  ───┐
  (tools, libraries)    TracerProvider  │
                                        ▼
  framework work ───────────────► spans (ScopedSpan, ManualSpan) ─┐
  (voice core, runtime,                                            │
   inference, AgentRun)  ───────► metrics (declared instruments) ──┼─► exporter ──┬─► LiveKit Cloud OTLP
                                                                   │  (per-       │   (traces, metrics, logs)
  session end ──────────────────► report records (OTLP logs) ──────┘   destination│
                                                                       snapshot)  └─► user OTLP endpoint
                                                                                      (traces, metrics)
  session end ──────────────────► session report + recording ────────────────────► Cloud upload

  Python logging, Rust log ─────► stdout (JSON in production) ───────────────────► hosted sidecar
```

| Signal | Produced in | Cloud | User endpoint | Wire |
| --- | --- | --- | --- | --- |
| Framework spans | `lk-agents-voice`, `lk-agents-runtime`, `lk-agents-inference`, `lk-agents-worker` | yes | yes | OTLP traces |
| Application spans | Python OTel API, through the Rust-backed provider | yes | yes | OTLP traces |
| Framework metrics | declared instruments, aggregated in `lk-agents-net` | yes | yes | OTLP metrics |
| Report records | `lk_agents_core::telemetry::report` | yes | no | OTLP logs (`chat_history`, `evaluations`) |
| Session report and recording | `lk_agents_core::telemetry::report`, uploaded by `lk-agents-net` | yes | no | report upload |
| Process logs | `log` facade and Python `logging` | hosted agents, through the sidecar | no | stdout; never OTLP, so self-hosted agents' logs stay out of Cloud |

### Spans

1. A span opens at its choke point with a typed declaration value and a parent
   (section [Span API](#span-api)).
2. Its W3C trace and span ids are generated in Rust. The voice core draws them
   from the session's id source, so the DST replays them exactly.
3. While the span is current, every host call made with its `cx` runs with the
   span active in the call's `contextvars` copy. Application spans created inside
   that call parent under it.
4. On end, the span becomes one immutable record: ids, parent, name, wall-clock
   start and end, attributes, events, status.
5. The exporter copies the record once per destination, applies that
   destination's policy and queues it.

### Application spans

OpenTelemetry's global tracer provider can be set once per process, and it is
the application's to set. The framework therefore takes it only when the
application has not: when the server starts, it reads the global provider.

| The global provider is | The framework |
| --- | --- |
| unset (the API's proxy) | installs its `TracerProvider` implementing `opentelemetry-api`; spans started through `opentelemetry.trace` start and end in Rust |
| an `opentelemetry-sdk` `TracerProvider` | attaches one span processor that converts each ended `ReadableSpan` into a span record for the Rust exporter; the application's processors and exporters keep receiving their spans unchanged |
| another implementation | leaves it alone and reports a configuration conflict; application spans are not exported by the framework |

Applications that configure an SDK provider do so before starting the server,
as OpenTelemetry requires for any instrumentation. In every case the parent of
an application span is the current span in `contextvars`, which inside a tool or
callback is the framework span HostInvoker activated. Framework spans never pass
through the application's SDK.

### What Python keeps

Python holds no telemetry state, policy, queue or exporter. The whole Python
side is one module, `livekit/agents/telemetry.py`, of about a hundred lines:

| Piece | What it is |
| --- | --- |
| Provider | The generated `_core` tracer, span and provider classes, registered as virtual subclasses of the `opentelemetry-api` ABCs and installed as the global provider |
| Bridge | One `SpanProcessor` for an application-installed SDK provider; imported only when `opentelemetry-sdk` is present |
| Public functions | `configure`, `set_capture_content` and `diagnostics`, forwarding to `_core` |

`opentelemetry-api` is the only OpenTelemetry dependency. Spans the framework
needs around Python-only code (RPC handlers on the `livekit.rtc` participant)
are ordinary OTel API spans and reach Rust like any application span.

Everything else leaves Python:

| Removed | Replaced by |
| --- | --- |
| `telemetry/_providers.py`, `_destination.py`, `_export.py`, `_redaction.py`, `_diagnostics.py`, `_transport.py`, `_context.py` | the exporter, destinations and policy in Rust; `telemetry.py` |
| `telemetry/traces.py` (session report upload, RPC tracing) | the Rust session report; RPC spans in `telemetry.py` |
| `telemetry/metrics.py`, `metrics.py` (`metrics_log_fields` logging) | declared instruments; metrics logged by Rust's `log` |
| `telemetry/trace_types.py`, `utils.py` | generated `semconv` constants |
| span handlers and `_spans` / `_root_span_context` in `voice/_session_events.py`, `_session_bindings.py` | spans recorded in Rust |
| job and shutdown spans in `ipc/child.py`; telemetry setup, `SessionReport` and report upload in `job.py` | the worker's and runtime's spans; the generated `SessionReport`; the runtime's job-end sequence |
| `judge_evaluation` spans in `evals.py` and `voice/run_result.py` | judge spans in `lk-agents-inference` |
| `capture_traceparent` / `use_traceparent` in the A2A client and endpoint | trace context carried by the native A2A client and endpoint |

About 2,700 lines of Python telemetry, plus the telemetry parts of `job.py`,
`ipc/child.py` and the voice bindings, become one module of about 100 lines.

### Metrics

Metrics are declared instruments (section [Metrics](#metrics-1)). A span end
field can record a histogram measurement as it ends; usage counters are fed by
the usage ledger. `lk-agents-net` aggregates measurements per destination and
exports them on a fixed interval with delta temporality to Cloud and the
temporality the user endpoint requests (cumulative by default).

### Report records

Chat history items, session options, the usage summary and evaluation results
are typed records with an exact OTLP log scope and shape (section
[Report records](#report-records-1)). They are built from session state the voice
core already holds and exported to Cloud only.

### Session report

At session end the voice core snapshots the session into a `SessionReport`,
`on_session_end` reads it, and the report becomes Cloud records and one
recording upload (section [Session report](#session-report-1)).

### Context across boundaries

| Boundary | Carrier |
| --- | --- |
| Rust → Python host call | the `CallContext`'s current span, activated in the call's `contextvars` copy |
| Python → Rust call | the current span in `contextvars`, captured at the operation's entry |
| Voice core → runtime | effects carry a `SpanRef` (trace id, span id, flags) of the core span the work runs under |
| `RemoteAgent` → A2A endpoint | W3C `traceparent` header; the endpoint's `a2a_request` continues it |
| Worker → job process | `traceparent` in the job's dispatch; `job_entrypoint` continues it |

Parentage always comes from explicit data. No span is found through a
process-global current span.

## Recording defaults

A session in a job connected to LiveKit Cloud records everything by default:
audio, traces, metrics, logs and the transcript. The transcript is recorded by
default whatever the GenAI message capture setting, because the Cloud session
view is built from it. The session's recording options turn individual parts
off; turning a part off never affects the application endpoint.

## Destinations

| | LiveKit Cloud | User endpoint |
| --- | --- | --- |
| Enabled by | hosted or Cloud-connected job | `OTEL_EXPORTER_OTLP_*` or `telemetry.configure(...)` |
| Signals | traces, metrics, report logs | traces, metrics |
| Scope | per job: a record is exported only while its job is registered and recording that signal | process |
| Before the recording decision | spans of the job are held, bounded, and released or dropped with the decision | exported |
| Content | sent as recorded, keyed for Cloud's redaction ([reference](telemetry-reference.md#cloud-ingest-and-redaction)) | `allow_pii` |
| Identity | the job's identity on every span, log and measurement, and on the resource (below) | the same on spans; measurements carry only declared labels |
| Batching | one job per request: a request never mixes jobs, so Cloud's per-batch session flags apply to one session | any |
| Temporality | delta | cumulative unless configured |
| Transport | OTLP/HTTP protobuf, gzip, with the job's Cloud token | OTLP/HTTP protobuf or JSON with configured headers |

The identity stamp is the one place job identity is written. It uses protocol's
shared keys, and also the spellings Cloud's consumers read today, until they read
the shared keys:

| Identity | Shared key (protocol) | Also written for current Cloud consumers |
| --- | --- | --- |
| Room session | `lk.room_id` | `room_id` (ingest room association, the session view, Agent Insights) |
| Job | `lk.job_id` | `job_id` |
| Room name | `lk.pii.room_name` | |
| Agent | `lk.agent_name`, `lk.cloud_agent_id`, `lk.deployment_id` (agents registry) | |
| SDK (resource) | `lk.sdk.name`, `lk.sdk.version` | |
| Session flags | `lk.redaction.enabled`, `lk.simulation.enabled` (native booleans, every record) | |

Each destination receives an independent snapshot. One destination's policy
cannot change another's data. The decisions (admission, holding, redaction,
byte charges) are synchronous functions in `lk_agents_core::telemetry::policy`;
the exporter calls them once per record and destination.

Queues are bounded by record count and retained bytes. A full queue drops the
record and counts it; `diagnostics()` reports the counts. Export never blocks a
session step, a tool call or the worker heartbeat, and an export failure never
fails a conversation. Flush and shutdown share the job's cleanup deadline with
the session report upload.

## Span API

### Declarations

`lk-agents-abi/src/telemetry/spans/` declares every framework span, one file per area.
Everything that carries attributes is one kind of thing, a struct deriving
`Attrs`; a role attribute says what it is for:

| Role | Declared as | Example |
| --- | --- | --- |
| Span (its start attributes) | `#[span(name, parent, end, events)]` | `LlmRequest` |
| Span end attributes | a plain `Attrs` struct, named by the span's `end =` | `LlmRequestEnd` |
| Span event | `#[event(name)]` | `Generation` |
| Attribute group, flattened with `#[attrs]` | a plain `Attrs` struct | `GenAiUsage`, `ParticipantAttrs` |
| Metric instrument (its labels) | `#[metric(name, kind, unit)]` | `LlmInputTokens` |
| Report record | `#[record(scope, body)]` | `ChatItemRecord` |

```rust
/// One request to an LLM provider, retries included.
#[span(name = "llm_request", parent = LlmNode | KeytermDetection | Amd | AgentRun, end = LlmRequestEnd)]
#[derive(Attrs)]
pub struct LlmRequest {
    #[attr("gen_ai.request.model", label)]
    pub model: String,
    #[attr("gen_ai.provider.name", label)]
    pub provider: Option<String>,
}

#[derive(Attrs)]
pub struct LlmRequestEnd {
    #[attr("lk.response.ttft", metric = "lk.agents.turn.llm_ttft")]
    pub ttft: Measured<Seconds>,
    #[attr("lk.llm_metrics")]
    pub legacy: Json<LegacyLlmMetrics>,
    #[attrs]
    pub usage: GenAiUsage,
}
```

| Part | Meaning |
| --- | --- |
| `name` | The span name on the wire. |
| `parent = A \| B` | Every span type it may be a direct child of. `External` continues a context from outside the process (a job dispatch, an A2A request). `Current` follows its caller: the span nests under whatever span is current, a framework span or the application's own (an `llm.chat()` called from application code, a judge, an RPC call). |
| `end = T` | The attributes `end(..)` requires. Without it, `end()` takes none. Required fields are what make an end value impossible to forget: setters set halfway through a span could be skipped silently. |
| `events = [E, ..]` | The events the span may carry. |
| `outlives_parent = "reason"` | The span may end after its parent. Without it, the DST requires the child to end within its parent. |
| `#[attr("key")]` | One attribute. The key is a declared or generated semconv constant; codegen rejects unknown keys. |
| `label` | The attribute is also a metric label for metrics recorded from this span. |
| `metric = "..."` | Ending the span records this field as a measurement of the named declared instrument. |
| `#[attrs]` | Flattens a typed attribute group (for example `GenAiUsage`). |

### Attribute types

The field type decides the wire encoding. Changing an encoding means changing a
type, which changes the generated schema.

| Type | Wire | Use |
| --- | --- | --- |
| `String`, `i64`, `f64`, `bool` | OTLP scalar | Plain values |
| `Option<T>` | omitted when `None` | Values that do not always exist |
| `Seconds` | double, seconds | Every duration and delay |
| `Measured<T>` | `T`, or the declared `unmeasured =` encoding (`omit` by default, `0`, `-1`) | Values that may be impossible to compute |
| `BoolString` | string `"true"` / `"false"` | Booleans consumers read as strings (`lk.is_interruption`) |
| `bool` session flags | native boolean on every span and log record, never only on the resource | `lk.redaction.enabled`, `lk.simulation.enabled`, which Cloud reads per record |
| `Content<T>`, `GenAiContent<T>` | the value, or removed by redaction | User content: transcripts, responses, tool arguments and output |
| `Json<T>` | a JSON string of `T` | Structured blobs consumers parse (`lk.llm_metrics`) |
| `KvList<T>` | an OTLP key-value list of `T` | Structured record attributes (`chat.item`) |
| `#[derive(SpanValue)]` enums | snake_case string; every variant is a registry enum member | Closed vocabularies (outcomes, origins, reasons) |

Content handling is part of the type, and the two content types differ only in
who can redact them:

| Type | Keys | Redacted by |
| --- | --- | --- |
| `Content<T>` | under `lk.pii.*` (codegen rejects any other key) | Cloud's ingest, by key |
| `GenAiContent<T>` | the GenAI message keys (`gen_ai.input.messages`, `gen_ai.tool.call.arguments`, ...) | the Cloud destination itself for a redacted session, since Cloud's ingest does not recognize these keys |

The application endpoint removes both unless `allow_pii`. GenAI message
capture (`set_capture_content`) is separate: it decides whether
`gen_ai.input.messages`, `gen_ai.output.messages` and judge explanations are
recorded at all, and does not gate any other data.

### Two span handles

The framework has two span handles, and they differ in one thing: **what ends
the span**. One is ended by Rust scope exit (drop), the other only by an
explicit call from the entity that owns it. Choosing the wrong one is the only
way to get span lifetimes wrong, so the rule is mechanical.

| | `ScopedSpan<D>` | `ManualSpan<D>` |
| --- | --- | --- |
| Where | async I/O code: `lk-agents-runtime`, `-inference`, `-worker`, `-net` | sans-I/O state machines: `lk-agents-voice`, `lk-agents-core` |
| Lifetime | a scope: the function or block doing the work | an entity: a turn, a speech, a session |
| Ends | **when dropped**, as cancelled, unless ended before | only explicitly, in the owning entity's one removal path |
| Time | the runtime's injected clock | the step's `now`, supplied by the caller |
| Parent check | debug assertion against the declared parents, and the DST | compile time: `child` takes the typed parent handle |

The rule for choosing is mechanical: work that is one async call uses a
`ScopedSpan`; state that lives across state-machine steps owns a `ManualSpan`.

### `ScopedSpan`: ends when its scope exits

A `ScopedSpan` follows Rust's RAII rule, as an OpenTelemetry Rust SDK span does:
dropping it ends the span. Ending is therefore guaranteed on every path,
including early returns and a future dropped by cancellation.

```rust
async fn complete(&self, cx: &Cx, request: ChatRequest) -> Result<LlmResponse, LlmError> {
    let span = cx.start_span(LlmRequest { model: request.model(), provider: self.provider() });
    let response = span.check(self.send(span.cx(), request).await)?;
    span.end(LlmRequestEnd::from(&response));
    Ok(response)
}
```

| Method | Effect |
| --- | --- |
| `cx.start_span(decl) -> ScopedSpan<D>` | Opens a child of `cx`'s current span. `#[must_use]`. |
| `span.cx() -> &Cx` | The child context. It borrows the span, so the span cannot be dropped while the context is in use, and `let _ = cx.start_span(..)` leaves no context to use. |
| `span.check(result) -> result` | On `Err`, records the error ([Errors](#errors)), then returns the result unchanged. |
| `span.event(event)` | Adds a declared event at the current time. |
| `span.end(end)` | Records the end attributes and ends the span. Consumes the span. |
| drop without `end` | Ends the span with `lk.cancelled = true`, keeping an error recorded by `check`. |

Context travels only in `cx`. Nothing is entered into a thread-local, so the
span stays correct across `.await` and concurrent tasks.

When a function is exactly one span and its return value carries the end
attributes, `#[span]` writes that code:

```rust
#[span(LlmRequest { model: request.model(), provider: self.provider() })]
async fn complete(&self, cx: &Cx, request: ChatRequest) -> Result<LlmResponse, LlmError> {
    self.send(cx, request).await
}

impl SpanEnd<LlmRequest> for LlmResponse {
    fn end(&self) -> LlmRequestEnd {
        LlmRequestEnd { ttft: self.ttft, legacy: Json(self.legacy_metrics()), usage: self.usage() }
    }
}
```

The attribute expands to the code of the first example: `cx` inside the body
is the span's context, the body's `Result` goes through `check`, and an `Ok`
value ends the span through its `SpanEnd` implementation.

### `ManualSpan`: ended by the entity that owns it

In the voice core a span lives exactly as long as the entity it describes. The
entity holds the `ManualSpan` in a private field, opens it in its constructor and
ends it in the one method that retires the entity:

```rust
pub(crate) struct Speech {
    span: ManualSpan<AgentTurn>,
    // ...
}

impl Speech {
    fn new(session: &ManualSpan<AgentSession>, trace: &mut Trace, now: Ts, id: SpeechId) -> Self {
        let span = session.child(trace, now, AgentTurn { speech_id: id.into(), .. });
        Speech { span, /* ... */ }
    }

    /// The only way a speech leaves the session.
    fn retire(self, trace: &mut Trace, now: Ts, outcome: SpeechOutcome) {
        self.span.end(trace, now, AgentTurnEnd::new(outcome, &self.result));
    }
}
```

| Rule | Enforced by |
| --- | --- |
| A child is opened only under a declared parent | `child` takes `&ManualSpan<P>` with `P: ParentOf<D>` |
| A span ends once | `end` consumes the handle |
| A span ends at all | a `debug_assert!` in `Drop`; `Trace::close` ends every span still open, child first, as cancelled; the DST fails if any span reaches the close sweep |
| Times are coherent | callers pass the step's monotonic `now`; `Trace` converts to wall time from the session's anchor; a backdated start (`child_at(started_at, ..)`) is clamped to the parent's start |
| One clock | the runtime's injected clock, the voice core's `now` and the `Trace` anchor all come from the `ProcessEnv` clock, so a `ScopedSpan` nested under a `ManualSpan` shares its timeline |

`span.event(trace, now, event)` adds a declared event and `span.fail(trace,
error)` records a failure ([Errors](#errors)) before the entity ends the span.

`Trace` is the session's span state: the id source, the wall-clock anchor and
the index of open span ids for the close sweep. Ended spans leave a step as
effects; the runtime hands them to the exporter.

Core work that continues in the runtime (an LLM node, a TTS node, a tool call)
receives the core span as a `SpanRef` in its effect, and the runtime opens its
`ScopedSpan` under it.

### Errors

Every handle records a failure the way the OpenTelemetry API's
`record_exception` does: status `ERROR` with the error's message, `error.type`,
and an `exception` event carrying `exception.type`, `exception.message` and
`exception.stacktrace`. `ScopedSpan::check` and `ManualSpan::fail` both record it
from a typed error:

- An application exception (a hook, a tool, a node) arrives as the
  HostInvoker's `HostError`, with the Python exception's type, message and
  traceback. The span records that exception, never a fixed description.
- A timeout records `error.type = "timeout"`; a provider error records the
  provider error's kind and message.
- The message and stacktrace are content. Cloud's redaction does not read them,
  so for a redacted session the Cloud destination replaces them with
  `"exception details redacted"` itself, as it does the status message; the
  application endpoint does the same unless `allow_pii`.

### The user turn

A user turn starts and ends in many ways, so its span is tied to the one owner
of turn identity: the `AudioRecognition` turn (`TurnId`). The turn's state holds
`ManualSpan<UserTurn>`; allocating a `TurnId` opens it and retiring the `TurnId`
ends it. There is no other open or end site.

```rust
#[span(name = "user_turn", parent = AgentSession, end = UserTurnEnd)]
#[derive(Attrs)]
pub struct UserTurn {
    #[attr("lk.user_turn.origin")]
    pub origin: UserTurnOrigin,
    #[attrs]
    pub participant: ParticipantAttrs,
    #[attr("gen_ai.request.model")]
    pub stt_model: Option<String>,
    #[attr("gen_ai.provider.name")]
    pub stt_provider: Option<String>,
}

#[derive(Attrs)]
pub struct UserTurnEnd {
    #[attr("lk.user_turn.outcome")]
    pub outcome: UserTurnOutcome,
    #[attr("lk.pii.user_transcript")]
    pub transcript: Content<String>,
    #[attr("lk.transcript_confidence")]
    pub confidence: Option<f64>,
    #[attr("lk.transcription_delay", metric = "lk.agents.turn.transcription_delay")]
    pub transcription_delay: Measured<Seconds>,
    #[attr("lk.end_of_turn_delay", unmeasured = 0, metric = "lk.agents.turn.end_of_turn_delay")]
    pub end_of_turn_delay: Measured<Seconds>,
    #[attr("lk.eou.resume_count")]
    pub resumes: u32,
    #[attr("lk.provider_request_ids")]
    pub provider_request_ids: Vec<String>,
}
```

**Opening.** The turn opens on its first evidence, with the time of that
evidence, not the time the core learned about it:

| `UserTurnOrigin` | Evidence | Span start |
| --- | --- | --- |
| `speech` | VAD start of speech | the VAD's speech start |
| `transcript` | an STT transcript with no VAD edge (STT turn detection, or speech the VAD missed) | the transcript's audio start, else its arrival |
| `server_speech` | a realtime model's input speech start | the server-reported start |
| `text` | text input to a voice session | its arrival |

**During the turn.** Children describe what happened inside it:

```text
user_turn
├── user_speaking            one per speech segment; every overlap verdict on it (lk.is_interruption
│                            "true" or "false", the last verdict wins)
├── eou_wait                 end of speech to the decision; events rearmed, user_resumed
│   └── eou_detection        one prediction; event superseded when a newer one replaces it
├── user_speaking            the resumed speech, in the same turn
└── on_user_turn_completed   the hook, for a committed turn; event stop_response
```

Speech that resumes before the decision continues the same turn. Only the end
of the turn ends `user_turn`.

**Ending.** Retiring the `TurnId` takes a `UserTurnOutcome`. The enum is
exhaustive and `UserTurnCommitVerdict` converts into it with an exhaustive
`From`, so a new way for a turn to end is a new variant that every match must
handle:

| `UserTurnOutcome` | When |
| --- | --- |
| `committed` | the turn was accepted for a reply (`Reply`, `LateTranscript`) |
| `too_few_words` | below the interruption word minimum; the transcript carries into the next turn |
| `empty` | no words and no reply requested |
| `backchannel` | a realtime backchannel: kept in the transcript, no reply |
| `dropped` | scheduling paused or new turns blocked |
| `cleared` | the application cleared the turn |
| `transcription_timeout` | speech ended and no transcript arrived in time |
| `closed` | the session closed while the turn was open |

A committed turn ends when its `on_user_turn_completed` hook returns, so the
hook's span is the turn's last child and the turn's duration includes it. The
agent turn that answers is a sibling under `agent_session`: `user_turn` never
contains `agent_turn`.

## Metrics

Instruments are declared next to spans:

```rust
#[metric(name = "lk.agents.turn.e2e_latency", kind = Histogram, unit = "s")]
#[derive(Attrs)]
pub struct TurnE2eLatency;

#[metric(name = "lk.agents.usage.llm_input_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmInputTokens {
    #[attr("model_provider")] pub provider: Label,
    #[attr("model_name")] pub model: Label,
}
```

- A span end field with `metric = "..."` records its value with the span's
  `label` attributes when the span ends. The span attribute and the measurement
  cannot diverge.
- Counters without a span are recorded at the one site that computes them:
  usage counters from the usage ledger fold.
- Labels have the `Label` type: an enum or a bounded string. Ids, transcripts
  and error messages cannot be labels.
- Identity is stamping, not labeling. Cloud aggregates its series per job, so its
  measurements also carry the job's identity attributes (`room_id`, `job_id`,
  agent identity) and the session export finds them by room; the user
  endpoint's measurements carry only declared labels.

## Report records

Records Cloud reads as OTLP logs are typed:

```rust
#[record(scope = "chat_history")]
#[derive(Attrs)]
pub struct ChatItemRecord {
    #[attr("chat.item")]
    pub item: KvList<ChatItemWire>,
    #[span_id]
    pub turn: Option<SpanRef>,
}

#[record(scope = "evaluations", body = "evaluation")]
#[derive(Attrs)]
pub struct EvaluationRecord {
    #[attr("evaluation")]
    pub evaluation: KvList<EvaluationWire>,
}
```

| Record | Scope | Shape |
| --- | --- | --- |
| Chat item | `chat_history` | `chat.item`: `message`, `agent_handoff`, `function_call`, `function_call_output`, `agent_config_update` |
| Session summary | `chat_history` | one record: `session.options`, `session.tags`, `usage` (the framework's [`Usage`](../usage.md) model, one entry per model and kind), `agent_name`, `sdk_version`, `session.report_timestamp` |
| Evaluation, tag, outcome | `evaluations` | body names the payload attribute: `evaluation` (`name`, `tag`, `verdict`, `reasoning`, `instructions`), `tag`, `outcome` (`outcome`, `reason`) |

These are the only record kinds. A new kind is a consumer change and follows the
compatibility rules.

The usage summary is the [`Usage`](../usage.md) model itself, so Cloud,
`result.usage` and `session.usage` read the same accounting. Its wire shape is
generated from the model's declaration and published in the schema. The same
shape is the session summary's `usage`; the session view and the agent
evaluations' session summary schema read it.

Recording options decide which records exist (`transcript` for chat items);
content inside them follows the destination's redaction. A chat item records the span id of the turn it belongs to, so a
consumer can link a transcript row to its trace without matching timestamps.

Queue bounds are sized so a session's end-of-session burst (every chat item,
the usage summary and the evaluations) fits without dropping; records beyond the
bound are dropped and counted, never truncated.

## Session report

The session report is the summary of one voice session: its transcript, usage,
options, events, tags and evaluations, and its audio recording. One value
carries it from the voice core to every reader.

```text
lk-agents-voice      VoiceSession::report(now) -> SessionReport      snapshot of session state
      │
lk-agents-runtime    job end: run on_session_end(SessionReport) through HostInvoker,
      │              then hand the report to the two outputs below, under the job's
      │              cleanup deadline
      ├──► lk-agents-core   report::records(&SessionReport, &RecordingOptions) -> Vec<LogRecord>
      │                     report::upload(&SessionReport, &RecordingOptions) -> RecordingUpload
      │
      └──► lk-agents-net    exporter: the records, with the job's final flush
                            recording::upload(RecordingUpload): the multipart request
```

| Crate | Owns | Does not |
| --- | --- | --- |
| `lk-agents-abi` | `SessionReport` (the read-only value and its generated Python view), `RecordingOptions`, the record and upload wire declarations (`ChatItemRecord`, `SessionSummaryRecord`, `EvaluationRecord`, `RecordingHeader`, `SessionSummary`) | behavior |
| `lk-agents-voice` | `VoiceSession::report`: chat history, events, usage, options, start time and recording path, read from the session's own state | encoding, I/O |
| `lk-agents-core` | `report::Tagger` (tags, evaluations, outcome); `report::records` and `report::upload`: pure projections into OTLP log records, the `chat_history.json` bytes, the header and the session summary, with their bounds | I/O, host calls |
| `lk-agents-runtime` | the job-end sequence: snapshot, `on_session_end`, projections, export flush and upload, under one deadline, inside the `job_shutdown` span (`on_session_end`, `session_end_upload`) | decisions about content or shape |
| `lk-agents-net` | the OTLP exporter; `recording::upload`: reading the audio file, the multipart request, auth, retries with `RetryInfo`, the recording-disabled response | report content |
| `lk-agents-py` | the generated `SessionReport` and `Tagger` classes | anything else |

`SessionReport` fields: `job_id`, `room_id`, `room`, `options`, `events`,
`chat_history`, `usage`, `tags`, `evaluations`, `outcome`, `started_at`,
`timestamp`, `duration`, `audio_recording_path`, `audio_recording_started_at`,
`sdk_version`. `JobContext.make_session_report()` returns the same value for an
application that builds its own report.

Rules:

- One snapshot. Every output reads the same `SessionReport`; no output reads
  live session state, so the transcript, the records and the summary agree.
- Projections are pure. `report::records` and `report::upload` take the report
  and the recording options and return data; the DST and golden tests call them
  directly.
- Bounds are decided before I/O. An oversized transcript drops whole items from
  the oldest end and records the count in the summary; it never fails the upload
  or drops the audio.
- The recording options decide which outputs exist. A redacted session always
  uploads `chat_history`, so Cloud can rebuild the transcript.

## Naming registry

`lk-agents-codegen` reads the declarations and writes:

| Output | Path | Reader |
| --- | --- | --- |
| Weaver registry | `otel/registry/` and `otel/manifest.yaml` | `weaver registry check`, protocol's live-check and combined docs |
| Consumer schema | `otel/schema.json` | Cloud consumers; the golden test |
| Python constants | `livekit/agents/_generated/semconv.py` | the framework's Python code |
| Trace rules | `tests/trace_schema.py` | trace tests and the DST rule set |

The agents registry depends on two registries, each pinned by git ref:

| Registry | Owns |
| --- | --- |
| `livekit/protocol` `otel/model/shared` | correlation keys shared with the client SDKs: `lk.room_id`, `lk.job_id`, `lk.participant_id`, `lk.track_id`, `lk.pii.room_name`, `lk.pii.participant_identity`, `lk.pii.remote_participant_identity`, `lk.sdk.name`, `lk.sdk.version`, `lk.track_source`, `lk.disconnect_reason`; and the naming policies |
| OpenTelemetry GenAI semantic conventions | `gen_ai.*` |

Declarations use the generated constants for these keys; they are never
redeclared here. Everything else, every agents span, attribute, event, metric
and record, is owned here. Adding one needs no change outside this repository. CI runs
`weaver registry check` with protocol's policies and `lk-agents-codegen --check`.
A key moves to protocol only when a second producer emits it.

## Verification

| Check | Catches |
| --- | --- |
| Compile errors | undeclared keys and parents in the core, wrong attribute types, double ends |
| `lk-agents-codegen --check` | declarations out of sync with generated outputs, `Content<T>` outside `lk.pii.*`, unknown semconv keys |
| `weaver registry check` | violations of protocol's naming policies |
| Golden OTLP | any wire change: a fixed fake session (pipeline and realtime voice turns, an interruption and a backchannel, a tool call, a handoff, AMD, a failed hook, a text run with a remote agent) exports OTLP JSON compared with `tests/telemetry/golden/` |
| Dashboard rendering | the golden traces and logs are published under `otel/golden/`; the session view's tests in `livekit/web` load them and assert the turns, stages, latencies, tokens, tool time, transcript and evaluations they render |
| DST span rules | every span ends exactly once; children end within their parent unless `outlives_parent`; parents are declared; an `Err` result is never recorded as cancelled; nothing reaches the close sweep |
| Conformance tests | an application span inside a tool parents under `function_tool`; a `RemoteAgent` call continues the trace; concurrent jobs keep their identity; Cloud disabled; queue saturation; bounded shutdown |

## Adding telemetry

**A span**

1. Declare it in `telemetry/spans/`: a `#[span]` struct with its name, parent and start
   attributes, and its `end =` struct if it has end attributes.
2. Open it at its one site: `#[span(..)]` on the function, a `ScopedSpan` in a
   block, or a `ManualSpan` field on the entity that owns its lifetime.
3. Run `lk-agents-codegen --write`, review the schema and golden diffs, commit.

**An attribute**: add a field to the span's start or end `Attrs` struct. Its key is new in this
registry, a protocol shared key, or a GenAI key.

**A metric**: add `metric = "..."` to a measured end field, or declare a
`#[metric]` instrument and record it where its value is computed.

**A report record**: declare a `#[record]` struct with its scope and emit it from
the report builder.

**A span around Python-only framework code** (an RPC handler): start it with
the OpenTelemetry API, using the generated `semconv` constants for its name and
keys, and declare it in `telemetry/spans/` like any other span so the schema and golden
cover it.

**An application span** needs nothing from the framework:
[telemetry.md](../telemetry.md#application-spans).
