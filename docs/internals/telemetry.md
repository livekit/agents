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
4. **Telemetry is spelled `telemetry::`.** Outside the telemetry modules, every
   telemetry touchpoint is a path that starts with `telemetry::`: no telemetry
   item is imported under a short name. A reader who cares only about the logic
   skips those lines; `rust/architecture/check.sh` rejects a `use` of a
   telemetry item outside a telemetry module. Telemetry is output only: logic
   never reads telemetry state, and removing every `telemetry::` line leaves
   behavior unchanged.
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
side is one module, `livekit/agents/telemetry.py`, of about 400 lines:

| Piece | What it is |
| --- | --- |
| Provider | `_Span`, `_Tracer` and `_TracerProvider` implementing `opentelemetry-api` over the generated `_core.TelemetrySpan`, installed as the global provider when the application set none |
| Bridge | One `SpanProcessor` for an application-installed SDK provider; imported only when `opentelemetry-sdk` is present |
| Public functions | `configure`, `set_capture_content` and `diagnostics`, forwarding to `_core` |
| Host hooks | `_install` and `_shutdown` (the process exporter; the global provider is adopted once per process), `_activate` and `_current` (HostInvoker's span activation and capture), `_LoopMonitor` (event loop stalls) |
| Framework spans | `room_connect` and `wait_for_participant` around `JobContext`'s room calls, `rpc_call` and `rpc_handler` from an `rtc.RpcInterceptor`, named with the generated `semconv` constants |

`opentelemetry-api` is the only OpenTelemetry dependency. Spans the framework
needs around Python-only code are started by the module's own tracer, so they
reach Rust whatever the global provider is. The worker's Prometheus process
metrics are `ipc/prometheus.py`, not telemetry.

Everything else left Python:

| Removed | Replaced by |
| --- | --- |
| `telemetry/_providers.py`, `_destination.py`, `_export.py`, `_redaction.py`, `_diagnostics.py`, `_transport.py`, `_context.py` | the exporter, destinations and policy in Rust; `telemetry.py` |
| `telemetry/traces.py` (session report upload, RPC tracing) | the Rust session report; RPC spans in `telemetry.py` |
| `telemetry/metrics.py` (OTel instruments, metric series admission) | declared instruments |
| `telemetry/trace_types.py`, `utils.py` | generated `semconv` constants |
| span handlers and `_spans` in `voice/_session_events.py`, `_session_bindings.py` | spans recorded in Rust |
| job and shutdown spans in `ipc/child.py`; telemetry setup, crash log buffering, `SessionReport` and report upload in `job.py` | the worker's and runtime's spans; the generated `SessionReport`; the runtime's job-end sequence |
| `judge_evaluation` spans in `evals.py` and `voice/run_result.py` | judge spans in `lk-agents-inference` |
| `capture_traceparent` / `use_traceparent` in the A2A client, endpoint and job dispatch | trace context carried by the native A2A client and endpoint; `start_job` continues the caller's span |

About 2,800 lines of Python telemetry, plus the telemetry parts of `job.py`,
`ipc/child.py` and the voice bindings, became one module of about 400 lines.

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
| Rust → Python host call | the `cx: &telemetry::Cx` every host call takes (no default), activated in the call's `contextvars` copy |
| Python stream returned by a host call | iterated in that call's own `contextvars` copy, the same copy for every pull and `aclose` |
| Python → Rust call | the current span in `contextvars`, captured at the operation's entry |
| Voice core → runtime | effects carry a `SpanContext` (trace id, span id, flags) of the core span the work runs under |
| `RemoteAgent` → A2A endpoint | W3C `traceparent` header; the endpoint's `a2a_request` continues it |
| Worker → job process | `traceparent` in the job's dispatch; `job_entrypoint` continues it |

**A host call names its span.** Every Rust → Python call goes through
HostInvoker, and every awaited proxy method takes the span it runs under as a
required `cx: &telemetry::Cx` parameter, which codegen refuses to generate
without; `CallContext` and `Invocation` carry no span. `telemetry::Cx` is
declared in `lk-agents-abi` so the proxy traits can name it, and records
through the process tracer (`lk_agents_net::telemetry::Tracer`, its `Recorder`).
A call that runs under no span passes `&telemetry::Cx::NONE`, so the choice is
visible at the call site. An exported function that needs its caller's span
takes a plain `cx: telemetry::Cx`; its binding fills it at the entry with
`telemetry::Cx::entry(..)`, reading Python's current span there and nowhere
else. Calls that run in place (`call`, `read_once`, `bind`) and notifications
activate no span and take none. Log forwarding is the one host entry outside
HostInvoker and runs no application code. Within `lk-agents-py`, Python objects
are called only from `py_calls.rs`, `py_streams.rs`, `py_logging.rs` and
generated output (`rust/architecture/check.sh`).

**Streams behave as `async for` does in Python.** A generator body runs only
when it is pulled, so a stream a host call returns (a node override's async
generator, a provider stream, a duplex stream's output) is iterated in the
`contextvars` copy that call ran in, with its span current: one copy per call,
the same one the call itself ran in, so what the call set before returning its
generator is visible in the generator's body.

- application code in the body parents under the call's span (`llm_node`, not
  whatever was current when the proxy was created);
- every pull and the final `aclose` run in the same copy, so a
  `with tracer.start_as_current_span(..)` held across a `yield` resets its
  token, and a context variable set before a `yield` is visible after it.

The parentage regression table (`tests/test_telemetry_invocations.py`) has one
row per application entry point and the stream cases above.

Parentage always comes from explicit data. No span is found through a
process-global current span.

**A request span exists where a request is made.** `llm_request` and
`tts_request` are the provider request; `llm_node` and `tts_node` are the
overridable callback and carry the same keys whatever it does. With no node
registered the reply's leg makes the request under its node span. A registered
node replaces the provider, so the leg opens no request around it: the default
node it may run (`session.default.llm_node`, `_core.llm_node`;
`session.default.tts_node`, `_core.tts_node`) makes the request, under the span
current at that call and for the step's job and conversation, which the node's
`ProviderCalls` carry. A node making its own output has no request span.

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
| Transport | OTLP/HTTP protobuf, gzip, with the job's Cloud token | OTLP/HTTP protobuf with configured headers |

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
| Report record | `#[record(scope, body)]` | `ChatItemLog` |

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
    #[attr("lk.response.ttft", metric = TurnLlmTtft)]
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
| `label`, `label = key` | The attribute is also a metric label for metrics recorded from this span, under its own key or `key` (`model_name` for `gen_ai.request.model` on the turn spans). Codegen checks the span's labels are exactly the instrument's. |
| `metric = Instrument` | Ending the span records this field as a measurement of the `#[metric]` struct `Instrument`, with its kind, unit and bounds. |
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
| Held by | the code doing the work | the entity it describes, in a field named `span` |
| Ends | **when dropped**, as cancelled, unless ended before | only by `self.telemetry.end(..)` in the entity's removal path; dropping it unended is a bug ([Sans-I/O](#sans-io-the-entity-owns-its-span)) |
| Time | the runtime's injected clock | the step's `now`, supplied by the caller |
| Parent check | debug assertion against the declared parents; DST rule `spans_nest_and_end` checks every recorded span | compile time: `open_in` takes the typed parent handle |

The rule for choosing is mechanical: work that is one async call uses a
`ScopedSpan`; state that lives across state-machine steps owns a
`telemetry::ManualSpan` in its `span` field.

### `ScopedSpan`: ends when its scope exits

A `ScopedSpan` follows Rust's RAII rule, as an OpenTelemetry Rust SDK span does:
dropping it ends the span. Ending is therefore guaranteed on every path,
including early returns and a future dropped by cancellation.

```rust
async fn complete(&self, cx: &telemetry::Cx, request: ChatRequest) -> Result<LlmResponse, LlmError> {
    let span = telemetry::start(cx, LlmRequest { model: request.model(), provider: self.provider() });
    let response = self.send(span.cx(), request).await;
    telemetry::end(span, response.as_ref().map(LlmResponse::end));   // an Err fails it
    response
}
```

| Method | Effect |
| --- | --- |
| `telemetry::start(cx, decl) -> ScopedSpan<D>` | Opens a child of `cx`'s current span. `#[must_use]`. |
| `span.cx() -> &telemetry::Cx` | The child context. It borrows the span, so the span cannot be dropped while the context is in use, and `let _ = telemetry::start(..)` leaves no context to use. |
| `telemetry::end(span, outcome)` | Ends the span with what it ended with: its end attributes, an error, a result (`result.as_ref()`; an `Err` fails it, [Errors](#errors)) or `telemetry::Failed { end, error }` (both). Consumes the span and returns nothing, so no control flow depends on telemetry. |
| `telemetry::event(&span, event)` | Adds a declared event at the current time; `telemetry::event_at(&span, event, at)` at an earlier instant. |
| `telemetry::start_at(cx, decl, at)` | Opens a span that began before the process knew of it (a job, from its receipt). |
| `telemetry::measure(cx, &instrument, value)` | Records one measurement of a declared instrument for `cx`'s job. |
| `telemetry::reserve::<D>(cx)` | Allocates a child context a voice core opens later: the session's startup work runs under its spans before the core exists. |
| drop without `end` | Ends the span with `lk.cancelled = true`. |

A failure the code learns before the span ends (a provider request that failed
and is retried, an entrypoint that raised while the job keeps shutting down) is
kept by the code and passed to `end`; nothing records into an open span.

`lk_agents_net::telemetry` defines these. Each I/O crate's telemetry module
(`lk-agents-runtime`'s and `lk-agents-inference`'s `telemetry.rs`,
`lk-agents-worker`'s `ipc/telemetry.rs`) re-exports them with the declarations
its logic names, so logic code imports only the module (`use crate::telemetry;`)
and writes `telemetry::start(cx, telemetry::LlmNode { .. })`. In `lk-agents-net`
the exporter (`telemetry::otlp`) and the session report upload
(`telemetry::session_report`) are telemetry modules.

Context travels only in `cx`. Nothing is entered into a thread-local, so the
span stays correct across `.await` and concurrent tasks.

A `cx: &telemetry::Cx` parameter is how context travels through I/O code; it is
the one telemetry item that appears in signatures.

When a function is exactly one span, `#[telemetry::span]` writes that code:

```rust
#[telemetry::span(JudgeEvaluation { name: name.into() })]
async fn judge(&self, cx: &telemetry::Cx, name: &str) -> Result<JudgeEvaluationEnd, JudgeError> {
    self.ask(cx, name).await
}
```

The attribute expands to the code of the first example: `cx` inside the body
is the span's context and the body's `Result` goes through `telemetry::end`, so
its `Ok` is the declaration's end attributes.

### Sans-I/O: the entity owns its span

In the voice core a span lives exactly as long as the entity it describes (a
turn, a speech segment, a hook run, a speech), so the entity holds it, in a
field named `span`. The machine's telemetry state, `self.telemetry`
(`telemetry::Trace`), opens and ends spans. Every touchpoint therefore reads
`self.telemetry`, `telemetry::` or `.span`:

```rust
struct Turn {
    id: TurnId,
    origin: Origin,
    span: telemetry::ManualSpan<telemetry::UserTurn>,
}

// The parent is implied: user_turn's one declared parent is the session.
let turn = Turn { id, origin, span: self.telemetry.open(telemetry::UserTurn { origin }) };

// A child, backdated to the VAD's speech start.
let segment = Segment {
    span: self.telemetry.open_in(&turn.span, telemetry::UserSpeaking { started_at: vad.start }),
};

// Ending consumes the span.
self.telemetry.end(segment.span, telemetry::UserSpeakingEnd { is_interruption, ..Default::default() });
self.telemetry.end(turn.span, telemetry::UserTurnEnd { outcome: UserTurnOutcome::Committed, transcript });
```

| `self.telemetry` | Effect |
| --- | --- |
| `step(now)` | Sets the step's time, once, at the machine's step entry. No other call takes a time. |
| `open(decl) -> telemetry::ManualSpan<D>` | Opens a span under `D`'s declared parent when that parent is the session. `#[must_use]`. |
| `open_in(&parent, decl) -> telemetry::ManualSpan<D>` | Opens a child of `parent`; `P: ParentOf<D>` is checked at compile time. |
| `end(span, outcome)` | Ends the span as `telemetry::end` does: end attributes, an error, a result or `Failed`. Consumes it. |
| `event(&span, event)` | A declared event. |
| `context(&span) -> Option<SpanContext>` | What an effect carries, so the runtime opens a `ScopedSpan` under it; `None` for a span opened outside the session's life, which records nothing. |
| `set(&span, decl)` | Replaces the start attributes, for what the entity learns after it opened. |

Calls take at most two arguments, of distinct types, and every value is a
struct literal with named fields. A span that starts or ends in the past names
it in its declaration, in a field marked `#[at]` (`started_at`, `ended_at`, an
event's `at`) that is not an attribute; `Trace` clamps a backdated start to the
parent's start.

A declaration holds the attributes its span carries, filled where the span is
opened. Participant attributes and the STT model and provider are ordinary
declared fields (`#[attrs] participant: Option<ParticipantAttrs>`,
`stt_model`), read from session state the opening code can see: the room link's
linked or local participant, the session's STT. Only `gen_ai.conversation.id`,
the session flags and job identity are written by `Trace` and the exporter on
every record, as the `EverySpan` group and the identity stamp.

| Rule | Enforced by |
| --- | --- |
| A span cannot drift from its entity | the entity owns it: moving or removing the entity moves the span |
| A span ends once | `end` consumes it |
| A span ends at all | `Drop` asserts it was ended (debug builds, so every test and the DST); the mistake fails at the line that dropped the entity |
| Nothing ends silently at close | a session close removes each live entity through its own removal path with the `closed` outcome; nothing ends spans in bulk. `Trace::close` then sweeps whatever remains: a non-zero sweep fails the DST and is a warning plus `telemetry.diagnostics()["swept"]` in production |
| A session's spans always reach the exporter | the session closes on every exit path before the job's final flush; DST rule `sessions_export_their_root` |
| A child is opened only under a declared parent | `open_in` takes `&telemetry::ManualSpan<P>` with `P: ParentOf<D>`; `open` exists only for declarations whose parent is the session |
| One clock | the runtime's injected clock, the voice core's `now` and the `Trace` anchor all come from the `ProcessEnv` clock, so a `ScopedSpan` nested under a sans-I/O span shares its timeline |

An entity that outlives its first owner hands its span over by value. A
speech's `agent_turn` lives in its speech scheduler entry, which the scheduler
returns when it releases the speech; a speech that ends while tool calls it
started still run moves its turn into the record of those calls (the tool
executor's), and the last call to end ends it. A preemptive generation
discarded for the speech that answers the same user turn hands that speech its
turn, moved from one entry to the other. A move is explicit, so a span is never
dropped or left open without the type system or the `Drop` assertion noticing.

| Span | Owner |
| --- | --- |
| `agent_session`, `session_start` | the session, from `start()`; `session_start` ends once the opening and the first `on_enter` are done |
| `session_close` | the session's close, from its request (the exits it issues run under it) |
| `user_turn` and its children | the activity's user turn ([The user turn](#the-user-turn)) |
| `agent_turn`, `agent_speaking` | the speech's scheduler entry, then the record of its running tool calls |
| `start_agent_activity` + `on_enter`, `drain_agent_activity` + `on_exit` | the callback's hook record |
| `update_agent` | the activity's agent update, ended once done and no callback runs under it |
| `amd` | the session's AMD detection, from the detector's attach to its verdict |
| `keyterm_detection` | the keyterm slot's pass in flight |

Core work that continues in the runtime (an LLM node, a TTS node, a tool call)
receives `self.telemetry.context(&speech.span)` in its effect, and the runtime
opens its `ScopedSpan` under it.

### Errors

Every handle records a failure the way the OpenTelemetry API's
`record_exception` does: status `ERROR` with the error's message, `error.type`,
and an `exception` event carrying `exception.type`, `exception.message` and
`exception.stacktrace`. `telemetry::end` and `self.telemetry.end` record it
from a typed error:

- An application exception (a hook, a tool, a node) arrives as the
  HostInvoker's `HostError`, with the Python exception's type, message and
  traceback. The span records that exception, never a fixed description, and
  its `error.type` is the exception's class name, as OpenTelemetry asks.
- A timeout records `error.type = "timeout"`; a provider error records the
  provider error's kind and message.
- The message and stacktrace are content. Cloud's redaction does not read them,
  so for a redacted session the Cloud destination replaces them with
  `"exception details redacted"` itself, as it does the status message; the
  application endpoint does the same unless `allow_pii`.

### The user turn

A user turn starts and ends in many ways, so its span is tied to the one owner
of the turn for its whole life: the activity's user turn entity. `AudioRecognition`
is a pure detector: it reports facts (speech started and ended, a transcript, an
end of turn verdict, a commit) and holds no span. The activity creates the user
turn's `telemetry::UserTurnSpan` on the turn's first evidence and owns it until
the turn ends: as the turn being heard (`user_turn`), then carried with the
decision through the turns waiting for their hook (`DecidedTurn`) and the one
parked on it (`PendingTurn`, whose hook span is its child). It holds
`user_turn` and the spans inside it (each speech segment's `user_speaking`, the
end of turn wait and its detection), updated from the recognizer's facts as the
session publishes them. Creating it opens `user_turn`; its one removal path,
`end(outcome)`, ends it and everything inside it. There is no other open or end
site.

The entity's identity exists only to own the span. The `TurnId` the recognizer
allocates at commit, which also filters stale end of turn verdicts, is
unchanged; the entity records it once the turn commits.

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
    #[attr("lk.transcription_delay", metric = TurnTranscriptionDelay)]
    pub transcription_delay: Measured<Seconds>,
    #[attr("lk.end_of_turn_delay", unmeasured = 0, metric = TurnEndOfTurnDelay)]
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

**Ending.** Removing the turn takes a `UserTurnOutcome`. The enum is
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
#[metric(name = "lk.agents.turn.e2e_latency", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct TurnE2eLatency;

#[metric(name = "lk.agents.usage.llm_input_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmInputTokens {
    #[attr("model_provider")] pub provider: Option<Label>,
    #[attr("model_name")] pub model: Option<Label>,
}
```

- A histogram declares its bucket bounds: `DURATION_BOUNDS` (seconds) or
  `TOKEN_BOUNDS`, the GenAI semantic conventions' advice. Codegen rejects a
  histogram without bounds and a counter with them; the schema publishes them.
- A span end field with `metric = Instrument` records its value with the span's
  `label` attributes when the span ends. The span attribute and the measurement
  cannot diverge.
- A label the instrument declares `Option` is omitted when its value is not
  known (a model without metadata); a required label is always written.
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
pub struct ChatItemLog {
    #[attr("chat.item")]
    pub item: KvList<ChatItemWire>,
    #[span_id]
    pub turn: Option<SpanContext>,
}

#[record(scope = "evaluations", body = "evaluation")]
#[derive(Attrs)]
pub struct EvaluationLog {
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
lk-agents-worker     job end: the session closes and is snapshotted, on_session_end reads the
      │              report through HostInvoker, then the report goes to the two outputs below,
      │              under the job's cleanup deadline
      ├──► lk-agents-core   report::records(&SessionReport, &RecordingOptions) -> Vec<LogRecord>
      │                     report::upload(&SessionReport, &RecordingOptions, simulated)
      │                         -> Option<RecordingUpload>
      │
      └──► lk-agents-net    session_report::export: the records to the exporter, the job's
                            flush, then recording::upload(RecordingUpload): the multipart request
```

| Crate | Owns | Does not |
| --- | --- | --- |
| `lk-agents-abi` | `SessionReport` (the read-only value and its generated Python view), `RecordingOptions`, the record and upload wire declarations (`ChatItemLog`, `SessionReportLog`, `EvaluationLog`, `TagLog`, `OutcomeLog`, `RecordingHeader`, `SessionSummary`) | behavior |
| `lk-agents-voice` | `VoiceSession::report`: chat history, events, usage, options, start time and recording path, read from the session's own state | encoding, I/O |
| `lk-agents-core` | `report::Tagger` (tags, evaluations, outcome); `report::records` and `report::upload`: pure projections into OTLP log records, the `chat_history.json` bytes, the header and the session summary, with their bounds | I/O, host calls |
| `lk-agents-worker` | the job-end sequence: snapshot, `on_session_end`, then the outputs under one deadline, inside the `job_shutdown` span (`on_session_end`, `session_end_upload`) | decisions about content or shape |
| `lk-agents-net` | the OTLP exporter; `session_report::export`: the records, the job's flush, then `recording::upload`: reading the audio file, the multipart request, auth, retries with `RetryInfo`, the recording-disabled response | report content |
| `lk-agents-py` | the generated `SessionReport` and `Tagger` classes | anything else |

`SessionReport` fields: `job_id`, `room_id`, `room`, `agent_name`, `options`,
`events`, `chat_history`, `usage`, `tags`, `evaluations`, `outcome`, `started_at`,
`timestamp`, `duration`, `audio_recording_path`, `audio_recording_started_at`,
`sdk_version`. `JobContext.make_session_report()` returns the same value for an
application that builds its own report: once the job's session closed, its snapshot,
with the tagger's tags, evaluations and outcome as of the call.

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
| Weaver registry | `rust/otel/registry/` and `rust/otel/manifest.yaml` | `weaver registry check` in CI |
| Consumer schema | `rust/otel/schema.json` | Cloud consumers; the golden test |
| Python constants | `livekit/agents/_generated/semconv.py` | the framework's Python code |

The agents registry depends on three registries, each pinned by git ref in `rust/otel/manifest.yaml`:

| Registry | Pin | Owns |
| --- | --- | --- |
| `livekit/protocol` `otel/model/shared` | a commit: the head of protocol PR #1866; move it to the merge commit once that PR merges | correlation keys shared with the client SDKs: `lk.room_id`, `lk.job_id`, `lk.participant_id`, `lk.track_id`, `lk.pii.room_name`, `lk.pii.participant_identity`, `lk.sdk.name`, `lk.sdk.version`, `lk.track_source`, `lk.disconnect_reason`, `lk.agent_name` |
| OpenTelemetry semantic conventions | `v1.44.0` | `error.type`, `service.name`, `exception.*`, `rpc.method` |
| OpenTelemetry GenAI semantic conventions | a commit | `gen_ai.*` |

Each of these keys is one constant in `lk_agents_abi::telemetry::semconv`, never
redeclared here. Everything else, every agents span, attribute, event, metric
and record, is owned here. Adding one needs no change outside this repository.
The generated registry references every `semconv` constant (the
`registry.livekit.agents.semconv` group), and the `Telemetry registry` workflow runs
`weaver registry check -r rust/otel` against the pins: a mistyped constant, or a key
a pinned registry no longer defines, fails CI. `lk-agents-codegen --check` keeps the
registry in sync with the declarations.
A key moves to protocol only when a second producer emits it.

## Verification

| Check | Catches |
| --- | --- |
| Compile errors | undeclared keys and parents in the core, wrong attribute types, double ends |
| Compile errors (I/O) | a key or event a declaration does not hold (`start`, `event` and `measure` take only declarations), an end of the wrong type, a span ended twice (`end` consumes it), telemetry steering control flow (`end` returns nothing) |
| `lk-agents-codegen --check` | declarations out of sync with generated outputs, `Content<T>` outside `lk.pii.*`, unknown semconv keys, a span missing from `spans::SPANS` |
| Clippy | a panic (`unwrap`, `expect`, `panic!`) in a telemetry module: telemetry never fails the agent |
| `rust/architecture/check.sh` (`cx-none`) | new work run under `Cx::NONE`: each use outside tests is counted per file and only shrinks |
| `ScopedSpan` itself | GenAI message content recorded with capture off: every `GENAI_CONTENT` key is dropped at the span's end, whatever the emitter wrote |
| `weaver registry check` (`Telemetry registry` workflow) | a `semconv` constant or declared reference that a pinned registry does not define; an invalid registry |
| Golden OTLP | any wire change: a fixed fake session (pipeline and realtime voice turns, an interruption and a backchannel, a tool call, a handoff, AMD, a failed hook, a text run with a remote agent) exports OTLP JSON compared with `tests/telemetry/golden/` |
| Dashboard rendering | the golden traces and logs are published under `rust/otel/golden/`; the session view's tests in `livekit/web` load them and assert the turns, stages, latencies, tokens, tool time, transcript and evaluations they render |
| DST span rules | every recorded span, scoped or manual, ends exactly once, under a declared parent and within it unless `outlives_parent`; a failed manual span is never recorded as cancelled; nothing reaches the close sweep |
| Conformance tests | an application span inside a tool parents under `function_tool`; a `RemoteAgent` call continues the trace; concurrent jobs keep their identity; Cloud disabled; queue saturation; bounded shutdown |

## Adding telemetry

**A span**

1. Declare it in `telemetry/spans/`: a `#[span]` struct with its name, parent and start
   attributes, and its `end =` struct if it has end attributes.
2. Open it at its one site: `#[telemetry::span(..)]` on the function, a
   `telemetry::start` in a block, or, in a sans-I/O machine, a `span` field on the
   entity that owns its lifetime, opened and ended through `self.telemetry`.
3. Run `lk-agents-codegen --write`, review the schema and golden diffs, commit.

**An attribute**: add a field to the span's start or end `Attrs` struct. Its key is new in this
registry, a protocol shared key, or a GenAI key.

**A metric**: add `metric = Instrument` to a measured end field, or declare a
`#[metric]` instrument and record it where its value is computed.

**A report record**: declare a `#[record]` struct with its scope and emit it from
the report builder.

**A span around Python-only framework code** (an RPC handler): start it with
the OpenTelemetry API, using the generated `semconv` constants for its name and
keys, and declare it in `telemetry/spans/` like any other span so the schema and golden
cover it.

**An application span** needs nothing from the framework:
[telemetry.md](../telemetry.md#application-spans).
