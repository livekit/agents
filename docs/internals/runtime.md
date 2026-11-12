# Runtime contracts

Effects, resources, streams, cancellation, backpressure and managed execution.

[Contributor index](README.md) · [Execution paths](architecture.md#execution-paths)

## Error contracts

Declare error families and their structured fields in ABI and generate the
catchable Python and Node types. Keep `APIError`, `APIStatusError`,
`APIConnectionError` and `APITimeoutError`; timeout is a connection-error subtype.
Do not collapse domain errors into one generic runtime exception.

An API error carries its message, retry eligibility, optional status code,
request ID, response body and cause. Timeout errors also carry a typed `timeout`, the limit that expired:
`pool`, `connect`, `read`, `write` or `total`. A foreign SDK timeout
that does not say which uses an absent one, never a guessed one. Preserve
the operation identity in the correlated failure event. Error messages are for
people; policy uses typed categories and fields, not class-name strings or
message parsing.

Preserve the original host exception, traceback and cause when returning it to
the same host. Retain diagnostic data for every observer even when ownership of
the original exception is transferred. Cross-process errors contain validated
data and cause records, never host references or arbitrary exception objects.
Response bodies crossing that boundary use the declared JSON/text/bytes types.

`retryable` means the error permits retry consideration. `recoverable` on a
provider event means the framework is actually recovering through retry or
fallback. Neither proves that repeating an external action is safe. The owning
protocol also checks progress, replay support and idempotency.

Cancellation and deliberate control outcomes, including input-required and
completion, are separate from failures. A provider's explicit cancellation
response ends its request without retry or a provider-health penalty. Tools
expose an intentional `ToolError` message to the model; unexpected exceptions
produce a generic tool failure while preserving the original diagnostic cause.
Developer-facing API exceptions retain diagnostic response bodies and causes.
Model-facing tool results and externally returned client errors use safe
rendering that does not expose credentials, raw bodies or tracebacks.

Conformance cases cover subtype matching, fields, original exception identity,
causes, multiple observers, foreign errors, timeouts, cancellation and
safe public error rendering. Retry disposition is tested in the protocol core,
not in exception conversion.
## Common runtime primitives

Provider plugins, custom nodes, transforms and I/O use the same Rust contracts
and generated language adapters. A feature does not implement its own iterator
pump, cancellation bridge, exception conversion or host-object registry.
Declarations live in ABI; executable bridges, queues and host handles live in
`lk-agents-runtime`, the host stream adapter lives in `lk-agents-net`, and async
I/O crates use the same source and sink contracts.
Cores consume data and effects rather than live streams.

| Primitive | Contract |
| --- | --- |
| `StreamOutput<T>` | One consumer pulls typed items; each pull produces an item, end-of-stream or an error |
| `StreamInput<T>` | A producer submits typed items; bounded admission is distinct from processing or playback completion. `flush` ends a segment; `end_input` allows output to drain |
| `DuplexStream<I, O>` | Independently owned input and output under one operation lifetime; consuming `split()` returns both halves without cloning a consumer |
| `ByteStream` | Independent byte reads and writes, flow control, half-close and reset |
| `RunStream` | A running agent operation, its event/text consumer, delivery acknowledgements and terminal result |
| Transform | A typed source-to-source operation; may filter, expand, buffer or change item types without owning the upstream resource implicitly |
| Host call | Invoke a callback on its owning language runtime, preserving invocation context and correlated outcome |
| Resource handle | Typed identity plus generation and explicit ownership; runtime retains the object and settles/releases it |
| Cancellation | Correlated termination request; completion and cleanup are observed outcomes, not implied by dropping a reference |
| Media frame | Typed format, sample/time information and bounded payload or opaque media reference; lifetime remains valid until the consumer releases it |

Ordinary streams contain no provider timing, metrics or retry hooks. Provider output explicitly
extends that contract through `ProviderStream<T>`; room events, byte chunks and node input do not
expose provider observations. The Python iterator adapter implements the same language-neutral
contract; a Node adapter owns its own iterator scheduling and references.

Shared native stream access admits one pull at a time. Closing stops admission and wakes a
blocked pull, then settles owned cleanup under the injected clock's deadline. Concurrent close
callers observe the same outcome. Dropping a close waiter does not replace or restart cleanup;
a later waiter resumes that cleanup. Input failure wakes the output, and closing output refuses
further input. Cancelling a whole operation may abandon its pending pull before closing it;
`cancel_safe` separately declares whether a pull can be abandoned and then resumed without loss.

### HostInvoker

`HostInvoker` is the internal runtime primitive for executing host-language
application and plugin code. Every Rust-to-host call goes through it, from any
crate: job entrypoints, setup/shutdown callbacks, tools, hooks, nodes, provider
and plugin proxies, CLI and dev-server callbacks and endpoint handlers. The one exception is log forwarding: `lk-agents-py` attaches Rust's
`log` facade to Python logging directly ([logging](processes.md#logging)). Each call carries the Invocation captured at its operation's entry
and its declared scopes; none writes its own dispatch, context or cancellation
wrapper. ABI declares the typed call contract and HostInvoker's port;
`lk-agents-runtime` owns the invocation lifecycle and dispatch as language-neutral
Rust; `lk-agents-py` implements the Python entry behind the port.
Owned host operations declare `#[host(cancel)]` when cancellation must wait for
authored asynchronous cleanup. The signal stays native. HostInvoker delivers it
to the host task and retains its settlement future until that task finishes,
including `finally` blocks. Dropping a waiter alone is not evidence of settlement.
Capabilities returned by either synchronous or asynchronous host calls inherit
the returning operation’s captured context, including capabilities inside
optional values and containers.
A proxy taken with no running loop (a provider facade built at import) captures
none: each call runs in the context of the operation driving it, the task-local
host context a `_core` entry sets for its native future
(`lk_agents_net::task::host_context`), which tasks spawned on the process spawner
inherit (a fallback chain's recovery probes).

Async I/O crates
contain no host-language code: they hold ABI-declared proxy traits whose generated
implementations enter through HostInvoker. An architecture rule enforces this.
It is not a public API that developers must call.

An internal call has this shape:

```rust
let outcome = host_invoker
    .invoke(callback, arguments, invocation_context)
    .await;
```

The invocation carries its operation identity, trace context and declared
invocation-local values explicitly. Capture them from the owning operation
when creating the invocation, not from whichever task later dispatches it or
from the callback's registration context. SDK objects remain host-owned.

This includes framework invocation-local bindings, not only tracing: the owning
job when present, agent/session, active speech handle when applicable, tool-call
identity and approval scope. Explicit `RunContext` arguments and context-local
accessors resolve the same invocation. A callback without a speech scope cannot
inherit another invocation's active speech handle.

The Python adapter establishes these bindings through `ContextVar`; Node uses
its asynchronous context mechanism. Application-defined context variables follow
normal host context propagation and remain host-owned, without serialization
or inspection by Rust. The shared invocation path preserves the originating
host context and applies the declared framework bindings without overwriting
unrelated application values. Retaining context does not extend an operation's
lifetime or make a stale resource handle valid.

Propagation works in both directions. When user code calls a context-aware Rust
API, the generated binding captures the host's invocation context and supplies
it to the native method as a hidden, typed argument. Developers do not pass
this argument in the public Python/Node API. Conceptually:

```python
# Public API
session.generate_reply()

# Equivalent generated binding logic at that call's entry
invocation = capture_invocation_context()
native_generate_reply(session_handle, invocation)
```

Capture context synchronously at call entry, before scheduling native work.
Preserve framework ownership bindings and capture the host's current trace
context, including user-created spans, as the parent for resulting native work.
Trace parentage can change within user code without changing its job, speech or
approval scope. Runtime retains this context with the work; further Rust calls
and subsequent `HostInvoker` callbacks carry it explicitly. The generated
boundary reads host context; domain code receives typed IDs and values, not
host objects or ambient lookups. Tokio tasks and OS threads do not implicitly
inherit host context.

ABI declares required and optional scopes for each call. A direct application
call such as `session.generate_reply()` can create work against its explicit
receiver with available job/trace context and no active callback. It must not
invent tool, speech or approval scope. Missing required scopes produce a clear
error. Generate these rules once; features do not add handwritten wrappers.

`get_job_context()` resolves the job bound to that invocation. Keep the host
`JobContext` object host-owned; Rust receives only the typed job identity and
resource handles it needs. Context accessors preserve their required/optional
behavior when no scope exists. Direct text runs and other work outside a job
do not acquire a job implicitly. Pure Rust operations need no invocation context.

Context-dependent operations validate their scope against their target
resource; an explicit receiver cannot accidentally use another
session's speech or job. Deferred work retains its originating scope rather
than consulting whichever agent, speech or job is active later.

The shared adapter activates context for actual user execution, including sync
calls and coroutine scheduling, and restores the caller's previous context on
every exit. Constructing a coroutine alone does not establish its execution
context. Nested invocations preserve their declared parentage; unrelated
concurrent invocations have isolated bindings. Context copies do not deep-copy
mutable application objects. Do not hold runtime state locks while executing
user code.

A host stream retains its execution context across resumptions and cleanup,
including context changes made by its generator. Do not recreate a fresh context
for every pull: scopes opened before a yield must close in the same execution
context. Serialize entry into that stream's context and keep it separate from
the consumer's context.
SDK context managers that require task affinity must also enter and exit on the same
host task, even when native callbacks arrive through separate invocations.

Framework-scheduled offload propagates context only across supported host worker
boundaries. Opaque host contexts and callbacks stay in their owning runtime;
processes and isolated workers receive only declared transferable data. Arbitrary
Node functions or context objects cannot be moved into a Worker isolate.

Release captured host contexts and references when their invocation or stream
settles. Outstanding cleanup retains only the scope it still owns and reports
its incomplete outcome; context capture cannot become an unbounded job/session
registry.

`HostInvoker` uses the shared result conversion, stream and cancellation
contracts below. A feature supplies its callback and invocation data; it does
not recreate those mechanisms. Context required for invocation is independent
of best-effort telemetry export.

Run a shared regression/conformance suite through the real host adapters. Cover
sync callbacks reached from Rust, async callbacks, iterator execution and close,
nested calls, concurrent operations, delayed dispatch, worker threads, exceptions and
cancellation. Assert user-visible invocation values, actual trace/parent IDs,
the correct speech handle and approval scope, preservation of application
context variables, isolation from invocations without a speech scope, and
context restoration and cleanup. Repeat the same behavioral cases for each
binding, including tracing disabled or spans not recorded. Core-only tests or
an exporter mock cannot establish correct host context propagation.
Include host-to-Rust-to-host round trips after a queue/thread hop, concurrent
jobs and speeches, missing job/speech scope, and stale or mismatched resource
handles. Cover direct calls outside a callback and a user-created span around a
native call. Assert that ownership bindings remain unchanged, native/callback
spans inherit the user span, and `get_job_context()` returns the same host
object after the round trip.
Exercise a generator that opens a context scope before yielding and closes it
during completion, explicit close or cancellation; assert scope continuity and
restoration, not just matching invocation IDs.

### Iterators and callback results

Generate both directions of each bridge: host iterables can feed Rust operations,
and Rust sources can be consumed through the host's normal iteration protocol.

| Rust/runtime shape | Python | Node |
| --- | --- | --- |
| Source consumed asynchronously | `AsyncIterable[T]` / `AsyncIterator[T]` | `AsyncIterable<T>` / `AsyncIterator<T>` |
| Synchronous source where declared | `Iterable[T]` / `Iterator[T]` | `Iterable<T>` / `Iterator<T>` |
| Awaitable operation | Coroutine/awaitable | Promise |
| Typed item/result | Generated record, enum or validated application value | Generated record, discriminated union or validated application value |

Normalize a node's declared return forms once at its generated adapter. A hook
may declare a direct value, an awaitable value, a stream or an awaitable stream.
Its item schema and signature determine which forms apply. Strings, byte buffers
and records remain single items unless the contract explicitly makes them a
source; their language-level iterability does not split them into characters,
bytes or fields.

`llm_node`, `stt_node`, `tts_node`, streamed `say` and user transforms all use
this normalization and stream lifecycle. A new node declares its signature,
item types and capabilities rather than writing another bridge. A normal
function return is not automatically a stream or a list materialization.

Every host-language node and transform executes through `HostInvoker`, including
opening its returned stream, pulling items and closing it. Apply the same rule
to `transcription_node`. Native Rust
implementations stay native and receive the relevant typed context explicitly.

Run node regression cases through the real generated adapters, not only the
invoker in isolation. Across awaits, yields, nested Rust calls and cleanup,
assert the expected job, agent/session, speech, application context variables
and trace parent for each applicable scope. Cover simultaneous node calls,
replacement, cancellation and calls with no job/speech scope. Scope values must
come from the owning invocation; replacement cannot substitute the session's
newly active agent or speech. Reuse these cases for future host node APIs.

Host callbacks run on their captured Python/Node event loop and invocation
context. Results return through runtime queues. Python execution follows normal
interpreter rules; the bridge does not retain the GIL across native waits.
Core steps neither invoke the host directly nor permit callbacks to mutate
core state.

### Stream lifecycle

A source has one active consumer. Each upstream pull follows downstream demand;
adapters do not eagerly drain a generator into an unbounded queue. Any bounded
buffer declares its capacity and overflow behavior. Fan-out is explicit and
defines whether a slow branch blocks, drops allowed data or fails.

Allow at most one outstanding pull per source. Reject an overlapping pull as
invalid stream state through the shared ABI error contract, before advancing
the source. Do not queue it, cancel the accepted pull or consume its item.
The reservation remains until the underlying pull settles, even if its caller
stops awaiting it. Apply this rule to native, Python and Node sources.

Synchronous iteration performs no awaits. Potentially blocking iteration uses a
bounded worker only where its host/source contract supports offload; otherwise
require an asynchronous or nonblocking source. It cannot block the core or
shared event loop. Async generators run on their owning Python/Node loop;
native Rust sources stay native. The bridge does not spawn a new producer task
per item merely to cross a language boundary.

End-of-stream, input end, segment flush, error and cancellation are separate
outcomes. A flush is typed stream control, not a special string or an implicit
close. Unexpected host cancellation is not silently reported as successful EOF.
Control and cancellation remain serviceable while data delivery is blocked.
Close stops new pull admission and uses the shared cancellation/cleanup
lifecycle to settle an outstanding pull before tearing down the iterator.
Its control path cannot wait behind the blocked pull it needs to terminate.

The runtime closes streams it owns, forwarding Python generator cleanup or
Node iterator `return()` where supported. Borrowed sources are detached without
closing their owner. Failures preserve the useful cause across the language
boundary. Successful cleanup leaves no outstanding pull or promise. Timed-out
cleanup retains an explicit owner; its late work cannot affect a replacement.

Use the [shared conformance cases](bindings.md#conformance) for native and host streams.
Stream-specific cases include split chunks, rejection of an overlapping pull
without disturbing the accepted pull, flush versus EOF and close/cancellation
while a pull is blocked.

### Cancellation and task ownership

Every runtime task has an explicit owner, identity and observable terminal
outcome. Cancelling an owner propagates to its owned work; borrowed resources
retain their owner's lifetime. Detached work requires another explicit owner.
Stopping an await is separate from cancelling the operation it observes.

Rust has one task-ownership primitive. The injected `Spawner` (ABI) hands back an
`OwnedTask<T>`: dropping it requests cancellation and never detaches; awaiting it
yields the value, or `TaskError::Cancelled`/`Panicked`, only after the work and
everything it owned were dropped; an optional settle step runs with the
`TaskOutcome` either way. ABI declares only this contract. Its implementation
(`lk_agents_net::task::launch`) is executor-agnostic, so Tokio, the host loop's
virtual time and the DST's scheduler run tasks the same way, and a panic still
reaches the executor after its outcome is reported. A group of tasks is a
`TaskSet` of owned tasks: named, cancellable by id, drained under a deadline;
tasks that end on their own (a pump until its input closes, a close that must
outlive its waiter) live in their owner's set, which the owner drains at its end.
Work nobody owns by choice goes through `spawn_detached`, with the reason at the
call site. Tokio's `JoinHandle` remains only inside executors; no `JoinSet` or
`TaskTracker` is used, and Tokio's blocking pool is not used (below).

Cancellation is idempotent and correlated. It can arrive before launch, during
an await/pull, after partial output or alongside completion. The owner records
which outcome won; stale completions cannot revive the operation. Completion
reports distinguish accepted results from cancellation or failure.

Generated adapters request Python task cancellation or Node cooperative
cancellation. Rust future drop/abort is an implementation mechanism, not proof
of host termination. All languages use the same cancellation contract.

Runtime waits raced against cancellation or other events preserve partial I/O
progress and ownership of unsent values. Use cancellation-safe operations or
retain that state outside the cancelled future. A partially consumed frame
cannot restart as a new frame; retain framing progress or retire the connection.
Test interrupted reads, writes and queue sends as well as final cleanup.

Waiting for settlement includes required asynchronous cleanup. Deadlines bound
that wait; expiry reports incomplete cleanup and retains responsibility for
late resources. Neither task-table removal nor scheduling a host cancel callback
proves cleanup finished. `aclose()` returns normally when its bounded wait ends.
If cleanup misses its deadline, the runtime logs an error, emits a typed
incomplete-cleanup observation and continues owning the resource. Tests treat
that observation as a failure; a normal return does not certify cleanup success.

Cancellation cannot forcibly stop arbitrary synchronous code or undo external
effects. Escalation that terminates a process is a process-supervision decision,
not a hidden behavior of a stream or tool wrapper. Conformance tests exercise
cancellation races and cleanup outcomes in both bindings.

### Lifecycle and ownership

Each rule below removes a class of bug: a hang, a leak, or work that was abandoned silently.
It does this by making the bad state impossible to build, not by relying on a convention.

- **Waits.** A wait is on an `lk_agents_net::signal` cell. Its `Producer` is the only writer,
  and it lives where the progress happens:
  - moved into the task that makes the progress (`TaskSet::spawn_settling`), or
  - held in the table of the loop that owns the state.

  When the producer drops (by a return, a cancel, a panic or a runtime drop), every waiter
  resolves. It gets `Err(Ended)` when the state can no longer come. So a waiter that is never
  woken cannot be written.
- **Shared state.** When several parties change one piece of state, it is a `signal::Shared`
  with a terminal value, such as closed, that every wait accepts. `Notify` and raw `watch`
  appear only inside `signal.rs`; Clippy rejects them anywhere else.
- **Closeable resources.** A resource that must be closed (a stream, a track, a provider
  socket) is held in `owned::Owned` from the moment it exists, before the next await. Dropping
  it runs its bounded close as a cleanup task, which the owner's `TaskSet` joins. A close that
  must outlive its caller goes through `owned::Background`, never `spawn_detached`.
- **Partition leases.** The only way to get a `Lease` is a `PendingLease`, which is minted per take
  under a fresh token. If the take is dropped while the request may already hold the partition (an
  abandoned future, an error, a lost acknowledgment), the `PendingLease` releases by token and
  ends the run `canceled`. A `Lease` renews until its commit resolves, and releases when it
  drops. Each owner has one `Holder`, and the owner's close awaits those releases.
- **Task deaths.** A task that feeds a machine reports its own death as an input
  (`TaskSet::scoped_feeding`), so a lost task cannot leave the machine waiting.
- **Session end.** Every wait a host can await (a speech handle, a reply, a run result)
  resolves when its session ends, whether it was drained or dropped. The voice runtime owns
  each speech's lifecycle producers, so closing the session finishes them.
- **Uncertain writes.** A storage write whose acknowledgment was lost is `Unacknowledged`. It is
  re-sent through `Resend` with a key that makes resending safe, until its outcome is known or
  the retries run out.

The DST checks these rules on every run:
- `every_wait_resolves`: each wait the driver started resolved.
- `no_waiter_at_quiescence`: nothing is still parked once the system goes idle.
- The lease, stream and transcript rules from AGENTS.md.

Health metrics (`telemetry/health.rs`) report what still slips through in production:
- `lk.agents.partition.acquire_wait`: how long a partition acquire waited;
- `lk.agents.partition.ownerless_wait`: time spent waiting behind a holder that stopped renewing;
- `lk.agents.partition.lease_lapsed`: leases taken over after they lapsed instead of being released;
- `lk.agents.teardown.overrun`: teardowns that ran past their bound;
- `lk.agents.storage.write_uncertain`: writes still unresolved when their retries ran out.

A call's storage records these metrics under the call's job, so they reach LiveKit Cloud with
the job's other metrics.
## Runtime limits and backpressure

Each runtime queue declares its capacity, accounting unit and full behavior.
Use bytes or media duration as well as item counts when payload sizes vary
widely; a queue whose large payload is one kind (audio) may count bytes for that
kind only, its other items bounded by count times their transport message limit.
Admission limits include in-flight operations and producers waiting for space;
a bounded channel with an unbounded population of blocked tasks is not a bound.
Retained buffers, pending results and resource tables have ownership and limits.

The owning interface specifies whether a full queue waits, rejects admission,
coalesces updates or discards replaceable data, and how that outcome is observed.
Preserve the delivery contract of the API that produces the data. Media dropping
accounts for lost samples/time; a capacity limit cannot silently change playback
accounting.

Audio into a consumer goes through a `lk_agents_media::media_queue`, its policy
chosen by what the consumer is for: `CatchUp` where what was said matters,
`Now` where only current audio does, the pre-connect head kept in full under
both. A bounded channel whose overflow fails the stream is not allowed for
audio, and local pressure (a slow socket, worker or plugin) is never a
non-retryable error: the queue drops old audio and reports it as a gap, and a
pump that hands audio to a slower consumer awaits it instead of refusing.

Cancellation, shutdown and required completion delivery remain serviceable
while data capacity is exhausted. Do not create a wait cycle where releasing
capacity requires an event blocked behind that capacity. EOF and flush retain
their stream ordering; they cannot overtake preceding data as if they were
urgent cancellation. Host synchronous callbacks never block their owning event
loop waiting for queue space.

Bound work per dispatch and service ready control work under sustained media
load. Do not spawn one task per item merely to cross a binding or evade a full
queue. Reuse buffers when ownership permits; avoid repeated whole-buffer copies
and linear front removal on hot queues. Measure allocation, copying and queue
latency before adding a more complex optimization.

Runtime schedules expensive CPU work and declared blocking integrations through
bounded workers when they would stall shared execution. Respect host thread
affinity and propagate invocation context. Pure algorithms remain synchronous
and executor-free; do not add a task per frame to offload trivial work.

Expose occupancy, capacity, oldest-item age, blocked time, rejection and drop
counts through observations. Observation export cannot block core progress.
Tests hold consumers stalled, saturate producers and cancel/close under load;
assert bounded retained work and eventual cleanup, not just successful sends.
## Request options, timeouts and retries

`lk-agents-net::tls` owns the native HTTP, WebSocket and WebTransport trust configuration.
The first use loads the system roots, or the explicit `SSL_CERT_FILE`/`SSL_CERT_DIR`
locations, once; subsequent clients reuse the cached verifier. Certificate changes take
effect after a process restart. Protocol-specific ALPN settings do not change the trusted
roots. Rust `livekit-api` receives the process HTTP client's connection pool and TLS
configuration. The Python host exports its CA paths when `run_app` starts so the
shared aiohttp session used by Python `livekit-api` and native clients use the same paths.
There is no Python OTLP transport: the exporter is Rust (`lk-agents-net/src/export/otlp`) on
the same native TLS configuration. The explicit insecure
worker WebTransport option is restricted to loopback destinations; `localhost` is pinned
to a loopback IP for that option so name resolution cannot send unverified traffic elsewhere.

`APIConnectOptions` is the shared immutable per-operation policy. Declare its
fields, defaults, validation and timeout error categories once in ABI and
generate the Python and Node APIs. Native and host providers consume the same
resolved policy; plugins do not maintain parallel definitions.

| Field | Scope |
| --- | --- |
| `pool_timeout` | Wait for transport capacity or a pooled connection |
| `connect_timeout` | Establish a usable connection, including DNS and TLS |
| `write_timeout` | Wait without progress while writing pending bytes |
| `read_timeout` | Wait without progress while a remote response is expected |
| `total_timeout` | Entire operation, including admission, retries and fallback |

Each timeout accepts a finite nonnegative duration in seconds or `None` to
disable that limit. Keep the limits' defaults at 10 seconds and `total_timeout=None`.
Keep `max_retry=3` and `retry_interval=2.0`; `max_retry` counts additional attempts
after the initial attempt. Retry counts are nonnegative integers. Reject invalid
values at the generated boundary before starting work.

Resolve provider defaults and explicit call options before execution. A supplied
options object replaces the provider's default policy in full. Forward the
resolved values without mutation or hidden SDK overrides.

### Timeout lifecycle

Phase limits apply to their own waits. Read and write limits restart on
transport progress, rather than waiting for a complete token or protocol event.
Writing does not time out while waiting for the application to produce input.
Downstream backpressure is not remote read inactivity. Passive STT/realtime
listening for future user input does not start a request read timer; the
protocol identifies when a response is expected. Heartbeats do not satisfy an
outstanding application acknowledgement or final result.

The total deadline starts when the operation is accepted for execution. It does
not reset on progress, retries, reconnects, stream pulls or backend fallback.
Phase limits cannot extend it or an inherited owner deadline. Use supplied
monotonic time for core policy; runtime executes timers and reports correlated
outcomes. Timeout initiates cancellation and bounded cleanup under the shared
operation lifecycle; it is not proof that remote work or cleanup has stopped.

### Retries and adapters

One owner per request chain makes retry decisions and keeps the accounting. Disable underlying SDK
retry loops for framework-managed requests. Keep delays and eligibility in the
shared policy rather than binding-specific branches. Fallback preserves the
operation's total deadline and declares each backend's attempt allowance.
Cancellation or an expired total deadline prevents further attempts.

Retry eligibility is separate from safe replay. Do not repeat externally
visible output or an uncertain side effect without a protocol that prevents
duplication. Preserve idempotency identity across retries where supported.
Model connection options do not implicitly retry tools or business actions.

Transport adapters, in Rust the shared `lk-agents-net` client, enforce the
declared timeout scopes and preserve the expired limit and cause in errors. Implement missing transport support or reject an unsupported
applicable option before I/O; never silently substitute a different scope.
HTTPX phase controls are not an overall deadline, and Reqwest's pool idle timeout
is not a pool acquisition timeout. See the [HTTPX timeout definitions](https://www.python-httpx.org/advanced/timeouts/)
and [Reqwest client controls](https://docs.rs/reqwest/latest/reqwest/struct.ClientBuilder.html)
for adapter APIs.

Test shared policy with virtual time and real adapters with controlled peers.
Cover stalled phases, duplex idle periods, consumer backpressure, disabled
limits, retries and fallback sharing a total deadline, cancellation, partial
output safety and generated error conversion in both bindings.
## Session I/O composition

`AudioInput`, `AudioOutput`, video and text endpoints are capabilities using the
same runtime primitives as provider plugins. An endpoint can be implemented in
Rust, Python or Node. Transform and attachment lifecycle are independent of the
implementation language. Playback output additionally reports actual progress,
cut and completion; accepting frames does not establish that they were played.

Core crates link neither `livekit` nor `livekit-ffi`. The host SDK owns rooms,
participants, tracks and audio sources. ABI declares their data/capabilities;
runtime proxies operate on live SDK objects. Pure media code processes supplied
frames without acquiring host objects.

A session owns attachments, media routing, playback/transcript accounting and
recording. Applications replace a selected source, transform or destination;
they do not reconstruct a hidden wrapper chain to preserve framework behavior.
Recording and telemetry observe stable session-owned taps, independently of the
currently selected destination.

```text
input source -> input transforms -> recognition
                   |
              session input tap

generated audio -> output transforms -> session routing -> playback destination
                                              |
                                    session output/recording tap

playback destination -> correlated progress/cut reports -> session accounting
```

An input transform that fails, such as noise cancellation, reports a typed
processing error (logged and observable) and the input continues with
unprocessed frames for that span. The input stream does not end and recognition
keeps receiving audio.

Replacing an endpoint changes only that attachment. It does not implicitly
remove recording, reset transcript history or replace unrelated transforms.
Attachment generations correlate frames, progress and cleanup with the correct
endpoint. Late reports from an old endpoint cannot advance a new one, but the
session still accounts for already delivered old media.

Selecting a replacement and revoking existing work are separate decisions.
Work permitted to continue retains its original endpoint binding and lifetime.
Pause, resume, stop and cleanup target that binding, including after selection
changes.

Recording keeps one session timeline across endpoint changes. Its outgoing
media accounting combines routed frames with reported playback progress; it
does not equate generated or accepted audio with heard audio. Sink capabilities
must state the playback information they can actually provide. A recorder cannot
infer precise remote playback where no such observation exists.

Endpoint replacement and recorder finalization have independent lifecycles.
Recorder queues have bounded capacity. Overflow fails the recording with a
typed event, releases buffered PCM and cleans up within its deadline while the
conversation continues. Other recording failures are reported explicitly.
Close waits for required recorder cleanup under its deadline.

The frontend contract is documented in [VoiceSession](../voice-session.md).
Tests replace native and host-language outputs during generation, buffering,
playback and close, verifying frame ownership, stale-report handling, transcript
alignment, continuous recording and cleanup.
## Managed turns

`lk_agents_core::a2a` decides arrival order, inbox policies, task transitions
and the writes each step needs. `lk-agents-storage` executes database requests,
fencing and lease renewal and reports typed errors; the runtime's A2A door
executes core's storage effects through it and the transport writes. Both language hosts use this
runtime-to-host handler bridge:

```text
handle(TurnContext) -> (Sink<Verdict>, Source<TurnEvent>)
```

These streams belong to the runtime bridge, not to a core. The A2A decisions
retain the synchronous `handle(event, now) -> effects` boundary.

The handler emits checkpoints before model calls and answer delivery. The turn
answers with `continue` or `yield` and consumes the handler's final output.

| Record | Contents |
| --- | --- |
| `TurnContext` | Thread/partition identity, incoming messages, history, explicitly saved fields, transport tasks and permitted caller history |
| Handler output | New history items, typed output or input-required outcome, usage, staged field writes and failure/yield information |

Generate these boundary records from ABI declarations. Make submission durable
before acknowledging it. Observed lease loss stops new work and cancels the
owning invocation; database fencing protects managed writes independently of
that observation. Worker loss does not implicitly rerun application code.
External actions follow the application transaction contract. Storage failures follow the
[storage contract](storage.md#the-unit-of-work).

The application constructs its context and explicitly loads/saves fields.
Only explicitly staged application fields join the managed write set. An explicit
`store.bind(app)` activates native session persistence and a live SessionState view; detached
reads do not bind. Routing frames and immutable history references are checkpointed under the
existing fence without committing application drafts. Python facades are generated from ABI;
dataclass discovery and application-object identity remain host adapters. No Python execution
stack is persisted. The handler checkpoint protocol
is separate from the public [semantic run stream](../agents.md#semantic-events).
## Telemetry

Rust owns telemetry end to end. [Telemetry internals](telemetry.md) is the full
contract: data flows, destinations, the span API, the naming registry, consumers
and verification. Placement:

| Crate | Owns |
| --- | --- |
| `lk-agents-abi` | Span, event, metric and record declarations (`telemetry::spans`); the attribute wire types; `SpanContext` |
| `lk-agents-telemetry` | Span recording: `ManualSpan` and `Trace` for sans-I/O machines; the tracer, `Cx`'s recorder, `ScopedSpan` |
| `lk-agents-core` | Destination policy, admission and byte charges; report records and tags |
| `lk-agents-voice` | The session's spans, each owned by the entity it describes (session, user turn, speech) |
| `lk-agents-runtime`, `-inference`, `-worker` | `ScopedSpan` spans at their I/O choke points; activating the current span for host calls |
| `lk-agents-net` | Export (`export`): the OTLP exporter, per-destination queues and metric aggregation; the session report and recording upload |
| `lk-agents-py` | The generated tracer, span and provider classes behind the Python OpenTelemetry API |

Rules:

- Parentage comes from explicit data: a `ManualSpan` parent handle, a `SpanContext`
  in an effect, the `cx` of a `ScopedSpan`, the Invocation of a host call, or a
  validated W3C `traceparent`. No span is found through a process-global or
  thread-local current span.
- Sans-I/O crates take time and span ids as inputs. Recording a span or a
  measurement is synchronous and never waits on export.
- Export is best-effort and bounded. A full queue drops and counts; an export
  failure cannot fail a conversation; flush and shutdown share the job's cleanup
  deadline and never block the session loop or the worker heartbeat.
- Each destination exports an independent snapshot under its own policy. Cloud
  export is gated per job; disabling it leaves the application destination
  unchanged.
- Declared metric labels are bounded by type. Ids, user content and error
  messages are never labels; the Cloud destination stamps job identity on its
  per-job series as attributes, outside the label set.
- Applications keep their own OpenTelemetry providers. The framework installs
  its provider only when the application set none, attaches one processor to an
  application SDK provider, and never shuts one down.
