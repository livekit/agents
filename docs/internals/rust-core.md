# Rust core

Rust owns framework decisions, I/O, IPC, storage, inference clients and media
processing. Decisions live in sans-I/O crates; I/O lives in the runtime, which is
the I/O layer of voice and text, and in async I/O crates for everything that can
be faked at its boundary. The Python host owns its developer API, application callbacks,
Python-authored provider implementations, SDK objects and process launch/bootstrap.

The workspace is `rust/`. The Python package is `livekit-agents-next/`, which
provides `livekit.agents` and includes the native `_core` module. Plugins depend
on that package. Node bindings use the same ABI/runtime contracts, with
`lk-agents-py` as the reference binding.

The [feature documents](../README.md) define public API behavior. This document
defines the Rust implementation boundaries and shared contracts that realize it.

Read the [core boundary](#cores-and-the-runtime), [binding semantics](#shared-binding-semantics)
and [component contracts](#component-contracts) for implementation ownership;
use [test boundaries](#test-boundaries) and [codebase rules](#codebase-rules)
for validation and review.

## Ownership boundaries

| Layer | Owns | Boundary |
| --- | --- | --- |
| Decision cores | Agent execution, tool/approval/output policy, session lifecycle, scheduling, MCP and A2A task decisions | Synchronous events in, effects out; no I/O, executor, or host objects |
| Data-plane core | Media transformations, buffering, gates, fan-out and playout accounting | Frames and control data in, transformed data and effects out; no device or network access |
| Async I/O crates | The shared transport, the host stream adapter, the delegation A2A client and the Cloud session report upload (`lk-agents-net`); provider clients with retry, fallback and reconnect execution, provider-side media execution and evaluation judge orchestration; worker registration, IPC and supervision; storage (all database interaction, reported as typed errors) | Ordinary async Rust, tested on their own against real peers and processes and faked in the DST; host access only through declared proxies |
| `lk-agents-runtime` | The I/O layer of voice and text: voice session executor, room I/O, avatar transport and recording ingress, AgentRun and TextSession drivers, A2A door and MCP session I/O, HostInvoker and the process runtime | Executes core effects, returns correlated events and wires the async I/O crates; makes no decisions |
| `lk-agents-py` | Python module registration, generated conversions and adapters, and HostInvoker's Python entry | The only crate containing pyo3; no framework policy or I/O implementation |
| Python package | Public API, user callbacks, provider implementations, host SDK objects and process launch/bootstrap | Calls `_core`; host objects are reached through generated proxies and HostInvoker |
| Build tools | Binding generation and architecture checks | Read export declarations from ABI; no product execution responsibility |

A provider implementation may perform its own I/O in Python. Framework access
to that provider goes through a generated proxy entered by HostInvoker; a core
receives only data and resource IDs. The same boundary applies to application
tools and host RTC objects.

## Crates and dependencies

| Crate | Responsibility |
| --- | --- |
| `lk-agents-abi` | Public export declarations and typed boundary records: chat items, events, effects, errors and resource descriptions; contracts and data only |
| `lk-agents-core` | Shared AgentRun, request planning, tool/approval/recovery and output acceptance; backend TextSession routing/frame composition; A2A endpoint policies and task transitions (`lk_agents_core::a2a`) over ABI-declared storage request/result records and error types; MCP tool listing, schema rendering, result and error mapping, pagination, close decisions and HTTP response classification; chat formatting, tool-argument processing, JSON repair, tool search and deterministic run/event assertions; telemetry destination policy and metric admission (`lk_agents_core::telemetry`); no voice lifecycle or transport dependency |
| `lk-agents-voice` | Frontend/session lifecycle, recognition, speech scheduling, interruption, IVR, AMD and media coordination; speech tokenization, markup and pacing decisions; composes agent execution from `lk-agents-core` |
| `lk-agents-storage` | Async I/O: all database interaction: schema and statements, the AgentDB and SQLite backends, retries, fencing, lease renewal and lost-acknowledgment resolution; every failure is a typed error: the database's `Uncertain`, `Rejected { running_task_id }`, `LeaseLost`, `Unavailable`, `NotFound` and `VersionConflict`, the caller's `Invalid`, `NotSerializable`, `TooLarge` and `Unauthorized`, and `Cancelled` |
| `lk-agents-media` | Synchronous decoding, encoding, TTS audio framing, image preparation and conversion, resampling, media filters, recording files and sample arithmetic; optional native codecs; synchronous CPU work with no async runtime, and a plain bounded thread pool only where a measured need exists |
| `lk-agents-net` | Async I/O: the shared transport: HTTP with phase deadlines, SSE, WebSocket connect, owned tasks, the shared attempt machine and gateway token signing; the host stream adapter through which native sources reach the host (one pull at a time, a close that settles the pull in flight); the delegation A2A client for remote agent calls; the LiveKit Cloud session report and recording upload with its own bounded admission; the process environment value (`lk_agents_net::env::ProcessEnv`) and the blocking I/O admission it lends |
| `lk-agents-inference` | Async I/O: provider clients for LLM, STT, TTS, VAD and EOT models, interruption, realtime, duplex and avatars, with their retry, fallback and reconnect execution; provider-side media execution: the TTS audio emitter (decode admission, drift timer, segment finalization), the streaming decoder's bounded output queue and provider image preparation on the injected media pool; speech stream adapters; evaluation judge orchestration |
| `lk-agents-worker` | Async I/O: worker registration and job assignment, IPC, process pool and supervision, CLI and endpoint serving |
| `lk-agents-runtime` | The I/O layer of voice and text: the voice session executor and its effect execution, room I/O, avatar transport, recording ingress and the console port adapter, AgentRun and TextSession drivers, the thin A2A door (executes core's storage effects through `lk-agents-storage` and feeds the typed results back), thin MCP session I/O over the SDK, HostInvoker and the process runtime with its fork guard and media pool instance; wires the async I/O crates and re-exports the implementations generated bindings select |
| `lk-agents-py` | Python extension registration, generated pyo3 conversions and adapters, HostInvoker's Python entry; the only crate containing pyo3 |
| `lk-agents-bindgen` | Binding declaration parser and the resolved declaration IR |
| `lk-agents-host-derive` | Inert `Host`, `host` and `Vocabulary` marker attributes read by codegen |
| `lk-agents-codegen` | Python models, stubs, provider bases, forwarders and architecture checks |
| `lk-agents-fuzz` | Deterministic simulation of the sans-I/O crates and behavioral contracts, running the runtime's real executors over faked SDK and provider ports |
| `lk-agents-test-support` | Simulated RTC peers (the SDK-port fakes) on a supplied clock, shared by the DST and the crates' own tests |
| `lk-agents-bench-steps` | Direct step measurements of the voice data plane; compiled in the launch gate, measured explicitly on qualified hardware |

```text
lk-agents-py -> lk-agents-runtime, lk-agents-abi
lk-agents-runtime -> sans-I/O crates, async I/O crates
lk-agents-voice -> lk-agents-core -> lk-agents-abi
lk-agents-voice -> lk-agents-media
lk-agents-inference -> lk-agents-media
lk-agents-inference -> lk-agents-core
lk-agents-net -> lk-agents-core
inference, worker, storage -> lk-agents-net
async I/O crates -> lk-agents-abi
fuzz, test-support, bench-steps -> sans-I/O crates + lk-agents-abi
lk-agents-fuzz -> lk-agents-runtime (its executors), lk-agents-storage (real storage under the A2A DST), lk-agents-test-support
every crate's tests -> lk-agents-test-support
```

The sans-I/O crates are `lk-agents-core`, `lk-agents-voice` and
`lk-agents-media`. `lk-agents-media` is a synchronous data-plane core with no
async runtime, so voice and inference compose it like `lk-agents-core`; inference
also uses `lk-agents-core`'s chat formatting, strict tool schemas and JSON repair
for the request bodies it sends.
`lk-agents-core` has no dependency on `lk-agents-storage`: its A2A decisions
emit storage effects over request/result records and error types declared in
ABI, and the runtime's A2A door executes them by calling storage.

`lk-agents-net` is the one shared async transport and hosts the delegation A2A
client: inference, the worker and the AgentDB backend use its HTTP client with
declared phase deadlines, SSE decoding, WebSocket connect, owned task handles
and gateway token signing rather than building their own. The delegation
client executes the client decisions of `lk_agents_core::a2a`.

The runtime depends on the async I/O crates to wire voice and text sessions:
providers, storage, the worker server, judges and the delegation client in
`lk-agents-net`. `lk-agents-py` depends only on the runtime and ABI: it uses ABI for the declared types its generated conversions translate, and the
runtime re-exports the implementations of declared classes that its generated
adapters select. Async I/O crates depend on ABI and never on the runtime,
`lk-agents-py` or pyo3; they reach host code only through ABI-declared proxies
and HostInvoker's declared port.

The runtime owns the process-wide Tokio runtime, the fork guard and the media
pool instance, and lends them as one value, `lk_agents_net::env::ProcessEnv`,
never through a dependency on the runtime. It is a plain struct:

| Field | Lends |
| --- | --- |
| `runtime`, `caller_runtime()` | The process runtime (started on first read, refused when fork-inherited) and the caller's |
| `spawner`, `clock` | The executor the host awaits a chain's futures on and the host `Clock` |
| `http`, `report_admission` | The shared HTTP pool and the session report admission |
| `media_pool` | The media pool, as the ABI-declared `MediaExecutor` (`lk_agents_media::pool::MediaPool` is its typed face) |
| `blocking_io` | The blocking I/O on the process runtime |
| `images`, `strip_assistant_markup`, `speech_text` | Image preparation and encoding, and the voice core's text algorithms providers and judges apply |

A declared class that needs the process declares `#[host(env)] type Env;` and
takes `env: &Self::Env` in its constructor or function; the generated binding
passes the runtime's value (`lk_agents_runtime::process::env()`), hidden from the
host's signature. Internal constructors take the handles they need; there is no
`Env` type parameter. Runtime-bound handles are read on use, so constructing a
class starts nothing. Rust callers and tests build their own value
(`ProcessEnv::current_runtime()`, with fakes as needed). The voice core's text
algorithms (`strip_assistant_markup`, `speech_text`) are the only functions it
lends from another layer: plain functions the runtime selects, so providers and
judges need no voice dependency; nothing registers into it at run time.

The sans-I/O crates and ABI have no runtime, executor, Tokio, futures,
async-trait or pyo3 dependency. Async I/O crates use Tokio, process and database
crates directly and reach the network through `lk-agents-net`. Runtime adapters depend on cores; cores never
depend on adapters. Build tools read declarations without adding runtime
dependencies to cores.

Reuse `livekit-protocol`, `livekit-api` and `livekit-token` from LiveKit's Rust
SDK. Protocol records may cross the ABI boundary; API clients belong to the I/O
code that uses them, and gateway token signing to `lk-agents-net`.

## Cores and the runtime

### Core interface

The composed core exposes these synchronous operations to its queue owner:

```text
handle(event, now) -> effects
wake_at() -> Option<Ts>
```

Each composed core defines its typed outer boundary. Related inputs may share
an outer envelope; private submachines keep domain-specific methods and outcomes.
Do not force all components into one global Event/Effect enum or require every
helper to return `Vec<Effect>`. Shared/exported contracts live in ABI; private
machine inputs and state remain with their owning subject module.

`now` supplies monotonic time and its wall-time mapping. Core-generated IDs use
counters seeded at construction. The core reads no clocks or entropy, starts no
tasks, takes no locks, and retains no channels, callbacks, provider objects or
futures.
Environment, process and OS state enter as supplied configuration or events.
Deterministic iteration order is required for all decision state.

Events and effects are typed data. Runtime-owned resources are identified by
`ResourceId`, with descriptions such as kind, capabilities, sample rate and
label. The runtime retains the actual providers, rooms, inputs, outputs and
recorders.

### Effect execution

The runtime owns the input queue and calls the core one event at a time. It
executes effects, waits for I/O or deadlines, and returns results as correlated
events. A due wake precedes another input at the same supplied instant; production
and simulation use that same rule. Other independent events remain schedulable.
Sessions share the runtime; there is no dedicated thread per session.

Runtime tasks serve asynchronous resource lifetimes and independent I/O. A
submachine does not need a task merely because it owns state. An effect names
an operation at a declared resource boundary; its adapter performs the required
transport reads/writes and SDK integration without turning every low-level I/O
step into another framework effect.

Each resource protocol defines command submission order. Independent resources
may progress concurrently. Dependencies on completion live in the owning core:
it authorizes dependent work after validating the required outcome. Position in
an effect collection does not imply that earlier work has completed.

Every outcome-bearing effect reports success or failure, including failure to
launch. Completion acknowledgments require the operation's declared
postconditions to hold. Late work cannot invalidate an acknowledged terminal
outcome.

Tokio drives I/O, not conversational policy. Interruptions, tool scheduling,
session and workflow deadline decisions belong to cores. Provider retries,
reconnects and fallback execute in `lk-agents-inference`, and process supervision
in `lk-agents-worker`, under the shared request and lifecycle contracts. An
adapter may sequence I/O and manage its resource lifecycle; it must not make
conversational or workflow decisions.

The runtime's executors (voice session, room I/O, avatar transport and their
drains and legs) use no Tokio runtime: no `tokio::spawn`, `tokio::time`,
`tokio::net`, `Handle` or `Runtime`. They spawn and sleep only through the
injected `Spawner` and `Clock`, so the [DST](#deterministic-simulation-testing)
runs the same code on its deterministic scheduler. Tokio's executor-agnostic
sync primitives (`watch`, `mpsc`, `oneshot`, `Notify`) are allowed, and every
`select!` is `biased;`: an unbiased select picks a branch at random. A socket
belongs to a boundary adapter, like the SDK room or the console's TCP port,
which the DST replaces with a port fake; everything above the port follows the
executor rules.

### Runtime layout

Every runtime module is I/O; the decisions it executes live in the sans-I/O
crates.

```text
lk-agents-runtime/src/
  lib.rs          module list; re-exports only the implementations lk-agents-py selects
  process.rs      process Tokio runtime, fork guard, media pool, the `ProcessEnv` it lends
  host/           HostInvoker: invocation lifecycle, scope admission, owned callback tasks
  agent_run/      AgentRun driver: runs effects (host calls, model streams); the declared AgentRunner
  text_session/   TextSession driver over agent_run's execution; the declared runner
  a2a/            turn: executes TaskMachine steps against storage and the handler;
                  door/: decodes, admits, calls storage, publishes response streams
  mcp/            the SDK session, stdio/host/HTTP/SSE transports, the SDK's HTTP client over lk-agents-net
  voice/          session executor (effects, legs, audio drains, storage, recorder observers)
    generation/         provider stream, TTS, playout and forwarding pumps; their decisions are voice's
    ivr.rs              IVR timers and feed over voice's navigator effects
    amd.rs              AMD timers, classification stream, tap stream, callee wait
    background_audio.rs mixer pump, file/stream reads, track publish/unpublish
    keyterm_detection.rs keyterm extraction's provider stream, reported to voice's detection pass
    console.rs          the console TCP port adapter
    remote_session/     host runtime loops and drain, room byte-stream transport
    workflows/          DTMF publish/collect; warm transfer dialing, room watching, provider HTTP via the host
  room_io/        RoomIo owner, participant input readers, audio and transcription output, pre-connect receiver
  avatar/         data-stream audio output, receiver and runner, each implementing its ABI class directly
  rtc/            room event fan-out and stream ingress admission, room waits, playout ledger lock
  recording.rs    recorder ingress and its handoff onto the media pool, terminal status
```

Tokenization, telemetry policy, A2A and MCP decisions, workflow and avatar
reply rules, recording bounds, codecs, the TTS emitter and image preparation do
not belong here: they live in voice, core, media or inference. Host video-frame
reading is a declared host proxy handed to the process at initialization, not
pyo3 code in the runtime.

### State-machine composition

A core is composed of smaller synchronous state machines. Each machine owns
one responsibility and its state: turn detection, speech scheduling,
interruption, agent execution, model requests, tool execution or lifecycle.
The parent owns event routing and composition, not all child decisions.

```text
Session core
├── Turn detection
├── Speech scheduling
├── Interruption
├── Agent execution
│   ├── Model request lifecycle
│   └── Tool execution
└── Session lifecycle
```

Submachines own their state and expose synchronous transition methods. They
have no independent task, channel, executor or event loop. A machine does not
mutate another machine's fields.

Keep lifecycle state private and phase-specific data together. Private phase
enums and consuming transitions serve different components; non-cloneable types
can enforce ownership transfer or permission to issue an operation. Independent
facts and synchronous accumulators retain their own data structures. Choose the
representation per component and validate it against its failure cases.

The parent coordinates child transitions through typed calls and outcomes,
collecting requested runtime effects. Internal signals may be buffered when
composition needs follow-up dispatch. The parent owns that local buffer and
processes signals in emission order until settled; no runtime I/O occurs during
this phase. Effects leave the outer `handle` for runtime execution after
synchronous decisions have settled. Internal return types and collectors are
implementation details, not a uniform submachine API or generic framework.

Use distinct types for internal communication and runtime requests. A private
collector may hold both while composing a step, but only external effects cross
the core/runtime boundary. Superseding work during settlement preserves its
caller outcome and cleanup responsibility; coalescing cannot erase an accepted
request or bypass its cancellation protocol.

```text
external event
  -> child transition
  -> buffer internal signals and runtime effects
  -> dispatch internal signals until settled
  -> return buffered effects to runtime
```

All transitions in a step use the same supplied time. The parent's next
deadline is the earliest deadline of itself or any child. Work follows what
changed; a VAD tick must not scan every subsystem.
Each step terminates. Internal signal dispatch and due-wake processing consume
work or advance state/deadlines; zero-time cycles cannot monopolize dispatch.

Submachines are tested directly, and composition tests verify signal ordering,
cancellation and deadline propagation. The simulation scheduler controls
external event delivery; it does not reorder synchronous internal signals.

### Events, signals and effects

| Term | Meaning | Example |
| --- | --- | --- |
| Event | Input delivered to the core by its queue owner | Model response received, playback stopped, deadline wake |
| Internal signal | Synchronous communication between machines during a core step | Turn detection reports `UserStartedSpeaking` to interruption handling |
| Effect | Declarative request for the runtime to perform work outside the core | Call a model, invoke a tool, cancel playback, publish a host update |

Effects describe requested work; they contain data and resource IDs, not
executable closures. Results and failures return as events. Internal signals
are consumed by the parent within the step; they never reach the runtime.
There is no conversion from an effect into an internal signal. Runtime results
may cause a subsequent core transition to produce a signal.

An internal state change alone does not require an effect. Helper computations
remain ordinary synchronous functions; they need no event or output wrapper.

### Requests and host views

Every scheduled API operation, hook, tool and provider request has a correlation
ID. Its owning core tracks it until an observed outcome settles it. Cancellation
and timeout requests do not establish cleanup completion. IDs are unique within
their declared scope and generation. Late answers cannot change settled
decisions; resources returned by late answers still require runtime cleanup.
Journal the disposition rather than silently adopting or leaking the result.

The runtime owns each resource-creation operation before launch and retains its
result through delivery, adoption and cleanup. Completion events carry IDs and
data; the runtime retains pending resources until adoption or retirement is decided.
Ownership transfers only when the recipient accepts it. Failed or refused
handoffs preserve the sender's cleanup responsibility.

Every accepted caller receives a correlated outcome. Repeated requests have
defined join, refusal or supersession semantics. Replacing pending work settles
its existing callers according to that contract. Successful close includes the
required resource cleanup.

Synchronous host properties, including speech state and resolved options, are
runtime-maintained views updated by core effects. `interrupt()` may immediately
record its request flag; the core determines the speech outcome, and a done
speech's view reads that outcome. The voice core owns each speech's state; the
host's requests and a reply's step reach it as queued inputs. A host request on
a speech (`interrupt()`, the interruption setters) is ordered at its call: the
runtime steps every request made before an input ahead of that input, and none
lands between a step and the speech views it publishes, so however a speech
ends, an `interrupt()` asked while the host read it running lands on it. A
speech created done (refused, or answered as the session closed) never ran: an
interrupt asked before its creation lands nowhere.
Other option changes become visible after the core accepts them. Tests compare
host views with settled core state.

### API validation and authority

| Boundary | Responsibility |
| --- | --- |
| ABI and generated entry adapter | Shared framework shape, field, default and range validation; equivalent native inputs use the same validation rules |
| Runtime | Resource identity/lifetime, invocation scope and correlation; retain captured context before scheduling |
| Owning core | Current lifecycle, generation, approval and eligibility decisions; a valid input shape does not authorize work |
| HostInvoker | Framework-invoked application validators and output functions execute as host code; return their result to the owning core for acceptance |

Cached host views and pre-queue checks cannot authorize core-dependent work.
The owner validates the request against its current state, including after
replacement or close. Keep framework validation shared without translating
arbitrary Python/Pydantic or Node application callbacks into Rust. Preserve the
public API's acceptance and visibility boundaries as these layers change.

## Component contracts

Components are organized around decisions and lifecycles. Each mutable fact has
one owner. Another component can observe a snapshot or send a typed input; it
cannot keep a second authoritative copy or mutate the owner's fields.

A component has a separate state machine only when it owns an independent
lifecycle. Parsing, formatting, validation and arithmetic remain synchronous
functions. Internal APIs use domain-specific methods and outcomes; no generic
machine trait, signal bus or universal result wrapper is required.

### State ownership

| Fact | Authority |
| --- | --- |
| Agent instructions, tools and output contract | Shared Agent definition |
| One model/tool conversation's progress and accepted output | Agent invocation |
| Backend parent/child routing and private histories | TextSession state supplied by the application |
| Active frontend role, floor and temporary live interactions | VoiceSession composition |
| Speech eligibility, authorization and terminal outcome | Speech scheduling/lifecycle machine |
| User transcript and recognition progress | Recognition child machines |
| Audio accepted, played or discarded | Data-plane accounting, using output observations |
| Remote provider's acknowledged conversation | Provider protocol machine |
| Live sockets, processes, SDK objects and host tasks | Runtime resource owner |
| Committed history and application data | Storage under the application's explicit save contract |

Host properties are views of these authorities. Observation records are not a
second control path. Persisted state is a committed snapshot, not an object that
runtime tasks may mutate independently of its owner.

### Agent execution

An Agent definition is reusable configuration. Each invocation has its own
identity, conversation view, runtime-owned application context reference, model
requests, tool calls, usage and accepted output. Neither definitions nor runtime
adapters own invocation progress.

`lk-agents-core` owns `AgentRun`, a synchronous state machine instantiated for
each agent invocation. It composes smaller model-step, tool/approval and output
machines. Direct `Agent.run()`, TextSession and VoiceSession reuse its code with
separate invocation state; the reusable Agent definition holds configuration.
When composed inside a session, AgentRun has no independent task or queue.

`TextSession` also lives in `lk-agents-core`. It is a separate request-scoped
coordinator that owns backend frames, parent/child routing and history selection,
and composes AgentRun for each invocation. It accepts explicitly supplied saved
state and returns updated state. Runtime resolves agent resources and performs
I/O; application code explicitly loads/saves session state. AgentRun and
TextSession do not compose A2A lane orchestration or VoiceSession; the separate
`a2a` module decides A2A tasks over ABI-declared storage records. Direct agent runs and voice execution
use AgentRun without requiring TextSession.

These entrypoints use one shared Rust implementation of agent execution:
request planning, model/tool step decisions, argument preparation, approval,
result interpretation, usage, recovery and output acceptance. Sharing contracts
or helpers while maintaining separate text and voice implementations of those
decisions does not satisfy this boundary.

The shared core has no session back-reference or audio lifecycle dependency.
Conversation coordinators supply explicit history snapshots, admission,
cancellation and delivery outcomes. TextSession owns routing/rehydration;
VoiceSession owns recognition, floor assignment, interruption and presentation.
Completion of an agent invocation is separate from speech/playback settlement.

Model output is incremental. Voice may synthesize and play authorized content
while generation continues; the common execution path cannot require collecting
a full text response before exposing it. Realtime models keep their protocol
state and response-initiation rules while composing the same tool, approval and
output machinery. A server-generated follow-up cannot also trigger a duplicate
client-generated request.

| Component | Owns | Contract |
| --- | --- | --- |
| Agent invocation | Invocation phase, request identities, model/tool progress and final outcome | Consume typed model/tool outcomes; request more work or return completed/input-required; no rooms or audio in this contract |
| Request planning | Construction of a request from the allowed history, instructions, tools and options | Pure transformation over an explicit snapshot; never reads mutable session state through a back-reference |
| Tool calls | Validated arguments, approval binding, call phase, progress and terminal result | Host executes the body; every result is correlated; changes to approved terms require fresh approval |
| Output acceptance | Validation and application acceptance of the final value | Conversational text is distinct from accepted typed output; rejection returns explicit feedback |
| Context construction | Application dependencies and explicitly persisted fields | Host creates runtime dependencies; load/save is explicit; control state never serializes host objects |
| TextSession | Backend routing frames, pending child calls and context/history ownership | New human input starts a fresh invocation from supplied state; child completion resolves the original parent call |
| Frontend remote interaction | Pending remote request identity and reply correlation | Owned by the frontend conversation; delivery eligibility follows the active agent's configured remote tool |

#### Shared tool execution

Both conversation modes use one tool preparation and execution path:

```text
model tool call
  -> resolve tool, parse/canonicalize and validate arguments
  -> prepare host arguments/context and validate application types
  -> bind approval and obtain the owning coordinator's admission
  -> runtime/HostInvoker executes the tool body
  -> normalize result/error/progress into correlated outcomes
  -> shared core records the outcome and decides continuation
```

Framework parsing/defaults, validated argument identity, approval binding and
result semantics live in shared Rust code. Python/Node signature binding,
application validators and object construction use the common host adapter
through `HostInvoker`. These language-specific operations are shared by text
and voice too; neither mode implements its own preparation wrapper. Approval
binds to validated arguments, not raw model JSON, and is checked before the body.

`HostInvoker` owns host execution context, cancellation and settlement; it does
not decide tool policy or whether to request another model response. Tools,
instruction functions, output functions and custom nodes use that same runtime
invocation mechanism. Tool ownership is explicit: session-owned background work
can outlive a voice turn without being owned by a completed invocation.

Keep recovery at its owning layer: `lk-agents-inference` executes provider retry
and fallback under the shared attempt policy; shared agent/tool logic turns argument errors and `ModelRetry` into
model feedback; runtime owns failed launch and cleanup settlement. Both modes
use those implementations rather than separate retry or exception branches.

A cancelled or timed-out tool may already have performed an external effect.
The framework reports that uncertainty without retrying a non-idempotent body
as though it had not run. Business operation identity and reconciliation remain
application contracts.

Tool lifecycle and progress are recorded as outcomes arrive. Model-order result
assembly is a separate responsibility: waiting for an earlier sibling's result
cannot prevent recognizing another call's detach, update or completion.

Tests cover nested/direct invocations, multi-turn input, approval changes,
invalid output, accepted completion, cancellation, duplicate/late completion
and context isolation.
Run the same argument/default/type/approval, recovery and outcome vectors through
both real text and voice entrypoints. Also verify streaming before model EOF,
voice interruption/admission, background-tool lifetime, playback settlement and
realtime follow-up without duplicate requests. Reuse the shared cases while
asserting each coordinator's distinct lifecycle.

### VoiceSession and AgentActivity

VoiceSession owns the frontend conversation and composes recognition, agent
execution, speech scheduling, interruption and lifecycle machines. AgentActivity
owns one active role's coordination. Neither becomes a container for all child
state or a second implementation of its children's decisions.

| Component | Owns | Contract |
| --- | --- | --- |
| Session lifecycle | Starting, running, draining and closing phases; outstanding lifecycle requests | One transition authority; close settles callers and reports resource/cleanup outcomes even when a hook fails |
| Active role | Agent activation identity, ordered replacement requests and held user input | Replacement respects exit/drain/entry semantics; input is explicitly delivered, held or rejected, never lost during a swap |
| Speech scheduler | Queued speeches, authorization, holds and current floor assignment | Speech cannot generate/play without permission; every speech reaches one terminal outcome |
| Interruption | Interruption eligibility and pause/resume policy for a particular speech | Distinguish requesting a cut from observing its completion; restore the correct speech or explicitly abandon it |
| Hooks | Pending callback identity, deadline and owning activation | Runtime invokes the callback; a late answer cannot affect a replacement activation |
| VoiceTask | Temporary frontend interaction and its typed result | Parent remains suspended in memory; restore it without replaying its enter/exit hooks |
| Filler scheduling | Filler scope, eligibility and deadlines | Filler obeys floor ownership and ends with its originating operation |
| User/agent state views | Derived externally visible state | Core transitions determine the view; delivery to observers cannot alter policy |

Drain and close are different operations. Drain waits for the agreed activity
to finish while preserving the session. Close stops admission, settles or
cancels owned work and waits for resource cleanup within its deadline. Expiry
reports incomplete cleanup, not successful release. Repeated requests join the
existing operation rather than start competing shutdown sequences.

Tests cover replacement during recognition, generation, tools and playback;
overlapping holds; hook failure; simultaneous close/drain; queued remote events;
and cancellation before and after resource creation.

### AudioRecognition

AudioRecognition is a coordinator with small owned children. Its job is to
interpret speech and propose user turns, not authorize replies or operate
providers. It contains detector IDs and capabilities, never detector objects.

| Child | Owns | Inputs and outputs |
| --- | --- | --- |
| Transcript accumulation | Final/interim/preflight text, language/confidence, held text and audio anchors | STT observations and gate changes in; transcript snapshots/updates out |
| Speech activity | User/agent speech boundaries and the current speech identity | VAD, STT and playback observations in; speech-boundary outcomes out |
| End-of-turn | Turn phase, pending prediction, countdown, manual-commit identity and agent-backchannel opportunity | Speech/transcript/prediction/time inputs in; prediction requests, correlated turn proposals and backchannel opportunities out |
| Overlap | Active overlap identity, verdict and interruption guard | Agent/user speech edges and detector outcomes in; backchannel/interruption outcomes out |
| Endpointing | Fixed settings or learned pause estimate | Speech timing in; delay bounds out; no timers or provider invocation |
| Transcription completeness | Transcript wait and timeout for the relevant speech/commit | Final transcript, speech end and timer inputs in; completeness/timeout outcome out |
| Turn limits | Accumulated words/duration for the defined user-turn scope | Transcript and boundary inputs in; limit-exceeded outcome out |

Each child changes its own state. The coordinator translates child outcomes
into inputs for other children in a defined synchronous order. It does not
reach into their fields. Related fields move together when extracting a child;
separate files alone do not satisfy this ownership contract.

Recognition proposes a turn with an identity and transcript snapshot. Activity
scheduling decides whether to reply and returns acceptance/refusal against that
identity. Recognition retains the transcript required by that decision; neither
side silently consumes data the other still needs.

Prediction, overlap, audio-source and turn identities prevent stale answers
from affecting replacements. Keep audio duration, monotonic time and wall time
distinct. Closing can preserve trailing text without reopening prediction or
generation. Unsupported languages and provider failure have explicit fallback
outcomes.

Backchannels have two distinct paths:

- **User backchannel:** overlap classification recognizes a short acknowledgment
  during agent speech. The overlap child supplies that verdict; interruption
  handling decides whether speech continues. It is distinct from a request to
  interrupt or take a full user turn.
- **Agent backchannel opportunity:** the turn detector identifies a suitable
  moment for the agent to acknowledge the user. End-of-turn handling owns the
  prediction/threshold, deduplication and opportunity identity. Recognition
  reports the opportunity; speech selection and floor authorization remain with
  the activity/scheduler. An opportunity does not itself generate speech or
  commit the user's turn.

A pending full-turn proposal and a backchannel opportunity must be arbitrated
against the same turn identity. Do not schedule both as independent replies to
one prediction. New speech, turn acceptance, replacement and close invalidate
opportunities that no longer apply. Speech-boundary cooldowns and transcript
holds use the relevant speech/overlap identity.

Tests cover resumed speech, manual commit, final/interim reordering, held-text
flush, stale prediction and timer delivery, audio-source replacement,
user backchannels, agent opportunities, opportunity/full-reply arbitration,
transcript timeout, turn limits and final speech at close.

### Generation, playback and transcripts

Generation describes what may be produced. Playback tracks progress reported
at the configured output's declared boundary. A local RTC source queue is not
an observation of remote rendering. Preserve that distinction when reporting
progress, synchronizing transcripts and recording media.

| Component | Owns | Contract |
| --- | --- | --- |
| Generation | Attempt identity, model output assembly and dependent stage progression | Start dependent work from explicit outcomes; retries obey the visible-output and idempotency contract |
| Synthesis | Text-to-audio segment state and synthesis outcomes | Runtime invokes TTS; segments retain identity across buffering, failure and cancellation |
| Speech lifecycle | Relation between generation, tool work and playback | Model EOF alone cannot complete speech that still owes tools, playback or required cleanup |
| Data plane | Gates, bounded frame buffers, conversion and fan-out accounting | Pure processing of supplied frames; no network/device/file operations |
| Playout ledger | Accepted, reported-played and discarded frame/sample counts | Retain the observation boundary and any estimation; a requested cut is not an observed cut |
| Transcript synchronization | Visible text and segment alignment against declared playback progress | Preserve raw text separately; interruption commits the prefix supported by the output contract, not the full generated reply |
| Realtime voice coordination | Mapping server responses to speeches and local floor ownership | Provider protocol owns server state; voice owns presentation and interruption decisions |

Backpressure limits memory without preventing cancellation and close. Resource
and segment replacement use generations so delayed progress cannot update a new
output. Control changes do not require every media frame to traverse the
session's decision queue. The data plane processes media frames separately from
that queue; its boundary reports remain schedulable in simulation.

Tests cover cuts at segment boundaries, zero-audio segments, stalled outputs,
partial synthesis followed by failure, output replacement, delayed acknowledgments
and realtime truncation while the model continues generating.

### Provider protocols and inference

Provider state is separate from session state. `lk-agents-inference` owns
provider connections, credentials, model execution, request/connection generations
and the protocols below as async I/O, tested against real peers. Voice and text
cores receive typed results and resource IDs; the DST replaces these clients with
fakes calibrated against the same peers.

| Component | Owns | Contract |
| --- | --- | --- |
| Request/option resolution | Pure validation and request construction | Explicit environment/configuration inputs; typed known fields; opaque provider extensions stay at their declared boundary |
| LLM protocol | Incremental response/tool-call assembly and usage attribution | Partial/malformed output has an explicit outcome; usage is attributed to the originating request |
| STT protocol | Transcript/speech phase and finalization state | Audio EOF, server EOF and transcript completion are distinct inputs with bounded settlement |
| TTS protocol | Request/segment progress and finish state | Empty input, final flush, partial audio and provider failure have explicit outcomes |
| VAD/local model | Window/input state and interpretation of model output | Media executes the model; measured durations and model results drive window transitions |
| EOT protocol | Prediction correlation, supersession and fallback state | Only the current request applies; cloud/local replacement preserves cancellation and transcript ownership |
| Interruption protocol | Overlap request and verdict lifecycle | Delayed verdicts cannot restore a guard or interrupt an unrelated speech |
| Fallback | Backend availability, selection, recovery probes and attempt identity | One policy per request chain performs attempts and probes; late recovery cannot revive closed work |
| Realtime protocol | Server conversation mirror, pending responses, reconnect, cancellation and truncation | Track acknowledged versus desired state; reconnect restores the intended conversation without duplicating server items |
| Duplex adaptation | Gate/segment state for continuous provider output | Audio and transcript fragments retain their burst/segment identity |
| MCP | Tool listing, schema rendering, result and error mapping, pagination, close decisions and HTTP response classification (`lk_agents_core::mcp::http`), in `lk-agents-core` | The runtime's thin session I/O uses the SDK for stdio/HTTP/SSE and only executes; do not duplicate the MCP protocol engine |
| Avatar provider | Create/terminate request policy and provider session identity | Gateway requests retry under one operation identity; a created session is delivered or terminated; voice owns presentation |

Protocol decoding validates external input before core dispatch. Secrets,
connection handles and provider-specific host objects never enter core state.
A transport reconnect does not by itself decide whether a business request is
safe to retry.

Network chunks are not protocol messages. Decode text and assemble frames
incrementally across arbitrary byte boundaries, including split Unicode.
Test the actual HTTP/SSE/WebSocket adapter against the protocol's framing rules;
preassembled strings passed directly to a parser do not cover that boundary.

Tests use split/malformed frames, EOF, duplicate events, partial output,
reconnect, supersession, cancellation and delayed cleanup. Real-peer fixtures
validate fake behavior, not just the fake's agreement with its own generator.

### Rooms, media and bounded interactions

| Component | Owns | Contract |
| --- | --- | --- |
| Room coordination | Linked participant, track/publication generations and attachment lifecycle | Runtime subscribes/publishes through host RTC proxies; late operations clean up obsolete resources |
| Pre-connect audio | Buffered input ownership until attachment | Transfer frames once; normalize to the session format before the live processing chain; source-format processors declare that capability |
| User I/O and nodes | Validated capabilities and attachment identities | Runtime invokes host nodes; core chooses policy from supplied capabilities and outcomes |
| Avatar transport | Audio segment and acknowledgment state | Runtime owns byte streams; byte-stream id plus output generation identifies the original segment, without a new protocol field |
| Avatar runner | Publication, generator/receiver progress, clear-buffer and close state | Frames remain opaque; runtime owns generator/receiver/synchronizer calls |
| Background audio | Clip/loop/fade decisions and mix settings | Pure sample processing and file decoding in media; runtime publishes audio |
| Recording | Timeline/accounting and flush/finish lifecycle | Runtime admits captured frames and hands them to the media pool; media owns the recording file (timeline, encoding, ordered writes) and its bounds; finalization waits for accepted data and occurs once |
| Remote session | Request correlation, cancellation, deadline and drain state | Typed messages over runtime-owned room/TCP transport; late replies cannot settle new requests |
| AMD | Classification phase and deadline | Runtime invokes the model; classification coordinates with the existing session floor |
| IVR and keyterms | Their bounded request/pass state | Invoke model/provider operations as effects; replacement or cancellation invalidates stale results |
| DTMF/end-call/transfer helpers | Their bounded live-interaction state | Obey VoiceTask/session ownership; restore temporary audio and release transfer resources on failure as well as success |
| Codecs and text processing | Deterministic conversion over supplied bytes, frames or text | Synchronous functions or stream accumulators; no hidden files, sockets or threads |

RTC records and capabilities are declared in ABI. Real SDK proxies belong to
runtime; simulated peers belong to test support. Tests verify that modeled
subscription, publication, buffering and playout behavior matches real RTC.
E2E verifies delivered audio/video, recordings and room cleanup at the client.

### Worker and job processes

`lk-agents-worker` is async I/O, tested against real servers and real processes;
the DST uses a fake worker. Worker connection state, process supervision and pool
allocation are separate lifecycles. The server composes them rather than
duplicating their decisions.

| Component | Owns | Contract |
| --- | --- | --- |
| Worker state | Registration, assignments, capacity/status and drain | Every assignment has an explicit disposition; stale connection results cannot revive old assignments |
| Process supervision | Initialization, heartbeat, shutdown, failure and exit state | The worker owns sockets/pipes and OS lifecycle; host adapter only launches/bootstraps the process |
| Process pool | Warm capacity, reservations, pending jobs and retry policy | Assign only ready processes; release failed reservations; drain stops admission |
| Endpoint exchange | HTTP/WebSocket exchange parsing and state | The worker serves bounded ASGI I/O; handler execution remains a host callback through HostInvoker |
| Capacity and authentication | Parsing/load calculations and credential contracts | OS/environment inputs are read at the process boundary; credentials are signed/verified with the LiveKit token SDK |
| Server lifecycle | Coordination of worker, pool, endpoint and health resources | Resource readiness is explicit; startup/shutdown report each owned operation's outcome, including incomplete cleanup |

Tests cover duplicate jobs, failed launch, initialization timeout, heartbeat
loss, worker disconnect, drain, orphan cleanup and late exit. Real IPC tests
exercise partial/malformed frames and process crashes. Python preload and
launch semantics remain host-specific; IPC policy and transport do not.

### Storage and A2A

Persistence stores committed records. It does not restore stack frames or
establish exactly-once execution of external effects.

Storage uses the Rust `Storage` lifecycle trait and the internal
`SqliteConnection` execution trait. AgentDB and local SQLite adapters share
`Thread`, `Lane`, `Run` and `Lease` engine types, including typed content reads
that require no lane lease. The [storage plan](storage.md#rust-interface)
defines the interfaces, backend contracts and failure behavior;
[thread content reads](storage.md#reading-thread-items-and-tasks) distinguish
runtime handles from Cloud service APIs. Host-visible records follow the ABI
and generated-binding rules. Application writes follow the
[context transaction contract](../context.md#application-records-and-transactions).
Cloud supplies an authorized threads API for browser reads. Local SQLite
supplies backend thread reads; browser access requires application-provided
HTTP routes. A hosted read API is not part of backend storage parity.

| Component | Owns | Contract |
| --- | --- | --- |
| Storage backend | Thread creation, metadata, lookup/listing, deletion/expiry and database resolution | Cloud services or a local registry and thread files; the same thread visibility and lifecycle contract |
| Database connection | Request correlation and connection generation | The backend executes the wire protocol; disconnect settles requests without claiming an uncertain write was rolled back |
| Schema and statements | Typed records, validated statements and schema compatibility | Upstream schema is authoritative; no duplicate Python SQL or handwritten extraction of known fields |
| Lane ownership | Acquisition, renewal, fencing and release | Storage performs them and reports `LeaseLost`; stale owners cannot commit |
| Managed run | Loaded revision, staged writes and commit/abandon outcome | One fenced commit; explicit application saves; transport acknowledgment and commit boundaries are distinct |
| A2A task | Arrival policy, message deduplication, pending input and terminal transition (`lk_agents_core::a2a`) | Typed handler/checkpoint outcomes drive policy; completed transport does not imply a persisted Python continuation |

`lk-agents-storage` is a plain async crate. It owns all database interaction:
both backends, retries, fencing, lease renewal and resolution of lost
acknowledgments. Every failure reaches the caller as a precise typed error.
A database failure is `Uncertain` (the outcome is unknown), `Rejected { running_task_id }`,
`LeaseLost`, `Unavailable` (the one retryable error), `NotFound` or
`VersionConflict`; a request the caller made wrongly is `Invalid`,
`NotSerializable`, `TooLarge` or `Unauthorized`; a turn the host cancelled is
`Cancelled`. The A2A task
decisions in `lk_agents_core::a2a` are sans-I/O over request/result records and
error types declared in ABI; core has no dependency on storage. The runtime's
thin A2A door executes core's storage effects by calling storage and feeds the
typed results back. Storage is tested against real SQLite and a fake AgentDB
server with injected faults; the A2A DST runs the door over real storage, on an
in-memory SQLite connection that injects every error kind.

Tests cover concurrent arrivals, lost ownership, duplicate messages, uncertain
writes, commit failure, worker loss and pending human input. The same storage
contract suite runs against SQLite and AgentDB, including restart persistence,
thread lifecycle, cross-process ownership and partial lifecycle failures.
Cloud routing information is not a substitute for authoritative write ownership.
Application transactions and external effects retain their own operation IDs.

Observed lease loss stops admission of new conversational work and cancels the
owning run's invocation. Managed writes remain fenced at the database even if
the former owner has not observed the loss. Cleanup remains owned and continues.
Lease expiry cannot prevent a paused former owner from resuming application code.
External mutations require the receiving system's conditional-write, fencing
or idempotency protocol. Idempotency prevents duplicate operations; it does not
by itself reject a stale owner's action. Already-started effects can remain
uncertain and require reconciliation under the application transaction contract.

### Support algorithms and observations

Pure algorithms belong to the subject that owns their behavior, under the
[crate responsibilities](#crates-and-dependencies). Chat formatting,
tool-argument processing and JSON repair belong to `lk-agents-core`; speech
tokenization, markup and pacing belong to `lk-agents-voice`; media codecs,
filters and sample arithmetic belong to `lk-agents-media`. Schema and protocol
transformations belong to their corresponding subjects. These algorithms take
supplied data and return data without I/O or host objects.

Stream decisions such as speech pacing remain with their subject cores;
provider retry belongs to the provider client. The runtime and the async I/O
crates implement bounded buffering, backpressure and async delivery under the
shared stream contract, each owning the clocks, locks, queues and connection
pools of its I/O; the runtime owns host references and exception/resource tables. ABI declares
exports and boundary types independently of these implementations.

`lk-agents-core` owns deterministic run/event assertions. Evaluation helpers
follow the owning subject's pure-algorithm rules; model calls and judge
orchestration are async I/O in `lk-agents-inference`, and application evaluators
run through HostInvoker. The supported judge API does not
require a separate evaluation subsystem in the agent core. Simulations own
scenario execution and simulator verdicts. Public contracts for SDK evals and
`SimulationContext` live in [testing.md](../testing.md).

Metrics and telemetry are typed observations, recorded synchronously and handed
to the host language's OpenTelemetry SDK, which owns buffering, export and its
failure behavior ([telemetry](#telemetry)).
Observers cannot become a second policy owner. Binding construction and
conversion follow the [binding rules](#binding-rules).

## Shared binding semantics

**Declare semantics once in Rust, generate both bindings, and run the same
conformance cases against each.** Features reuse these contracts rather than
adding their own cancellation wrapper, iterator pump or conversion rules.

ABI declarations describe API meaning independently of the Rust implementation
types. Runtime and binding implementations may change without changing that
meaning. A Rust future does not itself define cancellation propagation; an
`Option<T>` does not define omitted-argument behavior; a Rust iterator does not
choose between a lazy host iterator and an eagerly collected list.

### Declared contracts

| Concern | Declared semantics |
| --- | --- |
| Presence | Required, omitted, explicit null and value; defaults apply to the declared cases, not general truthiness |
| Scalars | Boolean and Unicode semantics, integer ranges, floating-point precision and permitted non-finite values; no implicit truthiness conversion |
| Collections | Typed lists, tuples, arrays and map keys/values; ordering, snapshot versus live access and validation of every element |
| Records and variants | Required fields, defaults and default factories, enum/union discriminants, unknown-field policy and validation paths |
| Buffers and media | Encoding, element type, layout, format, copying/borrowing, retained lifetime and permitted concurrent access |
| Time | Duration versus timestamp, units, precision and monotonic versus wall-clock domain |
| Calls | Typed constructors, functions, methods and properties; synchronous versus asynchronous execution, argument/default rules and result conversion |
| Callback | Arguments, allowed return forms, execution context, reentrancy, failure conversion and cancellation relationship |
| Stream | Item/control types, lazy consumption, buffering, backpressure, EOF, failure and cleanup |
| Notifications | Typed event payload, subscription lifetime, delivery order, buffering and unsubscribe behavior |
| Resource | Native handles versus host references, identity, ownership/borrowing, permitted sharing, lifetime and close responsibility |
| Errors | Typed failures, host exceptions, causes, validation errors and cancellation; conversion in both directions |
| Cancellation | Target operation, propagation to owned work, reason/outcome and observable settlement |

These are internal contract families, not a public wrapper class for each row.
Each declaration composes supported families; adding a type extends the shared
model and its conformance cases before either binding accepts it.

For example, a nullable update has three meanings: omit to preserve the current
value, explicit null to clear it, or supply a replacement. Rust can represent
this with `Maybe<Option<T>>`; Python uses `NOT_GIVEN`/`None`/a value and Node uses
`undefined`/`null`/a value. An invalid null is rejected when the declaration does
not permit it. False, zero, an empty string and an empty collection remain
values, not omission.

Time and numeric declarations state units, precision and accepted ranges.
Conversions cannot silently truncate an integer, reinterpret a timestamp as a
duration or lose precision crossing JavaScript's number range. Media/buffer
contracts state whether data is copied, borrowed or retained and how long it
remains valid.

Borrowed buffers preserve Rust aliasing and concurrency guarantees for their
full retained lifetime, in both binding directions. Retaining an exporter or
creating a read-only view does not establish immutable backing storage. Copy
mutable host buffers at the boundary unless an enforced ownership or
synchronization contract makes borrowing safe throughout native execution.
Obtain copies through the host's supported access rules. Releasing the GIL,
returning to the host or scheduling native work cannot leave Rust references
exposed to incompatible mutation through another alias.
Conformance tests include writable aliases and host mutation after handoff,
verifying copied values or the declared access protection during native work.

Duration conversion rejects unsupported non-finite or out-of-range values
before constructing native durations.

Records and collections cross as validated values unless explicitly declared as
live resources. Mutating a converted list or record does not silently mutate
Rust state. Mutable defaults are created per instance. Typed maps declare their
key conversion; arbitrary JSON is a separate, explicit value contract, not a
fallback for unknown types. Aliases, newtypes and generic parameters retain
their declared validation and conversion rules.

### Calls, references and failures

Constructors, getters, setters, methods and free functions use the same typed
conversion rules. Callback signatures declare whether a synchronous result,
awaitable or stream is accepted. A synchronous call cannot block an event loop
waiting for a callback on that same loop. Reentrant callbacks enter through the
runtime's normal event dispatch, never recursively into a core transition.

Native Rust objects cross through typed handles. Python/Node objects cross
through typed proxies or explicitly opaque host references. Opaque references
preserve identity within their originating host and are not inspected by core
code. Retaining a reference keeps its object alive; releasing or closing it
follows the declared ownership contract. Stale or foreign-runtime handles fail
explicitly rather than becoming null or another object. Host references and
native handles are not serializable application state. These bindings do not
transfer live Python objects into a JavaScript runtime or vice versa.

Notifications reuse typed callbacks or sources. Subscription ownership defines
when delivery starts and ends, whether unsubscribe waits for in-flight delivery,
and how late events are rejected. Delivery follows the declared buffering and
ordering policy; adding an event API does not introduce another queue protocol.

Declared domain errors map to the corresponding host error types with structured
details. Unclassified host exceptions retain diagnostic type, message and cause;
an exception returned to its originating host retains the original exception
where its lifetime permits. Validation errors identify the argument or field
path. Cancellation, end-of-stream and failure remain distinct. Binding entry
points contain unwinding Rust panics before the host boundary. Process aborts
are supervision failures, not recoverable API exceptions. Do not report success
or reuse potentially corrupted operation state after a panic.

Cancellation follows the [shared operation lifecycle](#cancellation-and-task-ownership).

### Conformance

The shared declaration model feeds Python and Node generation. Emitters express
that model in each language without independently choosing defaults, ownership
or error behavior. Common defaults live in the shared contract vocabulary;
features do not repeat metadata for rules that are already defined there.

Resolve declared types and aliases through that model. Do not infer semantics
from a last path segment such as `Value`, a callback's name, or whichever methods
a host object happens to expose. Unsupported or ambiguous declarations fail
generation explicitly. Add a missing contract capability to the generator,
rather than falling back to `Any`, string-based reflection or handwritten glue.

Run equivalent conformance cases against Python-to-Rust and Rust-to-Python,
and Node-to-Rust and Rust-to-Node. Cases cover absent/null/empty values, defaults,
invalid fields/tags, range/precision boundaries, collection and buffer ownership,
per-instance defaults, scalar/awaitable/stream callback results, synchronous
calls and reentrancy, subscription/unsubscribe races, error/cause conversion,
object identity, stale and foreign handles, partial output followed by failure,
backpressure, borrowed resources, cancellation before start and during a pull,
late completion and cleanup.
The specification and inputs are shared; adapters use each language's normal
APIs. Tests assert the same outcomes, not identical scheduler traces.

Feature tests add domain behavior on top of these primitives. They do not
reimplement the primitive or redefine its contract. Native Rust plugins and
host-language plugins pass the same capability contracts through their bindings.

## Error contracts

Declare error families and their structured fields in ABI and generate the
catchable Python and Node types. Keep `APIError`, `APIStatusError`,
`APIConnectionError` and `APITimeoutError`; timeout is a connection-error subtype.
Do not collapse domain errors into one generic runtime exception.

An API error carries its message, retry eligibility, optional status code,
request ID, response body and cause. Timeout errors also carry a typed `phase`:
`pool`, `connect`, `read`, `write` or `total`. A foreign SDK timeout
whose phase is unavailable uses an absent phase, never a guessed one. Preserve
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
causes, multiple observers, foreign errors, timeout phases, cancellation and
safe public error rendering. Retry disposition is tested in the protocol core,
not in exception conversion.

## Binding rules

### Declarations and placement

All handwritten API export declarations live in `lk-agents-abi`. This includes
`#[host(...)]`, `#[derive(Host)]` and their export metadata. Neither subject
cores nor runtime implementations declare a second export surface.

| ABI declaration | Defines | Implementation or generated output |
| --- | --- | --- |
| `#[derive(Host)]` | Boundary data: options, events, chat items, reports and error payloads | Generated models and conversions at the language boundary |
| `#[host]` for a synchronous operation | Constructor, transition or synchronous API contract | Subject core for decisions; runtime or an async I/O crate for host views and resources |
| `#[host(proxy)]` on a trait | Provider, user I/O, tool runner, hook or host SDK contract | Generated proxy in `lk-agents-py`, entered through HostInvoker |
| `#[host(proxy)]` with `#[python(provider(..))]` | Provider contract and Python hook metadata | Generated provider base and generated proxy |
| `#[host]` for an async operation, stream or awaitable | Operation awaited by the host | Runtime or async I/O implementation and generated async bridge |

ABI contains contracts and data, not I/O or host-language implementations. It
has no pyo3, Tokio, futures or async-trait dependency: dynamic ports return std
boxed futures and class traits use native `async fn`. Generated pyo3 conversions
are free functions in `lk-agents-py`, beside the generated adapters and proxies.
Generation must not introduce pyo3, executors or runtime dependencies into ABI.
ABI never imports subject-core or runtime implementations.

Core bindings take data and IDs and return data. Bind an item only when used
by the public API, a plugin base or host glue. Glue-only names use `_core_*`.
Avoid duplicate wire records, per-language schemas and bindings that only
rename another binding.

Keep export attributes in one documented vocabulary shared by parsing and
generation. New keys require multiple concrete uses, not feature-specific glue.

### Generation

`lk-agents-codegen` reads ABI export declarations once into a resolved IR and
generates, as checked-in files:

- A Rust trait for every declaration, classes and providers alike, which a core,
  the runtime or an async I/O crate implements; the compiler checks completeness.
- The pyo3 adapters, proxies and record conversions in `lk-agents-py`, which call
  those traits. The export attributes are inert
  markers: the macros emit nothing.
- The complete `_core.pyi`, `core_names.json` and Python models under `_generated/`.
- Provider bases, streams and forwarding modules.
- Node interfaces and napi records.

Provider traits define their Python hooks and `_core_<method>` roles. Missing
required roles fail generation. Generated Python passes ruff and mypy.
Python templates live only in codegen, one per output kind.

Generated files carry a first-line marker and are never edited manually.
`--check` rejects drift. Handwritten Python contains public API behavior,
user callback invocation, host SDK integration or process launch/bootstrap.
Modules that only forward or re-export are generated, with public names
re-exported by their owning package's `__init__`.

### Module registration

Maturin builds `livekit.agents._core` from `rust/crates/lk-agents-py`, configured
in `livekit-agents-next/pyproject.toml`. It ships inside the
`livekit-agents-next` wheel as `cp311-abi3` for Python 3.11 and later.

`lk-agents-py` is the only crate containing pyo3: the registration shell, the
generated conversions, adapters and proxies, and HostInvoker's Python entry
(GIL, context copy and coroutine scheduling) behind its declared port. Its
handwritten part stays minimal; whatever can be generated is generated. Runtime
initialization, fork guards, async bridging and resource handles belong in
`lk-agents-runtime` as language-neutral Rust. A link-time
registration table supplies the module's names. Tests compare `dir(_core)`
with `core_names.json`.

### Enforcement

Enforcement must cover:

- No pyo3 outside `lk-agents-py`; no executor dependencies in sans-I/O crates or
  ABI; no dependency of a core or an async I/O crate on the runtime.
- Every Rust-to-host call goes through HostInvoker, except log forwarding in
  `lk-agents-py`: no crate other than `lk-agents-py` calls host code, and async
  I/O crates hold only ABI-declared proxy traits.
- Handwritten `#[host(...)]`, `#[derive(Host)]`, `#[class]`, `#[binding]` and API
  export metadata only in ABI.
- No handwritten `#[pyclass]`, `#[pymethods]`, `#[pyfunction]`, `#[napi]` or
  `#[pymodule]` outside generated output and registration shells.
- Generated host conversions and proxies in `lk-agents-py`; the HostInvoker
  lifecycle and async bridges in runtime; never in ABI or subject crates.
- Generated forwarding modules and matching stubs.
- Runtime executor modules (`voice/`, `room_io/`, `avatar/`, `rtc/`) contain no
  Tokio spawn, timer, socket or runtime-handle use and no unbiased `select!`
  ([effect execution](#effect-execution)); boundary adapters hold their sockets
  behind a port.

Codegen `--check` enforces export placement and generated output, failing with
the file and item. The dependency check resolves the closure of `lk-agents-abi`,
`lk-agents-core`, `lk-agents-voice` and `lk-agents-media` with all of their own
features; a dependent's closure carries only the features that dependent enables. Those
closures must contain no runtime, executor, async transport or host SDK: no
tokio, futures, pyo3, async-trait, reqwest, tungstenite, rusqlite, `livekit` or
`livekit-ffi`. `livekit-protocol` records remain allowed. The dependency direction
is `py -> runtime -> sans-I/O and async I/O crates -> abi`, with the composition
edges `voice -> core`, `voice -> media`, `inference -> media`, `inference -> core` and
`net -> core`, `py -> abi`, async I/O crates `-> net`,
`fuzz`, `test-support` and `bench-steps -> sans-I/O crates + abi`, `fuzz -> runtime`,
`fuzz -> storage` (the A2A DST runs real storage), and
`-> test-support` from fuzz and every crate's tests described above;
core never depends on storage. These dependency bans apply to the sans-I/O
crates and ABI, not to runtime, the async I/O crates (storage included), fuzz
support, language bindings or build tools. pyo3 is banned from every closure except `lk-agents-py`'s.

The shared [`rust/architecture/clippy.toml`](../../rust/architecture/clippy.toml)
configures `disallowed-methods`, `disallowed-types` and `disallowed-macros`.
The architecture checker selects this config explicitly with `CLIPPY_CONF_DIR`
and runs Clippy only for `lk-agents-core`, `lk-agents-voice`, `lk-agents-abi`
and `lk-agents-media`. Packages requiring a different policy use a separate
Clippy invocation with their own config and explicit package list.
Media's optional thread pool, added only for a measured
need, is its one allowed spawn site. Storage, inference, worker and net are
unrestricted async crates. Banned constructs include task/thread spawns,
`select!`/`join!`, ambient monotonic/wall clocks, random or UUID generation,
environment/process reads, unbounded tokio/std/futures/crossbeam channels,
locks, host-shared atomics and randomly ordered hash collections in decision
state. Supplied inputs and runtime-owned resources keep decisions deterministic.
The file-length check reads git-tracked `.rs` files, skips generated-marker
files and rejects handwritten files over 1,250 lines.

Existing violations are recorded in checked-in baselines; clippy exemptions
also require `#[allow(...)]` with a reason. Baselines may only shrink by
violation kind and location identity. Removing an old violation cannot excuse
a different one, and neither baseline identities nor exemptions may grow or
broaden. Each check has deliberately failing inputs, including replacement of
an old violation with a new identity. The shared enforcement step is
[`rust/architecture/check.sh`](../../rust/architecture/check.sh), invoked by CI
and the launch gate. Run it from any directory in a full Git checkout.

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
| Source | One consumer pulls typed items; each pull produces an item, end-of-stream or an error |
| Sink | A producer submits typed items with backpressure; acceptance is distinct from processing or playback completion |
| Duplex stream | Independent source and sink halves under one resource lifetime; ending input can leave output draining |
| Transform | A typed source-to-source operation; may filter, expand, buffer or change item types without owning the upstream resource implicitly |
| Host call | Invoke a callback on its owning language runtime, preserving invocation context and correlated outcome |
| Resource handle | Typed identity plus generation and explicit ownership; runtime retains the object and settles/releases it |
| Cancellation | Correlated termination request; completion and cleanup are observed outcomes, not implied by dropping a reference |
| Media frame | Typed format, sample/time information and bounded payload or opaque media reference; lifetime remains valid until the consumer releases it |

### HostInvoker

`HostInvoker` is the internal runtime primitive for executing host-language
application and plugin code. Every Rust-to-host call goes through it, from any
crate: job entrypoints, setup/shutdown callbacks, tools, hooks, nodes, provider
and plugin proxies, CLI and dev-server callbacks, endpoint handlers and telemetry
observations. The one exception is log forwarding: `lk-agents-py` attaches Rust's
`log` facade to Python logging directly ([logging](#logging)). Each call carries the Invocation captured at its operation's entry
and its declared scopes; none writes its own dispatch, context or cancellation
wrapper. ABI declares the typed call contract and HostInvoker's port;
`lk-agents-runtime` owns the invocation lifecycle and dispatch as language-neutral
Rust; `lk-agents-py` implements the Python entry behind the port. Async I/O crates
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
to `transcription_node` and `realtime_audio_output_node`. Native Rust
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

Use the [shared conformance cases](#conformance) for native and host streams.
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
call site. Tokio's `JoinHandle` remains only inside executors and for the
blocking pool (`spawn_blocking` work cannot be cancelled); no `JoinSet` or
`TaskTracker` is used.

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
disable that limit. Keep phase defaults at 10 seconds and `total_timeout=None`.
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
declared timeout scopes and preserve timeout phase and cause in errors. Implement missing transport support or reject an unsupported
applicable option before I/O; never silently substitute a different scope.
HTTPX phase controls are not an overall deadline, and Reqwest's pool idle timeout
is not a pool acquisition timeout. See the [HTTPX timeout definitions](https://www.python-httpx.org/advanced/timeouts/)
and [Reqwest client controls](https://docs.rs/reqwest/latest/reqwest/struct.ClientBuilder.html)
for adapter APIs.

Test shared policy with virtual time and real adapters with controlled peers.
Cover stalled phases, duplex idle periods, consumer backpressure, disabled
limits, retries and fallback sharing a total deadline, cancellation, partial
output safety and generated error conversion in both bindings.

## Plugins

A plugin supplies an implementation of a Rust-defined capability, such as STT,
TTS, LLM, realtime, VAD, a transform or a media endpoint. Provider contracts and
export metadata live in ABI. Rust implementations satisfy the runtime interface;
Python/Node implementations satisfy generated host interfaces. Session and agent
code consume the same capability regardless of implementation language.

Plugin authors implement provider/domain behavior. Generated/runtime support
owns host invocation, stream adaptation, correlation, cancellation and error
conversion. Plugin code does not reproduce those mechanisms. Configuration,
capabilities, identity and supported update behavior are typed and validated
before use; unsupported operations fail explicitly.

Python loads `lk-agents-py` as the `_core` extension. Node loads the Node
binding. The host owns its interpreter, event loop and application objects;
Rust calls host implementations through the generated proxy for that binding.

```text
Python application -> Python binding -> Rust runtime -> native Rust capability
                                                   -> Python plugin/I/O proxy

Node application   -> Node binding   -> Rust runtime -> native Rust capability
                                                   -> Node plugin/I/O proxy
```

A Rust implementation is reusable through both bindings. Python authors
implement the Python-facing capability; Node authors implement the Node-facing
capability. Both adapt to the same Rust interface and receive the same lifecycle,
stream and error semantics. Binding generation owns language adaptation, not
the plugin's business logic.

Each plugin instance has an explicit owner and close lifecycle. Sharing a
stateless definition does not share live streams or session-specific state.
Capabilities determine permitted concurrency and updates. Sessions supply
runtime resources and cancellation scope; secrets and connections are never
persisted as application state.

Test native Rust implementations through both bindings. Test Python and Node
implementations through their respective generated proxies, using the same
capability/stream conformance cases. Provider-specific protocol tests remain
separate from binding adaptation tests.

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
| `TurnContext` | Thread/lane identity, incoming messages, history, explicitly saved fields, transport tasks and permitted caller history |
| Handler output | New history items, typed output or input-required outcome, usage, staged field writes and failure/yield information |

Generate these boundary records from ABI declarations. Make submission durable
before acknowledging it. Observed lease loss stops new work and cancels the
owning invocation; database fencing protects managed writes independently of
that observation. Worker loss does not implicitly rerun application code.
External actions follow the application transaction contract. Storage failures follow the
[storage contract](storage.md#the-unit-of-work).

The application constructs its context and explicitly loads/saves fields.
Only staged fields join the managed write set. `TextSessionState` is saved
application data, not a Python execution stack. The handler checkpoint protocol
is separate from the public [semantic run stream](../agents.md#semantic-events).

## Processes and IPC

`lk-agents-worker` owns socket/pipe creation, ownership, framing, protocol
messages, reads, writes, shutdown, exit monitoring, termination, supervision and
pool policy.
`_core.ipc_pair()` creates the control socket pair.

Each child has one Rust-owned control connection. Validate protocol
compatibility before accepting work and bound frame lengths before allocation.
Malformed/truncated input and unexpected EOF settle pending operations as
failures and enter supervision cleanup; they cannot appear as successful exit.

The host-language process adapter launches and bootstraps the child with the
Rust-created IPC endpoint. Python uses `multiprocessing`; Node supplies its own
launch adapter. The child runs application initialization and job callbacks.
Host adapters do not implement IPC loops, supervision or pool policy.

Provider clients and model execution run in the job process
(`lk-agents-inference`). There is no separate inference-process service.

The worker retains cleanup responsibility for every launched child, including
cancelled startup and failed initialization. Each start-method protocol defines
how exit is observed and who reaps the process; a forkserver child's OS parent
is the forkserver, not the logical worker owner.

Detect logical worker loss and initiate termination/escalation within configured
deadlines independently of a blocked host loop. Observe OS exit and completed
reaping separately; deadline expiry reports incomplete cleanup and retains its
owner. IPC queue saturation cannot disable supervision or shutdown.

### IPC stress and leak tests

Test IPC with generated event sequences against the worker and at the real OS
boundary:

- Generate supervision/pool event sequences covering startup, assignment,
  cancellation, timeout, drain and close races, as property tests of
  `lk-agents-worker` (the DST does not simulate the worker). Every accepted request settles;
  stale events cannot revive a process or lose cleanup ownership.
- Stress the real transport with fragmented/truncated frames, malformed and
  oversized messages, slow readers/writers, full queues and disconnects during
  pending operations. Assert bounded buffering and responsive supervision.
- Run repeated real-process launch/job/close cycles through each host launch
  adapter and supported start method. Mix concurrent jobs, initialization
  failure, child crashes, abrupt parent death and blocked host loops. Verify
  process exit and reaping separately; treating a zombie as gone does not prove
  successful reaping. Check inherited endpoint cleanup as well as live children.
- Measure resource retention after warm-up and after each settled batch:
  processes, descriptors/OS handles, threads, tasks, pending requests and host
  references. Baselines include declared caches and launch helpers, such as the
  forkserver/resource tracker. Counts return to their baseline; memory remains
  bounded under repeated fixed workloads. RSS alone does not establish a leak or its
  absence.

Use reproducible fault schedules and bounded watchdogs. Fixtures clean up the
whole test-owned process tree after failure, while preserving evidence of
orphans or leaked resources before cleanup. Cleanup must not turn a failed
assertion into a pass. Run a bounded stress campaign in the merge gate and
extended soak campaigns separately, retaining failing schedules and resource
measurements. Simulated resource counters do not replace real-process checks.

### Fork safety

Importing `_core` starts no runtime, thread or socket. Linux defaults to
`forkserver`; macOS and Windows use `spawn`. An explicit `forkserver` choice
outside Linux falls back to `spawn`.

Bindings use a lazy process-wide Tokio runtime after process initialization.
Python tests run on the real clock; deterministic time is the DST's.

`_core.init(role)` declares the process. A process that never declares a role
is a job. Entry points check the PID before using inherited native state;
a forked child must reinitialize before using state declared by its parent.
A stale declaration raises `RuntimeError`. No `pthread_atfork` handler is
installed; fork followed by exec remains allowed.

### Python preloading

```python
server = AgentServer(preload_modules=["myapp.preload"])
```

`preload_modules` contains importable module names in execution order. Importing
a module may synchronously prepare immutable catalogs, fork-safe weights or
other module-held data. The Python launch adapter owns this configuration.

| Start method | Execution |
| --- | --- |
| Linux `forkserver` | Import in the forkserver after framework/plugin preloads, before the final thread check, `gc.freeze()` and any job forks |
| `spawn` | Import in each child before `setup_fnc` and before reporting readiness |

Forked children inherit prepared objects through copy-on-write; their mutations
are private. Spawned children load independent copies. Use `setup_fnc` for
per-process clients and active runtimes after launch.

User preload failures stop initialization and report the module name and
original error. Detected background threads prevent forking. This check does
not prove arbitrary native-library state is fork-safe. Preloads must not start
threads, event loops, connections, GPU contexts or the Rust async runtime.

### Logging

Each process writes its own logs to inherited stdout. Logs do not travel over
the IPC control socket. Rust uses the structured `log` facade:

```rust
log::warn!(target: "livekit.agents", lane = lane.as_str(); "the lease was lost");
```

`lk-agents-py` forwards records to Python logging (`pyo3-log`) under their
logger target, with structured keys as `extra`. This is the one host entry
outside HostInvoker: a record carries no invocation and settles nothing. Logs are observations, not protocol messages or
inputs to decision policy.

## Telemetry

Rust emits typed observations with operation identity, supplied timestamps and
logical span relationships. Recording an observation is synchronous: Rust holds
no SDK spans, exporters, export queues or watchdog for user telemetry. Each
observation reaches the host through HostInvoker like every other host call;
there the host OpenTelemetry SDK turns it into spans, metrics and log records, so applications
intercept framework telemetry through their own provider, processors and
exporters alongside their own instrumentation. Runtime owns invocation-context
propagation. Export follows the host SDK boundary, just like a host provider:
the host's own SDK processors export, and no Rust worker pool runs host export
callbacks. Destination policy and metric admission are
synchronous decisions in `lk_agents_core::telemetry`; the host applies them.
The LiveKit Cloud session report and recording upload is not user telemetry: it
is ordinary async Rust in `lk-agents-net`, triggered at session end with its
own bounded admission.

Adopt the application's configured tracer provider by default. An explicitly
configured framework provider is also used when activating framework invocation
contexts; applications can instrument child work through the normal host
OpenTelemetry API. Do not silently replace the application's global provider,
sampler, resource or exporters. Framework-created providers are process-owned;
application-created providers remain application-owned and are never shut down
by a session or job.

Use [HostInvoker](#hostinvoker) to activate the owning operation's trace context
for user execution. Parentage comes from explicit invocation data, not a
process-global or dispatcher's current span. Merely using the same provider or
materializing completed spans later does not establish parentage. Correct
parenting cannot depend on a lossy export queue delivering a span-start
observation.

Capture thread, lane, job and operation attributes at observation creation.
Do not obtain them from whichever task happens to export a batch. Propagate
validated trace context over transport boundaries; never serialize SDK objects.
Concurrent sessions and jobs cannot overwrite one another's context or metadata.

Cloud and application export destinations retain independent ownership and
configuration. Framework-owned export queues are bounded; application processors
retain their configured synchronous or asynchronous behavior. Disabling Cloud
export for a job does not disable application exporters. Framework-managed
export is best-effort; export failure cannot fail a conversation. Observe dropped
records without recursive logging.
Flush and shutdown have bounded waits and do not block the session loop or
worker heartbeat. Do not invoke SDK processors while holding runtime state
locks; application processors may block.

Tracing, metrics and logs have separate SDK configuration. Respect configured
meter providers and readers; do not silently replace a user provider to attach
a Cloud reader. Register framework-owned exporters/readers during provider
setup and report configuration conflicts explicitly. Close only resources the
framework owns. Metrics use bounded-cardinality attributes; thread IDs, user
content and arbitrary exception messages are not metric labels. Conversation
content requires explicit capture and redaction policy. Each export destination
filters an independent snapshot; one destination's policy cannot mutate data
sent to another. See [telemetry integration](../telemetry.md).

Conformance tests create spans through the standard host API inside tools and
nodes and assert their actual parent and trace IDs. Also test concurrent jobs,
context restoration after failure/cancellation, metadata captured before export,
an existing user provider/exporter, Cloud disabled, export saturation and bounded
shutdown. Follow the [OpenTelemetry SDK contract](https://opentelemetry.io/docs/specs/otel/trace/sdk/)
for provider and processor behavior; the framework ownership rules above apply
to both bindings.

## Test boundaries

| Layer | Verifies | Dependencies and scheduling |
| --- | --- | --- |
| Core unit tests | Individual transitions, validation, effects and negative cases | Synchronous calls with supplied inputs and time |
| DST | Behavioral invariants across composed machines, failures, cancellation and event orderings | Real sans-I/O crates and the runtime's real executors, with fakes only at the SDK and provider ports: providers, the worker, the storage port (injecting every typed error), MCP transports and the room, under controlled time/scheduling |
| Async I/O and adapter integration | The shared transport, provider clients, worker IPC and supervision, storage (real SQLite and a fake AgentDB server with injected faults), telemetry delivery to the host SDK, serialization, bindings and RTC adapters | Real code against a controlled peer, real processes or the actual service |
| E2E | Complete user journeys through the deployed system | Real processes, bindings and transport; each test identifies which providers/services are live or scripted |

A contract can be exercised at several layers. DST explores its possible
orderings; adapter tests validate the real boundaries and fake assumptions;
E2E verifies that the assembled system delivers the result to the user.

For interruption, DST checks cancellation and completion orderings. E2E checks
that audio delivered through a real room stops at the caller within a specified
bound. An interrupted history flag alone does not establish that behavior.

### E2E contracts

Each test states its prerequisites, user actions, expected outcomes, forbidden
behavior and completion bounds. Assertions are standalone.
Verify business effects through recorded service calls or authoritative state,
and delivery through caller observations. Recording an event or enabling an
option does not prove the feature worked.

For approval, verify that the tool has not executed before confirmation, then
that the exact approved operation executes once and its result reaches the
caller. Recording tests verify a readable recording with expected media;
interruption tests verify playback at the receiving client.

Startup failure, missing artifacts, incomplete scenarios, empty selections,
timeouts and unexpected errors fail validation. Retrying retains each attempt
and reports instability; a later pass does not silently erase an earlier
failure. Performance measurements and exploratory recordings are reported
separately from correctness results.

### Test organization

Use ordinary Rust tests and pytest test functions, grouped by feature. A test
file contains related scenarios and assertions; reusable fixtures own worker,
room, caller and service lifecycles. Shared drivers perform actions and capture
observations without deciding whether a test passed. Fixture cleanup runs on
failure and cancellation, releases remote resources and reaps child processes.

Keep environment preparation, scenario execution, artifact capture and result
comparison separate. Command-line runners only select tests and invoke the
standard runner. Do not combine environment installation, an agent application,
caller simulation, normalization and custom pass/fail reporting in one script.
Do not build a second test framework inside a helper.

Split large files at responsibility boundaries. Keep a scenario's assertions
next to its actions; share fixtures rather than copying orchestration. Avoid
both monolithic runners and many single-use forwarding modules. Each module
must have a responsibility that a reader can name.

Python tests reach an offline room through the documented session, as
[programmatic tests](../testing.md#programmatic-tests) do: `session.start(agent)`
without a job, or `room=None` where a job would supply one. `_core` ships no
test-only room export. Room-level behavior is tested in Rust over the SDK-port
fakes in `lk-agents-test-support`.

## Deterministic simulation testing

The harness owns every core input, executor task and fake whose ordering can
affect behavior. The DST runs the runtime's real executors (voice session, room
I/O, avatar transport, their drains and legs) over faked SDK and provider ports,
never copies of them: fakes sit only at those ports, and boundary adapters such
as the SDK room and the console's TCP port are replaced by port fakes. Executors
spawn and sleep only through the injected `Spawner` and `Clock`
([effect execution](#effect-execution)), which the DST binds to its own
deterministic scheduler and virtual clock; every `select!` is `biased;`, so the
scheduler alone decides which ready work runs. The architecture check enforces
these rules on the runtime's `voice/`, `room_io/`, `avatar/` and `rtc/` modules
against a shrink-only baseline.

The scheduler chooses among enabled core event deliveries, executor tasks and
fake completions. Each core transition remains synchronous and atomic. Advance
virtual time to the next arrival or deadline only when nothing is enabled.

Bound total steps, steps without virtual-time progress and real execution time.
A real-time watchdog operates independently of the simulation so a
non-yielding step cannot disable it. Retain the seeds, delivered steps and last
known pending work. Report a violated contract, demonstrated deadlock and budget exhaustion
separately; a finite step limit alone does not prove livelock. Liveness checks
state the assumed progress of external services and scheduling.

### Simulated host execution

The DST harness supplies a deterministic fake of the ABI host-call contract
used by `HostInvoker`. It handles host-call effects from the real cores and
schedules correlated results, progress, stream items, failures, cancellation
and cleanup outcomes through the simulation scheduler. It stands at the
host-call port beneath the runtime's real executors and never executes
Python/Node callbacks.

Simulate stalled calls, cancellation refusal, late success, failed launch and
cleanup deadlines as well as successful execution. Preserve operation/resource
identity, source causality and declared ownership. The fake does not reproduce
agent policy: framework argument preparation, approval, recovery and continuation
still run in the real shared cores. Model application-specific validation as a
host outcome when it requires user code.

Use the fake to explore decision ordering and resource disposition. Verify real
ContextVar/async-context behavior, signature binding, application validation and
host object/task release through the actual `HostInvoker` adapters. Keep those
conformance cases aligned with the fake's contract so simulation assumptions are
checked at the real boundary.

### Scheduling boundaries

Cores receive events and time and retain no spawner or executor. The
simulation covers `lk-agents-voice`, `lk-agents-core` (with its A2A module),
`lk-agents-media` and the runtime's executors with realistic fakes. Providers,
the worker, storage (injecting every typed error), evaluation judges, the
delegation client, MCP transports and the SDK room are fakes in the simulation
and are tested on their own. Boundary adapters at those ports are outside it.
They are verified by conformance tests through the real adapters with
cancellation injected at every await point, stress tests (stalled consumers,
full queues, close under load), real-peer adapter tests and a regression test
per bug.

Campaigns report which cores and fakes they simulate. Control randomness inside
fakes as well as the scheduler: supply, seed or record those choices and vary
them across campaigns. Identical inputs, build, configuration and seeds
reproduce the schedule.

### Partial-order sampling

The scheduler uses basic partial-order sampling from
[Yuan, Yang and Gu, CAV 2018](https://doi.org/10.1007/978-3-319-96142-2_20):

1. Assign a random priority to each pending step: a core event delivery, an
   executor task woken through the injected `Spawner` or `Clock`, or a fake
   completion.
2. Dispatch the highest-priority enabled step, preserving source causality
   and due-wake precedence. A source's next step receives its own random
   priority.

The harness keeps no per-step read/write footprints and does not reassign
priorities after races.

The harness varies external event order, including decision/data-plane hops,
same-instant races and bursts before settlement. Internal machine signals keep
their defined order. Effect completions return as scheduled events; they do
not mutate cores from callbacks.

An arrival-order baseline runs alongside. Campaigns record the policy with each
seed and compare policies on the known-bug benchmark. Claims about explored
interleavings apply to these scheduled steps, not arbitrary instruction
interleavings in the whole program.

Test the scheduler against small programs with independently known legal
orders. Cover source causality, due-wake precedence, wake/cancel/completion and
exact replay. Benchmarks and known-bug hit rates supplement these checks; they
do not establish scheduler correctness.

### Fakes and findings

Fakes reproduce real boundary behavior: calibrated LLM chunk timings, TTS audio
proportional to text, realtime probe fixtures and SDK room lifecycle behavior.
The measured provider timings are in `rust/crates/lk-agents-fuzz/calibration.json`, for
reference; each fake hardcodes the deciles it draws from. Fix the
fake's missing behavior whenever a real-room failure reveals a gap.

A seed reproduces a run for the same build, configuration and scheduling policy.
Record those inputs and the delivered event trace. Fixed findings become
handwritten regressions. Key distinct findings by rule and subject kind, never
generated IDs. Compare both findings and reach on the same seeds when changing
scheduling or fakes.

The [launch gate](../../rust/crates/lk-agents-fuzz/launch-gate.sh) must run before
every merge. It covers affected subject cores and runtime/binding contracts,
reports omitted coverage and rejects empty runs. Workspace compilation is not
behavioral coverage of any component; the worker and the other async I/O crates
run their real-transport, real-process and controlled-peer suites in the gate.
Its `KNOWN_FINDINGS` baseline may only shrink. Architecture checks reject new
core spawns, `select!`, executor dependencies and unbounded channels, and
runtime executor Tokio runtime uses and unbiased selects, and ratchet existing
violations down.

### Observations and replay

Rules read journal observations and public outcomes, not private state. The
voice trace vocabulary is [trace.rs](../../rust/crates/lk-agents-voice/src/trace.rs).
Cover raw argument validation, pending answers, time ordering, bursts, resource
lifecycle and host-view consistency. Normalize only differences irrelevant to
the asserted contract; preserve the identities, ordering, timing and errors
needed to detect duplicate work, stale outcomes and deadline violations.

Export observations without retaining an unbounded in-memory call log.
Production replay feeds recorded events and times to cores without repeating
external effects. Recording and replay require sufficient input data, not just
an output journal. Apply retention, redaction and access policy to logs that
contain conversation text.

## Codebase rules

### Rust-first implementation

- Implement shared framework behavior once in Rust: defaults and option
  resolution, validation, lifecycle policy, stream adaptation, transforms,
  protocol handling, media accounting and built-in provider/I/O behavior.
- Python and Node expose generated APIs and thin host adapters. Handwritten host
  code serves application callbacks, host SDK objects, event-loop integration
  or language-specific process launch/bootstrap.
- A native implementation is consumed through either binding. Binding code
  converts types and invokes the host; it does not reproduce domain behavior.
- Extend ABI declarations and generation when a host needs an existing Rust
  capability. Do not port the implementation into Python or JavaScript to work
  around a missing export or stream bridge.
- User plugins and callbacks may be authored in their host language. Their
  custom behavior remains there; shared framing, stream lifecycle, cancellation
  and error adaptation remain in Rust and the generated bridges.

### Names and layout

- Use the specified public API names. For internal concepts, preserve the
  Python reference's names and module grouping, adapted to Rust conventions:
  `AgentActivity`, `SpeechHandle`, `schedule_speech`, `drain`, `handoff` and
  `dtmf`. Do not rename an existing concept merely because it moves to Rust.
- Introduce a new concept or name only for a distinct responsibility. Reuse
  the domain's vocabulary rather than adding generic framework terminology.
- Keep every git-tracked handwritten `.rs` file within 1,250 lines; skip marked
  generated output. Enforce the limit in architecture checks.
- Organize modules by subject and developer-visible behavior. Keep helpers
  beside their users and use module documentation to explain file structure.
- Extend existing mechanisms before adding new ones. Remove dead code and
  obsolete documentation.
- Methods preserve invariants or name nontrivial reusable computations. Use
  fields instead of bare getters/setters when no invariant requires a method.
- Reuse LiveKit crates rather than vendoring protocol definitions. Evaluate
  third-party dependencies against build time and wheel size.
- Keep lifecycle and decision code in safe Rust. Isolate necessary unsafe code
  in narrow OS/FFI/codec adapters and document its ownership, lifetime and
  thread-safety invariants. Validate those boundaries with focused tests.

### Types

- Public Python and Node APIs are fully typed, without `Any`/`any`.
- Preserve the public API's context and output types through Agent,
  `RunContext[Context]`, `RunResult[Output]` and session frames. Infer output
  types from the declared output contract rather than adding public parameters.
- Validate persisted fields, outputs and tool arguments at their boundaries.
- Generate typed wire records from one schema shared by the language bindings.
- Check generated stubs against Rust declarations and use `assert_type` /
  `expectType` tests for generic contracts.
- Follow [shared binding semantics](#shared-binding-semantics): declare once in
  Rust, generate both bindings, and verify equivalent conformance cases.

### Schemas and field access

- Export schemas have one source of truth in ABI. External protocol schemas
  use the upstream generated types. Generate models, serializers, conversions,
  field accessors and export registration from those declarations.
- Access known fields through typed records, enums and generated APIs, such as
  `options.name`. Do not use `dict.get("name")`, `value["name"]`, string-based
  attribute lookup or handwritten JSON construction for schema-defined fields.
- Decode and validate untyped input once at its adapter boundary. Core logic
  receives typed values; it does not repeatedly extract or cast JSON fields.
- Application-defined map keys and opaque provider payloads may remain dynamic.
  Keep them at their declared boundary. Known framework metadata keys use
  generated constants and typed accessors, not duplicated string literals.
- A missing generated accessor or conversion is fixed in the schema or
  generator. Do not work around it with a parallel handwritten representation.

### Helper tools

- Prefer compiler checks, existing test runners and generated schemas to custom
  source policing or bespoke verdict logic.
- Keep helpers focused on one operation. Make mutations explicit; check modes
  must not write files. Separate data capture from validation and publication.
- Validate inputs before running expensive work or replacing artifacts. Missing
  datasets, empty runs and partial output must not produce success or overwrite
  a valid baseline.
- Preserve subprocess exit codes and actionable errors. A parseable transcript
  does not turn a failed subprocess into a successful run.
- Test validation tools with deliberately failing inputs. State exactly what a
  check covers; a successful narrow check does not establish full architecture
  compliance.

### Tests

- Observe public APIs, core events/effects and journal observations; do not add
  test-only product APIs.
- Give rules negative cases. Prefer behavioral failures to shape/default checks,
  merge duplicate tests.
- Repair generator or fake coverage gaps in the harness after verifying the
  product's refusal. Product behavior changes require a product contract.
- Derive thresholds from the driver's contract rather than arbitrary constants.

## Design basis

These primary sources define the mechanisms and limits used by the contracts
above. Implementation and conformance tests verify the framework's integration.

| Area | Basis and limit |
| --- | --- |
| Sans-I/O | [Implementation guide](https://sans-io.readthedocs.io/how-to-sans-io.html) and [quinn-proto](https://docs.rs/quinn-proto/latest/quinn_proto/): synchronous, supplied-time protocol logic separated from I/O; domain-specific interfaces can preserve that boundary |
| Deterministic fault testing | [FoundationDB, SIGMOD 2021, section 4](https://www.foundationdb.org/files/fdb-paper.pdf): reproducible simulated faults; real adapter, OS and performance tests cover behavior outside the model |
| Scheduling | [POS, CAV 2018](https://doi.org/10.1007/978-3-319-96142-2_20): basic partial-order sampling, no race-based priority reassignment; sampling does not establish exhaustive coverage or leak freedom |
| Runtime cancellation and delivery | [Tokio cancellation safety](https://docs.rs/tokio/latest/tokio/macro.select.html#cancellation-safety) and [bounded sends](https://docs.rs/tokio/latest/tokio/sync/mpsc/struct.Sender.html#method.send): partial progress and enqueue success require explicit ownership handling |
| Host execution context | [Python contextvars](https://docs.python.org/3/library/contextvars.html), [PyO3 async bridge](https://docs.rs/pyo3-async-runtimes/latest/pyo3_async_runtimes/tokio/fn.future_into_py_with_locals.html) and [Node async context](https://nodejs.org/api/async_context.html): the bridge establishes execution scopes; thread crossings require supported context propagation |
| Generator and worker boundaries | [Python generator cleanup](https://docs.python.org/3.10/library/contextlib.html#contextlib.aclosing) and [Node Workers](https://nodejs.org/api/worker_threads.html#worker_threadsworkerdata): stream context persists through cleanup; opaque functions/context do not transfer between isolates |
| Tracing | [OpenTelemetry context](https://opentelemetry.io/docs/specs/otel/context/) and [tracing SDK](https://opentelemetry.io/docs/specs/otel/trace/sdk/): scoped propagation and provider ownership are separate from export delivery |
| Process cleanup | [Multiprocessing](https://docs.python.org/3.13/library/multiprocessing.html) and [process waiting](https://man7.org/linux/man-pages/man2/waitpid.2.html): start methods determine OS parentage; signalling, observed exit and reaping are separate |
| Memory measurements | [Linux process accounting](https://www.kernel.org/doc/html/latest/filesystems/proc.html): RSS includes effects that resource counts and retained-memory measurements must distinguish |
| Retry safety | [AWS timeout/retry analysis](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/): retries can amplify load and repeat side effects; use one retry owner and explicit replay safety |
| Foreign boundaries | [Rust FFI and unwinding](https://doc.rust-lang.org/nomicon/ffi.html#ffi-and-unwinding): unsafe interfaces require ownership/lifetime invariants; aborting panics cannot be caught |
| Borrowed buffers | [Rust slice safety](https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html#safety) and [Python read-only views](https://docs.python.org/3/library/stdtypes.html#memoryview.toreadonly): retained lifetime and a read-only view do not establish immutability or safe concurrent access |
| Ownership fencing | [Chubby, section 2.4](https://www.usenix.org/legacy/event/osdi06/tech/full_papers/burrows/burrows.pdf): resource-side checks reject stale ownership; a lease alone cannot stop a former owner's execution or external effects |
| Transport conformance | [SSE framing](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream), [WebSocket framing](https://www.rfc-editor.org/rfc/rfc6455#section-5.4), [ASGI](https://asgi.readthedocs.io/en/latest/specs/www.html) and [A2A](https://a2a-protocol.org/latest/specification/): test actual byte/message boundaries and the supported protocol version |
| Media observations | [AudioSource](https://docs.livekit.io/reference/python/livekit/rtc/audio_source.html) and [FFmpeg send/receive](https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html): local queue progress is distinct from receiver delivery; codec output may be buffered and requires draining |
| Coverage evidence | [Typing directives](https://typing.python.org/en/latest/spec/directives.html#assert-type), [pytest outcomes](https://docs.pytest.org/en/stable/how-to/skipping.html) and [Cargo test](https://doc.rust-lang.org/cargo/commands/cargo-test.html): static types, compilation, execution and skipped coverage establish different properties |
| Native safety validation | [Rust sanitizers](https://doc.rust-lang.org/unstable-book/compiler-flags/sanitizer.html) and [Miri limits](https://github.com/rust-lang/miri#limitations): supported runs detect classes of memory errors; unsupported FFI/targets and unexecuted paths remain outside their evidence |

Crate placement, generated API ownership, internal dispatch order, defaults and
file-size limits are repository contracts. Enforce them with architecture,
generation and behavioral checks. Calibrate capacities and timing budgets
against supported workloads; references alone do not establish suitable values
or prove the implementation correct.

## Implementation review

### Extraction unit

Establish `lk-agents-runtime` before extracting executors and host bridges.
The first migrated API uses a real generated boundary, runtime invocation and
synchronous core transition. Grow shared runtime services from working consumers;
a registration shell or forwarding path alone does not establish the design.

Migrate a lifecycle or pipeline leg end to end: its state owner, typed inputs
and outcomes, runtime adapter, host entrypoint and tests. Move its eligibility,
retry, cancellation and deadline decisions into synchronous transitions.
Moving the same policy loop to another async module does not complete extraction.
Keep I/O sequencing and independent resource tasks in runtime.

Executors move into the runtime by splitting, never by moving a
decision-bearing module wholesale. The decisions stay in their subject crate as
synchronous state machines emitting effects (voice for speech and generation
policy, core's AgentRun for tool dispatch); the runtime keeps the pumps, node
calls, timers and output binding that execute them.

Choose slices by dependency and failure risk across voice, inference, worker,
storage and A2A. Generation can move one leg at a time while preserving the
whole speech lifecycle. Each operation has one decision authority throughout
the change; partially updated old and new owners cannot both control it.
Remove the superseded path once its replacement satisfies the contract.

Validate shared interfaces against different real lifecycles before generalizing:
a streaming generation/tool path and an independently failing resource-creation
path. Include replacement/cancellation during open, late completion, rejected
delivery and actual cleanup. A common abstraction must simplify those cases
while preserving their concurrency and declared context.

Move a component's regression cases and observation rules with its owner.
Preserve typed diagnostic observations and the inputs required for replay when
changing message or journal representations. Serializable effects alone do not
establish replayability. Keep observers independent of policy ownership.

### Acceptance evidence

| Property | Required evidence |
| --- | --- |
| One authority | Each mutable fact has an owner; no duplicate lifecycle/retry policy remains in host glue or adapter tasks |
| Core purity | Compiler/dependency checks and source review establish supplied inputs/time, synchronous transitions and no retained live resources |
| Causality and settlement | Immediate launch failure, repeated requests, supersession, stale outcomes and cleanup deadlines settle callers without invalidating replacement work |
| Validation and views | Invalid framework/API input shapes fail before the requested operation's effects; valid-shaped requests against replaced/closed state reach core refusal; host views match state after the API's declared acceptance boundary |
| Host execution | Real callback/node round trips preserve job/speech/approval ownership, current trace parent, stream context and release of captured references |
| Deterministic decisions | Direct core stepping reproduces effects and observations for identical supplied inputs/time, including wire/history timestamps; event-level schedules explore permitted races |
| Real execution | Adapter, process, SDK, database and client tests observe the boundaries and cleanup outcomes defined by their component contracts |
| Resource and latency budgets | Fixed supported workloads measure retained resources, queue/control latency, copying/allocation and media throughput before and after the change |

Direct core simulation and runtime conformance, stress and real-peer tests
exercise different boundaries; each is required for the decision or execution
behavior it covers. Independent I/O tasks remain valid runtime mechanisms.
Campaigns name the policies and cores actually exercised and compare findings
and behavioral reach under matched seeds, workloads and budgets.

Ratchet remaining core violations by kind and location, including async policy,
ambient time/entropy, shared decision state and live objects in core messages.
Reduced spawn counts, smaller files or additional wrappers do not establish
semantic progress. Acceptance requires simpler ownership and failure handling,
preserved concurrency and the behavioral evidence above.

An audit finding names the violated rule and source evidence. Missing coverage
is distinguished from a demonstrated defect. Compilation, a narrow source check,
or one passing E2E run cannot establish all contracts.

Reviews compare implementation and evidence against these ownership and
behavioral contracts. They distinguish missing coverage from defects without
changing the architecture merely to accommodate existing code. Implementation
status and migration checklists are tracked separately from this contract.
