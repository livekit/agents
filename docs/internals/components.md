# Component contracts

Decision and execution responsibilities for each framework component.

[Contributor index](README.md) · [Execution paths](architecture.md#execution-paths)

Components are organized around decisions and lifecycles. Each mutable fact has
one owner. Another component can observe a snapshot or send a typed input; it
cannot keep a second authoritative copy or mutate the owner's fields.

A component has a separate state machine only when it owns an independent
lifecycle. Parsing, formatting, validation and arithmetic remain synchronous
functions. Internal APIs use domain-specific methods and outcomes; no generic
machine trait, signal bus or universal result wrapper is required.

Activity-local reply bookkeeping, interruption waiters, hook limits and realtime pending-work
transitions are methods on AgentActivity, colocated with their domain. Driver retains the ordering
that combines them with session floor, configuration, timers, storage and output bindings. Runtime
ReplyExecution contains an explicit pure plan and resource snapshot; execution does not implicitly
dereference into decision state. Related speech attribution and completion facts share one
activity-owned history, including the intentional late-event retention window.

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
TextSession do not compose A2A partition orchestration or VoiceSession; the separate
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
| Provider configuration | Base provider identities and ordered temporary overrides | Voice resolves precedence and restoration; runtime retains provider references; the host installs only the resulting changes |

Provider overrides use the same layer identity as turn settings, including a
block entered before `start()` and released after it. A base update under an
override is retained until it becomes visible; releasing layers out of order
preserves any newer layer still held. The voice core keeps only resource IDs.
`SessionOptions` retains the host references in the runtime and releases obsolete
references outside its lock. Language adapters keep no provider precedence stack.

Session defaults are declared in ABI and resolved by voice. The host constructs
Python provider objects from the resulting model plan. Native `SessionStorage`
opens the supplied thread's partition, else one on a new local SQLite thread, owns the context
activation and closes its storage resources. Python only wraps the supplied
activation and thread namespace.

`SessionOwner` exists from the public session's construction, before any room,
provider or storage is opened. Voice admits startup and closing once; runtime
owns their tasks, so cancelling an individual waiter does not abandon resource
acquisition or finalization. Startup opens storage, asks the language adapter for
typed native bindings, opens and attaches recording, and starts the voice driver.
Failure settles partial acquisition before returning the original host error.
Closing joins startup, closes the driver and storage, releases host hydration and
SDK processors, and finishes recording once. The recording outcome remains
available to the job reporter independently of the session's close result.

Python's session facade delegates lifecycle calls to this owner. Its separate
host adapters supply provider and RTC capabilities, application callbacks and
async iterators and public event objects. A native bindings
container is constructed on the Python loop so returned capabilities retain the
invocation context captured at that boundary.

The owner routes configuration updates and override releases before and after
startup through the same native entry. Python adapts SDK providers and installs
the returned effective references; it holds no override identity or choice of
configuration target. Voice validates construction-only interruption settings
and selects reply tools. The pending-agent view uses native request identities:
settling an older handoff cannot clear a newer selection. Runtime retains the
agent references and settles the view when the native handoff answers.

Public Python classes over a native class are generated facades ([Facades](bindings.md#facades)): the
fallback adapters, the inference clients and their per-request streams among them. Python keeps
handwritten subclasses only where SDK object conversion or callback binding is needed;
configuration types and VoiceTask authoring are separate from the session class.

The runtime owns a tool body's execution and first-progress notification. It
keeps the body alive after an interim result and returns subsequent completion
through the session queue; voice decides task completion and whether application
output validation is required. Hosts execute the callable, validate Python types
and classify returned objects. They keep no first-update race or detached-body
completion callback. DTMF collection belongs to the VoiceTask that requested it:
spoken completion, keypad completion, cancellation and parent restoration use
the same task lifecycle.

The runtime's private `ToolBodies` owner retains each actual host future until it
settles. This replaces the former first-progress-only `ToolProgress` map; it is
one lifetime owner, not another pending-work registry. Taking the first progress
reply does not erase that lifetime. A semantic terminal notification can retire
the core's call while host execution is still unwinding, so the body also watches
session close and requests cooperative cancellation through its captured
invocation. Core cancellation and close cleanup claim the same cancellation bit
before calling the host, after releasing the owner lock. This prevents both an
orphaned body and repeated cancellation interrupting host cleanup. A host that
refuses cancellation remains subject to the existing session cleanup deadline;
the runtime does not report its early semantic result as actual resource release.
Entries use the captured `(invocation_id, operation)` identity, so an older body's
completion or cancellation cannot affect a later reuse of its tool call ID.

Tool commands use the same captured identity through the typed ABI `VoiceCommandOrigin`.
This replaces the former string origin, rather than adding another attribution registry.
Tool origins carry invocation, operation and call IDs; hooks and systems retain their explicit
variants, and an omitted origin remains the host. Progress, completion, foreground holds,
fillers and speech self-wait checks cannot resolve an old body to a replacement call merely
because its provider reused a call ID. Core resolves the exact running owner; runtime owns
execution and cancellation but does not classify expert questions as terminal.

The native voice-run view replaces and orders recorded items, freezes them on
settlement and projects the final output. The host exposes those values and
adapts native failures to its future or promise. A2A handler normalization and
journal projection live in core; a run supplies its selected pending call as
typed data. No host reconstructs a pending request by parsing tool-output text.

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

Four protocols keep sans-I/O machines (inputs in, effects out) inside
`lk-agents-inference`, outside the purity lints that bind the cores: realtime, VAD,
the fallback chain and interruption; the media `AudioEmitter` is the fifth. The
LLM, STT, TTS and avatar protocols are plain async code in their pumps: a phase
enum and ordinary awaits, tested with the provider fakes.

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
| Agent authoring | Configuration, callback registration order and hook replacement, freezing, retry budgets and configuration projection | Core stores callable IDs and plain data; the host retains callable objects, inspects types and discovers module assignment names. Public retry and model-setting records are generated |
| Request streams | Consumer/read admission, event envelopes, text accumulation and presentation accounting | Core owns the consumption state; runtime binds native runner sources directly, executes cancellation and settles sources/resources with injected execution ports. Cursors retain admission through language delivery. Python hydrates typed results and retains exception identity; it owns no stream task or presentation buffer |
| Text invocation resources | Temporary namespace/thread creation, context borrowing and owned activation lifetime | One native owner surrounds a direct run or every frame of a text-session turn. Cancelled startup/close waiters cannot abandon cleanup. Toolset cleanup settles before owned context/storage close; Python only hydrates thread/store handles and releases cached language objects |
| Toolset lifetimes | Pending acquisitions, sharing, last-user release, cancellation and scope selection | Runtime executes declared setup/close callbacks through HostInvoker. Failed initialization settles acquired roots and nested children before returning the failure, in both text and voice. Failed parents do not run their authored close. Ready lifetimes retain no host callback; closed lease groups release theirs |
| MCP | Tool listing, schema rendering, result and error mapping, pagination, close decisions and HTTP response classification (`lk_agents_core::mcp::http`), in `lk-agents-core` | The runtime's thin session I/O uses the SDK for stdio and Streamable HTTP and only executes; do not duplicate the MCP protocol engine |
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
| RoomIO | Linked participant, track/publication generations and attachment lifecycle | Runtime subscribes/publishes through host RTC proxies; late operations clean up obsolete resources |
| Pre-connect audio | Buffered input ownership until attachment | Transfer frames once; normalize to the session format before the live processing chain; source-format processors declare that capability |
| User I/O and nodes | Validated capabilities and attachment identities | Runtime invokes host nodes; core chooses policy from supplied capabilities and outcomes |
| Avatar transport | Audio segment and acknowledgment state | Runtime owns byte streams. Between 2.0 peers, each stream's report route (`lk.playback_*/<stream id>/AUDIO_<output>_<segment>`) names its segment; the plain v1 methods stay answered, about the oldest streamed segment, and a stream named any other way reports on them |
| Avatar runner | Publication, generator/receiver progress, clear-buffer and close state | Frames remain opaque; runtime uses shared RTC audio/video source and publication capabilities, without an avatar-specific media host protocol |
| RTC frame pacing | Frame deadlines, stall recovery, bounded queue capacity and pacing observations | Voice supplies the media plan and clock-driven pacer; runtime owns the queue, injected clock waits and source calls. Source and track teardown survives cancellation of a close waiter |
| Queue audio | Segment admission, clear epochs, stale acknowledgment rejection and interrupted-report order | Voice owns the state machine, runtime wakes readers; Python only translates frames and delivers the resulting playback events |
| Background audio | Clip/loop/fade decisions and mix settings | Pure sample processing and file decoding in media; runtime publishes audio |
| Recording | Timeline/accounting and flush/finish lifecycle | Runtime admits captured frames and hands them to the media pool; media owns the recording file (timeline, encoding, ordered writes) and its bounds; finalization waits for accepted data and occurs once |
| Remote session | Request correlation, cancellation, deadline and drain state | Typed messages over runtime-owned room/TCP transport; late replies cannot settle new requests |
| AMD | Classification phase and deadline | Runtime invokes the model; classification coordinates with the existing session floor |
| IVR and keyterms | Their bounded request/pass state | Invoke model/provider operations as effects; replacement or cancellation invalidates stale results |
| DTMF/end-call/transfer helpers | Their bounded live-interaction state | Obey VoiceTask/session ownership; restore temporary audio and release transfer resources on failure as well as success |
| Codecs and text processing | Deterministic conversion over supplied bytes, frames or text | Synchronous functions or stream accumulators; no hidden files, sockets or threads |

RoomIO and participant input/output use the Python reference's domain vocabulary. In voice,
`room_io::RoomIo` owns pure attachment/readiness state and `ParticipantInput` owns reader
selection. Session-specific consequences live under `agent_session::room`. Runtime owns the
live room, stream pumps and a separate published `RoomView`; it does not store the decision
machine inside the watch channel. Avatar receiver RPC registrations belong to the room instance,
with registration identities so an obsolete receiver cannot unregister its replacement.

Session output assembly owns its transcription wrapper and closes it before room outputs.
Recording attachment facts live in `agent_session::RecorderIo`; `VoiceDriver` coordinates flush
deadlines and effects. Taps and playback observers are runtime objects. Reply plans retain
resource IDs for instruction/output-node callbacks; runtime resolves
those into an execution snapshot. Registry retirement drops objects outside its lock, while an
admitted snapshot stays alive until its execution settles.

The libsoxr wrapper disables default int16-output dithering, whose upstream seed depends on
wall time and handle address. Supplied PCM and conversion settings must reproduce the same PCM
across independent handles; quality recipes and channel/rate semantics remain explicit.

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
| Process pool | Warm capacity, reservations, pending jobs and retry policy | Assign only ready processes; release failed reservations; drain stops admission. The server runtime owns the pool and observes its job status; the host supplies only the process launcher (and forkserver), the started notification and the user's load function |
| Request stream | Bidirectional byte I/O and transport control after the node preamble | The worker dispatches opaque bytes and request metadata through HostInvoker; each binding owns HTTP/WebSocket serving, framework adaptation and application draining |
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
`ThreadDbConnection` execution trait. AgentDB and local SQLite adapters share
`Thread`, `Partition`, `Run` and `Lease` engine types, including typed content reads
that require no partition lease. The [storage plan](storage.md#rust-interface)
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
| Partition ownership | Acquisition, renewal, fencing and release | Storage performs them and reports `LeaseLost`; stale owners cannot commit |
| Managed run | Loaded revision, staged writes and commit/abandon outcome | One fenced commit; explicit application saves; transport acknowledgment and commit boundaries are distinct |
| A2A task | Arrival policy, message deduplication, pending input and terminal transition (`lk_agents_core::a2a`) | Typed handler/checkpoint outcomes drive policy; completed transport does not imply a persisted Python continuation |

`lk-agents-storage` is a plain async crate. It owns all database interaction:
both backends, retries, fencing, lease renewal and resolution of lost
acknowledgments. Every failure reaches the caller as a precise typed error.
A database failure is `Uncertain` (the outcome is unknown), `Rejected { running_task_id }`,
`LeaseLost`, `Unavailable` (the one retryable error), `NotFound` or
`VersionConflict`; a request the caller made wrongly is `Invalid`,
`NotSerializable`, `TooLarge` or `Unauthorized`; the handler's own save refused
through its turn is `Refused(HostError)`; a turn the host cancelled is
`Cancelled`. The A2A task
decisions in `lk_agents_core::a2a` are sans-I/O over request/result records and
error types declared in ABI; core has no dependency on storage. The runtime's
thin A2A door executes core's storage effects by calling storage and feeds the
typed results back. Storage is tested against real SQLite and a fake AgentDB
server with injected faults. A2A and TextSession behavior is covered by core,
runtime and storage component tests; the DST is limited to voice and room simulation.

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
[crate responsibilities](architecture.md#crates-and-dependencies). Chat formatting,
tool-argument processing, JSON repair, speech tokenization and expressive markup
belong to `lk-agents-core`; transcript pacing belongs to `lk-agents-voice`; media codecs,
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

Spans, metrics and report records are declared in `lk-agents-abi`, produced by
the subject that owns the work, recorded through `lk-agents-telemetry` and exported by
`lk_agents_net::export` ([telemetry](runtime.md#telemetry)).
Observers cannot become a second policy owner. Binding construction and
conversion follow the [binding rules](bindings.md#binding-rules).
