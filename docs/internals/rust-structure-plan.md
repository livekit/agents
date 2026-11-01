# Rust structure: refactoring plan, audit, and inventory

This is an implementation planning document, not a change to the public API specification.
It records the full restructuring discussion and independent static audit of the working tree
on 2026-11-01. No implementation changes or behavioral test campaigns were performed for this
audit. The checkout already contained substantial tracked and untracked work; the inventory
includes current files from both, and does not describe a clean committed release.

The feature documents in [the documentation index](../README.md) remain authoritative for
behavior. [Rust core](rust-core.md) defines implementation boundaries. The original Python
implementation on the local main branch was used to understand domain ownership, not as a
behavioral oracle. [Python ownership](python-ownership.md) covers the complementary host work.

## Executive summary

Organize around recognizable domain components: RoomIO, participant input/output, AgentActivity,
AudioRecognition, generation, transcription, tool execution, and session orchestration.
Preserve sans-I/O and the DST's control of execution. Reduce scattered ownership before changing
directories. A smaller file count alone is not success.

- Keep existing crate boundaries for this refactor. Do not add feature crates or merge core and
  voice merely to remove repeated folder names.
- Remove execution objects and runtime queue envelopes from decision components.
- Make state, resource lifetime, effect ordering, queue policy, and configuration authority explicit.
- Fold incidental abstractions into their domain owners. RoomCoordinator and TrackReaders are
  not privileged foundations to preserve; their necessary invariants are.
- Replace distributed manipulation of Driver and SessionShared with component ownership and narrow
  execution capabilities. Driver keeps decisions that span components.
- Keep useful decision/execution boundaries without mirroring every private helper across crates.
- Preserve behavior, operation correlation, cleanup settlement, and actual DST coverage per stage.
- Update architecture enforcement, code generation, bindings, and documentation with source moves.

## Non-negotiable contracts

1. Decision state in core and voice remains synchronous and sans-I/O: supplied time and IDs,
   deterministic iteration, ordinary records and resource identities, synchronous transitions,
   and effects executed elsewhere. No tasks, channels, locks, live SDK/provider objects or ambient
   clocks in the decision state being refactored.
2. Runtime owns asynchronous resources, pumps, timers, output binding and execution. Provider,
   transport, storage and worker crates keep their legitimate async I/O responsibilities.
3. DST continues to exercise the production components it currently exercises over fake ports.
   Do not replace real executors with test-only approximations to simplify a refactor.
4. Time, cancellation, publication order, resource retirement and late completions are part of
   behavior. Preserve their contracts even when an abstraction or wrapper disappears.
5. v2 feature documents govern public behavior. Python reference concepts guide ownership;
   historical Python quirks and giant Python files are not design requirements.
6. No public API changes, version bumps, new dependency edges, or broadened architecture
   exemptions are part of this structural project. Discuss unresolved API choices separately.
7. Never edit generated files manually. Change declarations/implementation metadata and regenerate.
8. Handwritten Rust stays within the 1,250-line rule. Split substantial behavior within its owner;
   do not recreate confetti by assigning one file to every enum, helper, or exported method.

## First-principles decision rules

Every abstraction must have a domain responsibility, exclusive state/invariants, an explicit
creation/retirement lifetime, and an interface callers can use without reaching into its fields.
Separate it when that separation materially helps local reasoning, reuse, or execution isolation.
Otherwise keep it private beside its owner. A struct does not automatically require a trait,
module, event enum, executor, lifecycle facade, and factory.

Sub-state machines are an implementation technique, not the organizing principle. A component can
have ordinary synchronous methods and private helpers; do not force a universal handle(event)
interface or a global Event/Effect enum on everything. Stateful components return meaningful
outcomes, and Driver decides interactions spanning components.

Repeated subject names can denote distinct roles. For example:

| Existing location | Actual role | Disposition |
| --- | --- | --- |
| abi/tts | Provider contract and boundary records | Keep contract ownership; consolidate tiny declaration fragments |
| core/tts | Shared speech text processing and markup | Keep shared algorithms; keep private helpers local |
| inference/tts | Provider synthesis, streaming and adapters | Keep provider execution; review incidental internal layers |
| voice/generation/tts | Synthesis within a spoken reply | Keep reply decisions distinct from provider behavior |
| runtime/voice/generation/tts | Execute that reply's synthesis pipeline | Keep necessary execution boundary |

Do not create matching directories to make these trees symmetrical. A small ABI contract need
not mirror its implementation's files. A private synchronous provider helper can remain in
inference; synchronous code does not automatically belong in core. Framework decisions that must
participate in the session simulation belong in the decision layer.

Merging core and voice would not remove ABI/runtime boundaries. Adding pairs of crates per
feature would multiply boundaries. Neither is justified by the current navigation problem.

## Crate ownership inventory

| Crate | Owns | Structural action |
| --- | --- | --- |
| lk-agents-abi | Export declarations, typed records/errors, shared ports, host vocabulary | Remove misplaced framework policy; consolidate declarations by coherent contract |
| lk-agents-core | AgentRun, tools/approval, TextSession, A2A/MCP policy, shared algorithms | Preserve meaningful machines; localize private helpers and history policy |
| lk-agents-voice | Voice/session decisions and synchronous media coordination | Main ownership refactor; remove execution-bearing types/adapters |
| lk-agents-runtime | Session executors, resource ownership, host invocation, process composition | Narrow capabilities; clarify lifetimes, queue policy and execution protocols |
| lk-agents-inference | Provider clients, retry/fallback execution, adapters, detectors, judges | Consolidate incidental stream layers; preserve provider-local boundaries |
| lk-agents-media | Synchronous media transforms, codecs, framing, sample arithmetic and media pool | Preserve; clarify shared playback/accounting ownership where needed |
| lk-agents-net | Transport, source lifecycle, tasks, clocks, blocking I/O, A2A client, report upload | Preserve distinct shared services; inspect mutable process state and helper forwarding |
| lk-agents-storage | Database operations, backends, fencing, leases, context and response streams | Keep storage semantics; clarify domain names and adapter surfaces |
| lk-agents-worker | Registration, jobs, IPC, process pool, supervision, endpoint bytes | Keep real lifetime boundaries; distinguish host launch from native execution |
| lk-agents-cli | Launcher, remote CLI, logs and shutdown deadline | Preserve; validate lifecycle integration when touched |
| lk-agents-py | Registration, conversions, host invocation entry and Python ownership | Keep pyo3 isolated; regenerate changes and preserve object/exception identity |
| lk-agents-fuzz | Controlled inputs, fake ports, deterministic scheduling, observations/contracts | Preserve campaign-specific boundaries and useful runtime interleavings |
| lk-agents-bindgen | Declaration parser and resolved declaration model | Update only as necessary; do not add a second declaration system |
| lk-agents-host-derive | Inert host/vocabulary markers | Preserve declaration roles |
| lk-agents-codegen | Generated bindings and architecture checks | Regenerate paths; maintain failing enforcement fixtures and drift checks |

## Confirmed structural findings and audit additions

These findings are based on source inspection. They are not claims that a runtime failure has
been reproduced. A source inventory entry alone is not evidence of a line-by-line semantic audit.

| ID | Finding and evidence | Required response |
| --- | --- | --- |
| F01 | Driver has inherent implementations across 28 voice source files, including activity, RoomIO, recorder, results, tools and workflows | Move local state/transitions behind domain owners; retain cross-component orchestration |
| F02 | room_io/mod.rs holds STT/VAD StreamOpened/Opened, InputProviders and RECORDER_FLUSH | Move provider-leg transport types and recorder policy to their real owners |
| F03 | RoomIO spans ABI room_io/room_owner/rtc, voice room_io, runtime room_io/rtc and session construction | Define each role and consolidate incidental navigation boundaries |
| F04 | voice/recorder_io.rs defines RecorderTap and RecorderPlaybackObserver, reads a clock and calls recorder methods | Move execution adapters to runtime; keep recording decisions pure |
| F05 | voice/generation::Providers holds live LLM/TTS objects; OutputNodes holds NodeFactory; commands carry HostAudio/HostRecorder | Separate execution snapshots/capabilities from decision inputs and state |
| F06 | voice/agent_session/driver.rs defines both the runtime queue envelope and decision input; Input::Room carries SDK RoomEvent while on_rtc handles only Close | Runtime owns queue messages and converts them to minimal decision observations |
| F07 | ABI llm/chat_context/history.rs implements copy/truncate/merge policy called by core ChatPlans | Move selection policy into core; retain appropriate contracts/codecs in ABI |
| F08 | runtime SessionShared exposes mutable publishers, resources, recorder, speech/call registries, queues, IDs and configuration | Give tasks narrow capabilities; establish one owner per mutable state |
| F09 | runtime voice Resources includes tool-progress response channels as well as resource objects | Separate tool-call response correlation from resource lookup |
| F10 | runtime voice/effects.rs publish_views partitions effects before other operations | Make publication phases explicit and preserve callback/receipt visibility guarantees |
| F11 | Out composes transitions, signals, planned work, timers and journal effects | Map guarantees before simplifying intermediate representations |
| F12 | Runtime queues mix bounded, drop-oldest, unbounded and terminating-on-overflow behavior | Inventory capacity, overflow, backpressure and shutdown per queue |
| F13 | OwnedTask cancels on drop; TaskSet retains work until explicit close/drain; acquisitions and handles have additional drop behavior | Map every lifetime before removing wrappers; cancellation request is not settlement |
| F14 | avatar/receiver.rs RX_ROUTES is process-global and keyed by room name/caller; fuzz start chooses unique room names because of global tables | Scope routing to the room instance and own registration cleanup; collision is a risk, not a reproduced failure |
| F15 | Voice DST uses real runtime execution, A2A uses real turn/storage logic over fake DB, TextSession campaign steps core directly | Preserve each distinct coverage boundary; add runner conformance where it is outside a campaign |
| F16 | Fuzz lib/boundary/strategy comments contain descriptions of an older design and transitional task policy | Reconcile descriptions with implementation; do not remove useful task interleavings based on stale labels |
| F17 | architecture/check.py EXECUTORS selects runtime voice/room_io/avatar/rtc by path | Update enforcement and failing fixtures alongside moves; prevent unchecked destination paths |
| F18 | ProviderLayers uses string provider slots and raw u64 values | Use closed provider-slot vocabulary and appropriate resource/layer identity types |
| F19 | SessionOptions, provider layers, active config and reply snapshots coexist | Document authoritative state and transfer/snapshot semantics before consolidation |
| F20 | Storage's SqliteConnection trait is backend-neutral and implemented for more than local SQLite | Rename the port to reflect thread database execution; keep backend names on implementations |
| F21 | Host launcher and native runner contracts both use JobExecutor terminology | Distinguish roles; preserve real process/thread/supervisor boundaries |
| F22 | Runtime's public surface mixes execution APIs and implementation re-exports consumed by generated bindings | Distinguish access paths and narrow ordinary module visibility |
| F23 | Avatar decision runner is top-level avatar_runner while runtime runner is under avatar; transcription executor is under session | Place each with its domain owner and eliminate incidental location asymmetry |

Evidence starting points (repository-relative paths):

- [Voice Driver boundary](../../rust/crates/lk-agents-voice/src/agent_session/driver.rs)
- [RoomIO subject root](../../rust/crates/lk-agents-voice/src/room_io/mod.rs)
- [Recorder adapters currently in voice](../../rust/crates/lk-agents-voice/src/recorder_io.rs)
- [Generation snapshots](../../rust/crates/lk-agents-voice/src/generation/mod.rs)
- [ABI history policy](../../rust/crates/lk-agents-abi/src/llm/chat_context/history.rs)
- [Session shared capabilities](../../rust/crates/lk-agents-runtime/src/voice/session/shared.rs)
- [Resource registry](../../rust/crates/lk-agents-runtime/src/voice/resources.rs)
- [Effect execution/publication](../../rust/crates/lk-agents-runtime/src/voice/effects.rs)
- [Task ownership contract](../../rust/crates/lk-agents-abi/src/utils/spawner.rs)
- [Task execution and sets](../../rust/crates/lk-agents-net/src/task.rs)
- [Avatar receiver routes](../../rust/crates/lk-agents-runtime/src/avatar/receiver.rs)
- [Campaign boundary inventory](../../rust/crates/lk-agents-fuzz/src/boundary.rs)
- [Real voice-session startup](../../rust/crates/lk-agents-fuzz/src/drive/start.rs)
- [Executor enforcement](../../rust/architecture/check.py)

## Workstream inventory and current-to-target map

All rows below are planned, not implemented. Target ownership is the commitment; exact private
type/file names remain implementation proposals. In particular, RoomIOState is a descriptive
name, not a requirement to introduce another facade around RoomCoordinator.

| ID | Current subjects | Target and concrete action | Findings |
| --- | --- | --- | --- |
| W01 | ABI exports, records, errors, primitive codecs | Group by coherent contract; retain declared host names; regenerate adapters | F07, F22 |
| W02 | core llm chat_plans; ABI chat_context/history | Core owns history selection; ABI owns declarations and suitable representation codecs | F07 |
| W03 | voice agent_session and scattered Driver impls | Driver/session owns lifecycle, active activity, floor and cross-component ordering | F01, F06, F11 |
| W04 | voice agent_activity hooks/handoff/voice_task/drain/idle | Activity owns agent-local state; explicit session interactions for handoff and floor changes | F01 |
| W05 | scheduling, speech, false_interruption, filler_scheduler | Activity-owned scheduling and interruption with local state/invariants; avoid a helper machine per field | F01 |
| W06 | audio_recognition, endpointing, recognition integration | Recognition owns transcripts, speech/EOT observations and local transitions; Driver coordinates reply requests | F01 |
| W07 | agent_activity/io; audio; room_io video_sampler and provider opening types | Agent input owns leg replacement, sampling and enablement; runtime owns stream-opening envelopes | F02, F05, F06 |
| W08 | RoomCoordinator and room link helpers | RoomIO's pure attachment, selection, readiness and close state; merge callers and transition ownership | F03 |
| W09 | TrackReaders/input_state, processor spans and input admission | Participant input owns selection/generation/attachment state and admission; runtime pumps frames | F03 |
| W10 | room_io output/publication/playout | Participant audio output owns publication/buffering/pause/refeed/reporting; share only genuinely reusable accounting | F03 |
| W11 | room_io transcription and output/segment_ids | Participant transcription output owns attachment and publication state; IDs scoped to correct lifetime | F03 |
| W12 | pre_connect_state and pre_connect_audio | Pre-connect audio owns buffer admission/replay decisions with runtime decode/pump | F03 |
| W13 | runtime rtc events/wait/playout/synchronizer; voice room_io ingress/wait | Shared RTC execution stays narrow; place pure helpers with actual consumers, not forced mirror directories | F03, F06 |
| W14 | voice room_io/detached; ABI room_io and room_owner | Detached implementation in runtime; coherent room contract declarations; preserve public surfaces | F02, F03 |
| W15 | runtime session class output selection and sync creation | Session I/O assembly chooses supplied/room legs and owns wrapper cleanup | F19, F23 |
| W16 | voice transcription lifecycle/pacer/synchronizer and runtime session/transcript_sync | Transcription owns sync decisions and corresponding runtime execution; private algorithms remain local | F23 |
| W17 | playback, room source ledger, generation playout and trackers | Distinguish shared tracking, source accounting and reply playout; preserve correlation and heard output | F10, F13 |
| W18 | recorder_io, runtime voice/recorder, runtime recording, media recorder, net report upload | Separate pure recording policy, playback observer, ingress, file work and upload; one session integration owner | F02, F04, F13 |
| W19 | generation, reply_run, runtime generation pipeline/realtime/tools/LLM/TTS | Reply owns planning, follow-ups and outcomes; runtime owns execution snapshots and pumps | F05, F10, F11 |
| W20 | tool_executor, tool_body, activity tools, shared core tool_calls/approval/outcome | Localize voice tool execution and response correlation; preserve shared approval policy | F01, F09, F13 |
| W21 | experts, workflows DTMF/end_call/warm_transfer, IVR, AMD, keyterms | Preserve distinct domain behavior and request/floor identities; interact through explicit session interfaces | F01 |
| W22 | turn, activity options, provider_layers, runtime options/providers | One declared configuration authority per lifetime; typed slots/IDs; explicit override release and snapshots | F18, F19 |
| W23 | runtime session owner/class/builder/mod/shared/calls and Runtime | Clear outer owner, construction, running executor, command handle, resources and storage roles | F08, F13, F22 |
| W24 | SessionShared, PeerHandle, Resources and tool-progress table | Narrow execution capabilities; separate resource registry, replies, publishers and observational access | F08, F09 |
| W25 | voice Event/Input/StreamOpened and runtime queue/effect execution | Runtime owns transport envelopes; pure inputs use observations/IDs; make effect phases explicit | F05, F06, F10, F11 |
| W26 | runtime queue/channel sites across voice, room, avatar, transcription and streams | Record queue protocols; decide overload behavior without mechanical bounded-channel conversion | F12 |
| W27 | OwnedTask, TaskSet, SessionOwner, RunResources, toolsets, speech, room attachments | Lifetime ownership/settlement map; retain owners with distinct contracts; consolidate only equivalent mechanisms | F13 |
| W28 | voice avatar/avatar_runner and runtime avatar/rtc AV sync | Avatar runner/receiver/output/queue under avatar; keep gateway provisioning in inference | F14, F23 |
| W29 | avatar RX_ROUTES and mutable process-global registries | Room-instance routing with explicit registration retirement; separate genuine process services/caches | F14 |
| W30 | inference LLM/STT/TTS/VAD/realtime/duplex/fallback/detectors | Review machine/lifecycle/stream/pump/ports/adapter divisions; localize plumbing; preserve modality contracts | F13 |
| W31 | core attempts, inference fallback, net HTTP/WS/source/pool | Preserve shared retry/transport semantics and replay safety; remove incidental forwarding only | F13 |
| W32 | media codecs/framing/images/resampling/filtering/pool; inference decode/emitter | Keep CPU work in media and provider-side scheduling in inference; preserve bounded admissions | F12, F13 |
| W33 | core tokenize/tts markup, voice tokenize and transform integration | Keep shared algorithms coherent; remove forwarding-only modules when justified | F22 |
| W34 | core/runtime AgentRun, TextSession, run_stream and resource setup | Keep substantive decision/execution split; localize helpers; verify runner/stream/resource settlement | F13, F15 |
| W35 | core/runtime A2A/MCP and net delegation client | Preserve protocol policy versus transport/storage execution; private details stay within subject | F15 |
| W36 | storage engine/db/SQL/backends/context/handles/voice/streams | Keep fencing, uncertain-commit resolution and explicit context semantics; clarify names and adapter surfaces | F20 |
| W37 | worker jobs/registration/server/transport/IPC/supervisor/pool and CLI | Preserve launch/supervision/OS lifetime boundaries; distinguish host/native executor names | F21 |
| W38 | telemetry observations/admission/projection, host SDK delivery, cloud reports | Keep policy, provider accounting, host delivery and report upload distinct; retain one accounting authority | F10 |
| W39 | Python binding implementation access, errors/refs, codegen/derive/bindgen | Preserve HostInvoker, GIL and exception/object ownership; declarations remain ABI-only; regenerate | F05, F22 |
| W40 | fuzz inputs/drive/scheduler/fakes/records/rules/search and campaigns | Preserve actual production coverage and task schedules; refresh boundary inventory and useful contracts | F15, F16 |
| W41 | architecture path rules, baselines, codegen placement, build manifests and docs | Update enforcement alongside moves; preserve failing fixtures, feature coverage and shrinking baselines | F17 |
| W42 | crate lib surfaces and module visibility throughout | Private by default; distinguish binding access paths from ordinary implementation APIs | F22 |

### RoomIO target in more detail

The Python reference's objects are the starting point: RoomIO, participant audio/video input,
participant audio output, participant transcription output, and pre-connect audio. The two
current room_io directories contain 20 production Rust files and approximately 6,850 lines,
before ABI, RTC and session integration. Multiple files are justified; unrelated ownership is not.

- RoomIO owns participant attachment identity, selection, readiness and room lifecycle decisions.
- Participant input owns track selection, obsolete-reader rejection, attachment generation,
  frame admission, processor failure spans, and bounded reader retirement.
- Participant audio output owns publication state, segment buffering, pause/refeed, clear and
  playout reporting. Publication may remain a private substantial helper.
- Participant transcription output owns participant/track attachment, stream and legacy caption
  publication state, and transcript encoding where appropriate.
- Pre-connect audio owns admission/replay decisions; decode and resource I/O remain runtime work.
- Driver owns session consequences: disconnect closes session, expert/floor coordination,
  session options and final room deletion requests.
- Runtime observes SDK objects, executes operations and returns correlated outcomes. It does not
  move attachment/selection policy out of the pure component.

RoomCoordinator/TrackReaders disappear as externally navigated concepts, not by losing safeguards.
Do not replace them with equally scattered state/factory/owner/adapter facades under new names.

### Session and activity authority

| Owner | Exclusive responsibility | Interaction |
| --- | --- | --- |
| Driver/session | Session lifecycle, active activity, floor and cross-component ordering | Route observations, combine outcomes, authorize dependent work |
| AgentActivity | Agent configuration, hooks, activity lifecycle and speech scheduling | Report requested replies, handoff outcomes and local state changes |
| AudioRecognition | Recognition/turn state and transcript accumulation | Report recognition outcomes; do not reach into activity/session fields |
| Agent input/output | Leg replacement, enablement and output binding policy | Return install/retire decisions over identities |
| Generation | One reply's planning, model/tool steps and final/interrupted outcome | Runtime executes provider and host operations |
| Tool execution | Calls, approvals, progress and settlement correlation | Shared core policy where reusable; runtime retains replies/resources |
| Workflows/experts | Domain interaction state with preserved request/floor identity | Explicit session interactions; no accidental VoiceTask collection extension |

Keep scheduling and false-interruption ownership together. Handoff and VoiceTask suspension span
activity lifetimes and therefore need explicit session coordination. Selecting an agent must
preserve session configuration and temporary override semantics.

## Required protocol inventories before implementation

Complete these tables for each touched vertical slice, linking the responsible workstream and
evidence/tests. These are implementation tasks, not already completed runtime audits.

### Lifetime checklist

For process, room attachment, outer session, running session, activity, speech, invocation,
provider stream, toolset acquisition/lease, storage lease and output segment, record:

- Creator and sole authority for start/close admission.
- Which caller may cancel, and whether cancelling an awaiter cancels the operation.
- Owner after the caller drops, and owner of late successful resource creation.
- Cancellation request, close request, drain, physical settlement and final acknowledgment.
- Deadline authority and retained cleanup when a deadline expires.
- Final drop behavior and how ownership cycles are prevented.

### Queue checklist

For session command/event/host queues, watcher taps, microphone/video buffers, provider openings,
realtime player, transcript forwarding, avatar frames/RPC/streams, recording ingress, tool progress,
AgentRun/TextSession updates and A2A delivery, record:

- Producer(s), consumer and ordering domain.
- Capacity and byte/frame/work budget, including deliberately unbounded queues.
- Full action: await, reject, drop-oldest, coalesce, fail subscriber/operation, or terminate.
- Whether completion/control traffic can be blocked behind payloads.
- Cancellation safety, close behavior, and treatment of unread items/resources.
- Observable overflow signal and a meaningful regression contract.

Do not adopt a single channel wrapper for all policies or bound every queue mechanically.

### Ordering and identity checklist

- Publish accepted views before callbacks/receipts that are entitled to observe them.
- Preserve supplied-time wake-before-same-instant-input ordering.
- Distinguish command submission order from operation completion/dependency authorization.
- Preserve attachment, participant lifetime, track selection, speech, request, invocation,
  playback segment, override layer and task identities. Introduce nominal types where misuse is
  plausible; do not create wrappers without a concrete invariant.
- Reject stale events, late opens and old handoff/override completions without losing cleanup.
- Preserve accepted-history order, tool approval/read-back identity, request snapshots and
  transcript/heard-audio correlation.
- Keep raw SDK observations and resource-bearing envelopes outside decision inputs.

### Configuration authority checklist

Map pre-start options, running session layers, provider override layers, active agent configuration,
and immutable per-reply snapshots. Record when ownership transfers, how updates under overrides
are retained, out-of-order override release, provider resource retention, retries of failed startup,
and agent replacement. Do not leave multiple mutable copies acting as independent authorities.

## DST and verification inventory

| Campaign/boundary | Current implementation evidence | Must preserve or add |
| --- | --- | --- |
| Voice | drive/start builds real VoiceSession; real runtime loop, room/avatar/RTC execution over fake ports | Real session entry, resource ownership, cancellation and execution schedules |
| Voice scheduling | Arrival, BasicPos and TaskBridge policies | Keep useful task-level interleavings; remove only after explicit equivalent-coverage evidence |
| A2A | Runtime turn with real storage logic over fault-injected in-memory backend | Lease/fence/uncertain-write behavior and turn correlation |
| TextSession | Core TextSession/AgentRun directly stepped | Keep core campaign; runtime runner, stream and binding conformance separately |
| Provider clients | Outside voice fake-provider boundary; provider transport/peer suites | Real provider stream/retry/fallback tests; fake outcomes cannot prove their implementation |
| Python binding | Cargo-linked embedded contracts in lk-agents-py | GIL release, HostInvoker scopes/cancellation, identity, ownership and generated conversion |
| Worker/IPC | Own process/transport/supervisor suites | Launch/exit/reap, health, process/thread distinctions and cleanup |
| Real RTC and DB backends | Outside in-memory port fakes | Relevant peer/backend conformance when those adapters change |

Refresh fuzz lib.rs, boundary.rs, TEST_STRATEGY.md, COVERAGE.md if present, manifest/target names and
launch-gate descriptions from implementation. Historical comments are evidence to reconcile,
not permission to reduce coverage. Preserve probes, counterexamples and regression cases; compare
meaningful contract reach as well as pass/fail. Seeds reproduce against the same implementation;
a refactor can change scheduling/generation, so keep authored minimal cases for known defects.

Behavioral checks per affected slice:

- Room startup/close races, cancellation before ready, participant replacement and identity reuse,
  stale track events, processor failures, pre-connect replay and room deletion once.
- Publication failures/late success, clear/pause/refeed, segment identity, unheard audio trimming,
  transcript timing and cleanup acknowledgment.
- Provider replacement while streams open, handoff during hooks/tools/replies, interruption and
  speculative reply adoption, out-of-order override release and input/output enablement.
- Approval repeat/read-back, expert request identity, human reply routing, VoiceTask parent
  restoration and workflow/floor interaction.
- Streaming consumer exclusivity, explicit cancellation, source settlement before result retrieval,
  and explicit application state/context persistence.
- Storage lease loss, uncertain commit, fencing, cancellation and final owned-resource settlement.
- Multi-session isolation, same-name room routing, overflow, slow consumers and close under load.

Use targeted tests first, then affected crate/DST/binding gates. Do not add tests that merely
mirror private implementation layout. Relevant commands from repository root include:

    make -C rust format-check
    make -C rust check
    bash rust/architecture/check.sh --tests
    uv run --no-sync cargo test --manifest-path rust/Cargo.toml -p lk-agents-py --features ffmpeg
    rust/crates/lk-agents-fuzz/launch-gate.sh
    uv run pytest --unit --audio_eot

Select ordinary Rust crate/target tests using their Cargo manifest declarations. Run Python gates
when host adapters/public behavior change; do not separately collect colocated embedded binding
tests. Use a private Cargo target per worktree, preserve required features, and do not share a
writable target across concurrent worktrees. These commands are validation requirements for
implementation; they have not been run as part of this planning audit.

## Implementation stages and acceptance

| Stage | Scope | Exit condition |
| --- | --- | --- |
| S0: Baseline | Current ownership map, dirty-checkout baseline, protocol/lifetime/queue/config inventories, test coverage and enforcement scope | Every affected owner and invariant mapped; distinguish confirmed facts/proposals/risks |
| S1: RoomIO | W07-W14, relevant W23-W29/W39-W42 | Recognizable room components across pure/execution boundary; stale/cancel/cleanup tests and real executor DST preserved |
| S2: Session I/O | W15-W18 and shared capability/lifetime work | Output selection/sync/recording have explicit ownership; execution adapters outside decision crate |
| S3: Voice orchestration | W03-W06, W19-W25, W27 | Component-local state no longer manipulated throughout Driver; cross-component decisions remain explicit |
| S4: Shared/provider work | W01-W02, W28-W33, W38 | ABI policy corrected; provider/avatar/markup internals localized without flattening real protocols |
| S5: Adjacent cleanup | W34-W37, W39-W42 as needed | Targeted naming/visibility/helper cleanup; no gratuitous rewrites of coherent infrastructure |
| Every stage | W26-W27, W39-W42 | Update inventory, enforcement, generated code, documentation and relevant behavioral evidence together |

Implement vertical slices across contracts, pure decisions, execution and tests. Avoid a wholesale
rename followed later by a behavioral rewrite. Preserve unrelated working-tree edits. Do not move
an exemption to a new identity or relax a baseline to make a source move pass.

Completion means a reader familiar with the Python domain can find the component, understand its
state/lifetime locally, and trace its execution boundary without a chain of incidental abstractions.
It does not mean every component fits one file or every subject disappears from all but one crate.

## Open implementation decisions and limits

- Exact private file/type names and component signatures require the vertical-slice design;
  this document fixes responsibilities rather than inventing all interfaces in advance.
- Room-instance RPC registration ownership and collision behavior need an authored regression
  before implementation; the audit identified coupling, not a reproduced production incident.
- Queue capacity and overflow changes may alter observable behavior. Preserve existing semantics
  during moves; resolve any intentional contract change against the owning feature document.
- Inventory callable capabilities separately from opaque retained host payloads and exceptions.
  Preserve host identity/refcount/GIL guarantees while making decision purity explicit.
- Do not generalize away modality differences, process versus thread lifetimes, uncertain database
  outcomes, or shared retry/replay safety because names look similar.
- No implementation, compile result, benchmark or runtime coverage result is implied by this plan.

## Progress ledger

- [x] Capture agreed principles and complete restructuring scope.
- [x] Record independent audit findings, evidence and risks.
- [x] Assign workstreams and implementation stages.
- [x] Inventory current workspace source files and explicit test targets below.
- [ ] Complete per-slice lifetime, queue, ordering, identity and configuration tables in S0.
- [ ] Record behavioral baseline and contract reach for the first slice.
- [ ] Implement S1-S5 with stage-specific validation and inventory updates.
- [ ] Reconcile Rust architecture and DST documentation with final implementation.

## Source and test inventory snapshot

The appendix below is a filesystem/Git inventory, not a claim that every listed file received a
semantic audit. It includes tracked files that still exist and nonignored untracked files. Deleted
files and ignored build products are excluded. Workstream assignments are routing aids: a source
can participate in more than its primary listed workstream. Generated source is counted separately
and must be regenerated, never edited manually. Test files retain their behavioral ownership when
implementations move; fixture and checker counterexample files remain verification infrastructure.

The inventory is a snapshot. Refresh it after structural stages and assign any new files to an
owner/workstream so the plan does not drift from the checkout.

Native C/C++ sources, headers, patches and pins are included for coverage of the workspace, not
as proposed framework refactors. Preserve vendored/upstream organization and native build inputs;
the media workstream owns checking compatibility when its Rust adapters change.

### Snapshot coverage

Snapshot commit: c859e2a4c70b8e3ad90dcba1e11331f019cc5965. Working-tree changes are included.

Total inventoried files under rust/: **1494**. Workspace packages: **15**.

| Category | Files |
| --- | ---: |
| Python host/test/tool source | 6 |
| Rust source | 869 |
| build support | 7 |
| configuration/asset | 397 |
| dependency lock | 1 |
| documentation | 4 |
| explicit test target | 40 |
| fixture/counterexample | 10 |
| gate/tool script | 5 |
| generated | 29 |
| package manifest | 16 |
| test/support | 110 |

No generated files are assigned handwritten refactoring work. Their listed workstreams identify
the declarations, generator, or owning implementation to update. File categories are mechanical
inventory labels; inline Rust tests remain inside their containing Rust source entry.

### Explicit Cargo test, benchmark, and example targets

| Package | Kind | Target | Source |
| --- | --- | --- | --- |
| lk-agents-codegen | example | architecture-fixtures | [../../architecture/fixtures.rs](../../rust/ai_slop/lk-agents-codegen/../../architecture/fixtures.rs) |
| lk-agents-codegen | example | io-fixtures | [../../architecture/io/fixtures.rs](../../rust/ai_slop/lk-agents-codegen/../../architecture/io/fixtures.rs) |
| lk-agents-cli | test | shutdown | [src/shutdown_tests.rs](../../rust/crates/lk-agents-cli/src/shutdown_tests.rs) |
| lk-agents-core | test | a2a_attachment_boundary | [src/a2a/attachments_tests.rs](../../rust/crates/lk-agents-core/src/a2a/attachments_tests.rs) |
| lk-agents-core | test | a2a_channel_precedence | [src/a2a/channels_tests.rs](../../rust/crates/lk-agents-core/src/a2a/channels_tests.rs) |
| lk-agents-core | test | a2a_client_metadata | [src/a2a/metadata_tests.rs](../../rust/crates/lk-agents-core/src/a2a/metadata_tests.rs) |
| lk-agents-fuzz | test | a2a_survey | [src/a2a_survey_tests.rs](../../rust/crates/lk-agents-fuzz/src/a2a_survey_tests.rs) |
| lk-agents-fuzz | test | pos | [src/pos_tests.rs](../../rust/crates/lk-agents-fuzz/src/pos_tests.rs) |
| lk-agents-fuzz | test | regressions | [src/regressions_tests.rs](../../rust/crates/lk-agents-fuzz/src/regressions_tests.rs) |
| lk-agents-fuzz | test | soak | [src/soak_tests.rs](../../rust/crates/lk-agents-fuzz/src/soak_tests.rs) |
| lk-agents-fuzz | test | survey | [src/survey_tests.rs](../../rust/crates/lk-agents-fuzz/src/survey_tests.rs) |
| lk-agents-fuzz | test | text_session_survey | [src/text_session_survey_tests.rs](../../rust/crates/lk-agents-fuzz/src/text_session_survey_tests.rs) |
| lk-agents-inference | test | inference | [src/inference_tests.rs](../../rust/crates/lk-agents-inference/src/inference_tests.rs) |
| lk-agents-inference | test | realtime_ctx_diff_scaling | [src/realtime/context_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/context_tests.rs) |
| lk-agents-media | test | native_cache | [build/native_cache_tests.rs](../../rust/crates/lk-agents-media/build/native_cache_tests.rs) |
| lk-agents-net | test | a2a_client | [src/a2a_client_tests.rs](../../rust/crates/lk-agents-net/src/a2a_client_tests.rs) |
| lk-agents-py | test | py_adapters | [src/py_adapter_tests.rs](../../rust/crates/lk-agents-py/src/py_adapter_tests.rs) |
| lk-agents-py | test | py_bindings | [src/py_bindings_tests.rs](../../rust/crates/lk-agents-py/src/py_bindings_tests.rs) |
| lk-agents-py | test | py_conversions | [src/py_conversion_tests.rs](../../rust/crates/lk-agents-py/src/py_conversion_tests.rs) |
| lk-agents-py | test | py_references | [src/py_reference_tests.rs](../../rust/crates/lk-agents-py/src/py_reference_tests.rs) |
| lk-agents-runtime | test | a2a_attachment_boundary | [src/a2a/attachments_tests.rs](../../rust/crates/lk-agents-runtime/src/a2a/attachments_tests.rs) |
| lk-agents-runtime | test | a2a_door | [src/a2a/door_tests.rs](../../rust/crates/lk-agents-runtime/src/a2a/door_tests.rs) |
| lk-agents-runtime | test | a2a_turns | [src/a2a/turns_tests.rs](../../rust/crates/lk-agents-runtime/src/a2a/turns_tests.rs) |
| lk-agents-runtime | test | eot | [src/eot_tests.rs](../../rust/crates/lk-agents-runtime/src/eot_tests.rs) |
| lk-agents-runtime | test | images | [src/images_tests.rs](../../rust/crates/lk-agents-runtime/src/images_tests.rs) |
| lk-agents-runtime | test | mcp_lifecycle | [src/mcp/lifecycle_tests.rs](../../rust/crates/lk-agents-runtime/src/mcp/lifecycle_tests.rs) |
| lk-agents-runtime | test | media_measurements | [src/media_tests.rs](../../rust/crates/lk-agents-runtime/src/media_tests.rs) |
| lk-agents-runtime | test | tts_gateway_live | [src/tts_live_tests.rs](../../rust/crates/lk-agents-runtime/src/tts_live_tests.rs) |
| lk-agents-runtime | test | tts_speech_text | [src/tts_tests.rs](../../rust/crates/lk-agents-runtime/src/tts_tests.rs) |
| lk-agents-storage | test | cloud_tasks | [src/cloud_tasks_tests.rs](../../rust/crates/lk-agents-storage/src/cloud_tasks_tests.rs) |
| lk-agents-storage | test | context | [src/context_tests.rs](../../rust/crates/lk-agents-storage/src/context_tests.rs) |
| lk-agents-storage | test | lease | [src/lease_tests.rs](../../rust/crates/lk-agents-storage/src/lease_tests.rs) |
| lk-agents-storage | test | lifecycle | [src/lifecycle_tests.rs](../../rust/crates/lk-agents-storage/src/lifecycle_tests.rs) |
| lk-agents-storage | test | memory | [src/memory_tests.rs](../../rust/crates/lk-agents-storage/src/memory_tests.rs) |
| lk-agents-storage | test | process | [src/process_tests.rs](../../rust/crates/lk-agents-storage/src/process_tests.rs) |
| lk-agents-storage | test | runs | [src/runs_tests.rs](../../rust/crates/lk-agents-storage/src/runs_tests.rs) |
| lk-agents-storage | test | schema | [src/schema_tests.rs](../../rust/crates/lk-agents-storage/src/schema_tests.rs) |
| lk-agents-storage | test | voice | [src/voice_tests.rs](../../rust/crates/lk-agents-storage/src/voice_tests.rs) |
| lk-agents-voice | test | amd_turns | [src/amd_tests.rs](../../rust/crates/lk-agents-voice/src/amd_tests.rs) |
| lk-agents-voice | test | ivr_loops | [src/ivr_tests.rs](../../rust/crates/lk-agents-voice/src/ivr_tests.rs) |
| lk-agents-voice | test | transcript_pacing | [src/transcript_tests.rs](../../rust/crates/lk-agents-voice/src/transcript_tests.rs) |
| lk-agents-worker | test | ipc_channel | [src/ipc/ipc_channel_tests.rs](../../rust/crates/lk-agents-worker/src/ipc/ipc_channel_tests.rs) |

### Complete file index

All paths link to their snapshot location. Primary workstream references point to the W01-W42
table above; they do not prescribe a move for every file. Keep entries for coherent infrastructure
as validation dependencies rather than inventing changes for them.

<details>
<summary>lk-agents-abi — 188 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-abi/Cargo.toml) | package manifest | W01 |
| [src/a2a/auth.rs](../../rust/crates/lk-agents-abi/src/a2a/auth.rs) | Rust source | W01 |
| [src/a2a/card.rs](../../rust/crates/lk-agents-abi/src/a2a/card.rs) | Rust source | W01 |
| [src/a2a/client.rs](../../rust/crates/lk-agents-abi/src/a2a/client.rs) | Rust source | W01 |
| [src/a2a/door.rs](../../rust/crates/lk-agents-abi/src/a2a/door.rs) | Rust source | W01 |
| [src/a2a/handler.rs](../../rust/crates/lk-agents-abi/src/a2a/handler.rs) | Rust source | W01 |
| [src/a2a/metadata.rs](../../rust/crates/lk-agents-abi/src/a2a/metadata.rs) | Rust source | W01 |
| [src/a2a/options.rs](../../rust/crates/lk-agents-abi/src/a2a/options.rs) | Rust source | W01 |
| [src/a2a/remote.rs](../../rust/crates/lk-agents-abi/src/a2a/remote.rs) | Rust source | W01 |
| [src/a2a/turn.rs](../../rust/crates/lk-agents-abi/src/a2a/turn.rs) | Rust source | W01 |
| [src/a2a.rs](../../rust/crates/lk-agents-abi/src/a2a.rs) | Rust source | W01 |
| [src/agent/assertions.rs](../../rust/crates/lk-agents-abi/src/agent/assertions.rs) | Rust source | W01 |
| [src/agent/authoring.rs](../../rust/crates/lk-agents-abi/src/agent/authoring.rs) | Rust source | W01 |
| [src/agent/resources.rs](../../rust/crates/lk-agents-abi/src/agent/resources.rs) | Rust source | W01 |
| [src/agent/run.rs](../../rust/crates/lk-agents-abi/src/agent/run.rs) | Rust source | W01 |
| [src/agent/stream.rs](../../rust/crates/lk-agents-abi/src/agent/stream.rs) | Rust source | W01 |
| [src/agent.rs](../../rust/crates/lk-agents-abi/src/agent.rs) | Rust source | W01 |
| [src/cli.rs](../../rust/crates/lk-agents-abi/src/cli.rs) | Rust source | W01 |
| [src/errors/api.rs](../../rust/crates/lk-agents-abi/src/errors/api.rs) | Rust source | W01 |
| [src/errors/host.rs](../../rust/crates/lk-agents-abi/src/errors/host.rs) | Rust source | W01 |
| [src/errors/host_capture.rs](../../rust/crates/lk-agents-abi/src/errors/host_capture.rs) | Rust source | W01 |
| [src/errors/mod.rs](../../rust/crates/lk-agents-abi/src/errors/mod.rs) | Rust source | W01 |
| [src/errors/provider.rs](../../rust/crates/lk-agents-abi/src/errors/provider.rs) | Rust source | W01 |
| [src/errors/side_table.rs](../../rust/crates/lk-agents-abi/src/errors/side_table.rs) | Rust source | W01 |
| [src/errors/stream.rs](../../rust/crates/lk-agents-abi/src/errors/stream.rs) | Rust source | W01 |
| [src/evals.rs](../../rust/crates/lk-agents-abi/src/evals.rs) | Rust source | W01 |
| [src/inference/avatar_session.rs](../../rust/crates/lk-agents-abi/src/inference/avatar_session.rs) | Rust source | W01 |
| [src/inference/fallback.rs](../../rust/crates/lk-agents-abi/src/inference/fallback.rs) | Rust source | W01 |
| [src/inference/mod.rs](../../rust/crates/lk-agents-abi/src/inference/mod.rs) | Rust source | W01 |
| [src/inference/tts_client.rs](../../rust/crates/lk-agents-abi/src/inference/tts_client.rs) | Rust source | W01 |
| [src/invoke.rs](../../rust/crates/lk-agents-abi/src/invoke.rs) | Rust source | W01 |
| [src/lib.rs](../../rust/crates/lk-agents-abi/src/lib.rs) | Rust source | W01 |
| [src/llm/chat_context/context_tests.rs](../../rust/crates/lk-agents-abi/src/llm/chat_context/context_tests.rs) | test/support | W01 |
| [src/llm/chat_context/history.rs](../../rust/crates/lk-agents-abi/src/llm/chat_context/history.rs) | Rust source | W02 |
| [src/llm/chat_context/plans.rs](../../rust/crates/lk-agents-abi/src/llm/chat_context/plans.rs) | Rust source | W01 |
| [src/llm/chat_context/protocol.rs](../../rust/crates/lk-agents-abi/src/llm/chat_context/protocol.rs) | Rust source | W01 |
| [src/llm/chat_context.rs](../../rust/crates/lk-agents-abi/src/llm/chat_context.rs) | Rust source | W01 |
| [src/llm/ctx_diff.rs](../../rust/crates/lk-agents-abi/src/llm/ctx_diff.rs) | Rust source | W01 |
| [src/llm/duplex.rs](../../rust/crates/lk-agents-abi/src/llm/duplex.rs) | Rust source | W01 |
| [src/llm/events.rs](../../rust/crates/lk-agents-abi/src/llm/events.rs) | Rust source | W01 |
| [src/llm/fallback.rs](../../rust/crates/lk-agents-abi/src/llm/fallback.rs) | Rust source | W01 |
| [src/llm/inference.rs](../../rust/crates/lk-agents-abi/src/llm/inference.rs) | Rust source | W01 |
| [src/llm/llm.py](../../rust/crates/lk-agents-abi/src/llm/llm.py) | Python host/test/tool source | W01 |
| [src/llm/llm.rs](../../rust/crates/lk-agents-abi/src/llm/llm.rs) | Rust source | W01 |
| [src/llm/mcp.rs](../../rust/crates/lk-agents-abi/src/llm/mcp.rs) | Rust source | W01 |
| [src/llm/mod.rs](../../rust/crates/lk-agents-abi/src/llm/mod.rs) | Rust source | W01 |
| [src/llm/persistence.rs](../../rust/crates/lk-agents-abi/src/llm/persistence.rs) | Rust source | W01 |
| [src/llm/provider_format/error.rs](../../rust/crates/lk-agents-abi/src/llm/provider_format/error.rs) | Rust source | W01 |
| [src/llm/provider_format/host.rs](../../rust/crates/lk-agents-abi/src/llm/provider_format/host.rs) | Rust source | W01 |
| [src/llm/provider_format/image.rs](../../rust/crates/lk-agents-abi/src/llm/provider_format/image.rs) | Rust source | W01 |
| [src/llm/provider_format/mod.rs](../../rust/crates/lk-agents-abi/src/llm/provider_format/mod.rs) | Rust source | W01 |
| [src/llm/provider_format/strict.rs](../../rust/crates/lk-agents-abi/src/llm/provider_format/strict.rs) | Rust source | W01 |
| [src/llm/provider_format/tools.rs](../../rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs) | Rust source | W01 |
| [src/llm/realtime.py](../../rust/crates/lk-agents-abi/src/llm/realtime.py) | Python host/test/tool source | W01 |
| [src/llm/realtime.rs](../../rust/crates/lk-agents-abi/src/llm/realtime.rs) | Rust source | W01 |
| [src/llm/schema/defaults.rs](../../rust/crates/lk-agents-abi/src/llm/schema/defaults.rs) | Rust source | W01 |
| [src/llm/schema/mod.rs](../../rust/crates/lk-agents-abi/src/llm/schema/mod.rs) | Rust source | W01 |
| [src/llm/to_dict.rs](../../rust/crates/lk-agents-abi/src/llm/to_dict.rs) | Rust source | W01 |
| [src/llm/tool_executor.rs](../../rust/crates/lk-agents-abi/src/llm/tool_executor.rs) | Rust source | W01 |
| [src/llm/toolsets.rs](../../rust/crates/lk-agents-abi/src/llm/toolsets.rs) | Rust source | W01 |
| [src/logging.rs](../../rust/crates/lk-agents-abi/src/logging.rs) | Rust source | W01 |
| [src/media/codecs.rs](../../rust/crates/lk-agents-abi/src/media/codecs.rs) | Rust source | W01 |
| [src/media/emitter.rs](../../rust/crates/lk-agents-abi/src/media/emitter.rs) | Rust source | W01 |
| [src/media/images.rs](../../rust/crates/lk-agents-abi/src/media/images.rs) | Rust source | W01 |
| [src/media.rs](../../rust/crates/lk-agents-abi/src/media.rs) | Rust source | W01 |
| [src/metrics/base.rs](../../rust/crates/lk-agents-abi/src/metrics/base.rs) | Rust source | W01 |
| [src/metrics/mod.rs](../../rust/crates/lk-agents-abi/src/metrics/mod.rs) | Rust source | W01 |
| [src/metrics/testkit.rs](../../rust/crates/lk-agents-abi/src/metrics/testkit.rs) | Rust source | W01 |
| [src/recorder.rs](../../rust/crates/lk-agents-abi/src/recorder.rs) | Rust source | W01 |
| [src/resources.rs](../../rust/crates/lk-agents-abi/src/resources.rs) | Rust source | W01 |
| [src/storage/context.rs](../../rust/crates/lk-agents-abi/src/storage/context.rs) | Rust source | W01 |
| [src/storage/handles.rs](../../rust/crates/lk-agents-abi/src/storage/handles.rs) | Rust source | W01 |
| [src/storage/journal.rs](../../rust/crates/lk-agents-abi/src/storage/journal.rs) | Rust source | W01 |
| [src/storage/mod.rs](../../rust/crates/lk-agents-abi/src/storage/mod.rs) | Rust source | W01 |
| [src/storage/rows.rs](../../rust/crates/lk-agents-abi/src/storage/rows.rs) | Rust source | W01 |
| [src/storage/thread.rs](../../rust/crates/lk-agents-abi/src/storage/thread.rs) | Rust source | W01 |
| [src/storage/voice.rs](../../rust/crates/lk-agents-abi/src/storage/voice.rs) | Rust source | W01 |
| [src/stt/events.rs](../../rust/crates/lk-agents-abi/src/stt/events.rs) | Rust source | W01 |
| [src/stt/fallback.rs](../../rust/crates/lk-agents-abi/src/stt/fallback.rs) | Rust source | W01 |
| [src/stt/inference.rs](../../rust/crates/lk-agents-abi/src/stt/inference.rs) | Rust source | W01 |
| [src/stt/mod.rs](../../rust/crates/lk-agents-abi/src/stt/mod.rs) | Rust source | W01 |
| [src/stt/stream_adapter.rs](../../rust/crates/lk-agents-abi/src/stt/stream_adapter.rs) | Rust source | W01 |
| [src/stt/stt.py](../../rust/crates/lk-agents-abi/src/stt/stt.py) | Python host/test/tool source | W01 |
| [src/stt/stt.rs](../../rust/crates/lk-agents-abi/src/stt/stt.rs) | Rust source | W01 |
| [src/telemetry/attributes.rs](../../rust/crates/lk-agents-abi/src/telemetry/attributes.rs) | Rust source | W01 |
| [src/telemetry/mod.rs](../../rust/crates/lk-agents-abi/src/telemetry/mod.rs) | Rust source | W01 |
| [src/telemetry/observation.rs](../../rust/crates/lk-agents-abi/src/telemetry/observation.rs) | Rust source | W01 |
| [src/telemetry/policy.rs](../../rust/crates/lk-agents-abi/src/telemetry/policy.rs) | Rust source | W01 |
| [src/telemetry/provider.rs](../../rust/crates/lk-agents-abi/src/telemetry/provider.rs) | Rust source | W01 |
| [src/telemetry/report/projection.rs](../../rust/crates/lk-agents-abi/src/telemetry/report/projection.rs) | Rust source | W01 |
| [src/telemetry/report.rs](../../rust/crates/lk-agents-abi/src/telemetry/report.rs) | Rust source | W01 |
| [src/telemetry/transport.rs](../../rust/crates/lk-agents-abi/src/telemetry/transport.rs) | Rust source | W01 |
| [src/telemetry/upload.rs](../../rust/crates/lk-agents-abi/src/telemetry/upload.rs) | Rust source | W01 |
| [src/testkit.rs](../../rust/crates/lk-agents-abi/src/testkit.rs) | Rust source | W01 |
| [src/text_session.rs](../../rust/crates/lk-agents-abi/src/text_session.rs) | Rust source | W01 |
| [src/tokenize/bindings.rs](../../rust/crates/lk-agents-abi/src/tokenize/bindings.rs) | Rust source | W01 |
| [src/tokenize/language/code.rs](../../rust/crates/lk-agents-abi/src/tokenize/language/code.rs) | Rust source | W01 |
| [src/tokenize/language/data.rs](../../rust/crates/lk-agents-abi/src/tokenize/language/data.rs) | Rust source | W01 |
| [src/tokenize/language/eot_thresholds.rs](../../rust/crates/lk-agents-abi/src/tokenize/language/eot_thresholds.rs) | Rust source | W01 |
| [src/tokenize/language/mod.rs](../../rust/crates/lk-agents-abi/src/tokenize/language/mod.rs) | Rust source | W01 |
| [src/tokenize/mod.rs](../../rust/crates/lk-agents-abi/src/tokenize/mod.rs) | Rust source | W01 |
| [src/tokenize/search.rs](../../rust/crates/lk-agents-abi/src/tokenize/search.rs) | Rust source | W01 |
| [src/tts/markup/bindings.rs](../../rust/crates/lk-agents-abi/src/tts/markup/bindings.rs) | Rust source | W01 |
| [src/tts/markup/expr.rs](../../rust/crates/lk-agents-abi/src/tts/markup/expr.rs) | Rust source | W01 |
| [src/tts/markup/mod.rs](../../rust/crates/lk-agents-abi/src/tts/markup/mod.rs) | Rust source | W01 |
| [src/tts/markup/mood.rs](../../rust/crates/lk-agents-abi/src/tts/markup/mood.rs) | Rust source | W01 |
| [src/tts/markup/provider.rs](../../rust/crates/lk-agents-abi/src/tts/markup/provider.rs) | Rust source | W01 |
| [src/tts/markup/steering.rs](../../rust/crates/lk-agents-abi/src/tts/markup/steering.rs) | Rust source | W01 |
| [src/tts/mod.rs](../../rust/crates/lk-agents-abi/src/tts/mod.rs) | Rust source | W01 |
| [src/tts/stream_adapter.rs](../../rust/crates/lk-agents-abi/src/tts/stream_adapter.rs) | Rust source | W01 |
| [src/tts/stream_pacer.rs](../../rust/crates/lk-agents-abi/src/tts/stream_pacer.rs) | Rust source | W01 |
| [src/tts/streams.rs](../../rust/crates/lk-agents-abi/src/tts/streams.rs) | Rust source | W01 |
| [src/tts/transcript.rs](../../rust/crates/lk-agents-abi/src/tts/transcript.rs) | Rust source | W01 |
| [src/tts/tts.py](../../rust/crates/lk-agents-abi/src/tts/tts.py) | Python host/test/tool source | W01 |
| [src/tts/tts.rs](../../rust/crates/lk-agents-abi/src/tts/tts.rs) | Rust source | W01 |
| [src/turn/detector.rs](../../rust/crates/lk-agents-abi/src/turn/detector.rs) | Rust source | W01 |
| [src/turn/eot.rs](../../rust/crates/lk-agents-abi/src/turn/eot.rs) | Rust source | W01 |
| [src/turn/events.rs](../../rust/crates/lk-agents-abi/src/turn/events.rs) | Rust source | W01 |
| [src/turn/inference.rs](../../rust/crates/lk-agents-abi/src/turn/inference.rs) | Rust source | W01 |
| [src/turn/interruption.rs](../../rust/crates/lk-agents-abi/src/turn/interruption.rs) | Rust source | W01 |
| [src/turn/keyterms.rs](../../rust/crates/lk-agents-abi/src/turn/keyterms.rs) | Rust source | W01 |
| [src/turn/mod.rs](../../rust/crates/lk-agents-abi/src/turn/mod.rs) | Rust source | W01 |
| [src/types/connect.rs](../../rust/crates/lk-agents-abi/src/types/connect.rs) | Rust source | W01 |
| [src/types/identity.rs](../../rust/crates/lk-agents-abi/src/types/identity.rs) | Rust source | W01 |
| [src/types/mod.rs](../../rust/crates/lk-agents-abi/src/types/mod.rs) | Rust source | W01 |
| [src/types/provider.rs](../../rust/crates/lk-agents-abi/src/types/provider.rs) | Rust source | W01 |
| [src/types/retry.rs](../../rust/crates/lk-agents-abi/src/types/retry.rs) | Rust source | W01 |
| [src/usage.rs](../../rust/crates/lk-agents-abi/src/usage.rs) | Rust source | W01 |
| [src/utils/audio.rs](../../rust/crates/lk-agents-abi/src/utils/audio.rs) | Rust source | W01 |
| [src/utils/byte_stream.rs](../../rust/crates/lk-agents-abi/src/utils/byte_stream.rs) | Rust source | W01 |
| [src/utils/clock.rs](../../rust/crates/lk-agents-abi/src/utils/clock.rs) | Rust source | W01 |
| [src/utils/connection_pool.rs](../../rust/crates/lk-agents-abi/src/utils/connection_pool.rs) | Rust source | W01 |
| [src/utils/dates.rs](../../rust/crates/lk-agents-abi/src/utils/dates.rs) | Rust source | W01 |
| [src/utils/filters.rs](../../rust/crates/lk-agents-abi/src/utils/filters.rs) | Rust source | W01 |
| [src/utils/host_ref.rs](../../rust/crates/lk-agents-abi/src/utils/host_ref.rs) | Rust source | W01 |
| [src/utils/maybe.rs](../../rust/crates/lk-agents-abi/src/utils/maybe.rs) | Rust source | W01 |
| [src/utils/mod.rs](../../rust/crates/lk-agents-abi/src/utils/mod.rs) | Rust source | W01 |
| [src/utils/resample.rs](../../rust/crates/lk-agents-abi/src/utils/resample.rs) | Rust source | W01 |
| [src/utils/spawner.rs](../../rust/crates/lk-agents-abi/src/utils/spawner.rs) | Rust source | W27 |
| [src/utils/stream.rs](../../rust/crates/lk-agents-abi/src/utils/stream.rs) | Rust source | W01 |
| [src/utils/strings.rs](../../rust/crates/lk-agents-abi/src/utils/strings.rs) | Rust source | W01 |
| [src/utils/timer.rs](../../rust/crates/lk-agents-abi/src/utils/timer.rs) | Rust source | W01 |
| [src/utils/video.rs](../../rust/crates/lk-agents-abi/src/utils/video.rs) | Rust source | W01 |
| [src/utils/vocabulary.rs](../../rust/crates/lk-agents-abi/src/utils/vocabulary.rs) | Rust source | W01 |
| [src/vad/events.rs](../../rust/crates/lk-agents-abi/src/vad/events.rs) | Rust source | W01 |
| [src/vad/inference.rs](../../rust/crates/lk-agents-abi/src/vad/inference.rs) | Rust source | W01 |
| [src/vad/mod.rs](../../rust/crates/lk-agents-abi/src/vad/mod.rs) | Rust source | W01 |
| [src/vad/vad.py](../../rust/crates/lk-agents-abi/src/vad/vad.py) | Python host/test/tool source | W01 |
| [src/vad/vad.rs](../../rust/crates/lk-agents-abi/src/vad/vad.rs) | Rust source | W01 |
| [src/voice/amd.rs](../../rust/crates/lk-agents-abi/src/voice/amd.rs) | Rust source | W01 |
| [src/voice/avatar.rs](../../rust/crates/lk-agents-abi/src/voice/avatar.rs) | Rust source | W01 |
| [src/voice/avatar_queue.rs](../../rust/crates/lk-agents-abi/src/voice/avatar_queue.rs) | Rust source | W01 |
| [src/voice/background_audio.rs](../../rust/crates/lk-agents-abi/src/voice/background_audio.rs) | Rust source | W01 |
| [src/voice/console.rs](../../rust/crates/lk-agents-abi/src/voice/console.rs) | Rust source | W01 |
| [src/voice/events.rs](../../rust/crates/lk-agents-abi/src/voice/events.rs) | Rust source | W01 |
| [src/voice/io.rs](../../rust/crates/lk-agents-abi/src/voice/io.rs) | Rust source | W01 |
| [src/voice/mod.rs](../../rust/crates/lk-agents-abi/src/voice/mod.rs) | Rust source | W01 |
| [src/voice/nodes.rs](../../rust/crates/lk-agents-abi/src/voice/nodes.rs) | Rust source | W01 |
| [src/voice/room_io.rs](../../rust/crates/lk-agents-abi/src/voice/room_io.rs) | Rust source | W01 |
| [src/voice/room_owner.rs](../../rust/crates/lk-agents-abi/src/voice/room_owner.rs) | Rust source | W01 |
| [src/voice/rtc/capabilities.rs](../../rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs) | Rust source | W01 |
| [src/voice/rtc/error.rs](../../rust/crates/lk-agents-abi/src/voice/rtc/error.rs) | Rust source | W01 |
| [src/voice/rtc/events.rs](../../rust/crates/lk-agents-abi/src/voice/rtc/events.rs) | Rust source | W01 |
| [src/voice/rtc/media.rs](../../rust/crates/lk-agents-abi/src/voice/rtc/media.rs) | Rust source | W01 |
| [src/voice/rtc/playout.rs](../../rust/crates/lk-agents-abi/src/voice/rtc/playout.rs) | Rust source | W01 |
| [src/voice/rtc/ports.rs](../../rust/crates/lk-agents-abi/src/voice/rtc/ports.rs) | Rust source | W01 |
| [src/voice/rtc.rs](../../rust/crates/lk-agents-abi/src/voice/rtc.rs) | Rust source | W01 |
| [src/voice/run_view.rs](../../rust/crates/lk-agents-abi/src/voice/run_view.rs) | Rust source | W01 |
| [src/voice/session.rs](../../rust/crates/lk-agents-abi/src/voice/session.rs) | Rust source | W01 |
| [src/voice/session_effects.rs](../../rust/crates/lk-agents-abi/src/voice/session_effects.rs) | Rust source | W01 |
| [src/voice/session_options.rs](../../rust/crates/lk-agents-abi/src/voice/session_options.rs) | Rust source | W01 |
| [src/voice/session_owner.rs](../../rust/crates/lk-agents-abi/src/voice/session_owner.rs) | Rust source | W01 |
| [src/voice/session_storage.rs](../../rust/crates/lk-agents-abi/src/voice/session_storage.rs) | Rust source | W01 |
| [src/voice/speech.rs](../../rust/crates/lk-agents-abi/src/voice/speech.rs) | Rust source | W01 |
| [src/voice/speech_handle.rs](../../rust/crates/lk-agents-abi/src/voice/speech_handle.rs) | Rust source | W01 |
| [src/voice/workflows/dtmf.rs](../../rust/crates/lk-agents-abi/src/voice/workflows/dtmf.rs) | Rust source | W01 |
| [src/voice/workflows/prompts.rs](../../rust/crates/lk-agents-abi/src/voice/workflows/prompts.rs) | Rust source | W01 |
| [src/voice/workflows/warm_transfer.rs](../../rust/crates/lk-agents-abi/src/voice/workflows/warm_transfer.rs) | Rust source | W01 |
| [src/voice/workflows.rs](../../rust/crates/lk-agents-abi/src/voice/workflows.rs) | Rust source | W01 |
| [src/worker/http/reset_codes.rs](../../rust/crates/lk-agents-abi/src/worker/http/reset_codes.rs) | Rust source | W01 |
| [src/worker/http.rs](../../rust/crates/lk-agents-abi/src/worker/http.rs) | Rust source | W01 |
| [src/worker/ipc.rs](../../rust/crates/lk-agents-abi/src/worker/ipc.rs) | Rust source | W01 |
| [src/worker/ipc_host.rs](../../rust/crates/lk-agents-abi/src/worker/ipc_host.rs) | Rust source | W01 |
| [src/worker/job.rs](../../rust/crates/lk-agents-abi/src/worker/job.rs) | Rust source | W01 |
| [src/worker/mod.rs](../../rust/crates/lk-agents-abi/src/worker/mod.rs) | Rust source | W01 |
| [src/worker/remote_session.rs](../../rust/crates/lk-agents-abi/src/worker/remote_session.rs) | Rust source | W01 |
| [src/worker/server_host.rs](../../rust/crates/lk-agents-abi/src/worker/server_host.rs) | Rust source | W01 |
| [src/worker/worker.rs](../../rust/crates/lk-agents-abi/src/worker/worker.rs) | Rust source | W01 |

</details>

<details>
<summary>lk-agents-bindgen — 6 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/ai_slop/lk-agents-bindgen/Cargo.toml) | package manifest | W39 |
| [src/bindgen_tests.rs](../../rust/ai_slop/lk-agents-bindgen/src/bindgen_tests.rs) | test/support | W39 |
| [src/error.rs](../../rust/ai_slop/lk-agents-bindgen/src/error.rs) | Rust source | W39 |
| [src/lib.rs](../../rust/ai_slop/lk-agents-bindgen/src/lib.rs) | Rust source | W39 |
| [src/meta.rs](../../rust/ai_slop/lk-agents-bindgen/src/meta.rs) | Rust source | W39 |
| [src/record.rs](../../rust/ai_slop/lk-agents-bindgen/src/record.rs) | Rust source | W39 |

</details>

<details>
<summary>lk-agents-cli — 8 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-cli/Cargo.toml) | package manifest | W37 |
| [src/args.rs](../../rust/crates/lk-agents-cli/src/args.rs) | Rust source | W37 |
| [src/lib.rs](../../rust/crates/lk-agents-cli/src/lib.rs) | Rust source | W37 |
| [src/logging.rs](../../rust/crates/lk-agents-cli/src/logging.rs) | Rust source | W37 |
| [src/remote_cli.rs](../../rust/crates/lk-agents-cli/src/remote_cli.rs) | Rust source | W37 |
| [src/shutdown.rs](../../rust/crates/lk-agents-cli/src/shutdown.rs) | Rust source | W37 |
| [src/shutdown_tests.rs](../../rust/crates/lk-agents-cli/src/shutdown_tests.rs) | explicit test target | W37 |
| [src/signals.rs](../../rust/crates/lk-agents-cli/src/signals.rs) | Rust source | W37 |

</details>

<details>
<summary>lk-agents-codegen — 40 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/ai_slop/lk-agents-codegen/Cargo.toml) | package manifest | W39 |
| [export-placement-baseline.json](../../rust/ai_slop/lk-agents-codegen/export-placement-baseline.json) | configuration/asset | W39 |
| [src/accessors.rs](../../rust/ai_slop/lk-agents-codegen/src/accessors.rs) | Rust source | W39 |
| [src/adapter.rs](../../rust/ai_slop/lk-agents-codegen/src/adapter.rs) | Rust source | W39 |
| [src/architecture/coverage.rs](../../rust/ai_slop/lk-agents-codegen/src/architecture/coverage.rs) | Rust source | W41 |
| [src/architecture/host_members.rs](../../rust/ai_slop/lk-agents-codegen/src/architecture/host_members.rs) | Rust source | W41 |
| [src/architecture/mod.rs](../../rust/ai_slop/lk-agents-codegen/src/architecture/mod.rs) | Rust source | W41 |
| [src/bases.rs](../../rust/ai_slop/lk-agents-codegen/src/bases.rs) | Rust source | W39 |
| [src/class_stub.rs](../../rust/ai_slop/lk-agents-codegen/src/class_stub.rs) | Rust source | W39 |
| [src/codegen_tests.rs](../../rust/ai_slop/lk-agents-codegen/src/codegen_tests.rs) | test/support | W39 |
| [src/decl/mod.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/mod.rs) | Rust source | W39 |
| [src/decl/resolve/class.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/resolve/class.rs) | Rust source | W39 |
| [src/decl/resolve/exposed.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/resolve/exposed.rs) | Rust source | W39 |
| [src/decl/resolve/native.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/resolve/native.rs) | Rust source | W39 |
| [src/decl/resolve/proxy.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/resolve/proxy.rs) | Rust source | W39 |
| [src/decl/resolve/record.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/resolve/record.rs) | Rust source | W39 |
| [src/decl/resolve/stream.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/resolve/stream.rs) | Rust source | W39 |
| [src/decl/resolve.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/resolve.rs) | Rust source | W39 |
| [src/decl/tree.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/tree.rs) | Rust source | W39 |
| [src/decl/vocabulary.rs](../../rust/ai_slop/lk-agents-codegen/src/decl/vocabulary.rs) | Rust source | W39 |
| [src/decl_callables_tests.rs](../../rust/ai_slop/lk-agents-codegen/src/decl_callables_tests.rs) | test/support | W39 |
| [src/decl_models_tests.rs](../../rust/ai_slop/lk-agents-codegen/src/decl_models_tests.rs) | test/support | W39 |
| [src/decl_tests.rs](../../rust/ai_slop/lk-agents-codegen/src/decl_tests.rs) | test/support | W39 |
| [src/exceptions.rs](../../rust/ai_slop/lk-agents-codegen/src/exceptions.rs) | Rust source | W39 |
| [src/facade.rs](../../rust/ai_slop/lk-agents-codegen/src/facade.rs) | Rust source | W39 |
| [src/forward.rs](../../rust/ai_slop/lk-agents-codegen/src/forward.rs) | Rust source | W39 |
| [src/implementations.rs](../../rust/ai_slop/lk-agents-codegen/src/implementations.rs) | Rust source | W39 |
| [src/imports.rs](../../rust/ai_slop/lk-agents-codegen/src/imports.rs) | Rust source | W39 |
| [src/main.rs](../../rust/ai_slop/lk-agents-codegen/src/main.rs) | Rust source | W39 |
| [src/native_enums.rs](../../rust/ai_slop/lk-agents-codegen/src/native_enums.rs) | Rust source | W39 |
| [src/parse.rs](../../rust/ai_slop/lk-agents-codegen/src/parse.rs) | Rust source | W39 |
| [src/placement.rs](../../rust/ai_slop/lk-agents-codegen/src/placement.rs) | Rust source | W39 |
| [src/proxy.rs](../../rust/ai_slop/lk-agents-codegen/src/proxy.rs) | Rust source | W39 |
| [src/python.rs](../../rust/ai_slop/lk-agents-codegen/src/python.rs) | Rust source | W39 |
| [src/records/boundary.rs](../../rust/ai_slop/lk-agents-codegen/src/records/boundary.rs) | Rust source | W39 |
| [src/records/emit.rs](../../rust/ai_slop/lk-agents-codegen/src/records/emit.rs) | Rust source | W39 |
| [src/records/models.rs](../../rust/ai_slop/lk-agents-codegen/src/records/models.rs) | Rust source | W39 |
| [src/records.rs](../../rust/ai_slop/lk-agents-codegen/src/records.rs) | Rust source | W39 |
| [src/shape_error.rs](../../rust/ai_slop/lk-agents-codegen/src/shape_error.rs) | Rust source | W39 |
| [src/surface.rs](../../rust/ai_slop/lk-agents-codegen/src/surface.rs) | Rust source | W39 |

</details>

<details>
<summary>lk-agents-core — 126 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-core/Cargo.toml) | package manifest | W34 |
| [fixtures/json_repair/LICENSE](../../rust/crates/lk-agents-core/fixtures/json_repair/LICENSE) | fixture/counterexample | W34 |
| [fixtures/json_repair/README.md](../../rust/crates/lk-agents-core/fixtures/json_repair/README.md) | fixture/counterexample | W34 |
| [fixtures/json_repair/cases.json](../../rust/crates/lk-agents-core/fixtures/json_repair/cases.json) | fixture/counterexample | W34 |
| [src/a2a/attachments.rs](../../rust/crates/lk-agents-core/src/a2a/attachments.rs) | Rust source | W35 |
| [src/a2a/attachments_tests.rs](../../rust/crates/lk-agents-core/src/a2a/attachments_tests.rs) | explicit test target | W35 |
| [src/a2a/auth.rs](../../rust/crates/lk-agents-core/src/a2a/auth.rs) | Rust source | W35 |
| [src/a2a/caller_history.rs](../../rust/crates/lk-agents-core/src/a2a/caller_history.rs) | Rust source | W35 |
| [src/a2a/caller_history_tests.rs](../../rust/crates/lk-agents-core/src/a2a/caller_history_tests.rs) | test/support | W35 |
| [src/a2a/card.rs](../../rust/crates/lk-agents-core/src/a2a/card.rs) | Rust source | W35 |
| [src/a2a/channels_tests.rs](../../rust/crates/lk-agents-core/src/a2a/channels_tests.rs) | explicit test target | W35 |
| [src/a2a/client/messages.rs](../../rust/crates/lk-agents-core/src/a2a/client/messages.rs) | Rust source | W35 |
| [src/a2a/client/tool.rs](../../rust/crates/lk-agents-core/src/a2a/client/tool.rs) | Rust source | W35 |
| [src/a2a/client.rs](../../rust/crates/lk-agents-core/src/a2a/client.rs) | Rust source | W35 |
| [src/a2a/client_tests.rs](../../rust/crates/lk-agents-core/src/a2a/client_tests.rs) | test/support | W35 |
| [src/a2a/delegation/retained.rs](../../rust/crates/lk-agents-core/src/a2a/delegation/retained.rs) | Rust source | W35 |
| [src/a2a/delegation.rs](../../rust/crates/lk-agents-core/src/a2a/delegation.rs) | Rust source | W35 |
| [src/a2a/delegation_tests.rs](../../rust/crates/lk-agents-core/src/a2a/delegation_tests.rs) | test/support | W35 |
| [src/a2a/handler.rs](../../rust/crates/lk-agents-core/src/a2a/handler.rs) | Rust source | W35 |
| [src/a2a/host.rs](../../rust/crates/lk-agents-core/src/a2a/host.rs) | Rust source | W35 |
| [src/a2a/metadata_tests.rs](../../rust/crates/lk-agents-core/src/a2a/metadata_tests.rs) | explicit test target | W35 |
| [src/a2a/outcome.rs](../../rust/crates/lk-agents-core/src/a2a/outcome.rs) | Rust source | W35 |
| [src/a2a/policy.rs](../../rust/crates/lk-agents-core/src/a2a/policy.rs) | Rust source | W35 |
| [src/a2a/rpc.rs](../../rust/crates/lk-agents-core/src/a2a/rpc.rs) | Rust source | W35 |
| [src/a2a/task/turn.rs](../../rust/crates/lk-agents-core/src/a2a/task/turn.rs) | Rust source | W35 |
| [src/a2a/task.rs](../../rust/crates/lk-agents-core/src/a2a/task.rs) | Rust source | W35 |
| [src/a2a/task_tests.rs](../../rust/crates/lk-agents-core/src/a2a/task_tests.rs) | test/support | W35 |
| [src/a2a/wire.rs](../../rust/crates/lk-agents-core/src/a2a/wire.rs) | Rust source | W35 |
| [src/a2a.rs](../../rust/crates/lk-agents-core/src/a2a.rs) | Rust source | W35 |
| [src/agent.rs](../../rust/crates/lk-agents-core/src/agent.rs) | Rust source | W34 |
| [src/agent_authoring.rs](../../rust/crates/lk-agents-core/src/agent_authoring.rs) | Rust source | W34 |
| [src/agent_run/cancellation.rs](../../rust/crates/lk-agents-core/src/agent_run/cancellation.rs) | Rust source | W34 |
| [src/agent_run/checkpoint.rs](../../rust/crates/lk-agents-core/src/agent_run/checkpoint.rs) | Rust source | W34 |
| [src/agent_run/coordinated.rs](../../rust/crates/lk-agents-core/src/agent_run/coordinated.rs) | Rust source | W34 |
| [src/agent_run/execution.rs](../../rust/crates/lk-agents-core/src/agent_run/execution.rs) | Rust source | W34 |
| [src/agent_run/mod.rs](../../rust/crates/lk-agents-core/src/agent_run/mod.rs) | Rust source | W34 |
| [src/agent_run/model_step.rs](../../rust/crates/lk-agents-core/src/agent_run/model_step.rs) | Rust source | W34 |
| [src/agent_run/result.rs](../../rust/crates/lk-agents-core/src/agent_run/result.rs) | Rust source | W34 |
| [src/agent_run/step_outcome.rs](../../rust/crates/lk-agents-core/src/agent_run/step_outcome.rs) | Rust source | W34 |
| [src/agent_run/text_session.rs](../../rust/crates/lk-agents-core/src/agent_run/text_session.rs) | Rust source | W34 |
| [src/agent_run_flow_tests.rs](../../rust/crates/lk-agents-core/src/agent_run_flow_tests.rs) | test/support | W34 |
| [src/agent_run_lifecycle_tests.rs](../../rust/crates/lk-agents-core/src/agent_run_lifecycle_tests.rs) | test/support | W34 |
| [src/agent_run_tests.rs](../../rust/crates/lk-agents-core/src/agent_run_tests.rs) | test/support | W34 |
| [src/approval.rs](../../rust/crates/lk-agents-core/src/approval.rs) | Rust source | W20 |
| [src/assertions.rs](../../rust/crates/lk-agents-core/src/assertions.rs) | Rust source | W34 |
| [src/assertions_tests.rs](../../rust/crates/lk-agents-core/src/assertions_tests.rs) | test/support | W34 |
| [src/attempts.rs](../../rust/crates/lk-agents-core/src/attempts.rs) | Rust source | W31 |
| [src/lib.rs](../../rust/crates/lk-agents-core/src/lib.rs) | Rust source | W34 |
| [src/llm/chat_dict.rs](../../rust/crates/lk-agents-core/src/llm/chat_dict.rs) | Rust source | W02 |
| [src/llm/chat_plans.rs](../../rust/crates/lk-agents-core/src/llm/chat_plans.rs) | Rust source | W02 |
| [src/llm/ctx_diff/incremental.rs](../../rust/crates/lk-agents-core/src/llm/ctx_diff/incremental.rs) | Rust source | W33 |
| [src/llm/ctx_diff.rs](../../rust/crates/lk-agents-core/src/llm/ctx_diff.rs) | Rust source | W33 |
| [src/llm/json_repair/mod.rs](../../rust/crates/lk-agents-core/src/llm/json_repair/mod.rs) | Rust source | W33 |
| [src/llm/json_repair/string.rs](../../rust/crates/lk-agents-core/src/llm/json_repair/string.rs) | Rust source | W33 |
| [src/llm/mod.rs](../../rust/crates/lk-agents-core/src/llm/mod.rs) | Rust source | W33 |
| [src/llm/provider_format/anthropic.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/anthropic.rs) | Rust source | W33 |
| [src/llm/provider_format/aws.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/aws.rs) | Rust source | W33 |
| [src/llm/provider_format/google.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/google.rs) | Rust source | W33 |
| [src/llm/provider_format/group.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/group.rs) | Rust source | W33 |
| [src/llm/provider_format/host.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/host.rs) | Rust source | W33 |
| [src/llm/provider_format/image.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/image.rs) | Rust source | W33 |
| [src/llm/provider_format/mistralai.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/mistralai.rs) | Rust source | W33 |
| [src/llm/provider_format/mod.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/mod.rs) | Rust source | W33 |
| [src/llm/provider_format/openai.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/openai.rs) | Rust source | W33 |
| [src/llm/provider_format/strict.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/strict.rs) | Rust source | W33 |
| [src/llm/provider_format/testkit.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/testkit.rs) | Rust source | W33 |
| [src/llm/provider_format/tools.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/tools.rs) | Rust source | W33 |
| [src/llm/provider_format/turns.rs](../../rust/crates/lk-agents-core/src/llm/provider_format/turns.rs) | Rust source | W33 |
| [src/llm/schema/defaults.rs](../../rust/crates/lk-agents-core/src/llm/schema/defaults.rs) | Rust source | W33 |
| [src/llm/schema/mod.rs](../../rust/crates/lk-agents-core/src/llm/schema/mod.rs) | Rust source | W33 |
| [src/llm/schema/openai.rs](../../rust/crates/lk-agents-core/src/llm/schema/openai.rs) | Rust source | W33 |
| [src/llm/tool_arguments.rs](../../rust/crates/lk-agents-core/src/llm/tool_arguments.rs) | Rust source | W33 |
| [src/mcp/http.rs](../../rust/crates/lk-agents-core/src/mcp/http.rs) | Rust source | W35 |
| [src/mcp.rs](../../rust/crates/lk-agents-core/src/mcp.rs) | Rust source | W35 |
| [src/request.rs](../../rust/crates/lk-agents-core/src/request.rs) | Rust source | W34 |
| [src/resource_lifecycle.rs](../../rust/crates/lk-agents-core/src/resource_lifecycle.rs) | Rust source | W27 |
| [src/run_stream/mod.rs](../../rust/crates/lk-agents-core/src/run_stream/mod.rs) | Rust source | W34 |
| [src/run_stream/projection.rs](../../rust/crates/lk-agents-core/src/run_stream/projection.rs) | Rust source | W34 |
| [src/telemetry/gen_ai.rs](../../rust/crates/lk-agents-core/src/telemetry/gen_ai.rs) | Rust source | W38 |
| [src/telemetry/metrics.rs](../../rust/crates/lk-agents-core/src/telemetry/metrics.rs) | Rust source | W38 |
| [src/telemetry/mod.rs](../../rust/crates/lk-agents-core/src/telemetry/mod.rs) | Rust source | W38 |
| [src/telemetry/policy.rs](../../rust/crates/lk-agents-core/src/telemetry/policy.rs) | Rust source | W38 |
| [src/telemetry/provider.rs](../../rust/crates/lk-agents-core/src/telemetry/provider.rs) | Rust source | W38 |
| [src/telemetry/records/metrics_log.rs](../../rust/crates/lk-agents-core/src/telemetry/records/metrics_log.rs) | Rust source | W38 |
| [src/telemetry/records/otel.rs](../../rust/crates/lk-agents-core/src/telemetry/records/otel.rs) | Rust source | W38 |
| [src/telemetry/records.rs](../../rust/crates/lk-agents-core/src/telemetry/records.rs) | Rust source | W38 |
| [src/telemetry/report/records.rs](../../rust/crates/lk-agents-core/src/telemetry/report/records.rs) | Rust source | W38 |
| [src/telemetry/report/records_tests.rs](../../rust/crates/lk-agents-core/src/telemetry/report/records_tests.rs) | test/support | W38 |
| [src/telemetry/report.rs](../../rust/crates/lk-agents-core/src/telemetry/report.rs) | Rust source | W38 |
| [src/telemetry/transport.rs](../../rust/crates/lk-agents-core/src/telemetry/transport.rs) | Rust source | W38 |
| [src/text.rs](../../rust/crates/lk-agents-core/src/text.rs) | Rust source | W33 |
| [src/text_session/continuation.rs](../../rust/crates/lk-agents-core/src/text_session/continuation.rs) | Rust source | W34 |
| [src/text_session/history.rs](../../rust/crates/lk-agents-core/src/text_session/history.rs) | Rust source | W34 |
| [src/text_session/mod.rs](../../rust/crates/lk-agents-core/src/text_session/mod.rs) | Rust source | W34 |
| [src/text_session/routing.rs](../../rust/crates/lk-agents-core/src/text_session/routing.rs) | Rust source | W34 |
| [src/text_session_tests.rs](../../rust/crates/lk-agents-core/src/text_session_tests.rs) | test/support | W34 |
| [src/tokenize/host.rs](../../rust/crates/lk-agents-core/src/tokenize/host.rs) | Rust source | W33 |
| [src/tokenize/mod.rs](../../rust/crates/lk-agents-core/src/tokenize/mod.rs) | Rust source | W33 |
| [src/tokenize/replace.rs](../../rust/crates/lk-agents-core/src/tokenize/replace.rs) | Rust source | W33 |
| [src/tokenize/sentence.rs](../../rust/crates/lk-agents-core/src/tokenize/sentence.rs) | Rust source | W33 |
| [src/tokenize/token_buffer.rs](../../rust/crates/lk-agents-core/src/tokenize/token_buffer.rs) | Rust source | W33 |
| [src/tokenize/word.rs](../../rust/crates/lk-agents-core/src/tokenize/word.rs) | Rust source | W33 |
| [src/tokenize/xml.rs](../../rust/crates/lk-agents-core/src/tokenize/xml.rs) | Rust source | W33 |
| [src/tool_calls.rs](../../rust/crates/lk-agents-core/src/tool_calls.rs) | Rust source | W20 |
| [src/tool_outcome.rs](../../rust/crates/lk-agents-core/src/tool_outcome.rs) | Rust source | W20 |
| [src/tool_search.rs](../../rust/crates/lk-agents-core/src/tool_search.rs) | Rust source | W20 |
| [src/toolsets.rs](../../rust/crates/lk-agents-core/src/toolsets.rs) | Rust source | W27 |
| [src/tts/markup/chat.rs](../../rust/crates/lk-agents-core/src/tts/markup/chat.rs) | Rust source | W33 |
| [src/tts/markup/expr.rs](../../rust/crates/lk-agents-core/src/tts/markup/expr.rs) | Rust source | W33 |
| [src/tts/markup/host.rs](../../rust/crates/lk-agents-core/src/tts/markup/host.rs) | Rust source | W33 |
| [src/tts/markup/instructions.rs](../../rust/crates/lk-agents-core/src/tts/markup/instructions.rs) | Rust source | W33 |
| [src/tts/markup/mod.rs](../../rust/crates/lk-agents-core/src/tts/markup/mod.rs) | Rust source | W33 |
| [src/tts/markup/mood.rs](../../rust/crates/lk-agents-core/src/tts/markup/mood.rs) | Rust source | W33 |
| [src/tts/markup/mood_data.rs](../../rust/crates/lk-agents-core/src/tts/markup/mood_data.rs) | Rust source | W33 |
| [src/tts/markup/prompts.rs](../../rust/crates/lk-agents-core/src/tts/markup/prompts.rs) | Rust source | W33 |
| [src/tts/markup/provider.rs](../../rust/crates/lk-agents-core/src/tts/markup/provider.rs) | Rust source | W33 |
| [src/tts/markup/steering.rs](../../rust/crates/lk-agents-core/src/tts/markup/steering.rs) | Rust source | W33 |
| [src/tts/markup/strip.rs](../../rust/crates/lk-agents-core/src/tts/markup/strip.rs) | Rust source | W33 |
| [src/tts/markup/summarize.rs](../../rust/crates/lk-agents-core/src/tts/markup/summarize.rs) | Rust source | W33 |
| [src/tts/markup/testkit.rs](../../rust/crates/lk-agents-core/src/tts/markup/testkit.rs) | Rust source | W33 |
| [src/tts/markup/transcript.rs](../../rust/crates/lk-agents-core/src/tts/markup/transcript.rs) | Rust source | W33 |
| [src/tts/markup_tests.rs](../../rust/crates/lk-agents-core/src/tts/markup_tests.rs) | test/support | W33 |
| [src/tts/mod.rs](../../rust/crates/lk-agents-core/src/tts/mod.rs) | Rust source | W33 |
| [src/tts/transcript.rs](../../rust/crates/lk-agents-core/src/tts/transcript.rs) | Rust source | W33 |
| [src/usage.rs](../../rust/crates/lk-agents-core/src/usage.rs) | Rust source | W38 |
| [src/usage_tests.rs](../../rust/crates/lk-agents-core/src/usage_tests.rs) | test/support | W34 |

</details>

<details>
<summary>lk-agents-fuzz — 153 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [.gitignore](../../rust/crates/lk-agents-fuzz/.gitignore) | configuration/asset | W40 |
| [COVERAGE.md](../../rust/crates/lk-agents-fuzz/COVERAGE.md) | documentation | W40 |
| [Cargo.toml](../../rust/crates/lk-agents-fuzz/Cargo.toml) | package manifest | W40 |
| [bench/baseline.tsv](../../rust/crates/lk-agents-fuzz/bench/baseline.tsv) | configuration/asset | W40 |
| [bench/bugs/amd_tap_at_close.patch](../../rust/crates/lk-agents-fuzz/bench/bugs/amd_tap_at_close.patch) | configuration/asset | W40 |
| [bench/bugs/background_tool_updates.patch](../../rust/crates/lk-agents-fuzz/bench/bugs/background_tool_updates.patch) | configuration/asset | W40 |
| [bench/bugs/driver_say_tts.patch](../../rust/crates/lk-agents-fuzz/bench/bugs/driver_say_tts.patch) | configuration/asset | W40 |
| [bench/bugs/host_interrupt_races_end.patch](../../rust/crates/lk-agents-fuzz/bench/bugs/host_interrupt_races_end.patch) | configuration/asset | W40 |
| [bench/bugs/mic_at_close.patch](../../rust/crates/lk-agents-fuzz/bench/bugs/mic_at_close.patch) | configuration/asset | W40 |
| [bench/bugs/realtime_server_onset.patch](../../rust/crates/lk-agents-fuzz/bench/bugs/realtime_server_onset.patch) | configuration/asset | W40 |
| [bench/bugs/say_kind_after_close.patch](../../rust/crates/lk-agents-fuzz/bench/bugs/say_kind_after_close.patch) | configuration/asset | W40 |
| [bench/known_bugs.sh](../../rust/crates/lk-agents-fuzz/bench/known_bugs.sh) | gate/tool script | W40 |
| [build.rs](../../rust/crates/lk-agents-fuzz/build.rs) | build support | W40 |
| [calibration.json](../../rust/crates/lk-agents-fuzz/calibration.json) | configuration/asset | W40 |
| [launch-gate.sh](../../rust/crates/lk-agents-fuzz/launch-gate.sh) | gate/tool script | W40 |
| [src/a2a/database.rs](../../rust/crates/lk-agents-fuzz/src/a2a/database.rs) | Rust source | W40 |
| [src/a2a/handler.rs](../../rust/crates/lk-agents-fuzz/src/a2a/handler.rs) | Rust source | W40 |
| [src/a2a/mod.rs](../../rust/crates/lk-agents-fuzz/src/a2a/mod.rs) | Rust source | W40 |
| [src/a2a/rules.rs](../../rust/crates/lk-agents-fuzz/src/a2a/rules.rs) | Rust source | W40 |
| [src/a2a_survey_tests.rs](../../rust/crates/lk-agents-fuzz/src/a2a_survey_tests.rs) | explicit test target | W40 |
| [src/audio_tests.rs](../../rust/crates/lk-agents-fuzz/src/audio_tests.rs) | test/support | W40 |
| [src/authored.rs](../../rust/crates/lk-agents-fuzz/src/authored.rs) | Rust source | W40 |
| [src/avatar_tests.rs](../../rust/crates/lk-agents-fuzz/src/avatar_tests.rs) | test/support | W40 |
| [src/boundary.rs](../../rust/crates/lk-agents-fuzz/src/boundary.rs) | Rust source | W40 |
| [src/check.rs](../../rust/crates/lk-agents-fuzz/src/check.rs) | Rust source | W40 |
| [src/conversation_tests.rs](../../rust/crates/lk-agents-fuzz/src/conversation_tests.rs) | test/support | W40 |
| [src/drive/apply/wait.rs](../../rust/crates/lk-agents-fuzz/src/drive/apply/wait.rs) | Rust source | W40 |
| [src/drive/apply.rs](../../rust/crates/lk-agents-fuzz/src/drive/apply.rs) | Rust source | W40 |
| [src/drive/collect.rs](../../rust/crates/lk-agents-fuzz/src/drive/collect.rs) | Rust source | W40 |
| [src/drive/executor.rs](../../rust/crates/lk-agents-fuzz/src/drive/executor.rs) | Rust source | W40 |
| [src/drive/finish.rs](../../rust/crates/lk-agents-fuzz/src/drive/finish.rs) | Rust source | W40 |
| [src/drive/mod.rs](../../rust/crates/lk-agents-fuzz/src/drive/mod.rs) | Rust source | W40 |
| [src/drive/queue.rs](../../rust/crates/lk-agents-fuzz/src/drive/queue.rs) | Rust source | W40 |
| [src/drive/start.rs](../../rust/crates/lk-agents-fuzz/src/drive/start.rs) | Rust source | W40 |
| [src/fakes/avatar.rs](../../rust/crates/lk-agents-fuzz/src/fakes/avatar.rs) | Rust source | W40 |
| [src/fakes/console.rs](../../rust/crates/lk-agents-fuzz/src/fakes/console.rs) | Rust source | W40 |
| [src/fakes/eot.rs](../../rust/crates/lk-agents-fuzz/src/fakes/eot.rs) | Rust source | W40 |
| [src/fakes/interruption.rs](../../rust/crates/lk-agents-fuzz/src/fakes/interruption.rs) | Rust source | W40 |
| [src/fakes/llm.rs](../../rust/crates/lk-agents-fuzz/src/fakes/llm.rs) | Rust source | W40 |
| [src/fakes/mod.rs](../../rust/crates/lk-agents-fuzz/src/fakes/mod.rs) | Rust source | W40 |
| [src/fakes/nodes.rs](../../rust/crates/lk-agents-fuzz/src/fakes/nodes.rs) | Rust source | W40 |
| [src/fakes/output.rs](../../rust/crates/lk-agents-fuzz/src/fakes/output.rs) | Rust source | W40 |
| [src/fakes/realtime/duplex.rs](../../rust/crates/lk-agents-fuzz/src/fakes/realtime/duplex.rs) | Rust source | W40 |
| [src/fakes/realtime/events.rs](../../rust/crates/lk-agents-fuzz/src/fakes/realtime/events.rs) | Rust source | W40 |
| [src/fakes/realtime/mod.rs](../../rust/crates/lk-agents-fuzz/src/fakes/realtime/mod.rs) | Rust source | W40 |
| [src/fakes/realtime/server.rs](../../rust/crates/lk-agents-fuzz/src/fakes/realtime/server.rs) | Rust source | W40 |
| [src/fakes/room/audio_source.rs](../../rust/crates/lk-agents-fuzz/src/fakes/room/audio_source.rs) | Rust source | W40 |
| [src/fakes/room/mod.rs](../../rust/crates/lk-agents-fuzz/src/fakes/room/mod.rs) | Rust source | W40 |
| [src/fakes/room/track.rs](../../rust/crates/lk-agents-fuzz/src/fakes/room/track.rs) | Rust source | W40 |
| [src/fakes/storage.rs](../../rust/crates/lk-agents-fuzz/src/fakes/storage.rs) | Rust source | W40 |
| [src/fakes/stt.rs](../../rust/crates/lk-agents-fuzz/src/fakes/stt.rs) | Rust source | W40 |
| [src/fakes/tools.rs](../../rust/crates/lk-agents-fuzz/src/fakes/tools.rs) | Rust source | W40 |
| [src/fakes/tts.rs](../../rust/crates/lk-agents-fuzz/src/fakes/tts.rs) | Rust source | W40 |
| [src/fakes/vad.rs](../../rust/crates/lk-agents-fuzz/src/fakes/vad.rs) | Rust source | W40 |
| [src/fakes/vocabulary.rs](../../rust/crates/lk-agents-fuzz/src/fakes/vocabulary.rs) | Rust source | W40 |
| [src/input.rs](../../rust/crates/lk-agents-fuzz/src/input.rs) | Rust source | W40 |
| [src/input_tests.rs](../../rust/crates/lk-agents-fuzz/src/input_tests.rs) | test/support | W40 |
| [src/known_findings.tsv](../../rust/crates/lk-agents-fuzz/src/known_findings.tsv) | configuration/asset | W40 |
| [src/lib.rs](../../rust/crates/lk-agents-fuzz/src/lib.rs) | Rust source | W40 |
| [src/metrics_tests.rs](../../rust/crates/lk-agents-fuzz/src/metrics_tests.rs) | test/support | W40 |
| [src/pos_tests.rs](../../rust/crates/lk-agents-fuzz/src/pos_tests.rs) | explicit test target | W40 |
| [src/realtime_lifecycle_tests.rs](../../rust/crates/lk-agents-fuzz/src/realtime_lifecycle_tests.rs) | test/support | W40 |
| [src/realtime_tools_tests.rs](../../rust/crates/lk-agents-fuzz/src/realtime_tools_tests.rs) | test/support | W40 |
| [src/realtime_turns_tests.rs](../../rust/crates/lk-agents-fuzz/src/realtime_turns_tests.rs) | test/support | W40 |
| [src/record/resources.rs](../../rust/crates/lk-agents-fuzz/src/record/resources.rs) | Rust source | W40 |
| [src/record/views.rs](../../rust/crates/lk-agents-fuzz/src/record/views.rs) | Rust source | W40 |
| [src/record.rs](../../rust/crates/lk-agents-fuzz/src/record.rs) | Rust source | W40 |
| [src/regressions_tests.rs](../../rust/crates/lk-agents-fuzz/src/regressions_tests.rs) | explicit test target | W40 |
| [src/room_io_tests.rs](../../rust/crates/lk-agents-fuzz/src/room_io_tests.rs) | test/support | W40 |
| [src/rules/aligned_transcript_follows_tts.rs](../../rust/crates/lk-agents-fuzz/src/rules/aligned_transcript_follows_tts.rs) | Rust source | W40 |
| [src/rules/api_calls_return.rs](../../rust/crates/lk-agents-fuzz/src/rules/api_calls_return.rs) | Rust source | W40 |
| [src/rules/approval_runs_on_the_answer.rs](../../rust/crates/lk-agents-fuzz/src/rules/approval_runs_on_the_answer.rs) | Rust source | W40 |
| [src/rules/audio_conservation.rs](../../rust/crates/lk-agents-fuzz/src/rules/audio_conservation.rs) | Rust source | W40 |
| [src/rules/avatar_av_in_sync.rs](../../rust/crates/lk-agents-fuzz/src/rules/avatar_av_in_sync.rs) | Rust source | W40 |
| [src/rules/avatar_interrupt_clears_within_a_frame.rs](../../rust/crates/lk-agents-fuzz/src/rules/avatar_interrupt_clears_within_a_frame.rs) | Rust source | W40 |
| [src/rules/avatar_leaves_nothing.rs](../../rust/crates/lk-agents-fuzz/src/rules/avatar_leaves_nothing.rs) | Rust source | W40 |
| [src/rules/avatar_playback_finished_once.rs](../../rust/crates/lk-agents-fuzz/src/rules/avatar_playback_finished_once.rs) | Rust source | W40 |
| [src/rules/close_completes_at_its_bound.rs](../../rust/crates/lk-agents-fuzz/src/rules/close_completes_at_its_bound.rs) | Rust source | W40 |
| [src/rules/close_finishes.rs](../../rust/crates/lk-agents-fuzz/src/rules/close_finishes.rs) | Rust source | W40 |
| [src/rules/cut_keeps_heard_words.rs](../../rust/crates/lk-agents-fuzz/src/rules/cut_keeps_heard_words.rs) | Rust source | W40 |
| [src/rules/drain_answers_once_settled.rs](../../rust/crates/lk-agents-fuzz/src/rules/drain_answers_once_settled.rs) | Rust source | W40 |
| [src/rules/driver_exits_only_on_close.rs](../../rust/crates/lk-agents-fuzz/src/rules/driver_exits_only_on_close.rs) | Rust source | W40 |
| [src/rules/dtmf_reports_what_was_pressed.rs](../../rust/crates/lk-agents-fuzz/src/rules/dtmf_reports_what_was_pressed.rs) | Rust source | W40 |
| [src/rules/eot_judged_by_the_detector.rs](../../rust/crates/lk-agents-fuzz/src/rules/eot_judged_by_the_detector.rs) | Rust source | W40 |
| [src/rules/every_resource_released.rs](../../rust/crates/lk-agents-fuzz/src/rules/every_resource_released.rs) | Rust source | W40 |
| [src/rules/every_task_finishes.rs](../../rust/crates/lk-agents-fuzz/src/rules/every_task_finishes.rs) | Rust source | W40 |
| [src/rules/first_audio_is_prompt.rs](../../rust/crates/lk-agents-fuzz/src/rules/first_audio_is_prompt.rs) | Rust source | W40 |
| [src/rules/forwarded_while_enabled.rs](../../rust/crates/lk-agents-fuzz/src/rules/forwarded_while_enabled.rs) | Rust source | W40 |
| [src/rules/idle_is_cheap.rs](../../rust/crates/lk-agents-fuzz/src/rules/idle_is_cheap.rs) | Rust source | W40 |
| [src/rules/idle_respects_holds.rs](../../rust/crates/lk-agents-fuzz/src/rules/idle_respects_holds.rs) | Rust source | W40 |
| [src/rules/interrupted_handle_is_journalled.rs](../../rust/crates/lk-agents-fuzz/src/rules/interrupted_handle_is_journalled.rs) | Rust source | W40 |
| [src/rules/interruption_has_cause.rs](../../rust/crates/lk-agents-fuzz/src/rules/interruption_has_cause.rs) | Rust source | W40 |
| [src/rules/items_carry_their_payload.rs](../../rust/crates/lk-agents-fuzz/src/rules/items_carry_their_payload.rs) | Rust source | W40 |
| [src/rules/mod.rs](../../rust/crates/lk-agents-fuzz/src/rules/mod.rs) | Rust source | W40 |
| [src/rules/no_premature_completion.rs](../../rust/crates/lk-agents-fuzz/src/rules/no_premature_completion.rs) | Rust source | W40 |
| [src/rules/nodes_are_the_active_agents.rs](../../rust/crates/lk-agents-fuzz/src/rules/nodes_are_the_active_agents.rs) | Rust source | W40 |
| [src/rules/nothing_left_live.rs](../../rust/crates/lk-agents-fuzz/src/rules/nothing_left_live.rs) | Rust source | W40 |
| [src/rules/on_exit_once_per_frame.rs](../../rust/crates/lk-agents-fuzz/src/rules/on_exit_once_per_frame.rs) | Rust source | W40 |
| [src/rules/option_comes_from_its_layer.rs](../../rust/crates/lk-agents-fuzz/src/rules/option_comes_from_its_layer.rs) | Rust source | W40 |
| [src/rules/output_replacement_is_clean.rs](../../rust/crates/lk-agents-fuzz/src/rules/output_replacement_is_clean.rs) | Rust source | W40 |
| [src/rules/pause_resolves.rs](../../rust/crates/lk-agents-fuzz/src/rules/pause_resolves.rs) | Rust source | W40 |
| [src/rules/playback_requires_authorization.rs](../../rust/crates/lk-agents-fuzz/src/rules/playback_requires_authorization.rs) | Rust source | W40 |
| [src/rules/processor_span_is_one_error.rs](../../rust/crates/lk-agents-fuzz/src/rules/processor_span_is_one_error.rs) | Rust source | W40 |
| [src/rules/provider_failure_is_accounted.rs](../../rust/crates/lk-agents-fuzz/src/rules/provider_failure_is_accounted.rs) | Rust source | W40 |
| [src/rules/provider_streams_closed.rs](../../rust/crates/lk-agents-fuzz/src/rules/provider_streams_closed.rs) | Rust source | W40 |
| [src/rules/realtime_reply_cancelled_with_cause.rs](../../rust/crates/lk-agents-fuzz/src/rules/realtime_reply_cancelled_with_cause.rs) | Rust source | W40 |
| [src/rules/realtime_reply_reads_tool_output.rs](../../rust/crates/lk-agents-fuzz/src/rules/realtime_reply_reads_tool_output.rs) | Rust source | W40 |
| [src/rules/realtime_server_accepts_requests.rs](../../rust/crates/lk-agents-fuzz/src/rules/realtime_server_accepts_requests.rs) | Rust source | W40 |
| [src/rules/realtime_server_holds_what_was_heard.rs](../../rust/crates/lk-agents-fuzz/src/rules/realtime_server_holds_what_was_heard.rs) | Rust source | W40 |
| [src/rules/refused_interrupt_is_no_interruption.rs](../../rust/crates/lk-agents-fuzz/src/rules/refused_interrupt_is_no_interruption.rs) | Rust source | W40 |
| [src/rules/released_provider_unused.rs](../../rust/crates/lk-agents-fuzz/src/rules/released_provider_unused.rs) | Rust source | W40 |
| [src/rules/reply_reaches_its_model.rs](../../rust/crates/lk-agents-fuzz/src/rules/reply_reaches_its_model.rs) | Rust source | W40 |
| [src/rules/retries_follow_conn_options.rs](../../rust/crates/lk-agents-fuzz/src/rules/retries_follow_conn_options.rs) | Rust source | W40 |
| [src/rules/shown_transcript_is_conserved.rs](../../rust/crates/lk-agents-fuzz/src/rules/shown_transcript_is_conserved.rs) | Rust source | W40 |
| [src/rules/spans_nest_and_end.rs](../../rust/crates/lk-agents-fuzz/src/rules/spans_nest_and_end.rs) | Rust source | W40 |
| [src/rules/speech_done_once.rs](../../rust/crates/lk-agents-fuzz/src/rules/speech_done_once.rs) | Rust source | W40 |
| [src/rules/speech_error_is_injected.rs](../../rust/crates/lk-agents-fuzz/src/rules/speech_error_is_injected.rs) | Rust source | W40 |
| [src/rules/speech_is_heard.rs](../../rust/crates/lk-agents-fuzz/src/rules/speech_is_heard.rs) | Rust source | W40 |
| [src/rules/speech_keeps_the_hosts_setting.rs](../../rust/crates/lk-agents-fuzz/src/rules/speech_keeps_the_hosts_setting.rs) | Rust source | W40 |
| [src/rules/speech_settles.rs](../../rust/crates/lk-agents-fuzz/src/rules/speech_settles.rs) | Rust source | W40 |
| [src/rules/synthesized_speech_is_played.rs](../../rust/crates/lk-agents-fuzz/src/rules/synthesized_speech_is_played.rs) | Rust source | W40 |
| [src/rules/text_conservation.rs](../../rust/crates/lk-agents-fuzz/src/rules/text_conservation.rs) | Rust source | W40 |
| [src/rules/tool_call_arguments_are_json.rs](../../rust/crates/lk-agents-fuzz/src/rules/tool_call_arguments_are_json.rs) | Rust source | W40 |
| [src/rules/tool_call_reaches_runner.rs](../../rust/crates/lk-agents-fuzz/src/rules/tool_call_reaches_runner.rs) | Rust source | W40 |
| [src/rules/tool_lifecycle.rs](../../rust/crates/lk-agents-fuzz/src/rules/tool_lifecycle.rs) | Rust source | W40 |
| [src/rules/tool_output_follows_outcome.rs](../../rust/crates/lk-agents-fuzz/src/rules/tool_output_follows_outcome.rs) | Rust source | W40 |
| [src/rules/tool_updates_are_told.rs](../../rust/crates/lk-agents-fuzz/src/rules/tool_updates_are_told.rs) | Rust source | W40 |
| [src/rules/transcript_follows_audio.rs](../../rust/crates/lk-agents-fuzz/src/rules/transcript_follows_audio.rs) | Rust source | W40 |
| [src/rules/turn_hook_is_awaited.rs](../../rust/crates/lk-agents-fuzz/src/rules/turn_hook_is_awaited.rs) | Rust source | W40 |
| [src/rules/turn_limit_cuts_in_once.rs](../../rust/crates/lk-agents-fuzz/src/rules/turn_limit_cuts_in_once.rs) | Rust source | W40 |
| [src/rules/virtual_time_advances.rs](../../rust/crates/lk-agents-fuzz/src/rules/virtual_time_advances.rs) | Rust source | W40 |
| [src/rules/voice_task_restores_its_parent.rs](../../rust/crates/lk-agents-fuzz/src/rules/voice_task_restores_its_parent.rs) | Rust source | W40 |
| [src/rules/voice_task_returns_its_value.rs](../../rust/crates/lk-agents-fuzz/src/rules/voice_task_returns_its_value.rs) | Rust source | W40 |
| [src/rules/voice_task_skips_parent_hooks.rs](../../rust/crates/lk-agents-fuzz/src/rules/voice_task_skips_parent_hooks.rs) | Rust source | W40 |
| [src/sched.rs](../../rust/crates/lk-agents-fuzz/src/sched.rs) | Rust source | W40 |
| [src/search/conversation/microphone.rs](../../rust/crates/lk-agents-fuzz/src/search/conversation/microphone.rs) | Rust source | W40 |
| [src/search/conversation/providers.rs](../../rust/crates/lk-agents-fuzz/src/search/conversation/providers.rs) | Rust source | W40 |
| [src/search/conversation/speaker.rs](../../rust/crates/lk-agents-fuzz/src/search/conversation/speaker.rs) | Rust source | W40 |
| [src/search/conversation.rs](../../rust/crates/lk-agents-fuzz/src/search/conversation.rs) | Rust source | W40 |
| [src/search/mod.rs](../../rust/crates/lk-agents-fuzz/src/search/mod.rs) | Rust source | W40 |
| [src/search/setup.rs](../../rust/crates/lk-agents-fuzz/src/search/setup.rs) | Rust source | W40 |
| [src/search/uniform.rs](../../rust/crates/lk-agents-fuzz/src/search/uniform.rs) | Rust source | W40 |
| [src/session_resources_tests.rs](../../rust/crates/lk-agents-fuzz/src/session_resources_tests.rs) | test/support | W40 |
| [src/session_state_tests.rs](../../rust/crates/lk-agents-fuzz/src/session_state_tests.rs) | test/support | W40 |
| [src/soak_tests.rs](../../rust/crates/lk-agents-fuzz/src/soak_tests.rs) | explicit test target | W40 |
| [src/speech_completion_tests.rs](../../rust/crates/lk-agents-fuzz/src/speech_completion_tests.rs) | test/support | W40 |
| [src/speech_playback_tests.rs](../../rust/crates/lk-agents-fuzz/src/speech_playback_tests.rs) | test/support | W40 |
| [src/survey_tests.rs](../../rust/crates/lk-agents-fuzz/src/survey_tests.rs) | explicit test target | W40 |
| [src/telephony_tests.rs](../../rust/crates/lk-agents-fuzz/src/telephony_tests.rs) | test/support | W40 |
| [src/text_session.rs](../../rust/crates/lk-agents-fuzz/src/text_session.rs) | Rust source | W40 |
| [src/text_session_survey_tests.rs](../../rust/crates/lk-agents-fuzz/src/text_session_survey_tests.rs) | explicit test target | W40 |
| [src/text_session_tests.rs](../../rust/crates/lk-agents-fuzz/src/text_session_tests.rs) | test/support | W40 |
| [src/tools_tests.rs](../../rust/crates/lk-agents-fuzz/src/tools_tests.rs) | test/support | W40 |

</details>

<details>
<summary>lk-agents-host-derive — 3 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/ai_slop/lk-agents-host-derive/Cargo.toml) | package manifest | W39 |
| [src/lib.rs](../../rust/ai_slop/lk-agents-host-derive/src/lib.rs) | Rust source | W39 |
| [src/table.rs](../../rust/ai_slop/lk-agents-host-derive/src/table.rs) | Rust source | W39 |

</details>

<details>
<summary>lk-agents-inference — 155 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-inference/Cargo.toml) | package manifest | W30 |
| [src/amd/detector.rs](../../rust/crates/lk-agents-inference/src/amd/detector.rs) | Rust source | W30 |
| [src/amd/mod.rs](../../rust/crates/lk-agents-inference/src/amd/mod.rs) | Rust source | W30 |
| [src/avatar/gateway.rs](../../rust/crates/lk-agents-inference/src/avatar/gateway.rs) | Rust source | W28 |
| [src/avatar/lifecycle.rs](../../rust/crates/lk-agents-inference/src/avatar/lifecycle.rs) | Rust source | W28 |
| [src/avatar/lifecycle_tests.rs](../../rust/crates/lk-agents-inference/src/avatar/lifecycle_tests.rs) | test/support | W28 |
| [src/avatar/mod.rs](../../rust/crates/lk-agents-inference/src/avatar/mod.rs) | Rust source | W28 |
| [src/avatar_tests.rs](../../rust/crates/lk-agents-inference/src/avatar_tests.rs) | test/support | W30 |
| [src/codecs/stream_buffer.rs](../../rust/crates/lk-agents-inference/src/codecs/stream_buffer.rs) | Rust source | W32 |
| [src/codecs.rs](../../rust/crates/lk-agents-inference/src/codecs.rs) | Rust source | W32 |
| [src/codecs_tests.rs](../../rust/crates/lk-agents-inference/src/codecs_tests.rs) | test/support | W30 |
| [src/credentials.rs](../../rust/crates/lk-agents-inference/src/credentials.rs) | Rust source | W30 |
| [src/duplex/gate.rs](../../rust/crates/lk-agents-inference/src/duplex/gate.rs) | Rust source | W30 |
| [src/duplex/host.rs](../../rust/crates/lk-agents-inference/src/duplex/host.rs) | Rust source | W30 |
| [src/duplex/mod.rs](../../rust/crates/lk-agents-inference/src/duplex/mod.rs) | Rust source | W30 |
| [src/duplex/segmenter.rs](../../rust/crates/lk-agents-inference/src/duplex/segmenter.rs) | Rust source | W30 |
| [src/duplex/session.rs](../../rust/crates/lk-agents-inference/src/duplex/session.rs) | Rust source | W30 |
| [src/duplex/session_tests.rs](../../rust/crates/lk-agents-inference/src/duplex/session_tests.rs) | test/support | W30 |
| [src/duplex_tests.rs](../../rust/crates/lk-agents-inference/src/duplex_tests.rs) | test/support | W30 |
| [src/eot/cloud.rs](../../rust/crates/lk-agents-inference/src/eot/cloud.rs) | Rust source | W30 |
| [src/eot/detector.rs](../../rust/crates/lk-agents-inference/src/eot/detector.rs) | Rust source | W30 |
| [src/eot/eot.rs](../../rust/crates/lk-agents-inference/src/eot/eot.rs) | Rust source | W30 |
| [src/eot/local.rs](../../rust/crates/lk-agents-inference/src/eot/local.rs) | Rust source | W30 |
| [src/eot/mod.rs](../../rust/crates/lk-agents-inference/src/eot/mod.rs) | Rust source | W30 |
| [src/eot/options.rs](../../rust/crates/lk-agents-inference/src/eot/options.rs) | Rust source | W30 |
| [src/eot/session.rs](../../rust/crates/lk-agents-inference/src/eot/session.rs) | Rust source | W30 |
| [src/eot/shell.rs](../../rust/crates/lk-agents-inference/src/eot/shell.rs) | Rust source | W30 |
| [src/eot/stream.rs](../../rust/crates/lk-agents-inference/src/eot/stream.rs) | Rust source | W30 |
| [src/eot/transport.rs](../../rust/crates/lk-agents-inference/src/eot/transport.rs) | Rust source | W30 |
| [src/eot_tests.rs](../../rust/crates/lk-agents-inference/src/eot_tests.rs) | test/support | W30 |
| [src/evals/mod.rs](../../rust/crates/lk-agents-inference/src/evals/mod.rs) | Rust source | W30 |
| [src/evals/prompts.rs](../../rust/crates/lk-agents-inference/src/evals/prompts.rs) | Rust source | W30 |
| [src/evals/transcript.rs](../../rust/crates/lk-agents-inference/src/evals/transcript.rs) | Rust source | W30 |
| [src/evals_tests.rs](../../rust/crates/lk-agents-inference/src/evals_tests.rs) | test/support | W30 |
| [src/extra.rs](../../rust/crates/lk-agents-inference/src/extra.rs) | Rust source | W30 |
| [src/fallback/adapter_tests.rs](../../rust/crates/lk-agents-inference/src/fallback/adapter_tests.rs) | test/support | W30 |
| [src/fallback/chain.rs](../../rust/crates/lk-agents-inference/src/fallback/chain.rs) | Rust source | W30 |
| [src/fallback/channel.rs](../../rust/crates/lk-agents-inference/src/fallback/channel.rs) | Rust source | W30 |
| [src/fallback/host_tests.rs](../../rust/crates/lk-agents-inference/src/fallback/host_tests.rs) | test/support | W30 |
| [src/fallback/llm.rs](../../rust/crates/lk-agents-inference/src/fallback/llm.rs) | Rust source | W30 |
| [src/fallback/machine.rs](../../rust/crates/lk-agents-inference/src/fallback/machine.rs) | Rust source | W30 |
| [src/fallback/machine_tests.rs](../../rust/crates/lk-agents-inference/src/fallback/machine_tests.rs) | test/support | W30 |
| [src/fallback/mod.rs](../../rust/crates/lk-agents-inference/src/fallback/mod.rs) | Rust source | W30 |
| [src/fallback/realtime/backlog.rs](../../rust/crates/lk-agents-inference/src/fallback/realtime/backlog.rs) | Rust source | W30 |
| [src/fallback/realtime/capabilities.rs](../../rust/crates/lk-agents-inference/src/fallback/realtime/capabilities.rs) | Rust source | W30 |
| [src/fallback/realtime/mod.rs](../../rust/crates/lk-agents-inference/src/fallback/realtime/mod.rs) | Rust source | W30 |
| [src/fallback/realtime/session.rs](../../rust/crates/lk-agents-inference/src/fallback/realtime/session.rs) | Rust source | W30 |
| [src/fallback/realtime_tests.rs](../../rust/crates/lk-agents-inference/src/fallback/realtime_tests.rs) | test/support | W30 |
| [src/fallback/stt.rs](../../rust/crates/lk-agents-inference/src/fallback/stt.rs) | Rust source | W30 |
| [src/fallback/tts/ports.rs](../../rust/crates/lk-agents-inference/src/fallback/tts/ports.rs) | Rust source | W30 |
| [src/fallback/tts.rs](../../rust/crates/lk-agents-inference/src/fallback/tts.rs) | Rust source | W30 |
| [src/fallback_tests.rs](../../rust/crates/lk-agents-inference/src/fallback_tests.rs) | test/support | W30 |
| [src/gateway_tests.rs](../../rust/crates/lk-agents-inference/src/gateway_tests.rs) | test/support | W30 |
| [src/headers.rs](../../rust/crates/lk-agents-inference/src/headers.rs) | Rust source | W30 |
| [src/ids.rs](../../rust/crates/lk-agents-inference/src/ids.rs) | Rust source | W30 |
| [src/images.rs](../../rust/crates/lk-agents-inference/src/images.rs) | Rust source | W32 |
| [src/inference_tests.rs](../../rust/crates/lk-agents-inference/src/inference_tests.rs) | explicit test target | W30 |
| [src/interruption/audio.rs](../../rust/crates/lk-agents-inference/src/interruption/audio.rs) | Rust source | W30 |
| [src/interruption/cache.rs](../../rust/crates/lk-agents-inference/src/interruption/cache.rs) | Rust source | W30 |
| [src/interruption/core.rs](../../rust/crates/lk-agents-inference/src/interruption/core.rs) | Rust source | W30 |
| [src/interruption/core_tests.rs](../../rust/crates/lk-agents-inference/src/interruption/core_tests.rs) | test/support | W30 |
| [src/interruption/detector.rs](../../rust/crates/lk-agents-inference/src/interruption/detector.rs) | Rust source | W30 |
| [src/interruption/event.rs](../../rust/crates/lk-agents-inference/src/interruption/event.rs) | Rust source | W30 |
| [src/interruption/leg.rs](../../rust/crates/lk-agents-inference/src/interruption/leg.rs) | Rust source | W30 |
| [src/interruption/mod.rs](../../rust/crates/lk-agents-inference/src/interruption/mod.rs) | Rust source | W30 |
| [src/interruption/options.rs](../../rust/crates/lk-agents-inference/src/interruption/options.rs) | Rust source | W30 |
| [src/interruption/proto.rs](../../rust/crates/lk-agents-inference/src/interruption/proto.rs) | Rust source | W30 |
| [src/interruption/retry.rs](../../rust/crates/lk-agents-inference/src/interruption/retry.rs) | Rust source | W30 |
| [src/interruption/transport.rs](../../rust/crates/lk-agents-inference/src/interruption/transport.rs) | Rust source | W30 |
| [src/interruption/transport_tests.rs](../../rust/crates/lk-agents-inference/src/interruption/transport_tests.rs) | test/support | W30 |
| [src/interruption/websocket_tests.rs](../../rust/crates/lk-agents-inference/src/interruption/websocket_tests.rs) | test/support | W30 |
| [src/lib.rs](../../rust/crates/lk-agents-inference/src/lib.rs) | Rust source | W30 |
| [src/llm/algorithms.rs](../../rust/crates/lk-agents-inference/src/llm/algorithms.rs) | Rust source | W30 |
| [src/llm/gateway/request.rs](../../rust/crates/lk-agents-inference/src/llm/gateway/request.rs) | Rust source | W30 |
| [src/llm/gateway.rs](../../rust/crates/lk-agents-inference/src/llm/gateway.rs) | Rust source | W30 |
| [src/llm/lifecycle.rs](../../rust/crates/lk-agents-inference/src/llm/lifecycle.rs) | Rust source | W30 |
| [src/llm/lifecycle_tests.rs](../../rust/crates/lk-agents-inference/src/llm/lifecycle_tests.rs) | test/support | W30 |
| [src/llm/live_tests.rs](../../rust/crates/lk-agents-inference/src/llm/live_tests.rs) | test/support | W30 |
| [src/llm/mod.rs](../../rust/crates/lk-agents-inference/src/llm/mod.rs) | Rust source | W30 |
| [src/llm/models.rs](../../rust/crates/lk-agents-inference/src/llm/models.rs) | Rust source | W30 |
| [src/llm/parse.rs](../../rust/crates/lk-agents-inference/src/llm/parse.rs) | Rust source | W30 |
| [src/llm/request.rs](../../rust/crates/lk-agents-inference/src/llm/request.rs) | Rust source | W30 |
| [src/llm/thinking.rs](../../rust/crates/lk-agents-inference/src/llm/thinking.rs) | Rust source | W30 |
| [src/llm_tests.rs](../../rust/crates/lk-agents-inference/src/llm_tests.rs) | test/support | W30 |
| [src/local_models.rs](../../rust/crates/lk-agents-inference/src/local_models.rs) | Rust source | W30 |
| [src/models.rs](../../rust/crates/lk-agents-inference/src/models.rs) | Rust source | W30 |
| [src/provider.rs](../../rust/crates/lk-agents-inference/src/provider.rs) | Rust source | W30 |
| [src/realtime/budget_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/budget_tests.rs) | test/support | W30 |
| [src/realtime/client_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/client_tests.rs) | test/support | W30 |
| [src/realtime/connection.rs](../../rust/crates/lk-agents-inference/src/realtime/connection.rs) | Rust source | W30 |
| [src/realtime/context_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/context_tests.rs) | explicit test target | W30 |
| [src/realtime/dialect.rs](../../rust/crates/lk-agents-inference/src/realtime/dialect.rs) | Rust source | W30 |
| [src/realtime/history_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/history_tests.rs) | test/support | W30 |
| [src/realtime/host.rs](../../rust/crates/lk-agents-inference/src/realtime/host.rs) | Rust source | W30 |
| [src/realtime/machine/context/preparation.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/context/preparation.rs) | Rust source | W30 |
| [src/realtime/machine/context/retirement.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/context/retirement.rs) | Rust source | W30 |
| [src/realtime/machine/context/server_items.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/context/server_items.rs) | Rust source | W30 |
| [src/realtime/machine/context.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/context.rs) | Rust source | W30 |
| [src/realtime/machine/contract_review_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/contract_review_tests.rs) | test/support | W30 |
| [src/realtime/machine/contract_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/contract_tests.rs) | test/support | W30 |
| [src/realtime/machine/effects.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/effects.rs) | Rust source | W30 |
| [src/realtime/machine/incoming.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/incoming.rs) | Rust source | W30 |
| [src/realtime/machine/mod.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/mod.rs) | Rust source | W30 |
| [src/realtime/machine/options.rs](../../rust/crates/lk-agents-inference/src/realtime/machine/options.rs) | Rust source | W30 |
| [src/realtime/mod.rs](../../rust/crates/lk-agents-inference/src/realtime/mod.rs) | Rust source | W30 |
| [src/realtime/openai/README.md](../../rust/crates/lk-agents-inference/src/realtime/openai/README.md) | documentation | W30 |
| [src/realtime/openai/encode.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/encode.rs) | Rust source | W30 |
| [src/realtime/openai/machine_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/machine_tests.rs) | test/support | W30 |
| [src/realtime/openai/mod.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/mod.rs) | Rust source | W30 |
| [src/realtime/openai/model.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/model.rs) | Rust source | W30 |
| [src/realtime/openai/queue.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/queue.rs) | Rust source | W30 |
| [src/realtime/openai/session/retirement.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/session/retirement.rs) | Rust source | W30 |
| [src/realtime/openai/session.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/session.rs) | Rust source | W30 |
| [src/realtime/openai/session_retirement_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/session_retirement_tests.rs) | test/support | W30 |
| [src/realtime/openai/testfix.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/testfix.rs) | Rust source | W30 |
| [src/realtime/openai/wire.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/wire.rs) | Rust source | W30 |
| [src/realtime/openai/writer.rs](../../rust/crates/lk-agents-inference/src/realtime/openai/writer.rs) | Rust source | W30 |
| [src/realtime/protocol.rs](../../rust/crates/lk-agents-inference/src/realtime/protocol.rs) | Rust source | W30 |
| [src/realtime/remote.rs](../../rust/crates/lk-agents-inference/src/realtime/remote.rs) | Rust source | W30 |
| [src/realtime/server_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/server_tests.rs) | test/support | W30 |
| [src/realtime/session_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/session_tests.rs) | test/support | W30 |
| [src/realtime/simulation_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/simulation_tests.rs) | test/support | W30 |
| [src/realtime/sync_tests.rs](../../rust/crates/lk-agents-inference/src/realtime/sync_tests.rs) | test/support | W30 |
| [src/realtime/testfix.rs](../../rust/crates/lk-agents-inference/src/realtime/testfix.rs) | Rust source | W30 |
| [src/realtime_tests.rs](../../rust/crates/lk-agents-inference/src/realtime_tests.rs) | test/support | W30 |
| [src/stt/lifecycle.rs](../../rust/crates/lk-agents-inference/src/stt/lifecycle.rs) | Rust source | W30 |
| [src/stt/machine.rs](../../rust/crates/lk-agents-inference/src/stt/machine.rs) | Rust source | W30 |
| [src/stt/mod.rs](../../rust/crates/lk-agents-inference/src/stt/mod.rs) | Rust source | W30 |
| [src/stt/options.rs](../../rust/crates/lk-agents-inference/src/stt/options.rs) | Rust source | W30 |
| [src/stt/pump.rs](../../rust/crates/lk-agents-inference/src/stt/pump.rs) | Rust source | W30 |
| [src/stt/retrying.rs](../../rust/crates/lk-agents-inference/src/stt/retrying.rs) | Rust source | W30 |
| [src/stt/retrying_tests.rs](../../rust/crates/lk-agents-inference/src/stt/retrying_tests.rs) | test/support | W30 |
| [src/stt/stream.rs](../../rust/crates/lk-agents-inference/src/stt/stream.rs) | Rust source | W30 |
| [src/stt/stream_adapter.rs](../../rust/crates/lk-agents-inference/src/stt/stream_adapter.rs) | Rust source | W30 |
| [src/stt/wire.rs](../../rust/crates/lk-agents-inference/src/stt/wire.rs) | Rust source | W30 |
| [src/stt_tests.rs](../../rust/crates/lk-agents-inference/src/stt_tests.rs) | test/support | W30 |
| [src/testkit.rs](../../rust/crates/lk-agents-inference/src/testkit.rs) | Rust source | W30 |
| [src/tts/adapter.rs](../../rust/crates/lk-agents-inference/src/tts/adapter.rs) | Rust source | W30 |
| [src/tts/checks.rs](../../rust/crates/lk-agents-inference/src/tts/checks.rs) | Rust source | W30 |
| [src/tts/connection_tests.rs](../../rust/crates/lk-agents-inference/src/tts/connection_tests.rs) | test/support | W30 |
| [src/tts/emitter.rs](../../rust/crates/lk-agents-inference/src/tts/emitter.rs) | Rust source | W30 |
| [src/tts/emitter_tests.rs](../../rust/crates/lk-agents-inference/src/tts/emitter_tests.rs) | test/support | W30 |
| [src/tts/machine.rs](../../rust/crates/lk-agents-inference/src/tts/machine.rs) | Rust source | W30 |
| [src/tts/mod.rs](../../rust/crates/lk-agents-inference/src/tts/mod.rs) | Rust source | W30 |
| [src/tts/pacer.rs](../../rust/crates/lk-agents-inference/src/tts/pacer.rs) | Rust source | W30 |
| [src/tts/ports.rs](../../rust/crates/lk-agents-inference/src/tts/ports.rs) | Rust source | W30 |
| [src/tts/pump.rs](../../rust/crates/lk-agents-inference/src/tts/pump.rs) | Rust source | W30 |
| [src/tts/sentence_adapter.rs](../../rust/crates/lk-agents-inference/src/tts/sentence_adapter.rs) | Rust source | W30 |
| [src/tts/sentences.rs](../../rust/crates/lk-agents-inference/src/tts/sentences.rs) | Rust source | W30 |
| [src/tts/stream.rs](../../rust/crates/lk-agents-inference/src/tts/stream.rs) | Rust source | W30 |
| [src/tts/wire.rs](../../rust/crates/lk-agents-inference/src/tts/wire.rs) | Rust source | W30 |
| [src/vad/machine.rs](../../rust/crates/lk-agents-inference/src/vad/machine.rs) | Rust source | W30 |
| [src/vad/mod.rs](../../rust/crates/lk-agents-inference/src/vad/mod.rs) | Rust source | W30 |
| [src/vad/stream.rs](../../rust/crates/lk-agents-inference/src/vad/stream.rs) | Rust source | W30 |
| [src/vad_tests.rs](../../rust/crates/lk-agents-inference/src/vad_tests.rs) | test/support | W30 |

</details>

<details>
<summary>lk-agents-media — 404 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [.gitignore](../../rust/crates/lk-agents-media/.gitignore) | configuration/asset | W32 |
| [Cargo.toml](../../rust/crates/lk-agents-media/Cargo.toml) | package manifest | W32 |
| [build/blingfire.rs](../../rust/crates/lk-agents-media/build/blingfire.rs) | build support | W32 |
| [build/cache.rs](../../rust/crates/lk-agents-media/build/cache.rs) | build support | W32 |
| [build/libraries.rs](../../rust/crates/lk-agents-media/build/libraries.rs) | build support | W32 |
| [build/native_cache_tests.rs](../../rust/crates/lk-agents-media/build/native_cache_tests.rs) | explicit test target | W32 |
| [build/settings.rs](../../rust/crates/lk-agents-media/build/settings.rs) | build support | W32 |
| [build.rs](../../rust/crates/lk-agents-media/build.rs) | build support | W32 |
| [csrc/blingfire/blingfire_wrap.cpp](../../rust/crates/lk-agents-media/csrc/blingfire/blingfire_wrap.cpp) | configuration/asset | W32 |
| [csrc/blingfire/blingfire_wrap.h](../../rust/crates/lk-agents-media/csrc/blingfire/blingfire_wrap.h) | configuration/asset | W32 |
| [csrc/blingfire/shim.cpp](../../rust/crates/lk-agents-media/csrc/blingfire/shim.cpp) | configuration/asset | W32 |
| [csrc/lk_agents_media.c](../../rust/crates/lk-agents-media/csrc/lk_agents_media.c) | configuration/asset | W32 |
| [src/audio_byte_stream.rs](../../rust/crates/lk-agents-media/src/audio_byte_stream.rs) | Rust source | W32 |
| [src/audio_decoder.rs](../../rust/crates/lk-agents-media/src/audio_decoder.rs) | Rust source | W32 |
| [src/blingfire.rs](../../rust/crates/lk-agents-media/src/blingfire.rs) | Rust source | W32 |
| [src/emitter/mod.rs](../../rust/crates/lk-agents-media/src/emitter/mod.rs) | Rust source | W32 |
| [src/emitter_tests.rs](../../rust/crates/lk-agents-media/src/emitter_tests.rs) | test/support | W32 |
| [src/error.rs](../../rust/crates/lk-agents-media/src/error.rs) | Rust source | W32 |
| [src/ffmpeg/audio_decoder.rs](../../rust/crates/lk-agents-media/src/ffmpeg/audio_decoder.rs) | Rust source | W32 |
| [src/ffmpeg/audio_encoder.rs](../../rust/crates/lk-agents-media/src/ffmpeg/audio_encoder.rs) | Rust source | W32 |
| [src/ffmpeg/fixtures_tests.rs](../../rust/crates/lk-agents-media/src/ffmpeg/fixtures_tests.rs) | fixture/counterexample | W32 |
| [src/ffmpeg/mod.rs](../../rust/crates/lk-agents-media/src/ffmpeg/mod.rs) | Rust source | W32 |
| [src/filters.rs](../../rust/crates/lk-agents-media/src/filters.rs) | Rust source | W32 |
| [src/images/encode.rs](../../rust/crates/lk-agents-media/src/images/encode.rs) | Rust source | W32 |
| [src/images/mod.rs](../../rust/crates/lk-agents-media/src/images/mod.rs) | Rust source | W32 |
| [src/images/prepare.rs](../../rust/crates/lk-agents-media/src/images/prepare.rs) | Rust source | W32 |
| [src/images/resolve.rs](../../rust/crates/lk-agents-media/src/images/resolve.rs) | Rust source | W32 |
| [src/images/serialize.rs](../../rust/crates/lk-agents-media/src/images/serialize.rs) | Rust source | W32 |
| [src/lib.rs](../../rust/crates/lk-agents-media/src/lib.rs) | Rust source | W32 |
| [src/mime.rs](../../rust/crates/lk-agents-media/src/mime.rs) | Rust source | W32 |
| [src/mixer.rs](../../rust/crates/lk-agents-media/src/mixer.rs) | Rust source | W32 |
| [src/pool.rs](../../rust/crates/lk-agents-media/src/pool.rs) | Rust source | W32 |
| [src/pool_tests.rs](../../rust/crates/lk-agents-media/src/pool_tests.rs) | test/support | W32 |
| [src/recorder/file.rs](../../rust/crates/lk-agents-media/src/recorder/file.rs) | Rust source | W18 |
| [src/recorder.rs](../../rust/crates/lk-agents-media/src/recorder.rs) | Rust source | W18 |
| [src/recorder_tests.rs](../../rust/crates/lk-agents-media/src/recorder_tests.rs) | test/support | W18 |
| [src/resample/frame.rs](../../rust/crates/lk-agents-media/src/resample/frame.rs) | Rust source | W32 |
| [src/resample/mod.rs](../../rust/crates/lk-agents-media/src/resample/mod.rs) | Rust source | W32 |
| [src/resample/stream.rs](../../rust/crates/lk-agents-media/src/resample/stream.rs) | Rust source | W32 |
| [src/source.rs](../../rust/crates/lk-agents-media/src/source.rs) | Rust source | W32 |
| [src/soxr.rs](../../rust/crates/lk-agents-media/src/soxr.rs) | Rust source | W32 |
| [src/wav.rs](../../rust/crates/lk-agents-media/src/wav.rs) | Rust source | W32 |
| [vendor/blingfire-18e9a19e5860/LICENSE](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/LICENSE) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAAllocator.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAAllocator.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAAllocatorA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAAllocatorA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAArrayCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAArrayCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAArray_cont_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAArray_cont_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAArray_pack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAArray_pack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FABrResultCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FABrResultCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAChains_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAChains_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAConfig.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAConfig.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FADictConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FADictConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FADictInterpreter_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FADictInterpreter_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FADigitizer_dct_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FADigitizer_dct_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FADigitizer_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FADigitizer_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAEncodeUtils.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAEncodeUtils.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAException.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAException.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAFsmConst.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAFsmConst.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAGetIWsCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAGetIWsCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAGetIWs_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAGetIWs_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAGlobalConfKeeper_packaged.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAGlobalConfKeeper_packaged.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAHmmTagger_l1.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAHmmTagger_l1.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAHyphConfKeeper_packaged.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAHyphConfKeeper_packaged.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAImageDump.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAImageDump.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAIwMap_pack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAIwMap_pack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FALDB.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FALDB.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FALexTools_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FALexTools_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FALimits.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FALimits.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMealyDfaCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMealyDfaCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMealyDfa_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMealyDfa_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMealyNfaCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMealyNfaCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMorphLDB_t_packaged.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMorphLDB_t_packaged.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMphInterpretTools_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMphInterpretTools_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMsg.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMsg.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMultiMapCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMultiMapCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMultiMap_pack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMultiMap_pack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMultiMap_pack_fixed.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMultiMap_pack_fixed.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMultiMap_pack_mph.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAMultiMap_pack_mph.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAOffsetTable_pack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAOffsetTable_pack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAOw2IwCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAOw2IwCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAOw2Iw_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAOw2Iw_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAParseTreeA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAParseTreeA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAParseTree_memlfp.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAParseTree_memlfp.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAParserConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAParserConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FARSDfaCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FARSDfaCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FARSDfa_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FARSDfa_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FARSNfaCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FARSNfaCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FARegexpTags_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FARegexpTags_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAResolveMatchA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAResolveMatchA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FASecurity.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FASecurity.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FASetImageA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FASetImageA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAState2OwCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAState2OwCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAState2Ow_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAState2Ow_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAState2OwsCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAState2OwsCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAState2Ows_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAState2Ows_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAStemmerConst_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAStemmerConst_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAStemmerLDB.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAStemmerLDB.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAStemmer_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAStemmer_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAStringArray_pack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAStringArray_pack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FASuffixInterpretToolsConst_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FASuffixInterpretToolsConst_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FASuffixInterpretTools_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FASuffixInterpretTools_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAT2PTable.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAT2PTable.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATaggedTextCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATaggedTextCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATokenSegmentationToolsCA_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATokenSegmentationToolsCA_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATokenSegmentationTools_1best_bpe_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATokenSegmentationTools_1best_bpe_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATokenSegmentationTools_1best_bpe_with_merges_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATokenSegmentationTools_1best_bpe_with_merges_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATokenSegmentationTools_1best_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATokenSegmentationTools_1best_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransformCA_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransformCA_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_capital_rev_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_capital_rev_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_capital_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_capital_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_cascade_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_cascade_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_hyph_redup_rev_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_hyph_redup_rev_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_hyph_redup_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_hyph_redup_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_prefix_rev_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_prefix_rev_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_prefix_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_prefix_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_unescape_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATransform_unescape_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATrsConfKeeper_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATrsConfKeeper_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATs2PTable.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATs2PTable.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATsConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FATsConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAUtf32Utils.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAUtf32Utils.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAUtf8Utils.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAUtf8Utils.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAUtils_cl.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAUtils_cl.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAW2PConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAW2PConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAW2SConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAW2SConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWREConfCA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWREConfCA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWREConf_pack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWREConf_pack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWbdConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWbdConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWftConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWftConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWgConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWgConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWordGuesser_prob_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWordGuesser_prob_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWordGuesser_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWordGuesser_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWordToProb_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWordToProb_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWreLexTools_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/FAWreLexTools_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/blingfire-client_src_pch.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/inc/blingfire-client_src_pch.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAAllocator.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAAllocator.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAArray_pack.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAArray_pack.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAChains_pack_triv.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAChains_pack_triv.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FADictConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FADictConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAException.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAException.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAGetIWs_pack_triv.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAGetIWs_pack_triv.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAGlobalConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAGlobalConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAHmmTagger_l1.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAHmmTagger_l1.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAHyphConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAHyphConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAImageDump.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAImageDump.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAIwMap_pack.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAIwMap_pack.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FALDB.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FALDB.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMealyDfa_pack_triv.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMealyDfa_pack_triv.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMsg.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMsg.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMultiMap_pack.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMultiMap_pack.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMultiMap_pack_fixed.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMultiMap_pack_fixed.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMultiMap_pack_mph.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAMultiMap_pack_mph.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FANormalizeDiacriticsMapPreserve.cxx](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FANormalizeDiacriticsMapPreserve.cxx) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FANormalizeDiacriticsMapProd.cxx](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FANormalizeDiacriticsMapProd.cxx) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FANormalizeDiacriticsMapRemove.cxx](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FANormalizeDiacriticsMapRemove.cxx) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAOffsetTable_pack.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAOffsetTable_pack.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAOw2Iw_pack_triv.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAOw2Iw_pack_triv.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAParseTree_memlfp.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAParseTree_memlfp.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAParserConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAParserConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FARSDfa_pack_triv.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FARSDfa_pack_triv.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAState2Ow_pack_triv.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAState2Ow_pack_triv.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAState2Ows_pack_triv.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAState2Ows_pack_triv.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAStemmerLDB.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAStemmerLDB.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAStringArray_pack.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAStringArray_pack.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAT2PTable.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAT2PTable.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FATs2PTable.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FATs2PTable.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FATsConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FATsConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtf32ToLower.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtf32ToLower.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtf32ToUpper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtf32ToUpper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtf32Utils.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtf32Utils.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtf8Utils.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtf8Utils.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtils_cl.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAUtils_cl.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAW2PConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAW2PConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAW2SConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAW2SConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAWREConf_pack.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAWREConf_pack.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAWbdConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAWbdConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAWftConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAWftConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAWgConfKeeper.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/FAWgConfKeeper.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfireclient.library/src/blingfire-client_src_pch.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfireclient.library/src/blingfire-client_src_pch.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAct2ArrA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAct2ArrA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAct2Arr_css.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAct2Arr_css.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAct2Arr_data.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAct2Arr_data.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAct2Arr_score_tag.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAct2Arr_score_tag.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAActionsA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAActionsA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAActionsArrayA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAActionsArrayA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAActionsArray_static.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAActionsArray_static.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAllocator.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAllocator.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAllocatorA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAllocatorA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAny2AnyOther_global_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAny2AnyOther_global_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAny2AnyOther_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAny2AnyOther_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAArrayPack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAArrayPack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAArray_cont_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAArray_cont_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAArray_p2ca.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAArray_p2ca.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAArray_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAArray_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutIOTools.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutIOTools.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools2_trbr_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools2_trbr_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools_fnfa_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools_fnfa_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools_pos_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools_pos_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools_trbr_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAAutInterpretTools_trbr_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FABitArray.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FABitArray.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FABitMatrix_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FABitMatrix_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FABrResult.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FABrResult.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FABrResultA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FABrResultA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACalcIwEqClasses.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACalcIwEqClasses.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACalcMealy1.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACalcMealy1.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACalcMealy2.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACalcMealy2.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChain2NumA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChain2NumA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChain2Num_hash.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChain2Num_hash.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChain2Num_judy.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChain2Num_judy.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChains2MinDfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChains2MinDfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChains2MinDfa_sort.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChains2MinDfa_sort.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChainsPack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAChainsPack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACmpTrBrOws_greedy.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACmpTrBrOws_greedy.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAColorGraph_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAColorGraph_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAConfParser.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAConfParser.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACorpusIOTools_utf8.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACorpusIOTools_utf8.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACssLDB.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FACssLDB.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfa2MealyNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfa2MealyNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfa2MinDfa_hg_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfa2MinDfa_hg_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfaPack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfaPack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfaTopoGraph.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfaTopoGraph.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfas2CommonNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADfas2CommonNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADict2Classifier.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADict2Classifier.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADictSplit.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADictSplit.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADictStr2SuffixRule.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FADictStr2SuffixRule.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoderA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoderA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_delta_pref.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_delta_pref.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_delta_pref_mask.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_delta_pref_mask.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_mask.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_mask.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_pm_dpm.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_pm_dpm.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_pref.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_pref.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_pref_mask.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEncoder_pref_mask.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEpsilonGraph.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEpsilonGraph.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEpsilonGraph_mealy.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEpsilonGraph_mealy.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEpsilonRemoval.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAEpsilonRemoval.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAException.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAException.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAExtendClassifier_cxp.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAExtendClassifier_cxp.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAFloatArrayPack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAFloatArrayPack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAFsmRenum.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAFsmRenum.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAGcInterpreter_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAGcInterpreter_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAGcLDB.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAGcLDB.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAHeap_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAHeap_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAHyphInterpreter_core_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAHyphInterpreter_core_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAHyphInterpreter_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAHyphInterpreter_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAIsDotNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAIsDotNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAIwMapPack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAIwMapPack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAIwOwSuffArr2Patterns.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAIwOwSuffArr2Patterns.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALad.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALad.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALadConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALadConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALadLDB.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALadLDB.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALad_dist.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALad_dist.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALessA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALessA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALexBreaker.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FALexBreaker.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMapA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMapA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMapIOTools.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMapIOTools.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMap_judy.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMap_judy.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMap_std.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMap_std.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyDfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyDfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyDfaA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyDfaA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyDfa_ro.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyDfa_ro.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyNfa2Dfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyNfa2Dfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyNfaA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMealyNfaA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMergeDumps.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMergeDumps.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMergeMwe.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMergeMwe.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMergeSets.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMergeSets.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMsg.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMsg.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMapA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMapA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMapPack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMapPack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMapPack_fixed.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMapPack_fixed.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMapPack_mph.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMapPack_mph.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMap_ar.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMap_ar.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMap_ar_uniq.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMap_ar_uniq.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMap_judy.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAMultiMap_judy.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfa2Dfa_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfa2Dfa_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfa2EqPairs.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfa2EqPairs.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaCreator_base.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaCreator_base.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaCreator_char.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaCreator_char.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaCreator_digit.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaCreator_digit.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaCreator_wre.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaCreator_wre.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaDelta_ro.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaDelta_ro.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaDelta_wo_ro.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfaDelta_wo_ro.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfas2CommonENfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfas2CommonENfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfas2TupleNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfas2TupleNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfstLookupTools_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FANfstLookupTools_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAOffsetTablePack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAOffsetTablePack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParseTree.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParseTree.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParseTree_rbt.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParseTree_rbt.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParsedRegexp2TrBrMaps.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParsedRegexp2TrBrMaps.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParser2WRE.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParser2WRE.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParser_base_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParser_base_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParser_nest_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParser_nest_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParser_triv_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAParser_triv_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAPosNfaPack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAPosNfaPack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAPosNfa_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAPosNfa_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAPrintUtils.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAPrintUtils.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAPrmInterpreter_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAPrmInterpreter_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa2MooreDfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa2MooreDfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa2PerfHash.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa2PerfHash.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfaA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfaA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfaRenum_depth_first.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfaRenum_depth_first.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfaRenum_remove_gaps.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfaRenum_remove_gaps.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_ar_judy.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_ar_judy.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_dynamic_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_dynamic_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_renum.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_renum.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_renum_iws.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_renum_iws.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_ro.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_ro.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_wo_ro.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSDfa_wo_ro.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfa2RevNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfa2RevNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfaA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfaA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfa_ar_judy.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfa_ar_judy.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfa_ro.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfa_ro.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfa_wo_ro.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARSNfa_wo_ro.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexp2MinDfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexp2MinDfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexp2Nfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexp2Nfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpLexerA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpLexerA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpLexer_char.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpLexer_char.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpLexer_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpLexer_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpLexer_wre.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpLexer_wre.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpParser_msyacc.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpParser_msyacc.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpReverse.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpReverse.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpSimplify.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpSimplify.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree2Funcs.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree2Funcs.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree2Hash.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree2Hash.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree2PosEqs.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree2PosEqs.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree2Str.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTree2Str.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTreeSimplify_disj.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTreeSimplify_disj.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTreeSort.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTreeSort.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTreeTopoGraph.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARegexpTreeTopoGraph.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARemoveUnreachable.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FARemoveUnreachable.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAResolveMatch_nest.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAResolveMatch_nest.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASegmentationTools_bf_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASegmentationTools_bf_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASelectTrPatterns.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASelectTrPatterns.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASetUtils.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASetUtils.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASortMultiMap.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASortMultiMap.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASplitSets.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASplitSets.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASplitStates.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASplitStates.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2Ow.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2Ow.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2OwA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2OwA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2Ows.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2Ows.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2OwsA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2OwsA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2Ows_amb.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2Ows_amb.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2Ows_ar_uniq.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2Ows_ar_uniq.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2TrBr_pack_triv.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAState2TrBr_pack_triv.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStr2IntA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStr2IntA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStr2Int_hash.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStr2Int_hash.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStr2Utf16.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStr2Utf16.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStrList2MinDfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStrList2MinDfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStringArrayPack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStringArrayPack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStringTokenizer.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAStringTokenizer.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASubstInterpretTools_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASubstInterpretTools_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASubstRules2Regexp.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASubstRules2Regexp.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASuffixRules2Chains.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASuffixRules2Chains.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASummTagScores.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FASummTagScores.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATagSet.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATagSet.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATaggedText.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATaggedText.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATaggedTextA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATaggedTextA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATaggedTextStat.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATaggedTextStat.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestCmpDfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestCmpDfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestCmpMultiMap.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestCmpMultiMap.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestCmpPosNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestCmpPosNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_b2w.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_b2w.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_tag_dict.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_tag_dict.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_w2b.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_w2b.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_w2h.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_w2h.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_w2s.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_w2s.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_w2t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATestMorph_w2t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAToken.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAToken.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATokenNfa2Nfa_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATokenNfa2Nfa_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATokenNfa2TupleNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATokenNfa2TupleNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATopoSort_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATopoSort_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATrBr2Symbol.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATrBr2Symbol.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATrBrNfa2MealyNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATrBrNfa2MealyNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATrWordIOTools_utf8.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATrWordIOTools_utf8.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATransClosure_acyc_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATransClosure_acyc_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATypeMinDfa2MinMooreDfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATypeMinDfa2MinMooreDfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATypesNfaList2TypeNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FATypesNfaList2TypeNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAUnAmbiguous.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAUnAmbiguous.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAUtf16Utils.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAUtf16Utils.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAUtf32ToEnc.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAUtf32ToEnc.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAUtils.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAUtils.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAW2VInterpreter_t.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAW2VInterpreter_t.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRECompiler.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRECompiler.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREConf.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREConf.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREConfA.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREConfA.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREConfPack.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREConfPack.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREIO.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREIO.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRERules2TokenNfa.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRERules2TokenNfa.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREToken.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWREToken.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRETokenParser.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRETokenParser.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRETokens2Dicts.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRETokens2Dicts.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRETokens2Digitizers.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWRETokens2Digitizers.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWord2WordSuff.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWord2WordSuff.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWreRulesConfKeeper.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/FAWreRulesConfKeeper.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/blingfire-compile_src_pch.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfirecompile.library/inc/blingfire-compile_src_pch.h) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfiretools/blingfiretokdll/BlingFireTokLibSbdData.cxx](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfiretools/blingfiretokdll/BlingFireTokLibSbdData.cxx) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfiretools/blingfiretokdll/BlingFireTokLibWbdData.cxx](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfiretools/blingfiretokdll/BlingFireTokLibWbdData.cxx) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfiretools/blingfiretokdll/blingfiretokdll.cpp](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfiretools/blingfiretokdll/blingfiretokdll.cpp) | configuration/asset | W32 |
| [vendor/blingfire-18e9a19e5860/blingfiretools/blingfiretokdll/blingfiretokdll.h](../../rust/crates/lk-agents-media/vendor/blingfire-18e9a19e5860/blingfiretools/blingfiretokdll/blingfiretokdll.h) | configuration/asset | W32 |
| [vendor/blingfire.pin](../../rust/crates/lk-agents-media/vendor/blingfire.pin) | configuration/asset | W32 |
| [vendor/fetch.sh](../../rust/crates/lk-agents-media/vendor/fetch.sh) | gate/tool script | W32 |
| [vendor/ffmpeg.pin](../../rust/crates/lk-agents-media/vendor/ffmpeg.pin) | configuration/asset | W32 |

</details>

<details>
<summary>lk-agents-net — 29 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-net/Cargo.toml) | package manifest | W31 |
| [src/a2a_auth.rs](../../rust/crates/lk-agents-net/src/a2a_auth.rs) | Rust source | W35 |
| [src/a2a_client/authority.rs](../../rust/crates/lk-agents-net/src/a2a_client/authority.rs) | Rust source | W35 |
| [src/a2a_client/authority_tests.rs](../../rust/crates/lk-agents-net/src/a2a_client/authority_tests.rs) | test/support | W35 |
| [src/a2a_client/host.rs](../../rust/crates/lk-agents-net/src/a2a_client/host.rs) | Rust source | W35 |
| [src/a2a_client/remote.rs](../../rust/crates/lk-agents-net/src/a2a_client/remote.rs) | Rust source | W35 |
| [src/a2a_client.rs](../../rust/crates/lk-agents-net/src/a2a_client.rs) | Rust source | W35 |
| [src/a2a_client_tests.rs](../../rust/crates/lk-agents-net/src/a2a_client_tests.rs) | explicit test target | W35 |
| [src/blocking_io.rs](../../rust/crates/lk-agents-net/src/blocking_io.rs) | Rust source | W31 |
| [src/clock.rs](../../rust/crates/lk-agents-net/src/clock.rs) | Rust source | W31 |
| [src/connection_pool.rs](../../rust/crates/lk-agents-net/src/connection_pool.rs) | Rust source | W31 |
| [src/dns.rs](../../rust/crates/lk-agents-net/src/dns.rs) | Rust source | W31 |
| [src/env.rs](../../rust/crates/lk-agents-net/src/env.rs) | Rust source | W31 |
| [src/http/exchange.rs](../../rust/crates/lk-agents-net/src/http/exchange.rs) | Rust source | W31 |
| [src/http.rs](../../rust/crates/lk-agents-net/src/http.rs) | Rust source | W31 |
| [src/lib.rs](../../rust/crates/lk-agents-net/src/lib.rs) | Rust source | W31 |
| [src/session_report/admission.rs](../../rust/crates/lk-agents-net/src/session_report/admission.rs) | Rust source | W38 |
| [src/session_report/events.rs](../../rust/crates/lk-agents-net/src/session_report/events.rs) | Rust source | W38 |
| [src/session_report/tagger.rs](../../rust/crates/lk-agents-net/src/session_report/tagger.rs) | Rust source | W38 |
| [src/session_report/transcript.rs](../../rust/crates/lk-agents-net/src/session_report/transcript.rs) | Rust source | W38 |
| [src/session_report.rs](../../rust/crates/lk-agents-net/src/session_report.rs) | Rust source | W38 |
| [src/session_report_tests.rs](../../rust/crates/lk-agents-net/src/session_report_tests.rs) | test/support | W38 |
| [src/source.rs](../../rust/crates/lk-agents-net/src/source.rs) | Rust source | W31 |
| [src/sse.rs](../../rust/crates/lk-agents-net/src/sse.rs) | Rust source | W31 |
| [src/strings.rs](../../rust/crates/lk-agents-net/src/strings.rs) | Rust source | W31 |
| [src/task.rs](../../rust/crates/lk-agents-net/src/task.rs) | Rust source | W27 |
| [src/token.rs](../../rust/crates/lk-agents-net/src/token.rs) | Rust source | W31 |
| [src/transport.rs](../../rust/crates/lk-agents-net/src/transport.rs) | Rust source | W31 |
| [src/websocket.rs](../../rust/crates/lk-agents-net/src/websocket.rs) | Rust source | W31 |

</details>

<details>
<summary>lk-agents-py — 45 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-py/Cargo.toml) | package manifest | W39 |
| [build.rs](../../rust/crates/lk-agents-py/build.rs) | build support | W39 |
| [src/generated/a2a.rs](../../rust/crates/lk-agents-py/src/generated/a2a.rs) | generated | W39 |
| [src/generated/accessors.rs](../../rust/crates/lk-agents-py/src/generated/accessors.rs) | generated | W39 |
| [src/generated/agent.rs](../../rust/crates/lk-agents-py/src/generated/agent.rs) | generated | W39 |
| [src/generated/cli.rs](../../rust/crates/lk-agents-py/src/generated/cli.rs) | generated | W39 |
| [src/generated/core_names.json](../../rust/crates/lk-agents-py/src/generated/core_names.json) | generated | W39 |
| [src/generated/errors.rs](../../rust/crates/lk-agents-py/src/generated/errors.rs) | generated | W39 |
| [src/generated/evals.rs](../../rust/crates/lk-agents-py/src/generated/evals.rs) | generated | W39 |
| [src/generated/implementations.rs](../../rust/crates/lk-agents-py/src/generated/implementations.rs) | generated | W39 |
| [src/generated/inference.rs](../../rust/crates/lk-agents-py/src/generated/inference.rs) | generated | W39 |
| [src/generated/invoke.rs](../../rust/crates/lk-agents-py/src/generated/invoke.rs) | generated | W39 |
| [src/generated/llm.rs](../../rust/crates/lk-agents-py/src/generated/llm.rs) | generated | W39 |
| [src/generated/media.rs](../../rust/crates/lk-agents-py/src/generated/media.rs) | generated | W39 |
| [src/generated/metrics.rs](../../rust/crates/lk-agents-py/src/generated/metrics.rs) | generated | W39 |
| [src/generated/mod.rs](../../rust/crates/lk-agents-py/src/generated/mod.rs) | generated | W39 |
| [src/generated/native_enums.rs](../../rust/crates/lk-agents-py/src/generated/native_enums.rs) | generated | W39 |
| [src/generated/recorder.rs](../../rust/crates/lk-agents-py/src/generated/recorder.rs) | generated | W39 |
| [src/generated/records.rs](../../rust/crates/lk-agents-py/src/generated/records.rs) | generated | W39 |
| [src/generated/storage.rs](../../rust/crates/lk-agents-py/src/generated/storage.rs) | generated | W39 |
| [src/generated/stt.rs](../../rust/crates/lk-agents-py/src/generated/stt.rs) | generated | W39 |
| [src/generated/telemetry.rs](../../rust/crates/lk-agents-py/src/generated/telemetry.rs) | generated | W39 |
| [src/generated/text_session.rs](../../rust/crates/lk-agents-py/src/generated/text_session.rs) | generated | W39 |
| [src/generated/tokenize.rs](../../rust/crates/lk-agents-py/src/generated/tokenize.rs) | generated | W39 |
| [src/generated/tts.rs](../../rust/crates/lk-agents-py/src/generated/tts.rs) | generated | W39 |
| [src/generated/turn.rs](../../rust/crates/lk-agents-py/src/generated/turn.rs) | generated | W39 |
| [src/generated/types.rs](../../rust/crates/lk-agents-py/src/generated/types.rs) | generated | W39 |
| [src/generated/utils.rs](../../rust/crates/lk-agents-py/src/generated/utils.rs) | generated | W39 |
| [src/generated/vad.rs](../../rust/crates/lk-agents-py/src/generated/vad.rs) | generated | W39 |
| [src/generated/voice.rs](../../rust/crates/lk-agents-py/src/generated/voice.rs) | generated | W39 |
| [src/generated/worker.rs](../../rust/crates/lk-agents-py/src/generated/worker.rs) | generated | W39 |
| [src/lib.rs](../../rust/crates/lk-agents-py/src/lib.rs) | Rust source | W39 |
| [src/py_adapter_tests.rs](../../rust/crates/lk-agents-py/src/py_adapter_tests.rs) | explicit test target | W39 |
| [src/py_bindings_tests.py](../../rust/crates/lk-agents-py/src/py_bindings_tests.py) | test/support | W39 |
| [src/py_bindings_tests.rs](../../rust/crates/lk-agents-py/src/py_bindings_tests.rs) | explicit test target | W39 |
| [src/py_calls.rs](../../rust/crates/lk-agents-py/src/py_calls.rs) | Rust source | W39 |
| [src/py_conversion_tests.rs](../../rust/crates/lk-agents-py/src/py_conversion_tests.rs) | explicit test target | W39 |
| [src/py_errors.rs](../../rust/crates/lk-agents-py/src/py_errors.rs) | Rust source | W39 |
| [src/py_logging.rs](../../rust/crates/lk-agents-py/src/py_logging.rs) | Rust source | W39 |
| [src/py_objects.rs](../../rust/crates/lk-agents-py/src/py_objects.rs) | Rust source | W39 |
| [src/py_reference_tests.rs](../../rust/crates/lk-agents-py/src/py_reference_tests.rs) | explicit test target | W39 |
| [src/py_streams.rs](../../rust/crates/lk-agents-py/src/py_streams.rs) | Rust source | W39 |
| [src/py_test_fixtures.py](../../rust/crates/lk-agents-py/src/py_test_fixtures.py) | fixture/counterexample | W39 |
| [src/py_test_support.rs](../../rust/crates/lk-agents-py/src/py_test_support.rs) | Rust source | W39 |
| [src/py_values.rs](../../rust/crates/lk-agents-py/src/py_values.rs) | Rust source | W39 |

</details>

<details>
<summary>lk-agents-runtime — 123 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-runtime/Cargo.toml) | package manifest | W23 |
| [examples/media_bench.rs](../../rust/crates/lk-agents-runtime/examples/media_bench.rs) | Rust source | W32 |
| [fixtures/README.md](../../rust/crates/lk-agents-runtime/fixtures/README.md) | fixture/counterexample | W23 |
| [fixtures/mcp_server.py](../../rust/crates/lk-agents-runtime/fixtures/mcp_server.py) | fixture/counterexample | W23 |
| [src/a2a/attachments_tests.rs](../../rust/crates/lk-agents-runtime/src/a2a/attachments_tests.rs) | explicit test target | W35 |
| [src/a2a/door/admission.rs](../../rust/crates/lk-agents-runtime/src/a2a/door/admission.rs) | Rust source | W35 |
| [src/a2a/door/mod.rs](../../rust/crates/lk-agents-runtime/src/a2a/door/mod.rs) | Rust source | W35 |
| [src/a2a/door/stream.rs](../../rust/crates/lk-agents-runtime/src/a2a/door/stream.rs) | Rust source | W35 |
| [src/a2a/door_tests.rs](../../rust/crates/lk-agents-runtime/src/a2a/door_tests.rs) | explicit test target | W35 |
| [src/a2a/turn_handle.rs](../../rust/crates/lk-agents-runtime/src/a2a/turn_handle.rs) | Rust source | W35 |
| [src/a2a/turns_tests.rs](../../rust/crates/lk-agents-runtime/src/a2a/turns_tests.rs) | explicit test target | W35 |
| [src/a2a.rs](../../rust/crates/lk-agents-runtime/src/a2a.rs) | Rust source | W35 |
| [src/agent_run/execution.rs](../../rust/crates/lk-agents-runtime/src/agent_run/execution.rs) | Rust source | W34 |
| [src/agent_run/mod.rs](../../rust/crates/lk-agents-runtime/src/agent_run/mod.rs) | Rust source | W34 |
| [src/agent_run/resources.rs](../../rust/crates/lk-agents-runtime/src/agent_run/resources.rs) | Rust source | W34 |
| [src/agent_run/resources_tests.rs](../../rust/crates/lk-agents-runtime/src/agent_run/resources_tests.rs) | test/support | W34 |
| [src/agent_run/runner.rs](../../rust/crates/lk-agents-runtime/src/agent_run/runner.rs) | Rust source | W34 |
| [src/agent_run_tests.rs](../../rust/crates/lk-agents-runtime/src/agent_run_tests.rs) | test/support | W23 |
| [src/avatar/media.rs](../../rust/crates/lk-agents-runtime/src/avatar/media.rs) | Rust source | W28 |
| [src/avatar/mod.rs](../../rust/crates/lk-agents-runtime/src/avatar/mod.rs) | Rust source | W28 |
| [src/avatar/output.rs](../../rust/crates/lk-agents-runtime/src/avatar/output.rs) | Rust source | W28 |
| [src/avatar/queue.rs](../../rust/crates/lk-agents-runtime/src/avatar/queue.rs) | Rust source | W28 |
| [src/avatar/receiver.rs](../../rust/crates/lk-agents-runtime/src/avatar/receiver.rs) | Rust source | W28 |
| [src/avatar/runner.rs](../../rust/crates/lk-agents-runtime/src/avatar/runner.rs) | Rust source | W28 |
| [src/eot_tests.rs](../../rust/crates/lk-agents-runtime/src/eot_tests.rs) | explicit test target | W23 |
| [src/host/cancellation.rs](../../rust/crates/lk-agents-runtime/src/host/cancellation.rs) | Rust source | W39 |
| [src/host/mod.rs](../../rust/crates/lk-agents-runtime/src/host/mod.rs) | Rust source | W39 |
| [src/images_tests.rs](../../rust/crates/lk-agents-runtime/src/images_tests.rs) | explicit test target | W32 |
| [src/lib.rs](../../rust/crates/lk-agents-runtime/src/lib.rs) | Rust source | W23 |
| [src/logging.rs](../../rust/crates/lk-agents-runtime/src/logging.rs) | Rust source | W23 |
| [src/mcp/http.rs](../../rust/crates/lk-agents-runtime/src/mcp/http.rs) | Rust source | W35 |
| [src/mcp/http_tests.rs](../../rust/crates/lk-agents-runtime/src/mcp/http_tests.rs) | test/support | W35 |
| [src/mcp/lifecycle_tests.rs](../../rust/crates/lk-agents-runtime/src/mcp/lifecycle_tests.rs) | explicit test target | W35 |
| [src/mcp/mod.rs](../../rust/crates/lk-agents-runtime/src/mcp/mod.rs) | Rust source | W35 |
| [src/mcp/session.rs](../../rust/crates/lk-agents-runtime/src/mcp/session.rs) | Rust source | W35 |
| [src/mcp/transport.rs](../../rust/crates/lk-agents-runtime/src/mcp/transport.rs) | Rust source | W35 |
| [src/mcp_tests.rs](../../rust/crates/lk-agents-runtime/src/mcp_tests.rs) | test/support | W23 |
| [src/media_tests.rs](../../rust/crates/lk-agents-runtime/src/media_tests.rs) | explicit test target | W32 |
| [src/monotone.rs](../../rust/crates/lk-agents-runtime/src/monotone.rs) | Rust source | W23 |
| [src/process.rs](../../rust/crates/lk-agents-runtime/src/process.rs) | Rust source | W23 |
| [src/recording.rs](../../rust/crates/lk-agents-runtime/src/recording.rs) | Rust source | W18 |
| [src/recording_tests.rs](../../rust/crates/lk-agents-runtime/src/recording_tests.rs) | test/support | W23 |
| [src/room_io/bounded_lossy.rs](../../rust/crates/lk-agents-runtime/src/room_io/bounded_lossy.rs) | Rust source | W26 |
| [src/room_io/input.rs](../../rust/crates/lk-agents-runtime/src/room_io/input.rs) | Rust source | W09 |
| [src/room_io/mod.rs](../../rust/crates/lk-agents-runtime/src/room_io/mod.rs) | Rust source | W08 |
| [src/room_io/output/segment_ids.rs](../../rust/crates/lk-agents-runtime/src/room_io/output/segment_ids.rs) | Rust source | W11 |
| [src/room_io/output/transcription.rs](../../rust/crates/lk-agents-runtime/src/room_io/output/transcription.rs) | Rust source | W11 |
| [src/room_io/output.rs](../../rust/crates/lk-agents-runtime/src/room_io/output.rs) | Rust source | W10 |
| [src/room_io/owner.rs](../../rust/crates/lk-agents-runtime/src/room_io/owner.rs) | Rust source | W08 |
| [src/room_io/pre_connect_audio.rs](../../rust/crates/lk-agents-runtime/src/room_io/pre_connect_audio.rs) | Rust source | W12 |
| [src/rtc/events.rs](../../rust/crates/lk-agents-runtime/src/rtc/events.rs) | Rust source | W13 |
| [src/rtc/mod.rs](../../rust/crates/lk-agents-runtime/src/rtc/mod.rs) | Rust source | W13 |
| [src/rtc/playout.rs](../../rust/crates/lk-agents-runtime/src/rtc/playout.rs) | Rust source | W13 |
| [src/rtc/synchronizer.rs](../../rust/crates/lk-agents-runtime/src/rtc/synchronizer.rs) | Rust source | W13 |
| [src/rtc/synchronizer_tests.rs](../../rust/crates/lk-agents-runtime/src/rtc/synchronizer_tests.rs) | test/support | W13 |
| [src/rtc/wait.rs](../../rust/crates/lk-agents-runtime/src/rtc/wait.rs) | Rust source | W13 |
| [src/run_stream/cursor.rs](../../rust/crates/lk-agents-runtime/src/run_stream/cursor.rs) | Rust source | W34 |
| [src/run_stream/input.rs](../../rust/crates/lk-agents-runtime/src/run_stream/input.rs) | Rust source | W34 |
| [src/run_stream/mod.rs](../../rust/crates/lk-agents-runtime/src/run_stream/mod.rs) | Rust source | W34 |
| [src/run_stream_tests.rs](../../rust/crates/lk-agents-runtime/src/run_stream_tests.rs) | test/support | W23 |
| [src/tcp.rs](../../rust/crates/lk-agents-runtime/src/tcp.rs) | Rust source | W23 |
| [src/text_session/mod.rs](../../rust/crates/lk-agents-runtime/src/text_session/mod.rs) | Rust source | W34 |
| [src/text_session/runner.rs](../../rust/crates/lk-agents-runtime/src/text_session/runner.rs) | Rust source | W34 |
| [src/toolsets/group.rs](../../rust/crates/lk-agents-runtime/src/toolsets/group.rs) | Rust source | W27 |
| [src/toolsets/lifetime.rs](../../rust/crates/lk-agents-runtime/src/toolsets/lifetime.rs) | Rust source | W27 |
| [src/toolsets/mod.rs](../../rust/crates/lk-agents-runtime/src/toolsets/mod.rs) | Rust source | W27 |
| [src/toolsets/scopes.rs](../../rust/crates/lk-agents-runtime/src/toolsets/scopes.rs) | Rust source | W27 |
| [src/toolsets_tests.rs](../../rust/crates/lk-agents-runtime/src/toolsets_tests.rs) | test/support | W23 |
| [src/tts_fallback_tests.rs](../../rust/crates/lk-agents-runtime/src/tts_fallback_tests.rs) | test/support | W30 |
| [src/tts_gateway_tests.rs](../../rust/crates/lk-agents-runtime/src/tts_gateway_tests.rs) | test/support | W30 |
| [src/tts_live_tests.rs](../../rust/crates/lk-agents-runtime/src/tts_live_tests.rs) | explicit test target | W30 |
| [src/tts_sentence_tests.rs](../../rust/crates/lk-agents-runtime/src/tts_sentence_tests.rs) | test/support | W30 |
| [src/tts_tests.rs](../../rust/crates/lk-agents-runtime/src/tts_tests.rs) | explicit test target | W30 |
| [src/voice/amd.rs](../../rust/crates/lk-agents-runtime/src/voice/amd.rs) | Rust source | W21 |
| [src/voice/audio.rs](../../rust/crates/lk-agents-runtime/src/voice/audio.rs) | Rust source | W07 |
| [src/voice/background_audio.rs](../../rust/crates/lk-agents-runtime/src/voice/background_audio.rs) | Rust source | W21 |
| [src/voice/console.rs](../../rust/crates/lk-agents-runtime/src/voice/console.rs) | Rust source | W21 |
| [src/voice/effects.rs](../../rust/crates/lk-agents-runtime/src/voice/effects.rs) | Rust source | W25 |
| [src/voice/generation/forward.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/forward.rs) | Rust source | W19 |
| [src/voice/generation/llm.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/llm.rs) | Rust source | W19 |
| [src/voice/generation/mod.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/mod.rs) | Rust source | W19 |
| [src/voice/generation/output.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/output.rs) | Rust source | W19 |
| [src/voice/generation/pipeline_reply.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/pipeline_reply.rs) | Rust source | W19 |
| [src/voice/generation/playout.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/playout.rs) | Rust source | W19 |
| [src/voice/generation/realtime_generation.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/realtime_generation.rs) | Rust source | W19 |
| [src/voice/generation/tools.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/tools.rs) | Rust source | W19 |
| [src/voice/generation/tts.rs](../../rust/crates/lk-agents-runtime/src/voice/generation/tts.rs) | Rust source | W19 |
| [src/voice/ivr.rs](../../rust/crates/lk-agents-runtime/src/voice/ivr.rs) | Rust source | W21 |
| [src/voice/keyterm_detection.rs](../../rust/crates/lk-agents-runtime/src/voice/keyterm_detection.rs) | Rust source | W21 |
| [src/voice/legs.rs](../../rust/crates/lk-agents-runtime/src/voice/legs.rs) | Rust source | W07 |
| [src/voice/mod.rs](../../rust/crates/lk-agents-runtime/src/voice/mod.rs) | Rust source | W23 |
| [src/voice/nodes.rs](../../rust/crates/lk-agents-runtime/src/voice/nodes.rs) | Rust source | W07 |
| [src/voice/playback.rs](../../rust/crates/lk-agents-runtime/src/voice/playback.rs) | Rust source | W17 |
| [src/voice/recorder.rs](../../rust/crates/lk-agents-runtime/src/voice/recorder.rs) | Rust source | W18 |
| [src/voice/remote_session/client.rs](../../rust/crates/lk-agents-runtime/src/voice/remote_session/client.rs) | Rust source | W21 |
| [src/voice/remote_session/driver.rs](../../rust/crates/lk-agents-runtime/src/voice/remote_session/driver.rs) | Rust source | W21 |
| [src/voice/remote_session/host.rs](../../rust/crates/lk-agents-runtime/src/voice/remote_session/host.rs) | Rust source | W21 |
| [src/voice/remote_session/room.rs](../../rust/crates/lk-agents-runtime/src/voice/remote_session/room.rs) | Rust source | W21 |
| [src/voice/remote_session.rs](../../rust/crates/lk-agents-runtime/src/voice/remote_session.rs) | Rust source | W21 |
| [src/voice/resources.rs](../../rust/crates/lk-agents-runtime/src/voice/resources.rs) | Rust source | W24 |
| [src/voice/session/builder.rs](../../rust/crates/lk-agents-runtime/src/voice/session/builder.rs) | Rust source | W23 |
| [src/voice/session/calls.rs](../../rust/crates/lk-agents-runtime/src/voice/session/calls.rs) | Rust source | W24 |
| [src/voice/session/class.rs](../../rust/crates/lk-agents-runtime/src/voice/session/class.rs) | Rust source | W15 |
| [src/voice/session/mod.rs](../../rust/crates/lk-agents-runtime/src/voice/session/mod.rs) | Rust source | W23 |
| [src/voice/session/options.rs](../../rust/crates/lk-agents-runtime/src/voice/session/options.rs) | Rust source | W22 |
| [src/voice/session/owner.rs](../../rust/crates/lk-agents-runtime/src/voice/session/owner.rs) | Rust source | W23 |
| [src/voice/session/owner_tests.rs](../../rust/crates/lk-agents-runtime/src/voice/session/owner_tests.rs) | test/support | W23 |
| [src/voice/session/providers.rs](../../rust/crates/lk-agents-runtime/src/voice/session/providers.rs) | Rust source | W22 |
| [src/voice/session/shared.rs](../../rust/crates/lk-agents-runtime/src/voice/session/shared.rs) | Rust source | W24 |
| [src/voice/session/storage.rs](../../rust/crates/lk-agents-runtime/src/voice/session/storage.rs) | Rust source | W36 |
| [src/voice/session/time.rs](../../rust/crates/lk-agents-runtime/src/voice/session/time.rs) | Rust source | W23 |
| [src/voice/session/transcript_sync.rs](../../rust/crates/lk-agents-runtime/src/voice/session/transcript_sync.rs) | Rust source | W16 |
| [src/voice/session/transcript_sync_tests.rs](../../rust/crates/lk-agents-runtime/src/voice/session/transcript_sync_tests.rs) | test/support | W16 |
| [src/voice/session_hooks_tests.rs](../../rust/crates/lk-agents-runtime/src/voice/session_hooks_tests.rs) | test/support | W21 |
| [src/voice/session_playback_tests.rs](../../rust/crates/lk-agents-runtime/src/voice/session_playback_tests.rs) | test/support | W21 |
| [src/voice/session_tests.rs](../../rust/crates/lk-agents-runtime/src/voice/session_tests.rs) | test/support | W21 |
| [src/voice/speech_handle.rs](../../rust/crates/lk-agents-runtime/src/voice/speech_handle.rs) | Rust source | W17 |
| [src/voice/speech_handle_tests.rs](../../rust/crates/lk-agents-runtime/src/voice/speech_handle_tests.rs) | test/support | W17 |
| [src/voice/storage.rs](../../rust/crates/lk-agents-runtime/src/voice/storage.rs) | Rust source | W36 |
| [src/voice/workflows/dtmf.rs](../../rust/crates/lk-agents-runtime/src/voice/workflows/dtmf.rs) | Rust source | W21 |
| [src/voice/workflows/mod.rs](../../rust/crates/lk-agents-runtime/src/voice/workflows/mod.rs) | Rust source | W23 |
| [src/voice/workflows/warm_transfer.rs](../../rust/crates/lk-agents-runtime/src/voice/workflows/warm_transfer.rs) | Rust source | W21 |
| [src/voice_tests.rs](../../rust/crates/lk-agents-runtime/src/voice_tests.rs) | test/support | W23 |

</details>

<details>
<summary>lk-agents-storage — 36 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-storage/Cargo.toml) | package manifest | W36 |
| [src/agentdb/conn/results.rs](../../rust/crates/lk-agents-storage/src/agentdb/conn/results.rs) | Rust source | W36 |
| [src/agentdb/conn.rs](../../rust/crates/lk-agents-storage/src/agentdb/conn.rs) | Rust source | W36 |
| [src/agentdb/conn_tests.rs](../../rust/crates/lk-agents-storage/src/agentdb/conn_tests.rs) | test/support | W36 |
| [src/agentdb/mod.rs](../../rust/crates/lk-agents-storage/src/agentdb/mod.rs) | Rust source | W36 |
| [src/agentdb/streams/cursor.rs](../../rust/crates/lk-agents-storage/src/agentdb/streams/cursor.rs) | Rust source | W36 |
| [src/agentdb/streams.rs](../../rust/crates/lk-agents-storage/src/agentdb/streams.rs) | Rust source | W36 |
| [src/agentdb/streams_tests.rs](../../rust/crates/lk-agents-storage/src/agentdb/streams_tests.rs) | test/support | W36 |
| [src/cloud_tasks_tests.rs](../../rust/crates/lk-agents-storage/src/cloud_tasks_tests.rs) | explicit test target | W36 |
| [src/context/db.rs](../../rust/crates/lk-agents-storage/src/context/db.rs) | Rust source | W36 |
| [src/context/store.rs](../../rust/crates/lk-agents-storage/src/context/store.rs) | Rust source | W36 |
| [src/context/transaction.rs](../../rust/crates/lk-agents-storage/src/context/transaction.rs) | Rust source | W36 |
| [src/context.rs](../../rust/crates/lk-agents-storage/src/context.rs) | Rust source | W36 |
| [src/context_tests.rs](../../rust/crates/lk-agents-storage/src/context_tests.rs) | explicit test target | W36 |
| [src/db.rs](../../rust/crates/lk-agents-storage/src/db.rs) | Rust source | W36 |
| [src/engine.rs](../../rust/crates/lk-agents-storage/src/engine.rs) | Rust source | W36 |
| [src/handles.rs](../../rust/crates/lk-agents-storage/src/handles.rs) | Rust source | W36 |
| [src/lease.rs](../../rust/crates/lk-agents-storage/src/lease.rs) | Rust source | W36 |
| [src/lease_tests.rs](../../rust/crates/lk-agents-storage/src/lease_tests.rs) | explicit test target | W36 |
| [src/lib.rs](../../rust/crates/lk-agents-storage/src/lib.rs) | Rust source | W36 |
| [src/lifecycle_tests.rs](../../rust/crates/lk-agents-storage/src/lifecycle_tests.rs) | explicit test target | W36 |
| [src/memory_tests.rs](../../rust/crates/lk-agents-storage/src/memory_tests.rs) | explicit test target | W36 |
| [src/namespace.rs](../../rust/crates/lk-agents-storage/src/namespace.rs) | Rust source | W36 |
| [src/process_tests.rs](../../rust/crates/lk-agents-storage/src/process_tests.rs) | explicit test target | W36 |
| [src/run.rs](../../rust/crates/lk-agents-storage/src/run.rs) | Rust source | W36 |
| [src/runs_tests.rs](../../rust/crates/lk-agents-storage/src/runs_tests.rs) | explicit test target | W36 |
| [src/schema.rs](../../rust/crates/lk-agents-storage/src/schema.rs) | Rust source | W36 |
| [src/schema_tests.rs](../../rust/crates/lk-agents-storage/src/schema_tests.rs) | explicit test target | W36 |
| [src/sql/script.rs](../../rust/crates/lk-agents-storage/src/sql/script.rs) | Rust source | W36 |
| [src/sql.rs](../../rust/crates/lk-agents-storage/src/sql.rs) | Rust source | W36 |
| [src/sqlite/exec.rs](../../rust/crates/lk-agents-storage/src/sqlite/exec.rs) | Rust source | W36 |
| [src/sqlite.rs](../../rust/crates/lk-agents-storage/src/sqlite.rs) | Rust source | W36 |
| [src/streams.rs](../../rust/crates/lk-agents-storage/src/streams.rs) | Rust source | W36 |
| [src/testing.rs](../../rust/crates/lk-agents-storage/src/testing.rs) | Rust source | W36 |
| [src/voice.rs](../../rust/crates/lk-agents-storage/src/voice.rs) | Rust source | W36 |
| [src/voice_tests.rs](../../rust/crates/lk-agents-storage/src/voice_tests.rs) | explicit test target | W36 |

</details>

<details>
<summary>lk-agents-voice — 113 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-voice/Cargo.toml) | package manifest | W21 |
| [TEST_STRATEGY.md](../../rust/crates/lk-agents-voice/TEST_STRATEGY.md) | documentation | W21 |
| [src/agent.rs](../../rust/crates/lk-agents-voice/src/agent.rs) | Rust source | W21 |
| [src/agent_activity/drain.rs](../../rust/crates/lk-agents-voice/src/agent_activity/drain.rs) | Rust source | W04 |
| [src/agent_activity/experts.rs](../../rust/crates/lk-agents-voice/src/agent_activity/experts.rs) | Rust source | W04 |
| [src/agent_activity/false_interruption.rs](../../rust/crates/lk-agents-voice/src/agent_activity/false_interruption.rs) | Rust source | W05 |
| [src/agent_activity/handoff.rs](../../rust/crates/lk-agents-voice/src/agent_activity/handoff.rs) | Rust source | W04 |
| [src/agent_activity/hooks.rs](../../rust/crates/lk-agents-voice/src/agent_activity/hooks.rs) | Rust source | W04 |
| [src/agent_activity/idle.rs](../../rust/crates/lk-agents-voice/src/agent_activity/idle.rs) | Rust source | W04 |
| [src/agent_activity/io.rs](../../rust/crates/lk-agents-voice/src/agent_activity/io.rs) | Rust source | W07 |
| [src/agent_activity/mod.rs](../../rust/crates/lk-agents-voice/src/agent_activity/mod.rs) | Rust source | W04 |
| [src/agent_activity/options.rs](../../rust/crates/lk-agents-voice/src/agent_activity/options.rs) | Rust source | W22 |
| [src/agent_activity/options_tests.rs](../../rust/crates/lk-agents-voice/src/agent_activity/options_tests.rs) | test/support | W22 |
| [src/agent_activity/realtime/realtime_state.rs](../../rust/crates/lk-agents-voice/src/agent_activity/realtime/realtime_state.rs) | Rust source | W04 |
| [src/agent_activity/realtime.rs](../../rust/crates/lk-agents-voice/src/agent_activity/realtime.rs) | Rust source | W04 |
| [src/agent_activity/recognition.rs](../../rust/crates/lk-agents-voice/src/agent_activity/recognition.rs) | Rust source | W06 |
| [src/agent_activity/reply_run.rs](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run.rs) | Rust source | W19 |
| [src/agent_activity/scheduling.rs](../../rust/crates/lk-agents-voice/src/agent_activity/scheduling.rs) | Rust source | W05 |
| [src/agent_activity/speech.rs](../../rust/crates/lk-agents-voice/src/agent_activity/speech.rs) | Rust source | W05 |
| [src/agent_activity/tools.rs](../../rust/crates/lk-agents-voice/src/agent_activity/tools.rs) | Rust source | W20 |
| [src/agent_activity/voice_task.rs](../../rust/crates/lk-agents-voice/src/agent_activity/voice_task.rs) | Rust source | W04 |
| [src/agent_session/agent_user_state.rs](../../rust/crates/lk-agents-voice/src/agent_session/agent_user_state.rs) | Rust source | W03 |
| [src/agent_session/closing.rs](../../rust/crates/lk-agents-voice/src/agent_session/closing.rs) | Rust source | W03 |
| [src/agent_session/commands.rs](../../rust/crates/lk-agents-voice/src/agent_session/commands.rs) | Rust source | W25 |
| [src/agent_session/driver.rs](../../rust/crates/lk-agents-voice/src/agent_session/driver.rs) | Rust source | W25 |
| [src/agent_session/effects.rs](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs) | Rust source | W25 |
| [src/agent_session/host_events.rs](../../rust/crates/lk-agents-voice/src/agent_session/host_events.rs) | Rust source | W03 |
| [src/agent_session/initial.rs](../../rust/crates/lk-agents-voice/src/agent_session/initial.rs) | Rust source | W03 |
| [src/agent_session/lifecycle.rs](../../rust/crates/lk-agents-voice/src/agent_session/lifecycle.rs) | Rust source | W03 |
| [src/agent_session/mod.rs](../../rust/crates/lk-agents-voice/src/agent_session/mod.rs) | Rust source | W03 |
| [src/agent_session/nodes.rs](../../rust/crates/lk-agents-voice/src/agent_session/nodes.rs) | Rust source | W03 |
| [src/agent_session/output.rs](../../rust/crates/lk-agents-voice/src/agent_session/output.rs) | Rust source | W03 |
| [src/agent_session/storage.rs](../../rust/crates/lk-agents-voice/src/agent_session/storage.rs) | Rust source | W03 |
| [src/agent_session/time.rs](../../rust/crates/lk-agents-voice/src/agent_session/time.rs) | Rust source | W03 |
| [src/agent_session_tests.rs](../../rust/crates/lk-agents-voice/src/agent_session_tests.rs) | test/support | W21 |
| [src/amd/classifier.rs](../../rust/crates/lk-agents-voice/src/amd/classifier.rs) | Rust source | W21 |
| [src/amd/mod.rs](../../rust/crates/lk-agents-voice/src/amd/mod.rs) | Rust source | W21 |
| [src/amd_tests.rs](../../rust/crates/lk-agents-voice/src/amd_tests.rs) | explicit test target | W21 |
| [src/audio.rs](../../rust/crates/lk-agents-voice/src/audio.rs) | Rust source | W07 |
| [src/audio_recognition/end_of_turn.rs](../../rust/crates/lk-agents-voice/src/audio_recognition/end_of_turn.rs) | Rust source | W06 |
| [src/audio_recognition/mod.rs](../../rust/crates/lk-agents-voice/src/audio_recognition/mod.rs) | Rust source | W06 |
| [src/audio_recognition/overlap.rs](../../rust/crates/lk-agents-voice/src/audio_recognition/overlap.rs) | Rust source | W06 |
| [src/audio_recognition/speech.rs](../../rust/crates/lk-agents-voice/src/audio_recognition/speech.rs) | Rust source | W06 |
| [src/av_sync.rs](../../rust/crates/lk-agents-voice/src/av_sync.rs) | Rust source | W28 |
| [src/avatar/output.rs](../../rust/crates/lk-agents-voice/src/avatar/output.rs) | Rust source | W28 |
| [src/avatar/queue.rs](../../rust/crates/lk-agents-voice/src/avatar/queue.rs) | Rust source | W28 |
| [src/avatar/receiver.rs](../../rust/crates/lk-agents-voice/src/avatar/receiver.rs) | Rust source | W28 |
| [src/avatar.rs](../../rust/crates/lk-agents-voice/src/avatar.rs) | Rust source | W28 |
| [src/avatar_runner.rs](../../rust/crates/lk-agents-voice/src/avatar_runner.rs) | Rust source | W28 |
| [src/avatar_runner_tests.rs](../../rust/crates/lk-agents-voice/src/avatar_runner_tests.rs) | test/support | W21 |
| [src/background_audio.rs](../../rust/crates/lk-agents-voice/src/background_audio.rs) | Rust source | W21 |
| [src/data_plane/mod.rs](../../rust/crates/lk-agents-voice/src/data_plane/mod.rs) | Rust source | W07 |
| [src/endpointing.rs](../../rust/crates/lk-agents-voice/src/endpointing.rs) | Rust source | W06 |
| [src/events.rs](../../rust/crates/lk-agents-voice/src/events.rs) | Rust source | W21 |
| [src/experts.rs](../../rust/crates/lk-agents-voice/src/experts.rs) | Rust source | W21 |
| [src/experts_tests.rs](../../rust/crates/lk-agents-voice/src/experts_tests.rs) | test/support | W21 |
| [src/filler_scheduler.rs](../../rust/crates/lk-agents-voice/src/filler_scheduler.rs) | Rust source | W05 |
| [src/generation/forward.rs](../../rust/crates/lk-agents-voice/src/generation/forward.rs) | Rust source | W19 |
| [src/generation/llm.rs](../../rust/crates/lk-agents-voice/src/generation/llm.rs) | Rust source | W19 |
| [src/generation/mod.rs](../../rust/crates/lk-agents-voice/src/generation/mod.rs) | Rust source | W19 |
| [src/generation/pipeline.rs](../../rust/crates/lk-agents-voice/src/generation/pipeline.rs) | Rust source | W19 |
| [src/generation/playout.rs](../../rust/crates/lk-agents-voice/src/generation/playout.rs) | Rust source | W19 |
| [src/generation/realtime.rs](../../rust/crates/lk-agents-voice/src/generation/realtime.rs) | Rust source | W19 |
| [src/generation/tts.rs](../../rust/crates/lk-agents-voice/src/generation/tts.rs) | Rust source | W19 |
| [src/ivr.rs](../../rust/crates/lk-agents-voice/src/ivr.rs) | Rust source | W21 |
| [src/ivr_tests.rs](../../rust/crates/lk-agents-voice/src/ivr_tests.rs) | explicit test target | W21 |
| [src/keyterm_detection.rs](../../rust/crates/lk-agents-voice/src/keyterm_detection.rs) | Rust source | W21 |
| [src/lib.rs](../../rust/crates/lk-agents-voice/src/lib.rs) | Rust source | W21 |
| [src/playback.rs](../../rust/crates/lk-agents-voice/src/playback.rs) | Rust source | W17 |
| [src/provider_layers.rs](../../rust/crates/lk-agents-voice/src/provider_layers.rs) | Rust source | W22 |
| [src/recorder_io.rs](../../rust/crates/lk-agents-voice/src/recorder_io.rs) | Rust source | W18 |
| [src/remote_session/console.rs](../../rust/crates/lk-agents-voice/src/remote_session/console.rs) | Rust source | W21 |
| [src/remote_session.rs](../../rust/crates/lk-agents-voice/src/remote_session.rs) | Rust source | W21 |
| [src/remote_session_tests.rs](../../rust/crates/lk-agents-voice/src/remote_session_tests.rs) | test/support | W21 |
| [src/room_io/coordinator.rs](../../rust/crates/lk-agents-voice/src/room_io/coordinator.rs) | Rust source | W08 |
| [src/room_io/detached.rs](../../rust/crates/lk-agents-voice/src/room_io/detached.rs) | Rust source | W14 |
| [src/room_io/ingress.rs](../../rust/crates/lk-agents-voice/src/room_io/ingress.rs) | Rust source | W13 |
| [src/room_io/input_state.rs](../../rust/crates/lk-agents-voice/src/room_io/input_state.rs) | Rust source | W09 |
| [src/room_io/mod.rs](../../rust/crates/lk-agents-voice/src/room_io/mod.rs) | Rust source | W08 |
| [src/room_io/output.rs](../../rust/crates/lk-agents-voice/src/room_io/output.rs) | Rust source | W10 |
| [src/room_io/playout.rs](../../rust/crates/lk-agents-voice/src/room_io/playout.rs) | Rust source | W17 |
| [src/room_io/pre_connect_state.rs](../../rust/crates/lk-agents-voice/src/room_io/pre_connect_state.rs) | Rust source | W12 |
| [src/room_io/publication.rs](../../rust/crates/lk-agents-voice/src/room_io/publication.rs) | Rust source | W10 |
| [src/room_io/transcription.rs](../../rust/crates/lk-agents-voice/src/room_io/transcription.rs) | Rust source | W11 |
| [src/room_io/video_sampler.rs](../../rust/crates/lk-agents-voice/src/room_io/video_sampler.rs) | Rust source | W07 |
| [src/room_io/wait.rs](../../rust/crates/lk-agents-voice/src/room_io/wait.rs) | Rust source | W13 |
| [src/run_result.rs](../../rust/crates/lk-agents-voice/src/run_result.rs) | Rust source | W34 |
| [src/run_view.rs](../../rust/crates/lk-agents-voice/src/run_view.rs) | Rust source | W34 |
| [src/session_start.rs](../../rust/crates/lk-agents-voice/src/session_start.rs) | Rust source | W23 |
| [src/speech_handle.rs](../../rust/crates/lk-agents-voice/src/speech_handle.rs) | Rust source | W17 |
| [src/speech_handle_tests.rs](../../rust/crates/lk-agents-voice/src/speech_handle_tests.rs) | test/support | W21 |
| [src/telemetry.rs](../../rust/crates/lk-agents-voice/src/telemetry.rs) | Rust source | W38 |
| [src/tokenize.rs](../../rust/crates/lk-agents-voice/src/tokenize.rs) | Rust source | W33 |
| [src/tool_body.rs](../../rust/crates/lk-agents-voice/src/tool_body.rs) | Rust source | W20 |
| [src/tool_executor.rs](../../rust/crates/lk-agents-voice/src/tool_executor.rs) | Rust source | W20 |
| [src/trace.rs](../../rust/crates/lk-agents-voice/src/trace.rs) | Rust source | W38 |
| [src/transcript_tests.rs](../../rust/crates/lk-agents-voice/src/transcript_tests.rs) | explicit test target | W21 |
| [src/transcription/filters.rs](../../rust/crates/lk-agents-voice/src/transcription/filters.rs) | Rust source | W16 |
| [src/transcription/liang/exceptions.txt](../../rust/crates/lk-agents-voice/src/transcription/liang/exceptions.txt) | configuration/asset | W16 |
| [src/transcription/liang/patterns.txt](../../rust/crates/lk-agents-voice/src/transcription/liang/patterns.txt) | configuration/asset | W16 |
| [src/transcription/lifecycle.rs](../../rust/crates/lk-agents-voice/src/transcription/lifecycle.rs) | Rust source | W16 |
| [src/transcription/mod.rs](../../rust/crates/lk-agents-voice/src/transcription/mod.rs) | Rust source | W16 |
| [src/transcription/pacer.rs](../../rust/crates/lk-agents-voice/src/transcription/pacer.rs) | Rust source | W16 |
| [src/transcription/pacer_tests.rs](../../rust/crates/lk-agents-voice/src/transcription/pacer_tests.rs) | test/support | W16 |
| [src/transcription/synchronizer.rs](../../rust/crates/lk-agents-voice/src/transcription/synchronizer.rs) | Rust source | W16 |
| [src/turn.rs](../../rust/crates/lk-agents-voice/src/turn.rs) | Rust source | W22 |
| [src/workflows/dtmf.rs](../../rust/crates/lk-agents-voice/src/workflows/dtmf.rs) | Rust source | W21 |
| [src/workflows/end_call.rs](../../rust/crates/lk-agents-voice/src/workflows/end_call.rs) | Rust source | W21 |
| [src/workflows/instructions.rs](../../rust/crates/lk-agents-voice/src/workflows/instructions.rs) | Rust source | W21 |
| [src/workflows/mod.rs](../../rust/crates/lk-agents-voice/src/workflows/mod.rs) | Rust source | W21 |
| [src/workflows/prompts.rs](../../rust/crates/lk-agents-voice/src/workflows/prompts.rs) | Rust source | W21 |
| [src/workflows/twilio.rs](../../rust/crates/lk-agents-voice/src/workflows/twilio.rs) | Rust source | W21 |
| [src/workflows/warm_transfer.rs](../../rust/crates/lk-agents-voice/src/workflows/warm_transfer.rs) | Rust source | W21 |

</details>

<details>
<summary>lk-agents-worker — 38 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [Cargo.toml](../../rust/crates/lk-agents-worker/Cargo.toml) | package manifest | W37 |
| [src/config.rs](../../rust/crates/lk-agents-worker/src/config.rs) | Rust source | W37 |
| [src/http.rs](../../rust/crates/lk-agents-worker/src/http.rs) | Rust source | W37 |
| [src/hw/cgroup.rs](../../rust/crates/lk-agents-worker/src/hw/cgroup.rs) | Rust source | W37 |
| [src/hw/cpu.rs](../../rust/crates/lk-agents-worker/src/hw/cpu.rs) | Rust source | W37 |
| [src/hw/mod.rs](../../rust/crates/lk-agents-worker/src/hw/mod.rs) | Rust source | W37 |
| [src/hw/paths.rs](../../rust/crates/lk-agents-worker/src/hw/paths.rs) | Rust source | W37 |
| [src/ipc/child.rs](../../rust/crates/lk-agents-worker/src/ipc/child.rs) | Rust source | W37 |
| [src/ipc/executor.rs](../../rust/crates/lk-agents-worker/src/ipc/executor.rs) | Rust source | W37 |
| [src/ipc/ipc_channel.rs](../../rust/crates/lk-agents-worker/src/ipc/ipc_channel.rs) | Rust source | W37 |
| [src/ipc/ipc_channel_tests.rs](../../rust/crates/lk-agents-worker/src/ipc/ipc_channel_tests.rs) | explicit test target | W37 |
| [src/ipc/ipc_tests.rs](../../rust/crates/lk-agents-worker/src/ipc/ipc_tests.rs) | test/support | W37 |
| [src/ipc/job_run.rs](../../rust/crates/lk-agents-worker/src/ipc/job_run.rs) | Rust source | W37 |
| [src/ipc/mod.rs](../../rust/crates/lk-agents-worker/src/ipc/mod.rs) | Rust source | W37 |
| [src/ipc/owner.rs](../../rust/crates/lk-agents-worker/src/ipc/owner.rs) | Rust source | W37 |
| [src/ipc/pool.rs](../../rust/crates/lk-agents-worker/src/ipc/pool.rs) | Rust source | W37 |
| [src/ipc/pool_tests.rs](../../rust/crates/lk-agents-worker/src/ipc/pool_tests.rs) | test/support | W37 |
| [src/ipc/process.rs](../../rust/crates/lk-agents-worker/src/ipc/process.rs) | Rust source | W37 |
| [src/ipc/process_pool.rs](../../rust/crates/lk-agents-worker/src/ipc/process_pool.rs) | Rust source | W37 |
| [src/ipc/supervisor.rs](../../rust/crates/lk-agents-worker/src/ipc/supervisor.rs) | Rust source | W37 |
| [src/ipc/wire.rs](../../rust/crates/lk-agents-worker/src/ipc/wire.rs) | Rust source | W37 |
| [src/jobs.rs](../../rust/crates/lk-agents-worker/src/jobs.rs) | Rust source | W37 |
| [src/lib.rs](../../rust/crates/lk-agents-worker/src/lib.rs) | Rust source | W37 |
| [src/load.rs](../../rust/crates/lk-agents-worker/src/load.rs) | Rust source | W37 |
| [src/load_driver.rs](../../rust/crates/lk-agents-worker/src/load_driver.rs) | Rust source | W37 |
| [src/server/health.rs](../../rust/crates/lk-agents-worker/src/server/health.rs) | Rust source | W37 |
| [src/server/host.rs](../../rust/crates/lk-agents-worker/src/server/host.rs) | Rust source | W37 |
| [src/server/http.rs](../../rust/crates/lk-agents-worker/src/server/http.rs) | Rust source | W37 |
| [src/server/lifecycle_tests.rs](../../rust/crates/lk-agents-worker/src/server/lifecycle_tests.rs) | test/support | W37 |
| [src/server/settings.rs](../../rust/crates/lk-agents-worker/src/server/settings.rs) | Rust source | W37 |
| [src/server.rs](../../rust/crates/lk-agents-worker/src/server.rs) | Rust source | W37 |
| [src/server_tests.rs](../../rust/crates/lk-agents-worker/src/server_tests.rs) | test/support | W37 |
| [src/transport/mod.rs](../../rust/crates/lk-agents-worker/src/transport/mod.rs) | Rust source | W37 |
| [src/transport/webtransport.rs](../../rust/crates/lk-agents-worker/src/transport/webtransport.rs) | Rust source | W37 |
| [src/transport/ws.rs](../../rust/crates/lk-agents-worker/src/transport/ws.rs) | Rust source | W37 |
| [src/worker.rs](../../rust/crates/lk-agents-worker/src/worker.rs) | Rust source | W37 |
| [src/worker_test_fixtures.rs](../../rust/crates/lk-agents-worker/src/worker_test_fixtures.rs) | fixture/counterexample | W37 |
| [src/worker_tests.rs](../../rust/crates/lk-agents-worker/src/worker_tests.rs) | test/support | W37 |

</details>

<details>
<summary>workspace and architecture — 27 files</summary>

| Current file | Category | Primary workstream |
| --- | --- | --- |
| [.cargo/mutants.toml](../../rust/.cargo/mutants.toml) | configuration/asset | W41 |
| [.config/nextest.toml](../../rust/.config/nextest.toml) | configuration/asset | W41 |
| [Cargo.lock](../../rust/Cargo.lock) | dependency lock | W41 |
| [Cargo.toml](../../rust/Cargo.toml) | package manifest | W41 |
| [Makefile](../../rust/Makefile) | configuration/asset | W41 |
| [architecture/README.md](../../rust/architecture/README.md) | documentation | W41 |
| [architecture/check.py](../../rust/architecture/check.py) | Python host/test/tool source | W41 |
| [architecture/check.sh](../../rust/architecture/check.sh) | gate/tool script | W41 |
| [architecture/clippy-baseline.json](../../rust/architecture/clippy-baseline.json) | configuration/asset | W41 |
| [architecture/clippy.toml](../../rust/architecture/clippy.toml) | configuration/asset | W41 |
| [architecture/dependencies-baseline.json](../../rust/architecture/dependencies-baseline.json) | configuration/asset | W41 |
| [architecture/dependencies-sqlite-baseline.json](../../rust/architecture/dependencies-sqlite-baseline.json) | configuration/asset | W41 |
| [architecture/executors-baseline.json](../../rust/architecture/executors-baseline.json) | configuration/asset | W41 |
| [architecture/fixtures.rs](../../rust/architecture/fixtures.rs) | fixture/counterexample | W41 |
| [architecture/host-calls-baseline.json](../../rust/architecture/host-calls-baseline.json) | configuration/asset | W41 |
| [architecture/io/clippy-baseline.json](../../rust/architecture/io/clippy-baseline.json) | configuration/asset | W41 |
| [architecture/io/clippy.toml](../../rust/architecture/io/clippy.toml) | configuration/asset | W41 |
| [architecture/io/fixtures.rs](../../rust/architecture/io/fixtures.rs) | fixture/counterexample | W41 |
| [architecture/lengths-baseline.json](../../rust/architecture/lengths-baseline.json) | configuration/asset | W41 |
| [architecture/paths-baseline.json](../../rust/architecture/paths-baseline.json) | configuration/asset | W41 |
| [architecture/pyo3-baseline.json](../../rust/architecture/pyo3-baseline.json) | configuration/asset | W41 |
| [architecture/test_check.py](../../rust/architecture/test_check.py) | test/support | W41 |
| [architecture/test_clippy.py](../../rust/architecture/test_clippy.py) | test/support | W41 |
| [architecture/uses-baseline.json](../../rust/architecture/uses-baseline.json) | configuration/asset | W41 |
| [backend-contracts.sh](../../rust/backend-contracts.sh) | gate/tool script | W41 |
| [bench-budgets.json](../../rust/bench-budgets.json) | configuration/asset | W41 |
| [rustfmt.toml](../../rust/rustfmt.toml) | configuration/asset | W41 |

</details>

