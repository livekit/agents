# Rust transition simplification inventory

Rust should preserve the concepts and names that a LiveKit Agents contributor already knows.
The goal is fewer primitives to learn: familiar domain state, ordinary methods, and runtime
effects. Remove intermediate representations and implicit bookkeeping before choosing names
for anything that remains. More specific names improve discovery but do not reduce the number
of concepts a contributor must understand.

This inventory covers the complete session transition path: component output, collection,
timer changes, internal signals, settlement, journaling, and runtime effects. It also covers
the related generation patterns and identifies boundaries to retain. It records the
implemented simplifications and their rationale, verified on 2026-11-01. Public behavior
continues to be defined by the [feature documents](../README.md); architectural requirements
remain in [Rust core](rust-core.md).

The companion [fuzz crate audit](rust-fuzz-inventory.md) covers the simulation harness,
contributor navigation, and verification infrastructure used to validate these changes.
The [lifecycle symmetry audit](rust-lifecycle-symmetry-audit.md) inventories hook, event,
effect, request/receipt, and terminal vocabulary across Rust and Python. Its 41 entries
distinguish useful normalization from lifecycle differences to preserve. Enter and exit
now share terminal acceptance and completion dispatch. Redundant `OnExitEnded` notification
plumbing is removed; no unused `OnEnterEnded` mirror was added. Preserve deferred completion
after dependent state is installed: inline no-host exit completion can strand a closing session.
The [shared policy audit](rust-shared-policy-audit.md) traces similar discrepancies across
agent outcomes, history, tool metadata, validation, retries, and cancellation. It also compares
OpenAI Agents/PydanticAI terminology and proposes `HandleToolResults` / `ToolResultsHandled`
with `AgentRunContinuation` as clearer names for the current coordinated step handshake.
Those names and the shared-run policy corrections are now implemented; see the companion
audit implementation checkpoint. Collector/scheduling and voice-run completion redesign are
implemented too; the final checkpoint records combined validation and its storage isolation.

## Plan audit checkpoint

The agreed architecture cleanup is implemented and validated; the
[final checkpoint](#behavioral-pass-checkpoint-verified) records the exact gates and isolated
storage baseline. The only deferred feature work is integration of PR #8256 and its stack
prerequisites. The three previously open internal contracts now have explicit implementations,
without adding question orchestration.

| Area | Current disposition | Acceptance evidence |
| --- | --- | --- |
| Shared-run reuse, canonical history/edits, duplicate policy, cleanup and names | Implemented | Shared-policy regressions; preserve canonical model order and operation correlation |
| Lossless input requirements | Implemented on successful results and failed-run envelopes/exceptions; singular status/question remains a compatibility projection | Preserve every requirement, including mixed terminal batches; no question queue or answer-routing policy added |
| Private outbox and automatic scheduling | Implemented; `Out`, mixed `Effects`, `Transition`, `Planned`, `StepOp` and manual `pump_schedule` removed | One private driver outbox, FIFO callbacks and automatic complete-transition advancement; final combined gates passed on the isolated storage baseline |
| Common hook completion | Implemented through the existing hook owner; exit continuation deferred | No-host, single terminal claim, abandonment and continuation-order regressions |
| Audio playback, transcript, output-binding and preparation grouping | Implemented | Preserve receipt bounds, heard-audio accounting, alignment, physical lease cleanup and reusable authorization |
| Room transcription and TTS emitter naming | Implemented | Preserve publication/wire compatibility, PCM flush and ordered delivery |
| Domain `machine` modules/category effect files | Implemented | Actual effect inventory below; previous structural gates passed, combined checks repeated for this pass |
| Scoped owner/event/result names | Implemented | `VoiceDriver`, `VoiceRuntime`, `VoiceSessionState`, `VoiceEvent`, `VoiceHostEvent`, `VoiceRuntimeEvent`, `AgentRunEvent`, `TextSessionEvent` and user-turn commit types |
| Close and scheduling symmetry | Implemented | `advance_close` owns progression; `resume_scheduling` matches pause; `SchedulingDrainFinished` is a real drain receipt; no `CloseProgress` or close-only pause callback |
| Timer API | Implemented | `set_deadline`, `clear_deadline`, observational `deadline`; current-transition changes suppress stale due selections |
| Voice-run completion | Implemented through lifecycle owners | Causal membership retained on speech/hooks/handoffs/holds; no `RunOwn` or duplicate work registry |
| `VoiceWorkTracker` / `VoiceWork` | Removed from the target design | Existing owner records answer liveness; a separate ledger was not justified |
| Small generation protocols | Implemented G01–G03 | `SpeechSegmentEvent` shared directly; first-output boolean and ordinary text-leg operations replace redundant enums |
| Type removals and DST cleanup | Implemented; companion inventories record verification and retain decisions | Voice/room over fakes only; audio/STT checks and contribution guide delivered |
| PR #8256 delegation/question behavior | Deferred, separate phase | Reconcile existing expert inbox, PR prerequisites and relay/claim/read-back contract after cleanup |

The resolved contracts are:

1. **Input requirements:** shared `AgentRun` retains all correlated requirements as result
   metadata: invocation identity, pending call, question and verbatim flag. Compatibility
   `status`/`question` still projects one terminal result, but does not discard the remaining
   requests. Failed runs also retain requirements in their terminal envelope and raised
   exception; a failed batch does not mark any question as compatibility-selected. Metadata
   is supplied-history scoped so reconstruction preserves it. Generic core
   does not choose question presentation, priority or answer routing; that remains outside this
   invocation loop and the colleague's PR remains deferred.
2. **Run membership and result:** lifecycle records carry membership at admission. Session-
   requested speech, handoffs and on-enter consequences participate while the run is open;
   incidental server turns/filler do not. A foreground hold inherits membership only from its
   attributed tool speech. Capture mode waits for initial admission. The last completed
   attributed speech supplies the selected result, preserving established failure precedence;
   unselected values are released and close clears every retained value. Completion queries
   existing owners rather than maintaining a second pending-work set.
3. **Deadline/event ties:** preserve deadline-first processing at equal time. Production runtime
   services an already-due wake before a supplied event; shared-run cleanup follows that same
   rule. A receipt queued earlier but processed at its cleanup bound encounters cleanup first.
   Current-transition set/clear operations are immediately visible to queries and suppress
   stale due-timer selections until the coordinator crosses its boundary. These are internal
   timing contracts, not a new public timer or question API.

“Outside core” means outside generic AgentRun policy. Voice conversation decisions remain
synchronous in `lk-agents-voice`; runtime and transports execute I/O.

### Final plan audit and next execution boundary

The accepted layout and behavioral reductions are present. No global effect hierarchy,
replacement shared collector, queue lookahead, per-handler dirty flag, or manual pumping
obligation was introduced. `PendingInputOperation` remains private solely to resolve flush
sink/overlap identity after the atomic transition installs it, preserving relative outbox order.
Domain owners instead return their actual outcomes: recognition actions/observations, drain
milestones, DTMF outcomes and output-binding operations. They do not receive the driver's outbox.

The combined gates are recorded in the final checkpoint. Historical counts in the structural
checkpoint remain historical. The shared working tree also contains a separately owned storage
refactor, so its whole-workspace integration gate must be repeated when that task settles.
PR #8256 is the only deferred implementation phase in this plan.

### Inventory completion and retained representations

Stable C/T/S/O/G IDs below preserve the original inventory and rationale. Old names in the
before/replacement tables identify the removed source, not outstanding work. C01–C10,
T01–T07, S01–S07 and O01–O10 are addressed by the private outbox, observational deadline API,
scoped callbacks/observations, ordinary component outcomes and explicit settlement phases.
G01–G05 and G12–G15 are implemented grouping/protocol changes. G06–G11 retain necessary
ordered generation operations under explicit owners; the live declaration map below names
them. Private playback child results prevent duplicated sink/cut/finish policy; they are not
additional public protocols for callers to learn. No conditional investigation remains an
untracked implementation task: retain decisions state the invariant they protect.

## Follow Python concepts

Use the existing Python domain vocabulary and responsibility boundaries as the reference.
Follow the specified v2 API when it differs from historical Python behavior. Historical
private Python names in Rust comments explain provenance; they are not additional contracts.

| Familiar concept | Current Rust representation | Direction |
| --- | --- | --- |
| Running agent | `AgentActivity` | Preserve the name and its component ownership |
| Speech queue and current speech | `SpeechScheduler` | Keep scheduling methods recognizable, such as `schedule_speech` and `pause_scheduling` |
| Recognition and user turn handling | `AudioRecognition` | Preserve recognition and end-of-turn vocabulary |
| Speech lifecycle | Core speech state and runtime `SpeechHandle` | Keep the shared speech identity and explain the representation boundary |
| `say`, `generate_reply`, interruption | Session commands and corresponding handlers | Preserve operation names through the implementation |
| Direct calls and callbacks between components | Methods, returned outcomes, and internal signals | Use ordinary methods and outcomes where deferred coordination adds no value |
| Async work | Runtime execution of core effects | Keep the core/runtime split explicit |

`Out`, `Transition`, `Planned` and `StepOp` were implementation mechanisms and are now
removed. Renaming them would have left the same machinery to learn. The rejected `StepOutput`,
`ComponentOutput` and `RecognitionDecision` replacements were not introduced. The tables below
retain before/after rationale for the implemented domain methods and private driver outbox;
final combined verification is tracked by the checkpoint above.

Apply these rules in order:

1. Preserve an existing domain name when the responsibility is the same.
2. Prefer ordinary methods and typed returns for local synchronous coordination.
3. Use internal signals when deferred ordering or coordination requires them.
4. Use effects when work crosses the core/runtime boundary.
5. Keep collectors, buffers, journal conversion, and settlement details private.

Do not move decisions into async runtime code to make the synchronous core smaller. Do not
introduce a global transition framework or a common effect enum for unrelated components.

## Target structure with fewer primitives

The contributor's model has three parts:

| Part | Meaning | Familiar examples |
| --- | --- | --- |
| State | Domain objects own their data and invariants | `VoiceSession`, `AgentActivity`, `SpeechHandle`, recognition, scheduling |
| Methods | Ordinary calls update state and return domain outcomes | `schedule_speech`, `interrupt_speech`, `on_end_of_turn` |
| Effects | Requests for work performed outside the synchronous core | Model calls, playback operations, host calls, publications |

Domain components expose ordinary methods and return meaningful domain values, `Option`, or
`Result` where needed. `VoiceDriver` coordinates those outcomes and privately records runtime
requests. Components do not receive a mutable `voice_effects` object as their default API.
A meaningful domain result is useful; a shared mixed-output framework is not required.

### Implemented replacement and ownership

`agent_session/machine/coordination.rs` now owns the driver's private callback dispatch,
input identity resolution and automatic advancement. `agent_session/effects.rs` contains
`VoiceEffect` declarations and payload records only. `Out`, mixed `Effects`, `Transition`,
`Planned` and `StepOp` are removed, rather than merely relocated.

Keep effect storage and deferred coordination inside `VoiceDriver`. A domain operation such
as `interrupt_speech` updates the appropriate domain state and privately records the runtime
requests that follow. The core returns `Vec<VoiceEffect>` to `VoiceRuntime` only after the
current transition and its required follow-ups settle.

The proposed shared `VoiceEffects` collector is withdrawn. Passing it through every component
would reduce wrapper types while spreading a dependency on transition machinery. The target
is a smaller domain API, not a shorter route for every component to manipulate an effect list.

| Existing type | Proposed replacement | Owner and purpose | Justification |
| --- | --- | --- | --- |
| `Out` and generic mixed `Effects` | Remove from domain method signatures; keep outbox storage private to `VoiceDriver` | The driver translates domain outcomes into runtime requests | Components need not learn or mutate a shared collector |
| `Effect` in session transitions | `VoiceEffect` | Produced by `VoiceDriver`; performed by `VoiceRuntime` | Names the actual execution boundary and stays clear when imported |
| `Transition` | Remove where ordinary calls and scoped outcomes replace its mixed component-output entries | Each component owns its state; the driver coordinates outcomes | No mandatory generic transition protocol between every component and its parent |
| `Planned` and `StepOp` | Remove from component APIs; retain only necessary ordered bookkeeping inside the driver | `VoiceDriver` resolves deferred timers and resource references before returning | Preserve ordering without making that representation a component API |
| `Signal` | Private `VoiceCallback` for necessary deferred domain calls | `VoiceDriver` invokes the handler after the current component transition returns | Preserve reentrancy protection and defined notification order |
| Behavioral `Decision` | Existing domain outcomes or a narrowly scoped result; the driver decides whether follow-up needs deferral | Recognition produces its result; the driver coordinates the response | Avoid a generic decision envelope and dispatcher |
| Journal-only `Decision` variants | Direct observation recording through the existing trace families | The observation owner reports what occurred; recording stays separate from behavioral dispatch | Preserve observations without treating them as commands |

Do not use a fixed type-count reduction as the acceptance criterion. Ordinary contributors
should follow state and domain methods without learning effect storage, staging, or queue
plumbing. The retained protocol names are `VoiceEffect` at the runtime boundary and
`VoiceCallback` for private deferred coordination. Any other private representation must
justify its ordering or ownership role; it is not a new interface for all components.

`VoiceCallback` is an enum of deferred domain handler calls, not a stored closure, a Python
callback, or another task. Host hooks still run through the existing host invocation boundary.

The driver's private bookkeeping must preserve the existing two-phase ordering: resolve
pending operations, invoke one deferred callback, then resolve the operations it produces
before invoking the next callback. Timer/resource requests keep their defined relative order.
Timer queries must not move entries or insert journal records. How the private fields represent
that ordering is an implementation decision; do not introduce another public collector type.

An outbox stores requests that have not yet been performed. That is distinct from a duplicate
pending-work registry mirroring which domain objects are alive. The run-completion audit below
must still justify any additional registry independently.

### Names at execution boundaries

Scope event and effect types to the domain whose protocol they represent, even inside a
module. A qualified path explains where a type is defined; the type name must also explain
its role when imported, shown in a signature, matched, or printed in diagnostics. Avoid aliases
such as `use avatar::AvatarEffect as Effect` that erase that information.

| Current boundary or type | Proposed name | Producer | Handler and state authority |
| --- | --- | --- | --- |
| Voice core `Input` | `VoiceEvent` | Runtime adapters translate host commands, provider observations, and completions | Voice core decides voice policy; components own their internal state |
| Voice session `Effect` | `VoiceEffect` | Voice core | Voice runtime performs requested work; it does not decide conversational policy |
| Runtime voice `Event` | `VoiceRuntimeEvent` | Runtime tasks and port adapters | Voice runtime handles execution envelopes and delivers `VoiceEvent` to the core |
| Voice `HostEvent` | `VoiceHostEvent` | Voice core/runtime integration | Host adapter handles framework-to-host coordination and outcomes |
| Existing public-facing `SessionEvent` | Keep `SessionEvent` | Voice core selects a session publication | Session event processing applies required bookkeeping; the host receives the public notification |
| Avatar runner `Event` and `Effect` | `AvatarEvent`, `AvatarEffect` | Avatar port observations and avatar core, respectively | Avatar core owns lifecycle decisions; avatar runtime performs transport work |
| AgentRun `Event` and `Effect` | `AgentRunEvent`, `AgentRunEffect` | Run driver observations and AgentRun core, respectively | AgentRun owns invocation decisions; its runtime driver performs effects |
| TextSession `Event` and `Effect` | `TextSessionEvent`, `TextSessionEffect` | Session driver observations and TextSession core, respectively | TextSession owns routing decisions; its runtime driver performs effects |
| A2A task completion | `A2aTaskEvent` | Runtime operation results and supplied wakes | A2a TaskMachine owns storage/handler progression; Completion::Event remains a payload variant |
| Room wait observations | `RoomWaitEvent` | RTC adapter translates room observations | RoomWait owns participant/publication readiness |
| Expert inbox queue entry | `ExpertInboxEvent` | Expert progress/questions/outcomes | Existing frontend inbox owns presentation eligibility; no new delegation queue |
| Existing room observations | Keep `RoomIoEvent` and `RoomEvent` | Room I/O adapter and SDK boundary, respectively | Room I/O handles participant/stream state; voice consumes relevant observations |
| Existing provider events | Keep `SpeechEvent`, `VadEvent`, and other meaningful provider names | Provider port | The consuming recognition or model component interprets them |

`VoiceEffect`, `AvatarEvent`, `AvatarEffect`, and scoped core effect types are implemented.
The scoped input, host, runtime and core event names in this table are now implemented;
imports retain those names rather than aliasing them back to generic `Event` or `Input`.

`VoiceEvent`, `VoiceRuntimeEvent`, and `SessionEvent` are distinct because their protocols
are distinct: a core input, an I/O execution envelope, and an observable session notification.
Do not rename all three to `VoiceEvent`. A model chunk or SDK room observation also retains
its source-specific type when wrapped for voice delivery.

The owner prefix identifies the protocol, not exclusive authorship: an event can be produced
by an adapter but handled by the named domain. Rust memory ownership still follows the actual
values. The producer/handler table identifies which component is allowed to change policy or
state; a descriptive type name alone does not enforce that boundary.

### Names for retained decisions and outcomes

The same ownership rule applies to decisions, verdicts, and outcomes. A type name must say
what is being decided without requiring the reader to look up its module. Keep the domain
result when it carries meaningful alternatives or data; remove a generic dispatch wrapper
when ordinary calls or existing domain results replace it.

| Existing type | Proposed name or replacement | Decision owner |
| --- | --- | --- |
| Generic session `Decision` | Remove as specified above; do not merely rename it `RecognitionDecision` | Recognition produces domain outcomes; the voice core coordinates the follow-up |
| Recognition `CommitDecision` | `UserTurnCommitDecision` | User-turn admission policy; carries its verdict and related commit facts |
| Recognition `CommitVerdict` | `UserTurnCommitVerdict` | The same user-turn admission policy; names the alternatives inside the decision |
| Voice tool executor `StepDecision` | Remove in favor of shared `AgentRun` step settlement | Use `HandleToolResults` / `ToolResultsHandled` with `AgentRunContinuation`; no replacement `ToolStepDecision` type |

These are proposed names for existing roles, not extra types layered over them. In particular,
renaming the useful user-turn admission result does not justify preserving the unrelated
generic `Decision` dispatcher. Avoid names such as `VoiceDecision` when the narrower subject
is known: user-turn commit and tool-step continuation are different decisions within voice.
Preserve scoped names in imports rather than aliasing them back to `Decision` or `Verdict`.

### Domain state and execution owners

| Existing owner | Proposed name | Responsibility |
| --- | --- | --- |
| Core session coordinator `Driver` | `VoiceDriver` | Coordinates recognition, activity, scheduling, timers, and lifecycle through component methods |
| Runtime voice `Runtime` | `VoiceRuntime` | Runs the voice loop, owns I/O execution, and feeds correlated events back |
| Python/runtime session handle `VoiceSession` | Keep `VoiceSession` | The familiar session API used by application code |
| Private core session-state struct `VoiceSession` | `VoiceSessionState` | Stores session-owned configuration, history, outputs, and lifecycle data; distinct from the public handle |
| `AgentActivity` | Keep `AgentActivity` | Owns the active agent's recognition, speech scheduling, hooks, and handoff components |
| `SpeechHandle`, `AudioRecognition`, `SpeechScheduler` | Keep existing names | Preserve familiar domain responsibilities and their invariants |
| Session run-completion bookkeeping `Runs` | Derive completion from existing work owners; `VoiceWorkTracker` only if justified | Tracks this voice run's attributed work, with membership/result semantics settled before restructuring |

These are renames of existing owners, not additional wrapper objects. Child lifecycle state
remains private. Coordination across components belongs to `VoiceDriver`; it calls
component operations rather than changing another component's bookkeeping directly.

### Replacements at call sites

Application-like domain code should read as ordinary operations. Effects are the driver's
implementation detail, not an object passed through every helper. The examples below explain the implemented ownership boundary. Explicit `now` arguments
remain where a domain operation needs its supplied instant; effect collectors do not.

#### Interrupt speech through its domain operation

```rust
// Before: domain methods pass a transition collector through their calls.
self.interrupt_speech(now, speech_id, cause, force, out)?;

// Implemented domain call: the driver owns the bookkeeping internally.
self.interrupt_speech(now, speech_id, cause, force)?;
```

`interrupt_speech` checks permission, updates speech state through its owner, and records any
required runtime work. The caller does not choose raw cancellation effects or manipulate a
queue. A refused interruption must still leave the output and unrelated speeches untouched.
Time is supplied explicitly or read from the driver's injected step clock, never from an ambient clock.

#### Keep raw effects at the execution boundary

Only the private driver implementation needs to translate an accepted domain transition into
runtime work. A helper inside that implementation might contain:

```rust
// Private VoiceDriver bookkeeping, not a component-facing API.
self.outbox.push(VoiceEffect::CancelSpeech(speech_id));
```

This records a request. It does not cancel a task at this line. `VoiceRuntime` performs the
request after the core returns. The outbox is a private field, not a new `VoiceEffects` type
or a service provided to recognition, scheduling, hooks, and handoff.

#### Expose draining, not mutable outbox access

Keep the field private and use one consuming operation:
`take_outbox(&mut self) -> Vec<VoiceEffect>` takes and clears the same storage. The current
driver already has this method; the refactor removes overlapping staging rather than adding
another independently consumed effect list.

Selected delivery contract:

- A public `start`/`handle`/`finish` transition settles its internal work, then drains that
  storage once and returns the effects. The returned vector is the runtime's execution batch.
- Work legitimately produced outside a public transition stays in the same storage until
  `take_outbox` or the next public transition drains it. Draining clears it; it is never also
  returned through a second path. The runtime never drains midway through an atomic step.
- A second drain with no intervening work returns an empty vector. Startup, undelivered
  commands, late refusals, and shutdown obey the same identity/order rule.

No mutable getter or per-component collector is exposed. Tests must establish exactly-once
batch delivery at each lifecycle entry point, including out-of-step work followed by `handle`.

#### Make scheduling advancement a driver guarantee

**Decision:** remove manual scheduling obligations from domain handlers. A domain operation
must not require its caller to remember `pump_schedule`, enqueue a scheduling callback, or
set a scheduling-dirty flag. Renaming the helper is insufficient.

The intended driver contract is: after handling an input, all immediately actionable internal
consequences have been processed and their runtime requests returned. Further progress needs
an external input or a deadline. “Actionable” excludes parked speech, an authorization hold,
a future speech-gap deadline, and work awaiting runtime completion. It does not mean every
live operation is finished or the whole session is idle.

**Feasibility audit:** use a driver-owned scheduling checkpoint after each complete external
input transition, deferred callback, and ordered waiter batch. No handler-specific pump call,
scheduling callback, or manually maintained dirty flag is needed. The source audit found no
handoff/close early-authorization counterexample under the atomic boundaries below. This is
a source-backed design conclusion, not a claim that the redesigned driver has been tested.

**Correction to the initial concern:** a handoff/drain/close admission pause does not stop
already admitted speech from being authorized. That speech must continue to drain, otherwise
the handoff or graceful close cannot complete. AMD's authorization hold is a different gate.
Likewise, requesting close is not the hard stop: `Requested`, `Exiting`, and `Draining` can
still progress accepted work. `Edge` and later phases prohibit authorization.

| Authoritative state | New speech admission | Already queued speech authorization |
| --- | --- | --- |
| Ordinary session | Normal domain checks | Allowed if floor, delay, and authorization gates permit |
| Handoff exiting | Existing goodbye/exit behavior; incoming turns are held | Existing queue can progress |
| Handoff draining / activity admission pause | Host speech is held for the arriving agent during handoff; other new speech is refused by the installed admission pause | Continue draining accepted work |
| Close exiting | Preserve existing on-exit speech behavior | Accepted work can progress |
| Close draining | New speech refused by admission pause | Continue draining accepted work |
| AMD authorization hold | Admission rules unchanged | Blocked until the authorization hold is released |
| Close hard edge or later | Refuse | Forbidden; terminal cleanup can still progress |

Do not turn this into two new public pause primitives. Keep the distinction explicit in the
existing owners' state, comments, and tests.

### Atomic transition boundaries

A checkpoint runs only after the whole operation below returns, before the next FIFO callback.
It never runs between that operation's internal domain calls, and runtime I/O does not execute
inside a checkpoint.

| Driver transition | What must finish before its automatic checkpoint |
| --- | --- |
| External input | Its complete input dispatch, including any synchronous phase/gate changes |
| `EndOfTurn` callback | Entire `on_end_of_turn`; authorize its eligible reply before a following `TurnSettled` callback |
| Shared hook completion (replaces `OnExitEnded`) | Complete common terminal accounting and the applicable exit-state/handoff/drain/close fan-out; no scheduling midway through completion |
| `PauseScheduling` callback while it still exists | Install the admission pause, cancel preemptive generation, and derive whether the drain settled |
| `Swap` callback | Recheck closing; install agent/config, lift the activity pause, replay held speech, issue/register on-enter, release handoff work, enqueue held-turn delivery and next-handoff |
| `NextHandoff` callback | Select/start the next handoff and establish its exiting/draining phase |
| `Paused(Close)` callback | Entire `close_edge`: install hard-edge state, perform its cleanup, cancel scheduling timers, and force-cut remaining speech |
| `CloseProgress` callback | Complete the close-phase transition and its domain actions |
| Other deferred callbacks | Complete their existing dispatch arm, then apply the same automatic checkpoint |
| Waiter batch | Preserve deferred-reply, expert, filler, turn-limit, and idle-waiter ordering; then advance scheduling and process generated callbacks |

Lifecycle entry points such as startup also use the coordinator's transition processing;
shutdown preserves its terminal-only restrictions. No alternate entry point may bypass the
applicable phase/gate checks.

### Checked source interleavings

These traces explain why a fixed post-transition checkpoint works without priority queues,
scanning callbacks for hidden intent, or globally running close callbacks first:

1. **Exit ends while speech remains queued.** Handoff changes `Exiting` to `Draining`
   synchronously, then queues `PauseScheduling`. A checkpoint may authorize already admitted
   speech; this is required drain progress. During the window before the pause callback, host
   speech is already held by `handoff.draining()`, and incoming turns by `handoff.switching()`.
   The later pause prevents further admission; it must not revoke valid queued work.
2. **Drain completes and queues a swap.** `settle_pause` emits its settled notification only
   when the speech map is empty. Admission remains frozen. Therefore a checkpoint between
   `Paused(Settled)` and `Swap` cannot start old-agent queued speech: there is none. `Swap`
   rechecks closing before installing the arriving agent.
3. **Close begins before a queued swap executes.** Close state is installed synchronously.
   The swap's closing recheck refuses the handoff. Do not reorder the FIFO or temporarily
   install the arriving agent. Existing close-exit/drain rules still apply to accepted work.
4. **Close drain reaches its hard edge.** `close_drain` installs `Draining` and pauses admission
   synchronously. A settled pause requires an empty speech map. `Paused(Close)` then performs
   the entire hard-edge operation before another checkpoint. Authorization sees `Edge` and
   cannot start work. A non-draining close first cuts open speech, but its on-exit phase keeps
   the existing hook-speech semantics; do not blanket-ban speech from the close request onward.
5. **End-of-turn followed by turn-settled.** Recognition enqueues those callbacks in that
   order. Finish end-of-turn, automatically advance scheduling, then handle turn-settled.
   Authorization releases any old paused-output binding as needed before false-interruption
   settlement can resume it. Draining the whole callback queue before scheduling is incorrect.
6. **Close edge and interruption watchdog coincide.** Complete the close edge first when that
   is the current transition. It cancels per-speech watchdogs; its forced interruptions do
   not rearm them because the hard edge is installed. The next checkpoint must honor those
   cancellations. The close bound owns later release of unreported speech. If the watchdog
   transition occurred first, close sees that terminal result; do not release the speech twice.
7. **Waiters create another reply.** Process the ordered batch, checkpoint, and settle the
   callbacks it produced before returning. Authorization does not depend on an unrelated
   future input. Run completion must not be checked ahead of newly admitted follow-up work.

These establish the relevant gating/order relationships in the current code. More frequent
checkpoints can still change watchdog timing and intermediate floor/trace observations.
Those differences need focused comparison against the chosen contract; source inspection
alone does not establish complete trace equivalence.

### Coordinator sequence and termination

**Selected rule: current state determines eligibility; FIFO callbacks determine transition
order.** The scheduler inspects the speech queue to select eligible work. It does not scan
ahead in the callback queue for a future pause, handoff, or close to decide whether the
current transition may schedule speech. Each completed transition installs its authoritative
state before the automatic checkpoint. Preserve the admission/authorization distinction above.

Remove `pump_schedule` calls from domain handlers and retire that caller-driven entry point.
Scheduling remains a private driver operation invoked by transition processing. Do not expose
a replacement `advance_schedule` method that callers must remember to invoke. No new callback
priority scheme, queue-lookahead mechanism, or scheduling primitive is part of this design.
Revisit lookahead only if a concrete failing interleaving demonstrates that the state contract
cannot express the requirement; document and test that exception before changing the rule.

Use the existing callback queue, lifecycle state, and private effect queue. The coordinator:

1. Completes one input/callback transition without reentrant scheduling.
2. Honors its ordered internal operations, including pending timer cancels/rearms, before
   choosing a due scheduler timer; runs terminal scheduler cleanup before eligible authorization.
3. Completes scheduling follow-ups and enqueues their domain callbacks; runtime requests stay
   queued for execution after the core returns.
4. Processes the next FIFO callback through the same transition/checkpoint sequence.
5. Processes eligible deferred work and the ordered waiter batch, then settles their generated
   work through the same sequence before the relevant publication/retirement boundaries.
   Run completion is checked against attributed obligations under the agreed membership rule;
   do not globally absorb every waiter-created operation into an open run.

Do not repeat waiter delivery on an unchanged state merely because work is parked. In
particular, consequences attributed to a run must settle before that run is declared complete.
First decide run membership/result semantics. Neither the existing pre-waiter `run_check` nor
a blanket post-waiter move is accepted without that contract.

Moving auto-advancement into `SpeechScheduler::enqueue` alone does not solve this. Eligibility
also depends on session close state, effective delay options, authorization pause, driver-owned
timers and pending timer changes. A scheduling pass can cancel tools, release paused output,
and submit a realtime request. Those remain driver composition responsibilities.

Separate deadline processing from speech selection if that gives the driver a clear phase
order; both operations must still run automatically. `pop_due` currently respects timer arms
and cancellations pending in the same step. Preserve that property and the journal's causal
order. Repeated evaluation of unchanged state must not duplicate effects, consume a timer
twice, or create a busy loop. Stop when no synchronous transition can progress; waiting on a
runtime receipt or future deadline is a valid stopping condition. Do not implement this as
an unrestricted “repeat until the session is idle” loop.

Before implementation, demonstrate these contracts with focused scenarios and then the
voice/room DST over the real runtime and port fakes:

- End-of-turn creates and authorizes a reply in the same input, in the correct relationship
  to the subsequent turn-settled callback.
- Deferred replies, expert presentation, fillers, and callback-created speech progress
  without a second unrelated input; completion checks do not settle ahead of follow-up work.
- Admission pause still lets previously accepted speech drain; AMD authorization hold blocks
  it. Host speech and user turns during handoff reach the arriving agent exactly once.
  Close racing swap refuses the handoff, and no speech authorizes after the hard edge.
  Resuming authorization makes eligible speech runnable immediately.
- A same-instant cancellation or rearm suppresses stale watchdog and speech-gap timers.
  Closing still processes required watchdog cleanup without authorizing new speech.
- Re-evaluating an unchanged blocked/current schedule produces no duplicate authorization,
  timer, realtime request, floor notification, or publication; future deadlines remain exposed.
- A callback appended later does not retroactively change eligibility at an earlier checkpoint.
  Inspect the speech queue for selection; do not inspect future callbacks for scheduling intent.
- Startup, late-command refusal, and shutdown satisfy their lifecycle contracts as well as
  normal `handle`; rejected speech cannot become unfinished run work after release.

Source anchors: [driver handling and timers](../../rust/crates/lk-agents-voice/src/agent_session/machine/driver.rs),
[callback settlement](../../rust/crates/lk-agents-voice/src/agent_session/machine/coordination.rs),
[scheduling and authorization](../../rust/crates/lk-agents-voice/src/agent_activity/scheduling.rs),
[idle/deferred work](../../rust/crates/lk-agents-voice/src/agent_activity/idle.rs), and
[close progress](../../rust/crates/lk-agents-voice/src/agent_session/closing.rs).
The automatic-scheduling redesign now implements these boundaries. The findings below retain
their audit evidence; final combined regressions are recorded separately from that earlier audit.

### Verification performed for this ordering audit

The release `lk-agents-fuzz` regression target compiled successfully. Five existing tests
passed against the current implementation:

| Existing regression | Relationship checked |
| --- | --- |
| `session_state::held_turn_idle::a_tool_waiting_for_idle_lets_a_drain_holding_a_turn_finish` | Held incoming turn does not deadlock handoff drain and reaches the arriving agent |
| `session_state::final_ends_pause::a_final_transcript_interrupts_the_speech_it_paused` | Final transcript resolves paused speech even while admission is draining |
| `session_state::hold_at_close::a_hold_ends_with_the_session` | Close ends the foreground hold |
| `session_state::idle_while_closing::idle_during_a_close_grants_no_hold` | Close prevents a new foreground hold at the idle boundary |
| `session_state::handoff_refused_releases::a_handoff_refused_after_the_end_releases_its_providers` | Late/refused handoff releases its provider resources |

`tools::tool_reply_scheduled_at_once::a_second_tool_update_reply_is_authorized_at_once`
initially failed with “there is no reactor running” during teardown. That harness issue is
now fixed by retaining the deterministic Tokio handle, and the regression passed in the
208-test deterministic suite. This is existing-behavior evidence, not verification of the
then-unimplemented automatic checkpoint sequence (now implemented; see the status checkpoint).

Reproduce any row with:

```sh
cargo test --manifest-path rust/Cargo.toml --release -p lk-agents-fuzz \
  --test regressions '<full test name above>' -- --exact --nocapture
```

These runs characterize existing behavior; they did not test the later driver
sequence. Dedicated assertions for callback/authorization order, hard-edge/watchdog
coincidence, authorization-vs-admission pause, and newly created waiter work remain required
when implementing the redesign. The checked source traces above resolve the architecture
question; they are not exhaustive runtime interleaving coverage.

#### Return a domain outcome from a component

The parent should learn what occurred in the component, not how to transfer its output
between collectors. For the existing STT delivery path:

```rust
// Existing STT path inside VoiceDriver.
let outcomes = self.activity.audio_recognition.on_stt_event(now, &event);
self.apply_recognition_effects(outcomes);
```

Recognition updates its own state and returns scoped `AudioRecognitionEffect` outcomes. `VoiceDriver`
chooses the required deferral. Recognition never receives `&mut VoiceEffects` and cannot
reach the session's private effect outbox.

This example does not specify every recognition method. Timer and overlap processing can
produce several ordered outcomes. Define each method's actual domain result and preserve its
order before removing the current mixed output. A small owner-specific result can be useful;
do not invent a replacement generic `Decision` or require all components to use one protocol.
Standalone recognition tests should inspect its domain outcomes without constructing a driver
or a session collector.

#### Defer a handler when ordering requires it

`VoiceDriver` may queue a private `VoiceCallback` instead of immediately reentering a component.
The callback runs later in the same synchronous step; it is not an async task or host callback.
For the existing end-of-turn path, the driver's callback handler performs both operations:

```rust
self.on_end_of_turn(info);
// Scheduling advancement belongs to the driver’s transition processing.
```

The former handler explicitly pumped scheduling; the implemented driver now supplies that
advancement automatically at its complete-transition boundary. A following turn-settled callback must remain behind
that handler and the work it creates. Deferred callback storage is private to the driver;
components return domain outcomes rather than enqueueing callbacks through a shared collector.

The three timing rules are:

| Operation | When it happens | Who owns it |
| --- | --- | --- |
| Domain method | Now, during the current synchronous transition | The component whose state it changes |
| `VoiceCallback` | After the current transition returns, before the core step finishes | `VoiceDriver` |
| `VoiceEffect` | Runtime work is submitted after the core returns; completion is a new event | `VoiceRuntime` executes the driver's request |

#### Process session events before publication

The parent receives a meaningful session observation and calls its own event-processing
operation. That operation performs state, usage, tracing, DTMF, keyterm, and context work
before adding the final publication request to its private outbox. A child never pushes a
final `VoiceEffect::Emit` directly.

Enforce this with private storage and driver-owned publication operations, rather than an
unrestricted shared `push(VoiceEffect)` interface. Preserve separate internal paths for
already-processed trace spans and startup/shutdown publication outside an active step; those
paths must not repeat normal event processing.

#### Keep observation and detector operations meaningful

Driver code records a trace observation through its private tracing operation. Detector work
uses a named operation such as `notify_interruption_detector`, with overlap identity resolved
at the defined point. Neither requires a shared `voice_effects` parameter. Journal-only
recognition observations never enter the behavioral callback queue.

### Shared AgentRun integration

`VoiceDriver` uses the existing `AgentRun` in `lk-agents-core` for each model reply. Keep that
one shared invocation state machine for direct agent runs, text sessions, and voice. The
voice-specific adapter binds it to speech, floor access, and runtime execution; it must not
grow a second agent loop.

The current implementation is in
[reply_run.rs](../../rust/crates/lk-agents-voice/src/agent_activity/reply_run.rs), with the
shared coordinated contract in
[agent_run/coordinated.rs](../../rust/crates/lk-agents-core/src/agent_run/machine/coordinated.rs).
`ReplyRun` currently owns an `AgentRun` alongside its speech/request correlations and waiters.
Keep that ownership visible rather than hiding it behind another generic run abstraction.

| Responsibility | Owner | Voice integration responsibility |
| --- | --- | --- |
| Tool admission, argument preparation, approvals, retries, and accepted output | `AgentRun` and its shared core components | Supply outcomes and execute requests when the voice floor permits |
| History snapshot and request planning for a coordinated voice reply | Voice coordination, using shared request-planning code | Plan the admitted speech against its intended history and resource snapshot |
| Recognition, speech scheduling, interruption, and playback | `VoiceDriver` and its domain components | Feed relevant constraints to the shared invocation; retain audio lifecycle outside `AgentRun` |
| Whether a settled step may continue under voice constraints | Voice coordination, through `AgentRunContinuation` | Report interruption, stop, or allowed continuation; do not reverse a core failure or accepted output |
| Agent invocation completion | `AgentRun` | Store/deliver its result without treating it as playback completion |
| Completion of the surrounding voice activity | Voice session lifecycle | Account for playback and required hook/handoff/hold work without another agent execution loop |

The event/effect handshake should be visible in one short path:

```text
VoiceDriver delivers AgentRunEvent
    -> AgentRun updates invocation state
    -> AgentRunEffect requests planning, tool/host work, or reports settlement
    -> voice adapter binds that request to the speech and permitted floor
    -> VoiceRuntime executes I/O
    -> correlated result returns to the same AgentRun
```

**Implemented decision: use the shared AgentRun step outcome.** Voice `StepDecision` and
its duplicate raw-outcome interpretation are removed rather than renamed to `ToolStepDecision`.
`AgentRun` owns tool outcome selection, retry/output precedence, canonical recorded items,
and whether the invocation would continue. Voice consumes that result and supplies only its
explicit domain constraints through the existing coordinated handshake.

The target path is:

```text
Tool outcomes -> AgentRun
  -> HandleToolResults { step, items, continues, terminal_call }
  -> VoiceDriver applies permitted voice restrictions / host-hook handling
  -> ToolResultsHandled { step, continuation, reply_required } -> same AgentRun
  -> shared run effects / terminal result -> voice presentation and speech lifecycle
```

Reuse `StepId` and the shared `HandleToolResults` / `ToolResultsHandled` handshake with
`AgentRunContinuation`. Do not introduce a second
step-result enum, parallel commit list, or independent terminal-selection algorithm. Remove
voice `next_step`, `StepDecision::settle`, and related `NextStep`/`ReplyRequest`/`StepInputs`
wrappers where their responsibilities are covered by the shared run. Keep only actual voice
state and ordinary driver methods for scheduling, interruption, realtime delivery, history
publication, and host callback coordination; do not remove unrelated methods merely because
they share a name.

The shared API supplies canonical items, a continuation flag, a selected compatibility terminal
call and lossless correlated input-requirement metadata. Failure/accepted-output precedence
remains invocation policy; selecting the compatibility projection does not discard other requests. `AgentRunContinuation` permits continue, stop, or interruption with the documented
failure/accepted-output precedence. The handled-results event carries permitted per-call
`reply_required` edits, not arbitrary replacement tool outputs. Core and voice apply that same
patch to canonical history; ordinary continuation can change without reselecting a terminal
outcome. Keep host-owned accepted values correlated with the
shared accepted output. Do not expose core's private `Next` enum simply to reuse a type name.

Required regressions: multiple terminal tools in model order with varied completion order;
shared retry exhaustion versus later successful outputs; accepted output versus interruption;
voice step budget/drain/handoff restrictions; host handler edits; realtime and pipeline history
consistency; and interrupted/cancelled cleanup. Assert that presented question, reported result,
and answer-tool routing remain correlated. For human questions, additionally verify the
lossless reporting seam; the existing first-only test is not acceptance of multi-question policy. Invocation completion still
does not imply speech or playout completion.

The implemented adapter now preserves these ownership contracts:

- `open_reply_run` extracts the initial planning effect and processes all remaining startup
  effects through the ordinary owner; it no longer drops them with `find_map`.
- `HandleToolResults` supplies canonical calls-then-outputs, each in model order. This matches
  the existing documented shared contract and voice's timestamp-ordered storage. Voice consumes
  the items rather than independently reconstructing them. End-to-end tests inspect the next
  actual model request as well as emitted history events.
- Invocation code preserves terminal precedence and all correlated input requirements, including
  mixed completion/question batches. The compatibility terminal projection does not implement
  a question queue or decide future delegation presentation/routing.
- Host tool-executed handlers can patch only `reply_required` through the shared contract.
  Both histories receive that patch. Voice also re-reads current interruption state after
  awaited hooks, preventing a stale executor snapshot from creating a new question afterward.
- Duplicate policy uses supplied facts about existing admitted/executing voice bodies,
  including detached work and concurrent replies. Validated argument identity and stable
  operation IDs preserve scope and cancellation correlation without another live registry.
- Advertised function tools retain installed policy metadata. Unknown policy metadata is
  refused/diagnosed rather than replaced with permissive defaults.
- Prepared arguments are owned by runtime call tasks; core's resource ID is synthetic at
  that boundary. Validation and refusal callbacks are speech-owned, precisely cancellable,
  and retired on speech release. Deferred reply rendering and detached bodies retain their
  distinct longer lifetime. Shared request cancellation reaches producer interruption and
  explicit stream cleanup, bounded by the existing watchdog.
- Voice wakeups include shared-run cleanup deadlines. Missing cancellation receipts cannot
  leave the shared machine waiting forever without its deadline being driven. Failed text
  validation also no longer waits on the validation result already received.

Implementation, tests, original evidence, and retained distinctions are recorded in the
[shared policy audit](rust-shared-policy-audit.md). Scheduling, outbox, hook completion and
owner-derived run work are implemented too; their combined validation is recorded separately.

Use the existing focused core, runtime, and voice tests to verify these boundaries. Clearer
names and a shorter file alone do not establish that the adapter is correct.

### Removal rules

| Current pattern | Preferred direction | Evidence needed to keep additional machinery |
| --- | --- | --- |
| `Out` plus `Effects` and staging | Remove transition plumbing from domain calls; keep one driver-owned runtime outbox | A concrete ordered outcome that ordinary component returns cannot express clearly |
| `Transition` then `Planned` | Remove duplicate representations instead of replacing them with equivalent wrappers | A specific invalid operation or lifecycle bug prevented by the distinction |
| Local call through `Signal` and a dispatcher | Call the domain method directly where safe | A follow-up must wait until the current transition finishes, or deferred fan-out/order is required |
| Shared `StepOp` | Use explicit timer or resource operations inside their owner where safe | A request must use state at a later, precisely defined point |
| Generic recognition `Decision` | Return a domain outcome directly to the caller | Several independent outcomes genuinely require ordered composition beyond an ordinary return |
| Journal-only decision | Record the observation directly | Preserve recording time and causal order; no behavioral dispatch is needed |
| Tiny event/effect pair | Use ordinary methods and a small return value | A real asynchronous boundary or state machine with meaningful phases |
| Identical stream enums on both sides of an adapter | Use one existing domain protocol type | Different semantics or ownership obligations, not just different modules |

`VoiceEffect` remains the voice runtime boundary. Cross-component deferred callbacks and
runtime requests are private to `VoiceDriver`; child components retain their own domain state
and outcomes. Additional private operations inside a component are exceptions to
justify per behavior, not a shared mechanism every component must participate in. Separate
buffers or differently shaped enum variants do not alone justify keeping their types.

### Complete recognition and interruption path

Recognition produces domain information; `VoiceDriver` decides when to handle it. This
uses the actual STT delivery methods; other observations have their own scoped returns:

```rust
let outcomes = self.activity.audio_recognition.on_stt_event(now, &event);
self.apply_recognition_effects(outcomes);
```

An immediate call is not correct for every existing recognition path. End-of-turn detection
currently requests `EndOfTurn` followed by `TurnSettled`; overlap classification also requests
ordered session publications and interruption work. Preserve those dependencies explicitly:

1. A provider observation becomes a `VoiceEvent`. `VoiceDriver` routes it to recognition.
2. Recognition updates its own state and returns its domain outcomes/observations. It does
   not receive the driver's outbox or a shared effects object. The driver translates any
   requested detector or session work at the boundary.
3. For accepted overlap, the core selects the overlap and metrics publications before the
   interruption follow-up. The driver receives the recognition outcome and selects the
   corresponding private callback; recognition does not call back into the session while
   still mutably borrowed.
4. The core resolves the current batch of pending operations, then invokes the next private
   `VoiceCallback`. End-of-turn calls `on_end_of_turn`; interruption calls the existing guarded
   interruption operation, including its required `end_agent_speech_after_overlap` follow-up.
   End-of-turn also performs its scheduling pass. These remain ordinary domain methods.
5. `interrupt_speech` checks interruption permission and the active/paused speech. It chooses
   the bound output, marks speech state, and requests clear/cancel/watchdog work. Refusal does
   not clear another speech's output. The clear request retains its position before resume.
6. The core processes subsequent follow-ups in their defined order. For end-of-turn,
   `TurnSettled` follows the turn handler and its pending work. Handoff, drain, and close
   notifications occur after the current component transition finishes.
7. After core finalization, `VoiceRuntime` performs `VoiceEffect` requests. Playback
   receipts and provider completions return as correlated `VoiceEvent` values.

The reader follows recognition, turn completion, and interruption. Only a maintainer changing
deferred ordering needs to inspect `VoiceCallback` or the driver's private pending-operation
representation. No shared collector argument or generic decision dispatcher participates in
the domain call path.

### When an additional primitive is justified

Keep an additional primitive only with a concrete explanation of the ordering, ownership, or
lifecycle bug that an ordinary method and return value would otherwise permit. Examples are
reentrant handoff while a component is transitioning, a timer firing after same-instant
cancellation, a stale reply completing the wrong request, or a resource released twice.

Use the smallest local mechanism that preserves the obligation. Keep real asynchronous
boundaries and meaningful domain states explicit. Architectural consistency, hypothetical
future use, and making every helper follow one pattern are not sufficient reasons.

### Migration rationale

The implemented migration keeps deferred timing/resource semantics intact while collapsing
their representation. Eliminating the remaining private queues is a separate behavioral
change, not a prerequisite for removing transition plumbing from domain methods. No alias should retain
both the old and replacement vocabularies after migration.

Pin the mixed pending-operation/callback ordering in focused tests. Define concrete component
return contracts, move transition bookkeeping behind driver-owned methods, and migrate one
behavior before removing the old wrappers. Apply owner/event/effect renames in separate mechanical changes so reviewers
can distinguish semantic changes from naming changes. Update runtime, simulation, and trace
consumers with the boundary they use, and preserve all journal observations. Move behavioral
`Decision` recording into turn/interruption trace families; distinguish detection from the
existing post-admission `TurnStep::EndOfTurn` observation. Observation-only variants never
enter the callback queue.

## Historical transition path

The central implementation is
[session coordination](../../rust/crates/lk-agents-voice/src/agent_session/machine/coordination.rs);
[VoiceEffect declarations](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs)
are separate.
The outer orchestration is in
[voice driver](../../rust/crates/lk-agents-voice/src/agent_session/machine/driver.rs), and
runtime execution is in
[voice/effects.rs](../../rust/crates/lk-agents-runtime/src/voice/effects.rs).

```text
Component returns Effects containing Transition entries
    -> Out.machine stages those entries
    -> Out.absorb journals and separates them
        -> pending Planned entries containing VoiceEffect or StepOp
        -> internal Signal queue
    -> settle_step applies local operations and dispatches signals
    -> speech updates and session views are published
    -> Vec<VoiceEffect> reaches the runtime
```

Some components write directly through `Out`; others return `Effects`. `VoiceDriver.outbox`
also holds effects produced outside the active collector, including lifecycle paths.

These buffers are not one FIFO queue. `absorb` separates signals from pending work;
`settle_step` drains pending work before dispatching the next signal. Work produced by that
signal is collected before another signal is dispatched. Replacing this with one FIFO queue
or inline signal calls would change ordering unless equivalence is established.

| Type | Responsibility today | Main issue |
| --- | --- | --- |
| `Effect` | Work returned to the runtime, including publication and journal records | Necessary boundary; distinguish requested work from observations |
| `Effects` | Returned component output containing effects, local operations, and signals | The name suggests a collection of `Effect`, but its contents are broader |
| `Transition` | One mixed component-output entry | The name suggests a state transition, but the entry requests follow-up work |
| `Out` | Staging, journal insertion, signal separation, and pending work | A vague name hides several responsibilities |
| `Planned` | Pending effect or local operation after signals have been separated | Another representation of largely the same entries |
| `StepOp` | Timer mutation and requests resolved against session state | Two different responsibilities share one wrapper |
| `Signal` | Synchronous communication consumed within the step | Useful boundary; several variants hide their subject |
| `Decision` | Recognition outcomes, control changes, and journal-only observations | Different meanings share one dispatch path |

## Collector improvements

This table retains the original scope/risk assessment. C01–C10 are implemented; the private
coordinator keeps only ordered callback dispatch and late input-identity resolution. Necessary
resource-lifetime boundaries remain, without exposing collector types to domain components.

| ID | Improvement | Disposition | Scope and constraint |
| --- | --- | --- | --- |
| C01 | Remove `out.machine()` and effect-collector arguments from domain calls | Implemented | Medium. Components return domain outcomes; driver-owned methods handle transition bookkeeping |
| C02 | Remove the staging `machine: Effects` buffer | Implemented | Medium. Preserve ordered component outcomes and driver finalization without a shared mutable collector |
| C03 | Remove `absorb()` | Implemented | Medium. No staging collection remains to drain; read-only queries do not perform bookkeeping |
| C04 | Remove the separate `Planned` wrapper | Implemented | Medium. Flatten private ordered bookkeeping; do not introduce a named replacement primitive for components to learn |
| C05 | Replace `Out` and generic mixed `Effects` with private driver bookkeeping | Implemented | Medium. Keep the outbox private and define actual domain returns per producer; no replacement `VoiceEffects` parameter |
| C06 | Use scoped names for the surviving boundaries and owners | Implemented | Small to medium. Apply the ownership table mechanically after the collector change, without retaining aliases |
| C07 | Remove `Transition` | Implemented | Small to medium. No generic returned-component entry list remains |
| C08 | Reduce visibility | Reviewed; owner/result APIs crate-private where possible | Small to medium. Check runtime, simulation, and test consumers before restricting types and methods |
| C09 | Encapsulate transition storage | Implemented | Small. Keep buffer representation private; expose only operations required by components and focused tests |
| C10 | Preserve ignored-output protection | Implemented with must-use outcomes | Small. Retain `#[must_use]` or equivalent protection when changing returned component output |

The implemented recognition/interruption path addresses C01 through C05 together. The migration
removes domain-facing transition plumbing while retaining deferred ordering. Replacing an
`out` argument with a renamed collector argument or forwarding method does not complete it.

Do not require every helper to return an output collection. An ordinary computation or local
state update should remain an ordinary method.

## Timer and deferred operation improvements

The removed `StepOp` was a private deferred instruction list. Its four variants
set a timer, cancel a timer, flush the current input stream, or send an interruption signal.
The last two resolve live stream/overlap identity during settlement. This combines local
bookkeeping with runtime-request preparation and exposes unnecessary machinery to components.

Selected direction: remove `StepOp` and `Planned` from component APIs, not rename them to
another operation type. Timers belong to their state owner; stream flush and interruption
remain explicit domain operations, with runtime requests prepared by the coordinator at the
required boundary. Retain ordering privately only where needed, without making components
construct deferred instructions.

Use `set_deadline(timer, at)`, `clear_deadline(timer)`, and `deadline(timer) -> Option<Ts>`
for the intended timer API. Setting replaces the existing deadline; clearing is idempotent;
reading is observational and reflects accepted changes in the current transition. These
replace `arm`/`cancel` terminology on the timer API without compatibility aliases. Clearing
a deadline does not cancel its underlying operation. Existing `arm`, `arms`, and `StepOp`
spellings below identify the pre-refactor source, not surviving component APIs.

| ID | Current operation | Opportunity | Constraint |
| --- | --- | --- | --- |
| T01 | `Arm(Timer, Ts)` | Use owner-local `set_deadline(timer, at)`; investigate whether deferred application is necessary | Preserve replacement semantics and the existing due-timer pass |
| T02 | `Cancel(Timer)` | Use owner-local `clear_deadline(timer)`; evaluate with deadline setting | Preserve idempotence and suppression of later same-instant fires |
| T03 | `FlushInput(StreamKind)` | Make stream-to-sink resolution explicit; investigate resolving at insertion | The selected sink must correspond to the intended point in the transition |
| T04 | `Interruption(InterruptionSignal)` | Make overlap lookup and trace changes explicit; investigate removing the generic wrapper | Preserve when the overlap becomes available and when its end is recorded |
| T05 | `arms`, `cancels`, `timers_touched` | Keep pending-operation inspection private; expose an observational `deadline(timer)` view | These queries answer different questions; they cannot simply become aliases |
| T06 | Timer queries that call `absorb` | Remove hidden journal and buffer mutation from apparently observational APIs | Resolve together with C02 and C03 |
| T07 | `StepOp` as a separate wrapper | Remove the component-facing wrapper; no named replacement primitive | Preserve timer updates and late resource resolution in the first migration; do not expose a second operation vocabulary |

`VoiceDriver::pop_due` skips deadlines revised in the current atomic transition. Queries
reflect accepted set/clear changes immediately without moving buffers or writing observations.
The coordinator clears that transient exclusion at its boundary; general timer wakes and
scheduler-owned timer processing retain their distinct roles.

## Internal signal improvements

| ID | Improvement | Disposition | Constraint |
| --- | --- | --- | --- |
| S01 | Distinguish processing a session event from publishing it | Implemented | `Effects::emit` creates a signal; `Effect::Emit` reaches the runtime. Preserve all processing before publication |
| S02 | Disambiguate `signal` methods | Implemented | `Out::signal` means internal coordination; `Effects::signal` means an interruption-detector signal |
| S03 | Remove unnecessary signal variants before naming survivors | Implemented | Prefer a direct domain method; add a subject only to a signal whose deferred behavior is justified |
| S04 | Remove `Signal::Recognition(Decision)` nesting | Implemented | Use existing direct domain results when safe, or specific `VoiceCallback` variants when ordered deferral is required |
| S05 | Remove journal-only observations from behavioral dispatch | Implemented scoped observations | `EndpointingDelayUpdated` and `TurnDeclared` currently reach no-op decision match arms |
| S06 | Prefer ordinary calls and domain returns to signal hops | Implemented; necessary FIFO hops retained | Retain a hop only where deferred FIFO handling, reentrancy prevention, or fan-out has a concrete requirement |
| S07 | Document shared signal recipients and ordering | Implemented | Preserve coordination among handoff, drain, close, and scheduling |

The original session `Signal` responsibilities are mapped below; coordination declarations now live
in `agent_session/machine/coordination.rs`, not the session effect declaration file. Keeping a signal is a valid result when its
deferred delivery protects coordination or avoids reentrant transitions.

| Responsibility | Variants | Review action |
| --- | --- | --- |
| Session event processing | `SessionEvent` | Make processing versus publication explicit; preserve state, metrics, and tracing hooks |
| Recognition | `Recognition` | Separate behavioral outcomes from journal-only observations |
| Scheduling | `PauseScheduling`, `Paused`, `Quiet` | Use matching pause/resume methods; any retained drain-result callback names the completed attempt, not successful pause state; see scheduling/close symmetry below |
| Hook completion | `HookFinished` (formerly `OnExitEnded`) | Implemented common completion dispatch; preserve exit FIFO ordering and fan-out when migrating to VoiceCallback |
| Run completion tracking | `RunOwn` | Remove the signal through the owner-derived run-completion redesign; do not replace it with another ownership command |
| Applying or refusing a handoff | `Swap`, `RefuseHandoff` | Use recognizable handoff vocabulary and preserve caller outcomes |
| Recovering after a handoff | `AgentStays`, `DeliverHeldTurn`, `ReplayHeldSpeech` | Audit deferred hops while preserving held turn and speech ordering |
| Continuing handoffs | `NextHandoff` | Preserve queue order and the scheduling pass |
| Advancing close | `CloseProgress` | Replace with coordinator-owned `advance_close` from phase prerequisites when deferred ordering is preserved; no manual pump/dirty flag |
| DTMF | `InterruptForDtmf`, `ReadBackDtmf`, `DtmfFinished` | Keep collection vocabulary; assess whether each needs the shared queue |

The implemented scoped callback renames are: `Quiet` becomes
`SpeechesSettled`, `Swap` becomes `ApplyHandoff`,
and `NextHandoff` becomes `StartNextHandoff`. Keep `AgentStays`: it notifies drain handling
that the agent stayed, rather than commanding the driver to select an agent.
Preserve the implemented common `HookFinished` completion; `OnExitEnded` is removed. See the
lifecycle symmetry audit for the no-host ordering constraint. Keep meaningful held-turn, drain, and DTMF names.
Surviving callbacks are private variants of `VoiceCallback`, not separate new types. Keep their existing FIFO position until a specific
hop is proven safe to replace with an immediate call. `RunOwn` is removed through the
run-completion redesign below rather than renamed to `UpdateRunOwnership` or
`TrackVoiceRunWork`.

The `Decision` variants have explicit destinations rather than a generic replacement enum:

| Current variant | Proposed replacement | Observation destination |
| --- | --- | --- |
| `EndOfTurn(info)` | Existing `EndOfTurnInfo` return or private `VoiceCallback::EndOfTurn(info)` calling `on_end_of_turn` | A pre-admission detection record in `TurnStep`, distinct from its existing post-admission record |
| `TurnSettled(outcome)` | `VoiceCallback::TurnSettled(outcome)` calling `on_turn_settled` after turn handling | Turn settlement in `TurnStep`, retaining the outcome |
| `AudioInterruptGuard` | Recognition returns the guard outcome; driver-owned set/restore operations and callbacks apply it at the defined point | Guard change/restore in `InterruptStep`, retaining the instruction and payload |
| `InterruptByAudioActivity(trigger)` | Private callback calling the existing guarded interruption method | Interruption request in `InterruptStep`, retaining the trigger |
| `EndpointingDelayUpdated` | Direct trace recording; no callback | Endpointing observation in `TurnStep`, retaining the minimum delay |
| `TurnDeclared` | Direct trace recording; no callback | Turn-declaration observation in `TurnStep`, retaining the owner |

## Settlement and observation improvements

| ID | Improvement | Disposition | Constraint |
| --- | --- | --- | --- |
| O01 | Move journal conversion beside tracing | Implemented | Leave the recording point explicit in collection; moving code must not move recording time |
| O02 | Explain journal records in `Effect` | Implemented | They record decisions for the runtime to consume; they do not introduce another decision layer |
| O03 | Audit session-event publication paths | Implemented | Compare normal emission, direct `Effect::Emit`, and `emit_now`; make intentional bypasses explicit |
| O04 | Constrain `VoiceDriver.outbox` | Implemented | Preserve work outside an active step, including startup and shutdown; investigate overlap with step output |
| O05 | Make settlement phases visible | Implemented | Separate pending local work, signals, deferred replies/calls, floor publication, speech updates, views, and retirement |
| O06 | Reassess the two settlement passes in `handle` | Implemented with explicit membership/result contract | Preserve attributed follow-up obligations and provider cleanup; do not mechanically retain or move the run-check position |
| O07 | Keep lifecycle behavior beside its domain owner | Implemented | Recognition, event accounting, transcript forwarding, and handoff should not require reading collector internals |
| O08 | Improve diagnostics for buffered work | Implemented | Removed opaque Out diagnostics; driver-private outbox/callback/input queues have explicit responsibilities and DST retains dispatch/pending trails |
| O09 | Reduce test coupling to collector representation | Implemented | Assert behavior and order where that is the contract; retain focused representation tests when necessary |
| O10 | Document termination of synchronous settlement | Implemented | Explain what consumes work or advances state in each repeated pass; avoid arbitrary iteration limits that drop accepted work |

Publication is behaviorally significant. Session events can update user state, usage,
tracing, DTMF collection, end-call handling, keyterms, and forwarded chat context. A shorter
publication path is safe only when it preserves the required processing.

## General symmetry rule

Symmetry is a general rule across all Rust crates, including core, voice, runtime, media,
async I/O, and bindings. It applies to methods, events, effects, callbacks, states, and
ownership lifecycles. Scheduling and close are examples, not the scope of the rule.

- Use consistent subjects and verb forms for matching operations and milestones: open/close,
  start/stop, pause/resume, attach/detach, acquire/release, and enter/exit where semantics match.
- For each operation, identify its counterpart or explain why it is irreversible. State who
  owns each transition, when it is allowed, what completes it, and how failure, cancellation,
  timeout, and duplicate or late completion are handled.
- Keep request, accepted state change, completed work, and resource release distinguishable.
  A request must not sound like successful completion; cancellation must not imply release.
- Reuse shared lifecycle/completion paths. Add a counterpart callback or effect only when an
  actual consumer or ordering obligation requires it. Synchronous operations may return
  directly while their asynchronous counterparts require completion notifications.
- Document intentional asymmetry beside the owner. Remove unnecessary primitives before
  renaming survivors; symmetry must make behavior predictable without adding concepts.


Keep internal coordination inside its owner. Names such as `CloseProgress`, `Pump`, `Tick`,
`Recheck`, or `Continue` must not become cross-component callbacks/effects merely to tell an
owner to run its own logic again. Prefer automatic advancement from authoritative state at
complete transition boundaries. Reserve callbacks for concrete domain facts or genuinely
deferred completion, and effects for work another executor must perform. A real timer input
or protocol-defined tick remains valid; a generic progress notification needs a documented
ordering obligation before it survives. Do not replace removed callbacks with per-handler
pump calls, dirty flags, or additional bookkeeping primitives.

Apply this checklist to every planned lifecycle change and its call sites. Existing inventory
findings are examples, not an exhaustive list of places where the rule applies.

## Scheduling and close symmetry

Apply the Rust guide's symmetry rule to operations, results, and states separately. Prefer
matching method names for genuine inverse operations; require a callback only when a caller
actually waits for completion. The [symmetry audit](rust-lifecycle-symmetry-audit.md#scheduling-and-close-symmetry)
records source evidence. The replacements in this table are implemented; original spellings identify
the pre-refactor mechanisms.

| Former mechanism | Meaning | Implemented replacement / justification |
| --- | --- | --- |
| `PauseScheduling(cause)` / `pause_scheduling` | Install an admission pause, cancel preemptive generation, and await existing speech release; already accepted speech can still play | Prefer the ordinary `pause_scheduling` domain operation at the established transition boundary; retain a FIFO request only where its ordering is needed |
| `resume_scheduling_task` | At handoff swap, synchronously lift handoff/drain admission pauses and refuse any still-waiting pause callers; never lift a close pause | Rename the method `resume_scheduling` to match `pause_scheduling`; there is no Rust task here and no deferred resume result to justify a new callback |
| `Paused(cause, result)`, including `Paused(Close, Refused)` | Scheduling drain completion, not a pause-success acknowledgment | `VoiceCallback::SchedulingDrainFinished(cause, result)` retains handoff/host-drain receipts. Close derives progression from its own prerequisites; its callback/cause variant is removed |
| `CloseProgress` / `close_progress` | Reconsider close's current wait after phase entry, speech release, trailing transcript completion, or realtime closure | Prefer an internal `advance_close` operation driven by the coordinator from current phase/prerequisite state; remove generic callback plumbing only after preserving its deferred boundary |

Close's drain is followed by the hard edge on **either** settled or refused completion, or
its deadline. This is intentional teardown behavior, not a successful pause acknowledgment.
Keep the exact result visible to handoff/drain consumers; do not generalize close's treatment
to them. Normal resume must never reopen closing scheduling. An admission pause, an
AMD/playout authorization hold, and the irreversible close edge are different states.

Removing `CloseProgress` must not create another `pump_close`, a dirty flag, or a new
per-handler obligation. The coordinator advances close at complete transition boundaries and
stops when its current prerequisite is outstanding. Phase state must be installed before
local completion can dispatch, and a phase's effects/consequences must precede its successor.
The implementation preserves those boundaries using state-owned advancement.
No speculative `ResumeScheduling`, `Resumed`, or mirror close notifications are required.

Acceptance covers handoff pause/resume, host drain retained until its defined release,
multiple pause callers, close taking over an existing pause, both close drain results,
deadline/late completion, immediate/no-host closure, trailing commit and realtime completion,
and repeated unchanged advancement. Assert one terminal delivery per waiter, no speech
launch between atomic close actions, no reopening after the hard edge, and no repeated effects
or same-step spin. Keep the already-open deadline/input tie contract explicit.

## Voice run completion

Keep explicit completion accounting for a voice run, but integrate it with the existing work
lifecycles. Remove the scattered ownership protocol. A retained tracker does not execute the agent,
own Rust memory, determine scheduling, or replace `AgentRun`. It tracks when the voice run can
settle after speeches, on-enter hooks, handoffs, and foreground holds finish.

This requirement comes from real behavior: a speech can finish while a handoff is still
starting the next agent, and an on-enter hook can produce more speech. A run must not settle
in the gap between those operations. Waiting on one model invocation or the current speech
alone is insufficient.

The existing Python implementation watched speech handles and futures with completion
callbacks. Its `SpeechHandle.add_done_callback` also notified watchers for an already-finished
handle. The Rust pending-work set is a valid replacement, but manual registration/release
must preserve that lifecycle guarantee. Python did not require this exact architecture;
the synchronous Rust coordinator can enforce the contract at explicit transition boundaries.

### Scope: voice-run completion, including speech playout

This bookkeeping is specific to voice. It does not track the shared core `AgentRun` used by
text and voice, and it does not represent all background work in a voice session. It answers:
“Has the work belonging to this voice run finished?”

| Participating work | Completion boundary |
| --- | --- |
| Speech | The speech lifecycle releases it; normal audio speech includes output playout settlement, not just model or TTS completion |
| On-enter hook | The hook finishes after admitting any follow-up work |
| Handoff | The handoff lands or is refused, preserving membership through queued and active states |
| Foreground hold | Its participating request/hold finishes or is released |

Playout is covered **through the speech lifecycle**, not by another work entry for every
frame, segment, or sink receipt. Playback state and the runtime output adapter still own
those details. Interruption, failure, and close use their terminal speech paths; completion
does not promise every generated sample was heard. Preserve playback receipt, timeout, and
cancellation semantics when changing those paths.

The audit considered `VoiceWorkTracker` if a separate tracker survived; none was needed. `SpeechWorkTracker`
would hide hooks, handoffs, and holds; `RunWorkTracker` would obscure the voice-only scope.
The optional work-key enum follows the same domain: `VoiceWork`. Prefer removing either
abstraction when the existing lifecycle owners can provide the answer directly.

### Storage design: owner-derived completion implemented

The existing speech scheduler, hook table, handoff state and hold state now retain their
work's run membership and already own its terminal semantics. Completion queries those
records directly. Neither `VoiceWorkTracker`, `VoiceWork` nor a duplicate pending set was
introduced; the earlier naming alternatives below are retained only as rejected-design rationale.

The implementation selected the first design after comparing both costs:

| Design | Membership and liveness | Cost and risk |
| --- | --- | --- |
| Owner-derived completion | Associate existing lifecycle records with `RunId` at admission; ask each owner whether that run has unfinished work | Avoid duplicate registration/release, but include queued handoffs, hold requests before grant, and owner-defined terminal semantics |
| Explicit completion ledger | Retain a set of work IDs associated with the active run, updated by common lifecycle boundaries | A direct membership query, but duplicate liveness can drift unless every terminal path updates both owners |

Owner-derived completion is implemented. The generic work-key enum and duplicate set are
removed; active-run state retains result selection and lifecycle queries provide liveness. Do not substitute global `is_idle()` or raw
collection emptiness: preexisting/unrelated work is not this run's obligation, and a retained
record is not necessarily unfinished. Preserve membership through waiting-to-granted holds
and queued-to-active handoffs. Check completion after the full transition and follow-ups.

No concrete complexity or measured performance reason justified a ledger.
The following naming table and struct comment archive the rejected conditional ledger design, not an
approved new collection. The lifecycle invariants apply to either design; association at
admission need not mean inserting into a second registry.

### Names and ownership

| Existing name or mechanism | Replacement | Responsibility |
| --- | --- | --- |
| `Runs` | Existing run state over owner-derived liveness; `VoiceWorkTracker` only if needed | Preserve run identity/result state without prescribing a second completion component |
| `Owned` | `VoiceWork` if the existing key enum remains necessary | Identify speech, on-enter hook, handoff, or foreground hold using their existing IDs |
| `owned` set | Remove with owner-derived liveness; `pending_work` only if a ledger is justified | Unfinished work attributed to this run |
| `RunOwn` signal | Remove | Register through the work's existing creation/admission path |
| Scattered `run_own` and `run_release` coordination | Existing owners maintain membership and terminal state; update a ledger only if selected | Keep attribution and completion at the same lifecycle boundary |
| Session-wide output map with scattered removal | Result data attached to the work or run whose lifecycle governs it | Preserve required result data until selection, then release all remaining entries |

If a separate ledger is retained, use one consistent name family: `VoiceWorkTracker` tracks
`VoiceWork` in `pending_work`.
Both type names identify the voice domain. `pending_work` remains a private field; it does
not need a redundant prefix inside its owner. Neither type is a new required primitive.

`VoiceWork` would rename the existing key enum, not add a second resource registry or
per-work wrapper. Do not introduce task futures, locks, RAII completion guards, or generic
completion tokens into the synchronous core for this change. Existing work IDs and the
session's synchronous ordering are sufficient unless a concrete correlation gap is found.

If a tracker is justified, use a comment explaining the invariant, not merely translating
the type name:

```rust
/// Tracks unfinished work that keeps a voice run open.
///
/// Speech (including playout settlement), on-enter hooks, handoffs,
/// and foreground holds register
/// before they can complete. The run settles after all tracked work
/// finishes and its run-attributed follow-up work has settled.
/// Unrelated session work does not extend this run.
///
/// This is completion bookkeeping, not Rust memory ownership.
struct VoiceWorkTracker {
    // Existing state, reorganized around the lifecycle contract below.
}
```

### Lifecycle contract

1. Register run work at its creation/admission boundary, before any path can finish, refuse,
   interrupt, or release it. The speech admission owner must cover successful and rejected
   speech through the same lifecycle; an already-finished speech cannot become pending.
2. Complete or remove work at the corresponding terminal boundary. The owner handles success,
   refusal, timeout, interruption, and close. Other components request a domain transition;
   they do not remember a separate ownership update afterward.
3. Replace work without a premature completion gap. In particular, register the incoming
   agent's on-enter hook and any immediately created work before releasing the handoff.
   For queued/refused handoffs, settle the exact handoff ID.
4. Check run completion only after the current transition and its required follow-ups settle.
   An empty set observed midway through a handoff or hook transition does not finish a run.
5. Keep caller result selection separate from unfinished-work accounting. Preserve the result
   until its defined selection point; release non-selected values and all remaining result
   references when their owning run/work ends, including session close.
6. Handle duplicate terminal notifications idempotently, but make attempted registration of
   terminal work and missing terminal paths detectable in focused tests and diagnostics.

Existing domain operations maintain their lifecycle records and run attribution. If a tracker
is justified, only those operations update it. For handoff, the parent can
register the handoff ID returned by admission before advancing the transition; it does not
need a general `RunOwn` signal. Preserve the registration-before-completion ordering even if
admission and immediate refusal happen synchronously in one method.

### Audit findings to fix

These are source-level failure paths requiring regression coverage before the redesign is
treated as verified:

| Finding | Current path | Required correction |
| --- | --- | --- |
| A rejected speech can be tracked after it finishes | `start_speech` releases rejected speech before its later `run_own` call; realtime also attaches after `start_speech` returns | Register before terminal paths, or consume rejection as a terminal outcome without inserting finished work |
| Non-final owned outputs can remain retained | `release` replaces the last-speech reference; settlement removes only that speech's output and close does not clear the map | Give intermediate outputs an explicit lifetime and release every non-selected value |

Sources: [run_result.rs](../../rust/crates/lk-agents-voice/src/run_result.rs),
[speech admission](../../rust/crates/lk-agents-voice/src/agent_activity/speech.rs),
[realtime admission](../../rust/crates/lk-agents-voice/src/agent_activity/realtime/mod.rs), and
[handoff](../../rust/crates/lk-agents-voice/src/agent_activity/handoff.rs).

### Resolved run behavior contracts

| Question | Implemented contract |
| --- | --- |
| What belongs to a run? | Existing lifecycle records carry explicit membership for session-requested speech, handoffs and on-enter work; incidental provider/filler work is excluded. Holds inherit attributed tool-speech membership. |
| Which result wins? | Last completed attributed speech selects output/error; established failure precedence remains. Nonselected values are released. |
| Can a run open before work exists? | Native capture mode waits for first participating admission; empty capture does not settle immediately. |
| What does cancelling the awaiter do? | Python await shielding leaves voice work running; waiter cancellation remains separate from cancelling work. |
| What happens on close? | Exactly-once run settlement clears result references; existing domain owners stop resources. |

No `VoiceWorkTracker`, `VoiceWork` or duplicate pending-work ledger was needed. The historical
conditional design below explains why it was considered; it is not another implementation task.

## Generation inventory

The driver's private effect handling must not become an API imposed on every generation helper.
Review each type according to whether it expresses domain policy, an ordered operation,
or a redundant representation.

Source groups:

- [LLM generation](../../rust/crates/lk-agents-voice/src/generation/machine/llm.rs) and its
  [runtime adapter](../../rust/crates/lk-agents-runtime/src/voice/generation/llm.rs).
- [Forwarding and preparation](../../rust/crates/lk-agents-voice/src/generation/machine/forward.rs).
- [TTS and transcripts](../../rust/crates/lk-agents-voice/src/generation/machine/tts.rs).
- [Playout and completion](../../rust/crates/lk-agents-voice/src/generation/machine/playout.rs).
- [Pipeline startup](../../rust/crates/lk-agents-voice/src/generation/machine/pipeline.rs) and
  [realtime generation](../../rust/crates/lk-agents-voice/src/generation/machine/realtime_generation.rs).

| ID | Types or boundary | Opportunity | Direction and risk |
| --- | --- | --- | --- |
| G01 | Former `WordEffect` / runtime `SegEvent` | Duplicate segment protocol | Implemented: one `SpeechSegmentEvent` used directly by generation/runtime, preserving end semantics |
| G02 | Former `ForwardEffect::StartedSpeaking` | Single-variant observation wrapper | Implemented: ordinary first-output boolean from the turn accumulator; existing publications remain canonical |
| G03 | Former `TextLegEvent` / `TextLegEffect` | Small capture/flush protocol | Implemented: ordinary capture/flush-once operations, policy retained in the core |
| G04 | Former gate `PlaybackEvent` and `PlaybackEffect` | Implemented: ordinary authorization methods and `PlaybackAuthorization::{Pending, Authorized, Interrupted}` | Preserve authorization outcomes inside GenerationPreparation; no gate event/effect wrapper to recreate |
| G05 | Former `PreparationEvent` and `PreparationEffect` | Implemented: GenerationPreparation owns readiness and step-wide authorization | Preserve speculative synthesis and multi-segment/text-only authorization; extract the existing scoped protocol without another layer |
| G06 | `TurnAccumulatorEvent` and `TurnAccumulatorEffect` | Sequence reading, forwarding, and ending | Retain when they express policy; make the owner and operation names explicit |
| G07 | Private `AudioPlaybackAccountingEffect` | Implemented: AudioPlayback consumes accounting results and exposes its own publications | Keep accounting private and preserve trace/start ordering; further helper removal is optional |
| G08 | `StreamingTtsEffect` | Orders transcript capture, TTS input, and completion | Keep ordered operations; avoid bypassing capture-before-push behavior |
| G09 | `AlignedTranscriptEffect` | Orders aligned or fallback transcript capture, warnings, and flush | Keep alignment and flush-once guarantees; simplify adapters where redundant |
| G10 | `SynthesisLossEvent` and `SynthesisLossEffect` | Express generation failure and interruption outcomes | Retained as scoped SynthesisLoss policy; preserve failure/terminal semantics |
| G11 | `ReplyStartupEvent` and `ReplyStartupEffect` | Express reply startup | Retained as scoped ReplyStartup protocol; preserve startup admission/resources |
| G12 | `RealtimeEffect` and `RealtimeControl` | Realtime-generation output and controls need explicit owner scope | Implemented: RealtimeGenerationEffect / RealtimeGenerationControl under RealtimeGeneration; preserve provider generation versus physical playback/connection distinctions |
| G13 | `CutEvent` and `AudioPlaybackCutEffect` | Bound the opportunity to receive a cut playback receipt | Implemented inside AudioPlayback; retain private receipt policy and preserve probes, bound, closing-clock behavior, and unheard fallback |
| G14 | `FinishEvent` and `AudioPlaybackFinishEffect` | Sequence drain, receipt acceptance, cutting, and completion | Implemented in AudioPlayback completion phases; preserve explicit transitions and heard-text/error accounting |
| G15 | `FlowEvent` and `AudioPlaybackFlowEffect` | Sequence interruption checks, reads, launch, frames, tails, and boundaries | Implemented through AudioPlaybackEvent / AudioPlaybackEffect; retain private forwarding policy and decisions in core |

The strongest adjacent simplification candidates are G01 through G03. A small enum is not
automatically unnecessary: a three-way domain verdict is often clearer than booleans or a
tuple. Conversely, a one-variant wrapper does not automatically represent a useful concept.

### Playback and forwarding vocabulary review

Former spellings in the table identify migration provenance; the main status table and
current effect inventory record what is implemented. Responsibilities must be apparent
without opening the module:

| Current vocabulary | Actual responsibility | Simplification / naming direction |
| --- | --- | --- |
| `CutReceipt`, `CutEvent`, `AudioPlaybackCutEffect` | After audio is cut, give its asynchronous playback receipt a bounded chance to arrive so heard text/position remain accurate | Make receipt waiting an internal phase of `AudioPlayback`; preserve the policy without a separate runtime-facing cut protocol |
| Gate `Wait`, `Play`, `Refuse` | Await authorization, permit forwarding under existing authorization, or stop because interrupted | These do not physically play/refuse an output. Prefer gate methods and a meaningful three-way outcome as in G04; remove event wrappers if they add no ordering/correlation value |
| `SegmentFlow`, `FlowEvent`, `AudioPlaybackFlowEffect` | Decide audio-source reads, first-frame launch permission, frame/tail forwarding, boundaries, and completion | Selected family: `AudioPlayback`, `AudioPlaybackEvent`, `AudioPlaybackEffect`; combine with receipt/completion ownership, keeping decisions in core |
| `AudioForwarder`, `ForwarderCommand`, `ForwarderEffect` | Microphone input fan-out to STT/VAD/other installed sinks, gating, resampling, ending inputs, and reports | Selected family: `AudioInputRouter`, `AudioInputRouterCommand`, `AudioInputRouterEffect`; distinct from generated audio flowing toward playback |
| `Pacer`, `PaceEvent`, `PaceEffect` | Release transcript words against audio playhead/alignment and report heard text, flushes, and warnings | Implemented: direct TranscriptSync effects replace nested pacing instructions; TranscriptPacer and its protocol remain private |

These dispositions cover retained and already-grouped responsibilities; they do not authorize
adding five new protocols. Prefer fewer ordinary methods/results first. Keep scoped names in
imports; no aliases back to `FlowEvent`, `PaceEffect`, or generic `Effect`. Update matching
producer/consumer names together and use imperative variants only for actual requested work.
Words like `Flow`, `Pace`, `Cut`, and `Forwarder` alone are insufficient domain scope.
One-variant `ForwardEffect::StartedSpeaking` is separate from `AudioInputRouterEffect`;
G02 still assesses removing that wrapper instead of expanding its name.

### Forwarding scope and long-term direction

Implemented input owner: `AudioForwarder` is now `AudioInputRouter`, with
`AudioInputRouterCommand` and `AudioInputRouterEffect`. This supersedes the earlier
`AudioInputForwarder*` and retained-`AudioForwarder*` proposals. `AudioInput` stays the source
interface; routing, resampling, gating, and attached consumers belong to the router.
`PlaybackAuthorization` replaces the proposed `PlaybackGateOutcome` name, with `Pending`,
`Authorized`, and `Interrupted` outcomes.

Verified naming constraint: `AudioInput` already names the public Python frame-source class
and the ABI source trait (`next_frame`, attach/detach notifications), both used by runtime.
The existing chain is `AudioInput` source -> runtime reader -> `AudioForwarder` -> consumers.
Renaming the forwarder to `AudioInput` would give two different responsibilities the same name;
module separation would not remove that ambiguity. `SessionAudioInput` was considered and withdrawn; the implemented name `AudioInputRouter`
keeps the distinction visible without renaming the existing source interface.
See the [ABI source contract](../../rust/crates/lk-agents-abi/src/voice/io.rs),
[public Python source](../../livekit-agents-next/livekit/agents/voice/io.py), and
[runtime reader](../../rust/crates/lk-agents-runtime/src/voice/mod.rs).

Today this forwarder accepts `AudioFrame` and implements audio-specific channel fan-out,
resampling, silence substitution, and input accounting. Video does not use it: video input is
read by a runtime task, sampled in the voice core, and accepted images are sent to the realtime
model. See [audio forwarding](../../rust/crates/lk-agents-voice/src/data_plane/mod.rs) and
[video decisions](../../rust/crates/lk-agents-voice/src/agent_activity/io.rs).

The long-term goal is understandable owners for session input, generation, playback, and
transcript synchronization, not a generic media-forwarding framework. Keep common attachment,
resource, and teardown behavior shared where their contracts match; keep audio processing and
video sampling distinct. A future shared owner is justified by concrete shared transitions,
not by both carrying frames. Do not add a speculative video event/effect pair, a media enum,
or another top-level primitive for symmetry. Choose the shortest name that unambiguously
identifies the actual owner in an import; specificity does not require stacking qualifiers.

Output counterpart: the existing `AudioOutput` is the sink interface, symmetric with the
`AudioInput` source interface. There is no current `AudioOutputRouter` matching the input
router. Generated audio passes through playback authorization, segment forwarding/resampling,
and playback receipt/completion accounting before/alongside the output sink. These owners
have different obligations from microphone fan-out. Keep their state machines; do not add or
merge an output-router abstraction solely to make the input/output names symmetrical.
See [output sink](../../rust/crates/lk-agents-abi/src/voice/io.rs),
[playback decisions](../../rust/crates/lk-agents-voice/src/generation/machine/playout.rs), and
[playback execution](../../rust/crates/lk-agents-runtime/src/voice/generation/playout.rs).

### AudioPlayback groups the playback lifecycle

`AudioPlayback` is the selected owner, replacing the earlier `AudioSegmentForwarder*` and
standalone `PlaybackCutReceipt*` proposals. Its one runtime-facing protocol is
`AudioPlaybackEvent` / `AudioPlaybackEffect`. Group source/frame forwarding, segment boundary
handling, interruption, bounded cut-receipt waiting, and completion under this lifecycle.
`AudioPlaybackCutEffect` is therefore not merely renamed into another independently exposed protocol.

Preserve explicit internal states and the existing policy. Reuse the current forwarding,
receipt, finishing, and accounting logic as private state/helpers where useful; grouping does
not require flattening them into one large match or moving decisions into runtime. Do not add
`AudioPlaybackEffect::Cut(AudioPlaybackCutEffect)` and keep the old protocols underneath as additional
runtime-facing concepts: the integration must reduce the boundary vocabulary.

The core owner and runtime migration are implemented; declarations now live in
`generation/effects.rs`, with owner transitions in `generation/machine/`. The runtime previously composed these
responsibilities itself in one segment's forwarding and completion path; it now executes the
core owner's protocol. Focused tests and deterministic regressions verify the selected paths,
not an exhaustive proof of all interleavings. Pin transitions from
forwarding into clean finish, interruption/receipt wait, failure, and terminal completion;
preserve receipt bounds, closing-clock behavior, source-without-sink behavior, first-frame
launch holds, resampler-tail/flush order, and exact heard-text/position accounting. Stale and
duplicate outcomes must not repeat effects or terminal delivery.

Keep speech/floor scheduling outside this owner. The generation-step `GenerationPreparation` owner and its
`PlaybackAuthorization` result also remain distinct: their authorization spans synthesis
segments and can apply when no audio output exists. Group by lifecycle ownership, not by every
type containing the word “playback.”

### Concrete names for retained state machines

Keep the state machines and their explicit transitions. The following names are the selected
replacement vocabulary for the current responsibilities. Playback, transcript, and preparation
groupings and the `AudioInputRouter` rename are implemented in the working tree. Effect
definitions have moved to dedicated category files and passed the structural validation gates.
Naming no longer depends on deciding whether a later simplification removes a wrapper. A
retained protocol uses its owner's full name consistently; removed types get no alias.

| Existing owner / protocol | Selected owner / protocol | Why |
| --- | --- | --- |
| `CutReceipt`, `CutEvent`, `AudioPlaybackCutEffect` | Receipt-wait phase within `AudioPlayback`, using its event/effect boundary | Preserve bounded playback receipt accounting without another protocol to learn |
| Core `PlaybackGate`, `PlaybackEvent`, `PlaybackEffect`, `SynthesisPreparation` | `GenerationPreparation` and `PlaybackAuthorization` | One step-wide preparation/authorization owner; distinct from physical sink `PlaybackEvent` records |
| `SegmentFlow`, `FlowEvent`, `AudioPlaybackFlowEffect`, `SegmentFinish` and finish protocol | `AudioPlayback`, `AudioPlaybackEvent`, `AudioPlaybackEffect` | Own forwarding, interruption, receipt waiting, and segment completion as one lifecycle |
| `AudioForwarder`, `ForwarderCommand`, `ForwarderEffect` | `AudioInputRouter`, `AudioInputRouterCommand`, `AudioInputRouterEffect` | Incoming microphone audio routed to consumers, not generated playback |
| `Pacer`, `PaceEvent`, `PaceEffect`, `PaceWarning`, `PacerOptions` | Public `TranscriptSyncEvent` / `TranscriptSyncEffect`; private `TranscriptPacer` state | One runtime boundary; scoped configuration/warnings remain exposed, internal pacing protocols do not |

Clarify ambiguous variants at the same time, preserving their payloads and behavior:

| Existing qualified variant | Replacement |
| --- | --- |
| `PlaybackEffect::Wait` | `PlaybackAuthorization::Pending` |
| `PlaybackEffect::Play` | `PlaybackAuthorization::Authorized` |
| `PlaybackEffect::Refuse` | `PlaybackAuthorization::Interrupted` |
| `AudioPlaybackCutEffect::Probe` | `AudioPlaybackEffect::PollCutReceipt` |
| `AudioPlaybackCutEffect::Yield` | `AudioPlaybackEffect::YieldForCutReceipt` |
| `AudioPlaybackCutEffect::WaitUntil` | `AudioPlaybackEffect::WaitForCutReceiptUntil` |
| `AudioPlaybackFlowEffect::Read` | `AudioPlaybackEffect::ReadSource` |
| `AudioPlaybackFlowEffect::WaitLaunch` | `AudioPlaybackEffect::WaitForLaunchPermission` |
| `AudioPlaybackFlowEffect::Frame` | `AudioPlaybackEffect::ForwardFrame` |
| `AudioPlaybackFlowEffect::Tail` | `AudioPlaybackEffect::ForwardResamplerTail` |
| `FlowEvent::Launched(bool)` | `AudioPlaybackEvent::LaunchPermissionResolved { allowed: bool }` |
| `ForwarderEffect::Push` | `AudioInputRouterEffect::PushFrame` |
| `ForwarderEffect::Report` | `AudioInputRouterEffect::ReportToSession` |
| `PaceEffect::Forward` | `TranscriptSyncEffect::PublishWord` |
| `PaceEffect::Flush` | `TranscriptSyncEffect::FlushTranscript` |
| `PaceEffect::Heard` | `TranscriptSyncEffect::ReportHeard` |

The first-frame launch hold is separate from the generation-step playback authorization gate;
the names must preserve that distinction.

Already descriptive variants such as `EndInput`, `PlaybackFinished`, and `BoundaryFlushed`
need no change solely to lengthen them. Qualify generic completion/readiness variants by the
retained owner; inspect remaining event names with their actual producers before renaming.
No extra command, callback, queue, or state machine is introduced by these replacements.
A name-only pass preserves core policy, effect order, interruption precedence, receipt bounds,
source-without-sink behavior, transcript alignment, and flush/terminal accounting. Keep it
separate from optional removal of adapters or event wrappers, with no compatibility aliases.

## Current effect ownership map

This source snapshot contains **36 effect enums** (core 5, voice 23, media 1, inference 7).
This inventory covers handwritten production Rust declarations in product crates,
excluding generated bindings, build tools, simulations, and test subjects. It counts explicitly
named effect types, not every command/action/operation that might request work. It is a map
of current declarations after the completed refactor, not a proposed hierarchy.

There is no global root effect enum. Most families are independent protocols interpreted by
the executor for their owner. A module/subsystem relationship is not an enum-containment edge.

### Actual nesting and projection

- `TextSessionEffect::Run { effect, .. }` contains `AgentRunEffect`.
- `AgentRunEffect` flattens `ToolCallEffect` through conversion into invocation variants;
  it does not carry a nested tool effect enum.
- Voice `agent_session::effects::VoiceEffect` contains `AudioOp`, `VoiceHostEvent`, and
  `AudioInputRouterCommand` payloads. Those are operations, host requests, and commands, respectively.
  It does not contain `AudioPlaybackEffect`, `TranscriptSyncEffect`, or `RoomTranscriptionEffect`.
- `AudioPlayback` owns private Flow/Cut/Finish/Playout machines and translates them into its
  one public effect protocol. The old helpers are not nested runtime-facing effect variants.
- `TranscriptSync` owns its private pacer; pacing effects become direct sync effects.
- Avatar `AvatarEffect::Io` contains `IoOp`; A2A `A2aTaskEffect::Execute` contains `Operation`.
- `TransferToolEffect::Settled` contains the `TransferEffects` aggregate.
- Provider/inference effect families are peers. In particular, native TTS synthesis effects
  do not contain the PCM emitter's effects.

### Room boundary: current versus proposed

`RoomIo` is the integration owner, but **there is currently no `RoomIoEffect` enum**. Its
participant/attachment methods use ordinary return values; room transcription has the separate
`RoomTranscriptionEffect` protocol. Other room children use their own operations/results.
Renaming `WriterEffect` did not unify those protocols. A unified room effect boundary remains
a design candidate, not an implemented hierarchy or a requirement to wrap every child enum.
The earlier explanation of room ownership must not be read as actual effect containment.

### Complete enum inventory

`shared type` means visible beyond its immediate module; it does not mean public Python API
or that runtime consumes it directly. All entries below link to their actual category
declaration files; private helper effects are not public crate-index exports.

#### lk-agents-core

| Current type and declaration file | Visibility / responsibility |
| --- | --- |
| [`a2a::task::effects::A2aTaskEffect`](../../rust/crates/lk-agents-core/src/a2a/task/effects.rs) | shared type: Durable A2A task progression; Execute actually contains Operation. |
| [`agent_run::effects::AgentRunEffect`](../../rust/crates/lk-agents-core/src/agent_run/effects.rs) | shared type: Shared invocation work; ToolCallEffect is projected into these variants. |
| [`text_session::effects::TextSessionEffect`](../../rust/crates/lk-agents-core/src/text_session/effects.rs) | shared type: Backend routing/continuation; Run actually contains AgentRunEffect. |
| [`tool_calls::effects::ToolCallEffect`](../../rust/crates/lk-agents-core/src/tool_calls/effects.rs) | shared type: Tool argument preparation, approval, admission, execution and cancellation. |
| [`toolsets::effects::ToolsetEffect`](../../rust/crates/lk-agents-core/src/toolsets/effects.rs) | shared type: Shared setup, grants, release, rollback and close. |

#### lk-agents-voice

| Current type and declaration file | Visibility / responsibility |
| --- | --- |
| [`agent_session::effects::VoiceEffect`](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs) | shared type: Session coordination; contains AudioOp, VoiceHostEvent and AudioInputRouterCommand payloads. |
| [`amd::effects::AmdEffect`](../../rust/crates/lk-agents-voice/src/amd/effects.rs) | shared type: Answering-machine classification, deadlines and voice-floor coordination. |
| [`avatar::effects::AvatarEffect`](../../rust/crates/lk-agents-voice/src/avatar/effects.rs) | shared type: Avatar runner work; Io actually contains IoOp. |
| [`audio_recognition::effects::AudioRecognitionEffect`](../../rust/crates/lk-agents-voice/src/audio_recognition/effects.rs) | shared type: Recognition domain actions; journal-only observations use a separate scoped observation type. |
| [`data_plane::effects::AudioInputRouterEffect`](../../rust/crates/lk-agents-voice/src/data_plane/effects.rs) | shared type: Microphone frame routing, conversion, gating, sink input closure and core reports. |
| [`generation::effects::StreamingTtsEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | shared type: Ordered transcript capture and TTS text feed. |
| [`generation::effects::AlignedTranscriptEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | shared type: Aligned/fallback TTS transcript capture and flush. |
| [`generation::effects::SynthesisLossEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | shared type: Synthesis failure/interruption policy. |
| [`generation::effects::AudioPlaybackAccountingEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | private helper: AudioPlayback sink accounting/publication helper. |
| [`generation::effects::AudioPlaybackCutEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | private helper: AudioPlayback bounded cut-receipt helper. |
| [`generation::effects::AudioPlaybackFinishEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | private helper: AudioPlayback completion/error helper. |
| [`generation::effects::AudioPlaybackFlowEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | private helper: AudioPlayback source/forwarding helper. |
| [`generation::effects::RealtimeGenerationEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | shared type: Voice realtime generation text/audio/tool/boundary projection. |
| [`generation::effects::AudioPlaybackEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | shared type: Generated-audio forwarding, interruption, bounded cut receipts and completion. |
| [`generation::effects::GenerationPreparationEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | shared type: Step-wide readiness, speculative synthesis admission and authorization. |
| [`generation::effects::ReplyStartupEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | shared type: Reply startup, distinct from generation-step preparation. |
| [`generation::effects::TurnAccumulatorEffect`](../../rust/crates/lk-agents-voice/src/generation/effects.rs) | shared type: Sequential generation segment reads, forwarding and end. |
| [`ivr::effects::IvrEffect`](../../rust/crates/lk-agents-voice/src/ivr/effects.rs) | shared type: IVR navigation work and outcomes. |
| [`keyterm_detection::effects::KeytermEffect`](../../rust/crates/lk-agents-voice/src/keyterm_detection/effects.rs) | shared type: Keyterm request/push/cancel work. |
| [`room_io::effects::RoomTranscriptionEffect`](../../rust/crates/lk-agents-voice/src/room_io/effects.rs) | shared type: Room transcript publication; independent executor protocol, not a RoomIoEffect child. |
| [`transcription::effects::TranscriptSyncEffect`](../../rust/crates/lk-agents-voice/src/transcription/effects.rs) | shared type: Transcript attachment/close lifecycle and direct word/flush/heard output. |
| [`transcription::effects::TranscriptPacerEffect`](../../rust/crates/lk-agents-voice/src/transcription/effects.rs) | private helper: TranscriptSync word-pacing helper. |
| [`workflows::effects::TransferToolEffect`](../../rust/crates/lk-agents-voice/src/workflows/effects.rs) | shared type: Warm-transfer participant movement/settlement; Settled contains TransferEffects. |

#### lk-agents-media

| Current type and declaration file | Visibility / responsibility |
| --- | --- |
| [`emitter::effects::AudioEmitterEffect`](../../rust/crates/lk-agents-media/src/emitter/effects.rs) | shared type: PCM TTS emission/flush scheduling/diagnostics; performed by inference. |

#### lk-agents-inference

| Current type and declaration file | Visibility / responsibility |
| --- | --- |
| [`effects::SttStreamEffect`](../../rust/crates/lk-agents-inference/src/effects.rs) | shared type: Provider recognition stream/finalization; executed by SttStreamExecutor. |
| [`effects::TtsSynthesisEffect`](../../rust/crates/lk-agents-inference/src/effects.rs) | shared type: Native provider synthesis; does not contain AudioEmitterEffect. |
| [`effects::LlmRequestEffect`](../../rust/crates/lk-agents-inference/src/effects.rs) | shared type: One provider model request, including attempts/cleanup. |
| [`effects::FallbackChainEffect`](../../rust/crates/lk-agents-inference/src/effects.rs) | shared type: Provider retry/fallback-chain coordination. |
| [`effects::RealtimeConnectionEffect`](../../rust/crates/lk-agents-inference/src/effects.rs) | shared type: Provider connection/reconnection, distinct from voice realtime generations. |
| [`effects::AvatarSessionEffect`](../../rust/crates/lk-agents-inference/src/effects.rs) | shared type: Avatar provider session lifecycle. |
| [`effects::InterruptionEffect`](../../rust/crates/lk-agents-inference/src/effects.rs) | shared type: Interruption provider transport lifecycle. |

### Containers and other operation vocabulary

These are not additional effect enums:

| Type | Role |
| --- | --- |
| Voice warm-transfer `TransferEffects` | Aggregate of cleanup/I/O changes, also carried by `TransferToolEffect::Settled` |
| ABI `voice::avatar_queue::QueueEffects` | Host-boundary result record for queue start/finish/clear/flush/reset observations |

`AudioOp`, `AvatarIoOperation`, A2A `Operation`, and provider commands also represent work
without `Effect` in their name. The generic session collector and `StepOp` are removed. Events, callbacks, and commands are distinct directions
or roles, not more levels in one effect tree. No named effect enums were found in runtime,
net, storage, worker, or CLI production sources by this declaration scan; those crates still
execute work and expose their own operation/port APIs.

Surviving protocols carry scoped owner names and retain their ordering or ownership purpose. Do not manufacture an all-purpose `Effect` hierarchy or add wrappers merely to
make this map look like a tree.

## Effect definition layout and navigation

The following category declarations, domain machine modules, and scoped owner/effect names
are **implemented and verified**. This pass changes
organization and names, not scheduling, timer semantics, or conversation policy. Additional
protocol reduction and behavioral consolidation are now implemented as recorded in the status checkpoint; final combined validation remains separate.

### Dedicated declaration files per category

Put effect definitions in a dedicated `effects.rs` for the smallest coherent category inside
the owning crate. Group related declarations in one file, not one tiny file per enum and not
one workspace-wide declarations crate. Keep state, transitions, and executor code in their
owner files. The effect file contains contracts, payload documentation, and owner/consumer
links; it must not accumulate dispatch loops, collectors, timer mutation, or state machines.

Provide one crate-level `effects.rs` navigation index where the crate has several categories.
It re-exports the same scoped types without aliases and has a compact ownership map identifying
actual nesting, conversion, and independent execution. It is not another enum or wrapper.
A category's own `effects.rs` is the single-file view of that category's actual definitions.

```text
lk-agents-voice/src/
  effects.rs                    crate overview and named public exports
  agent_session/effects.rs      VoiceEffect declarations only
  agent_session/machine/coordination.rs  private outbox, VoiceCallback FIFO and automatic advancement
  generation/effects.rs         generation/preparation/playback/realtime effect definitions
  transcription/effects.rs      transcript synchronization and private pacing effects
  room_io/effects.rs            room integration effect definitions
  avatar/effects.rs             avatar execution contract
  amd/effects.rs                classification contract
  ...                           other coherent owners, not files per variant

lk-agents-inference/src/
  effects.rs                    actual provider effect definitions, grouped by owner
  llm/, tts/, stt/, ...          each domain has machine/ owners plus execution adapters

lk-agents-core/src/
  effects.rs                    shared execution/conversation overview and named exports
  agent_run/effects.rs           AgentRunEffect
  text_session/effects.rs        TextSessionEffect
  a2a/task/effects.rs            A2aTaskEffect
  ...                           tool-call/toolset declaration categories as appropriate

lk-agents-media/src/
  emitter/effects.rs             AudioEmitterEffect
  emitter/machine.rs             AudioEmitter framing/flush transitions
```

This is ownership grouping, not a promise that every leaf needs its own public protocol or
that the crate index wraps category effects. Private helper effects stay private to their
category and are omitted from public index exports. The generation category keeps private forwarding/receipt/finish/accounting
contracts in `generation/effects.rs` with restricted visibility; playback exposes only
`AudioPlaybackEffect` as its runtime-facing effect boundary.

The inference crate now has one actual declaration file: its seven current effect
families are grouped in `src/effects.rs`. This gives a real single-file view, not merely an
index pointing to scattered definitions. Split into provider-local effect files only if real
owner/dependency or size constraints justify it, retaining one overview. Do not move ABI/host
export declarations out of `lk-agents-abi` to satisfy this layout. Ordinary result aggregates
and commands remain their own roles; relocation is not a reason to recast them as effects.

### Naming corrections grounded in current responsibilities

These scoped declarations and owner names are implemented in the working tree. Former
spellings are retained only to identify the mechanical migration. Generic generation helpers
listed after the table remain separate review candidates.

| Former name | Implemented declaration / owner | Reason |
| --- | --- | --- |
| `SttProtocol` / `SttEvent` / `SttEffect` | `SttStream` / `SttStreamEvent` / `SttStreamEffect` | One provider recognition stream, including audio EOF, final transcript/acknowledgment, quiet-period completion, and deadlines; not one utterance or a transport socket |
| `tts::machine::Effect` / `Event` / `Synthesis` | `TtsSynthesisEffect` / `TtsSynthesisEvent` / `TtsSynthesis` | A native provider synthesis stream; module placement must not carry the missing domain name |
| `LlmRequestEffect` | Retain | Already identifies one provider request, including attempt/cleanup behavior |
| Generation `RealtimeEffect` / `RealtimePlayout` | `RealtimeGenerationEffect` / `RealtimeGeneration` | Handles a provider generation's messages, text/audio, TTS fallback, tool calls, completion/failure, and retained transcript; not just physical playout or all realtime work |
| Generation `RealtimeControl`, `RealtimeObservation` | `RealtimeGenerationControl`, `RealtimeGenerationObservation` | The incoming facts and controls must identify the same generation boundary |
| Inference `ConnectionEffect` | `RealtimeConnectionEffect` | Provider connection/reconnection, separate from voice realtime-generation output |
| Core `agent_run::Effect`, `text_session::Effect`, `a2a::task::Effect` | `AgentRunEffect`, `TextSessionEffect`, `A2aTaskEffect` | Declare the domain in the type, not only in import aliases |
| Voice `agent_session::effects::Effect` | `VoiceEffect` | Session coordination, distinct from generation/playback/room child protocols |

Consistency means explicit subjects and matching owner families. It does not mean calling
LLM requests, TTS synthesis streams, STT recognition streams, provider connections, and
fallback chains the same kind of request. Preserve their different lifetimes and protocol
contracts. Review remaining generic names (`ReplyStartupEffect`, `WordEffect`, `StreamingTtsEffect`,
`AlignedTranscriptEffect`, `SynthesisLossEffect`, `TurnAccumulatorEffect`, and any remaining unspecific
local effect names) against their owner before choosing a concrete scoped replacement or deletion.

### Domain modules with visibly separate state machines

Keep domain names first: `generation`, `room_io`, `transcription`, `stt`, `tts`, and `llm`.
Use a plainly named **`machine` module** inside the domain for synchronous effect-producing
owners. The earlier `state_machines` directory and `_machine.rs` suffix proposals are
withdrawn. Owner filenames and types keep meaningful domain names, with no repeated suffix:
`generation::machine::AudioPlayback` produces `generation::effects::AudioPlaybackEffect`.

```text
generation/
  mod.rs                           domain overview
  effects.rs                       effect definitions and producer/executor links
  machine/
    mod.rs                         owner map and canonical owner exports
    audio_playback.rs              AudioPlayback
    preparation.rs                 GenerationPreparation
    realtime_generation.rs         RealtimeGeneration
  ...                              non-machine helpers and records

room_io/
  mod.rs
  effects.rs
  machine/
    mod.rs
    attachment.rs
    transcription.rs
    playout.rs

# Within lk-agents-inference (async I/O crate)
effects.rs                         provider effect definitions, grouped by producer
stt/
  mod.rs                           client and stream entry point
  machine/
    mod.rs                         exports SttStream; maps private child state
    stream.rs                      stream lifecycle/finalization
    transcript.rs                  transcript accumulator
  executor.rs                      SttStreamExecutor: self-running async I/O
  stream.rs                        live input/output adapter
  wire.rs                          provider message decoding
```

The producer is visible in its module path, and the `effects.rs` file records the exact
mapping, for example:

```text
generation::machine::AudioPlayback         -> AudioPlaybackEffect
generation::machine::GenerationPreparation -> GenerationPreparationEffect
stt::machine::SttStream                    -> effects::SttStreamEffect [inference crate]
```

Input/event declarations remain with their machine unless a separate domain event contract
already owns them; moving effect definitions does not require another events wrapper.

Each effect declaration names its producer and executor in rustdoc, with links to the owner
and execution code. The machine module's `mod.rs` gives the compact ownership map and exports
only the required domain owners. This is a source-organization convention, not a new common
machine trait, public wrapper, registry, or requirement to create an effect for every helper.
Pure records, parsers and utilities stay outside `machine`; private child state stays private.

The STT owner can span multiple utterances. Audio EOF, final transcript, provider finalization
acknowledgment, server EOF, and quiet-period completion remain distinct facts. The transcript
accumulator remains a child responsibility; do not add another effect enum solely to match
the stream naming family. Use the same domain-first arrangement for TTS synthesis and LLM
requests. A `machine` module is synchronous decision code even inside an async I/O crate;
it does not own sockets/tasks or read ambient time. Executors supply outcomes and time.

Each machine documents its explicit states/phases, input/time contract, effects, executor,
and terminal/cancellation behavior. Category effect files contain declarations; machine files
contain transitions; I/O belongs to executors/adapters. The directory/import path therefore
shows directly which code produces effects. The file/module moves are now present in the
working tree; their final full validation is tracked separately below.

### No downstream manual pumping

Downstream code starts an operation, supplies domain input, and awaits/reads its result. It
must not need a subsequent `pump`, `advance`, `step`, or `schedule` call to make accepted
input take effect. The owning coordinator/executor handles returned effects, correlated I/O
outcomes, complete callback transitions, and deadlines automatically until the operation is
terminal or legitimately waiting. A supplied time/wake input to a pure core remains valid;
remembering to drive that input is the executor's responsibility, not the application's or
an unrelated domain handler's. Preserve backpressure and legitimate waiting without spinning.

The STT executor is now `executor.rs` / `SttStreamExecutor` (formerly `pump.rs` / `Pump`).
It is a private, long-running async executor, not a downstream manual-pumping API. Stream
creation starts its `run` once in an owned task; that task handles
socket I/O, queued audio, updates, reconnect attempts, and finalization deadlines. Downstream
consumers use the stream's input/output halves. The rename identifies the existing role and does not change
its lifecycle. See [stream startup](../../rust/crates/lk-agents-inference/src/stt/mod.rs) and
[current executor](../../rust/crates/lk-agents-inference/src/stt/executor.rs).

Voice now follows the same no-manual-pumping rule. `pump_schedule` is removed; the driver's
private complete-transition coordinator advances scheduling and close. Provider executor
naming and effect extraction introduce no new downstream pumping obligation.

### What the single-file overview must show

For each effect family, record its state-machine owner, executor, and visibility. Indent
private child machines under their actual owner. Mark literal payload nesting separately
from conversion into a parent's variants and from independent runtime performers. Show
peer protocols as peers. For example, room owns transcription publication but currently has
no `RoomIoEffect`; moving definitions into `room_io/effects.rs` does not create that enum.
A future unified room boundary needs its own demonstrated ordering/coordination benefit.

### Migration and verification

1. Apply the agreed names in the table above and selected AudioInputRouter family. Extract
   each effect definition once, then move its producer under the domain `machine` module and
   update canonical imports. Review name-only and location-only changes separately from
   behavior. The remaining generic-effect candidates are a later owner-by-owner audit, not
   permission to invent unspecified replacements during this pass.
2. Preserve effective access scope, not literal visibility tokens. Moving a helper under
   `machine/` changes what `pub(super)` permits. Scope or re-export contract payload records
   deliberately so category effects can use them without exposing state or broadly adding
   `pub(crate)`/`pub`. Private child protocols remain absent from public crate-index exports.
3. Preserve payloads, operation identity, variant order, derives, and behavior. `Vocabulary`
   derives emit type-name-based companion modules, coverage functions, and optional kind
   enums at the declaration location. Move/rename those identities atomically with producer,
   fuzzer, sampler, and coverage imports. Do not derive duplicate copies or silently drop
   coverage to preserve an old path; retain every covered/skipped variant obligation.
4. Preserve host ABI/export declarations in ABI and generated output through codegen only.
   Category files may depend on subject/ABI contract records, never on an execution adapter
   merely to name its executor; use rustdoc links for that relationship. Add no crate edges.
   Check location-based architecture/exemption identities before moves. If a move would add a
   forbidden identity, resolve the underlying violation first; do not transfer or enlarge a
   baseline/exemption as a shortcut.
5. Update the effect inventory's current paths, declaration-to-producer links, machine-module
   ownership maps, and source/rustdoc links in the same pass. Category files must show actual
   definitions; crate indexes only re-export canonical scoped names, without compatibility
   aliases or fake containment. No dispatch/collector logic belongs in declaration files.
6. Compile all consumers and run affected core/runtime/DST checks, formatting, architecture
   and generated-drift enforcement. Verify identical dispatch/receipt/timer ordering and
   no manual pumping obligation. Only then mark the structural pass implemented in the
   authoritative status table. Existing grouping test results do not validate future moves.

A Rust file-length threshold alone is not a reason to add another category. Conversely, do
not let a declaration file grow beyond the handwritten-source limit; split along real owner
boundaries while preserving the crate overview as the single navigation entry point.

## Other transition boundaries

These boundaries remain distinct from the driver's private bookkeeping. Their inclusion prevents a
local simplification from expanding into a uniform framework across the workspace.

| Boundary | Current forms | Direction |
| --- | --- | --- |
| Agent execution | `AgentRunEvent`, `AgentRunEffect` | Scoped `AgentRunEvent` and effect implemented. Retain invocation boundary without voice collector types |
| Shared tool execution | `ToolCallEffect` in `tool_calls/effects.rs` | Scoped declaration implemented; retain shared lifecycle policy |
| Text sessions | `TextSessionEvent`, `TextSessionEffect` | Scoped `TextSessionEvent` and effect implemented. Retain routing/invocation composition |
| A2A tasks | `A2aTaskEvent`, `A2aTaskEffect`, `Operation` | Scoped event/effect implemented; retain correlated storage and handler operations |
| Toolset lifetime | `ToolsetEffect` in `toolsets/effects.rs` | Scoped declaration implemented; retain lease/setup/rollback/close policy |
| AMD | Classifier returns `Vec<AmdEffect>` | Removed the single-field `Out` wrapper; `attached` and `step` require callers to use the returned state/effects. `SetDeadline` / `ClearDeadline` match the shared timer vocabulary, without an extra collector protocol. |
| Avatar execution | `AvatarEvent`, `AvatarEffect`, `AvatarIoOperation` | Scoped boundary implemented; retain operation correlation and I/O ownership |
| Audio data plane | `AudioInputRouterCommand`, `AudioInputRouterEffect` | Scoped router family implemented; preserve frame/control ordering and core reports |
| Room transcription | `RoomTranscriptionWriter` / `RoomTranscriptionEffect` (formerly `Writer` / `WriterEffect`) | Explicit names implemented; preserve ordered stream operations and legacy publication behavior |
| Transcript synchronization | `TranscriptSyncEvent`, `TranscriptSyncEffect`, `TranscriptOutput`; private `TranscriptPacerEffect` | Grouping implemented; preserve direct sync effects, private pacing state, ordered words/flush/heard reports and warnings during declaration extraction |
| TTS audio emitter | `AudioEmitter` / `AudioEmitterEffect` (formerly `HostEffect` plus `Effects` alias) | Explicit scoped effects and flush scheduling implemented; retain synchronous framing versus inference decoding/delivery/timer boundary |

Module paths do not replace ownership in event, effect, or decision type names. Use scoped names
even when the defining module is clear, and preserve those names in imports. Do not add an
event/effect pair merely to make every local helper conform to that naming scheme. A private
method can still return a meaningful domain value without another protocol type.

## Related lifecycle groupings: implemented boundaries

The approved follow-ups group by an obligation's lifetime, not by file adjacency:

| Owner | What is grouped | Kept separate |
| --- | --- | --- |
| `AudioPlayback` | Segment forwarding, interruption, bounded cut-receipt waiting, finish and heard-audio accounting | Speech/floor scheduling, generation-step authorization, room transport |
| `TranscriptSync` | Transcript attachment/lifecycle, segment queue, per-segment pacing; direct word/flush/heard/warning effects | Audio sink playback and its receipts |
| Existing `Hooks` | Issue, first accepted response/timeout/local outcome, terminal dispatch, and abandonment | Consumer-specific turn/filler/tool/enter/exit behavior; exit continuation retains FIFO deferral |
| Existing `OutputBindings` / runtime `OutputBinding` | One scoped reserve/activate/cleanup-pending/release protocol; private acquisition and late-settlement guards | Core lease policy versus runtime-owned audio/text resources; not absorbed into audio-only playback |
| `GenerationPreparation` | Readiness, speculative synthesis admission, reusable step authorization with `PlaybackAuthorization` | Per-segment launch holds and physical audio playback |

`OnExitEnded` and the handoff/drain notification variants have been removed. The hook owner's
accepted record stays live until completion dispatch; a queued exit continuation cannot run
after abandonment. Non-exit hooks preserve their inline ordering. No-host fallbacks install
their waiting state before issuance and do not pretend a host operation happened.

Output requests now reach their existing owner through `OutputBindingEvent`; they no longer
spread four unrelated-looking variants through generation dispatch. Cleanup-pending keeps
the reserved/active phase and exact lease intact. Physical settlement can outlive cancellation,
and neither a stale speech ID nor a duplicate release can retire another binding.

The transcript runtime no longer imports the pacing/segment protocols or matches nested
`TranscriptSyncEffect::Pace(PaceEffect)` instructions. `GenerationPreparation` replaces the
separate readiness and gate event/effect loops while retaining one authorization timestamp
across segments, including text-only output. These are explicit synchronous state machines;
I/O and waits still run through runtime ports.

Source: [playback owner](../../rust/crates/lk-agents-voice/src/generation/machine/audio_playback.rs),
[transcript owner](../../rust/crates/lk-agents-voice/src/transcription/machine/lifecycle.rs),
[hook owner](../../rust/crates/lk-agents-voice/src/agent_activity/hooks.rs),
[binding owner](../../rust/crates/lk-agents-voice/src/agent_session/output.rs), and
[preparation owner](../../rust/crates/lk-agents-voice/src/generation/machine/preparation.rs).

### Verification of lifecycle grouping implementation

- Voice: 195 library tests and 9 integration tests passed, including new playback, hook,
  preparation, transcript-ordering, and correlated binding-retirement cases.
- Runtime: 174 library tests passed. Seven socket tests initially hit sandbox bind restrictions;
  rerunning with local socket access passed the full suite.
- Voice/room DST: all 208 release deterministic regressions passed.
- Architecture enforcement passed, with zero warnings in 15 warning-clean crates and no
  broadened baseline/exemption. Checker counterexamples: 47 passed; codegen tests: 132 passed.
- Formatting, scoped whitespace, documentation links/anchors, and stable inventory IDs passed.

Those lifecycle-grouping changes add no public ABI export declarations. Their verification
does not certify the subsequent structural or behavioral passes. Their changes and validation
are recorded separately; PR #8256 question orchestration remains deferred.

### Structural pass checkpoint: verified

The accepted domain/effect layout is present after checkpoint `d6a3fba44`:

- Core's five scoped effect families have dedicated domain declaration files and one crate
  effect index. Machine owners/private transition children moved under domain `machine` modules;
  public owner/input/result paths are re-exported without generic effect aliases.
- Voice effect declarations are grouped by category. `VoiceDriver`, `VoiceRuntime`, and
  `AudioInputRouter` are the implemented names; mixed session coordination lives beside the
  driver in `agent_session/machine/coordination.rs`. Public indexes omit private child effects.
- Inference's seven provider effect families are actual declarations in `effects.rs`; STT,
  TTS, LLM, fallback, connection and avatar/interruption owners keep their distinct lifecycles.
  The STT executor is named `SttStreamExecutor` in `stt/executor.rs`.
- Media's `AudioEmitterEffect` declaration lives in `emitter/effects.rs`, with framing and
  flush transitions in `emitter/machine.rs`. No host export declarations moved out of ABI.

The counts in the preceding lifecycle subsection belong to the earlier grouping checkpoint.
Structural-pass verification:

- Core: 426 library and 5 integration tests passed.
- Voice: 195 library and 9 integration tests passed.
- Inference: 485 library and 58 integration tests passed; 4 tests remained ignored.
- Runtime: all 174 library tests passed with local socket access; automatic review timed out
  before the first launch, and the single permitted retry completed successfully.
- Media: 123 library and 9 native-cache tests passed.
- Fuzz: all-target compilation, all 208 release deterministic regressions, and 25 release
  library/oracle contracts passed.
- Architecture enforcement passed: zero warnings in 15 warning-clean crates, no broadened
  baseline/exemption, and successful purity/I/O/default/all/export-placement gates.
- Checker counterexamples: 47 passed; codegen tests: 132 passed.
- Formatting, whitespace, source inventory links, and stable inventory IDs passed. Rustdoc
  built all four subject/provider crates with broken intra-doc links treated as errors;
  preexisting private-link/style warnings are not counted as code warning-clean checks.

Source moves and documentation reconciliation passed these historical structural gates. The
subsequent behavioral pass implements timer APIs, automatic scheduling, outbox reduction,
run membership/results and lossless requirements; its final combined validation is separate.
PR #8256 remains deferred. No `RoomIoEffect` was introduced.

### Behavioral pass checkpoint: verified

The subsequent [final audit corrections](rust-shared-policy-audit.md#final-audit-corrections-and-python-reference)
close the three follow-up findings: per-call approval answers, full-identity tool receipts,
and metadata on voice failure exceptions. That section records the Python-reference comparison
and the newer combined validation counts. The table below preserves the earlier pass's evidence.

All C/T/S/O/G inventory items now have implemented changes or an explicit retain decision;
there is no remaining architecture implementation batch in this plan. Only PR #8256 is deferred.
The final changes also remove AMD's single-field output wrapper and preserve failed-run input
requirements without converting failures into results or adding question orchestration.

Final verification:

| Gate | Result |
| --- | --- |
| Core | 431 library and 5 integration tests passed |
| Voice | 217 library and 9 integration tests passed |
| Runtime | 176 library tests passed, including actual tool-body lifetime/cancellation ownership |
| Inference | 485 library, 58 integration and 1 scaling test passed; 4 integration cases ignored |
| ABI/media | 67 ABI, 123 media and 9 native-cache tests passed |
| Cargo-linked Python bindings | 129 Python contracts plus 27 native library, 10 adapter, 17 model and 2 reference tests passed; the final exception-metadata assignment also passed its two focused contracts |
| DST | All default targets passed, including 210 production regressions, 25 library/oracle contracts and 4 registration checks |
| Randomized DST | Original eight-seed campaign, 3,000 inputs per seed and per-seed policies passed; all eight runs completed; exact former failing seeds 2 and 5 also passed |
| Architecture | Zero warnings in 15 warning-clean crates; purity and I/O default/all captures and export/dependency/file-length checks passed |
| Checker/codegen | 47 checker counterexamples and 132 codegen tests passed |
| Documentation/style | Four-crate Rustdoc with broken links denied, scoped declaration/link/ID checks, Rust formatting, affected Python Ruff and whitespace checks passed |

The randomized campaign exposed a real tool-body cleanup gap. Runtime now retains the actual
host future through one existing lifetime owner (`ToolBodies`), keyed by captured invocation
and operation. Semantic terminal progress cannot retire its cancellation obligation. Core and
close share a single cancellation claim; stale completion cannot remove a newer body with the
same tool call ID. Cancelled work cannot publish a second terminal result or start new completion
validation. Cooperative and refusing-host regressions preserve the existing cleanup bound.
The refusal oracle's obsolete task label was corrected with the same one-refusal/one-late-body
cardinality; no known-finding baseline or architecture exemption was broadened.

**Concurrent storage isolation:** final whole-workspace architecture and DST verification ran
in an isolated snapshot with byte-identical refactor product sources and committed storage from
`d6a3fba44`. The user confirmed another task was editing storage; those edits were preserved.
The shared working tree's combined integration gate must be repeated after that independent
storage task settles. This qualification does not defer any implementation item in this plan.

Strict mypy still reports 39 errors in 16 files across 739 public/plugin modules; its normalized
diagnostics are unchanged by this refactor. The affected audio/room/STT Python checks pass
(64 + 30 + 2 cases). The full repository Python type gate is therefore not claimed as passing.

### Room scope and emitter clarification

`RoomIo` already provides the room integration entry point. Keep room decisions under
`lk-agents-voice::room_io` and their executors/adapters under `lk-agents-runtime::room_io`;
ABI RTC ports/records remain the SDK capability boundary. All room integration responsibilities
should be discoverable there: participant linking, input tracks/subscription, output
publication/playback, text ingress, pre-connect audio, transcription and coordinated teardown.
Do not add a second room controller or a flat `RoomEffect` envelope merely for grouping.
Child state machines keep explicit names and their own lifecycle obligations.

Implemented room transcription names: `RoomTranscriptionWriter` and
`RoomTranscriptionEffect`. This is room publication, not upstream word pacing. Other generic
room helper names, such as `Publication`, `Admission`, or `Subject`, remain local audit
candidates; inspect their real boundary before renaming or changing visibility. The room
structure is already present; no broader room rearchitecture is claimed by this naming pass.

The emitter is specifically TTS `AudioEmitter`, not a generic `MediaEmitter`. PCM framing
lives in media; inference supplies decoding, delivery, and the injected-clock flush timer.
The public ABI/Python name remains familiar. Native `HostEffect` becomes `AudioEmitterEffect`,
and the `Effects` alias is removed rather than renamed. The effect variants are
`EmitAudio`, `ScheduleFlushAfter`, `CancelScheduledFlush`, `ReportDroppedBytes`, and
`ReportProviderRequestIds`; the timer input is `on_flush_deadline`. This pass preserves
relative delays, ordered delivery outside locks, tail/final-marker behavior, and decoder
settlement. Source methods still use ordinary explicit operations, with no new event wrapper.

Focused verification: room transcription core 6/6, runtime room tests 6/6, room transcription
DST 4/4, PCM emitter 28/28, and inference emitter 7/7 passed. The runtime compilation and
architecture checks cover the renamed native boundary without changing public ABI exports.

These owners are distinct from `AudioPlayback` and `TranscriptSync`: synthesis emission,
voice playback, word pacing, and room publication must not be merged merely because they
carry related audio/text. See the [Rust room guide](rust-core.md#roomio-is-the-room-integration-owner)
and [emitter guide](rust-core.md#tts-audio-emission).

## Verification requirements

Structural changes must preserve these observable and internal ordering contracts. Extend
existing focused tests and deterministic simulations where necessary; this matrix specifies
required evidence, not an assertion that every case already has a dedicated test.

| Contract | Required cases |
| --- | --- |
| Pending work and signal order | Interleave effects, timer operations, and multiple signals; include a signal producing more work and another signal |
| Signal fan-out | Exit completion and speech settlement reaching handoff, drain, and close in the intended order |
| Same-instant timers | An earlier due callback cancels or replaces another due timer; scheduler-owned timers retain their special pass |
| Pending timer queries | Arm then cancel, cancel then arm, replacement, and queries before settlement |
| Deferred resource resolution | Stream replacement/detachment and overlap creation/end within the same step |
| Journal order | Decisions recorded before dependent publications; equivalent behavior with journaling enabled and disabled |
| Host object lifetime | Journal copies do not retain host exceptions; buffers do not prolong resources unexpectedly |
| Session-event processing | Required state, usage, tracing, DTMF, end-call, keyterm, and STT context work occurs before publication |
| Lifecycle without an active step | Startup, close, undelivered commands, and outbox work still publish and settle |
| Two settlement passes | Waiters, runs, deferred replies, scheduling, and provider release progress without needing an unrelated wake |
| Accepted caller outcomes | Refused, superseded, cancelled, and closed requests still receive their defined outcomes and cleanup |
| Voice run work lifecycle | Speech rejected while scheduling is paused/drained; synchronous completion during admission; hook and handoff refusal/timeouts; duplicate terminal notifications; queued handoffs |
| Voice run continuation | Handoff-to-on-enter and hook-to-speech replacements do not settle the run between operations; settle only after required follow-ups |
| Voice run results and retention | Multiple tracked outputs, selected versus incidental speech, earlier failure and later success, and release of non-selected/closed-run references |
| Voice run scope and cancellation | No-input capture, activity membership, nested-run refusal, waiter cancellation, and session-close settlement follow the agreed behavior |
| Speech retirement | Events and updates referencing a speech precede its retirement |
| Generation protocols | Segment end, first speaking observation, flush-once behavior, alignment fallback, and capture-before-push ordering |
| Playback termination | Interruption, missing/late receipts, provider failure, resampler tails, and shutdown preserve heard-text accounting |

Run targeted Rust tests for affected components, relevant runtime contracts, and the affected
deterministic simulations. Binding changes also require the Cargo-linked Python contracts.
Run the applicable architecture and generated-drift checks. Do not broaden exemptions or
manually edit generated output to accommodate a cleanup.

## Architecture and liveness acceptance rules

The cleanup must reduce obligations on callers, not merely rename the same fragile protocol.
Use the existing domain owners, one driver transition sequence, private runtime-effect
storage, and the shared hook-completion route. No additional lifecycle framework, completion
registry, callback priority queue, or scheduling-dirty protocol is approved by this plan.

| Obligation | Owner and rule | Failure to prevent |
| --- | --- | --- |
| Work admission | Its existing domain owner associates identity and run membership before any terminal path | Rejected/already-finished speech becomes pending forever |
| Hook completion | Existing hook lifecycle retains enough identity until shared completion dispatch or abandonment wins | Removing a hook from `pending` while its queued continuation still owes work makes owner-derived completion report idle too early |
| Completion races | One terminal transition wins for a hook ID; timeout, host reply, local fallback, and late duplicate cannot each perform cleanup | Double release, duplicate handoff/close progress, or resurrected abandoned work |
| State before local completion | Caller installs dependent state before queued no-host completion can dispatch | Close waits on a notification already consumed; turn/filler/tool handling misses its result |
| Follow-up work | Complete domain consequences and admit their child work before final run-completion checks | Hook-to-speech or handoff-to-enter creates an observable completion gap |
| Scheduling | Driver automatically checkpoints after complete transitions; eligibility comes from authoritative current state | New work waits for an unrelated event, or a handler must remember to pump |
| Cancel versus settle | Resource/task owner retains cleanup responsibility after requesting cancellation | Cancellation request is mistaken for resource release or operation completion |
| Teardown admission | Owner closes new-task admission before cancellation and joined teardown where permanent shutdown is intended | Concurrently admitted work escapes a cancellation snapshot or keeps teardown replenishing work |
| Effects | Each request has one delivery path through the private outbox/step result contract | Startup/shutdown work is lost or runtime work is executed twice |
| Publication and retirement | Publish required updates before retiring their referenced records/resources | Host receives dangling identities or late terminal updates recreate removed state |
| Abandonment | Invalidate all relevant waits before releasing their run participation; use explicit per-kind fallback rules | Queued completion advances an abandoned exit or replays a turn after close |

For shared hook completion, claim the terminal outcome **when the driver accepts it**,
using the existing hook owner's record. A reply/timeout/local outcome can move an active hook
to completion-pending once; later duplicate outcomes cannot overwrite the accepted cause or
enqueue another completion. Keep that record live through queued dispatch. Dispatch applies
the outcome, admits its consequences, and finishes the hook's lifecycle in one atomic driver
transition. A run settles only after its attributed callbacks/follow-up work, under the
selected membership contract; unrelated waiter work must not silently extend it.

Abandonment invalidates active and completion-pending records before any participating work
is released. A queued callback whose record was invalidated is discarded and cannot advance
an abandoned exit or restore a torn-down turn. Preserve the existing tools-executed fallback
where required. If an outcome was already accepted, retain that terminal cause once and
suppress its continuation; abandonment changes delivery disposition rather than emitting a
second terminal observation/release. If no outcome was accepted, abandonment is the single
terminal cause. Tests must pin both races. No-host hooks also get an owner-managed local lifecycle; preserve
local provenance and do not emit a fake host request or start a host timeout for them.

This may require a private lifecycle state on the existing record, not a second registry of
“completion work,” a new tracker, or an additional public type. The record provides correlation,
terminal ownership, and liveness together. Removing its `pending` entry as soon as an outcome
is received would lose that guarantee while its queued consequences still owe work.

### Progress and legitimate waiting

For each live operation, identify the existing state/ID it waits on and the event or deadline
that can advance it. A blocked speech, outstanding host hook, foreground hold, or future timer
is a valid wait; synchronous processing must stop rather than spin. Preserve intentionally
unbounded user operations where the feature contract allows them. Cancellation and session
close must still have their specified way to release or abandon those waits.

Every immediate-processing iteration must discharge an obligation without an equivalent
replacement, or advance an owner lifecycle toward its next blocking/terminal phase. Consuming
and re-enqueueing the same unchanged callback is not progress. A blocked/unchanged scheduler
does not count as progress. No-op callbacks must not resubmit themselves, duplicate I/O, or
create a same-instant timer cycle; do not hide a loop behind a work-dropping iteration cap. Never resubmit the same
runtime request while waiting for its response. For cleanup owned by an existing timeout,
retain the bound and ensure its timeout path reaches terminal accounting even if a provider
never answers. A cleanup deadline expiring does not mean the underlying resource is gone:
keep physical cleanup ownership explicit until the defined release/retirement boundary.

When a DST reports a stall, report the existing pending hook/speech/request IDs, phase, next
deadline, relevant queue contents, and expected external response. Derive this from owners;
do not add a production tracking framework just to produce diagnostics.

### Required adversarial sequences

Use the voice and room simulations with real runtime executors and port fakes:

- No-host close; instant host completion; timeout then late response; duplicate response;
  abandonment with completion already queued. Assert the terminal consumer runs at most once.
- Hook completion creates speech; handoff creates enter work; waiters create deferred replies.
  Assert run settlement follows admission and processing of those consequences.
- Speech is synchronously refused at admission, interrupted while queued, or cut at close.
  Assert no terminal speech becomes pending and every result reference is eventually released.
- Admission pause with accepted queued speech; AMD authorization pause/resume; handoff
  while host speech and a user turn arrive; close while a swap is queued. Assert required
  progress and the existing owner/gate restrictions, not an invented blanket launch ban.
- Same-instant timer cancellation/rearm and close/watchdog races; missing/late playback receipt;
  failed provider cleanup. Assert stale events cannot revive work and bounded cleanup advances.
- Permanent task-set close races task admission, including admission before the task handle
  is installed. Assert new work is refused or cancelled and joined teardown does not detach it.
- Startup, normal step, late-command refusal, and shutdown each produce their required effects
  once. Repeating an unchanged blocked transition produces no extra effects. An unchanged
  self-requeued callback and a zero-time rearm cycle must be detected as non-progress, not
  accepted because a queue item was consumed.
- A cleanup receipt before, exactly at, and after the shared-run deadline follows the chosen
  event/deadline policy. Also test a receipt queued before the bound but dispatched afterward;
  do not infer arrival order from processing time.

The deferred-reply regression and missing-reactor harness issue are resolved. The coordinator,
owner-derived membership and timer contracts are now implemented too; final combined checks
must validate them independently of the earlier structural gates. Keep mechanical renames,
behavioral changes and generated declaration updates separately reviewable. Source completion
is not a claim that every interleaving has been explored.

## Deferred delegation and question integration

Finish the agreed Rust architecture cleanup before integrating
[PR #8256](https://github.com/livekit/agents/pull/8256). Reconcile it with the existing Rust
expert inbox as a separate behavior change, including its stack prerequisites and outstanding
relay/claim/read-back design. Do not build a second question/delegation implementation during
the core cleanup.

The generic `AgentRun` must report correlated outcomes and input requirements without owning
conversation question prioritization, presentation, or answer routing. Multiple pending expert
questions must be retained and presented one at a time by the conversation owner. The implemented `RunResult.input_requirements` metadata retains all correlated requests even
when the compatibility projection exposes a singular question or completed result. Unselected
requests are not considered answered. Delegation can consume those facts later without a new
question queue inside generic core.

Keep the shared-run reuse, canonical history, duplicate-policy, and cleanup fixes. Review the
question presentation/routing separately when implementing the deferred colleague PR.
See the [question ownership and PR boundary](rust-shared-policy-audit.md#question-policy-is-outside-the-generic-invocation-loop).

## Implementation order and completion criteria

### Phase 1: agreed Rust architecture cleanup

Implemented source changes:

1. Lossless input-requirement metadata, explicit lifecycle-owned run attribution/result
   selection, and preservation of deadline-first equal-time dispatch.
2. One private driver outbox; automatic scheduling/close advancement at complete transitions;
   no component-facing collector, manual pump, dirty flag or callback lookahead.
3. Common hook completion with caller state installed first, deferred exit fan-out, single
   accepted terminal result, no-host fallbacks and abandonment invalidation.
4. Owner-derived completion without `RunOwn`, `VoiceWork` or a second pending-work registry;
   rejected-speech/result-retention cleanup and close value release.
5. Scoped events, effects, owners and results; ordinary generation methods/shared segment
   events replacing G01–G03 wrappers; dedicated declaration files and domain machine modules.
6. Type-removal verification, voice/room DST scope, contributor walkthroughs, fake contracts,
   faithful Arrival baseline and explicit outcome reporting.

Final combined validation remains the acceptance boundary. Targeted regressions cover outbox
exactly-once delivery, deadline queries/before-equal-after wakes, callback-created scheduling,
admission-versus-authorization during handoff, and no launch after close's hard edge. Earlier
structural suite counts remain separate from these results. No new question orchestration is
included; generic metadata preserves every requirement for the later conversation owner.

### Phase 2: delegation/question behavior, after Phase 1

Reconcile PR #8256 and its stack prerequisites with the existing expert inbox and outstanding
relay/claim/read-back design. Implement retention, sequential presentation, answer correlation,
and cross-text/call continuation in the proper owners. Extend the lossless generic metadata with the agreed conversation behavior. This is a separate acceptance gate, not a hidden last step inside
architecture cleanup. Implementing the lossless seam in Phase 1 does not implement Phase 2.

Completion means an ordinary change to scheduling, end-of-turn handling, or interruption can
be understood through its familiar domain methods and the necessary core/runtime boundary.
Contributors should not need to reconstruct collector buffers or journal conversion to
understand that behavior. Fewer wrappers, dispatch steps, and concepts must be necessary to
follow the path; clearer names alone do not meet the goal. Every surviving intermediate type
must have a clear owner and a concrete reason it cannot be an ordinary method or returned
value. Its explanation should name the bug it prevents.
