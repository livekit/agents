# Lifecycle symmetry and effect vocabulary audit

The goal is predictable lifecycles with fewer concepts. Use matching names for matching
operations and milestones. A familiar pair helps a contributor predict where to look; an
unused mirror callback adds another thing to learn.

This audit supplements the [transition plan](rust-transition-inventory.md). The shared-run naming/history cleanup and comment fixes have shipped into the working tree;
common hook terminal handling and playback/transcript/preparation grouping are also implemented.
Other symmetry entries remain proposals unless
the main plan or shared-policy checkpoint explicitly records implementation.
The public feature documents remain the API contracts.

## Hook grouping checkpoint

The common hook owner now retains the first accepted terminal result through dispatch. Exit
uses deferred `HookFinished`; handoff/drain/close continuations run from that shared dispatch,
and `OnExitEnded` is removed. Other hooks retain their inline timing. Local/no-host paths share
the owner lifecycle without issuing a fake host request or deadline. Abandonment invalidates
records before fallback, retains any accepted terminal cause, and suppresses queued continuation.

The original findings below remain review evidence; SY01 is implemented, while unrelated
renames and automatic scheduling/close advancement are not implied complete. The
[main status table](rust-transition-inventory.md#plan-audit-checkpoint) owns sequencing.

## Scope and limits

Repository-wide lifecycle searches covered Rust crates, the public Python package, provider
plugins, the console, and examples. The detailed source audit followed hooks, speech and run
completion, scheduling, output binding, playback, provider requests, room/avatar I/O, worker
processes, storage, and their runtime/host boundaries. Generated bindings were evidence of
contracts, not editing targets. Build-tool code and every individual provider implementation
were not exhaustively reviewed. A search finding is not proof of lifecycle equivalence.

The inventory below records both changes and intentional asymmetries. An operation may end
through an ordinary return, an existing typed result, a shared completion path, or a resource
owner's cleanup. A separate event is needed only when a consumer actually needs that boundary.

## The enter/exit question

`OnEnterEnded` is not missing functionality. Both hooks already share the same lifecycle:

| Layer | Enter | Exit |
| --- | --- | --- |
| Public Python | `Agent.on_enter` | `Agent.on_exit` |
| Host ABI | `SessionHooks::on_enter` | `SessionHooks::on_exit` |
| Runtime request | `HookCall::OnEnter` | `HookCall::OnExit` |
| Host response | `Command::HookDone { id, outcome }` | Same |
| Common terminal processing | `accept_hook_completion` claims the first terminal outcome; `dispatch_hook_completion` performs accounting and continuation once | Same; exit dispatch is deferred |
| Observation | `HookIssued` / `HookFinished`, tagged with `HookKind::OnEnter` | Same, tagged with `HookKind::OnExit` |
| Further domain continuation | No separate activation barrier | Common `HookFinished` dispatch advances exit state, handoff, drain, and close |

Sources: [Python hooks](../../livekit-agents-next/livekit/agents/voice/agent.py),
[ABI hooks](../../rust/crates/lk-agents-abi/src/voice/session_effects.rs),
[hook calls](../../rust/crates/lk-agents-voice/src/agent_session/host_events.rs),
[common hook lifecycle](../../rust/crates/lk-agents-voice/src/agent_activity/hooks.rs),
[runtime execution](../../rust/crates/lk-agents-runtime/src/voice/session/mod.rs), and
[deferred continuation](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs).

Exit gates leaving the current activity. Enter does not gate activation: `update_agent`
resolves when enter is issued, not when it returns. Enter can still produce speech and keep
an open voice run unfinished. The comments saying “Nothing waits on it” are therefore too
broad. Those comments now distinguish activation from participation in an open voice run.
See the [installation contract](../../rust/crates/lk-agents-abi/src/voice/session.rs).

**Implemented direction:** the common hook completion route replaces the redundant
`OnExitEnded` continuation. Do not rename it to `ContinueAfterExitHook`, `TaskComplete`, or
add a paired `OnEnterEnded`. Hook completion is the shared concept; handoff/drain/close are
consumers of an exit hook's completion, not a second hook lifecycle.

The first accepted hook outcome and later abandonment disposition are separate facts. Keep
an already-accepted returned/failed/timeout cause once; abandonment can suppress its queued
continuation without producing a second terminal observation or release. If abandonment wins
before any outcome is accepted, record it as the single terminal cause. Pin both orders in tests.
Run completion must use attributed obligations; changing its position relative to all session
waiters first requires the membership/result contract in the main plan.

### Reuse feasibility and required ordering

The pre-change path was host `HookDone` → `finish_hook` → `hook_ended`, with an additional
exit-only `OnExitEnded` hop. The following historical trace explains the deferred ordering
that the implemented common route must preserve. Current host reply, timeout, and local
outcomes all use terminal acceptance followed by common completion dispatch; exit queues
`HookFinished`, while other kinds keep their inline timing.

**Reuse is feasible, but deleting the queue boundary and calling exit consumers inline is
incorrect.** A concrete source trace demonstrates why:

1. `begin_closing` calls `plan_exit`.
2. `plan_exit` claims the exit and calls `issue_claimed_exits`.
3. Before the refactor, no-host `issue_hook` returned false and `OnExitEnded` was queued.
   The implemented route instead records local completion and queues common `HookFinished`.
4. Only after `plan_exit` returns does close install `session.closing` and its `Exiting` phase.
5. Deferred completion then sees the installed phase and advances close to draining.

If step 3 directly calls the current exit fan-out, `close_exit_done` sees no matching closing
phase. Step 4 then installs a wait whose only completion has already been consumed. Preserve
completion-after-state-installation, including the no-host case.

Use one driver-owned, hook-kind-aware completion dispatch at the established FIFO boundary.
For the proposed driver this can be the shared `VoiceCallback::HookFinished` case, replacing
exit-specific callback plumbing rather than adding a second completion event. Reuse existing
`HookId`, `HookKind`, `HookOutcome`, and terminal-cause data; do not create a new completion
registry or generic task framework. The host response remains a boundary input, not an
additional completion of the operation.

Claim the terminal result when the driver accepts it in the existing hook owner, and keep
that record completion-pending until dispatch. A later timeout/reply cannot replace the
winner or enqueue a second completion. Abandonment invalidates active and completion-pending
records before releasing work; an already queued callback checks that ownership and cannot
revive an abandoned operation. No-host hooks use a local owner-managed lifecycle without fake
host calls/timeouts. This is one record lifecycle, not another completion registry. Run
settlement follows dispatch and follow-up work attributed to that run, not mere result receipt
or every unrelated waiter in the session.

The shared dispatch performs common pending/timer/trace/run-work accounting exactly once,
then calls the actual domain consumers for that hook kind in the same atomic transition.
For exit, preserve exit-state → handoff → drain → close ordering. For enter, common completion
accounting is sufficient. The corresponding `OnExitEnded` variants in `HandoffEvent` and
`DrainEvent` have been replaced with ordinary owner methods called by shared dispatch.
Do not reintroduce those notification variants under new names.

The implementation integrates local/no-host outcomes into the same lifecycle without fake
host responses. Each caller's dependent state is installed before completion can dispatch.
The audited six `issue_hook` call sites are: enter, user-turn completion, user-turn exceeded, filler source,
tools-executed handling, and claimed exit issuance.

| Completion path | Required behavior in the reused route |
| --- | --- |
| Host reply | Correlate a pending hook, process its outcome once, then notify its consumers |
| Timeout | Preserve timed-out provenance and each hook's fallback, including exit progress |
| No host | Complete locally after dependent state is installed; retain the existing hook-specific fallback |
| Enter | Release run participation without introducing an activation barrier |
| Exit | Advance exit state/handoff/drain/close without re-invoking the hook |
| Abandonment | Release bookkeeping without advancing abandoned exits; preserve tools-executed fallback where required |
| Late/duplicate reply | Ignore after terminal processing; never repeat release or continuation |

Do not blanket-defer every existing `hook_ended` call without checking behavior: turn edits,
filler source handling, tools-executed handling, and turn-limit reset are currently inline.
Moving them behind earlier FIFO callbacks changes timing. Compare those paths against their
contracts and the automatic scheduling checkpoint sequence. Install pending-turn, filler
`asking`, tools `handling`, turn-limit, and close/handoff state before any local completion.
No-host filler refusal must still terminate the filler rather than repeatedly decline at the
same instant; no-host turn processing must still use its default edits.

Verification before implementation is accepted: no-host close completes instead of waiting
forever; close/drain/handoff share one exit invocation; enter keeps its existing run/activation
semantics; success/failure/timeout/abandonment/late duplicates preserve per-kind behavior;
preexisting queued callbacks and same-instant timers retain the required order. The shared
route is now implemented with seven focused hook regressions; see the checkpoint and main
plan for validation. Preserve those guarantees during future module/declaration moves; these
results do not prove the pending automatic-scheduling or run-membership redesign.

This naming cleanup must preserve success, host failure, timeout, no-host, abandoned, late,
and duplicate answer behavior. “Finished” describes terminal processing, not guaranteed
success. No-host completion is local; it does not fabricate a host response. The existing
`HookOutcome`/`HookEnd` values preserve the cause.

## Explain the operation before choosing its name

The removed `OnExitEnded` was an internal continuation, not the developer's `on_exit`
callback. Runtime invokes `on_exit` and reports its outcome; common hook terminal dispatch
now accounts for that outcome and advances exit state, handoff, drain, and close. “The hook finished” does not mean the agent has already been swapped or the session
has closed. Prefer an ordinary driver operation such as `finish_agent_exit` for this fan-out
where ordering permits; retain a deferred callback only when its FIFO position is required.
The selected direction above supersedes the earlier `OnExitFinished` rename: reuse shared
hook completion and remove the redundant exit-specific notification, retaining its ordering.

`TaskSet::abort_all` requests cancellation of every task owned by that particular set. It
returns immediately, leaves admission open, and does not guarantee cleanup is complete.
`close` also closes admission; `cancel_and_wait` requests cancellation and waits for the set
to empty. The comments now state these distinctions. `cancel_all` was proposed to replace
misleading “abort” terminology, not to cancel all framework work. If the non-waiting nature
still needs emphasis in the name, consider `request_cancel_all`; choose the method family
together rather than introducing aliases. No cancellation-method rename has shipped here.

`RenderTemplate` asks the host to produce **tool-handling prompt text** from the configured
`tool_handling` templates. Existing `ToolTemplate` cases cover duplicate rejection/confirmation
and instructions for deferred tool replies (`ReplyAtTail`, `ReplyMaybeCovered`). The host uses
the applicable toolset override or agent configuration; a template can be a formatted string
or callable. `None` preserves framework text. This does not call an LLM, synthesize speech,
or render UI. Prefer `RenderToolHandlingTemplate` / `ToolHandlingTemplateRendered` if these
messages remain; the generic `RenderTemplate` recommendation alone was insufficiently scoped.
Update producers, runtime/host adapters, receipt handlers, and declarations together if that
rename is adopted. The existing Python `tool_handling` authoring vocabulary stays the reference.

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

Current source has a real pause/resume operation pair, but different completion requirements:

- `pause_scheduling` installs an admission pause, cancels preemptive generation, and queues
  `Paused(cause, result)` after outstanding speeches are released, or when the wait is refused.
  `PauseScheduling` is an internal request callback used by handoff/drain; close calls the
  method directly. This does not stop accepted speech from draining.
- `resume_scheduling_task` is a synchronous method used at handoff swap. It removes only
  handoff/drain admission pauses and refuses any outstanding callers. There is **no**
  `ResumeScheduling` callback. Close pauses remain installed. Rename the method
  `resume_scheduling`; its `_task` suffix is inherited vocabulary, not a Rust task.
- `Paused(Close, result)` means close's scheduling-drain attempt finished. `Settled` and
  `Refused` both enter the hard edge if close is still draining; the close deadline can also
  enter that edge. The name implies a successful state that the result does not guarantee.
  If this FIFO notification remains necessary, prefer the existing private `VoiceCallback`
  with `SchedulingDrainFinished { cause, result }`; reuse the result enum and routing.
- `CloseProgress` is a request to reconsider the current close phase, not an observation that
  close advanced. Phase entry, speech release, transcript completion, and realtime closure
  trigger it. Prefer coordinator-owned `advance_close` using authoritative prerequisites;
  preserve the phase-installation and deferred ordering before removing the notification.
  Do not replace it with a per-handler pump, dirty flag, or no-op mirror callback.

The symmetry is in method vocabulary and explicit contracts, not identical numbers of events.
Keep admission pause/resume separate from speech authorization holds and terminal close.
The [plan](rust-transition-inventory.md#scheduling-and-close-symmetry) records migration and
adversarial acceptance cases; these replacements remain unimplemented.

Evidence: [pause/resume and settlement](../../rust/crates/lk-agents-voice/src/agent_activity/scheduling.rs),
[callback routing](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs),
[handoff swap](../../rust/crates/lk-agents-voice/src/agent_activity/handoff.rs), and
[close phases](../../rust/crates/lk-agents-voice/src/agent_session/closing.rs).

## Naming rules

1. Identify the owner in event/effect/outcome type names, following the core naming rule.
2. Runtime effects request work using an imperative verb: `StartSpeech`, `CancelSpeech`,
   `RenderTemplate`, `ValidateOutput`. Observations describe what happened:
   `TemplateRendered`, `OutputValidated`. Incoming requests can still arrive as inputs;
   their request role must remain explicit.
3. Normalize words only when the lifecycle milestone is the same. Input EOF, provider EOF,
   accepted output, speech completion, playout settlement, and resource retirement differ.
4. Use common completion handling when behavior is shared. Specialize only the continuation
   that needs domain-specific ordering. Keep its owner and consumer discoverable.
5. Never add a mirror event/effect, alias, generic wrapper, or lifecycle stage for appearance.
6. Preserve Python/provider/SDK vocabulary at those boundaries. Internal Rust renames do not
   require public Python aliases, wire-string changes, or new public notifications.

## Change inventory

Priority measures readability and risk of misunderstanding; all entries still require their
owner/runtime call sites and relevant tests to move together. No new enum is justified by
this table alone; prefer ordinary domain returns when the transition plan removes a wrapper.

| ID | Priority | Current surface | Proposed change and justification | Source |
| --- | --- | --- | --- | --- |
| SY01 | High | `HookDone`, `HookFinished`, `hook_ended`, `OnExitEnded` | Reuse one shared, hook-kind-aware completion dispatch at the FIFO boundary; remove exit-specific notification plumbing. Preserve no-host state-installation ordering; no no-op enter mirror. | [Hooks](../../rust/crates/lk-agents-voice/src/agent_activity/hooks.rs), [dispatch](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs) |
| SY02 | High | Enter-hook comments say nothing waits | State activation versus voice-run completion separately; the current comments hide an existing obligation. | [HookCall](../../rust/crates/lk-agents-voice/src/agent_session/host_events.rs), [issue_on_enter](../../rust/crates/lk-agents-voice/src/agent_activity/hooks.rs) |
| SY03 | High | `SpeechStart` / `CancelSpeech` | Use `StartSpeech` / `CancelSpeech` in scoped `VoiceEffect`. Same imperative convention, same existing variants. | [Voice effects](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs) |
| SY04 | High | `SttStreamOpen`, `VadStreamOpen`, `RealtimeClose`, `CloseRealtimePlayer` | Use `OpenSttStream`, `OpenVadStream`, `CloseRealtimeSession`, `CloseRealtimePlayer`; preserve stream/session/player distinctions. | [Voice effects](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs) |
| SY05 | High | `ActivateLeg` / `DiscardLeg` | Name accepting/rejecting a pending acquisition, e.g. `AdoptOpenedInput` / `DiscardOpenedInput`. These are not activate/deactivate of an already live input. | [Runtime adoption](../../rust/crates/lk-agents-runtime/src/voice/effects.rs) |
| SY06 | High | `TaskSet::cancel(id)` versus `abort_all()` | Rename `abort_all` to `cancel_all` and fix “abort” comments: both invoke cooperative `OwnedTask::cancel`. Keep `cancel_and_wait` as a distinct joined operation. | [Task ownership](../../rust/crates/lk-agents-net/src/task.rs) |
| SY07 | High | `BindOutput`, `ActivateOutput`, `ReleaseOutput` inside `GenEvent` | Make incoming requests recognizable, e.g. `OutputBindingRequested`, `OutputActivationRequested`, `OutputReleaseRequested`, if these wrappers survive. Inspect release semantics before choosing any past-tense name. | [Generation messages](../../rust/crates/lk-agents-voice/src/generation/mod.rs), [handlers](../../rust/crates/lk-agents-voice/src/agent_activity/speech.rs) |
| SY08 | High | `Answer::OutputBinding` / `OutputActivated` | Normalize parallel result names to `OutputBinding` / `OutputActivation`, or both completed-action names if semantics support them. Both currently return the same typed result shape. | [Answers](../../rust/crates/lk-agents-voice/src/agent_session/commands.rs) |
| SY09 | Medium | Core `InstructionsRender`, `OutputValidate`, `TemplateRender` versus voice `ValidateOutput`, `RenderTemplate` | Use `RenderInstructions`, `ValidateOutput`, and scoped `RenderToolHandlingTemplate`; pair with `InstructionsRendered`, `OutputValidated`, and `ToolHandlingTemplateRendered` if the wrappers remain. | [AgentRun](../../rust/crates/lk-agents-core/src/agent_run/mod.rs), [voice](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs) |
| SY10 | Medium | STT `InputEof` / `Wake`, TTS `EndInput` / `Tick` | For equivalent local stimuli, use `InputEnded` / `Wake`; retain `ServerEof` as a different observation. Do not rename the public Python `end_input` method. | [STT](../../rust/crates/lk-agents-inference/src/stt/lifecycle.rs), [TTS](../../rust/crates/lk-agents-inference/src/tts/machine.rs) |
| SY11 | Medium | Generic TTS `Event` / `Effect` versus scoped STT/LLM types | Use `TtsSynthesisEvent` / `TtsSynthesisEffect` if retained, consistent with the existing synthesis owner. No common provider event hierarchy. | [TTS](../../rust/crates/lk-agents-inference/src/tts/machine.rs), [LLM](../../rust/crates/lk-agents-inference/src/llm/lifecycle.rs) |
| SY12 | Medium | TTS `TtsCompleted` / `TtsFailed`, STT/LLM `Finished(Result)` | Evaluate a single typed terminal result for TTS, removing variants if consumers permit. First preserve cancellation and delivery semantics; do not manufacture a terminal receipt for a caller that gave up. | [TTS](../../rust/crates/lk-agents-inference/src/tts/machine.rs), [STT](../../rust/crates/lk-agents-inference/src/stt/lifecycle.rs), [LLM](../../rust/crates/lk-agents-inference/src/llm/lifecycle.rs) |
| SY13 | Medium | Core `LlmEnded`, generation `ChatFinished`, inference `Ended` / `Finished` | Name the actual milestone: provider EOF versus request result versus whole speech result. Consider `LlmRequestFinished` for `ChatFinished` after confirming payload/consumer meaning. No global `Completed` replacement. | [AgentRun](../../rust/crates/lk-agents-core/src/agent_run/mod.rs), [generation](../../rust/crates/lk-agents-voice/src/generation/mod.rs), [LLM](../../rust/crates/lk-agents-inference/src/llm/lifecycle.rs) |
| SY14 | Medium | Generation `PlaybackEvent` / `PlaybackEffect` overlap physical playback and `PlayoutEffect` | Prefer ordinary gate methods; if needed, return `PlaybackAuthorization::{Pending, Authorized, Interrupted}`. Existing `Wait`, `Play`, `Refuse` are authorization outcomes, not physical playback effects. | [Playback gate](../../rust/crates/lk-agents-voice/src/generation/forward.rs), [playout](../../rust/crates/lk-agents-voice/src/generation/playout.rs) |
| SY15 | Medium | `StartedSpeaking`, `PlayoutEffect::Started`, `FirstFrame`, `PlaybackStarted` | Document producer and milestone, collapse forwarding wrappers where possible. Keep physical playback start distinct from the text-only first-output fallback. | [Forwarding](../../rust/crates/lk-agents-voice/src/generation/forward.rs), [playout](../../rust/crates/lk-agents-voice/src/generation/playout.rs), [generation](../../rust/crates/lk-agents-voice/src/generation/mod.rs) |
| SY16 | Medium | Transcript `AttachText`, `DetachText`, `RetireText`, `TextRetired` | Explain logical attachment versus physical retirement. Use ordinary local methods plus meaningful physical requests/receipts; do not add a mirror detach effect automatically. | [Transcript lifecycle](../../rust/crates/lk-agents-voice/src/transcription/lifecycle.rs) |
| SY17 | Medium | `StorageCommit` → `Settle` → `Settled`; `StorageAbort` → `Abort` → `Abandoned` | Document one cross-boundary map, then normalize names only where the operation is identical. Trace commit/fencing/abandon semantics before selecting replacements; retain lease-loss as its own cause. | [Voice effects](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs), [storage port](../../rust/crates/lk-agents-abi/src/storage/voice.rs), [storage engine](../../rust/crates/lk-agents-storage/src/engine.rs) |
| SY18 | Medium | Recording `Started` / `Finished` / `Finish` | Scope segment playback observations versus finalizing the recording file, e.g. `PlaybackStarted`, `PlaybackFinished`, `FinishRecording`. A recording finish must not look like a segment completion. | [Recording commands](../../rust/crates/lk-agents-media/src/recorder/file.rs) |
| SY19 | Low | Avatar runner `Event`, `Effect`, `IoOp` | Scope to the avatar runner while preserving existing request/receipt pairs. Share `Reported { finished, result }` if sufficient; no extra mirror receipt types. | [Avatar runner](../../rust/crates/lk-agents-voice/src/avatar/runner.rs) |
| SY20 | Low | `PoolEvent` variants and wire strings describe individual processes | Consider `ProcessEvent`; keep `initialize` → `Ready` because readiness is the useful observation. No extra `Initialized` stage. | [Process pool](../../rust/crates/lk-agents-worker/src/ipc/pool.rs) |
| SY21 | Medium | Python `agent._enter(ctx)` versus direct `agent.on_exit()` | Document context/decorator compatibility and exit-specific cleanup. Unify private helpers only if that removes code while preserving those contracts; public hook names stay familiar. | [Callbacks](../../livekit-agents-next/livekit/agents/voice/_session_callbacks.py), [Agent](../../livekit-agents-next/livekit/agents/voice/agent.py), [runtime](../../rust/crates/lk-agents-runtime/src/voice/session/mod.rs) |

Additional scheduling/close findings extend the inventory without renumbering prior entries:

| ID | Priority | Current mechanism | Direction | Evidence |
| --- | --- | --- | --- | --- |
| SY39 | Medium | `pause_scheduling` / `resume_scheduling_task` | Match method vocabulary with `resume_scheduling`; document synchronous resume versus awaited drain. Add no unused resume callback. | [Scheduling](../../rust/crates/lk-agents-voice/src/agent_activity/scheduling.rs) |
| SY40 | High | `Paused(Close, Refused)` | Name any retained callback `SchedulingDrainFinished { cause, result }`; preserve settled/refused outcomes and recipient-specific semantics. | [Dispatch](../../rust/crates/lk-agents-voice/src/agent_session/effects.rs) |
| SY41 | High | `CloseProgress` | Prefer coordinator-owned `advance_close` from phase prerequisites; preserve deferred ordering and remove manual advancement obligations. | [Closing](../../rust/crates/lk-agents-voice/src/agent_session/closing.rs) |

## Symmetry to preserve and asymmetry to explain

These entries are explicit retain decisions, not forgotten cleanup. Add the explanation near
the owner when needed; do not add aliases, event pairs, or abstraction layers.

| ID | Existing family | Why it stays | Source |
| --- | --- | --- | --- |
| SY22 | Python `on_enter` / `on_exit`, ABI HookCall pair, common hook response | Existing symmetry already covers both hooks. Exit has extra waiting consumers and toolset-scope cleanup. | [Hooks](../../rust/crates/lk-agents-voice/src/agent_activity/hooks.rs), [runtime](../../rust/crates/lk-agents-runtime/src/voice/session/mod.rs) |
| SY23 | `SpeechCreated` → `SpeechDone`, `RunStarted` → `RunSettled` | Creation occurs before admission/refusal; voice-run settlement waits for participating work. These are different units, not missing mirror events. | [Session events](../../rust/crates/lk-agents-voice/src/events.rs) |
| SY24 | Public `speech_created`, internal speech-done wire, SpeechHandle completion | Public observations and private handle updates have different consumers. Keep handle completion canonical; no new public done event just for symmetry. | [Public events](../../livekit-agents-next/livekit/agents/voice/events.py), [delivery](../../livekit-agents-next/livekit/agents/voice/_session_events.py), [handle](../../livekit-agents-next/livekit/agents/voice/speech_handle.py) |
| SY25 | `VoiceSession.start`, `shutdown`, `aclose` | Start, synchronous shutdown request, and awaited teardown are distinct. Keep Python conventions; no duplicate close alias. | [VoiceSession](../../livekit-agents-next/livekit/agents/voice/agent_session.py) |
| SY26 | Speech `interrupt`, background playback `stop` | Speech interruption is policy-controlled; background stop can fade out. Both already provide `done` / `wait_for_playout`. | [SpeechHandle](../../livekit-agents-next/livekit/agents/voice/speech_handle.py), [background playback](../../livekit-agents-next/livekit/agents/voice/background_audio.py) |
| SY27 | Stream `cancel`, context-manager cleanup, iterator `aclose` | Cancelling the operation, awaiting its cleanup, and closing a consumer cursor differ. No inverse resume after cancellation. | [Run streams](../../livekit-agents-next/livekit/agents/voice/run_stream.py) |
| SY28 | Ports `flush`, `end_input`, `clear_buffer`, `aclose`; `pause` / `resume`, attachment pair | Segment boundary, input half-close, output discard, terminal cleanup, and reversible gates are distinct. Keep the existing familiar pairs. | [Audio ports](../../livekit-agents-next/livekit/agents/voice/io.py), [TTS adapter](../../livekit-agents-next/livekit/agents/tts.py) |
| SY29 | Room connected/disconnected, published/unpublished, subscribed/unsubscribed | Existing SDK-aligned pairs are useful; connected versus active is an extra readiness milestone. No new generic lifecycle wrapper. | [RTC observations](../../rust/crates/lk-agents-abi/src/voice/rtc/events.rs), [room adapter](../../livekit-agents-next/livekit/agents/voice/room_io.py) |
| SY30 | Forwarder `Install` / `Remove` → `EndInput` | Installation associates an already opened stream; removal/replacement ends its input. No fabricated `BeginInput` event. | [Forwarder](../../rust/crates/lk-agents-voice/src/data_plane/mod.rs) |
| SY31 | TextSession invocation started/settled/released/parked | Invocation result, delivery/cleanup release, and durable routing state differ. A transition table is clearer than flattening everything to ended. | [TextSession](../../rust/crates/lk-agents-core/src/text_session/mod.rs) |
| SY32 | A2A `Execute { id, operation }` → `Completed { id, result }` → outer `Finished` | Shared correlated receipt already covers the operation family. No per-operation mirror events needed. | [A2A task](../../rust/crates/lk-agents-core/src/a2a/task/mod.rs) |
| SY33 | LLM request result, `AttemptClosed`, `CleanupIncomplete` | Caller result and physical attempt cleanup differ. LLM cancellation deliberately has no caller terminal result after the consumer gives up. | [LLM lifecycle](../../rust/crates/lk-agents-inference/src/llm/lifecycle.rs) |
| SY34 | Avatar create/created/create-finished/adopted, retiring/terminated | Acquisition, delivery/adoption, and provider-session cleanup differ; abandoned sessions still need termination. | [Avatar provider lifecycle](../../rust/crates/lk-agents-inference/src/avatar/lifecycle.rs) |
| SY35 | Realtime connect/close effects and connected/failed/closed methods | Ordinary methods can supply receipts; no new Event enum required for visual pairing. | [Realtime connection](../../rust/crates/lk-agents-inference/src/realtime/connection.rs) |
| SY36 | Worker connect result and `TransportError::Closed`; drain outcomes and stopped notification | Returns/errors already supply outcomes. Worker draining and process termination are different milestones. | [Transport](../../rust/crates/lk-agents-worker/src/transport/mod.rs), [worker](../../rust/crates/lk-agents-worker/src/worker/mod.rs) |
| SY37 | Lease `stop`, `release`, `release_if_drained`; acquisition elsewhere | Stop ends renewal, not the lease; release is a fenced database operation. The acquired handle does not need a second acquire method. | [Lease](../../rust/crates/lk-agents-storage/src/lease.rs) |
| SY38 | Registration without unregister; console resubscribe after transport EOF | Configuration/process registration is not a reversible live attachment. Transport EOF is not durable task completion; resubscribe is recovery, not undoing cancel. | [Worker](../../livekit-agents-next/livekit/agents/worker.py), [console stream](../../console/src/lib/a2a.ts), [conversation recovery](../../console/src/lib/use-conversation.ts) |

The step-decision audit also has a selected removal: voice consumes shared `AgentRun`
settlement instead of renaming `StepDecision` to `ToolStepDecision`. See the
[shared-run integration decision](rust-transition-inventory.md#shared-agentrun-integration).
Keep necessary voice constraints in the existing coordinated handshake; do not introduce a
second outcome type to achieve naming symmetry.

## Implementation and verification order

1. Write each affected owner's short lifecycle map: operation, identity, effect, returned
   observation/result, terminal owner, and who waits. Include failure/cancel/late cleanup.
2. Correct hook comments and exact semantic mismatches such as cooperative `cancel_all`.
   Apply mechanical effect verb-order renames at producers, consumers, traces, and fakes
   together. Regenerate bindings from declarations where necessary; never edit generated code.
3. Integrate common hook completion and retained continuations with the driver-owned atomic
   transition/checkpoint design. Preserve run-work release and exit fan-out before any run
   completion decision or next scheduling checkpoint.
4. Re-evaluate output/gate/provider wrappers for deletion before renaming their types. Choose
   the conditional result consolidations only after comparing success, error, cancellation,
   cleanup, and caller delivery contracts. Do not introduce a generic lifecycle framework.
5. Add focused contract regressions for behavior-bearing changes, then run the relevant Rust,
   Python binding, and voice/room DST checks. Pure naming changes use existing checks.

Required hook assertions include enter completion participates in an open voice run; activation
resolves on issue; exit completion advances handoff/drain/close once; failures/timeouts continue
the appropriate lifecycle; no-host, late, duplicate, and abandoned replies do not leak work or
re-run continuations; exit toolset cleanup remains owned after hook cancellation. These are
requirements for implementation, not newly executed test results from this audit.

For the inventory as a whole, verify request/receipt correlation, one terminal decision per
operation, idempotent late cleanup, no new public event aliases, and preserved milestone
ordering. A matching pair of names is not a substitute for any of these contracts.
