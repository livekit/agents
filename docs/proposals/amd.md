# Multi-turn AMD on a TurnHooks seam

This is a design proposal for answering-machine detection (AMD) in the voice core.
The behavior it ports is specified by the Python implementation on `main`:
[spec/amd.md](https://github.com/livekit/agents/blob/4de940988/spec/amd.md) (cited
below as "spec §N"), `voice/amd/detector.py` and `voice/_turn_hooks.py`. Nothing here
is implemented. Existing contracts remain in [VoiceSession](../voice-session.md) and
[internal testing](../internals/testing.md). After implementation, move accepted
public behavior and internal contracts into their owning guides and retire this
proposal.

## Objective

AMD follows an outbound call across turns: call screeners, voicemail, IVR menus and
hold queues. At each committed turn it classifies the participant, keeps a stage, and
decides whether and how the agent replies. It completes when it finds a person, finds
that the call cannot complete, enters a stage the application handles, or reaches a
limit.

A run is a bounded phase. One `AMD` instance runs once, and only one runs on a
session at a time. The application usually enters it when the call starts, but may
enter a new instance later, for example after a transfer. IVR handling exists only
inside a run, as the `machine-ivr` stage; there is no session-wide IVR navigator.

The port must keep three properties that a test can check at the audio sink:

- a turn's reply is decided by that turn's own prediction, and a superseded turn
  never replies
- a machine-stage reply never plays before the participant has been silent for
  `machine_silence_threshold` (spec §10.3)
- nothing but the opening plays before a prediction allows a reply (spec §10.6)

## Background

The earlier v2.0 AMD was one-shot and ran as an executor beside the session. It read
the session's event tap after the fact and sent fire-and-forget commands back through
the session queue: pause or resume authorization, skip replies, interrupt. It has been
removed from `v2.0`, together with the IVR navigator, the authorization pause,
skip-replies, the tap fan-out, the SIP attribute wait and session-added tool pools.

That shape cannot carry the multi-turn design. Python's AMD decides per turn, adds
instructions and tools to the reply it allows, and watches the reply's speech handle.
Across an executor boundary every one of those messages is its own scheduling step,
so the DST can deliver a newer turn's reply before the pause for it arrives. The
guarantees above hold only if the decisions happen inside the core's own transitions.

## Decisions

| Decision | Status |
| --- | --- |
| The seam is a trait, `TurnHooks`, mirroring Python's protocol. AMD is the only production implementor; a scripted implementor drives the seam in tests. | agreed |
| The late AMD-final wait (spec §6.2) lives inside AMD, as classifier input. | agreed |
| The wait in `awaiting_reply` is not surfaced in telemetry for now. | agreed |
| Realtime per-response instructions are passed through unchanged. Whether a provider appends them to or replaces its session instructions is plugin-dependent; the core does not assume either. | agreed |
| Preemptive generation follows Python: it may run while authorization is held, is cancelled on any stage change, and is adopted only if the gate allows the reply. | proposed |
| The scheduler asks AMD `may_authorize(speech)` instead of keeping its own copy of the hold. The answer changes only when AMD steps. | proposed |
| A speech held at the head of the queue counts as current for the commit interrupt, preemptive refusal and completion cancellation, as Python's paused current speech does. | proposed |
| A promoted-interim correction re-applies the turn's reply decision to the replacement reply. | proposed |

## Layers

| Layer | Contents |
| --- | --- |
| `lk-agents-voice::amd::machine` (pure) | Stage transitions (a port of `_fsm.py`), turn registry, classifier history, the spec §10.1 reply rules, silence hold, voicemail record, counters, deadlines |
| Seam in `agent_session::machine::activity` | Hook calls at commit, generation, the reply gate, reply creation and speech events; the authorization query; the parked reply |
| Runtime legs | Classifier request, menu extraction, AMD's own STT stream, the wait for the callee's track and SIP answer |

Leg results enter the session as inputs and are stepped inside the session core's
transition. Nothing outside the core changes a reply decision.

## The seam

```rust
/// Python `_turn_hooks.TurnHooks`, stepped inside the activity's transition.
pub(crate) trait TurnHooks {
    /// At commit, before the commit interrupt, the customer hook and `waiting_turns`.
    fn on_user_turn_committed(&mut self, now: Ts, info: &EndOfTurnInfo) -> (TurnBinding, Vec<HookEffect>);
    /// The turn left the activity by any path.
    fn on_user_turn_completed(&mut self, now: Ts, b: TurnBinding) -> Vec<HookEffect>;
    /// Every generation: turn reply, preemptive, manual. Adds `send_dtmf_events` in `machine-ivr`.
    fn on_reply_generation(&self, tools: Arc<Toolset>) -> Arc<Toolset>;
    /// After the customer hook. `Pending` parks the turn until a `GateReady` effect.
    fn should_reply(&mut self, now: Ts, b: TurnBinding, agent: &str) -> (Gate, Vec<HookEffect>);
    /// The reply exists: created, adopted from a preemptive generation, or rebuilt after a
    /// correction.
    fn on_agent_turn_committed(&mut self, b: TurnBinding, speech: SpeechId) -> Vec<HookEffect>;
    /// Every speech: created, item added, playout advanced, done.
    fn on_speech(&mut self, now: Ts, ev: SpeechObservation) -> Vec<HookEffect>;
    /// Asked by the scheduler for the head speech, after the consecutive-speech gap. Reads
    /// AMD's state only, never the clock.
    fn may_authorize(&self, speech: SpeechId) -> Authorization;
}

pub(crate) enum TurnBinding {
    /// AMD was active at commit and bound the turn.
    Bound { run: AmdRunId, turn: AmdTurnId },
    /// AMD was installed but not active, or the overall timeout passed at commit
    /// (spec §7.1). The gate uses spec §10.1 row 1.
    Unbound { run: AmdRunId },
}

pub(crate) enum Gate { Pending, Decided(ReplyDecision) }

pub(crate) struct ReplyDecision {
    allow: bool,
    /// Stage instructions. A temporary user message for a pipeline reply; per-response
    /// instructions for a realtime reply and each of its tool follow-ups.
    instructions: Option<String>,
    track_voicemail: bool,
    /// Classifier history keeps the reply's words, not "(the agent spoke)" (spec §8.2).
    history_words: bool,
}

pub(crate) enum Authorization {
    Allow,
    Hold(HoldReason),
}

pub(crate) enum HoldReason {
    /// AMD entered; no turn has been allowed yet.
    Entry,
    /// The latest turn has no allowed gate yet: its prediction is pending, or it left
    /// without reaching the gate (`skip_reply`, the uninterruptible drop).
    Undecided { turn: AmdTurnId },
    /// The latest turn was denied (`wait`, voicemail already sent, superseded).
    Denied { turn: AmdTurnId },
    /// An allowed machine-stage reply waits for silence. `until` is `None` while the
    /// participant speaks. Only `Timeout(Silence)` ends it.
    Silence { turn: AmdTurnId, until: Option<Ts> },
}
```

A commit when no run is installed gets no binding and no gate, as after completion
(spec §9.5). `should_reply` returns effects because several spec §10.1 rows act: they
complete AMD (`agent_changed`, `timeout`), start the silence hold, or mark the
voicemail turn. It takes the driving agent's id for the `agent_changed` row.

## Call sites

Function names refer to `agent_session/machine/activity/` on `v2.0`.

| Hook | Call site |
| --- | --- |
| `on_user_turn_committed` | `on_end_of_turn`, once the verdict commits a turn and before `waiting_turns.push_back`. The binding is stored on `DecidedTurn`. Classification starts at commit, so an older turn parked on its customer hook does not delay a newer turn's prediction. The `Dropped` verdict and the `in_window` hold return earlier and carry no binding, as in Python. |
| Commit interrupt | For a bound turn without `skip_reply`, in `begin_turn` before the customer hook, as Python does: `interrupt_for_user_turn`, which also interrupts a held head, and the realtime interrupt. A turn later predicted `wait` still cuts off a playing voicemail message (spec §10.5). `supersede_unheard_reply` stays after an allowed gate. |
| `on_user_turn_completed` | Every exit after the bind: `finish_turn` (reply, `StopResponse`, `skip_reply`, `hold_user_turn`), `supersede_turn`, the uninterruptible drop in `begin_turn`, `user_turn_dropped`, and the paths that abandon parked turns without `finish_turn`: the close and the handoff swap. |
| `should_reply` | `finish_turn`, after the customer hook's edits are applied and before `reply_to_turn`. |
| `on_reply_generation` | `start_speech`, on `SpeechSpec.toolset`. The toolset is fixed per speech, so a stage change cancels the preemptive generation (`HookEffect::CancelPreemptive`). |
| `on_agent_turn_committed` | The speech `reply_to_turn` creates, the preemptive generation it adopts, and the replacement `reply_again_unheard` builds. |
| `on_speech` | Speech creation, item added, playout progress and done, in the scheduler and output paths. |
| `may_authorize` | `scheduling_pass`, for the head candidate. A held candidate stays chosen in a held scheduling step, distinct from `Current`, until AMD emits `HookEffect::Reschedule`. |
| Correction | `correct_user_words` also searches `awaiting_reply`, so a parked turn replies to the corrected words. While a run is active, `reply_again_unheard` keeps the turn's `ReplyDecision` and applies it to the replacement: instructions, toolset, `on_agent_turn_committed`, silence hold and voicemail tracking. |

### Realtime

AMD refuses a realtime model with server-side turn detection at entry (spec §5.1),
because server-initiated replies never reach `should_reply`. With client-side
detection, both realtime commit paths currently skip the gate: `on_turn(Commit)` calls
`realtime_user_turn` and returns, and `commit_user_turn` forces `skip_reply` and
returns early when session STT is disabled. With a run installed and `!skip_reply`:

- the core commits the server audio buffer at commit, before the gate, as Python does,
  so a denied `wait` turn leaves no audio for the next allowed reply
- the turn goes through `on_end_of_turn`, even with no session STT; its classifier
  text then comes from AMD's late final (spec §6.2)
- `reply_to_turn` sends the response request once, after an allowed gate
- the request carries the stage instructions, and the session tools are updated to
  include `send_dtmf_events` for the response and restored after it, since
  `SpeechSpec.toolset` alone does not reach the server
- AMD inserts the user item (spec §7.2) only for a model without user transcription
  and without session STT, so the item is never added twice

## The reply gate

```rust
// AgentActivity
awaiting_reply: Option<AwaitingReply>,
struct AwaitingReply { info: EndOfTurnInfo, edits: TurnEdits, binding: TurnBinding }
```

- `should_reply` → `Pending`: `finish_turn` stores the turn in `awaiting_reply`, and
  `next_turn` continues.
- A classifier result arrives as a session input. AMD steps and emits
  `GateReady(turn)`. The activity asks `should_reply` again in the same transition. An
  allowed turn goes to `reply_to_turn`. A denied turn commits only its user item and
  cancels the preemptive generation.
- `Finish` emits `GateReady` for a parked turn, so it takes the "AMD completed" row.
- A newer commit decides the parked turn (deny) in the same transition. `waiting_turns`
  is empty whenever a turn parks, so at most one turn is parked.
- Code that inspects parked turns also inspects `awaiting_reply`: the idle check, the
  re-decide check in `decide_user_turn`, `correct_user_words`, and the close, which
  drains it with `Dropped { append_user_message }` semantics.

## Authorization

The scheduler keeps no AMD state. AMD answers `may_authorize` from its own state, and
that state changes only when AMD steps, so the two cannot disagree. The answer follows
Python's pause and resume points: held from entry, released only by an allowed reply
that is not held, by `Timeout(Silence)`, or by completion. A denied turn keeps it held.

Cancellation runs before any release that could let a stale speech play:

- At a commit, with or without text, AMD cancels the held reply and clears the silence
  deadline, then emits `Reschedule`.
- At completion, AMD cancels speeches first (spec §12.3), including an unplayed
  opening when the result forbids a reply, then releases the hold.
- The overall timeout during a silence hold completes AMD and releases the allowed
  reply at once (spec §10.3).
- Participant speech before the opening's first playout cancels the opening, even
  when it produces no commit (spec §10.6).

Other rules:

- The opening is marked at creation and is the only speech that `may_authorize`
  allows while held. It is created through an internal origin, since `say()` and
  `generate_reply()` are refused while a run is active.
- Pipeline generation already runs before authorization, which gates playout. A
  realtime request is sent at authorization, so a realtime reply starts generating
  after the silence, as spec §10.3 requires.
- A change of hold reason is traced as `SpeechStep::AuthorizationHeld(HoldReason)`,
  once per change, not per scheduling pass.

## Reply payload

- **Stage instructions.** A pipeline reply gets them as a temporary user message, in a
  new `SpeechSpec` field. They do not use `TurnEdits.instructions` or
  `instructions_override`, which replace the agent's prompt. Setting the field counts
  as an edit, so a preemptive generation made without it is not adopted (spec §10.2).
  A realtime reply carries them on the `UserTurn` request and on each `ToolFollowUp`.
- **Tools.** `on_reply_generation` adds `send_dtmf_events` in `machine-ivr`
  (spec §11.1). Successful DTMF calls enter classifier history from the tool results.

## AMD machine

```rust
pub struct Amd {
    run: AmdRunId,
    lifecycle: Lifecycle,                     // Pending | Active | Finished(Outcome)
    stage: Stage,
    likely_human: bool,
    turns: BTreeMap<AmdTurnId, Turn>,
    latest: AmdTurnId,
    request: Option<Request>,                 // at most one; a stale answer is dropped by id
    late_final: Option<LateFinalWait>,
    history: ClassifierHistory,               // agent replies inserted at their start time
    voicemail: VoicemailRecord,
    uncertain: u8,
    timeouts: u8,
    speech_ended_at: Option<Ts>,
    speaking: bool,
    held_reply: Option<(AmdTurnId, Option<SpeechId>)>,
}

enum In {
    Commit(..), Classified { req, result }, MenuExtracted { req, .. },
    SpeechStarted { at }, SpeechEnded { at }, SessionFinal(..), AmdFinal(..),
    Speech(SpeechObservation), Callee(Callee), Timeout(AmdTimer), LegsClosed, Close,
}

enum HookEffect {
    Classify(ClassifyRequest), CancelClassify, ExtractMenu(..),
    SetDeadline(AmdTimer, Ts), ClearDeadline(AmdTimer),
    Reschedule, CancelSpeech(SpeechId), CancelPreemptive, GateReady(AmdTurnId),
    InsertUserItem(..), Emit(AmdEvent), Finish { cancel_speeches: bool },
}

enum AmdTimer { Inference(ReqId), LateFinal(AmdTurnId), Silence, Idle, Overall, Menu(ReqId) }
```

- `stage::next(stage, category, likely_human)` is a line-for-line port of `_fsm.py`.
- The spec §10.1 table is one `match` in `should_reply`, row for row.
- `AmdTimer` joins the session timers as `Timer::Amd(_)`.
- Speech boundaries use the VAD boundary time, not event creation time (spec §16.2).

### Turn ownership

Three things can belong to different turns, and the machine tracks each separately
(spec §6.2, §7.3, §7.4):

- **The request** belongs to the newest turn with text. A newer turn with text cancels
  it. An empty turn does not; it waits for the result and gets a `reused` prediction.
- **The late-final wait** belongs to the newest committed turn. A newer commit takes it
  over, and an older turn's AMD final then goes to classifier history only.
- **The right to reply** belongs to the latest turn. An empty turn can take it from an
  older turn whose request is still pending; that request's result still gives the
  empty turn its `reused` prediction, and a later AMD final does not revise it.

The `inference_timeout` deadline starts at commit for a turn with text, including the
late-final wait, and when the AMD final arrives for a turn committed without text.

## Completion

`Finish` runs inside the transition, in spec §12.3 order: the lifecycle becomes
`Finished` and the request is cancelled; the run's hooks are removed from the session
slot, so later commits get no binding; interruption settings are restored; pending
speeches or only the preemptive generation are cancelled; then the hold is released.
The run's `AmdOutcome` stays reachable from the bindings it issued and answers only
those. `amd_completed` is emitted after the runtime reports the legs closed
(`In::LegsClosed`), so `execute()` returns after cleanup.

## Session rules while a run is active

One predicate, `amd_active()`, gates these, so DST rules read the same flag
(spec §5.3): skip `min_words` at end of turn, force interruptions and restore them at
completion, refuse `say()` and `generate_reply()` except the opening's internal origin,
cancel handoffs, skip `on_user_turn_exceeded`, drop deferred tool replies, do not
regenerate after a realtime fallback swap, and discard participant audio while
`Pending`.

## DST

The DST runs the real core with AMD inside it, so every ordering it explores already
includes AMD's decisions.

- **Fakes.** The classifier leg is a fake LLM. The search picks each `record_result`
  category from a callee persona (screener → human → human; voicemail → message →
  menu; hold → `wait`) and injects delays, failures and stale answers. A decision-model
  fake records `choice` calls. AMD's STT is a fake stream with its own lag. The fake
  LLM and the realtime fake record each request's messages, instructions and tools.
- **Trace.**
  - `AmdStep::TurnBound { turn, has_text, late_wait }`
  - `RequestIssued { req, turn }` and `RequestCancelled { req }`
  - `Predicted { turn, req, raw, category, stage, reason, likely_human }`
  - `GateDecided { turn, allow }`
  - `ReplyBound { turn, speech }`: an adopted preemptive generation has no turn of its
    own
  - `VoicemailPlayed`, `OpeningBypass { speech }`, `Completed`
  - `SpeechStep::AuthorizationHeld(HoldReason)`
- **Driver.** The developer model enters AMD, places the call, waits for
  `amd_completed`, and varies the options: stages without instructions, the opening,
  `machine_silence_threshold=0`.
- **Authored stories.** Port the scenarios of `tests/test_amd.py` as authored DST
  stories, and add stories for a correction before and after the reply exists, and
  for the classifier result and late final arriving in both orders. Authored stories
  are the only check against a rule and an implementation that share the same
  misreading of the spec.

Each spec §14 invariant has one owner:

| # | Invariant | Owner |
| --- | --- | --- |
| 1 | One prediction per turn at most | DST rule on `TurnBound`, `RequestIssued`/`RequestCancelled` and `Predicted` |
| 2 | A stale request never decides | DST rule: `Predicted.req` is the turn's uncancelled request, and only it precedes the turn's `GateDecided` |
| 3 | `wait` never replies | DST rule `reply_follows_gate` |
| 4 | `uncertain` and `wait` keep the stage | DST rule on the `Predicted` sequence |
| 5 | Likely-human mark | DST rule on `raw`, `category` and `likely_human` |
| 6 | One voicemail message at most | DST rule on `VoicemailPlayed`, confirmed at the sink |
| 7 | Silence hold | DST sink rule, with silence computed from the fake VAD boundaries, not from AMD's own anchor |
| 8 | Stage instructions never saved | DST rule: no stage-instruction item in recorded items or storage |
| 9 | DTMF only in `machine-ivr` | DST rule: the tool appears only in requests generated in `machine-ivr`, and runs only for a reply whose gate allowed it |
| 10 | Cleanup on every path | DST rule after `Completed`: no hold, no further `AmdStep`, interruption settings equal their values at entry; host test for listener removal |
| 11 | AMD closes only models it created | Runtime leg unit test |
| 12 | `execute()` returns once | Python host test; DST driver asserts one `Completed` |
| 13 | A customer-handled stage never replies | DST rule `reply_follows_gate` plus the `Completed` reason |
| 14 | Decision model: one `choice`, no menu | Leg unit test; DST decision-model fake sees no menu request |
| 15 | Only the opening plays before a prediction | DST sink rule with `OpeningBypass`, and participant speech before first playout |

`reply_follows_gate`: a denied gate yields no reply for its turn, and an allowed gate
yields at most one. An allowed gate yields none only with a traced refusal, such as
paused scheduling, the close, or a refused realtime request. `speech_is_heard` treats a
speech as waiting legitimately while a run is active and no allowed gate has released
it, and under `Silence` only until `until`.

The DST does not measure classifier accuracy, prompt quality, real STT lag or the SIP
answer path. Live simulations and room tests own those.

## Open questions

- Whether `awaiting_reply` becomes visible in telemetry later, and as what span.
- Whether a denied realtime turn should commit or clear the server audio buffer. This
  proposal commits it, as Python does.
- Whether a held head should become `Current` while held, as in Python, instead of
  counting as current in the three places listed in Decisions.

## Implementation order

1. The pure AMD machine and its unit tests, ported from `_fsm.py` and the reply table.
2. The seam with a scripted `TurnHooks` implementor, and the DST rules that read it.
3. AMD behind the seam, with the classifier, STT and callee legs.
4. Realtime commit paths, per-response instructions and tools.
5. The Python host API and the authored stories from `tests/test_amd.py`.
