# Multi-turn AMD on an effects-based reply gate

This is a design proposal for answering-machine detection (AMD) in the voice core.
The behavior it ports is specified by the Python implementation on `main`:
[spec/amd.md](https://github.com/livekit/agents/blob/4de940988/spec/amd.md) (cited
below as "spec §N") and `voice/amd/detector.py`. Nothing here is implemented.
Existing contracts remain in [VoiceSession](../voice-session.md) and
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

The earlier v2.0 AMD was one-shot and ran as an executor beside the session. It held
authorization from attach to verdict and skipped replies while it listened, through
fire-and-forget commands on the session queue: pause or resume authorization, skip
replies, interrupt. It has been removed from `v2.0`, together with the IVR navigator,
the authorization pause, skip-replies, the tap fan-out, the SIP attribute wait and
session-added tool pools.

One hold and one decision per run suited one-shot AMD. Multi-turn AMD must hold again
at every commit and release per turn, and with global commands that races. AMD's
release for turn A and the session's commit of turn B cross in the queue, so A's
release lands after B commits and lets a stale reply play. A `Pause` that AMD sends
when it hears of B arrives a step late, and B's own reply can be authorized in that
gap. The fault is not that AMD reacts to effects. It is that the session acted by
default, the commands named no turn, and the session relied on AMD to tell it that a
turn was stale.

## Decisions

| Decision | Status |
| --- | --- |
| AMD and the session talk only through effects and inputs. The session emits observations and gate requests; AMD answers with decisions and releases keyed by turn. No call crosses the boundary inside a transition. | agreed |
| The session owns the hold. It holds at commit in the same transition and drops any answer that names a turn other than the latest gated one. | agreed |
| The late AMD-final wait (spec §6.2) lives inside AMD, as classifier input. | agreed |
| The parked gate is not surfaced in telemetry for now. | agreed |
| Realtime per-response instructions are passed through unchanged. Whether a provider appends them to or replaces its session instructions is plugin-dependent; the core does not assume either. | agreed |
| The AMD machine runs in its own executor. The protocol does not depend on delivery order, so the DST explores when AMD answers. | proposed |
| Preemptive generation follows Python: it may run while authorization is held, is discarded when AMD publishes a new reply profile, and is adopted only if the gate allows the reply. | proposed |
| A speech held at the head of the queue counts as current for the commit interrupt, preemptive refusal and detach cancellation, as Python's paused current speech does. | proposed |

## The protocol

Three rules make the protocol safe under any delivery order:

1. **Hold by default.** While AMD is attached, the session holds authorization from
   attach, holds again at every commit in the same transition, and parks each gated
   turn until its decision arrives. It never replies on its own.
2. **Key every answer.** Each decision and release names its turn. The session
   applies an answer only to the latest gated turn and drops any other, as
   `on_hook_done` drops a late hook answer.
3. **Supersede locally.** When a newer turn commits, the session denies the parked
   turn and cancels an older turn's held reply itself, without waiting for AMD.

`on_user_turn_completed` already works this way: `issue_turn_hook` emits a host-hook
effect and parks the turn, and `on_hook_done` matches the answer by id. The reply gate
has the same shape, answered by the AMD machine instead of the host.

### Session to AMD

Effects the session emits while AMD is attached. Each carries the time it describes,
so AMD does not depend on when it receives them.

```rust
pub enum AmdObservation {
    /// A turn committed. The session already holds authorization for it.
    TurnCommitted { turn: TurnId, transcript: String, eot_delay: Option<Duration>, at: Ts },
    /// The customer hook returned; the turn waits at the gate.
    GateRequested { turn: TurnId },
    /// The turn left the activity by any path, including local supersession.
    TurnEnded { turn: TurnId, how: TurnEnd },
    /// A reply exists for the turn: created, adopted from a preemptive generation, or
    /// rebuilt after a correction.
    ReplyCreated { turn: TurnId, speech: SpeechId },
    /// Every speech: created, item added (with its words), playout advanced, done.
    Speech { speech: SpeechId, event: SpeechObservation, at: Ts },
    /// Participant speech boundaries and session STT finals.
    UserSpeech(UserSpeechObservation),
    /// Executed tool calls, for DTMF in classifier history (spec §11.1).
    ToolsExecuted { calls: Vec<ToolCallRecord> },
}
```

### AMD to session

Inputs the session accepts from AMD. Each names the turn it is about.

```rust
pub enum AmdAnswer {
    /// Answers `GateRequested`.
    ReplyDecided { turn: TurnId, decision: ReplyDecision },
    /// Ends the silence hold of the turn's allowed reply.
    Released { turn: TurnId },
    /// Tools and instructions for the next reply. A preemptive generation made under an
    /// older profile is discarded.
    ReplyProfile { version: u64, tools: Arc<Toolset> },
    /// The opening (spec §10.6), created with a bypass of the hold.
    OpeningRequested(Opening),
    /// The transcript for a realtime model without user transcription (spec §7.2).
    InsertUserItem { turn: TurnId, text: String },
    /// AMD completed. The session restores its settings and stops gating.
    Detach { cancel: DetachCancel },
}

pub struct ReplyDecision {
    allow: bool,
    /// Stage instructions: a temporary user message for a pipeline reply, per-response
    /// instructions for a realtime reply and each of its tool follow-ups.
    instructions: Option<String>,
    tools: Arc<Toolset>,
    /// The reply is created now, but authorization stays held until `Released`.
    hold: bool,
}
```

Attaching AMD is a session command, as entering any other mode is. It installs the
gate, sets the rules in [Session rules while attached](#session-rules-while-attached)
and starts the hold.

## The session side

### Gate state

```rust
// AgentActivity
gate: Option<Gate>,                    // Some while AMD is attached

struct Gate {
    parked: Option<ParkedTurn>,        // at most one, see below
    hold: Hold,
    profile: (u64, Arc<Toolset>),
}
struct ParkedTurn { info: EndOfTurnInfo, edits: TurnEdits }

enum Hold {
    /// AMD attached; no turn has been allowed yet.
    Entry,
    /// The latest turn has no decision yet.
    Undecided { turn: TurnId },
    /// The latest turn was denied.
    Denied { turn: TurnId },
    /// An allowed reply waits for `Released`.
    Silence { turn: TurnId, reply: SpeechId },
    /// An allowed reply without a hold: authorization is free.
    Open,
}
```

- At commit, in `on_end_of_turn`, the session emits `TurnCommitted`, sets
  `Hold::Undecided` and, if a turn is parked, denies it (rule 3). The `Dropped`
  verdict and the `in_window` hold return earlier and are not gated, as in Python.
- After the customer hook, `finish_turn` emits `GateRequested` and stores the turn in
  `parked` instead of calling `reply_to_turn`. `waiting_turns` is empty whenever a
  turn parks, so at most one turn is parked.
- `ReplyDecided` for the parked turn either calls `reply_to_turn` with the decision's
  instructions and tools, or commits only the user item and discards the preemptive
  generation. `Hold` becomes `Silence`, `Open` or `Denied`.
- `Released` for the latest turn moves `Silence` to `Open`. Any other answer is dropped
  and traced as `GateStep::AnswerDropped`.
- The scheduler authorizes the head speech only under `Hold::Open`, or when the head
  is the opening. A held head stays chosen in a held scheduling step, distinct from
  `Current`.

### Local rules

These need no answer from AMD:

- **Commit interrupt.** For a gated turn without `skip_reply`, `begin_turn` runs
  `interrupt_for_user_turn`, which also interrupts a held head, and the realtime
  interrupt, before the customer hook. A turn later predicted `wait` still cuts off a
  playing voicemail message (spec §10.5). `supersede_unheard_reply` stays after an
  allowed decision.
- **Supersession.** A newer commit denies the parked turn and cancels an older turn's
  held reply. The session emits `TurnEnded` for it, so AMD learns from the session
  instead of telling it.
- **Corrections.** `correct_user_words` also searches `parked`, so a parked turn
  replies to the corrected words. `reply_again_unheard` keeps the turn's
  `ReplyDecision` and applies it to the replacement, with the same hold, and emits
  `ReplyCreated` for it.
- **Turn exits.** `TurnEnded` fires on every exit: reply, `StopResponse`,
  `skip_reply`, `hold_user_turn`, `supersede_turn`, the uninterruptible drop, local
  supersession, and the close and handoff paths that abandon turns without
  `finish_turn`. The close drains `parked` with `Dropped { append_user_message }`
  semantics.
- **Inspection.** The idle check and the re-decide check in `decide_user_turn` also
  inspect `parked`.

### Realtime

AMD refuses a realtime model with server-side turn detection at entry (spec §5.1),
because server-initiated replies never reach the gate. With client-side detection,
both realtime commit paths currently skip it: `on_turn(Commit)` calls
`realtime_user_turn` and returns, and `commit_user_turn` forces `skip_reply` and
returns early when session STT is disabled. With the gate attached and `!skip_reply`:

- the session commits the server audio buffer at commit, before the gate, as Python
  does, so a denied `wait` turn leaves no audio for the next allowed reply
- the turn goes through `on_end_of_turn`, even with no session STT; its classifier
  text then comes from AMD's late final (spec §6.2)
- `reply_to_turn` sends the response request once, after an allowed decision
- the request carries the decision's instructions, and the session tools are updated
  to the decision's tools for the response and restored after it, since
  `SpeechSpec.toolset` alone does not reach the server
- `InsertUserItem` is sent only for a model without user transcription and without
  session STT, so the item is never added twice

### Reply payload

- **Stage instructions.** A pipeline reply gets them as a temporary user message, in a
  new `SpeechSpec` field. They do not use `TurnEdits.instructions` or
  `instructions_override`, which replace the agent's prompt. Setting the field counts
  as an edit, so a preemptive generation made without it is not adopted (spec §10.2).
- **Tools.** The decision's toolset includes `send_dtmf_events` in `machine-ivr`
  (spec §11.1). A preemptive generation uses the current `ReplyProfile`, and is
  adopted only when its profile version matches the decision.

### Session rules while attached

The gate's presence gates these, so DST rules read the same flag (spec §5.3): skip
`min_words` at end of turn, force interruptions and restore them at detach, refuse
`say()` and `generate_reply()` except the opening, cancel handoffs, skip
`on_user_turn_exceeded`, drop deferred tool replies, do not regenerate after a realtime
fallback swap, and discard participant audio while AMD is `Pending`.

## The AMD side

### Layers

| Layer | Contents |
| --- | --- |
| `lk-agents-voice::amd::machine` (pure) | Stage transitions (a port of `_fsm.py`), turn registry, classifier history, the spec §10.1 reply rules, silence hold, voicemail record, counters, deadlines |
| `lk-agents-runtime` AMD executor | Steps the machine on observations, leg results and timers; sends answers to the session |
| Runtime legs | Classifier request, menu extraction, AMD's own STT stream, the wait for the callee's track and SIP answer |

### Machine

```rust
pub struct Amd {
    lifecycle: Lifecycle,                     // Pending | Active | Finished(Outcome)
    stage: Stage,
    likely_human: bool,
    turns: BTreeMap<TurnId, Turn>,
    latest: Option<TurnId>,
    request: Option<Request>,                 // at most one; a stale answer is dropped by id
    late_final: Option<LateFinalWait>,
    history: ClassifierHistory,               // agent replies inserted at their start time
    voicemail: VoicemailRecord,
    uncertain: u8,
    timeouts: u8,
    speech_ended_at: Option<Ts>,
    speaking: bool,
    held_reply: Option<(TurnId, SpeechId)>,
}

enum In {
    Observed(AmdObservation), Classified { req, result }, MenuExtracted { req, .. },
    AmdFinal(..), Callee(Callee), Timeout(AmdTimer), LegsClosed, Close,
}

enum AmdEffect {
    Answer(AmdAnswer), Classify(ClassifyRequest), CancelClassify, ExtractMenu(..),
    SetDeadline(AmdTimer, Ts), ClearDeadline(AmdTimer), Emit(AmdEvent),
}

enum AmdTimer { Inference(ReqId), LateFinal(TurnId), Silence, Idle, Overall, Menu(ReqId) }
```

- `stage::next(stage, category, likely_human)` is a line-for-line port of `_fsm.py`.
- The spec §10.1 table is one `match` that produces `ReplyDecided`, row for row. A
  turn committed while AMD is `Pending` or past its overall timeout gets row 1 at once.
- A stage change publishes a new `ReplyProfile`.
- The silence hold ends on `Timeout(Silence)`, which sends `Released`. New participant
  speech restarts it. A commit observed while a reply is held clears it.
- Speech boundaries use the VAD boundary time carried in the observation, not event
  creation or arrival time (spec §16.2).

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

### Completion

AMD completes inside its own transition, in spec §12.3 order: the lifecycle becomes
`Finished` and the request is cancelled; every parked turn gets its final
`ReplyDecided` (the "AMD completed" row); then AMD sends `Detach`, which tells the
session to cancel pending speeches, including an unplayed opening when the result
forbids a reply, or only the preemptive generation. The session cancels first, then
restores interruption settings, removes the gate and frees authorization. Later
commits are not gated (spec §9.5). `amd_completed` is emitted after the runtime
reports the legs closed (`In::LegsClosed`), so `execute()` returns after cleanup.

## DST

The DST runs the session core and the AMD executor as separate scheduled work, so it
explores every order of observations and answers. The protocol must hold under all of
them.

- **Fakes.** The classifier leg is a fake LLM. The search picks each `record_result`
  category from a callee persona (screener → human → human; voicemail → message →
  menu; hold → `wait`) and injects delays, failures and stale answers. A decision-model
  fake records `choice` calls. AMD's STT is a fake stream with its own lag. The fake
  LLM and the realtime fake record each request's messages, instructions and tools.
- **Scripted AMD.** Before the AMD machine exists, a scripted answerer drives the
  session's gate. It answers late, out of order and for stale turns.
- **Trace.**
  - `GateStep::{Requested, Decided, AnswerDropped, Superseded}` and
    `SpeechStep::AuthorizationHeld(Hold)`, once per change
  - `AmdStep::TurnBound { turn, has_text, late_wait }`
  - `RequestIssued { req, turn }` and `RequestCancelled { req }`
  - `Predicted { turn, req, raw, category, stage, reason, likely_human }`
  - `ReplyBound { turn, speech }`: an adopted preemptive generation has no turn of its
    own
  - `VoicemailPlayed`, `OpeningBypass { speech }`, `Completed`
- **Driver.** The developer model enters AMD, places the call, waits for
  `amd_completed`, and varies the options: stages without instructions, the opening,
  `machine_silence_threshold=0`.
- **Authored stories.** Port the scenarios of `tests/test_amd.py` as authored DST
  stories, and add stories for a correction before and after the reply exists, for
  the classifier result and late final arriving in both orders, and for a release that
  crosses a newer commit. Authored stories are the only check against a rule and an
  implementation that share the same misreading of the spec.

Each spec §14 invariant has one owner:

| # | Invariant | Owner |
| --- | --- | --- |
| 1 | One prediction per turn at most | DST rule on `TurnBound`, `RequestIssued`/`RequestCancelled` and `Predicted` |
| 2 | A stale request never decides | DST rule: `Predicted.req` is the turn's uncancelled request, and only it precedes the turn's `GateStep::Decided` |
| 3 | `wait` never replies | DST rule `reply_follows_gate` |
| 4 | `uncertain` and `wait` keep the stage | DST rule on the `Predicted` sequence |
| 5 | Likely-human mark | DST rule on `raw`, `category` and `likely_human` |
| 6 | One voicemail message at most | DST rule on `VoicemailPlayed`, confirmed at the sink |
| 7 | Silence hold | DST sink rule, with silence computed from the fake VAD boundaries, not from AMD's own anchor |
| 8 | Stage instructions never saved | DST rule: no stage-instruction item in recorded items or storage |
| 9 | DTMF only in `machine-ivr` | DST rule: the tool appears only in requests generated under a `machine-ivr` profile, and runs only for a reply whose decision allowed it |
| 10 | Cleanup on every path | DST rule after `Completed`: no gate, no hold, interruption settings equal their values at attach; host test for listener removal |
| 11 | AMD closes only models it created | Runtime leg unit test |
| 12 | `execute()` returns once | Python host test; DST driver asserts one `Completed` |
| 13 | A customer-handled stage never replies | DST rule `reply_follows_gate` plus the `Completed` reason |
| 14 | Decision model: one `choice`, no menu | Leg unit test; DST decision-model fake sees no menu request |
| 15 | Only the opening plays before a prediction | DST sink rule with `OpeningBypass`, and participant speech before first playout |

Two rules cover the protocol itself:

- `reply_follows_gate`: a denied decision yields no reply for its turn, and an allowed
  one yields at most one. An allowed decision yields none only with a traced refusal,
  such as paused scheduling, the close, or a refused realtime request.
- `gate_answers_are_keyed`: the session applies a decision or release only to the
  latest gated turn; every other answer appears as `AnswerDropped`.

`speech_is_heard` treats a speech as waiting legitimately while the gate holds it, and
under `Hold::Silence` only until AMD's silence deadline. The DST does not measure
classifier accuracy, prompt quality, real STT lag or the SIP answer path. Live
simulations and room tests own those.

## Open questions

- Whether the parked gate becomes visible in telemetry later, and as what span.
- Whether a denied realtime turn should commit or clear the server audio buffer. This
  proposal commits it, as Python does.
- Whether a held head should become `Current` while held, as in Python, instead of
  counting as current in the three places listed in Decisions.

## Implementation order

1. The session gate: protocol types, gate state, local rules, and the DST rules that
   read them, driven by a scripted answerer.
2. The pure AMD machine and its unit tests, ported from `_fsm.py` and the reply table.
3. The AMD executor with the classifier, STT and callee legs.
4. Realtime commit paths, per-response instructions and tools.
5. The Python host API and the authored stories from `tests/test_amd.py`.
