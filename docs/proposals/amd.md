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
default, the commands named no turn, and the session relied on AMD for facts it
already had: which turn is latest, and whether the participant is speaking.

## Decisions

| Decision | Status |
| --- | --- |
| AMD and the session talk only through effects and inputs. The session emits observations; AMD answers with decisions keyed by run and turn. No call crosses the boundary inside a transition. | agreed |
| The session owns the hold. It holds at commit in the same transition and drops any answer that names a turn other than the latest committed one. | agreed |
| The late AMD-final wait (spec §6.2) lives inside AMD, as classifier input. | agreed |
| The pending reply decision is not surfaced in telemetry for now. | agreed |
| Realtime per-response instructions are passed through unchanged. Whether a provider appends them to or replaces its session instructions is plugin-dependent; the core does not assume either. | agreed |
| The session times the silence hold from its own speech boundaries. AMD decides only which replies need it and for how long. | proposed |
| AMD's final message settles every turn committed under the gate, including one still in its customer hook. The session acknowledges it before `execute()` returns. | proposed |
| The AMD machine runs in its own executor. The protocol depends only on per-direction FIFO delivery, so the DST explores when AMD answers. | proposed |
| Preemptive generation may run while authorization is held. An allowed reply adopts it when the reply's request is unchanged: no stage instructions, the same DTMF tool state, and the session's usual adoption check. A stage change or completion does not cancel it by itself, unlike Python (spec §3, §12.3), so a direct human answer keeps its head start. | proposed |
| `amd_prediction` and `amd_completed` are informational. Applications do not change the agent's instructions or tools while AMD is attached; the proposal documents this and does not enforce it. | proposed |
| A speech held at the head of the queue counts as current for the commit interrupt, preemptive refusal and detach cancellation, as Python's paused current speech does. | proposed |

## The protocol

Each direction is FIFO: the session applies AMD's answers in the order AMD sent them,
and AMD observes the session's effects in the order they happened. The scheduler may
interleave the two directions in any way. Four rules make the protocol safe under
every such interleaving:

1. **Hold by default.** While AMD is attached, the session holds authorization from
   attach, holds again at every commit in the same transition, and never replies to
   a gated turn without its decision.
2. **Key every answer.** Each answer names its run, and each decision names its turn.
   The session applies a decision only to the latest committed turn and drops any
   other, as `on_hook_done` drops a late hook answer.
3. **Supersede locally.** When a newer turn commits, the session denies the older turn
   and cancels its held reply itself, without waiting for AMD.
4. **Keep live facts local.** What can change between AMD's decision and the session
   applying it stays in the session: whether the participant is speaking, how long
   they have been silent, and whether the opening is still eligible. AMD's answers
   carry policy, not those facts.

`on_user_turn_completed` already works like rules 1 and 2: `issue_turn_hook` emits a
host-hook effect and parks the turn, and `on_hook_done` matches the answer by id. The
reply decision has the same shape, answered by the AMD machine instead of the host.

### Session to AMD

Effects the session emits while AMD is attached. Each carries the time it describes,
so AMD does not depend on when it receives them.

```rust
pub enum AmdObservation {
    /// A turn committed. The session already holds authorization for it.
    TurnCommitted {
        turn: TurnId,
        transcript: String,
        eot_delay: Option<Duration>,
        skip_reply: bool,
        /// The participant was still speaking at commit (spec §7.1 speech duration).
        speaking: bool,
        agent: AgentId,
        at: Ts,
    },
    /// The customer hook returned and the turn waits for its decision. AMD needs it
    /// only for idle tracking; decisions do not wait for it.
    TurnParked { turn: TurnId },
    /// The turn left the activity by any path, including local supersession.
    TurnEnded { turn: TurnId, how: TurnEnd },
    /// A reply exists for the turn: created, adopted from a preemptive generation, or
    /// rebuilt after a correction.
    ReplyCreated { turn: TurnId, speech: SpeechId },
    /// Every speech: created (with `add_to_chat_ctx`), item added (with its words and
    /// whether it was interrupted), playout advanced, done (interrupted, failed). At
    /// attach the session also reports each speech already queued or playing.
    Speech { speech: SpeechId, event: SpeechObservation, at: Ts },
    /// Whether the agent is busy for the idle timer: a paused speech, pending
    /// playback, thinking, a pending false interruption, or a running user-turn task
    /// (Python's `_is_agent_busy`).
    AgentBusy { busy: bool, at: Ts },
    /// Participant speech boundaries and session STT finals.
    UserSpeech(UserSpeechObservation),
    /// Executed tool calls, for DTMF in classifier history (spec §11.1).
    ToolsExecuted { calls: Vec<ToolCallRecord> },
    /// The session applied `Detach`: no turn committed under the gate is left
    /// undecided.
    Detached { run: AmdRunId },
}
```

### AMD to session

Inputs the session accepts from AMD.

```rust
pub enum AmdAnswer {
    /// Listening started (spec §5.2): the session stops discarding participant audio.
    /// The opening, if any, comes with it, so its eligibility is checked at the same
    /// point.
    Activated { run: AmdRunId, opening: Option<Opening> },
    /// The decision for a committed turn. It may arrive before or after the customer
    /// hook returns.
    ReplyDecided { run: AmdRunId, turn: TurnId, decision: ReplyDecision },
    /// Whether generations started from now on, preemptive ones included, get
    /// `send_dtmf_events`. Sent on entering and leaving `machine-ivr` (spec §11.1).
    Dtmf { run: AmdRunId, enabled: bool },
    /// The transcript for a realtime model without user transcription (spec §7.2),
    /// also for an older turn's late final and a turn whose prediction was reused.
    InsertUserItem { run: AmdRunId, turn: TurnId, text: String },
    /// AMD completed. `pending` settles every turn committed under the gate that has no
    /// decision: the "AMD completed" row of spec §10.1.
    Detach { run: AmdRunId, pending: ReplyDecision, cancel: DetachCancel },
}

pub struct ReplyDecision {
    allow: bool,
    /// Stage instructions for this reply and its tool follow-ups.
    instructions: Option<String>,
    /// Whether the reply's tools include `send_dtmf_events`. FIFO delivery puts the
    /// matching `Dtmf` before this decision.
    dtmf: bool,
    /// Silence the participant must keep before the reply plays (spec §10.3). The
    /// session times it.
    silence: Option<Duration>,
}
```

Attaching AMD is a session command, as entering any other mode is. It installs the
gate for the run, sets the rules in
[Session rules while attached](#session-rules-while-attached) and starts the hold.

## The session side

### Gate state

```rust
// AgentActivity
gate: Option<Gate>,

struct Gate {
    run: AmdRunId,
    active: bool,                      // set by `Activated`
    /// The latest committed turn and its decision, once known. A decision can arrive
    /// while the turn is still in its customer hook; `finish_turn` consumes it.
    latest: Option<(TurnId, Option<ReplyDecision>)>,
    parked: Option<ParkedTurn>,        // at most one
    hold: Hold,
    dtmf: bool,                        // the last `Dtmf`
    /// Participant speech since activation, for the opening.
    spoke_since_active: bool,
    /// Set by `Detach`: the gate stays only until the turns committed under it exit.
    detached: Option<ReplyDecision>,
}
struct ParkedTurn { info: EndOfTurnInfo, edits: TurnEdits }

enum Hold {
    /// AMD attached; no turn has been allowed yet.
    Entry,
    /// The latest turn has no decision yet.
    Undecided { turn: TurnId },
    /// The latest turn was denied, or ended without a reply (`skip_reply`,
    /// `StopResponse`). It stays held until the next commit, as in Python.
    Denied { turn: TurnId },
    /// An allowed reply waits until the participant has been silent for `needed`,
    /// measured from the session's latest speech boundary.
    Silence { turn: TurnId, needed: Duration },
    /// Authorization is free.
    Open,
}
```

### Transitions

- **Commit** (`on_end_of_turn`): emit `TurnCommitted`, set `latest` to the new turn
  with no decision and `Hold::Undecided`. If a turn is parked or holds a reply, deny
  it and cancel the reply (rule 3). The `Dropped` verdict and the `in_window` hold
  return earlier and are not gated, as in Python.
- **Decision** (`ReplyDecided`): if it names `latest`, store it; otherwise drop it and
  trace `GateStep::AnswerDropped`. If the turn is parked, apply it now.
- **Customer hook returns** (`finish_turn`): emit `TurnParked`. If `latest` has a
  decision, apply it; otherwise park the turn. `waiting_turns` is empty whenever a
  turn parks, so at most one turn is parked.
- **Apply a decision.** A denied turn commits only its user item, discards the
  preemptive generation and sets `Hold::Denied`. An allowed turn calls `reply_to_turn`
  with the decision's instructions and DTMF state, and sets `Hold::Silence` if the
  decision asks for silence, else `Hold::Open`.
- **Silence.** Under `Hold::Silence`, a `Timer::AmdSilence` is armed for the latest
  speech boundary plus `needed`. If there was no boundary since the commit, the
  boundary is the speech end estimated from `eot_delay`, else the commit, as in
  Python. Start of speech disarms it; end of speech re-arms it. When it fires while
  the participant is silent, the hold becomes `Open`.
- **Turn exits without a decision applied** (`skip_reply`, `StopResponse`, the
  uninterruptible drop): emit `TurnEnded`, set `Hold::Denied`.
- **Authorization.** The scheduler authorizes the head speech only under `Hold::Open`,
  or when the head is the opening. A held head stays chosen in a held scheduling step,
  distinct from `Current`.

### Activation and the opening

`Activated` ends the `Pending` audio discard. If it carries an opening, the session
creates it with the hold bypass only if the participant is not speaking and has not
spoken since activation started; otherwise it drops it (spec §10.6). Start of speech
before the opening's first playout cancels it. All of these are local facts, so a
late `Activated` cannot let a stale opening play. The opening is the only speech that
bypasses the hold, and it is created through an internal origin, since `say()` and
`generate_reply()` are refused while the gate is attached.

### Detach

`Detach` is applied in this order (spec §12.3):

1. When `cancel` forbids replies, cancel queued speeches, an unplayed opening and the
   preemptive generation, but keep a speech that is playing or paused, as
   `_cancel_pending_speeches` does: a voicemail message keeps playing when the box
   hangs up. Otherwise keep the preemptive generation; adoption decides whether it
   still matches (spec §12.3 cancels it).
2. Restore interruption settings, set `dtmf` to false, and clear the rules of
   [Session rules while attached](#session-rules-while-attached).
3. Settle turns: `pending` replaces any decision the session has not applied yet,
   stored for a turn in its customer hook or for a parked turn, because Python
   decides when the reply is about to be made and a finished run answers with the
   "AMD completed" row. A parked turn has it applied now; a turn still in its
   customer hook keeps it for `finish_turn`. `pending` never asks for silence or DTMF.
4. Free authorization. A reply already held for silence plays at once, for every
   completion reason (spec §10.3); only a `cancel` that forbids replies removes it.
5. Once no turn committed under the gate remains, remove the gate and emit
   `Detached`. Later commits are not gated (spec §9.5).

### Local rules

These need no answer from AMD:

- **Commit interrupt.** For a gated turn without `skip_reply`, `on_end_of_turn` runs
  `interrupt_for_user_turn`, which also interrupts a held head, and the realtime
  interrupt, next to the hold. It does not wait for an older turn's customer hook. A
  turn later predicted `wait` still cuts off a playing voicemail message (spec §10.5).
  `supersede_unheard_reply` stays after an allowed decision.
- **Corrections.** `correct_user_words` also searches `parked`, so a parked turn
  replies to the corrected words. While the reply's turn is `latest` and its hold is
  `Silence` or `Open`, `reply_again_unheard` applies the same decision to the
  replacement and emits `ReplyCreated` for it. `reply_again_unheard` currently picks
  the newest reply regardless of turn, so it gains that guard.
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

- the session commits the server audio buffer at commit, before the decision, as
  Python does, so a denied `wait` turn leaves no audio for the next allowed reply
- the turn goes through `on_end_of_turn`, even with no session STT; its classifier
  text then comes from AMD's late final (spec §6.2)
- `reply_to_turn` sends the response request once, after an allowed decision. The
  request is sent at authorization, so a reply held for silence starts generating
  after it, as spec §10.3 requires
- the request carries the decision's instructions, and the session tools are updated
  to the decision's tools for the response and restored after it, since
  `SpeechSpec.toolset` alone does not reach the server
- `InsertUserItem` is sent only for a model without user transcription and without
  session STT, so the item is never added twice

### Reply payload

- **Stage instructions.** `SpeechSpec.turn_instructions` already carries per-reply
  instructions that never enter saved history. Stage instructions use it, with two
  changes. It renders as a system message, and Python's AMD adds a user message; the
  role is an open question below. It is cleared for tool follow-ups, and spec §10.2
  keeps stage instructions for them, so a gated reply keeps it. It is not
  `TurnEdits.instructions` or `instructions_override`, which replace the agent's
  prompt.
- **Tools.** The only tool AMD adds is `send_dtmf_events`, in `machine-ivr` (spec
  §11.1). This replaces Python's `on_reply_generation` hook, which the session called
  on every generation. A generation gets the tool when the gate's last `Dtmf` enabled
  it; an allowed reply gets it when `decision.dtmf` is set.
- **Preemptive generation.** An allowed reply adopts the preemptive generation when
  the decision has no instructions and the session's usual check passes: same
  transcript, chat context, tools and tool choice. A DTMF change fails the tools
  comparison, so no version is needed. Stage instructions change the request, so a
  reply that carries them generates again (spec §10.2). On a direct human answer the
  request is the one a new generation would send, so the head start is kept. Python
  cancels the preemptive generation on every stage change and at completion; this
  port relies on the request comparison instead.

### Session rules while attached

The gate's presence gates these, so DST rules read the same flag (spec §5.3): skip
`min_words` at end of turn, force interruptions and restore them at detach, refuse
`say()` and `generate_reply()` except the opening, cancel handoffs, skip
`on_user_turn_exceeded`, drop deferred tool replies, do not regenerate after a realtime
fallback swap, and discard participant audio until `Activated`.

## The AMD side

### Layers

| Layer | Contents |
| --- | --- |
| `lk-agents-voice::amd::machine` (pure) | Stage transitions (a port of `_fsm.py`), turn registry, classifier history, the spec §10.1 reply rules, voicemail record, counters, deadlines |
| `lk-agents-runtime` AMD executor | Steps the machine on observations, leg results and timers; sends answers to the session |
| Runtime legs | Classifier request, menu extraction, AMD's own STT stream, the wait for the callee's track and SIP answer |

AMD's STT hears the participant's real frames. The session substitutes silence for
its own STT and realtime input during AEC warmup and uninterruptible speech, but AMD,
VAD and the interruption detector keep the real frame (`agent_activity.push_audio`).
The removed tap fan-out provided this, so the port needs one participant-audio feed
for the AMD leg, taken before the discard; its mechanism is an open question. If the
AMD stream fails or will not open, the leg reports `SttFailed`: AMD closes it, stops
late-final waits, and uses session transcripts from then on (spec §6.3). A blank
final does not consume a turn's late-final wait.

### Machine

```rust
pub struct Amd {
    run: AmdRunId,
    lifecycle: Lifecycle,                     // Pending | Active | Finished(Outcome)
    stage: Stage,
    previous_stage: Option<Stage>,            // for the completed category (spec §12.2)
    category: Category,                       // latest accepted category; `wait` denies replies
    last_prediction: Option<Prediction>,      // not changed by reused predictions
    likely_human: bool,
    had_machine_stage: bool,                  // set when a machine stage is accepted, never cleared
    turns: BTreeMap<TurnId, Turn>,
    latest: Option<TurnId>,
    request: Option<Request>,                 // at most one; a stale answer is dropped by id
    late_final: Option<LateFinalWait>,
    history: ClassifierHistory,               // agent replies inserted at their start time
    voicemail: VoicemailRecord,
    uncertain: u8,
    timeouts: u8,
    speech: SpeechClock,                      // speaking, started_at, duration, since_commit
    agent_busy: bool,
}

enum In {
    Observed(AmdObservation), Classified { req, result }, MenuExtracted { req, .. },
    AmdFinal(..), SttFailed, LegFailed, Callee(Callee), Timeout(AmdTimer), LegsClosed, Close,
}

enum AmdEffect {
    Answer(AmdAnswer), Classify(ClassifyRequest), CancelClassify, ExtractMenu(..),
    SetDeadline(AmdTimer, Ts), ClearDeadline(AmdTimer), Emit(AmdEvent),
}

enum AmdTimer { Inference(ReqId), LateFinal(TurnId), Idle, Overall, Menu(ReqId) }
```

- `stage::next` is a line-for-line port of `_fsm.py`. The rules around it live in
  `detector.py` and are ported as machine logic: the likely-human downgrade and when
  the mark clears, the counters (any accepted non-`uncertain` category resets the
  uncertain count, `wait` included; only an accepted prediction resets the timeout
  count; a result after its deadline counts as a timeout), `_forget_category` after
  an inference error or timeout, and a customer-handled stage completing instead of
  applying its effect.
- The spec §10.1 table is one `match` that produces `ReplyDecided` as soon as the
  turn's prediction exists, without waiting for `TurnParked`. A prediction that
  completes AMD sends no `ReplyDecided`; `Detach.pending` decides its turn with the
  "AMD completed" row, as Python's `_authorize_reply` checks `FINISHED` first. A turn committed while
  AMD is `Pending` or past its overall timeout gets row 1 at once. The
  `agent_changed` row reads the agent from `TurnCommitted`.
- Entering or leaving `machine-ivr` while active sends `Dtmf` before any decision
  under the new stage. Other stage changes send nothing.
- `human_instructions` apply only to the reply of the turn whose `human` prediction
  completes AMD, and only if `had_machine_stage` is set (spec §10.2). A direct human
  answer replies under the agent's own instructions.
- In a machine stage, an allowed decision carries `silence =
  machine_silence_threshold`. AMD does not time it.
- The idle and inference-timeout rules count a reply as held from `ReplyCreated` until
  its first playout or its end, from `Speech` observations.
- **Idle** (`_update_idle`). The run is busy while a request is pending, the category
  is `wait`, a reply is held, the participant speaks, a watched speech is live, or
  `AgentBusy` is set. When not busy, the deadline is armed once, with
  `voicemail_idle_timeout` in `machine-vm` and `idle_timeout` otherwise. A stage
  change clears it, and a played voicemail message restarts it. The
  `max_inference_timeouts` completion waits while a request is pending or a reply is
  held.
- **Voicemail** (spec §10.4). The reservation belongs to the turn that was allowed
  the message and is released when a newer turn commits. A reply is denied while an
  attempt is in flight. The message counts as played only if its speech finished not
  interrupted and not failed, no assistant item was interrupted (a TTS failure
  partway), playout advanced, and the run was still active. An interrupted attempt
  can be retried. After a played message, `ivr_instructions` gain the "already
  played" sentence.
- **Classifier history** (spec §8.2). Each watched speech records, when it is allowed
  or first seen, whether the classifier sees its words or `(the agent spoke)`.
  Speeches with `add_to_chat_ctx` false are left out; speeches queued before attach
  are watched; items and tool calls from before listening started are filtered out.
  DTMF entries need a successful, non-empty output and are de-duplicated by call id.
  At most 20 items go into a request.
- **Speech clock.** Speech duration starts at activation if the participant is
  already speaking; boundaries before activation are ignored.
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

AMD completes inside its own transition: the lifecycle becomes `Finished` and the
request is cancelled, then AMD sends `Detach` with the decision for any turn still
undecided. `amd_completed` is emitted, and `execute()` returns, after the session
reports `Detached` and the runtime reports the legs closed (`In::LegsClosed`).

The completed category is the stage, except that after a played voicemail message a
`machine-unavailable` stage reports `previous_stage` (spec §12.2). `LegFailed`, or a
failing prediction or menu listener in the host, completes with `internal_error`; a
classifier `APIError`, invalid output or timeout gives an `inference_error`
prediction instead (spec §8.4). A menu extraction failure is silent.

### Ported unchanged

The proposal changes how AMD and the session talk, not AMD's public behavior. These
spec rules carry over as written; each is owned by the layer named, and each needs a
test or story before the port is done:

| Spec | Rules | Owner |
| --- | --- | --- |
| §4.1 | Options, defaults (`idle_timeout` 10 s, `voicemail_idle_timeout` 10 s, `timeout` 60 s, `inference_timeout` 2 s, `machine_silence_threshold` 1.5 s, `max_uncertain_turns` 3, `max_inference_timeouts` 3, `wait_until_answered` true, built-in stage instructions) and every `ValueError` | Python host and AMD constructor |
| §4.1 | Model selection: auto models when credentials exist, else the session's; `None` inherits a non-realtime Agent LLM; a decision model must support `choice`; the source of each model is logged; `llm_conn_options`; the `AMD` feature header only on Inference requests | Runtime legs |
| §4.3, §12.1 | Event fields: prediction (`speech_duration`, `delay`, `inference_duration`, `transcript`), completion (`voicemail_message_played`, `last_prediction`, `is_error`, the ten reasons), menu (`session_id`, `extraction_duration`). Turn ids are local to each run, start at 1 and may skip | `AmdEffect::Emit` |
| §5.1 | Entry checks: session started, no handoff or `AgentTask` in flight, one run at a time, realtime needs per-response tools and client tool replies and some STT, opening text needs TTS or realtime `say()`, opening instructions need an LLM | Attach command |
| §5.2 | Listening setup: 5 s track wait (`participant_missing`), `participant_disconnected` during setup or later, participant I/O links when no identity is given, no timeout on the SIP answer, overall deadline armed at activation | Callee leg and `In::Callee` |
| §5.3 | Attached rules beyond the list above: `update_agent()` and inline `AgentTask` switches raise; a realtime model without session STT drops a backchannel over agent speech | Session |
| §8.1 | Request contents: `stage`, `previous_prediction` (shown as "uncertain (likely human)" when marked), recommended list, `speech_duration`; the dual-transcript format; the decision-model transcript format; exactly one `record_result` call | Machine builds `ClassifyRequest`; leg renders it |
| §10.6 | Opening: `opening_instructions` generate with `tool_choice = none`; a failed opening is logged and AMD keeps listening | Session, on `Activated` |
| §11.2 | Menu extraction: 5 s deadline, at most 20 options, only a non-empty menu is emitted, only a newer turn with text cancels it, skipped for customer-handled stages and decision models, dropped after finish | Machine and menu leg |
| §13 | Remote sessions get each prediction (turn id, stage, inference duration or unset) and the completion (last prediction or unset) before `amd_completed` fires; menus are not forwarded | Host |

## DST

The DST runs the session core and the AMD executor as separate scheduled work, FIFO in
each direction, so it explores every interleaving of observations and answers. The
protocol must hold under all of them.

- **Fakes.** The classifier leg is a fake LLM. The search picks each `record_result`
  category from a callee persona (screener → human → human; voicemail → message →
  menu; hold → `wait`) and injects delays, failures and stale answers. A decision-model
  fake records `choice` calls. AMD's STT is a fake stream with its own lag. The fake
  LLM and the realtime fake record each request's messages, instructions and tools.
- **Scripted AMD.** Before the AMD machine exists, a scripted answerer drives the
  session's gate. It answers early (before the customer hook returns), late, for stale
  turns, detaches with a turn in its hook, and activates after participant speech.
- **Trace.**
  - `GateStep::{Committed, Decided, AnswerDropped, Superseded, Settled, Detached}` and
    `SpeechStep::AuthorizationHeld(Hold)`, once per change
  - `AmdStep::TurnBound { turn, has_text, late_wait }`
  - `RequestIssued { req, turn }` and `RequestCancelled { req }`
  - `Predicted { turn, source_req, raw, category, stage, reason, likely_human }`:
    `source_req` is the request whose result decided the turn, which for a `reused`
    prediction belongs to an older turn
  - `ReplyBound { turn, speech }`: an adopted preemptive generation has no turn of its
    own
  - `VoicemailPlayed`, `OpeningBypass { speech }`, `Completed`
- **Driver.** The developer model enters AMD, places the call, waits for
  `amd_completed`, and varies the options: stages without instructions, the opening,
  `machine_silence_threshold=0`.
- **Authored stories.** Port the scenarios of `tests/test_amd.py` as authored DST
  stories, and add stories for:
  - a correction before and after the reply exists
  - the classifier result and late final arriving in both orders
  - participant speech that restarts during a silence hold without a commit
  - completion while a turn is in its customer hook, for `human` and for a stage
    without instructions
  - a decision that arrives before the customer hook returns
  - `Activated` with an opening delivered after participant speech
  - a direct human answer adopting its preemptive generation, and a screening or
    human-after-machine reply discarding it
  - completion during a silence hold for each reason, releasing the held reply
    except for `machine-unavailable`
  - completion after a decision is stored but before it is applied, replacing it
    with the "AMD completed" row
  - a voicemail box hanging up while the message plays, which keeps playing
  - AEC warmup at the start of the opening, with AMD's STT still transcribing

  Authored stories are the only check against a rule and an implementation that share
  the same misreading of the spec.

Each spec §14 invariant has one owner:

| # | Invariant | Owner |
| --- | --- | --- |
| 1 | One prediction per turn at most | DST rule on `TurnBound`, `RequestIssued`/`RequestCancelled` and `Predicted` |
| 2 | A stale request never decides | DST rule: `Predicted.source_req` was never cancelled, and was issued for this turn or for an older turn whose result this empty turn reuses (spec §7.3) |
| 3 | `wait` never replies | DST rule `reply_follows_decision` |
| 4 | `uncertain` and `wait` keep the stage | DST rule on the `Predicted` sequence |
| 5 | Likely-human mark | DST rule on `raw`, `category` and `likely_human` |
| 6 | One voicemail message at most | DST rule on `VoicemailPlayed`, confirmed at the sink |
| 7 | Silence hold | DST sink rule, with silence computed from the fake VAD's ground truth, not from the session's or AMD's view |
| 8 | Stage instructions never saved | DST rule: no stage-instruction item in recorded items or storage |
| 9 | DTMF only in `machine-ivr` | DST rule: the tool appears only in requests generated while the gate's `Dtmf` was enabled, and runs only for a reply whose decision set `dtmf` |
| 10 | Cleanup on every path | DST rule after `Detached`: no gate, no hold, interruption settings equal their values at attach; host test for listener removal |
| 11 | AMD closes only models it created | Runtime leg unit test |
| 12 | `execute()` returns once | Python host test; DST driver asserts one `Completed`, after `Detached` |
| 13 | A customer-handled stage never replies | DST rule `reply_follows_decision` plus the `Completed` reason, including turns settled by `Detach` |
| 14 | Decision model: one `choice`, no menu | Leg unit test; DST decision-model fake sees no menu request |
| 15 | Only the opening plays before a prediction | DST sink rule with `OpeningBypass`; no opening plays after participant speech since activation |

Three rules cover the protocol itself. They check outcomes, not only that the session
followed its own rules:

- `reply_follows_decision`: a turn committed under the gate replies only after an
  allowed decision for it, from `ReplyDecided` or `Detach`. A denied turn yields no
  reply, and an allowed one at most one, or none with a traced refusal such as paused
  scheduling, the close, or a refused realtime request.
- `gate_settles`: every turn committed under the gate reaches a decision, a
  supersession or an exit within a bound, while AMD and the classifier make progress.
- `gate_answers_are_keyed`: the session applies a decision only to the latest
  committed turn of the current run; every other answer appears as `AnswerDropped`.

`speech_is_heard` treats a speech as waiting legitimately while the gate holds it, and
under `Hold::Silence` only until the fake VAD's ground truth shows the required
silence. The DST does not measure classifier accuracy, prompt quality, real STT lag
or the SIP answer path. Live simulations and room tests own those.

## Open questions

- How AMD's STT leg receives the participant's frames before the session's
  silence substitution, now that the tap fan-out is gone.

- Whether the pending reply decision becomes visible in telemetry later, and as what
  span.
- Whether stage instructions render as a system message, as `turn_instructions` does,
  or as a user message, as Python's AMD does. Model behavior may differ.
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
