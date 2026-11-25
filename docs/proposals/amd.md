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
| AMD's final message settles every turn committed under the gate, including one still in its customer hook. The session removes the gate and acknowledges at once; a turn still in its hook carries its settled decision, so completion never waits for customer code. | proposed |
| The AMD machine runs in its own executor. The protocol depends only on per-direction FIFO delivery, so the DST explores when AMD answers. | proposed |
| Preemptive generation may run while authorization is held, never with AMD's DTMF tool. An allowed reply adopts it only when the decision adds no instructions and no DTMF, and the session's adoption check passes. A stage change or completion does not cancel it by itself, unlike Python (spec §3, §12.3), so a direct human answer keeps its head start. | proposed |
| When AMD completes on a turn the session has already superseded, `Detach.pending` applies to the newer turn, `human_instructions` included. Python cannot reach this state, because its completion is synchronous; the newer turn is the same person's next words, so the instructions still fit. | proposed |
| The commit interrupt does not wait for an older turn's customer hook, unlike Python (`agent_activity.py:3017`). | proposed |
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
        /// With no speech boundary since the previous commit, the silence starts at
        /// `at` minus this delay.
        eot_delay: Option<Duration>,
        skip_reply: bool,
        /// The gate was active at commit. AMD gives an unbound turn row 1.
        bound: bool,
        /// The participant was still speaking at commit (spec §7.1 speech duration).
        speaking: bool,
        agent: AgentId,
        at: Ts,
    },
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
    /// playback, thinking, a pending false interruption, a running user-turn task or
    /// a parked turn (Python's `_is_agent_busy`). Sent at attach, then on change.
    AgentBusy { busy: bool, at: Ts },
    /// The session entered or left `Hold::Silence` for a turn. `held: false` ends the
    /// reply's hold for AMD's inference-timeout and idle rules.
    SilenceHold { turn: TurnId, held: bool, at: Ts },
    /// Participant speech boundaries and session STT finals.
    UserSpeech(UserSpeechObservation),
    /// Executed tool calls, for DTMF in classifier history (spec §11.1).
    ToolsExecuted { calls: Vec<ToolCallRecord> },
    /// The session applied `Detach`, or closed, and removed the gate.
    Detached { run: AmdRunId },
}
```

Observations that name a turn (`TurnEnded`, `ReplyCreated`, `SilenceHold`) go only to the
run that gated the turn, and to none once that run is detached. A turn that `Detach` settled
and that leaves its hook after a second run attached reports nothing to the second run.

### AMD to session

Inputs the session accepts from AMD.

```rust
pub enum AmdAnswer {
    /// Listening starts (spec §5.2). The session applying it is the activation
    /// instant: it stops discarding participant audio there, for its own paths and
    /// AMD's feed alike, so no participant speech exists before it on either side.
    /// The opening, if any, comes with it and is checked at the same point.
    Activated { run: AmdRunId, opening: Option<Opening> },
    /// The decision for a committed turn. It may arrive before or after the customer
    /// hook returns.
    ReplyDecided { run: AmdRunId, turn: TurnId, decision: ReplyDecision },
    /// A turn's transcript on a realtime session (spec §7.2), also for an older turn's
    /// late final and a turn whose prediction was reused. The session keeps it only for
    /// a model without user transcription and without session STT, read when it
    /// arrives. It goes to the agent's context and history, never to the server.
    InsertUserItem { run: AmdRunId, turn: TurnId, text: String },
    /// AMD completed. `pending` settles every turn committed under the gate whose
    /// decision is not applied yet: the `wait` and "AMD completed" rows of spec §10.1.
    Detach { run: AmdRunId, pending: ReplyDecision, cancel: DetachCancel },
}

pub struct ReplyDecision {
    allow: bool,
    /// Stage instructions for this reply and its tool follow-ups.
    instructions: Option<String>,
    /// AMD may add the built-in `send_dtmf_events` to this reply (`machine-ivr`,
    /// spec §11.1). An agent tool with the same id is kept instead.
    dtmf: bool,
    /// Silence the participant must keep before the reply plays (spec §10.3). The
    /// session times it.
    silence: Option<Duration>,
}
```

Attaching AMD is a session command, as entering any other mode is. It installs the
gate for the run, sets the rules in
[Session rules while attached](#session-rules-while-attached) and starts the hold.
`AmdCommand::Attach` carries the `ToolSpec` of AMD's built-in `send_dtmf_events`, as the
host declares it. The core does not keep its own copy of the schema. It also carries
`amd_stt`, which says whether the run has its own STT. The session needs it for the
realtime entry check in [Realtime](#realtime).

## The session side

### Gate state

```rust
// AgentActivity
gate: Option<Gate>,

struct Gate {
    run: AmdRunId,
    active: bool,                      // set by `Activated`
    latest: Option<Latest>,
    hold: Hold,
    dtmf_tool: ToolSpec,               // from `AmdCommand::Attach`
}
struct Latest {
    turn: TurnId,
    /// The gate was active at commit. Only a bound turn's decision moves the hold.
    bound: bool,
    reply: TurnReply,
}
enum TurnReply {
    Waiting,                           // in its customer hook, no decision
    Parked(ParkedTurn),                // back from its hook, no decision; at most one
    Decided(ReplyDecision),            // decided while still in its hook
    Applied,
    Ended,                             // left without a decision applied
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
    /// measured from the session's latest speech boundary. `deadline` is the armed
    /// expiry; a timer for any other deadline is stale.
    Silence { turn: TurnId, needed: Duration, deadline: Option<Ts> },
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
  trace `GateStep::AnswerDropped`. If the turn is parked, apply it now. The existing
  `waiting_turns` check runs first, so a decision for a turn behind another turn's
  hook is never applied to the older turn.
- **Customer hook returns** (`finish_turn`): if the turn carries a decision settled by
  `Detach`, apply it without a gate. Otherwise, if `latest` has a decision, apply it;
  otherwise park the turn. `waiting_turns` is empty whenever a turn parks, so at most
  one turn is parked.
- **Apply a decision.** A denied turn commits only its user item, discards the
  preemptive generation and sets `Hold::Denied`. An allowed turn calls `reply_to_turn`
  with the decision's instructions and DTMF permission, and sets `Hold::Silence` if
  the decision asks for silence, else `Hold::Open`. The commit records whether the gate
  was `active`. A turn committed before `Activated` is not bound, so its allow keeps
  `Hold::Entry`, even when its hook returns after activation. Python never resumes
  authorization for a turn it did not bind (`detector.py:1047`). `TurnCommitted`
  carries `bound`. AMD can be active before the gate applies `Activated`, so AMD
  cannot tell from its own state.
- **Decision for an ended turn.** A decision for a turn that ended without one
  (`skip_reply`, `StopResponse`, a drop) is dropped as `TurnEnded`. A second decision
  for a turn is dropped as `Duplicate`.
- **Silence.** Under `Hold::Silence`, a `Timer::AmdSilence` is armed for the latest
  speech boundary plus `needed`. Boundaries use the VAD boundary time, backdated as
  Python's are (`detector.py:668`), not event time, or every reply waits an extra VAD
  `min_silence`. If no boundary came since the previous commit, the boundary is the
  speech end estimated from `eot_delay`, else the commit (`detector.py:708`). Start of
  speech disarms it; end of speech re-arms it with a new deadline. A firing opens the
  hold only if its deadline is the one armed, the turn is still `latest` and the
  participant is silent. Entering and leaving the hold emit `SilenceHold`.
- **Turn exits without a decision applied** (`skip_reply`, `StopResponse`, the
  uninterruptible drop): emit `TurnEnded`, set `Hold::Denied`.
- **Authorization.** The scheduler authorizes the head speech only under `Hold::Open`,
  or when the head is the opening. A held head stays chosen in a held scheduling step,
  distinct from `Current`. A speech queued before attach and held at the head keeps the
  opening behind it, as in Python (spec §10.6).

### Activation and the opening

`Activated` ends the `Pending` audio discard, and applying it is the activation
instant for both sides. AMD starts its overall deadline and speech clock when it sends
`Activated`, but receives no frames before the session applies it. If it carries an
opening, the session creates it with the hold bypass only if the participant is not
speaking; otherwise it drops it (spec §10.6). Start of speech before the opening's
first playout cancels it. Both are local facts, so a late `Activated` cannot let a
stale opening play. The opening is the only speech that
bypasses the hold, and it is created through an internal origin, since `say()` and
`generate_reply()` are refused while the gate is attached.

### Detach

`Detach` is applied in this order (spec §12.3):

1. When `cancel` forbids replies (`machine-unavailable` or a stage the application
   handles), cancel every speech with no playout yet, including an authorized reply
   still generating, an unplayed opening and the preemptive generation. Keep a speech
   that is playing or paused, as `_cancel_pending_speeches` does: a voicemail message
   keeps playing when the box hangs up. Otherwise keep the preemptive generation;
   adoption decides whether it still matches (spec §12.3 cancels it).
2. Restore interruption settings and clear the rules of
   [Session rules while attached](#session-rules-while-attached).
3. Settle turns: `pending` replaces any decision the session has not applied yet,
   stored for a turn in its customer hook or for a parked turn. Python decides when
   the reply is about to be made: a `wait` category denies first, then a finished run
   answers with the "AMD completed" row (`detector.py:1060`). AMD builds `pending` the
   same way: `allow` is false when replies are forbidden or the category is `wait`,
   and it never asks for silence or DTMF. A parked turn has it applied now; a turn
   still in its customer hook carries it to `finish_turn`.
4. Free authorization. A reply already held for silence plays at once, for every
   completion reason (spec §10.3); only a `cancel` that forbids replies removes it.
5. Remove the gate and emit `Detached`. Later commits are not gated (spec §9.5), and
   a new AMD run can attach while an old turn is still in its hook.

A correction after `Detach` remakes an unheard gated reply with the same stage
instructions and tools. The session keeps the reply's request payload until the reply
ends.

The close ends an attached run as its completion would. When closing begins, the session
releases the gate locally with `Keep` semantics: a parked turn keeps its words with no
reply, a turn in its hook is settled to no reply, the hold opens so a draining close plays
what the gate held, and the session emits `Detached`. Spec §12.3 resumes authorization on
every completion, the close included.

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
  the newest reply regardless of turn, so it gains that guard. AMD moves voicemail
  tracking to the replacement and ignores the first speech's interrupted end, as
  `_track_voicemail` does (`detector.py:1191`).
- **Turn exits.** `TurnEnded` fires on every exit: reply, `StopResponse`,
  `skip_reply`, `supersede_turn`, the uninterruptible drop, local supersession, and the
  close. No run is attached inside a handoff's window, so `hold_user_turn` never ends a
  gated turn. The close ends a parked turn as abandoned, with its words kept.
- **Inspection.** The idle check and the re-decide check in `decide_user_turn` also
  inspect `parked`.

### Realtime

The session checks a realtime model at attach, in `detector.py`'s order (spec §5.1).
Each check has its own `AmdAttachRefused` reason:

1. `RealtimeServerTurnDetection`: the server detects turns. Server-initiated replies
   never reach the gate. A model whose server detection the session switched off is
   client-side.
2. `RealtimeTools`: the model cannot take per-response tools, or it replies to tool
   results on its own.
3. `RealtimeNoStt`: the session has no STT and the run has none (`amd_stt`). AMD
   would have nothing to classify.

All three are session refusals, as the handoff and task checks are. The runtime
reports AMD's STT in `Attach` and does not check it again.

Without the gate, both realtime commit paths skip `on_end_of_turn`: `on_turn(Commit)`
calls `realtime_user_turn` and returns, and `commit_user_turn` forces `skip_reply` and
returns early when session STT is disabled. With the gate attached and `!skip_reply`:

- the session commits the server audio buffer at commit, before the decision, as
  Python does, so a denied `wait` turn leaves no audio for the next allowed reply
- the turn goes through `on_end_of_turn`, even with no session STT; its classifier
  text then comes from AMD's late final (spec §6.2). A manual commit still waits the
  endpointing delay, as `_run_eou_detection` does
- `reply_to_turn` sends the response request once, after an allowed decision. The
  request is sent at authorization, so a reply held for silence starts generating
  after it, as spec §10.3 requires. The server takes one response at a time, so the
  request may leave after an older response settles
- the request carries the decision's stage instructions, passed through unchanged.
  The reply's tool follow-ups carry them too
- a decision that allows DTMF resolves its toolset once: the agent's tools, the
  session's tools and `send_dtmf_events`, unless an agent tool has that name. The
  speech runs that toolset. The session sets the server's tools to it with
  `UpdateTools` before the reply's first request, because `SpeechSpec.toolset` alone
  does not reach the server. The agent's tools come back when the reply's speech ends,
  after its last tool follow-up. A fallback swap's frame also sends the agent's tools
- the target is a per-response `tools` field on the realtime `GenerateReply`, as
  Python's `generate_reply(tools=...)` sends. It would replace the swap and its
  restore. `GenerateReply` is a host-recorded type on the realtime wire, so the field
  changes the host ABI and every realtime plugin. This step keeps the swap
- `commit_gated_turn_item` is the one owner of a gated turn's words, for a reply, a
  denial and a turn superseded in its hook. A pipeline keeps the STT's words. A
  realtime model with user transcription keeps none: the server's transcript of the
  committed audio is the turn's item. A model without it keeps the session's STT
  words, when there are some
- AMD sends `InsertUserItem` for every turn of a realtime run. The session decides
  on its live model and STT when the answer arrives: it keeps the words only for a
  model without user transcription and without session STT, so the item is never
  added twice. It adds them to the agent's context and history and emits
  `conversation_item_added`, as `_add_realtime_transcript` does. In any other case it
  drops them as `NotNeeded`. The runtime does not decide from the model at entry: a
  handoff can change the model's transcription before AMD enters
- both items stay off the wire. The server holds the turn's audio, so the realtime
  projection (`for_realtime`) leaves these items out, as it leaves out the server's
  own items. A later `UpdateChatCtx`, such as a host `update_chat_ctx` or a transcript
  correction, does not send the turn again

The opening also works on a realtime model (spec §10.6). Opening text uses the model's
`say()` when it supports it and the session has no TTS. Opening instructions ask the
model for a reply with `tool_choice = none`. Both bypass the hold like a pipeline
opening.

### Reply payload

- **Stage instructions.** `SpeechSpec.turn_instructions` already carries per-reply
  instructions that never enter saved history. Stage instructions use it, with two
  changes. It renders as a system message, and Python's AMD adds a user message; the
  role is an open question below. It is cleared for tool follow-ups, and spec §10.2
  keeps stage instructions for them, so a gated reply keeps it. It is not
  `TurnEdits.instructions` or `instructions_override`, which replace the agent's
  prompt.
- **Tools.** The only tool AMD adds is the built-in `send_dtmf_events`, to a reply
  whose decision sets `dtmf` (spec §11.1), unless the agent already has a tool with
  that id. This replaces Python's `on_reply_generation` hook. Preemptive generations
  never get it: every allowed `machine-ivr` reply carries stage instructions and
  generates again anyway, except a likely-human reply, which waits for silence and
  gains nothing from a head start.
- **Preemptive generation.** v2.0's adoption check compares the turn, hook edits, the
  chat-history version and the words (`activity/speech.rs:306`), not the request's
  tools or per-reply instructions. So the gate adds its own condition: an allowed
  reply adopts the preemptive generation only if the decision has no instructions and
  no `dtmf`; otherwise it discards it and generates again. On a direct human answer
  the request is the one a new generation would send, so the head start is kept.
  Speeches fix `allow_interruptions` at creation (`recognition.rs:939`), and AMD
  forces it on while attached, so a generation adopted after `Detach` takes the
  restored value. Python cancels the preemptive generation on every stage change and
  at completion; this port relies on these conditions instead.

### Session rules while attached

The gate's presence gates these, so DST rules read the same flag (spec §5.3): skip
`min_words` at end of turn, force interruptions, refuse `say()` and `generate_reply()`
except the opening, cancel handoffs, skip `on_user_turn_exceeded`, drop deferred tool
replies, do not regenerate after a realtime fallback swap, and discard participant audio
until `Activated`.

- **Realtime backchannels.** A realtime model without session STT drops a backchannel
  over agent speech instead of committing it (spec §5.3). While the gate is attached,
  the session asks whether the server's turn detection runs, not whether the model
  has it. A model whose server detection the session switched off then drops the
  backchannel too. Without the gate the check is unchanged: it reads the model's
  capability, while Python's `_rt_turn_detection_enabled` reads whether detection is
  enabled. That divergence predates AMD.
- **False interruptions.** The same value decides `cancel_false_interruption` at
  commit. Under the gate, a model whose server detection the session switched off
  cancels a pending false interruption at commit, as Python does.

- **Interruptions.** AMD's requirement is laid over every option layer while the run is
  attached, so a host `override_options` entered during AMD does not turn interruptions
  off. At detach the remaining layers decide again.
- **Deferred tool replies.** A tool update that arrives while a run is attached keeps
  its output on the record and owes no reply, before or after detach. A reply whose
  instructions were rendering when the run attached is not scheduled.

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
AMD's stream has its own slot on the audio router, `AudioChannel::Amd`. It is a raw
channel on the real side of the gate, so it hears the room frame unconverted and is
never substituted. The router drops every frame until the gate applies `Activated`,
so AMD hears nothing from before activation, as the session does not.

The stream opens when the run starts listening. Python opens it on the first frame
and keeps that frame. Here the executor sends `HoldAmd(true)` to the router just
before `Activated`, on the same queue. From then on, the router holds the channel's
frames until the stream's sink is installed, and delivers them first. The hold counts
media duration and keeps the last `AMD_BACKLOG` (30 s); beyond it the oldest frames
go. `HoldAmd(false)` drops the hold when the stream fails before its sink is installed
or the run stops. An Inference STT's connections, retries included, carry AMD's client
feature header (`stream(extra_headers=...)`); another STT gets none.

If the AMD stream fails or will not open, the leg reports `SttFailed`: AMD closes it,
ends every late-final wait at once, and uses session transcripts from then on (spec
§6.3). This differs from Python on purpose. Python lets each wait run out, but no
final can come. A blank final does not consume a turn's late-final wait.

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
    speech: SpeechClock,                      // speaking, started_at, duration, ended_at
    agent_busy: bool,
    held: Option<TurnId>,                     // the turn whose reply waits for silence
}

enum In {
    Observed(AmdObservation), Classified { req, result }, MenuExtracted { req, .. },
    AmdFinal(..), SttFailed, LegFailed, Callee(Callee), Timeout(AmdTimer), LegsClosed, Close,
}

enum AmdEffect {
    Answer(AmdAnswer), Classify(ClassifyRequest), CancelClassify, ExtractMenu(..),
    CancelMenu, SetDeadline(AmdTimer, Ts), ClearDeadline(AmdTimer), Emit(AmdEvent),
}

enum AmdTimer { Inference(ReqId), LateFinal(TurnId), Idle, Overall, Menu(ReqId) }
```

`In::Classified` carries the model time the leg measured around the model call.
Python times `classify` alone, after the late-final wait.

- `stage::next` is a line-for-line port of `_fsm.py`. The rules around it live in
  `detector.py` and are ported as machine logic: the likely-human downgrade and when
  the mark clears, the counters (any accepted non-`uncertain` category resets the
  uncertain count, `wait` included; only an accepted prediction resets the timeout
  count; a result after its deadline counts as a timeout), `_forget_category` after
  an inference error or timeout, and a customer-handled stage completing instead of
  applying its effect.
- The spec §10.1 table is one `match` that produces `ReplyDecided` as soon as the
  turn's prediction exists, without waiting for the customer hook. A prediction that
  completes AMD sends no `ReplyDecided`; `Detach.pending` decides its turn.
  `_authorize_reply` checks `wait` first and `FINISHED` second (`detector.py:1060`),
  and `pending` follows the same order. A turn committed while AMD is `Pending`, an
  unbound turn (`bound: false`) and a turn past the overall timeout get row 1 at once.
  AMD keeps no record of an unbound turn. The `agent_changed` row reads the agent from
  `TurnCommitted`.
- An allowed decision in `machine-ivr` while active sets `dtmf`. Stage changes send
  nothing else to the session.
- `human_instructions` apply only to the reply of the turn whose `human` prediction
  completes AMD, and only if `had_machine_stage` is set (spec §10.2). A direct human
  answer replies under the agent's own instructions.
- In a machine stage, an allowed decision carries `silence =
  machine_silence_threshold`. AMD does not time it.
- **Held reply.** The idle and inference-timeout rules count a reply as held from an
  allowed decision with `silence` until the session reports `SilenceHold { held: false }`
  for that turn, or until a turn end that prevents the reply. Every end except
  `Replied` prevents it. `TurnEnded { Replied }` does not release the reply: the gate
  sends `ReplyCreated`, then `TurnEnded { Replied }`, and only then enters the hold.
  A turn that asked for no reply (`skip_reply`), or that ended before its prediction,
  is never held. This matches Python's `_reply_held_at` (`detector.py:1126`), which
  is true from the prediction until the silence ends, whether or not a speech exists
  yet. Counting from `ReplyCreated` instead would let the third inference timeout
  complete AMD before the reply exists, and `Detach` would then replace its silence
  and stage instructions.
- **Idle** (`_update_idle`). The run is busy while a request is pending, the category
  is `wait`, a reply is held, the participant speaks, or `AgentBusy` is set.
  `AgentBusy` covers every queued or playing speech except the preemptive generation.
  When not busy, the deadline is armed once, with `voicemail_idle_timeout` in
  `machine-vm` and `idle_timeout` otherwise. A stage change clears it, and a played
  voicemail message restarts it while the stage is `machine-vm` (`detector.py:1217`).
- **Inference timeout limit** (`detector.py:1288`). Once `max_inference_timeouts` is
  reached and no request is pending, AMD completes with `inference_timeout` unless a
  reply is held. A prediction checks this before the turn's decision, as Python's
  `_record_prediction` runs `_update_idle` first. AMD counts the latest turn's reply as
  held there only if it would wait for silence: the stage is a machine stage, the
  category is not `wait`, `machine_silence_threshold` is not zero, and the participant
  speaks or has been silent for less than the threshold. Otherwise `Detach.pending`
  settles the turn with the "AMD completed" row, without stage instructions. AMD's
  speech clock measures that silence as the gate does: from the latest speech end, or
  from the commit minus `eot_delay` when no boundary came since the previous commit.
- **Voicemail** (spec §10.4). The reservation belongs to the turn that was allowed
  the message and is released when a newer turn commits. A reply is denied while an
  attempt is in flight. The message counts as played only if its speech finished not
  interrupted and not failed, no assistant item was interrupted (a TTS failure
  partway), playout advanced, and the run was still active. An interrupted attempt
  can be retried. After a played message, `ivr_instructions` gain the "already
  played" sentence.
- **Classifier history** (spec §8.2). Each watched speech records, when it is allowed
  or first seen, whether the classifier sees its words or `(the agent spoke)`.
  Speeches with `add_to_chat_ctx` false are left out. Speeches queued before attach
  are watched, and their words count when they play. A speech playing at attach is
  watched, but its words stay out of the history: they started before AMD's entry
  (Python's `_started_at` filter). Tool calls created before AMD's entry are filtered
  out the same way.
  DTMF entries need a successful, non-empty output and are de-duplicated by call id.
  At most 20 items go into a request.
- **Speech clock.** The gate reports speech boundaries and finals only once it is
  active, which is after AMD activates. Speech duration counts from activation.
- **Menus** (spec §11.2). A menu result at or after its 5 s deadline is dropped, as a
  late classifier result is. A menu with more than 20 options is not observed: it
  fails Python's validation (`_inference.py:112`). `CancelMenu` stops the menu leg
  wherever Python cancels `_menu_atask` (a newer turn with text, `detector.py:749`),
  when a new extraction replaces it, at its deadline and at completion.
- **Turn registry.** An ended turn older than the newest turn with a prediction is
  dropped, unless a late final or the request still waits on it.
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
request is cancelled, then AMD sends `Detach` with the decision for any turn whose
decision is not applied. `amd_completed` is emitted, and `execute()` returns, after
the session reports `Detached` and the runtime reports the legs closed
(`In::LegsClosed`). The session sends `Detached` when it applies `Detach`, so neither
waits for a customer hook.

AMD's view lags the session's. It can complete on turn N's result after the session
has committed N+1; `pending` then settles N+1 (see Decisions). A held reply can exist
at completion only for `timeout`, `cancelled`, `participant_disconnected` and
`internal_error`: a completion caused by a prediction needs a newer commit, which has
already cancelled the older reply.

The completed category is the stage, except that after a played voicemail message a
`machine-unavailable` stage reports `previous_stage` (spec §12.2). `LegFailed`, or a
failing prediction or menu listener in the host, completes with `internal_error`; a
classifier `APIError`, invalid output or timeout gives an `inference_error`
prediction instead (spec §8.4). A menu extraction failure is silent. Python also
treats a model's `ValueError` and `asyncio.TimeoutError` as `inference_error`. The
core cannot tell those apart from other host errors, because a host error outside the
API taxonomy carries only its class name. Here they end the run with
`internal_error`.

The callee leg reports `participant_disconnected` when the participant leaves, and
also when the room disconnects or closes, during setup or later.

### Runtime

**Entry.** `enter()` checks the options and the STT, then hands the run to one task on
the session's AMD task set. That task routes the run's observations, sends `Attach`,
reads the answer and runs the executor. Observations are routed by run, so a refused
second run never displaces the attached one. A refused run closes the models AMD
created before `enter()` returns its error, as `__aenter__` does. A caller that stops
waiting for `enter()` cancels the run's close token: if the session accepts the
attach, the run completes at once with `cancelled`, and its models close. An entry
that the task set refuses never sends `Attach`, and its models close as a cleanup on
that task set.

**Legs.** Each leg feeds the machine through `TaskSet::scoped_feeding`. A leg that
dies reports its death as an input: `LegFailed` for the classifier, menu and callee
legs, `SttFailed` for AMD's STT. This matches Python's `_on_task_done`, which ends
the run with `internal_error`. At completion the executor asks every leg to stop. The
callee leg closes its room events within the teardown bound, and AMD's STT closes its
stream. The executor joins the legs, then closes each model AMD created, each close
bounded by the teardown bound, then steps `LegsClosed`.

**Session close.** After the session's loop ends, its close joins every AMD run's
task within the teardown bound, as `_teardown_activity` awaits `amd.aclose()`. A run
that misses the bound is cancelled and logged.

**Queues.** Observations reach the executor on an unbounded channel: the session's
loop never waits on AMD, and the executor reads them as it steps. Leg reports are
bounded by their producers. The host's events queue holds `EVENTS_CAPACITY` (128)
events; past that the newest is dropped and logged, and `execute()` still returns the
outcome.

**Decision model.** A decision model is the `DecisionModel` port: `choose(cx,
ChoiceRequest)` makes one `evaluate` call with one `Choice` decision. The request
carries the input (request state and transcript), the decision's name and prompt, the
options in order without descriptions, the connection options and the extra headers.
It returns the chosen option, or `None` when the response holds no choice. A decision
model never extracts menus.

### Ported unchanged

The proposal changes how AMD and the session talk, not AMD's public behavior. These
spec rules carry over as written; each is owned by the layer named, and each needs a
test or story before the port is done:

| Spec | Rules | Owner |
| --- | --- | --- |
| §4.1 | Options, defaults (`idle_timeout` 10 s, `voicemail_idle_timeout` 10 s, `timeout` 60 s, `inference_timeout` 2 s, `machine_silence_threshold` 1.5 s, `max_uncertain_turns` 3, `max_inference_timeouts` 3, `wait_until_answered` true, built-in stage instructions) and every `ValueError` | Python host and AMD constructor |
| §4.1 | Model selection: auto models when credentials exist, else the session's; `None` inherits a non-realtime Agent LLM; a decision model must support `choice`; the source of each model is logged; `llm_conn_options`; the `AMD` feature header only on Inference requests and Inference STT connections | Runtime legs |
| §4.3, §12.1 | Event fields: prediction (`speech_duration`, `delay`, `inference_duration`, `transcript`), completion (`voicemail_message_played`, `last_prediction`, `is_error`, the ten reasons), menu (`session_id`, `extraction_duration`). Turn ids are local to each run, start at 1 and may skip | `AmdEffect::Emit` |
| §5.1 | Entry checks: session started, no handoff or `AgentTask` in flight, one run at a time, realtime needs client-side turn detection, per-response tools and client tool replies, and session STT or AMD STT, opening text needs TTS or realtime `say()`, opening instructions need an LLM | Attach command (`amd_stt`) |
| §5.2 | Listening setup: 5 s track wait (`participant_missing`), `participant_disconnected` during setup or later, the participant the room I/O links when no identity is given (waited for within the same 5 s), no timeout on the SIP answer, overall deadline armed at activation | Callee leg and `In::Callee` |
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
  The realtime fake also records each socket's model transcription and each `say()`.
  The session journals the speech each realtime request is for
  (`RealtimeStep::Requested`), so the rules join a response to its turn by speech, not
  by a time window.
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
  - participant speech starting right after `Activated`, before the opening's first
    playout
  - a direct human answer adopting its preemptive generation, and a screening or
    human-after-machine reply discarding it
  - completion during a silence hold for each reachable reason, releasing the held
    reply, and completion into a stage the application handles while a reply is
    generating but not yet playing, which cancels it
  - completion after a decision is stored but before it is applied, replacing it
    with the "AMD completed" row
  - a voicemail box hanging up while the message plays, which keeps playing
  - the third inference timeout in `machine-vm`, with the hook returning before and
    after it
  - a `wait` turn in its hook at `timeout` and at `aclose()`
  - a `machine-ivr` stage followed by a direct `human` completion, with an agent that
    sets `allow_interruptions=False`
  - a hook that awaits `execute()`, and a second attach while the old turn is in its
    hook
  - a correction while the voicemail reply is held
  - silence that expires, speech that restarts and ends, then the stale expiry
  - an agent that defines its own `send_dtmf_events`, across IVR entry and exit
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
| 7 | Silence hold | DST rule judged at the instant the gate lets the reply go, with silence computed from the fake VAD's ground truth, not from the session's or AMD's view. After the release, new speech follows ordinary interruption, as in Python, so a later first frame says nothing about the hold. A detach exempts only a reply it let go. |
| 8 | Stage instructions never saved | DST rule: no stage-instruction item in recorded items or storage |
| 9 | DTMF only in `machine-ivr` | DST rule: AMD's built-in tool appears only in a reply whose decision set `dtmf`, never in a preemptive generation |
| 10 | Cleanup on every path | DST rule after `Detached`: no gate, no hold, interruption settings equal their values at attach; host test for listener removal |
| 11 | AMD closes only models it created | Runtime leg unit test |
| 12 | `execute()` returns once | Python host test; DST driver asserts one `Completed`, after `Detached` |
| 13 | A customer-handled stage never replies | DST rule `reply_follows_decision` plus the `Completed` reason, including turns settled by `Detach` |
| 14 | Decision model: one `choice`, no menu | Leg unit test; DST decision-model fake sees no menu request |
| 15 | Only the opening plays before a prediction | DST sink rule with `OpeningBypass`; no opening plays after participant speech that started after activation |

Six rules cover the protocol itself. They check outcomes, not only that the session
followed its own rules:

- `reply_follows_decision`: a turn committed under the gate replies only after an
  allowed decision for it, from `ReplyDecided` or `Detach`. A denied turn yields no
  reply, and an allowed one at most one, or none with a traced refusal such as paused
  scheduling, the close, or a refused realtime request.
- `gate_settles`: every turn committed under the gate reaches a decision, a
  supersession or an exit within a bound, while AMD and the classifier make progress.
- `gate_answers_are_keyed`: the session applies a decision only to the latest
  committed turn of the current run; every other answer appears as `AnswerDropped`.
- `adopted_request_matches`: an adopted preemptive generation's recorded request
  (messages, instructions, tools) and interruptibility equal those a new generation
  would have at adoption.
- `completion_respects_wait`: no turn whose AMD category at `Completed` is `wait` gets
  a reply from `Detach`.
- `detached_is_prompt`: `Detached` follows `Detach` within a bound that does not
  depend on a customer hook.

On a realtime model, `gated_turn_has_one_user_item` checks the words rule from what AMD
sent and the fake server's commits: a turn has no item of the session's own when the
socket's model transcribes, else exactly one, and none is empty.
`realtime_turn_audio_committed` checks that every gated turn commits the server's buffer
once, before its reply's request.

`speech_is_heard` treats a speech as waiting legitimately while the gate holds it. The
opening never waits on the hold. A reply to a bound turn with an applied allow waits only
under `Hold::Silence`, until the fake VAD's ground truth shows the required silence.
`observations_pair` checks that `AgentBusy` is reported only on change and that
`SilenceHold` enters and leaves once per turn, only for a decision that asked for
silence. The DST does not measure classifier accuracy, prompt quality, real STT lag
or the SIP answer path. Live simulations and room tests own those.

## Open questions

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
