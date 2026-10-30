//! No dead air: a speech the agent was given plays within a bound, or the run says why it has not.
//!
//! From its creation, a speech's audio reaches the speaker (`SinkCalled { Push }`) within
//! [`BOUND_MS`] of virtual time, or by then it has ended (how is other rules' to judge), or
//! something it rightly waits on still holds at the bound: a model, voice, tool, hook or server
//! response the case has yet to answer; the floor held by another speech that is playing, or by
//! the developer (`wait_for_idle_and_hold`); authorization held (AMD); the person speaking; the
//! session closing. What is left is an agent with something to say, nothing to wait for, saying
//! nothing: a reply queued behind a speech paused for good, as c6881abb9's realtime onset left one.
use std::collections::BTreeMap;

use lk_agents_abi::{RealtimeCommandKind, voice::events::UserState};
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    agent_session::effects::{GateCause, GateOp, Timer},
    events::{SessionEvent, SpeechOutcome},
    trace::{
        CoreStep, GenerationStep, InterruptCause, InterruptStep, RealtimeStep, PlayoutStep, RoomEventKind, RoomStep, SinkCall,
        SpeechStep, Step, TimerName, TurnStep,
    },
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{Input, RoomIn, RtEvent},
    record::{Handle, Kind, Observed, Recording, SpeechView},
};

/// How long a speech with nothing to wait on may stay silent.
pub const BOUND_MS: u64 = 15_000;

/// What the case owes the driver while it is open: a request it has not answered, a `say` stream
/// it has not ended.
const OWED: [Kind; 6] =
    [Kind::Chat, Kind::Tts, Kind::Tool, Kind::Hook, Kind::Generation, Kind::Stream];

pub struct SpeechIsHeard;

impl Rule for SpeechIsHeard {
    const CONTRACT: Contract = Contract {
        id: "speech_is_heard",
        guarantee: "A speech plays within a bound of its creation, ends, or waits on something that still holds: an unanswered provider, tool or hook, the floor held by a playing speech or the developer, authorization held, the person speaking, the close.",
        examples: &[
            played_in_time,
            waited_on_the_model,
            waited_behind_the_persons_utterance,
            waited_on_a_stalled_voice,
        ],
        counterexamples: &[
            silent_behind_a_paused_speech,
            silent_after_the_persons_utterance,
            silent_with_the_voice_due,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let journal = r.journal();
        let end = r.now().at_ms;
        let closing = r.close_begun().map(|c| c.at_ms);
        // When each speech first pushed audio.
        let mut pushed: BTreeMap<u64, u64> = BTreeMap::new();
        // When each speech was announced done.
        let mut done: BTreeMap<u64, (u64, SpeechOutcome)> = BTreeMap::new();
        // Each speech's journalled interruption cause.
        let mut cut: BTreeMap<u64, InterruptCause> = BTreeMap::new();
        // When the scheduler first let each speech go.
        let mut authorized: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        // When a preemptive reply's turn ended and the driver took or dropped the guess.
        let mut reconciled: BTreeMap<u64, Observed> = BTreeMap::new();
        // When the person was heard to start speaking.
        let mut cancelled: std::collections::BTreeSet<u64> = Default::default();
        // In occurrence order: an end and a start at one instant (the realtime server's
        // `speech_stopped` then `speech_started`) leave the person speaking.
        let mut onsets: Vec<Observed> = Vec::new();
        let mut offsets: Vec<Observed> = Vec::new();
        for (at, step) in journal {
            match step {
                Step::Playout(PlayoutStep::SinkCalled {
                    speech: Some(speech),
                    call: SinkCall::Push { .. },
                }) => {
                    pushed.entry(speech.0).or_insert(at.at_ms);
                }
                // The console's speaker plays a speech with no sink push (seed 83204).
                Step::Speech(SpeechStep::AgentStartedSpeaking { speech }) => {
                    pushed.entry(speech.0).or_insert(at.at_ms);
                }
                Step::Emitted(SessionEvent::SpeechDone { speech, outcome }) => {
                    done.entry(speech.0).or_insert((at.at_ms, *outcome));
                }
                Step::Interrupt(InterruptStep::Interrupted { speech, cause, .. }) => {
                    cut.entry(speech.0).or_insert(*cause);
                }
                Step::Speech(SpeechStep::Authorized { speech }) => {
                    authorized.entry(speech.0).or_default().push(at.at_ms);
                }
                Step::Turn(TurnStep::PreemptiveReconciled { speech, .. }) => {
                    reconciled.entry(speech.0).or_insert(*at);
                }
                // A realtime reply whose response the server cancelled ends `Completed` with
                // nothing said: as cut as an interrupted one.
                Step::Realtime(RealtimeStep::GenerationStatus { speech, status })
                    if status == "Cancelled" =>
                {
                    cancelled.insert(speech.0);
                }
                Step::Emitted(SessionEvent::UserStateChanged { new, .. }) => {
                    if *new == UserState::Speaking {
                        onsets.push(*at);
                    } else {
                        offsets.push(*at);
                    }
                }
                _ => {}
            }
        }
        for speech in r.host_view().speeches().into_iter().filter(SpeechView::is_known) {
            let Some(created) = speech.created_at() else { continue };
            // A preemptive reply waits on the turn it guessed, which may never end: its bound runs
            // from the turn's end, when the driver took the guess.
            let created = if speech.is_speculative() {
                match reconciled.get(&speech.id) {
                    Some(at) if speech.is_scheduled() => *at,
                    _ => continue,
                }
            } else {
                created
            };
            // The bound runs from the creation, and again from when the scheduler let it go: a
            // speech that waited its turn (the floor, the gap, authorization, the person) has the
            // whole bound once it may play.
            // Each step is let go again once its tools are done (a tool the case answers late):
            // a let-go within the bound gives the whole bound again.
            let mut deadline = created.at_ms + BOUND_MS;
            for a in authorized.get(&speech.id).into_iter().flatten() {
                if *a <= deadline {
                    deadline = deadline.max((*a).max(created.at_ms) + BOUND_MS);
                }
            }
            if end <= deadline || closing.is_some_and(|c| c <= deadline) {
                continue;
            }
            let played = pushed.get(&speech.id).is_some_and(|at| *at <= deadline);
            let ended = done.get(&speech.id).filter(|(at, _)| *at <= deadline);
            // A speech that ended unheard, cut short: something after it was made must have cut
            // it -- the person starting to speak, the host, the close, the scheduler's own
            // refusal -- not a stale interruption meant for another.
            let unheard_cut = match ended {
                Some(&(at, SpeechOutcome::Interrupted)) => Some(at),
                // A failed one is `speech_error_is_injected`'s to judge (its voice gave up).
                Some(&(at, outcome))
                    if outcome != SpeechOutcome::Failed && cancelled.contains(&speech.id) =>
                {
                    Some(at)
                }
                _ => None,
            };
            if let (false, Some(at)) = (played, unheard_cut) {
                let rightly = match cut.get(&speech.id) {
                    Some(InterruptCause::ByAudioActivity) | None => {
                        speaking_at(&onsets, &offsets, created)
                            || person_acted(r, created.at_ms, at)
                            || onset_cancel(r, created.at_ms, at)
                            || client_cancel(r, created.at_ms, at)
                            || host_cut(r, created.at_ms, at)
                            || closing.is_some_and(|c| c <= at)
                    }
                    Some(_) => true,
                };
                check.expect_with(Subject::Speech(speech.id), rightly, || {
                    format!(
                        "created at {} ms, cut unheard at {at} ms ({:?}) with the person silent since",
                        created.at_ms,
                        cut.get(&speech.id)
                    )
                });
                continue;
            }
            let why = (played || ended.is_some())
                .then_some("heard or ended")
                .or_else(|| waits_on(r, speech.id, deadline));
            check.expect_with(Subject::Speech(speech.id), why.is_some(), || {
                format!(
                    "created at {} ms, silent and not done at {deadline} ms with nothing to wait on",
                    created.at_ms
                )
            });
        }
    }
}

/// Whether the person was speaking at `at`, by their last state change at or before it.
fn speaking_at(onsets: &[Observed], offsets: &[Observed], at: Observed) -> bool {
    let on = onsets.iter().filter(|o| **o <= at).max();
    let off = offsets.iter().filter(|o| **o <= at).max();
    on.is_some_and(|on| off.is_none_or(|off| on > off))
}

/// Whether the person, or the host on their behalf, did anything that ends or claims a turn
/// between two instants: spoke into a detector or recogniser (the session's or the realtime
/// server's), typed, pressed a key, or had their turn committed or claimed.
fn person_acted(r: &Recording, from: u64, to: u64) -> bool {
    r.applied().iter().filter(|a| a.accepted && from <= a.at.at_ms && a.at.at_ms <= to).any(|a| {
        matches!(
            a.input,
            Input::VadStart { .. }
                | Input::VadInference { speaking: true, .. }
                | Input::SttStart { .. }
                | Input::SttInterim { .. }
                | Input::SttPreflight { .. }
                | Input::SttFinal { .. }
                | Input::RealtimeEvent { event: RtEvent::SpeechStarted | RtEvent::SpeechStopped, .. }
                | Input::Room { event: RoomIn::Chat }
                | Input::Dtmf { .. }
                | Input::CommitUserTurn { .. }
                | Input::CommitTurn { .. }
                | Input::ClaimUserTurn {}
                | Input::Run { .. }
                | Input::GenerateReply { user_input: Some(_), .. }
                | Input::Console { .. }
        )
    })
}

/// How long before a realtime reply's creation an interrupt may already be aimed at it: the
/// server announces a response after its request, and a cancel sent between hits it.
const ANNOUNCE_MS: u64 = 2_000;

/// Whether the realtime server heard the person start just before a reply was announced: the
/// onset interrupts what is playing, and the cancel that sends (session-wide, as the reference's
/// `interrupt()`) reaches the server after it made the reply, as [`host_cut`]'s interrupt does.
fn onset_cancel(r: &Recording, created: u64, to: u64) -> bool {
    let from = created.saturating_sub(ANNOUNCE_MS);
    r.applied().iter().filter(|a| a.accepted && from <= a.at.at_ms && a.at.at_ms <= to).any(|a| {
        matches!(a.input, Input::RealtimeEvent { event: RtEvent::SpeechStarted, .. })
    })
}

/// Whether the client sent the realtime server a `response.cancel` around a reply's life: one
/// aimed at a request the server had not answered yet names no response, and stops whichever one
/// is active, a reply the server made of its own accord included.
fn client_cancel(r: &Recording, created: u64, to: u64) -> bool {
    let from = created.saturating_sub(ANNOUNCE_MS);
    r.journal().iter().any(|(at, step)| {
        from <= at.at_ms
            && at.at_ms <= to
            && matches!(
                step,
                Step::Realtime(RealtimeStep::Sent { kind: RealtimeCommandKind::CancelReply, .. })
            )
    })
}

/// Whether the host interrupted, or the realtime server dropped or failed, around a speech's
/// life: from just before it was announced to its end.
fn host_cut(r: &Recording, created: u64, to: u64) -> bool {
    let from = created.saturating_sub(ANNOUNCE_MS);
    r.applied().iter().filter(|a| a.accepted && from <= a.at.at_ms && a.at.at_ms <= to).any(|a| {
        matches!(
            a.input,
            Input::Interrupt { .. }
                | Input::InterruptSpeech { .. }
                // Taking the speaker away cuts what plays, and on a realtime socket cancels the
                // server's response in flight, an unasked one included.
                | Input::SetAudioOutput { attach: false, .. }
                | Input::RealtimeEvent {
                    event: RtEvent::Disconnected
                        | RtEvent::ResponseFailed
                        | RtEvent::ServerError
                        | RtEvent::FatalError,
                    ..
                }
        )
    })
}

/// When the person last stopped speaking.
fn last_quiet(journal: &[&(Observed, Step)]) -> Option<u64> {
    journal.iter().rev().find_map(|(a, s)| match s {
        Step::Emitted(SessionEvent::UserStateChanged { old: UserState::Speaking, .. }) => {
            Some(a.at_ms)
        }
        _ => None,
    })
}

/// What a still-silent speech waits on at `deadline`, if anything holds.
fn waits_on(r: &Recording, speech: u64, deadline: u64) -> Option<&'static str> {
    let before = |a: &Observed| a.at_ms <= deadline;
    let open = |kind: Kind| {
        r.opened().iter().filter(|o| o.kind == kind && before(&o.at)).any(|o| {
            !r.closed().iter().any(|(c, k, n)| *k == kind && *n == o.nth && before(c))
        })
    };
    if OWED.iter().any(|k| open(*k)) {
        return Some("owed by the case");
    }
    // A voice asked before the bound whose first audio comes on its own clock after it: the
    // model answered late, and the voice's time to first audio is still running (seed 62764), or
    // the voice stalls past the bound and the speech is cut before its audio comes (seed 12082).
    let voicing = r.opened().iter().filter(|o| o.kind == Kind::Tts && before(&o.at)).any(|o| {
        r.delivered().get(&(Kind::Tts, o.nth)).is_some_and(|at| !before(at))
            || r.first_audio_due().get(&o.nth).is_some_and(|due| *due > deadline)
    });
    if voicing {
        return Some("the voice's first audio");
    }
    let journal: Vec<&(Observed, Step)> = r.journal().iter().filter(|(a, _)| before(a)).collect();
    let last = |pick: &dyn Fn(&Step) -> Option<bool>| journal.iter().filter_map(|(_, s)| pick(s)).last();
    let held = last(&|s| match s {
        Step::Speech(SpeechStep::AuthorizationPaused) => Some(true),
        Step::Speech(SpeechStep::AuthorizationResumed) => Some(false),
        _ => None,
    });
    if held == Some(true) {
        return Some("authorization held");
    }
    // The gap the scheduler holds between speeches (`min_consecutive_speech_delay`).
    let gap = last(&|s| match s {
        Step::Speech(SpeechStep::FloorArmed { .. }) => Some(true),
        Step::Speech(SpeechStep::FloorReleased) => Some(false),
        _ => None,
    });
    if gap == Some(true) {
        return Some("the gap between speeches");
    }
    // An activity draining (a drain, a handoff) schedules nothing new until the next takes over.
    if journal.iter().any(|(a, s)| {
        a.at_ms + BOUND_MS >= deadline && matches!(s, Step::Speech(SpeechStep::SchedulingPaused))
    }) {
        return Some("scheduling paused");
    }
    // The agent's voice on the room's track, with nobody in the room: it plays once someone
    // subscribes (`_subscribed_fut`), as the reference's does.
    let room_track = journal.iter().any(|(_, s)| {
        matches!(s, Step::Room(RoomStep::Event(RoomEventKind::LocalTrackPublished)))
    });
    let present = journal.iter().fold(0i64, |n, (_, s)| match s {
        Step::Room(RoomStep::Event(RoomEventKind::ParticipantConnected)) => n + 1,
        Step::Room(RoomStep::Event(RoomEventKind::ParticipantDisconnected)) => n - 1,
        _ => n,
    });
    // AMD found nobody answered: the callee never joined to subscribe.
    let unanswered = journal.iter().any(|(_, s)| {
        matches!(s, Step::Emitted(SessionEvent::AmdResult { verdict })
            if verdict.reason == lk_agents_abi::voice::amd::AmdReason::ParticipantMissing)
    });
    // The case had no audio output of its own by then: the voice goes to the room's own speaker
    // (the room's legs), which holds it until somebody subscribes, whether or not the room's
    // events reached the journal.
    let room_speaker = !r.opened().iter().any(|o| {
        o.kind == Kind::Output
            && before(&o.at)
            && o.detail == crate::record::Detail::Output { audio: true }
    });
    if (room_track || unanswered || room_speaker) && present <= 0 {
        return Some("nobody in the room to hear it");
    }
    // A provider failing within the bound: the reply waits on retries, or on a connection.
    if journal.iter().any(|(a, s)| {
        a.at_ms + BOUND_MS >= deadline
            && matches!(
                s,
                Step::Emitted(SessionEvent::Error(_))
                    | Step::Generation(GenerationStep::RetryArmed { .. })
            )
    }) {
        return Some("a provider failing");
    }
    // A request the realtime server has not answered, within its own timeout. Keyed: a cancel
    // of a timer already fired or never armed is no answer to another.
    let mut asked = std::collections::BTreeSet::new();
    for (_, s) in &journal {
        match s {
            Step::Core(CoreStep::TimerArmed {
                timer: TimerName::Session(Timer::RealtimeRequest(key)), ..
            }) => {
                asked.insert(*key);
            }
            Step::Core(
                CoreStep::TimerCancelled(TimerName::Session(Timer::RealtimeRequest(key)))
                | CoreStep::TimerFired(TimerName::Session(Timer::RealtimeRequest(key)), _),
            ) => {
                asked.remove(key);
            }
            _ => {}
        }
    }
    if !asked.is_empty() {
        return Some("a realtime request the server has not answered");
    }
    let holds = journal.iter().fold(0i64, |n, (_, s)| match s {
        Step::Speech(SpeechStep::IdleAnswered { hold: Some(_), .. }) => n + 1,
        Step::Speech(SpeechStep::IdleHoldReleased { .. }) => n - 1,
        _ => n,
    });
    if holds > 0 {
        return Some("the floor held by the developer");
    }
    let speaking = last(&|s| match s {
        Step::Emitted(SessionEvent::UserStateChanged { new, .. }) => Some(*new == UserState::Speaking),
        _ => None,
    });
    if speaking == Some(true) {
        return Some("the person speaking");
    }
    // Held at its launch, or paused, for the person the session's own detector hears, until the
    // pause resolves: a realtime server doing turn detection need not have heard them (no user
    // state change).
    // Each cause pauses and resumes on its own: a launch released leaves an interruption's pause.
    let mut gates: Vec<(GateCause, bool)> = Vec::new();
    for (_, s) in &journal {
        if let Step::Interrupt(InterruptStep::PlayoutGate { speech: sp, op, cause }) = s
            && sp.0 == speech
        {
            gates.retain(|(c, _)| c != cause);
            gates.push((*cause, *op == GateOp::Pause));
        }
    }
    let paused = gates.iter().any(|(_, paused)| *paused);
    // The detector's view of the person: speaking, or quiet since when.
    let mut quiet_since: Option<u64> = None;
    let mut vad_speaking = false;
    for a in r.applied().iter().filter(|a| a.accepted && before(&a.at)) {
        let speaking = match a.input {
            Input::VadStart { .. } => true,
            Input::VadInference { speaking, .. } => speaking,
            Input::VadEnd { .. } => false,
            _ => continue,
        };
        if vad_speaking && !speaking {
            quiet_since = Some(a.at.at_ms);
        }
        vad_speaking = speaking;
    }
    // The recogniser's view: an utterance it opened and has not ended. Without a VAD the session
    // never reports the person speaking (the reference takes the recogniser's onset only when it
    // detects turns), yet that utterance is what pauses the speech playing (seed 8893).
    let mut stt_speaking = false;
    for a in r.applied().iter().filter(|a| a.accepted && before(&a.at)) {
        match a.input {
            Input::SttStart { .. } => stt_speaking = true,
            Input::SttEnd { .. } => stt_speaking = false,
            _ => {}
        }
    }
    let allowance = super::pause_resolves::allowance_ms(r.journal(), deadline);
    if paused
        && (vad_speaking
            || stt_speaking
            || quiet_since.is_some_and(|quiet| deadline <= quiet + allowance))
    {
        return Some("paused for the person's speech");
    }
    // The person stopped, and their turn's end is still being decided (seed 99458).
    let endpointing = last(&|s| match s {
        Step::Core(CoreStep::TimerArmed { timer: TimerName::Session(Timer::EndpointingDelay), .. }) => {
            Some(true)
        }
        Step::Core(
            CoreStep::TimerCancelled(TimerName::Session(Timer::EndpointingDelay))
            | CoreStep::TimerFired(TimerName::Session(Timer::EndpointingDelay), _),
        ) => Some(false),
        _ => None,
    });
    if endpointing == Some(true) {
        return Some("the person's turn being decided");
    }
    // The floor's holder, if another speech: waiting behind it is right while it plays, or while
    // it sits paused for the person's speech no longer than `pause_resolves` allows; not after.
    let holder = journal
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Emitted(SessionEvent::FloorChanged { holder }) => Some(holder.map(|h| h.0)),
            _ => None,
        })
        .last()
        .flatten();
    if let Some(other) = holder.filter(|h| *h != speech) {
        let paused = last(&|s| match s {
            Step::Interrupt(InterruptStep::PlayoutGate { speech: sp, op, .. }) if sp.0 == other => {
                Some(*op == GateOp::Pause)
            }
            _ => None,
        });
        if paused != Some(true) {
            return Some("the floor held by a speech that plays");
        }
        let quiet = last_quiet(&journal);
        if stt_speaking || quiet.is_some_and(|q| deadline <= q + allowance) {
            return Some("the floor held by a speech paused for the person");
        }
    }
    None
}

fn floor(holder: u64) -> Step {
    Step::Emitted(SessionEvent::FloorChanged { holder: Some(SpeechId(holder)) })
}

/// A reply that reached the speaker a moment after it was made.
fn played_in_time() -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechKind;
    Authored::new()
        .speech_created(4, SpeechKind::GenerateReply)
        .speech(4, Handle::live())
        .journal(floor(4))
        .journal(Step::Playout(PlayoutStep::SinkCalled {
            speech: Some(SpeechId(4)),
            call: SinkCall::Push { frames: 10 },
        }))
        .later(BOUND_MS * 2)
        .done()
}

/// A reply still waiting on its model at the bound.
fn waited_on_the_model() -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechKind;
    Authored::new()
        .speech_created(4, SpeechKind::GenerateReply)
        .speech(4, Handle::live())
        .chat(4)
        .later(BOUND_MS * 2)
        .done()
}

/// A `say` whose voice stalls past the bound before its first audio, cut before that audio came.
fn waited_on_a_stalled_voice() -> Recording {
    voiced(BOUND_MS * 3)
}

/// The same voice due long before the bound, its audio never handed to the speaker.
fn silent_with_the_voice_due() -> Recording {
    voiced(100)
}

fn voiced(first_audio_ms: u64) -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechKind;
    Authored::new()
        // The case's own speaker, not the room's: nobody need subscribe.
        .output(true)
        .speech_created(4, SpeechKind::Say)
        .speech(4, Handle::live())
        .tts()
        .first_audio_due(0, first_audio_ms)
        .closed(Kind::Tts, 0)
        .later(BOUND_MS * 2)
        .done()
}

/// A reply queued behind a speech paused for the person's voice, the person long quiet.
fn silent_behind_a_paused_speech() -> Recording {
    behind_a_paused_speech(Authored::new())
}

/// The same, the recogniser still hearing the person at the bound: the reply waits on them.
fn waited_behind_the_persons_utterance() -> Recording {
    use crate::input::Ref;
    behind_a_paused_speech(Authored::new().sent(Input::SttStart { stt: Ref::new(0) }))
}

/// The same, the recogniser's utterance ended long before the bound.
fn silent_after_the_persons_utterance() -> Recording {
    use crate::input::Ref;
    let heard = Authored::new()
        .sent(Input::SttStart { stt: Ref::new(0) })
        .sent(Input::SttEnd { utterance: Ref::new(0) });
    behind_a_paused_speech(heard)
}

fn behind_a_paused_speech(case: Authored) -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechKind;
    case
        // The case's own speaker, not the room's: nobody need subscribe.
        .opens(Kind::Output, crate::record::Detail::Output { audio: true })
        .speech_created(3, SpeechKind::Realtime)
        .journal(floor(3))
        .journal(Step::Interrupt(InterruptStep::PlayoutGate {
            speech: SpeechId(3),
            op: GateOp::Pause,
            cause: GateCause::ByAudioActivity,
        }))
        .speech_created(4, SpeechKind::Realtime)
        .speech(4, Handle::live())
        .later(BOUND_MS * 2)
        .done()
}
