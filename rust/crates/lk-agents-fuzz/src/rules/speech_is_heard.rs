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

use lk_agents_abi::voice::events::UserState;
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    agent_session::effects::{GateCause, GateOp, Timer},
    events::{SessionEvent, SpeechOutcome},
    trace::{
        CoreStep, GenerationStep, InterruptCause, InterruptStep, RealtimeStep, PlayoutStep, RoomEventKind, RoomStep, SinkCall,
        SpeechStep, Step, TimerName,
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
        examples: &[played_in_time, waited_on_the_model],
        counterexamples: &[silent_behind_a_paused_speech],
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
        let mut authorized: BTreeMap<u64, u64> = BTreeMap::new();
        // When the person was heard to start speaking.
        let mut cancelled: std::collections::BTreeSet<u64> = Default::default();
        let mut onsets: Vec<u64> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new();
        for (at, step) in journal {
            match step {
                Step::Playout(PlayoutStep::SinkCalled {
                    speech: Some(speech),
                    call: SinkCall::Push { .. },
                }) => {
                    pushed.entry(speech.0).or_insert(at.at_ms);
                }
                Step::Emitted(SessionEvent::SpeechDone { speech, outcome }) => {
                    done.entry(speech.0).or_insert((at.at_ms, *outcome));
                }
                Step::Interrupt(InterruptStep::Interrupted { speech, cause, .. }) => {
                    cut.entry(speech.0).or_insert(*cause);
                }
                Step::Speech(SpeechStep::Authorized { speech }) => {
                    authorized.entry(speech.0).or_insert(at.at_ms);
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
                        onsets.push(at.at_ms);
                    } else {
                        offsets.push(at.at_ms);
                    }
                }
                _ => {}
            }
        }
        for speech in r.host_view().speeches().into_iter().filter(SpeechView::is_known) {
            let Some(created) = speech.created_at() else { continue };
            // A preemptive reply waits on the turn it guessed, which may never end.
            if speech.is_speculative() && !speech.is_scheduled() {
                continue;
            }
            // The bound runs from the creation, and again from when the scheduler let it go: a
            // speech that waited its turn (the floor, the gap, authorization, the person) has the
            // whole bound once it may play.
            let deadline = match authorized.get(&speech.id) {
                Some(a) if *a <= created.at_ms + BOUND_MS => (*a).max(created.at_ms) + BOUND_MS,
                _ => created.at_ms + BOUND_MS,
            };
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
                Some(&(at, _)) if cancelled.contains(&speech.id) => Some(at),
                _ => None,
            };
            if let (false, Some(at)) = (played, unheard_cut) {
                let rightly = match cut.get(&speech.id) {
                    Some(InterruptCause::ByAudioActivity) | None => {
                        speaking_at(&onsets, &offsets, created.at_ms)
                            || person_acted(r, created.at_ms, at)
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
fn speaking_at(onsets: &[u64], offsets: &[u64], at: u64) -> bool {
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

/// Whether the host interrupted, or the realtime server dropped or failed, around a speech's
/// life: from just before it was announced to its end.
fn host_cut(r: &Recording, created: u64, to: u64) -> bool {
    let from = created.saturating_sub(ANNOUNCE_MS);
    r.applied().iter().filter(|a| a.accepted && from <= a.at.at_ms && a.at.at_ms <= to).any(|a| {
        matches!(
            a.input,
            Input::Interrupt { .. }
                | Input::InterruptSpeech { .. }
                | Input::RealtimeEvent {
                    event: RtEvent::Disconnected | RtEvent::ResponseFailed | RtEvent::ServerError,
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
    if (room_track || unanswered) && present <= 0 {
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
    // A request the realtime server has not answered, within its own timeout.
    let rt = |s: &Step| {
        let timer = match s {
            Step::Core(CoreStep::TimerArmed { timer, .. }) => Some((timer, 1)),
            Step::Core(CoreStep::TimerCancelled(timer) | CoreStep::TimerFired(timer, _)) => {
                Some((timer, -1))
            }
            _ => None,
        };
        match timer {
            Some((TimerName::Session(Timer::RealtimeRequest(_)), n)) => n,
            _ => 0,
        }
    };
    if journal.iter().map(|(_, s)| rt(s)).sum::<i64>() > 0 {
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
        let allowance = super::pause_resolves::allowance_ms(r.journal(), deadline);
        if quiet.is_some_and(|q| deadline <= q + allowance) {
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

/// A reply queued behind a speech paused for the person's voice, the person long quiet.
fn silent_behind_a_paused_speech() -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechKind;
    Authored::new()
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
