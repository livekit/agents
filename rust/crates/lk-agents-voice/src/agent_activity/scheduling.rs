//! `_scheduling_task` and its queue: the [`SpeechScheduler`], the speech vocabulary, and the
//! driver's scheduling, authorization, pause and `_current_speech` queries.

use crate::agent_session::Driver;
use crate::agent_session::effects::{Out, PauseResult, Signal, Timer};
use crate::events::SessionEvent;
use crate::speech_handle::{SpeechState, SpeechUpdate};
use crate::trace::{CloseAct, CloseStep, InterruptCause, InterruptStep, SpeechStep, TurnStep};
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::AgentState;
use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

const HANDOFF: &str = "a handoff resumed scheduling before the drain completed";

const CLOSING: &str = "the session is closing";

use speech_scheduler::{NextSpeech, Requeue, SchedulingPass};
pub use speech_scheduler::{SpeechScheduler, Status};

/// The scheduling state: the speeches, the queue and the step keep their invariants behind
/// methods; the driver reaches only the plain switches.
mod speech_scheduler {
    use super::*;

    /// `_scheduling_paused`: new speech refused (tool replies excepted), queued speech still plays,
    /// every caller answered when nothing is outstanding.
    #[derive(Debug)]
    struct Pause {
        cause: PauseCause,
        /// Every cause asked for it, answered together.
        callers: Vec<PauseCause>,
    }

    /// Where `_scheduling_task`'s loop is: waiting for `_speech_q`, sleeping out
    /// `min_consecutive_speech_delay`, waiting for `_authorization_allowed`, or with a
    /// `_current_speech`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SchedulingStep {
        Idle,
        /// `min_consecutive_speech_delay` still owed, until [`Timer::ConsecutiveSpeechDelay`]
        /// fires. `next` is `None` once that candidate finished early; the gap still stands.
        Delaying {
            next: Option<SpeechId>,
        },
        /// The head is chosen but authorization is held (AMD's `_authorization_paused`).
        AuthorizationPaused {
            next: SpeechId,
        },
        /// Authorized (`_current_speech`): playing, or inside a tool.
        Current(SpeechId),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Status {
        /// On the queue, or popped as the next speech.
        Queued,
        /// A live preemptive generation: asking the model, not on the queue, waiting for its turn
        /// to commit.
        Preemptive,
        Current,
        /// Step spoken, tools running: neither current nor queued until requeued or finished.
        Working,
        /// Rejected or discarded: the driver interrupted it and it is winding down to its end.
        Discarded,
    }

    #[derive(Debug, Clone)]
    struct Entry {
        /// The speech's state, from scheduling until release: the queue is where the loop holds
        /// it.
        state: SpeechState,
        /// What it was created as. Provenance: an adopted preemptive generation stays a `Preemptive`
        /// here.
        kind: SpeechKind,
        add_to_chat_ctx: bool,
        status: Status,
    }

    /// `AgentActivity`'s scheduling state (`_speech_q`, `_current_speech`, `_scheduling_paused`,
    /// `_authorization_paused`, `_preemptive_generation`), and every live speech's state. Part of
    /// the [`AgentActivity`](crate::agent_activity::AgentActivity).
    #[derive(Debug)]
    pub struct SpeechScheduler {
        speeches: BTreeMap<SpeechId, Entry>,
        /// `_speech_q`. Every speech has `SPEECH_PRIORITY_NORMAL` (nothing calls `say(priority=)`),
        /// so the reference's priority queue is a queue.
        speech_q: VecDeque<SpeechId>,
        step: SchedulingStep,
        /// `None` until a speech has played out. Not `Ts::ZERO`, which would delay the first
        /// speech.
        last_playout: Option<Ts>,
        paused: Option<Pause>,
        /// AMD's `_authorization_paused`: the head is chosen and held unauthorized. A flag, not a
        /// counter: two pauses and one resume resume.
        pub(crate) authorization_paused: bool,
        /// At most one live preemptive generation, with the transcript the model was asked about.
        preemptive_generation: Option<(SpeechId, String)>,
        /// The preflight a preemptive generation is started for, consumed by the `schedule_speech`
        /// that creates it.
        pub(crate) preflight: Option<String>,
        preemptive_generation_count: u32,
        /// What released speeches published last, then their release, until taken.
        released: Vec<(SpeechId, SpeechUpdate)>,
    }

    impl Default for SpeechScheduler {
        fn default() -> Self {
            Self::new()
        }
    }

    impl SpeechScheduler {
        pub fn new() -> Self {
            SpeechScheduler {
                speeches: BTreeMap::new(),
                speech_q: VecDeque::new(),
                step: SchedulingStep::Idle,
                last_playout: None,
                paused: None,
                authorization_paused: false,
                preemptive_generation: None,
                preflight: None,
                preemptive_generation_count: 0,
                released: Vec::new(),
            }
        }

        /// `_current_speech`.
        pub fn current_speech(&self) -> Option<SpeechId> {
            match self.step {
                SchedulingStep::Current(speech) => Some(speech),
                SchedulingStep::Idle
                | SchedulingStep::Delaying { .. }
                | SchedulingStep::AuthorizationPaused { .. } => None,
            }
        }

        fn entry(&self, speech: SpeechId) -> Option<&Entry> {
            self.speeches.get(&speech)
        }

        /// What a speech the queue holds was created as (an adopted preemptive generation stays a
        /// `Preemptive`).
        pub fn kind(&self, speech: SpeechId) -> Option<SpeechKind> {
            self.entry(speech).map(|e| e.kind)
        }

        /// Where a speech the queue holds is in its life.
        pub fn status(&self, speech: SpeechId) -> Option<Status> {
            self.entry(speech).map(|e| e.status)
        }

        /// The state of a speech the queue holds (from scheduling until its release).
        pub fn speech(&self, speech: SpeechId) -> Option<&SpeechState> {
            self.speeches.get(&speech).map(|e| &e.state)
        }

        pub fn speech_mut(&mut self, speech: SpeechId) -> Option<&mut SpeechState> {
            self.speeches.get_mut(&speech).map(|e| &mut e.state)
        }

        /// What every speech published since the last call, released ones first.
        pub fn take_updates(&mut self) -> Vec<(SpeechId, SpeechUpdate)> {
            let mut updates = std::mem::take(&mut self.released);
            for (id, entry) in &mut self.speeches {
                updates.extend(entry.state.take_updates().into_iter().map(|u| (*id, u)));
            }
            updates
        }

        pub fn contains(&self, speech: SpeechId) -> bool {
            self.speeches.contains_key(&speech)
        }

        /// No speech is held, live or winding down.
        pub fn is_empty(&self) -> bool {
            self.speeches.is_empty()
        }

        /// Every speech held, oldest first.
        pub fn speech_ids(&self) -> impl DoubleEndedIterator<Item = SpeechId> + '_ {
            self.speeches.keys().copied()
        }

        /// Every speech held with its state, oldest first.
        pub fn speeches(&self) -> impl DoubleEndedIterator<Item = (SpeechId, &SpeechState)> {
            self.speeches.iter().map(|(id, e)| (*id, &e.state))
        }

        pub fn preemptive_generation(&self) -> Option<SpeechId> {
            self.preemptive_generation.as_ref().map(|(speech, _)| *speech)
        }

        /// `say(add_to_chat_ctx=..)` for a speech still held; `true` for one already released.
        pub fn add_to_chat_ctx(&self, speech: SpeechId) -> bool {
            self.speeches.get(&speech).is_none_or(|e| e.add_to_chat_ctx)
        }

        /// What is left of `min_delay` (`Duration::ZERO` disables the gap) since the last playout.
        fn consecutive_speech_delay(&self, now: Ts, min_delay: Duration) -> Option<Duration> {
            if min_delay.is_zero() {
                return None;
            }
            let elapsed = now.saturating_duration_since(self.last_playout?);
            min_delay.checked_sub(elapsed).filter(|d| !d.is_zero())
        }

        pub fn is_paused(&self) -> bool {
            self.paused.is_some()
        }

        pub fn preemptive_generation_count(&self) -> u32 {
            self.preemptive_generation_count
        }

        /// The current speech has not been interrupted.
        pub fn current_speech_uninterrupted(&self) -> bool {
            let current = self.current_speech().and_then(|c| self.entry(c));
            current.is_some_and(|e| !e.state.interrupted())
        }

        /// No current speech and none chosen next: at most the gap is owed.
        pub fn no_current_speech(&self) -> bool {
            match self.step {
                SchedulingStep::Idle | SchedulingStep::Delaying { next: None, .. } => true,
                SchedulingStep::Delaying { next: Some(_), .. }
                | SchedulingStep::AuthorizationPaused { .. }
                | SchedulingStep::Current(_) => false,
            }
        }

        /// The next speech, chosen but not yet current (AMD's hold or the consecutive-speech gap).
        pub fn next_speech(&self) -> Option<SpeechId> {
            match self.step {
                SchedulingStep::AuthorizationPaused { next }
                | SchedulingStep::Delaying { next: Some(next), .. } => Some(next),
                SchedulingStep::Idle
                | SchedulingStep::Delaying { next: None, .. }
                | SchedulingStep::Current(_) => None,
            }
        }

        /// `_speech_q` in pop order.
        pub fn speech_q(&self) -> &VecDeque<SpeechId> {
            &self.speech_q
        }

        /// The speech's entry, holding its state from here to its release.
        pub(super) fn admit(
            &mut self,
            speech: SpeechId,
            state: SpeechState,
            kind: SpeechKind,
            add_to_chat_ctx: bool,
            status: Status,
        ) {
            let entry = Entry { state, kind, add_to_chat_ctx, status };
            self.speeches.insert(speech, entry);
        }

        pub(super) fn hold_preemptive_generation(&mut self, speech: SpeechId, preflight: String) {
            self.preemptive_generation = Some((speech, preflight));
            self.preemptive_generation_count += 1;
        }

        pub(super) fn enqueue(&mut self, speech: SpeechId) {
            self.speech_q.push_back(speech);
        }

        /// `_scheduling_task`'s pass: the owed gap expires where it is read, and the head is
        /// authorized, armed behind the gap, or held for AMD.
        /// `min_delay` is the effective `min_consecutive_speech_delay` (the driving agent's, else
        /// the session's), read at each pass.
        pub(super) fn scheduling_pass(
            &mut self,
            now: Ts,
            gap_elapsed: bool,
            min_delay: Duration,
        ) -> SchedulingPass {
            let mut released = false;
            let candidate = loop {
                let candidate = match self.step {
                    SchedulingStep::Current(_) => {
                        return SchedulingPass { released, next: NextSpeech::Stay };
                    }
                    SchedulingStep::Delaying { .. } if !gap_elapsed => {
                        return SchedulingPass { released, next: NextSpeech::Stay };
                    }
                    // A candidate that finished during the gap: the queue is looked at.
                    SchedulingStep::Delaying { next, .. } => {
                        released = true;
                        let Some(next) = next else {
                            self.step = SchedulingStep::Idle;
                            continue;
                        };
                        next
                    }
                    SchedulingStep::AuthorizationPaused { next } => next,
                    SchedulingStep::Idle => {
                        let Some(speech) = self.speech_q.pop_front() else {
                            return SchedulingPass { released, next: NextSpeech::Stay };
                        };
                        speech
                    }
                };
                let live = self.speeches.get(&candidate);
                if live.is_some_and(|e| !e.state.interrupted() && e.status == Status::Queued) {
                    break candidate;
                }
                // Gone, interrupted or discarded while it waited: forget it and look again.
                self.step = SchedulingStep::Idle;
            };
            if let Some(delay) = self.consecutive_speech_delay(now, min_delay) {
                let until = now.saturating_add(delay);
                self.step = SchedulingStep::Delaying { next: Some(candidate) };
                return SchedulingPass { released, next: NextSpeech::Delayed { until } };
            }
            // AMD's hold keeps the head chosen and unauthorized.
            if self.authorization_paused {
                self.step = SchedulingStep::AuthorizationPaused { next: candidate };
                return SchedulingPass { released, next: NextSpeech::Stay };
            }
            self.step = SchedulingStep::Current(candidate);
            self.set_status(candidate, Status::Current);
            SchedulingPass { released, next: NextSpeech::Authorized(candidate) }
        }

        /// A `Working` speech's next step: back on the queue; refused while closing.
        pub(super) fn requeue(&mut self, speech: SpeechId, closing: bool) -> Requeue {
            let Some(entry) =
                self.speeches.get_mut(&speech).filter(|e| e.status == Status::Working)
            else {
                return Requeue::NotWorking;
            };
            if closing {
                entry.status = Status::Discarded;
                return Requeue::Refused;
            }
            entry.status = Status::Queued;
            self.speech_q.push_back(speech);
            Requeue::Requeued
        }

        /// The speech is over: its entry (and state) go, the current speech and its gap move.
        /// `Some(quiet)`: no current speech and nothing in its tools (`listening`).
        pub(super) fn finish(&mut self, speech: SpeechId, now: Ts) -> Option<bool> {
            let mut entry = self.speeches.remove(&speech)?;
            let last = entry.state.take_updates().into_iter().chain([SpeechUpdate::Released]);
            self.released.extend(last.map(|u| (speech, u)));
            // A preemptive generation that wound down on its own before the turn committed.
            if self.preemptive_generation.as_ref().is_some_and(|(held, _)| *held == speech) {
                self.preemptive_generation = None;
            }
            self.step = match self.step {
                // The gap measures *end of playout* to start of the next.
                SchedulingStep::Current(current) if current == speech => {
                    self.last_playout = Some(now);
                    SchedulingStep::Idle
                }
                // The gap owed to a finished speech stands for whoever is behind.
                SchedulingStep::Delaying { next: Some(next) } if next == speech => {
                    SchedulingStep::Delaying { next: None }
                }
                SchedulingStep::AuthorizationPaused { next } if next == speech => {
                    SchedulingStep::Idle
                }
                step => step,
            };
            let working = self.speeches.values().any(|e| e.status == Status::Working);
            Some(
                !working
                    && matches!(
                        self.step,
                        SchedulingStep::Idle | SchedulingStep::Delaying { next: None, .. }
                    ),
            )
        }

        /// `_pause_scheduling_task`: one more promise; the close's drain outranks the pause it
        /// joins.
        pub(super) fn pause(&mut self, cause: PauseCause) {
            match &mut self.paused {
                Some(pause) => {
                    pause.callers.push(cause);
                    if cause == PauseCause::Close {
                        pause.cause = PauseCause::Close;
                    }
                }
                None => self.paused = Some(Pause { cause, callers: vec![cause] }),
            }
        }

        /// The pause a handoff or a host drain caused, lifted: its callers.
        pub(super) fn resume_activity_pause(&mut self) -> Option<Vec<PauseCause>> {
            let caused = |p: &mut Pause| matches!(p.cause, PauseCause::Handoff | PauseCause::Drain);
            self.paused.take_if(caused).map(|p| p.callers)
        }

        pub(super) fn take_pause(&mut self) -> Option<Vec<PauseCause>> {
            self.paused.take().map(|p| p.callers)
        }

        /// The close: an owed or held next speech freed (`true` when an owed gap was released).
        pub(super) fn close(&mut self) -> bool {
            match self.step {
                SchedulingStep::Delaying { .. } => {
                    self.step = SchedulingStep::Idle;
                    true
                }
                SchedulingStep::AuthorizationPaused { .. } => {
                    self.step = SchedulingStep::Idle;
                    false
                }
                SchedulingStep::Idle | SchedulingStep::Current(_) => false,
            }
        }

        /// The pause's callers, once no speech is held.
        pub(super) fn settle_pause(&mut self) -> Vec<PauseCause> {
            let Some(pause) = &mut self.paused else { return Vec::new() };
            if !self.speeches.is_empty() {
                return Vec::new();
            }
            std::mem::take(&mut pause.callers)
        }

        /// `_cancel_preemptive_generation`: the preflight goes; the live preemptive generation, if
        /// it still has an entry, is discarded and returned for its interruption.
        pub(super) fn cancel_preemptive_generation(&mut self) -> Option<SpeechId> {
            self.preflight = None;
            let (speech, _) = self.preemptive_generation.take()?;
            self.entry(speech)?;
            self.set_status(speech, Status::Discarded);
            Some(speech)
        }

        /// The turn committed `committed`: the preemptive generation asked about those words is
        /// queued as its reply, any other discarded. The per-turn retry count starts over.
        pub(super) fn adopt_preemptive_generation(
            &mut self,
            committed: &str,
        ) -> Option<(SpeechId, PreemptiveVerdict)> {
            (self.preemptive_generation_count, self.preflight) = (0, None);
            let (speech, preflight) = self.preemptive_generation.take()?;
            let interrupted = self.entry(speech)?.state.interrupted();
            let verdict = reconcile_preemptive_generation(&preflight, committed, interrupted);
            match verdict {
                PreemptiveVerdict::Reuse => {
                    self.set_status(speech, Status::Queued);
                    self.speech_q.push_back(speech);
                }
                PreemptiveVerdict::Discard(_) => self.set_status(speech, Status::Discarded),
            }
            Some((speech, verdict))
        }

        /// The step was spoken: the current speech is released and becomes `Working` (requeueing
        /// is the way back). No `listening`: tools may still run.
        pub(in crate::agent_activity) fn spoken(&mut self, speech: SpeechId, now: Ts) {
            if self.step != SchedulingStep::Current(speech) {
                return;
            }
            self.step = SchedulingStep::Idle;
            self.last_playout = Some(now);
            self.set_status(speech, Status::Working);
        }

        fn set_status(&mut self, speech: SpeechId, status: Status) {
            if let Some(entry) = self.speeches.get_mut(&speech) {
                entry.status = status;
            }
        }

        /// Debug builds only, at the end of each pass: the state is consistent between wakes.
        pub fn check_invariants(&self, closing: bool) {
            let status = |id: &SpeechId| self.speeches.get(id).map(|e| e.status);
            let preemptive = self.speeches.iter().filter(|(_, e)| e.status == Status::Preemptive);
            debug_assert!(
                preemptive.map(|(id, _)| *id).eq(self.preemptive_generation()),
                "the preemptive generation slot and the Preemptive entries disagree"
            );
            match self.step {
                SchedulingStep::Current(speech) => debug_assert_eq!(
                    status(&speech),
                    Some(Status::Current),
                    "the current speech is not current"
                ),
                SchedulingStep::Delaying { next: Some(next), .. }
                | SchedulingStep::AuthorizationPaused { next } => debug_assert_eq!(
                    status(&next),
                    Some(Status::Queued),
                    "the next speech {next:?} was released or is not queued"
                ),
                SchedulingStep::Delaying { next: None, .. } | SchedulingStep::Idle => {}
            }
            let current = self.speeches.values().filter(|e| e.status == Status::Current).count();
            debug_assert!(current <= 1, "two speeches are current");
            // The queue holds queued speeches (and stale ids of released ones).
            for id in &self.speech_q {
                let s = status(id);
                debug_assert!(s.is_none_or(|s| s == Status::Queued), "{id:?} is queued as {s:?}");
            }
            if closing {
                debug_assert!(
                    self.preemptive_generation.is_none(),
                    "a preemptive generation outlived Closing"
                );
                debug_assert!(self.paused.is_none(), "a pause outlived Closing");
            }
            if self.paused.is_some() {
                debug_assert!(
                    self.preemptive_generation.is_none(),
                    "a preemptive generation outlived a pause"
                );
            }
        }
    }

    /// What a scheduling pass did: an owed gap expired (`FloorReleased`), then what the next
    /// speech is.
    pub(super) struct SchedulingPass {
        pub(super) released: bool,
        pub(super) next: NextSpeech,
    }

    pub(super) enum NextSpeech {
        /// A current speech, a gap owed, nothing queued, or held for AMD: nothing to authorize.
        Stay,
        /// The gap is owed until `until`.
        Delayed {
            until: Ts,
        },
        Authorized(SpeechId),
    }

    pub(super) enum Requeue {
        NotWorking,
        /// Closing: discarded, to be refused.
        Refused,
        Requeued,
    }
}

/// Not `SpeechHandle::id`, the Python-visible string the pure tier cannot mint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpeechId(pub u64);

/// Priority and `user_initiated` derive from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum SpeechKind {
    TurnReply,
    Say,
    GenerateReply,
    /// A speculative reply on a preflight transcript, **not queued** until the turn adopts it.
    Preemptive,
    Realtime,
}

impl SpeechKind {
    /// `SpeechCreatedEvent.source`: only `say()` is `say`; everything else is `generate_reply`.
    pub const fn as_source_str(self) -> &'static str {
        match self {
            SpeechKind::Say => "say",
            SpeechKind::TurnReply
            | SpeechKind::GenerateReply
            | SpeechKind::Preemptive
            | SpeechKind::Realtime => "generate_reply",
        }
    }

    /// `speech_created(user_initiated=..)`: the host asked for this one by name (`say`,
    /// `generate_reply`), so an open run owns it.
    pub const fn user_initiated(self) -> bool {
        match self {
            SpeechKind::Say | SpeechKind::GenerateReply => true,
            SpeechKind::TurnReply | SpeechKind::Preemptive | SpeechKind::Realtime => false,
        }
    }

    /// A preemptive generation must not be authorised before the turn commits.
    pub const fn scheduled(self) -> bool {
        match self {
            SpeechKind::Preemptive => false,
            SpeechKind::TurnReply
            | SpeechKind::Say
            | SpeechKind::GenerateReply
            | SpeechKind::Realtime => true,
        }
    }
}

/// Either way the speech is force-interrupted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error, lk_agents_abi::Vocabulary)]
pub enum Rejected {
    /// Scheduling is paused (draining). The driver raises the reference's `RuntimeError` verbatim.
    #[error(
        "cannot schedule new speech, the speech scheduling is draining/pausing, the speech will be cancelled"
    )]
    Draining,
    /// The session is closing. The reference logs a warning and returns normally.
    #[error(
        "attempting to schedule a new SpeechHandle, but the speech_scheduler is not running, the speech will be cancelled"
    )]
    Stopped,
}

/// A handoff lifts only its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseCause {
    Handoff,
    /// The host's `drain()`: nothing new is scheduled, lifted by the agent that replaces it.
    Drain,
    /// Closing: speeches in flight finish (tools included); the hard edge follows the settle.
    Close,
}

/// Why no preemptive generation is started, so a trace names the guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum PreemptiveRefusal {
    Disabled,
    Realtime,
    SchedulingPaused,
    AgentSpeaking,
    RetriesExhausted,
    TooLong,
    ManualTurn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum PreemptiveDiscard {
    /// The committed words are not the words the model was asked about.
    WordDelta,
    Interrupted,
    /// A newer preflight asked again; the older preemptive generation is not raced, it is replaced.
    Superseded,
    /// The turn the preemptive generation was for went away without a reply, or the preemptive
    /// generation's request failed.
    TurnAbandoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptiveVerdict {
    /// The model was asked the question the turn ended up committing: queue the parked reply.
    Reuse,
    Discard(PreemptiveDiscard),
}

/// Word-level equivalence: case folded, punctuation ignored, whitespace normalised.
pub fn same_words(a: &str, b: &str) -> bool {
    words(a).eq(words(b))
}

fn words(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
}

/// Interrupted wins over the words.
pub fn reconcile_preemptive_generation(
    preflight: &str,
    committed: &str,
    interrupted: bool,
) -> PreemptiveVerdict {
    if interrupted {
        return PreemptiveVerdict::Discard(PreemptiveDiscard::Interrupted);
    }
    if !same_words(preflight, committed) {
        return PreemptiveVerdict::Discard(PreemptiveDiscard::WordDelta);
    }
    PreemptiveVerdict::Reuse
}

impl Driver {
    /// `_schedule_speech`. Every speech is born here: announced (`SpeechCreated`), recorded
    /// (`ReplyCreated`), then queued, held as the preemptive generation, or refused: journalled,
    /// interrupted (forced) and `false`.
    pub(crate) fn schedule_speech(
        &mut self,
        now: Ts,
        speech: SpeechId,
        state: SpeechState,
        kind: SpeechKind,
        add_to_chat_ctx: bool,
        host_asked: bool,
        out: &mut Out,
    ) -> bool {
        // The reference announces every `say` and `_generate_reply` speech (turn replies and
        // preemptive generations included) as user-initiated; only a generation the realtime
        // server started on its own is not.
        let user_initiated = kind != SpeechKind::Realtime || host_asked;
        out.machine().emit(SessionEvent::SpeechCreated { speech, kind, user_initiated });
        out.trace(SpeechStep::ReplyCreated { speech });
        // A preemptive generation's preflight is taken before the previous preemptive generation is
        // discarded, which clears it.
        let preflight = if kind == SpeechKind::Preemptive {
            self.activity.speech_scheduler.preflight.take()
        } else {
            None
        };
        // Closing first, so a forced request into a closing session is still `Stopped`. `Closing`
        // and a pause discard the preemptive generation, so a preemptive generation behind them is
        // refused too.
        let rejected =
            match (self.session.closing_past_edge(), self.activity.speech_scheduler.is_paused()) {
                (true, _) => Some(Rejected::Stopped),
                (false, true) => Some(Rejected::Draining),
                (false, false) => None,
            };
        let status = match (rejected, kind) {
            (Some(_), _) => Status::Discarded,
            (None, SpeechKind::Preemptive) => Status::Preemptive,
            (None, _) => Status::Queued,
        };
        if status == Status::Preemptive {
            // Asking again supersedes the previous preemptive generation rather than racing it.
            self.cancel_preemptive_generation(now, PreemptiveDiscard::Superseded, out);
        }
        let s = &mut self.activity.speech_scheduler;
        s.admit(speech, state, kind, add_to_chat_ctx, status);
        match rejected {
            Some(reason) => self.reject_speech(now, speech, reason, out),
            None if status == Status::Preemptive => {
                s.hold_preemptive_generation(speech, preflight.unwrap_or_default())
            }
            None => s.enqueue(speech),
        }
        rejected.is_none()
    }

    /// A refused speech is still interrupted (forced), or its eager task parks in
    /// `wait_for_authorization` forever: `Closing` past the close, else `ScheduleRejected`.
    fn reject_speech(&mut self, now: Ts, speech: SpeechId, reason: Rejected, out: &mut Out) {
        out.trace(SpeechStep::ScheduleRejected { speech, reason });
        let cause = match reason {
            Rejected::Stopped => InterruptCause::Closing,
            Rejected::Draining => InterruptCause::ScheduleRejected,
        };
        let _ = self.interrupt_speech(now, speech, cause, true, out);
        // The server keeps generating a cancelled realtime turn until told.
        if self.activity.rt_session.is_some() {
            self.realtime_interrupt(cause, out);
        }
        // Born past the close's edge: released here, since the loop may stop first.
        if reason == Rejected::Stopped {
            out.trace(CloseStep::Performed(CloseAct::Released { speech }));
            self.release_speech(now, speech, out);
        }
    }

    /// One `_scheduling_task` pass per cycle, so `Authorized` lands next to the steps that led to
    /// it; then the turn machine's `PreemptiveGeneration` asks.
    pub(crate) fn pump_schedule(&mut self, now: Ts, out: &mut Out) {
        self.run_scheduling_pass(now, out);
        self.activity.speech_scheduler.check_invariants(self.session.closing_past_edge());
    }

    /// `_scheduling_task`: due interruption watchdogs, then the current speech. Its timers fire
    /// here, wherever the pass runs, as `_scheduling_task` wakes between the loop's callbacks: a
    /// watchdog due in the instant another timer fires goes off in that timer's pass.
    fn run_scheduling_pass(&mut self, now: Ts, out: &mut Out) {
        let watchdog = |t| matches!(t, Timer::InterruptionTimeout(_));
        while let Some(Timer::InterruptionTimeout(speech)) = self.pop_due(now, out, watchdog) {
            out.trace(SpeechStep::WatchdogCleared { speech });
            // `SpeechHandle._cancel`: cancel, mark done and release, since the aborted task will
            // never report `Done`.
            if let Some(state) = self.activity.speech_scheduler.speech_mut(speech)
                && state.take_interruption_timeout()
            {
                state.mark_done(None);
                out.machine().push(crate::agent_session::effects::Effect::CancelSpeech(speech));
                out.trace(InterruptStep::InterruptionTimedOut { speech });
                if self.session.closing.is_none() {
                    self.end_live_tools(now, speech, out);
                }
                self.release_speech(now, speech, out);
            }
        }
        if self.session.closing_past_edge() {
            return;
        }
        let gap_elapsed = self.pop_due(now, out, |t| t == Timer::ConsecutiveSpeechDelay).is_some();
        let min_delay = self.session.config.effective.speech.min_consecutive_speech_delay();
        let SchedulingPass { released, next } =
            self.activity.speech_scheduler.scheduling_pass(now, gap_elapsed, min_delay);
        if released {
            out.trace(SpeechStep::FloorReleased);
        }
        let candidate = match next {
            NextSpeech::Stay => return,
            NextSpeech::Delayed { until } => {
                out.machine().arm(Timer::ConsecutiveSpeechDelay, until);
                return out.trace(SpeechStep::FloorArmed { until });
            }
            NextSpeech::Authorized(candidate) => candidate,
        };
        let Some(state) = self.activity.speech_scheduler.speech_mut(candidate) else {
            return;
        };
        state.authorize_generation();
        // Whoever was paused has lost the floor to this speech, or is this speech: either way the
        // sink plays again. A pause left on a speech that ended under it would hold every later
        // speech's audio until the close.
        if let Some(paused) = self.activity.paused_speech.paused_handle() {
            self.release_paused_speech(paused, out);
        }
        out.trace(SpeechStep::Authorized { speech: candidate });
        self.activity.launching.insert(candidate);
        self.realtime_authorized(now, candidate, out);
    }

    /// The reply's tools are done and it has a next step: a `Working` speech goes back on the
    /// queue behind whatever was queued since (any other is left alone); closing refuses it.
    pub(super) fn requeue_speech(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let closing = self.session.closing_past_edge();
        match self.activity.speech_scheduler.requeue(speech, closing) {
            Requeue::NotWorking => {}
            Requeue::Refused => self.reject_speech(now, speech, Rejected::Stopped, out),
            // The step boundary ends the interruption's watchdog too.
            Requeue::Requeued => {
                if self.disarm(Timer::InterruptionTimeout(speech), out) {
                    out.trace(SpeechStep::WatchdogCleared { speech });
                }
            }
        }
    }

    /// The speech is over (`Done`, `Failed` or the watchdog): the current speech and its gap, then
    /// `_on_pipeline_reply_done`'s `listening` when nothing speaks and nothing is in its tools.
    pub(super) fn speech_finished(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let Some(quiet) = self.activity.speech_scheduler.finish(speech, now) else { return };
        if self.disarm(Timer::InterruptionTimeout(speech), out) {
            out.trace(SpeechStep::WatchdogCleared { speech });
        }
        // A self-replying realtime model is still composing the reply its tools asked for: the
        // agent stays `thinking`.
        if quiet && !self.activity.realtime_state.auto_reply_expected() {
            self.set_agent_state(now, AgentState::Listening, out);
        }
        self.settle_scheduling_pause(out);
        if self.activity.speech_scheduler.is_empty() {
            out.signal(Signal::Quiet);
        }
    }

    /// `_pause_scheduling_task`: new speech is refused (tool replies excepted) and queued speech
    /// still plays; its caller is answered once nothing outstanding remains. Two pauses before the
    /// drain completes are two promises; a closing session refuses (drops) it.
    pub(crate) fn pause_scheduling(&mut self, now: Ts, cause: PauseCause, out: &mut Out) {
        out.trace(SpeechStep::SchedulingPaused);
        if self.session.closing_past_edge() {
            return refuse_pause(vec![cause], CLOSING, out);
        }
        self.activity.speech_scheduler.pause(cause);
        self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
        self.settle_scheduling_pause(out);
    }

    /// `_resume_scheduling_task` at the handoff's swap: a preemptive generation for the old agent
    /// is discarded, and only the pause the handoff (or the host's drain) caused is lifted.
    pub(crate) fn resume_scheduling_task(&mut self, now: Ts, out: &mut Out) {
        self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
        if let Some(callers) = self.activity.speech_scheduler.resume_activity_pause() {
            refuse_pause(callers, HANDOFF, out);
        }
    }

    /// The close: watchdogs go with their speeches, the current speech is released, a pending
    /// preemptive generation discarded, a drain-waiting pause refused, and nothing is authorized
    /// again.
    pub(crate) fn close_scheduling(&mut self, now: Ts, out: &mut Out) {
        let speeches: Vec<SpeechId> = self.activity.speech_scheduler.speech_ids().collect();
        for speech in speeches {
            if self.disarm(Timer::InterruptionTimeout(speech), out) {
                out.trace(SpeechStep::WatchdogCleared { speech });
            }
        }
        if self.activity.speech_scheduler.close() {
            self.disarm(Timer::ConsecutiveSpeechDelay, out);
            out.trace(SpeechStep::FloorReleased);
        }
        self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, out);
        if let Some(callers) = self.activity.speech_scheduler.take_pause() {
            refuse_pause(callers, CLOSING, out);
        }
    }

    /// Answer every pending pause once nothing it waits on is outstanding (released, tools
    /// included).
    pub(super) fn settle_scheduling_pause(&mut self, out: &mut Out) {
        for cause in self.activity.speech_scheduler.settle_pause() {
            out.signal(Signal::Paused(cause, PauseResult::Settled));
            out.trace(SpeechStep::SchedulingPauseSettled);
        }
    }

    /// `_cancel_preemptive_generation`: a live preemptive generation is discarded.
    pub(crate) fn cancel_preemptive_generation(
        &mut self,
        now: Ts,
        reason: PreemptiveDiscard,
        out: &mut Out,
    ) {
        if let Some(speech) = self.activity.speech_scheduler.cancel_preemptive_generation() {
            self.discard_preemptive_generation(now, speech, reason, out);
        }
    }

    /// The turn committed: a preemptive generation asked about the committed words is its reply and
    /// is queued; any other is discarded. The per-turn retry count starts over.
    pub(super) fn adopt_preemptive_generation(
        &mut self,
        now: Ts,
        committed: &str,
        out: &mut Out,
    ) -> Option<SpeechId> {
        let (speech, verdict) =
            self.activity.speech_scheduler.adopt_preemptive_generation(committed)?;
        match verdict {
            // Adopted by construction: a pause and `Closing` discard the preemptive generation
            // first.
            PreemptiveVerdict::Reuse => {
                if let Some(state) = self.activity.speech_scheduler.speech_mut(speech) {
                    state.mark_scheduled();
                }
                out.trace(TurnStep::PreemptiveReconciled { speech, verdict });
                Some(speech)
            }
            PreemptiveVerdict::Discard(reason) => {
                self.discard_preemptive_generation(now, speech, reason, out);
                None
            }
        }
    }

    /// A superseded preemptive generation must be interrupted, or it parks forever holding its LLM
    /// stream.
    fn discard_preemptive_generation(
        &mut self,
        now: Ts,
        speech: SpeechId,
        reason: PreemptiveDiscard,
        out: &mut Out,
    ) {
        let verdict = PreemptiveVerdict::Discard(reason);
        out.trace(TurnStep::PreemptiveReconciled { speech, verdict });
        let _ = self.interrupt_speech(now, speech, InterruptCause::Superseded, true, out);
    }

    /// `_current_speech` and its state.
    pub(crate) fn current_speech(&self) -> Option<(SpeechId, &SpeechState)> {
        let scheduling = &self.activity.speech_scheduler;
        scheduling.current_speech().and_then(|id| Some((id, scheduling.speech(id)?)))
    }

    pub(crate) fn is_realtime_speech(&self, speech: SpeechId) -> bool {
        self.activity.speech_scheduler.kind(speech) == Some(SpeechKind::Realtime)
    }
}

/// A pause that can no longer complete truthfully: its callers are refused and the journal has
/// the reason.
fn refuse_pause(callers: Vec<PauseCause>, cause: &'static str, out: &mut Out) {
    for caller in callers {
        out.signal(Signal::Paused(caller, PauseResult::Refused));
        out.trace(SpeechStep::SchedulingPauseRefused { cause });
    }
}
