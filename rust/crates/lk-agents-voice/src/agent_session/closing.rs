//! `AgentSession._aclose_impl`: the close in flight (`_closing_task`), the step it is at and the
//! `drain()` callers it answers, and the sequence: unwind, `on_exit`, drain, hard edge, trailing
//! transcript, legs. Also the commands refused past the edge and the errors that close.

use lk_agents_abi::Ts;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::events::CloseReason;

use super::{HookId, RequestId};

use crate::agent_activity::false_interruption::InterruptOutcome;
use crate::agent_activity::handoff::HandoffEvent;
use crate::agent_activity::on_exit::Exit;
use crate::agent_activity::scheduling::{PauseCause, SpeechId};
use crate::agent_session::effects::{Effect, Out, TimerOp, TurnTimer};
use crate::agent_session::{
    ActivityCommand, Answer, Close, Command, DrainRefused, Driver, DtmfCommand, HandoffRefused,
    IdleRefused, SpeechCommand,
};
use crate::audio::{AudioChannel, ChannelOp};
use crate::events::SessionEvent;
use crate::trace::{
    CloseAct, CloseStep, CoreStep, HandoffStep, InputStep, InterruptCause, InterruptStep,
    SpeechStep, TurnStep,
};
use crate::workflows::builtins::dtmf::DtmfOutcome;
use std::time::Duration;

/// Silence pushed into the recogniser so a streaming STT finalises: `stt_flush_duration`.
const CLOSE_FLUSH: Duration = Duration::from_secs(2);

/// The error output a tool still running at the close's bound is completed with.
pub const SESSION_CLOSING: &str = "session closing";

/// How long a close's drain waits for the speeches in flight before the hard edge.
pub const DRAIN_BOUND: Duration = crate::speech_handle::INTERRUPTION_TIMEOUT;

/// How long the legs' close is awaited; past it the leg is abandoned.
pub const TEARDOWN_BOUND: Duration = Duration::from_secs(5);

/// How long the edge waits for cut speeches to report their end before releasing them.
pub const EDGE_BOUND: Duration = crate::speech_handle::INTERRUPTION_TIMEOUT;

pub(crate) struct Closing {
    pub(crate) reason: CloseReason,
    pub(crate) error: Option<ProviderError>,
    pub(crate) phase: ClosePhase,
    /// Host `drain()`s answered by the close's own drain (the reference's drain waits on the
    /// activity lock `_aclose_impl` holds). Answered `Ok` once the edge has settled.
    drains: Vec<RequestId>,
    /// `drains` was answered: a later caller is answered at once.
    drains_answered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClosePhase {
    /// An inline task chain is being unwound, innermost first, before the remaining frame's
    /// `on_exit`. `deadline` is `None` once the bound passed and the rest was abandoned.
    Unwinding {
        deadline: Option<Ts>,
    },
    Exiting {
        hook: HookId,
    },
    Draining {
        deadline: Ts,
    },
    /// The hard edge was performed; `until` bounds the cut speeches' reports.
    Edge {
        until: Ts,
    },
    Committing {
        until: Ts,
    },
    /// The realtime socket's `aclose` was asked for; `until` bounds it, `closed` is its answer.
    ClosingActivity {
        until: Ts,
        closed: bool,
    },
    Closed,
}

impl ClosePhase {
    pub(crate) fn deadline(self) -> Option<Ts> {
        match self {
            ClosePhase::Unwinding { deadline } => deadline,
            ClosePhase::Draining { deadline } => Some(deadline),
            ClosePhase::Edge { until }
            | ClosePhase::Committing { until }
            | ClosePhase::ClosingActivity { until, .. } => Some(until),
            ClosePhase::Exiting { .. } | ClosePhase::Closed => None,
        }
    }

    /// Past the hard edge no command may take effect any more.
    pub(crate) fn past_edge(self) -> bool {
        !matches!(
            self,
            ClosePhase::Unwinding { .. } | ClosePhase::Exiting { .. } | ClosePhase::Draining { .. }
        )
    }
}

impl Closing {
    /// Begins unwinding until `deadline`, answering `drains` with its drain.
    pub(crate) fn new(
        reason: CloseReason,
        error: Option<ProviderError>,
        deadline: Ts,
        drains: Vec<RequestId>,
    ) -> Self {
        let phase = ClosePhase::Unwinding { deadline: Some(deadline) };
        Closing { reason, error, phase, drains, drains_answered: false }
    }

    /// A second request joins the close: a drain asked for again restarts its bound.
    pub(crate) fn rejoin(&mut self, drain: bool, deadline: Ts) {
        if let (true, ClosePhase::Draining { deadline: bound }) = (drain, &mut self.phase) {
            *bound = deadline;
        }
    }

    /// Whether a caller of `drain()` can still wait on the close's own drain.
    pub(crate) fn drains_open(&self) -> bool {
        !self.drains_answered
    }

    pub(crate) fn join_drains(&mut self, waiters: Vec<RequestId>) {
        self.drains.extend(waiters);
    }

    /// The close's `drain()` callers, once: empty when answered already or nobody waits.
    pub(crate) fn take_drains(&mut self) -> Vec<RequestId> {
        if std::mem::replace(&mut self.drains_answered, true) {
            return Vec::new();
        }
        std::mem::take(&mut self.drains)
    }

    /// Whoever still waits on the close's drain when the loop ends, answered or not.
    pub(crate) fn remaining_drains(&mut self) -> Vec<RequestId> {
        std::mem::take(&mut self.drains)
    }
}

impl Driver {
    /// `AgentSession._aclose_impl`: an inline task chain unwound innermost first, the frame's
    /// `on_exit`, the drain, the hard edge, the trailing transcript, the legs, `close`. A second
    /// request joins the close in progress (a drain asked for again restarts its bound).
    pub(crate) fn begin_closing(&mut self, now: Ts, close: Close, out: &mut Out) {
        let Close { reason, drain, error } = close;
        if let Some(closing) = &mut self.session.closing {
            closing.rejoin(drain, now.saturating_add(DRAIN_BOUND));
            return;
        }
        out.trace(CloseStep::Requested { reason, drain });
        // The systems (AMD, IVR) close themselves on this, in this instant.
        self.watch.publish(crate::agent_session::SystemTap::CloseRequested);
        // The keyterm pass: the request in flight is cancelled, no push after this.
        self.keyterms_step(now, crate::keyterm_detection::KeytermIn::Close, out);
        let exit = (!self.activity.tasks.any_live()).then(|| self.plan_exit(out));
        // A host `drain()` in progress is adopted: its `on_exit` and its pause become the close's,
        // and the caller is answered with the close's drain.
        let drains = self.activity.drain.take_waiters();
        let deadline = now.saturating_add(DRAIN_BOUND);
        self.session.closing = Some(Closing::new(reason, error, deadline, drains));
        // A non-drain close: every open speech force-interrupted regardless of
        // `allow_interruptions`, a false-interruption pause released so a later turn never
        // resolves it as false, and every pending socket request settled `Closing`.
        if !drain {
            self.force_cut_open(now, out);
            self.cancel_speech_pause(now, out);
            if self.activity.rt_session.is_some() {
                self.realtime_on_close(out);
            }
        }
        match exit {
            Some(exit) => self.close_on_exit(now, exit, out),
            None => self.close_unwind_step(now, out),
        }
    }

    /// Pending handoffs are refused; the agent's `on_exit` runs once: lent by a handoff that
    /// already issued it, skipped if it ran, otherwise issued here.
    fn plan_exit(&mut self, out: &mut Out) -> Exit {
        self.handoff_step(HandoffEvent::Closing, out);
        let exit = self.leave().claim();
        self.issue_claimed_exits(out);
        exit
    }

    /// The remaining agent's `on_exit`, then the drain.
    fn close_on_exit(&mut self, now: Ts, exit: Exit, out: &mut Out) {
        let Some(hook) = exit.hook() else { return self.close_drain(now, out) };
        if let Some(closing) = &mut self.session.closing {
            closing.phase = ClosePhase::Exiting { hook };
        }
    }

    /// The `on_exit` hook `id` ended (answered, timed out, or with nobody to ask).
    pub(crate) fn close_exit_done(&mut self, now: Ts, id: HookId, out: &mut Out) {
        if self.session.close_phase() == Some(ClosePhase::Exiting { hook: id }) {
            self.close_drain(now, out);
        }
    }

    /// `PauseScheduling { cause: Close }`, bounded by [`DRAIN_BOUND`]; settled or refused, the hard
    /// edge follows.
    fn close_drain(&mut self, now: Ts, out: &mut Out) {
        let deadline = now.saturating_add(DRAIN_BOUND);
        if let Some(closing) = &mut self.session.closing {
            closing.phase = ClosePhase::Draining { deadline };
        }
        // No reply plans a follow-up once the close began.
        self.shared.draining.send_replace(true);
        self.pause_scheduling(now, PauseCause::Close, out);
    }

    fn force_cut_open(&mut self, now: Ts, out: &mut Out) {
        let open: Vec<SpeechId> = self.activity.speech_scheduler.speech_ids().collect();
        for speech in open {
            match self.interrupt_speech(now, speech, InterruptCause::Closing, true, out) {
                Ok(InterruptOutcome::Now { .. }) => {}
                // Already stopped: the journal still carries the close's own step for it.
                Ok(InterruptOutcome::Already) => out.trace(InterruptStep::Interrupted {
                    speech,
                    cause: InterruptCause::Closing,
                    force: true,
                    position: None,
                }),
                // Unreachable: `force` overrides the hold. Said rather than swallowed.
                Err(_) => out.trace(CoreStep::PortFailure("a forced close interrupt was refused")),
            }
        }
    }

    pub(crate) fn join_close_drain(&mut self, waiters: Vec<RequestId>, out: &mut Out) {
        if waiters.is_empty() {
            return;
        }
        let exit_hook = self.activity.on_exit.exiting();
        if let Some(closing) = self.session.closing.as_mut().filter(|c| c.drains_open()) {
            out.trace(HandoffStep::DrainStarted { exit_hook });
            closing.join_drains(waiters);
        } else {
            // Answered already, or no close is running (the loop is ending).
            let refused = self.session.closing.is_none().then_some(DrainRefused::Closing);
            for call in waiters {
                out.trace(HandoffStep::Drained { refused });
                out.answer(Some(call), Answer::Drain(refused.map_or(Ok(()), Err)));
            }
        }
    }

    fn answer_close_drains(&mut self, out: &mut Out) {
        let waiters = self.session.closing.as_mut().map(Closing::take_drains).unwrap_or_default();
        if waiters.is_empty() {
            return;
        }
        out.trace(HandoffStep::Drained { refused: None });
        for call in waiters {
            self.answer(Some(call), Answer::Drain(Ok(())));
        }
    }

    /// A task left the chain during unwinding: cancel the next innermost, or go on to the
    /// remaining frame's `on_exit`.
    pub(crate) fn close_unwind_step(&mut self, now: Ts, out: &mut Out) {
        let unwinding = matches!(self.session.close_phase(), Some(ClosePhase::Unwinding { .. }));
        if unwinding && !self.cancel_innermost_task(now, out) {
            let exit = self.plan_exit(out);
            self.close_on_exit(now, exit, out);
        }
    }

    /// After every cycle, advance the close one phase per cycle so the journal never shows a later
    /// step before an earlier one's effects. Returns once waiting on something or stopped.
    pub(crate) fn close_advance(&mut self, now: Ts) {
        loop {
            let mut out = self.out();
            match self.session.close_phase().zip(self.session.closing.as_ref().map(|c| c.reason)) {
                // Every speech the edge cut has reported (or the bound released it).
                Some((ClosePhase::Edge { .. }, reason))
                    if self.activity.speech_scheduler.is_empty() =>
                {
                    let timeout =
                        self.session.config.effective.timers.session_close_transcript_timeout;
                    if reason != CloseReason::Error && self.transcript_pending(now) {
                        self.close_await_transcript(now, now.saturating_add(timeout), &mut out);
                    } else {
                        self.close_legs_begin(now, &mut out);
                    }
                }
                Some((ClosePhase::Committing { .. }, _))
                    if !self.activity.audio_recognition.commit_pending() =>
                {
                    self.close_legs_begin(now, &mut out);
                }
                Some((ClosePhase::ClosingActivity { until, closed }, _))
                    if closed || until <= now =>
                {
                    self.close_legs(&mut out);
                }
                Some((
                    ClosePhase::Unwinding { .. }
                    | ClosePhase::Exiting { .. }
                    | ClosePhase::Draining { .. }
                    | ClosePhase::Edge { .. }
                    | ClosePhase::Committing { .. }
                    | ClosePhase::ClosingActivity { .. }
                    | ClosePhase::Closed,
                    _,
                ))
                | None => return,
            }
            self.pump_schedule(now, &mut out);
            self.execute(out);
        }
    }

    /// Flush the recogniser and wait for its final until `until`.
    fn close_await_transcript(&mut self, now: Ts, until: Ts, out: &mut Out) {
        if let Some(closing) = &mut self.session.closing {
            closing.phase = ClosePhase::Committing { until };
        }
        self.answer_close_drains(out);
        let timeout = until.saturating_duration_since(now);
        // Nobody awaits it: the commit's `EndOfTurn` and user item are the record.
        self.begin_commit(now, timeout, CLOSE_FLUSH, true, true, None, out);
    }

    /// Tell every leg to close. The realtime socket's `aclose` runs off the loop, bounded by
    /// [`TEARDOWN_BOUND`]; its answer comes back as `RealtimeWake::Closed`.
    fn close_legs_begin(&mut self, now: Ts, out: &mut Out) {
        let rt = self.activity.rt_session.clone();
        let until = now.saturating_add(TEARDOWN_BOUND);
        if let Some(closing) = &mut self.session.closing {
            closing.phase = ClosePhase::ClosingActivity { until, closed: rt.is_none() };
        }
        self.answer_close_drains(out);
        let _ = self.shared.closing.send(true);
        out.machine().close_turn_detector();
        if let Some(rt) = rt {
            let tx = self.realtime_tx.clone();
            lk_agents_abi::SpawnerExt::spawn_unboxed(&*self.shared.spawner, async move {
                rt.aclose().await;
                let _ = tx.send(crate::agent_activity::realtime::RealtimeWake::Closed);
            });
        }
    }

    /// A close bound passed. `true` when the close moved to a later phase.
    pub(crate) fn close_tick(&mut self, now: Ts, out: &mut Out) -> bool {
        let Some(phase) = self.session.close_phase() else { return false };
        if !phase.deadline().is_some_and(|d| d <= now) {
            return false;
        }
        match phase {
            // The unwind's bound: every pushed frame popped without a handoff, every awaiter
            // released, then the remaining frame's `on_exit`.
            ClosePhase::Unwinding { .. } => {
                if let Some(closing) = &mut self.session.closing {
                    closing.phase = ClosePhase::Unwinding { deadline: None };
                }
                self.abandon_tasks(out);
                let exit = self.plan_exit(out);
                self.close_on_exit(now, exit, out);
                true
            }
            ClosePhase::Draining { .. } => {
                self.close_edge(now, out);
                true
            }
            // The edge's bound passed with a cut speech unreported: released by the close.
            ClosePhase::Edge { .. } => {
                let open: Vec<SpeechId> = self.activity.speech_scheduler.speech_ids().collect();
                for speech in open {
                    out.trace(CloseStep::Performed(CloseAct::Released { speech }));
                    self.release_speech(now, speech, out);
                }
                false
            }
            ClosePhase::Exiting { .. }
            | ClosePhase::Committing { .. }
            | ClosePhase::ClosingActivity { .. }
            | ClosePhase::Closed => false,
        }
    }

    /// The legs closed or their bound passed: the session is closed.
    /// The loop stops here; [`Driver::finish`] sends the last event.
    fn close_legs(&mut self, out: &mut Out) {
        if let Some(closing) = &mut self.session.closing {
            closing.phase = ClosePhase::Closed;
        }
        self.answer_close_drains(out);
        self.stopped = true;
    }

    /// Whether the recogniser may still owe the last final: the user is speaking, words are held
    /// uncommitted, or speech ended within the last second. Only with an STT leg.
    fn transcript_pending(&self, now: Ts) -> bool {
        if self.input.leg(AudioChannel::Stt).is_none() {
            return false;
        }
        let turn = &self.activity.audio_recognition;
        if turn.user_speaking() || !turn.current_transcript().trim().is_empty() {
            return true;
        }
        match (turn.last_speaking_time(), turn.last_final_transcript_time()) {
            (Some(spoke), last_final) => {
                last_final.is_none_or(|f| f < spoke)
                    && now.saturating_duration_since(spoke) <= Duration::from_secs(1)
            }
            (None, _) => false,
        }
    }

    /// A command past the hard edge: its promise is answered, nothing takes effect.
    pub(crate) fn refuse_command(&mut self, command: Command, out: &mut Out) {
        let (what, lost, call) = match command {
            Command::Speech(SpeechCommand::Say { call, .. }) => {
                ("say", "a say() reached the session after it began closing: it never spoke", call)
            }
            Command::Speech(SpeechCommand::GenerateReply { call, .. }) => (
                "generate_reply",
                "a generate_reply() reached the session after it began closing: it never replied",
                call,
            ),
            // A handoff somebody awaits: refused as closing, explicitly.
            Command::Activity(ActivityCommand::Update(update)) => {
                self.refuse_handoff(None, update, HandoffRefused::Closing, out);
                return;
            }
            // An inline task somebody awaits: refused as closing, its awaiter released.
            Command::Task(cmd) => {
                self.refuse_task_command(cmd, out);
                return;
            }
            // A collection that never starts: answered as stopped, as after a `Stop`.
            Command::Dtmf(DtmfCommand::Start { call, .. }) => {
                self.answer(call, Answer::Dtmf(DtmfOutcome::Stopped));
                return;
            }
            // A drain somebody awaits: the close's drain is the one it waits on.
            Command::Activity(ActivityCommand::Drain { call }) => {
                self.join_close_drain(call.into_iter().collect(), out);
                return;
            }
            // A run somebody awaits: refused as closing, on the stream.
            Command::Run(_) => {
                self.emit_now(SessionEvent::RunRefused {
                    cause: crate::events::RunRefusal::Closing,
                });
                return;
            }
            // Past the edge nothing is speaking, so an interrupt is `Ok`; a wait is refused.
            Command::Speech(SpeechCommand::Interrupt { call, .. }) => {
                self.answer(call, Answer::Interrupt(Ok(())));
                return;
            }
            Command::Speech(SpeechCommand::WaitForIdle { call, .. }) => {
                out.trace(SpeechStep::IdleAnswered { hold: None, closing: true });
                self.answer(call, Answer::Idle(Err(IdleRefused::Closing)));
                return;
            }
            Command::Speech(SpeechCommand::WaitForIdleAndHold { call }) => {
                out.trace(SpeechStep::IdleAnswered { hold: None, closing: true });
                self.answer(call, Answer::Hold(Err(IdleRefused::Closing)));
                return;
            }
            // A commit the loop never ran: `""`, as the reference answers a closing session.
            Command::Speech(SpeechCommand::CommitUserTurn { call, .. }) => {
                out.trace(TurnStep::CommitAnswered { transcript: String::new() });
                self.answer(call, Answer::Commit(Ok(String::new())));
                return;
            }
            // Nothing awaits these: a dropped one is a lost instruction, not a lost promise.
            Command::Speech(
                SpeechCommand::InterruptSpeech { .. }
                | SpeechCommand::ReleaseIdleHold { .. }
                | SpeechCommand::ClaimUserTurn
                | SpeechCommand::ReleaseUserTurn
                | SpeechCommand::SetAllowInterruptions { .. }
                | SpeechCommand::SetRunOutput { .. }
                | SpeechCommand::ToolUpdate { .. }
                | SpeechCommand::FillerStart { .. }
                | SpeechCommand::FillerStop { .. },
            )
            | Command::Activity(
                ActivityCommand::UpdateInstructions(_)
                | ActivityCommand::UpdateTools(_)
                | ActivityCommand::SetAudioInput(_)
                | ActivityCommand::SetVideoInput(_)
                | ActivityCommand::SetIoEnabled { .. }
                | ActivityCommand::UpdateChatCtx(_)
                | ActivityCommand::SetAudioOutput(_)
                | ActivityCommand::SetTranscriptionOutput(_)
                | ActivityCommand::UpdateModels { .. }
                | ActivityCommand::SetSpeakerFilter(_),
            )
            | Command::Turn(_)
            | Command::Video(_)
            | Command::Close(_)
            | Command::HookDone { .. }
            | Command::HostMetrics(_)
            | Command::SetStaticKeyterms(_)
            | Command::Recorder(_)
            | Command::Ivr(_)
            | Command::Dtmf(DtmfCommand::Digit { .. } | DtmfCommand::Stop)
            | Command::Speech(
                SpeechCommand::PauseAuthorization | SpeechCommand::ResumeAuthorization,
            ) => return,
        };
        out.trace(CoreStep::PortFailure(lost));
        self.settle_undelivered(what, call);
    }

    /// The hard edge: scheduling stopped, timers cancelled, countdown committed or torn down, what
    /// is open interrupted; after it only the close's own steps happen. The close moves on once
    /// every cut speech has reported `SpeechDone`, as the reference awaits `current_speech`, or
    /// [`EDGE_BOUND`] releases them.
    pub(crate) fn close_edge(&mut self, now: Ts, out: &mut Out) {
        if let Some(closing) = &mut self.session.closing {
            closing.phase = ClosePhase::Edge { until: now.saturating_add(EDGE_BOUND) };
        }
        let reason =
            self.session.closing.as_ref().map(|c| c.reason).unwrap_or(CloseReason::UserInitiated);
        self.session.on_close_edge();
        // A run still open ends here, `Closed`, before the close's own last event.
        self.run_close(out);
        // Detach the microphone: the recogniser hears only the commit's silence flush.
        self.apply_leg_op(ChannelOp::SetSource(None), out);
        // And the recorder: its last flush, its timer cancelled at the edge.
        self.detach_recorder(now, out);
        // `wait_for_idle` raises `ActivityClosedError` on a closing session; the trailing-turn
        // commit below commits what the turn holds.
        self.refuse_idle_waiters(out);
        // A pending host `commit_user_turn()` is answered below from the close's commit, never
        // with words that were not committed.
        let host_commit_pending = self.activity.audio_recognition.commit_pending();
        // Refuse a handoff in flight; a running hook is left to the host and its wait abandoned.
        self.abandon_handoffs(out);
        // The per-leg frame census, read off the one table.
        for leg in AudioChannel::ALL {
            let (frames, silent) = self.input.legs().counters(leg);
            if frames > 0 {
                out.trace(InputStep::LegFrames { leg, frames, silent });
            }
        }
        // The shapes those frames had: the microphone's and the fan-out's, then each row's.
        if let Some((input, converted)) = self.input.legs().input_shape() {
            out.trace(InputStep::InputShape {
                sample_rate: input.sample_rate,
                num_channels: input.num_channels,
                samples_per_channel: input.samples_per_channel,
                converted_rate: converted.sample_rate,
                converted_samples_per_channel: converted.samples_per_channel,
            });
            for leg in AudioChannel::ALL {
                if let Some(shape) = self.input.legs().shape(leg) {
                    out.trace(InputStep::LegShape {
                        leg,
                        sample_rate: shape.sample_rate,
                        num_channels: shape.num_channels,
                        samples_per_channel: shape.samples_per_channel,
                    });
                }
            }
        }
        // Cut a running endpointing delay short, unless the close is an error's: that commits no
        // trailing transcript, so the countdown is torn down.
        let eff = self.activity.audio_recognition.on_closing(now, reason != CloseReason::Error);
        out.machine().append(eff);
        if host_commit_pending {
            self.perform_pending_commit(now, out);
        }
        // An in-flight prediction answers a turn nobody will decide: end its task.
        out.machine().close_turn_detector();
        // Cancel the turn's armed timers through the cycle so each is journalled, skipping those
        // the turn machine just cancelled; the others are cancelled with what owns them.
        {
            let already: Vec<TurnTimer> = out
                .peek()
                .iter()
                .filter_map(|e| match e {
                    Effect::Timer(TimerOp::Cancel(t)) => Some(*t),
                    Effect::Timer(TimerOp::Arm(..))
                    | Effect::Provider(_)
                    | Effect::Emit(_)
                    | Effect::Decision(_)
                    | Effect::Realtime(_)
                    | Effect::Audio(_)
                    | Effect::Detector(_) => None,
                })
                .collect();
            for t in [
                TurnTimer::EotPredictionTimeout,
                TurnTimer::EndpointingDelay,
                TurnTimer::TranscriptionTimeout,
                TurnTimer::BackchannelBoundary,
                TurnTimer::FalseInterruption,
                TurnTimer::AecWarmup,
                TurnTimer::UserAway,
                TurnTimer::UserTurnDuration,
            ] {
                if self.timers.is_armed(t) && !already.contains(&t) {
                    out.machine().cancel(t);
                }
            }
        }
        // Watchdogs go with their speeches, the current speech is released, a pending preemptive
        // generation discarded, a drain-waiting pause refused, and nothing is authorized again.
        self.close_scheduling(now, out);
        // What the drain did not finish is force-interrupted; a speech that does not report its
        // end by the edge's bound is released by the close. `BTreeMap` order keeps runs stable.
        self.force_cut_open(now, out);
        // Release a held pause so no turn during the transcript wait resolves it as false.
        self.cancel_speech_pause(now, out);
        // The socket too, settling every `generate_reply()` still awaiting `GenerationCreated`.
        self.realtime_interrupt(out);
    }

    pub(crate) fn settle_waiters_at_end(&mut self) {
        for w in std::mem::take(&mut self.activity.interrupt_waiters) {
            self.answer(w.call, Answer::Interrupt(Ok(())));
        }
        for waiter in self.session.idle.take_waiters() {
            self.answer(waiter.call, waiter.answer(Err(IdleRefused::Closing)));
        }
        if let Some(pending) = self.activity.audio_recognition.take_pending_commit() {
            self.answer(pending.call, Answer::Commit(Ok(String::new())));
        }
        // The close ran to its end: whoever still waits on its drain is answered as drained.
        for call in self.session.closing.as_mut().map(Closing::remaining_drains).unwrap_or_default()
        {
            self.answer(Some(call), Answer::Drain(Ok(())));
        }
        // The loop ended with no close to answer them (its queue gone).
        for call in self.activity.drain.take_waiters() {
            self.answer(Some(call), Answer::Drain(Err(DrainRefused::Closing)));
        }
        let call = self.activity.dtmf.take().and_then(|d| d.call);
        self.answer(call, Answer::Dtmf(DtmfOutcome::Stopped));
    }

    /// `_on_error` (activity then session): the `ErrorEvent` for every failed attempt, then the
    /// close policy on unrecoverable ones ([`ErrorCounts`](crate::agent_session::ErrorCounts)).
    pub(crate) fn report_error(
        &mut self,
        now: Ts,
        source: crate::events::ErrorSource,
        recoverable: bool,
        error: lk_agents_abi::errors::ProviderError,
        speech: Option<SpeechId>,
        out: &mut Out,
    ) {
        let ev = crate::events::ErrorEvent { source, recoverable, error: error.clone(), speech };
        out.machine().emit(SessionEvent::Error(Box::new(ev)));
        if recoverable || self.session.closing.is_some() {
            return;
        }
        if self.session.errors.exhausted_by(source) {
            let close = Close { reason: CloseReason::Error, drain: false, error: Some(error) };
            self.begin_closing(now, close, out);
        }
    }
}
