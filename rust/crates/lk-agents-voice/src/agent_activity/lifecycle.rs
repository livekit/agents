//! The session's close (`_aclose_impl`): unwind, `on_exit`, drain, hard edge, transcript, legs;
//! the commands refused past it, the room link, the errors that close, and `AgentSession.run()`.

use crate::agent_session::cycle::Cycle;
use crate::agent_activity::pause::InterruptOutcome;
use crate::agent_activity::scheduling::{PauseCause, SpeechId};
use crate::agent_activity::speech::IdleDone;
use crate::agent::Source;
use crate::agent_session::{
    Close, ClosePhase, Command, Driver, FrameCommand, HandoffRefused, HoldId, HookId, HookKind,
    HostEvent, IdleRefused, Origin, SpeechCommand,
};
use crate::audio::{LegId, LegOp};
use crate::events::{RunId, RunOutcome, RunRefusal, SessionEvent};
use crate::room_io::{DeleteRoomRequested, RoomIoEvent};
use crate::run_result::{Decided, Owned, RunOutput, RunSpec};
use crate::speech_handle::SpeechHandle;
use crate::trace::{
    CloseAct, CloseStep, CoreStep, HandoffStep, InputStep, InterruptCause, InterruptStep,
    RoomLinkStep, RoomStep, SpeechStep, Step, TurnStep,
};
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::rtc::DisconnectReason;
use lk_agents_abi::Ts;
use std::sync::Arc;
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

/// What the scheduling task tells the close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseIn {
    /// The close ended a speech. `live_tools` counts blocking tool calls whose reply task will
    /// never report.
    SpeechReleased { speech: SpeechId, live_tools: u32 },
}

impl Driver {
    /// `AgentSession._aclose_impl`: an inline task chain unwound innermost first, the frame's
    /// `on_exit`, the drain, the hard edge, the trailing transcript, the legs, `close`. A second
    /// request joins the close in progress (a drain asked for again restarts its bound).
    pub(crate) fn begin_closing(&mut self, now: Ts, close: Close, cycle: &mut Cycle) {
        let Close { reason, drain, error } = close;
        if self.session.rejoin_close(drain, now.saturating_add(DRAIN_BOUND)) {
            return;
        }
        cycle.trace(CloseStep::Requested { reason, drain });
        // The systems (AMD, IVR) close themselves on this, in this instant.
        self.watch.publish(crate::agent_session::Watched::CloseRequested);
        // The keyterm pass: the request in flight is cancelled, no push after this.
        self.keyterms_close(now, cycle);
        let exit = (!self.activity.tasks().any_live()).then(|| self.plan_exit(cycle));
        // A host `drain()` in progress is adopted: its `on_exit` and its pause become the close's,
        // and the caller is answered with the close's drain.
        let drains = self.activity.hooks_mut().take_drain_waiters();
        self.session.start_close(reason, error, now.saturating_add(DRAIN_BOUND), drains);
        if !drain {
            self.cut_speeches(now, cycle);
        }
        match exit {
            Some(exit) => self.close_on_exit(now, exit, cycle),
            None => self.close_unwind_step(now, cycle),
        }
    }

    /// Pending handoffs are refused; the frame's `on_exit` runs once: lent by a handoff that
    /// already issued it, skipped if it ran, otherwise issued here.
    fn plan_exit(&mut self, cycle: &mut Cycle) -> (Option<HookId>, bool) {
        if let Some((id, update)) = self.activity.hooks_mut().take_handoff() {
            self.refuse_handoff(Some(id), update, HandoffRefused::Closing, cycle);
        }
        self.refuse_queued_handoffs(cycle);
        // A turn parked for the arriving frame: no frame arrives now.
        self.activity.hooks_mut().clear_parked_turn();
        self.activity.hooks_mut().frame_exit()
    }

    /// The remaining frame's `on_exit`, then the drain.
    fn close_on_exit(&mut self, now: Ts, (hook, issue): (Option<HookId>, bool), cycle: &mut Cycle) {
        let Some(hook) = hook else { return self.close_drain(now, cycle) };
        self.session.set_close_phase(ClosePhase::Exiting { hook });
        if issue && self.issue_on_exit(now, hook, cycle) {
            self.close_exit_done(now, hook, cycle);
        }
    }

    /// The `on_exit` hook `id` ended (answered, timed out, or abandoned).
    pub(crate) fn close_exit_done(&mut self, now: Ts, id: HookId, cycle: &mut Cycle) {
        if self.session.close_phase() == Some(ClosePhase::Exiting { hook: id }) {
            self.close_drain(now, cycle);
        }
    }

    /// `PauseScheduling { cause: Close }`, bounded by [`DRAIN_BOUND`].
    fn close_drain(&mut self, now: Ts, cycle: &mut Cycle) {
        let deadline = now.saturating_add(DRAIN_BOUND);
        // No reply plans a follow-up once the close began.
        let (done, settled) = tokio::sync::oneshot::channel();
        self.session.close_draining(deadline, settled);
        self.shared.draining.send_replace(true);
        self.pause_scheduling(now, PauseCause::Close, done, cycle);
        self.close_ready(now, cycle);
    }

    /// The hard edge; the close moves on once every cut speech has reported `SpeechDone`, as the
    /// reference awaits `current_speech`, or [`EDGE_BOUND`] releases them.
    fn close_edge(&mut self, now: Ts, cycle: &mut Cycle) {
        self.session.set_close_phase(ClosePhase::Edge { until: now.saturating_add(EDGE_BOUND) });
        self.hard_edge(now, cycle);
    }

    /// Blocking tool calls still running are answered with the closing error before the speech's
    /// own `SpeechDone`; nothing is journalled after `close`.
    pub(crate) fn close_step(&mut self, now: Ts, input: CloseIn, cycle: &mut Cycle) {
        let CloseIn::SpeechReleased { speech, live_tools } = input;
        if live_tools > 0 && self.session.close_phase().is_some_and(|p| p != ClosePhase::Closed) {
            self.end_live_tools(now, speech, cycle);
        }
    }

    /// A non-drain close: every open speech force-interrupted regardless of
    /// `allow_interruptions`, and the socket told once.
    fn cut_speeches(&mut self, now: Ts, cycle: &mut Cycle) {
        self.force_cut_open(now, cycle);
        // Release a false-interruption pause so a later turn never resolves it as false.
        self.cancel_speech_pause(now, cycle);
        // Every pending socket request settles `Closing`; the open generation is cancelled.
        if self.activity.rt_session().is_some() {
            self.realtime_close(cycle);
        }
    }

    fn force_cut_open(&mut self, now: Ts, cycle: &mut Cycle) {
        let open: Vec<SpeechId> = self.activity.scheduling_task().speech_ids().collect();
        for speech in open {
            match self.interrupt_speech(now, speech, InterruptCause::Closing, true, cycle) {
                Ok(InterruptOutcome::Now { .. }) => {}
                // Already stopped: the journal still carries the close's own step for it.
                Ok(InterruptOutcome::Already) => cycle.trace(InterruptStep::Interrupted {
                    speech,
                    cause: InterruptCause::Closing,
                    force: true,
                    position: None,
                }),
                // Unreachable: `force` overrides the hold. Said rather than swallowed.
                Err(_) => {
                    cycle.trace(CoreStep::PortFailure("a forced close interrupt was refused"))
                }
            }
        }
    }

    pub(crate) fn join_close_drain(
        &mut self,
        waiters: Vec<tokio::sync::oneshot::Sender<Result<(), crate::agent_session::DrainRefused>>>,
        cycle: &mut Cycle,
    ) {
        if waiters.is_empty() {
            return;
        }
        let exit_hook = self.activity.hooks().exit_hook();
        if self.session.close_drains_open() {
            cycle.trace(crate::trace::HandoffStep::DrainStarted { exit_hook });
            self.session.join_close_drains(waiters);
        } else {
            // Answered already, or no close is running (the loop is ending).
            let refused =
                (!self.session.is_closing()).then_some(crate::agent_session::DrainRefused::Closing);
            for done in waiters {
                self.answer_drain(Some(done), refused, cycle);
            }
        }
    }

    fn answer_close_drains(&mut self, cycle: &mut Cycle) {
        let waiters = self.session.take_close_drains();
        if waiters.is_empty() {
            return;
        }
        cycle.trace(crate::trace::HandoffStep::Drained { refused: None });
        for done in waiters {
            let _ = done.send(Ok(()));
        }
    }

    /// Read after every cycle: the scheduling task answering or refusing the close's pause ends the
    /// drain.
    pub(crate) fn close_ready(&mut self, now: Ts, cycle: &mut Cycle) {
        if self.session.close_drain_settled() {
            self.close_edge(now, cycle);
        }
    }

    /// A task left the chain during unwinding: cancel the next innermost, or go on to the
    /// remaining frame's `on_exit`.
    pub(crate) fn close_unwind_step(&mut self, now: Ts, cycle: &mut Cycle) {
        let unwinding = matches!(self.session.close_phase(), Some(ClosePhase::Unwinding { .. }));
        if unwinding && !self.cancel_innermost_task(now, cycle) {
            let exit = self.plan_exit(cycle);
            self.close_on_exit(now, exit, cycle);
        }
    }

    /// After every cycle, advance the close one phase per cycle so the journal never shows a later
    /// step before an earlier one's effects. Returns once waiting on something or stopped.
    pub(crate) async fn close_advance(&mut self) {
        loop {
            let now = self.shared.time.now();
            let mut cycle = self.ports();
            match self.session.close_phase().zip(self.session.close_reason()) {
                // Every speech the edge cut has reported (or the bound released it).
                Some((ClosePhase::Edge { .. }, reason))
                    if self.activity.scheduling_task().is_empty() =>
                {
                    let timeout = self.session.options().timers.session_close_transcript_timeout;
                    if reason != CloseReason::Error && self.transcript_pending(now) {
                        self.close_await_transcript(now, now.saturating_add(timeout), &mut cycle);
                    } else {
                        self.close_legs_begin(&mut cycle);
                    }
                }
                Some((ClosePhase::Committing { .. }, _))
                    if !self.activity.audio_recognition().commit_pending() =>
                {
                    self.close_legs_begin(&mut cycle);
                }
                Some((ClosePhase::ClosingLegs, _)) => {
                    self.close_legs(now).await;
                    continue;
                }
                Some((
                    ClosePhase::Unwinding { .. }
                    | ClosePhase::Exiting { .. }
                    | ClosePhase::Draining { .. }
                    | ClosePhase::Edge { .. }
                    | ClosePhase::Committing { .. }
                    | ClosePhase::Closed,
                    _,
                ))
                | None => return,
            }
            self.pump_schedule(now, &mut cycle);
            self.execute(cycle).await;
        }
    }

    /// Flush the recogniser and wait for its final until `until`.
    fn close_await_transcript(&mut self, now: Ts, until: Ts, cycle: &mut Cycle) {
        self.session.set_close_phase(ClosePhase::Committing { until });
        self.answer_close_drains(cycle);
        let timeout = until.saturating_duration_since(now);
        // Nothing reads the answer: the commit's `EndOfTurn` and user item are the record.
        let (done, _unread) = tokio::sync::oneshot::channel();
        self.begin_commit(now, timeout, CLOSE_FLUSH, true, true, done, cycle);
    }

    /// Tell every leg to close; the realtime socket is awaited by `close_legs`.
    fn close_legs_begin(&mut self, cycle: &mut Cycle) {
        self.session.set_close_phase(ClosePhase::ClosingLegs);
        self.answer_close_drains(cycle);
        let _ = self.shared.closing.send(true);
        cycle.machine().close_turn_detector();
    }

    /// A close bound passed. `true` when the close moved to a later phase.
    pub(crate) fn close_tick(&mut self, now: Ts, cycle: &mut Cycle) -> bool {
        let Some(phase) = self.session.close_phase() else { return false };
        if !phase.deadline().is_some_and(|d| d <= now) {
            return false;
        }
        match phase {
            // The unwind's bound: every pushed frame popped without a handoff, every awaiter
            // released, then the remaining frame's `on_exit`.
            ClosePhase::Unwinding { .. } => {
                self.session.set_close_phase(ClosePhase::Unwinding { deadline: None });
                self.abandon_tasks(now, cycle);
                let exit = self.plan_exit(cycle);
                self.close_on_exit(now, exit, cycle);
                true
            }
            ClosePhase::Draining { .. } => {
                self.close_edge(now, cycle);
                true
            }
            // The edge's bound passed with a cut speech unreported: released by the close.
            ClosePhase::Edge { .. } => {
                let open: Vec<SpeechId> = self.activity.scheduling_task().speech_ids().collect();
                for speech in open {
                    cycle.trace(CloseStep::Performed(CloseAct::Released { speech }));
                    self.release_speech(now, speech, cycle);
                }
                false
            }
            ClosePhase::Exiting { .. }
            | ClosePhase::Committing { .. }
            | ClosePhase::ClosingLegs
            | ClosePhase::Closed => false,
        }
    }

    /// The realtime socket's `aclose`, bounded by [`TEARDOWN_BOUND`] on the session clock.
    async fn close_legs(&mut self, now: Ts) {
        let mut cycle = self.ports();
        if let Some(rt) = self.activity.rt_session().cloned() {
            let time = self.shared.time.clone();
            tokio::select! {
                biased;
                () = rt.aclose() => {}
                () = time.wait_until(now.saturating_add(TEARDOWN_BOUND)) => {}
            }
        }
        self.session.set_close_phase(ClosePhase::Closed);
        self.answer_close_drains(&mut cycle);
        // Answer queued commands before the last event: nothing settles after `close`.
        self.report_undelivered_commands();
        if let Some((reason, error)) = self.session.close_outcome() {
            cycle.machine().emit(SessionEvent::Close { reason, error });
        }
        self.stopped = true;
        self.execute(cycle).await;
    }

    /// Whether the recogniser may still owe the last final: the user is speaking, words are held
    /// uncommitted, or speech ended within the last second. Only with an STT leg.
    fn transcript_pending(&self, now: Ts) -> bool {
        if self.input.leg(LegId::Stt).is_none() {
            return false;
        }
        let turn = self.activity.audio_recognition();
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
    pub(crate) fn refuse_command(&mut self, command: Command, cycle: &mut Cycle) {
        let (what, lost, waiting) = match command {
            Command::Speech(SpeechCommand::Say { handle, .. }) => (
                "say",
                "a say() reached the session after it began closing: it never spoke",
                handle,
            ),
            Command::Speech(SpeechCommand::GenerateReply { handle, .. }) => (
                "generate_reply",
                "a generate_reply() reached the session after it began closing: it never replied",
                handle,
            ),
            // A handoff somebody awaits: refused as closing, explicitly.
            Command::Frame(FrameCommand::Update(update)) => {
                self.refuse_handoff(None, update, HandoffRefused::Closing, cycle);
                return;
            }
            // An inline task somebody awaits: refused as closing, its awaiter released.
            Command::Task(cmd) => {
                let now = self.shared.time.now();
                self.refuse_task_command(now, cmd, cycle);
                return;
            }
            // A collection that never starts: answered as stopped, as after a `Stop`.
            Command::Dtmf(crate::agent_session::DtmfCommand::Start { done, .. }) => {
                let _ = done.send(crate::workflows::builtins::dtmf::DtmfOutcome::Stopped);
                return;
            }
            // A drain somebody awaits: the close's drain is the one it waits on.
            Command::Frame(FrameCommand::Drain { done }) => {
                self.join_close_drain(done.into_iter().collect(), cycle);
                return;
            }
            // A run somebody awaits: refused as closing, on the stream.
            Command::Run(_) => {
                self.refuse_run();
                return;
            }
            // Past the edge nothing is speaking, so an interrupt is `Ok`; a wait is refused.
            Command::Speech(SpeechCommand::Interrupt { done, .. }) => {
                if let Some(done) = done {
                    let _ = done.send(Ok(()));
                }
                return;
            }
            Command::Speech(SpeechCommand::WaitForIdle { done, .. }) => {
                cycle.trace(SpeechStep::IdleAnswered { hold: None, closing: true });
                let _ = done.send(Err(IdleRefused::Closing));
                return;
            }
            Command::Speech(SpeechCommand::HoldFloor { done }) => {
                cycle.trace(SpeechStep::IdleAnswered { hold: None, closing: true });
                let _ = done.send(Err(IdleRefused::Closing));
                return;
            }
            // A commit the loop never ran: `""`, as the reference answers a closing session.
            Command::Speech(SpeechCommand::CommitUserTurn { done, .. }) => {
                cycle.trace(TurnStep::CommitAnswered { transcript: String::new() });
                let _ = done.send(String::new());
                return;
            }
            // Nothing awaits these: a dropped one is a lost instruction, not a lost promise.
            Command::Speech(
                SpeechCommand::InterruptSpeech { .. }
                | SpeechCommand::ReleaseFloor { .. }
                | SpeechCommand::ClaimUserTurn
                | SpeechCommand::ReleaseUserTurn
                | SpeechCommand::SetAllowInterruptions { .. }
                | SpeechCommand::SetRunOutput { .. }
                | SpeechCommand::ToolUpdate { .. }
                | SpeechCommand::FillerStart { .. }
                | SpeechCommand::FillerStop { .. },
            )
            | Command::Frame(
                FrameCommand::UpdateInstructions(_)
                | FrameCommand::UpdateTools(_)
                | FrameCommand::SetAudioInput(_)
                | FrameCommand::SetVideoInput(_)
                | FrameCommand::SetIoEnabled { .. }
                | FrameCommand::UpdateChatCtx(_)
                | FrameCommand::SetAudioOutput(_)
                | FrameCommand::SetTranscriptionOutput(_)
                | FrameCommand::UpdateModels { .. }
                | FrameCommand::SetSpeakerFilter(_),
            )
            | Command::Turn(_)
            | Command::Video(_)
            | Command::Close(_)
            | Command::HookDone { .. }
            | Command::HostMetrics(_)
            | Command::SetStaticKeyterms(_)
            | Command::Recorder(_)
            | Command::Ivr(_)
            | Command::Dtmf(
                crate::agent_session::DtmfCommand::Digit { .. }
                | crate::agent_session::DtmfCommand::Stop,
            )
            | Command::Speech(
                SpeechCommand::PauseAuthorization | SpeechCommand::ResumeAuthorization,
            ) => return,
        };
        cycle.trace(CoreStep::PortFailure(lost));
        self.settle_undelivered(what, waiting);
    }

    pub(crate) fn on_room_io(&mut self, now: Ts, ev: RoomIoEvent, cycle: &mut Cycle) {
        match &ev {
            RoomIoEvent::ParticipantUnlinked(unlinked) => {
                let reason = unlinked.reason;
                // A deletion in flight is the end of the call already (`not self._delete_room_task`).
                let closes = self.session.disconnect_closes(reason);
                cycle.trace(RoomStep::Link(RoomLinkStep::Unlinked { reason, closes }));
                if closes {
                    self.close_for_room(now, CloseReason::ParticipantDisconnected, cycle);
                }
            }
            // Journalled so a re-link is a step the rules can read; no driver state changes.
            RoomIoEvent::ParticipantLinked(_) => cycle.trace(RoomStep::Link(RoomLinkStep::Linked)),
            RoomIoEvent::TextInput(_) | RoomIoEvent::DeleteRoomRequested(_) => {}
        }
        let _ = self.events.send(HostEvent::Room(ev));
    }

    pub(crate) fn on_session_closed(&mut self) {
        if let Some(room) = self.session.request_room_deletion() {
            self.journal(Step::Room(RoomStep::Link(RoomLinkStep::DeleteRoomRequested)));
            let _ = self.events.send(HostEvent::Room(RoomIoEvent::DeleteRoomRequested(
                DeleteRoomRequested { room },
            )));
        }
    }

    pub(crate) fn close_for_room(&mut self, now: Ts, reason: CloseReason, cycle: &mut Cycle) {
        self.begin_closing(now, Close { reason, drain: false, error: None }, cycle)
    }

    /// The close's hard edge: scheduling stopped, timers cancelled, countdown committed or torn
    /// down, what is open interrupted. After it only the close's own steps happen.
    pub(crate) fn hard_edge(&mut self, now: Ts, cycle: &mut Cycle) {
        let reason = self.session.close_reason().unwrap_or(CloseReason::UserInitiated);
        self.session.on_close_edge();
        // A run still open ends here, `Closed`, before the close's own last event.
        self.run_close(cycle);
        // Detach the microphone: the recogniser hears only the commit's silence flush.
        self.apply_leg_op(LegOp::SetSource(None), cycle);
        // And the recorder: its last flush, its timer cancelled at the edge.
        self.detach_recorder(now, cycle);
        // `wait_for_idle` raises `ActivityClosedError` on a closing session; the trailing-turn
        // commit below commits what the turn holds.
        self.refuse_idle_waiters(cycle);
        // A pending host `commit_user_turn()` is answered below from the close's commit, never
        // with words that were not committed.
        let host_commit_pending = self.activity.audio_recognition().commit_pending();
        // Refuse a handoff in flight; a running hook is left to the host and its wait abandoned.
        self.abandon_handoffs(cycle);
        // The per-leg frame census, read off the one table.
        for leg in LegId::ALL {
            let (frames, silent) = self.input.legs().counters(leg);
            if frames > 0 {
                cycle.trace(InputStep::LegFrames { leg, frames, silent });
            }
        }
        // The shapes those frames had: the microphone's and the fan-out's, then each row's.
        if let Some((input, converted)) = self.input.legs().input_shape() {
            cycle.trace(InputStep::InputShape {
                sample_rate: input.sample_rate,
                num_channels: input.num_channels,
                samples_per_channel: input.samples_per_channel,
                converted_rate: converted.sample_rate,
                converted_samples_per_channel: converted.samples_per_channel,
            });
            for leg in LegId::ALL {
                if let Some(shape) = self.input.legs().shape(leg) {
                    cycle.trace(InputStep::LegShape {
                        leg,
                        sample_rate: shape.sample_rate,
                        num_channels: shape.num_channels,
                        samples_per_channel: shape.samples_per_channel,
                    });
                }
            }
        }
        // Cut a running endpointing delay short -- unless the close is an error's: that commits
        // no trailing transcript, so the countdown is torn down.
        let eff =
            self.activity.audio_recognition_mut().on_closing(now, reason != CloseReason::Error);
        cycle.machine().append(eff);
        if host_commit_pending {
            self.perform_pending_commit(now, cycle);
        }
        // An in-flight prediction answers a turn nobody will decide: end its task.
        cycle.machine().close_turn_detector();
        // Cancel every armed timer through the cycle so each is journalled, skipping those the
        // turn machine just cancelled. No timer may outlive `Closing`.
        {
            use crate::agent_session::cycle::Effect;
            use crate::agent_session::cycle::TimerOp;
            use crate::agent_session::cycle::TurnTimer;
            let already: Vec<TurnTimer> = cycle
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
                    cycle.machine().cancel(t);
                }
            }
        }
        // Watchdogs go with their speeches, the current speech is released, a pending preemptive
        // generation discarded, a drain-waiting pause refused, and nothing is authorized again.
        self.close_scheduling(now, cycle);
        // What the drain did not finish is force-interrupted; a speech that does not report its
        // end by the edge's bound is released by the close. `BTreeMap` order keeps runs stable.
        self.force_cut_open(now, cycle);
        // Release a held pause so no turn during the transcript wait resolves it as false.
        self.cancel_speech_pause(now, cycle);
        // The socket too, settling every `generate_reply()` still awaiting `GenerationCreated`.
        self.realtime_interrupt(cycle);
    }

    pub(crate) fn settle_waiters_at_end(&mut self) {
        for w in self.activity.take_interrupt_waiters() {
            if let Some(done) = w.done {
                let _ = done.send(Ok(()));
            }
        }
        for waiter in self.session.take_idle_waiters() {
            match waiter.done {
                IdleDone::Wait(done) => {
                    let _ = done.send(Err(IdleRefused::Closing));
                }
                IdleDone::Hold { done, .. } => {
                    let _ = done.send(Err(IdleRefused::Closing));
                }
            }
        }
        if let Some(pending) = self.activity.audio_recognition_mut().take_pending_commit() {
            let _ = pending.done.send(String::new());
        }
        // The close ran to its end: whoever still waits on its drain is answered as drained.
        for done in self.session.remaining_close_drains() {
            let _ = done.send(Ok(()));
        }
        // The loop ended with no close to answer them (its queue gone).
        for done in self.activity.hooks_mut().take_drain_waiters() {
            let _ = done.send(Err(crate::agent_session::DrainRefused::Closing));
        }
        if let Some(done) = self.activity.hooks_mut().take_dtmf_done() {
            let _ = done.send(crate::workflows::builtins::dtmf::DtmfOutcome::Stopped);
        }
    }

    /// `_on_error` (activity then session): the `ErrorEvent` for every failed attempt, then the
    /// close policy on unrecoverable ones ([`AgentSession`](crate::agent_session::AgentSession)'s counters).
    pub(crate) fn report_error(
        &mut self,
        now: Ts,
        source: crate::events::ErrorSource,
        recoverable: bool,
        error: lk_agents_abi::errors::ProviderError,
        speech: Option<SpeechId>,
        cycle: &mut Cycle,
    ) {
        let ev = crate::events::ErrorEvent { source, recoverable, error: error.clone(), speech };
        cycle.machine().emit(SessionEvent::Error(Box::new(ev)));
        if recoverable || self.session.is_closing() {
            return;
        }
        if self.session.on_unrecoverable_error(source) {
            self.begin_closing(
                now,
                Close {
                    reason: lk_agents_abi::voice::events::CloseReason::Error,
                    drain: false,
                    error: Some(error),
                },
                cycle,
            );
        }
    }

    /// `Command::Run`: `AgentSession.run()` / `start(capture_run=True)`.
    pub(crate) fn on_run(&mut self, now: Ts, spec: RunSpec, cycle: &mut Cycle) {
        let closing = self.session.is_closing();
        let run = match self.session.runs_mut().open(&spec, closing) {
            Ok(run) => run,
            Err(cause) => {
                cycle.machine().emit(SessionEvent::RunRefused { cause });
                return;
            }
        };
        // Before the reply, so the run's start precedes its items on the host stream.
        cycle.machine().emit(SessionEvent::RunStarted { run });
        if let Some(text) = spec.user_input {
            self.run_reply(now, Some(text), None, cycle);
        }
    }

    /// The run's own `generate_reply(..)`, pipeline or realtime; owned like any session reply.
    fn run_reply(
        &mut self,
        now: Ts,
        user_input: Option<String>,
        instructions: Option<String>,
        cycle: &mut Cycle,
    ) {
        if self.activity.rt_session().is_some() {
            let (waiter, _rx) = tokio::sync::oneshot::channel();
            self.realtime_reply(now, user_input, instructions, None, None, waiter, cycle);
            return;
        }
        let _ = self.generate_reply_speech(
            now,
            user_input,
            instructions,
            None,
            None,
            None,
            None,
            false,
            cycle,
        );
    }

    /// `SpeechCommand::SetRunOutput`: `AgentTask.complete(result)` from inside `speech`'s tool.
    pub(crate) fn on_run_output(&mut self, speech: SpeechId, output: RunOutput, cycle: &mut Cycle) {
        cycle.trace(HandoffStep::RunOutput { speech, ok: matches!(output, RunOutput::Value(_)) });
        self.session.runs_mut().record_output(speech, output);
    }

    pub(crate) fn run_own(&mut self, what: Owned, cycle: &mut Cycle) {
        if let Some(run) = self.session.runs_mut().own(what) {
            cycle.trace(HandoffStep::RunOwned { run, what });
        }
    }

    pub(crate) fn run_release(&mut self, what: Owned, cycle: &mut Cycle) {
        if let Some(run) = self.session.runs_mut().release(what, None) {
            cycle.trace(HandoffStep::RunReleased { run, what });
        }
    }

    /// A speech left the loop (`release_speech`): the run's verdict moves to it.
    pub(crate) fn run_speech_ended(&mut self, handle: &Arc<SpeechHandle>, cycle: &mut Cycle) {
        let what = Owned::Speech(handle.id());
        match self.session.runs_mut().release(what, Some(handle)) {
            Some(run) => cycle.trace(HandoffStep::RunReleased { run, what }),
            None => self.session.runs_mut().forget_speech(handle.id()),
        }
    }

    /// A session-level speech created while a run is open is owned; turn replies, preemptive
    /// generations and server-created generations are not.
    pub(crate) fn run_own_created(
        &mut self,
        speech: SpeechId,
        kind: crate::agent_activity::scheduling::SpeechKind,
        cycle: &mut Cycle,
    ) {
        if kind.session_level() {
            self.run_own(Owned::Speech(speech), cycle);
        }
    }

    /// A `HoldFloor` from an owned speech's tool is the run's.
    pub(crate) fn run_own_hold(&mut self, origin: &Origin, hold: HoldId, cycle: &mut Cycle) {
        let Origin::Tool { call_id } = origin else { return };
        if let Some(speech) = self.activity.tools().speech_of(call_id)
            && self.session.runs().owns(Owned::Speech(speech))
        {
            self.run_own(Owned::Hold(hold), cycle);
        }
    }

    /// The run an item belongs to: the open run for its own speeches and the handoff record. The
    /// `run(user_input=)` user message is not the run's, as in the reference (`RunResult.events`
    /// opens with the reply), and neither is a committed call: a run call became the run's when it
    /// started (`RunItemAdded`); a call answered without running is no run's (generation.py:766).
    pub(crate) fn run_of_item(&self, source: &Source) -> Option<RunId> {
        let open = self.session.runs().id()?;
        match source {
            Source::Reply { speech, .. }
            | Source::Say { speech }
            | Source::ToolOutput { speech, .. }
            | Source::ToolUpdate { speech, .. } => {
                self.session.runs().owns(Owned::Speech(*speech)).then_some(open)
            }
            Source::Handoff => Some(open),
            Source::UserInput { .. }
            | Source::ToolCall { .. }
            | Source::UserTurn { .. }
            | Source::ConfigUpdate
            | Source::HostCtx
            | Source::UserTyped
            | Source::Remote { .. } => None,
        }
    }

    /// An inline task's frame landed: suspend the awaiting speech, the parent's running `on_enter`
    /// and every foreground hold, so the run settles on the task (#6500): a `foreground()` scope
    /// waiting on the task can't end before it either (agent.py:1000-1008).
    pub(crate) fn run_suspend_for_task(
        &mut self,
        speech: Option<SpeechId>,
        cycle: &mut Cycle,
    ) -> Vec<Owned> {
        let mut suspended = Vec::new();
        if let Some(s) = speech
            && let Some(run) = self.session.runs_mut().release(Owned::Speech(s), None)
        {
            cycle.trace(HandoffStep::RunReleased { run, what: Owned::Speech(s) });
            suspended.push(Owned::Speech(s));
        }
        for id in self.activity.hooks().pending_of(HookKind::OnEnter) {
            if let Some(run) = self.session.runs_mut().release(Owned::Hook(id), None) {
                cycle.trace(HandoffStep::RunReleased { run, what: Owned::Hook(id) });
                suspended.push(Owned::Hook(id));
            }
        }
        for hold in self.session.runs().owned_holds() {
            if let Some(run) = self.session.runs_mut().release(Owned::Hold(hold), None) {
                cycle.trace(HandoffStep::RunReleased { run, what: Owned::Hold(hold) });
                suspended.push(Owned::Hold(hold));
            }
        }
        suspended
    }

    /// The task is over: what was suspended is the current run's again, if still live.
    pub(crate) fn run_resume_after_task(&mut self, suspended: Vec<Owned>, cycle: &mut Cycle) {
        for what in suspended {
            let live = match what {
                Owned::Speech(s) => self.activity.scheduling_task().contains(s),
                Owned::Hook(h) => self.activity.hooks().is_pending(h),
                Owned::Hold(h) => self.session.hold_live(h),
                Owned::Handoff(_) => false,
            };
            if live {
                self.run_own(what, cycle);
            }
        }
    }

    /// The settle check after every wake; returns whether a re-prompt was created.
    pub(crate) fn run_check(&mut self, now: Ts, cycle: &mut Cycle) -> bool {
        match self.session.runs_mut().decide() {
            Decided::Open => false,
            Decided::Retry { run, attempt, instructions } => {
                cycle.trace(HandoffStep::RunRetried { run, attempt });
                self.run_reply(now, None, Some(instructions), cycle);
                true
            }
            Decided::Settled { run, outcome, speech } => {
                cycle.machine().emit(SessionEvent::RunSettled { run, outcome, speech });
                false
            }
        }
    }

    /// The close's hard edge: an open run ends `Closed`, before `Close`.
    pub(crate) fn run_close(&mut self, cycle: &mut Cycle) {
        if let Some((run, speech)) = self.session.runs_mut().close() {
            cycle.machine().emit(SessionEvent::RunSettled {
                run,
                outcome: RunOutcome::Closed,
                speech,
            });
        }
    }

    /// The loop ended with a run open: settled `Closed` directly on the stream.
    pub(crate) fn run_close_at_end(&mut self) {
        if let Some((run, speech)) = self.session.runs_mut().close() {
            let ev = SessionEvent::RunSettled { run, outcome: RunOutcome::Closed, speech };
            self.journal(Step::Emitted(ev.clone()));
            let _ = self.events.send(crate::agent_session::HostEvent::Session(ev));
        }
    }

    /// A `Command::Run` never performed: refused as closing, directly on the stream.
    pub(crate) fn refuse_run(&mut self) {
        let ev = SessionEvent::RunRefused { cause: RunRefusal::Closing };
        self.journal(Step::Emitted(ev.clone()));
        let _ = self.events.send(crate::agent_session::HostEvent::Session(ev));
    }
}

/// `RoomOptions`' two session-level switches and the room they apply to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomLinkConfig {
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
    pub room: String,
}

/// Does this disconnect end the session? The linked-participant check lives at the call site.
pub fn closes_session(close_on_disconnect: bool, reason: DisconnectReason) -> bool {
    close_on_disconnect && reason.closes_session_by_default()
}
