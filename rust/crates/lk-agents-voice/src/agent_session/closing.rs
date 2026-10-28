//! `AgentSession._aclose_impl`: the close in flight (`_closing_task`), the step it is at and the
//! `drain()` callers it answers, and the sequence: `on_exit`, drain, hard edge, trailing
//! transcript, legs. Also the commands refused past the edge and the errors that close.

use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::CloseReason;

use super::{HookId, RequestId};

use crate::agent_activity::false_interruption::InterruptOutcome;
use crate::agent_activity::handoff::HandoffEvent;
use crate::agent_activity::hooks::Exit;
use crate::agent_activity::scheduling::{PauseCause, SpeechId};
use crate::agent_session::effects::{Effect, Out, Signal, Timer};
use crate::agent_session::{
    ActivityCommand, Answer, Close, Command, DrainRefused, Driver, DtmfCommand, HandoffRefused,
    IdleRefused, SpeechCommand,
};
use crate::audio::AudioChannel;
use crate::data_plane::{ForwarderCommand, InputCensus};
use crate::events::SessionEvent;
use crate::trace::{
    CloseAct, CloseStep, CoreStep, HandoffStep, InputStep, InterruptCause, InterruptStep,
    SpeechStep, TurnStep,
};
use crate::workflows::dtmf::DtmfOutcome;
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
    pub(crate) error: Option<crate::events::ErrorEvent>,
    pub(crate) phase: ClosePhase,
    /// Host `drain()`s answered by the close's own drain (the reference's drain waits on the
    /// activity lock `_aclose_impl` holds), `Ok` once the edge has settled. `None` once answered:
    /// a later caller is answered at once.
    drains: Option<Vec<RequestId>>,
}

/// Draining, the edge and the legs are bounded by [`Timer::Close`], armed as each begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClosePhase {
    /// Inside `begin_closing`, before the `on_exit` is issued or the drain begins.
    Requested,
    Exiting {
        hook: HookId,
    },
    Draining,
    /// The hard edge was performed; its bound releases the cut speeches that never reported.
    Edge,
    /// Bounded by the trailing commit's own timer.
    Committing,
    /// The realtime socket's `aclose` was asked for; `closed` once it answered or its bound passed.
    ClosingActivity {
        closed: bool,
    },
    Closed,
}

impl ClosePhase {
    /// Past the hard edge no command may take effect any more.
    pub(crate) fn past_edge(self) -> bool {
        !matches!(self, ClosePhase::Requested | ClosePhase::Exiting { .. } | ClosePhase::Draining)
    }
}

impl Driver {
    /// `AgentSession._aclose_impl`: the agent's `on_exit`, the drain, the hard edge, the trailing
    /// transcript, the legs, `close`. A second request joins the close in progress (a drain asked
    /// for again restarts its bound).
    pub(crate) fn begin_closing(&mut self, now: Ts, close: Close, out: &mut Out) {
        let Close { reason, drain, error } = close;
        if let Some(closing) = &self.session.closing {
            if drain && closing.phase == ClosePhase::Draining {
                out.machine().arm(Timer::Close, now.saturating_add(DRAIN_BOUND));
            }
            return;
        }
        out.trace(CloseStep::Requested { reason, drain });
        let mut spans = Vec::new();
        let wall = self.step.wall_secs_at(now);
        let error_type = error.as_ref().map(|e| e.source.as_str());
        self.session.trace.closing(wall, reason.as_str(), drain, error_type, &mut spans);
        spans.into_iter().for_each(|ev| out.machine().emit(ev));
        // The systems (AMD, IVR) close themselves on this, in this instant.
        out.machine().push(Effect::Tap(crate::agent_session::SystemTap::CloseRequested));
        // The keyterm pass: the request in flight is cancelled, no push after this.
        self.keyterms_step(now, crate::keyterm_detection::KeytermIn::Close, out);
        // DTMF collection stop, the first thing the close runs, stops listening: a key pressed
        // during the drain is not collected, a pending debounce never answers.
        self.dtmf_step(crate::workflows::dtmf::DtmfTaskEvent::Stop, out);
        // Every VoiceTask ends first: the agent the close exits is the one they suspended.
        self.close_voice_tasks(now, out);
        let exit = self.plan_exit(out);
        // A host `drain()` in progress is adopted: its `on_exit` and its pause become the close's,
        // and the caller is answered with the close's drain.
        let drains = self.activity.drain.take_waiters();
        let phase = ClosePhase::Requested;
        self.session.closing = Some(Closing { reason, error, phase, drains: Some(drains) });
        // A non-drain close: every open speech force-interrupted regardless of
        // `allow_interruptions`, a false-interruption pause released so a later turn never
        // resolves it as false, and every pending socket request settled `Closing`.
        if !drain {
            // The reference's close runs the session-level `interrupt(force=True)`.
            let generating = self.activity.realtime_state.generating();
            self.force_cut_open(now, out);
            self.cancel_speech_pause(now, out);
            if self.activity.rt_session.is_some() {
                self.realtime_on_close(out);
                self.realtime_interrupt_session(generating, out);
            }
        }
        self.close_on_exit(now, exit, out);
    }

    /// The close moves to `phase`, bounded by `bound` from now; an unbounded phase cancels the
    /// bound of the one it left.
    fn enter_close_phase(
        &mut self,
        phase: ClosePhase,
        bound: Option<Duration>,
        now: Ts,
        out: &mut Out,
    ) {
        if let Some(closing) = &mut self.session.closing {
            closing.phase = phase;
        }
        match bound {
            Some(bound) => out.machine().arm(Timer::Close, now.saturating_add(bound)),
            None => {
                self.disarm(Timer::Close, out);
            }
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
        self.enter_close_phase(ClosePhase::Exiting { hook }, None, now, out);
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
        self.enter_close_phase(ClosePhase::Draining, Some(DRAIN_BOUND), now, out);
        // No reply plans a follow-up once the close began.
        out.machine().push(Effect::Draining);
        self.pause_scheduling(now, PauseCause::Close, out);
    }

    fn force_cut_open(&mut self, now: Ts, out: &mut Out) {
        let open: Vec<SpeechId> = self.activity.speech_scheduler.speech_ids().collect();
        for speech in open {
            match self.interrupt_speech(now, speech, InterruptCause::Closing, true, out) {
                Ok(InterruptOutcome::Now) => {}
                // Already stopped: the journal still carries the close's own step for it.
                Ok(InterruptOutcome::Already) => out.trace(InterruptStep::Interrupted {
                    speech,
                    cause: InterruptCause::Closing,
                    force: true,
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
        if let Some(drains) = self.session.closing.as_mut().and_then(|c| c.drains.as_mut()) {
            out.trace(HandoffStep::DrainStarted { exit_hook });
            drains.extend(waiters);
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
        let waiters =
            self.session.closing.as_mut().and_then(|c| c.drains.take()).unwrap_or_default();
        if waiters.is_empty() {
            return;
        }
        out.trace(HandoffStep::Drained { refused: None });
        for call in waiters {
            out.answer(Some(call), Answer::Drain(Ok(())));
        }
    }

    /// The close moves on when what its phase waits for is over: every speech the edge cut has
    /// reported (or the bound released it), the trailing commit, the legs. Raised on each of those
    /// and on entering a phase, so each phase lands after the effects of the one before.
    pub(crate) fn close_progress(&mut self, now: Ts, out: &mut Out) {
        let Some(Closing { phase, reason, .. }) = &self.session.closing else { return };
        match (*phase, *reason) {
            (ClosePhase::Edge, reason) if self.activity.speech_scheduler.is_empty() => {
                let timers = &self.session.config.effective.timers;
                let timeout = timers.session_close_transcript_timeout();
                if reason != CloseReason::Error && self.transcript_pending(now) {
                    self.close_await_transcript(now, timeout, out);
                } else {
                    self.close_legs_begin(now, out);
                }
            }
            (ClosePhase::Committing, _) if !self.activity.audio_recognition.commit_pending() => {
                self.close_legs_begin(now, out);
            }
            (ClosePhase::ClosingActivity { closed: true }, _) => {
                self.close_legs(now, out);
            }
            (
                ClosePhase::Requested
                | ClosePhase::Exiting { .. }
                | ClosePhase::Draining
                | ClosePhase::Edge
                | ClosePhase::Committing
                | ClosePhase::ClosingActivity { .. }
                | ClosePhase::Closed,
                _,
            ) => return,
        }
        self.pump_schedule(now, out);
    }

    /// Flush the recogniser and wait for its final for `timeout`.
    fn close_await_transcript(&mut self, now: Ts, timeout: Duration, out: &mut Out) {
        self.enter_close_phase(ClosePhase::Committing, None, now, out);
        out.signal(Signal::CloseProgress);
        self.answer_close_drains(out);
        // Nobody awaits it: the commit's `EndOfTurn` and user item are the record.
        self.begin_commit(now, timeout, CLOSE_FLUSH, true, true, None, out);
    }

    /// Tell every leg to close. The realtime socket's `aclose` runs off the loop, bounded by
    /// [`TEARDOWN_BOUND`]; its answer comes back as `RealtimeWake::Closed`.
    fn close_legs_begin(&mut self, now: Ts, out: &mut Out) {
        let rt = self.activity.rt_session.as_ref().map(|s| s.id);
        let phase = ClosePhase::ClosingActivity { closed: rt.is_none() };
        self.enter_close_phase(phase, rt.map(|_| TEARDOWN_BOUND), now, out);
        out.signal(Signal::CloseProgress);
        self.answer_close_drains(out);
        // `send_replace`: `send` stores nothing while no task subscribes, so a later one would
        // never see the close.
        out.machine().push(Effect::Closing);
        // The activity's `aclose()` cancels its tools before it closes the models.
        self.cancel_running_tools(now, out);
        out.machine().close_turn_detector();
        if let Some(rt) = rt {
            out.machine().push(Effect::RealtimeClose(rt));
        }
    }

    /// A close bound passed. `true` when the close moved to a later phase.
    pub(crate) fn close_tick(&mut self, now: Ts, out: &mut Out) -> bool {
        let Some(phase) = self.session.close_phase() else { return false };
        match phase {
            ClosePhase::Draining => {
                self.close_edge(now, out);
                true
            }
            // The edge's bound passed with a cut speech unreported: released by the close.
            ClosePhase::Edge => {
                let open: Vec<SpeechId> = self.activity.speech_scheduler.speech_ids().collect();
                for speech in open {
                    out.trace(CloseStep::Performed(CloseAct::Released { speech }));
                    self.release_speech(now, speech, out);
                }
                false
            }
            // The legs' bound: the session closes without them.
            ClosePhase::ClosingActivity { .. } => {
                if let Some(closing) = &mut self.session.closing {
                    closing.phase = ClosePhase::ClosingActivity { closed: true };
                }
                out.signal(Signal::CloseProgress);
                false
            }
            ClosePhase::Requested
            | ClosePhase::Exiting { .. }
            | ClosePhase::Committing
            | ClosePhase::Closed => false,
        }
    }

    /// The legs closed or their bound passed: the session is closed.
    /// The loop stops here; [`Driver::finish`] sends the last event.
    fn close_legs(&mut self, now: Ts, out: &mut Out) {
        self.enter_close_phase(ClosePhase::Closed, None, now, out);
        // A callback issued past the edge (a cut step's `function_tools_executed`) is abandoned as
        // the edge abandoned the ones before it: once the loop stops, nobody would hear its answer.
        self.abandon_hooks(out);
        self.answer_close_drains(out);
        self.stopped = true;
    }

    /// Whether the recogniser may still owe the last final: the user is speaking, words are held
    /// uncommitted, or speech ended within the last second. Only with an STT leg.
    fn transcript_pending(&self, now: Ts) -> bool {
        if !self.input.sinks.contains_key(&AudioChannel::Stt) {
            return false;
        }
        let turn = &self.activity.audio_recognition;
        // A manual turn is committed whatever it holds: the reference's commit completes it even
        // with no words.
        let manual =
            turn.turn_detection_mode() == crate::audio_recognition::TurnDetectionMode::Manual;
        if manual || turn.user_speaking() || !turn.current_transcript().trim().is_empty() {
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

    /// The close's frame census: per channel, then the shapes those frames had (the microphone's
    /// and the conversion's, then each channel's).
    pub(crate) fn on_input_census(&mut self, census: InputCensus, out: &mut Out) {
        for &(leg, frames, silent, _) in &census.channels {
            out.trace(InputStep::LegFrames { leg, frames, silent });
        }
        let Some((input, converted)) = census.input else { return };
        out.trace(InputStep::InputShape {
            sample_rate: input.sample_rate,
            num_channels: input.num_channels,
            samples_per_channel: input.samples_per_channel,
            converted_rate: converted.sample_rate,
            converted_samples_per_channel: converted.samples_per_channel,
        });
        for (leg, _, _, shape) in census.channels {
            out.trace(InputStep::LegShape {
                leg,
                sample_rate: shape.sample_rate,
                num_channels: shape.num_channels,
                samples_per_channel: shape.samples_per_channel,
            });
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
            // A collection that never starts: answered as stopped, as after a `Stop`.
            Command::Dtmf(DtmfCommand::Start { call, .. }) => {
                out.answer(call, Answer::Dtmf(DtmfOutcome::Stopped));
                return;
            }
            // A drain somebody awaits: the close's drain is the one it waits on.
            Command::Activity(ActivityCommand::Drain { call }) => {
                self.join_close_drain(call.into_iter().collect(), out);
                return;
            }
            // A task somebody awaits: refused as closing; a late cancel has no task left.
            Command::VoiceTask(crate::agent_activity::voice_task::VoiceTaskCommand::Run(run)) => {
                let why = crate::agent_activity::voice_task::VoiceTaskEnded::Closing;
                out.trace(crate::trace::HandoffStep::TaskRefused { why });
                out.answer(Some(run.call), Answer::VoiceTask(Err(why)));
                return;
            }
            Command::VoiceTask(_) => return,
            // A run somebody awaits: refused as closing, on the stream.
            Command::Run(spec) => {
                self.emit_now(SessionEvent::RunRefused {
                    cause: crate::events::RunRefusal::Closing,
                    host: spec.host,
                });
                return;
            }
            // Past the edge nothing is speaking, so an interrupt is `Ok`; a wait is refused.
            Command::Speech(SpeechCommand::Interrupt { call, .. }) => {
                out.answer(call, Answer::Interrupt(Ok(())));
                return;
            }
            Command::Speech(SpeechCommand::WaitForIdle { call, .. }) => {
                out.trace(SpeechStep::IdleAnswered { hold: None, closing: true });
                out.answer(call, Answer::Idle(Err(IdleRefused::Closing)));
                return;
            }
            Command::Speech(SpeechCommand::WaitForIdleAndHold { call }) => {
                out.trace(SpeechStep::IdleAnswered { hold: None, closing: true });
                out.answer(call, Answer::Hold(Err(IdleRefused::Closing)));
                return;
            }
            // A commit the loop never ran: `""`, as the reference answers a closing session.
            Command::Speech(SpeechCommand::CommitUserTurn { call, .. }) => {
                out.trace(TurnStep::CommitAnswered { transcript: String::new() });
                out.answer(call, Answer::Commit(Ok(String::new())));
                return;
            }
            // Nothing awaits these: a dropped one is a lost instruction, not a lost promise.
            Command::Speech(
                SpeechCommand::InterruptSpeech { .. }
                | SpeechCommand::ReleaseIdleHold { .. }
                | SpeechCommand::ClaimUserTurn
                | SpeechCommand::ReleaseUserTurn
                | SpeechCommand::SetAllowInterruptions { .. }
                | SpeechCommand::HoldInterruptions { .. }
                | SpeechCommand::EndCall { .. }
                | SpeechCommand::SetRunOutput { .. }
                | SpeechCommand::ToolUpdate { .. }
                | SpeechCommand::FillerStart { .. }
                | SpeechCommand::FillerStop { .. },
            )
            | Command::Activity(
                ActivityCommand::UpdateInstructions(_)
                | ActivityCommand::UpdateTools(_)
                | ActivityCommand::UpdateExpressive(_)
                | ActivityCommand::SetAudioInput(_)
                | ActivityCommand::SetVideoInput(_)
                | ActivityCommand::SetIoEnabled { .. }
                | ActivityCommand::UpdateChatCtx(_)
                | ActivityCommand::SetAudioOutput(_)
                | ActivityCommand::SetTranscriptionOutput(_)
                | ActivityCommand::UpdateModels { .. },
            )
            | Command::Turn(_)
            | Command::Video(_)
            | Command::Close(_)
            | Command::HookDone { .. }
            | Command::HostMetrics(_)
            | Command::AmdSpan(_)
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
        self.enter_close_phase(ClosePhase::Edge, Some(EDGE_BOUND), now, out);
        out.signal(Signal::CloseProgress);
        let reason =
            self.session.closing.as_ref().map(|c| c.reason).unwrap_or(CloseReason::UserInitiated);
        self.session.on_close_edge();
        // A hold the host has not released ends with the session: past the edge its release is
        // dropped.
        for hold in self.session.idle.take_holds() {
            out.trace(SpeechStep::IdleHoldReleased { hold });
            self.run_release(crate::run_result::Owned::Hold(hold), out);
        }
        // A run still open ends here, `Closed`, before the close's own last event.
        self.run_close(out);
        // Detach the microphone: the recogniser hears only the commit's silence flush.
        out.machine().push(Effect::Forwarder(ForwarderCommand::SetSource(None)));
        // And the recorder: its last flush, its timer cancelled at the edge.
        self.detach_recorder(now, out);
        // `wait_for_idle` raises `ActivityClosedError` on a closing session; the trailing-turn
        // commit below commits what the turn holds.
        self.refuse_idle_waiters(out);
        self.refuse_turn_limit(out);
        // A pending host `commit_user_turn()` is answered below from the close's commit, never
        // with words that were not committed.
        let host_commit_pending = self.activity.audio_recognition.commit_pending();
        // Refuse a handoff in flight; a running hook is left to the host and its wait abandoned.
        self.abandon_handoffs(out);
        // The per-leg frame census, journalled when the data plane answers.
        out.machine().push(Effect::Forwarder(ForwarderCommand::ReportCensus));
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
        for t in [
            Timer::EotPredictionTimeout,
            Timer::EndpointingDelay,
            Timer::TranscriptionTimeout,
            Timer::BackchannelBoundary,
            Timer::FalseInterruption,
            Timer::AecWarmup,
            Timer::UserAway,
            Timer::UserTurnDuration,
        ] {
            if self.timers.is_armed(t) && !out.cancels(t) {
                out.machine().cancel(t);
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
        self.realtime_interrupt(InterruptCause::Closing, out);
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
        for call in self.session.closing.as_mut().and_then(|c| c.drains.take()).unwrap_or_default()
        {
            self.answer(Some(call), Answer::Drain(Ok(())));
        }
        // The loop ended with no close to answer them (its queue gone).
        for call in self.activity.drain.take_waiters() {
            self.answer(Some(call), Answer::Drain(Err(DrainRefused::Closing)));
        }
        let call = self.activity.dtmf.take_call();
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
        let ev = crate::events::ErrorEvent { source, recoverable, error, speech };
        out.machine().emit(SessionEvent::Error(Box::new(ev.clone())));
        if recoverable || self.session.closing.is_some() {
            return;
        }
        if self.session.errors.exhausted_by(source) {
            let close = Close { reason: CloseReason::Error, drain: false, error: Some(ev) };
            self.begin_closing(now, close, out);
        }
    }
}
