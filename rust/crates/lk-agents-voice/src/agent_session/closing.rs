//! `AgentSession._aclose_impl`: the close in flight (`_closing_task`), the step it is at and the
//! `drain()` callers it answers, and the sequence: `on_exit`, drain, hard edge, trailing
//! transcript, legs. Also the commands refused past the edge and the errors that close.

use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::CloseReason;

use super::{HookId, RequestId};

use crate::agent_activity::false_interruption::InterruptOutcome;
use crate::agent_activity::handoff::HandoffEvent;
use crate::agent_activity::hooks::Exit;
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::effects::{Timer, VoiceEffect};
use crate::agent_session::{
    ActivityCommand, Answer, Close, Command, DrainRefused, DtmfCommand, HandoffRefused,
    IdleRefused, SpeechCommand, VoiceDriver,
};
use crate::audio::AudioChannel;
use crate::data_plane::{AudioInputRouterCommand, InputCensus};
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

impl VoiceDriver {
    /// `AgentSession._aclose_impl`: the agent's `on_exit`, the drain, the hard edge, the trailing
    /// transcript, the legs, `close`. A second request joins the close in progress (a drain asked
    /// for again restarts its bound).
    pub(crate) fn begin_closing(&mut self, now: Ts, close: Close) {
        let Close { reason, drain, error } = close;
        if let Some(closing) = &self.session.closing {
            if drain && closing.phase == ClosePhase::Draining {
                self.set_deadline(Timer::Close, now.saturating_add(HOST_TEARDOWN_TIMEOUT));
            }
            return;
        }
        self.record_trace(CloseStep::Requested { reason, drain });
        self.session.spans.closing(now, reason, drain, error.as_ref());
        // The systems (AMD, IVR) close themselves on this, in this instant.
        self.record_effect(VoiceEffect::Tap(crate::agent_session::SystemTap::CloseRequested));
        // The keyterm pass: the request in flight is cancelled, no push after this.
        self.keyterms_step(now, crate::keyterm_detection::KeytermIn::Close);
        // DTMF collection stop, the first thing the close runs, stops listening: a key pressed
        // during the drain is not collected, a pending debounce never answers.
        self.dtmf_step(crate::workflows::dtmf::DtmfTaskEvent::Stop);
        // Every VoiceTask ends first: the agent the close exits is the one they suspended.
        self.close_voice_tasks(now);
        let exit = self.plan_exit();
        // A host `drain()` in progress is adopted: its `on_exit` and its pause become the close's,
        // and the caller is answered with the close's drain.
        let drains = self.activity.take_drain_waiters();
        let phase = ClosePhase::Requested;
        self.session.closing = Some(Closing { reason, error, phase, drains: Some(drains) });
        // A non-drain close: every open speech force-interrupted regardless of
        // `allow_interruptions`, a false-interruption pause released so a later turn never
        // resolves it as false, and every pending socket request settled `Closing`.
        if !drain {
            // The reference's close runs the session-level `interrupt(force=True)`.
            let generating = self.activity.realtime_generating();
            self.force_cut_open(now);
            self.cancel_speech_pause(now);
            if self.activity.realtime().is_some() {
                self.realtime_on_close();
                self.realtime_interrupt_session(generating);
            }
        }
        self.close_on_exit(now, exit);
    }

    /// The close moves to `phase`, bounded by `bound` from now; an unbounded phase cancels the
    /// bound of the one it left.
    fn enter_close_phase(&mut self, phase: ClosePhase, bound: Option<Duration>, now: Ts) {
        if let Some(closing) = &mut self.session.closing {
            closing.phase = phase;
        }
        match bound {
            Some(bound) => self.set_deadline(Timer::Close, now.saturating_add(bound)),
            None => {
                self.clear_deadline_if_set(Timer::Close);
            }
        }
    }

    /// Pending handoffs are refused; the agent's `on_exit` runs once: lent by a handoff that
    /// already issued it, skipped if it ran, otherwise issued here.
    fn plan_exit(&mut self) -> Exit {
        self.handoff_step(HandoffEvent::Closing);
        let exit = self.activity.claim_exit();
        self.issue_claimed_exits();
        exit
    }

    /// The remaining agent's `on_exit`, then the drain.
    fn close_on_exit(&mut self, now: Ts, exit: Exit) {
        let Some(hook) = exit.hook() else { return self.close_drain(now) };
        self.enter_close_phase(ClosePhase::Exiting { hook }, None, now);
    }

    /// The `on_exit` hook `id` ended (answered, timed out, or with nobody to ask).
    pub(crate) fn close_exit_done(&mut self, now: Ts, id: HookId) {
        if self.session.close_phase() == Some(ClosePhase::Exiting { hook: id }) {
            self.close_drain(now);
        }
    }

    /// `PauseScheduling { cause: Close }`, bounded by [`HOST_TEARDOWN_TIMEOUT`]; settled or refused, the hard
    /// edge follows.
    fn close_drain(&mut self, now: Ts) {
        self.enter_close_phase(ClosePhase::Draining, Some(HOST_TEARDOWN_TIMEOUT), now);
        self.pause_scheduling_for_close(now);
    }

    fn force_cut_open(&mut self, now: Ts) {
        let open: Vec<SpeechId> = self.activity.speeches().collect();
        for speech in open {
            match self.interrupt_speech(now, speech, InterruptCause::Closing, true) {
                Ok(InterruptOutcome::Now) => {}
                // Already stopped: the journal still carries the close's own step for it.
                Ok(InterruptOutcome::Already) => self.record_trace(InterruptStep::Interrupted {
                    speech,
                    cause: InterruptCause::Closing,
                    force: true,
                }),
                // Unreachable: `force` overrides the hold. Said rather than swallowed.
                Err(_) => {
                    self.record_trace(CoreStep::PortFailure("a forced close interrupt was refused"))
                }
            }
        }
    }

    pub(crate) fn join_close_drain(&mut self, waiters: Vec<RequestId>) {
        if waiters.is_empty() {
            return;
        }
        let exit_hook = self.activity.exiting_hook();
        if let Some(drains) = self.session.closing.as_mut().and_then(|c| c.drains.as_mut()) {
            drains.extend(waiters);
            self.record_trace(HandoffStep::DrainStarted { exit_hook });
        } else {
            // Answered already, or no close is running (the loop is ending).
            let refused = self.session.closing.is_none().then_some(DrainRefused::Closing);
            for call in waiters {
                self.record_trace(HandoffStep::Drained { refused });
                self.answer(Some(call), Answer::Drain(refused.map_or(Ok(()), Err)));
            }
        }
    }

    fn answer_close_drains(&mut self) {
        let waiters =
            self.session.closing.as_mut().and_then(|c| c.drains.take()).unwrap_or_default();
        if waiters.is_empty() {
            return;
        }
        self.record_trace(HandoffStep::Drained { refused: None });
        for call in waiters {
            self.answer(Some(call), Answer::Drain(Ok(())));
        }
    }

    /// The close moves on when what its phase waits for is over: every speech the edge cut has
    /// reported (or the bound released it), the trailing commit, the legs. Raised on each of those
    /// and on entering a phase, so each phase lands after the effects of the one before.
    pub(crate) fn advance_close(&mut self, now: Ts) {
        let Some(Closing { phase, reason, .. }) = &self.session.closing else { return };
        match (*phase, *reason) {
            (ClosePhase::Draining, _) if self.activity.speech_queue_empty() => {
                self.close_edge(now);
            }
            (ClosePhase::Edge, reason) if self.activity.speech_queue_empty() => {
                let timers = &self.session.config.effective.timers;
                let timeout = timers.session_close_transcript_timeout();
                if reason != CloseReason::Error && self.transcript_pending(now) {
                    self.close_await_transcript(now, timeout);
                } else {
                    self.close_legs_begin(now);
                }
            }
            (ClosePhase::Committing, _) if !self.activity.has_pending_commit() => {
                self.close_legs_begin(now);
            }
            (ClosePhase::ClosingActivity { closed: true }, _) => {
                self.close_legs(now);
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
            ) => (),
        }
    }

    /// Flush the recogniser and wait for its final for `timeout`.
    fn close_await_transcript(&mut self, now: Ts, timeout: Duration) {
        self.enter_close_phase(ClosePhase::Committing, None, now);

        self.answer_close_drains();
        // Nobody awaits it: the commit's `EndOfTurn` and user item are the record.
        self.begin_commit(now, timeout, CLOSE_FLUSH, true, true, None);
    }

    /// Tell every leg to close. The realtime socket's `aclose` runs off the loop, bounded by
    /// [`HOST_TEARDOWN_TIMEOUT`]; its answer comes back as `RealtimeWake::Closed`.
    fn close_legs_begin(&mut self, now: Ts) {
        let rt = self.activity.realtime().map(|s| s.id);
        let phase = ClosePhase::ClosingActivity { closed: rt.is_none() };
        self.enter_close_phase(phase, rt.map(|_| HOST_TEARDOWN_TIMEOUT), now);

        self.answer_close_drains();
        // `send_replace`: `send` stores nothing while no task subscribes, so a later one would
        // never see the close.
        self.record_effect(VoiceEffect::Closing);
        // The activity's `aclose()` cancels its tools before it closes the models.
        self.cancel_running_tools(now);
        self.record_effect(VoiceEffect::EotClose);
        if let Some(rt) = rt {
            self.record_effect(VoiceEffect::CloseRealtimeSession(rt));
        }
    }

    /// A close bound passed. `true` when the close moved to a later phase.
    pub(crate) fn close_tick(&mut self, now: Ts) -> bool {
        let Some(phase) = self.session.close_phase() else { return false };
        match phase {
            ClosePhase::Draining => {
                self.close_edge(now);
                true
            }
            // The edge's bound passed with a cut speech unreported: released by the close.
            ClosePhase::Edge => {
                let open: Vec<SpeechId> = self.activity.speeches().collect();
                for speech in open {
                    self.record_trace(CloseStep::Performed(CloseAct::Released { speech }));
                    self.release_speech(now, speech);
                }
                false
            }
            // The legs' bound: the session closes without them.
            ClosePhase::ClosingActivity { .. } => {
                if let Some(closing) = &mut self.session.closing {
                    closing.phase = ClosePhase::ClosingActivity { closed: true };
                }

                false
            }
            ClosePhase::Requested
            | ClosePhase::Exiting { .. }
            | ClosePhase::Committing
            | ClosePhase::Closed => false,
        }
    }

    /// The legs closed or their bound passed: the session is closed.
    /// The loop stops here; [`VoiceDriver::finish`] sends the last event.
    fn close_legs(&mut self, now: Ts) {
        self.enter_close_phase(ClosePhase::Closed, None, now);
        // A callback issued past the edge (a cut step's `function_tools_executed`) is abandoned as
        // the edge abandoned the ones before it: once the loop stops, nobody would hear its answer.
        self.abandon_hooks();
        self.answer_close_drains();
        self.stopped = true;
    }

    /// Whether the recogniser may still owe the last final: the user is speaking, words are held
    /// uncommitted, or speech ended within the last second. Only with an STT leg.
    fn transcript_pending(&self, now: Ts) -> bool {
        if !self.input.sinks.contains_key(&AudioChannel::Stt) {
            return false;
        }
        self.activity.transcript_pending(now)
    }

    /// The close's frame census: per channel, then the shapes those frames had (the microphone's
    /// and the conversion's, then each channel's).
    pub(crate) fn on_input_census(&mut self, census: InputCensus) {
        for &(leg, frames, silent, _) in &census.channels {
            self.record_trace(InputStep::LegFrames { leg, frames, silent });
        }
        let Some((input, converted)) = census.input else { return };
        self.record_trace(InputStep::InputShape {
            sample_rate: input.sample_rate,
            num_channels: input.num_channels,
            samples_per_channel: input.samples_per_channel,
            converted_rate: converted.sample_rate,
            converted_samples_per_channel: converted.samples_per_channel,
        });
        for (leg, _, _, shape) in census.channels {
            self.record_trace(InputStep::LegShape {
                leg,
                sample_rate: shape.sample_rate,
                num_channels: shape.num_channels,
                samples_per_channel: shape.samples_per_channel,
            });
        }
    }

    /// A command past the hard edge: its promise is answered, nothing takes effect.
    pub(crate) fn refuse_command(&mut self, command: Command) {
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
                self.refuse_handoff(None, update, HandoffRefused::Closing);
                return;
            }
            // A collection that never starts: answered as stopped.
            Command::Dtmf(DtmfCommand::Start { call, .. }) => {
                self.answer(call, Answer::Dtmf(DtmfOutcome::Stopped));
                return;
            }
            // A drain somebody awaits: the close's drain is the one it waits on.
            Command::Activity(ActivityCommand::Drain { call }) => {
                self.join_close_drain(call.into_iter().collect());
                return;
            }
            // A task somebody awaits: refused as closing; a late cancel has no task left.
            Command::VoiceTask(crate::agent_activity::voice_task::VoiceTaskCommand::Run(run)) => {
                let why = crate::agent_activity::voice_task::VoiceTaskEnded::Closing;
                self.record_trace(crate::trace::HandoffStep::TaskRefused { why: why.clone() });
                self.answer(Some(run.call), Answer::VoiceTask(Err(why)));
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
                self.answer(call, Answer::Interrupt(Ok(())));
                return;
            }
            Command::Speech(SpeechCommand::WaitForIdle { call, .. }) => {
                self.record_trace(SpeechStep::IdleAnswered { hold: None, closing: true });
                self.answer(call, Answer::Idle(Err(IdleRefused::Closing)));
                return;
            }
            Command::Speech(SpeechCommand::WaitForIdleAndHold { call }) => {
                self.record_trace(SpeechStep::IdleAnswered { hold: None, closing: true });
                self.answer(call, Answer::Hold(Err(IdleRefused::Closing)));
                return;
            }
            // A commit the loop never ran: `""`, as the reference answers a closing session.
            Command::Speech(SpeechCommand::CommitUserTurn { call, .. }) => {
                self.record_trace(TurnStep::CommitAnswered { transcript: String::new() });
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
                | SpeechCommand::HoldInterruptions { .. }
                | SpeechCommand::EndCall { .. }
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
                | ActivityCommand::UpdateModels(_),
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
            | Command::Dtmf(DtmfCommand::Cancel { .. })
            | Command::Speech(
                SpeechCommand::PauseAuthorization | SpeechCommand::ResumeAuthorization,
            ) => return,
        };
        self.record_trace(CoreStep::PortFailure(lost));
        self.settle_undelivered(what, call);
    }

    /// The hard edge: scheduling stopped, timers cancelled, countdown committed or torn down, what
    /// is open interrupted; after it only the close's own steps happen. The close moves on once
    /// every cut speech has reported `SpeechDone`, as the reference awaits `current_speech`, or
    /// [`HOST_TEARDOWN_TIMEOUT`] releases them.
    pub(crate) fn close_edge(&mut self, now: Ts) {
        self.enter_close_phase(ClosePhase::Edge, Some(HOST_TEARDOWN_TIMEOUT), now);

        let reason =
            self.session.closing.as_ref().map(|c| c.reason).unwrap_or(CloseReason::UserInitiated);
        self.session.on_close_edge();
        // A hold the host has not released ends with the session: past the edge its release is
        // dropped.
        for hold in self.session.idle.take_holds() {
            self.record_trace(SpeechStep::IdleHoldReleased { hold });
        }
        // A run still open ends here, `Closed`, before the close's own last event.
        self.run_close();
        // Detach the microphone: the recogniser hears only the commit's silence flush.
        self.record_effect(VoiceEffect::UpdateAudioInputRouting(
            AudioInputRouterCommand::SetSource(None),
        ));
        // And the recorder: its last flush, its timer cancelled at the edge.
        self.detach_recorder(now);
        // `wait_for_idle` raises `ActivityClosedError` on a closing session; the trailing-turn
        // commit below commits what the turn holds.
        self.refuse_idle_waiters();
        if let Some(reason) = self.activity.refuse_turn_limit() {
            self.record_trace(crate::trace::TurnStep::TurnLimitSkipped(reason));
        }
        // A pending host `commit_user_turn()` is answered below from the close's commit, never
        // with words that were not committed.
        let host_commit_pending = self.activity.has_pending_commit();
        // Refuse a handoff in flight; a running hook is left to the host and its wait abandoned.
        self.abandon_handoffs();
        // The per-leg frame census, journalled when the data plane answers.
        self.record_effect(VoiceEffect::UpdateAudioInputRouting(
            AudioInputRouterCommand::ReportCensus,
        ));
        // Cut a running endpointing delay short, unless the close is an error's: that commits no
        // trailing transcript, so the countdown is torn down.
        let effects = self.activity.close_recognition(now, reason != CloseReason::Error);
        self.apply_recognition_effects(effects);
        if host_commit_pending {
            self.perform_pending_commit(now);
        }
        // An in-flight prediction answers a turn nobody will decide: end its task.
        self.record_effect(VoiceEffect::EotClose);
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
            if self.deadline(t).is_some() {
                self.clear_deadline(t);
            }
        }
        // Watchdogs go with their speeches, the current speech is released, a pending preemptive
        // generation discarded, a drain-waiting pause refused, and nothing is authorized again.
        self.close_scheduling(now);
        // What the drain did not finish is force-interrupted; a speech that does not report its
        // end by the edge's bound is released by the close. `BTreeMap` order keeps runs stable.
        self.force_cut_open(now);
        // Release a held pause so no turn during the transcript wait resolves it as false.
        self.cancel_speech_pause(now);
        // The socket too, settling every `generate_reply()` still awaiting `GenerationCreated`.
        self.realtime_interrupt(InterruptCause::Closing);
    }

    pub(crate) fn settle_waiters_at_end(&mut self) {
        let effects = self.activity.finish_interrupts();
        for effect in effects {
            self.record_effect(effect);
        }
        for waiter in self.session.idle.take_waiters() {
            self.answer(waiter.call, waiter.answer(Err(IdleRefused::Closing)));
        }
        if let Some(effect) = self.activity.finish_pending_commit() {
            self.record_effect(effect);
        }
        // The close ran to its end: whoever still waits on its drain is answered as drained.
        for call in self.session.closing.as_mut().and_then(|c| c.drains.take()).unwrap_or_default()
        {
            self.answer(Some(call), Answer::Drain(Ok(())));
        }
        let effects = self.activity.finish_drain_and_dtmf();
        for effect in effects {
            self.record_effect(effect);
        }
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
    ) {
        let ev = crate::events::ErrorEvent::provider(source, recoverable, error, speech);
        self.publish_event(SessionEvent::Error(Box::new(ev.clone())));
        if recoverable || self.session.closing.is_some() {
            return;
        }
        if self.session.errors.exhausted_by(source) {
            let close = Close { reason: CloseReason::Error, drain: false, error: Some(ev) };
            self.begin_closing(now, close);
        }
    }
}
