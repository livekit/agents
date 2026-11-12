use crate::agent_session::effects::{AudioOp, VoiceEffect};
use crate::agent_session::effects::{GateCause, GateOp, Timer};
use crate::agent_session::machine::activity::scheduling::{PreemptiveDiscard, SpeechId, Status};
use crate::agent_session::options::AgentSessionOptions;
use crate::agent_session::{Answer, RequestId, VoiceCommandOrigin, VoiceDriver};
use crate::audio_recognition::TurnDetectionMode;
use crate::events::SessionEvent;
use crate::speech_handle::SpeechError;
use crate::trace::{InputStep, InterruptCause, InterruptStep, SpeechStep};
use crate::turn::InterruptionOptions;
use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_core::tokenize::spoken_words;
use std::time::Duration;

pub use paused_speech::PausedSpeech;

mod paused_speech {
    use super::*;

    /// The false-interruption pause, its timer, and the sink facts that permit pausing.
    #[derive(Debug, Clone)]
    pub struct PausedSpeech {
        interruption: InterruptionOptions,
        /// As of the last `refresh_pause_facts`.
        sink: AudioOutputCapabilities,
        /// An audio output exists and is enabled, as of the last `refresh_pause_facts`.
        audio_enabled: bool,
        paused_speech: Option<PausedSpeechInfo>,
        /// Tracked so a cancel is issued only when armed.
        false_interruption_timer: bool,
        /// The timeout elapsed with a turn decision open, so the turn owns the verdict.
        false_interruption_pending: bool,
    }

    impl PausedSpeech {
        /// A session with no sink passes `AudioOutputCapabilities::default()` and `false`.
        pub fn new(
            cfg: &AgentSessionOptions,
            sink: AudioOutputCapabilities,
            audio_enabled: bool,
        ) -> Self {
            PausedSpeech {
                interruption: cfg.turn.interruption,
                sink,
                audio_enabled,
                paused_speech: None,
                false_interruption_timer: false,
                false_interruption_pending: false,
            }
        }

        /// The one predicate for pausing and resuming, with its timeout; `None` is pausing off.
        pub fn pause_timeout(&self) -> Option<Duration> {
            let i = &self.interruption;
            let enabled = i.resume_false_interruption() && self.audio_enabled && self.sink.pause;
            i.false_interruption_timeout().filter(|_| enabled)
        }

        pub fn paused_handle(&self) -> Option<SpeechId> {
            self.paused_speech.map(|p| p.handle)
        }

        /// Whether the timer was armed (its cancel is the caller's); the pending flag clears, the
        /// pause itself stays.
        pub fn cancel_false_interruption_timer(&mut self) -> bool {
            let armed = std::mem::take(&mut self.false_interruption_timer);
            self.false_interruption_pending = false;
            armed
        }

        /// The paused speech's resume timeout, captured at pause time.
        pub fn paused_timeout(&self) -> Option<Duration> {
            self.paused_speech.map(|p| p.timeout)
        }

        /// The caller armed `Timer::FalseInterruption`.
        pub fn false_interruption_timer_armed(&mut self) {
            self.false_interruption_timer = true;
        }

        /// `true` when the interruption is known to be false now; an open turn decision owns the
        /// verdict instead.
        pub fn false_interruption_timed_out(&mut self, turn_open: bool) -> bool {
            if self.paused_speech.is_none() || self.false_interruption_pending {
                return false;
            }
            self.false_interruption_timer = false;
            if turn_open {
                self.false_interruption_pending = true;
                return false;
            }
            true
        }

        /// `true` when the pending verdict resolves now: the turn was decided or torn down.
        pub fn turn_settled(&mut self, settled: TurnSettled) -> bool {
            if !self.false_interruption_pending {
                return false;
            }
            match settled {
                TurnSettled::Superseded => false,
                TurnSettled::TornDown | TurnSettled::Decided => {
                    self.false_interruption_pending = false;
                    true
                }
            }
        }

        pub fn take_paused(&mut self) -> Option<PausedSpeechInfo> {
            self.paused_speech.take()
        }

        pub fn audio_enabled(&self) -> bool {
            self.audio_enabled
        }

        /// A change that no longer permits the pause releases it, returning the released speech.
        pub fn refresh_facts(
            &mut self,
            sink: AudioOutputCapabilities,
            audio_enabled: bool,
        ) -> Option<SpeechId> {
            if self.sink == sink && self.audio_enabled == audio_enabled {
                return None;
            }
            (self.sink, self.audio_enabled) = (sink, audio_enabled);
            let paused = self.paused_speech.filter(|_| self.pause_timeout().is_none())?;
            self.paused_speech = None;
            Some(paused.handle)
        }

        /// Re-pausing the same handle updates only the timeout.
        pub fn update_paused_speech(
            &mut self,
            handle: SpeechId,
            agent_state: AgentState,
            timeout: Duration,
        ) {
            match &mut self.paused_speech {
                Some(paused) if paused.handle == handle => paused.timeout = timeout,
                slot => *slot = Some(PausedSpeechInfo { handle, agent_state, timeout }),
            }
        }
    }
}

/// What an interruption did. A refusal is journalled as `ByAudioActivityRefused` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterruptReport {
    pub speech: SpeechId,
    pub trigger: AudioInterruptTrigger,
    /// The playout was paused, nothing was interrupted.
    pub paused: bool,
    /// The handle was told to stop, and whether it landed *now*; `None` for a pause or a refusal.
    pub interrupted: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptOutcome {
    /// Interrupted now; a speech in the sink has its buffer cleared.
    Now,
    /// Already interrupted, already finished, or already released: nothing to do, nothing said.
    Already,
}

pub use lk_agents_abi::turn::interruption::InterruptionPort;

pub(crate) struct InterruptWaiter {
    pub(crate) speeches: Vec<SpeechId>,
    pub(crate) call: Option<RequestId>,
}

/// One variant per path, with the measurement its gate reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum AudioInterruptTrigger {
    /// A local VAD `inference_done`: `duration` against `min_interruption_duration`; `raw_silence`
    /// is the "VAD speech still ongoing" half of the own-turn-tail gate. Clamped by the caller.
    VadThreshold { duration: Duration, raw_silence: Duration },
    /// An interim transcript with **no VAD**: its first words are the onset. Empty is no onset.
    SttInterim { words: usize },
    /// A final transcript: the failsafe for a VAD that missed the segment.
    SttFinal { words: usize },
    /// The adaptive detector ruled an interruption. The instant from which the user's words over
    /// the agent are not replayed into the next turn.
    DetectorVerdict { ignore_user_transcript_until: Ts },
}

/// Which guard refused, so "interruption did nothing" is debuggable; journalled as `ByAudioActivityRefused`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, lk_agents_abi::Vocabulary)]
pub enum InterruptRefusal {
    /// Turn detection is `manual` or `realtime_llm`: no local audio activity may stop the agent.
    ManualTurnDetection,
    /// The VAD segment is shorter than `min_interruption_duration`; a cough.
    TooShort,
    /// `stt` detection, the STT ended the turn, and the VAD accumulates silence: the turn's tail.
    OwnTurnTail,
    /// InterruptByAudioActivity-in by audio activity is off, or taken over by the adaptive detector.
    Disabled,
    /// AEC has not settled, so the "user speech" may be the agent's own echo.
    AecWarmup,
    /// The realtime model owns turn taking.
    RealtimeTurnDetection,
    /// Fewer than `min_words` transcribed this turn, or the asking event carried no words.
    TooFewWords,
    NoCurrentSpeech,
    /// The current speech is already interrupted.
    AlreadyInterrupted,
    NotAllowed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PausedSpeechInfo {
    pub handle: SpeechId,
    /// The agent state at the **first** pause of this handle; a re-pause updates only the timeout.
    pub agent_state: AgentState,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnSettled {
    /// A newer decision superseded this one; keep waiting.
    Superseded,
    /// Torn down instead of decided.
    TornDown,
    Decided,
}

impl VoiceDriver {
    /// `session.interrupt()`, or the AMD's verdict: a detector's cut, not a host interrupt, so
    /// it records no `HostInterrupt` step.
    pub(crate) fn host_interrupt_all(&mut self, now: Ts, force: bool, call: Option<RequestId>) {
        let cause = match self.command_origin {
            VoiceCommandOrigin::Amd => InterruptCause::Detector,
            VoiceCommandOrigin::Host
            | VoiceCommandOrigin::Tool { .. }
            | VoiceCommandOrigin::Hook { .. }
            | VoiceCommandOrigin::Ivr => InterruptCause::Host,
        };
        if cause == InterruptCause::Host {
            self.record_trace(InterruptStep::HostInterrupt { speech: None, force });
        }
        match self.interrupt_all(now, force, cause) {
            Ok(speeches) => {
                self.activity.interrupt_waiters.push(InterruptWaiter { speeches, call })
            }
            // Refused before the socket and the queue are touched.
            Err(refused) => self.answer(call, Answer::Interrupt(Err(refused))),
        }
    }

    /// `SpeechHandle.interrupt()`: the whole speech, wherever it is: queued, playing, or in its
    /// tools (which finish and commit, with no follow-up).
    pub(crate) fn host_interrupt_speech(&mut self, now: Ts, speech: SpeechId, force: bool) {
        self.record_trace(InterruptStep::HostInterrupt { speech: Some(speech), force });
        let result = if force {
            Ok(self.force_interrupt_speech(now, speech, InterruptCause::Host))
        } else {
            self.interrupt_speech(now, speech, InterruptCause::Host)
        };
        let tell_socket = match result {
            // The socket too, once, for a pipeline speech on a realtime session.
            Ok(InterruptOutcome::Now) => !self.activity.is_realtime_speech(speech),
            // A no-op, so another speech's live generation is not cancelled under it.
            Ok(InterruptOutcome::Already) => false,
            Err(_) => {
                self.record_trace(InterruptStep::HostInterruptRefused { speech });
                if let Some(state) = self.activity.speech_scheduler.speech_mut(speech) {
                    state.interrupt_refused();
                }
                return;
            }
        };
        if tell_socket {
            self.realtime_interrupt(InterruptCause::Host);
        }
        self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned);
    }

    /// `disallow_interruptions()` and its release. A done handle refuses quietly; an unknown id is
    /// a no-op.
    pub(crate) fn hold_interruptions(&mut self, speech: SpeechId, hold: bool) {
        let Some(state) = self.activity.speech_scheduler.speech_mut(speech) else {
            return;
        };
        if hold {
            if state.hold_interruptions().is_err() {
                return;
            }
        } else {
            state.release_interruptions();
        }
        let allow = state.allow_interruptions();
        self.record_trace(InterruptStep::InterruptionsToggled { speech, allow });
    }

    /// `SpeechHandle.allow_interruptions = ..`: refused quietly on a done handle.
    pub(crate) fn set_allow_interruptions(&mut self, speech: SpeechId, allow: bool) {
        let Some(state) = self.activity.speech_scheduler.speech_mut(speech) else {
            return;
        };
        if state.set_allow_interruptions(allow).is_err() {
            return;
        }
        let allow = state.allow_interruptions();
        self.record_trace(InterruptStep::InterruptionsToggled { speech, allow });
    }

    /// The only writer of a speech's interruption. `Err` when forbidden and not forced; an ended
    /// speech is `Ok(Already)`. A cut playout cancels its tools, a speech in its tools lets them
    /// finish and commit, and no follow-up is spoken.
    pub(crate) fn interrupt_speech(
        &mut self,
        now: Ts,
        speech: SpeechId,
        cause: InterruptCause,
    ) -> Result<InterruptOutcome, SpeechError> {
        let Some(state) = self.activity.speech_scheduler.speech_mut(speech) else {
            return Ok(InterruptOutcome::Already);
        };
        if !state.interrupt()? {
            return Ok(InterruptOutcome::Already);
        }
        Ok(self.speech_interrupted(now, speech, cause, false))
    }

    pub(crate) fn force_interrupt_speech(
        &mut self,
        now: Ts,
        speech: SpeechId,
        cause: InterruptCause,
    ) -> InterruptOutcome {
        let Some(state) = self.activity.speech_scheduler.speech_mut(speech) else {
            return InterruptOutcome::Already;
        };
        if !state.force_interrupt() {
            return InterruptOutcome::Already;
        }
        self.speech_interrupted(now, speech, cause, true)
    }

    fn speech_interrupted(
        &mut self,
        now: Ts,
        speech: SpeechId,
        cause: InterruptCause,
        force: bool,
    ) -> InterruptOutcome {
        // Only a speech whose audio is in the sink has anything to clear; clearing for a queued
        // speech would cut the current one.
        let in_the_sink = self.activity.speech_scheduler.current_speech() == Some(speech)
            || self.activity.paused_speech.paused_handle() == Some(speech);
        let sink = self.session.output_bindings.active_speech_audio(speech).filter(|_| in_the_sink);
        self.record_trace(InterruptStep::Interrupted { speech, cause, force });
        if let Some(turn) = self.agent_turn(speech) {
            turn.interrupted(cause);
        }
        self.fillers_speech_ended(speech);
        // A tool's idle ask is refused when the cut lands during playout, or it would wait on the
        // speech itself.
        if !self.activity.tools_in_flight.contains(&speech) {
            self.withdraw_idle_asks(speech);
        }
        if let Some(output) = sink {
            let (op, cause) = (GateOp::ClearBuffer, GateCause::Interrupted);
            self.record_effect(VoiceEffect::Audio(AudioOp::Gate { speech, op, cause, output }));
        }
        let realtime = self.activity.is_realtime_speech(speech);
        // No watchdog after the close: every handle is interrupted there and the loop exits.
        let closing = self.session.closing_past_edge();
        let watchdog = Timer::InterruptionTimeout(speech);
        if !closing && self.deadline(watchdog).is_none() {
            let at = now.saturating_add(HOST_TEARDOWN_TIMEOUT);
            self.set_deadline(watchdog, at);
            self.record_trace(SpeechStep::WatchdogArmed { speech, at });
        }
        // A realtime server keeps generating (into its own mirror) until told not to.
        if realtime {
            self.realtime_interrupt(cause);
        }
        InterruptOutcome::Now
    }

    /// The one place the loop moves the playout sink. No sink, no op.
    fn gate_op(&mut self, speech: SpeechId, op: GateOp, cause: GateCause) {
        if let Some(output) = self
            .session
            .output_bindings
            .speech_outputs(speech)
            .and_then(|outputs| outputs.audio.as_ref())
            .map(|a| a.id)
        {
            self.record_effect(VoiceEffect::Audio(AudioOp::Gate { speech, op, cause, output }));
        }
    }

    /// Refused for a handle already interrupted or never interruptible; a hold taken since is
    /// kept.
    fn interrupt_paused(&mut self, now: Ts, speech: SpeechId) {
        let Some(state) = self.activity.speech_scheduler.speech(speech) else { return };
        if !state.interruptible() {
            return;
        }
        if self.interrupt_speech(now, speech, InterruptCause::UserTurn).is_err() {
            self.record_trace(InterruptStep::ByAudioActivityRefused(InterruptRefusal::NotAllowed));
        }
    }

    /// Guard 1 back to the session's option, not `true`: a manual or realtime-detected session had
    /// it off.
    pub(crate) fn restore_audio_activity_guard(&mut self) {
        self.session.by_audio_activity =
            self.session.config.effective.turn.interruption.by_audio_activity();
    }

    /// A leg failure hands interruption back to the VAD and never errors; an unrecoverable one
    /// also stops the sentinels.
    pub(crate) fn on_overlap_verdict(
        &mut self,
        _now: Ts,
        ev: lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict>,
    ) {
        use lk_agents_abi::turn::detector::LegEvent;
        use lk_agents_abi::voice::events::Leg;
        let (req, v) = match ev {
            LegEvent::Verdict { req, verdict } => (req, verdict),
            LegEvent::Error { recoverable, .. } => {
                self.record_trace(InputStep::LegFailed { leg: Leg::Interruption, recoverable });
                self.restore_audio_activity_guard();
                if !recoverable {
                    self.activity.audio_recognition.set_interruption_active(false);
                }
                return;
            }
            LegEvent::Closed => {
                self.record_trace(InputStep::ChannelEnded {
                    leg: Leg::Interruption,
                    cause: crate::trace::ChannelEndCause::SourceEnded,
                });
                self.restore_audio_activity_guard();
                self.activity.audio_recognition.set_interruption_active(false);
                return;
            }
        };
        self.record_trace(InputStep::LegVerdict {
            leg: Leg::Interruption,
            req: req.map_or(0, |r| r.0),
        });
        self.record_trace(InterruptStep::OverlapVerdict {
            is_interruption: v.is_interruption,
            agent_ended: v.agent_ended,
        });
        // Every verdict is published before it is acted on.
        if let Some(event) = &v.reported {
            self.publish_event(crate::events::SessionEvent::OverlappingSpeech(
                std::sync::Arc::clone(event),
            ));
        }
        if let Some(m) = &v.metrics {
            self.publish_event(crate::events::SessionEvent::MetricsCollected {
                metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::Interruption(
                    m.clone(),
                )),
                speech: None,
                counted: false,
            });
        }
        // A verdict for an overlap no longer open changes nothing; one inside the boundary
        // cooldown is not trusted. The rest is on the effect list, in order.
        let (applied, eff) = self.activity.audio_recognition.on_overlap_verdict(req, &v);
        self.apply_recognition_effects(eff);
        if let Err(refusal) = applied {
            self.record_trace(InterruptStep::OverlapRefused(refusal));
        }
    }

    /// The path's own gate, five guards, then pause or interrupt. Every refusal is journalled by
    /// name. Returns whether the realtime socket was told.
    pub(crate) fn interrupt_by_audio_activity(
        &mut self,
        now: Ts,
        trigger: AudioInterruptTrigger,
    ) -> bool {
        // The sink as it is *now*, not as it was at start.
        self.refresh_pause_facts(now);
        let speech = match self.audio_interrupt_guards(trigger) {
            Ok(speech) => speech,
            Err(refusal) => {
                self.record_trace(InterruptStep::ByAudioActivityRefused(refusal));
                return false;
            }
        };
        // The pause itself survives; only the resume timer goes.
        self.cancel_false_interruption_timer();
        let mut report = InterruptReport { speech, trigger, paused: false, interrupted: None };
        match self.activity.paused_speech.pause_timeout() {
            // The false-interruption timer decides later whether to resume.
            Some(timeout) => {
                report.paused = true;
                let ignore_user_transcript_until = match trigger {
                    AudioInterruptTrigger::DetectorVerdict { ignore_user_transcript_until } => {
                        ignore_user_transcript_until
                    }
                    AudioInterruptTrigger::VadThreshold { .. }
                    | AudioInterruptTrigger::SttInterim { .. }
                    | AudioInterruptTrigger::SttFinal { .. } => now,
                };
                // The record before the sink goes quiet, so the resume has something to resume.
                let agent_state = self.session.states.agent_state();
                self.activity.paused_speech.update_paused_speech(speech, agent_state, timeout);
                self.gate_op(speech, GateOp::Pause, GateCause::ByAudioActivity);
                // `listening` before the flush below re-enters recognition.
                self.set_agent_state(now, AgentState::Listening);
                let turn = &mut self.activity.audio_recognition;
                let eff = turn.on_end_of_agent_speech(now, ignore_user_transcript_until);
                self.apply_recognition_effects(eff);
                if self.session.config.effective.turn.interruption.detection_enabled() {
                    self.session.by_audio_activity = true;
                }
                // Paused after the person stopped: no end of speech is coming to arm the resume.
                if !self.activity.audio_recognition.user_speaking() {
                    self.start_false_interruption_timer(now);
                }
            }
            // Only the host may override `allow_interruptions`.
            None => match self.interrupt_speech(now, speech, InterruptCause::ByAudioActivity) {
                Ok(outcome) => report.interrupted = Some(outcome != InterruptOutcome::Already),
                Err(_) => self.record_trace(InterruptStep::ByAudioActivityRefused(
                    InterruptRefusal::NotAllowed,
                )),
            },
        }
        // A pause keeps the realtime generation so playout can resume.
        let socket_told = self.activity.realtime().is_some() && report.interrupted == Some(true);
        self.record_trace(InterruptStep::ByAudioActivity(Box::new(report)));
        socket_told
    }

    /// The calling path's gate, then the guards in order: the current speech to barge into, or
    /// the guard that refused.
    fn audio_interrupt_guards(
        &self,
        trigger: AudioInterruptTrigger,
    ) -> Result<SpeechId, InterruptRefusal> {
        use InterruptRefusal as R;
        let (turn, session) = (&self.activity.audio_recognition, &self.session);
        let options = &self.session.config.effective.turn.interruption;
        let owns_turn = match turn.turn_detection_mode() {
            TurnDetectionMode::Vad | TurnDetectionMode::Stt => true,
            TurnDetectionMode::Manual | TurnDetectionMode::RealtimeLlm => false,
        };
        match trigger {
            AudioInterruptTrigger::VadThreshold { duration, raw_silence } => {
                if !owns_turn {
                    return Err(R::ManualTurnDetection);
                }
                // `>=`: a segment exactly on the threshold interrupts.
                if duration < options.min_duration() {
                    return Err(R::TooShort);
                }
                // STT may end the turn before the VAD does: its tail does not interrupt.
                let stt_eos_received =
                    turn.turn_detection_mode() == TurnDetectionMode::Stt && !turn.user_speaking();
                if stt_eos_received && !raw_silence.is_zero() {
                    return Err(R::OwnTurnTail);
                }
            }
            AudioInterruptTrigger::SttInterim { words }
            | AudioInterruptTrigger::SttFinal { words } => {
                if !owns_turn {
                    return Err(R::ManualTurnDetection);
                }
                if words == 0 {
                    return Err(R::TooFewWords);
                }
            }
            AudioInterruptTrigger::DetectorVerdict { .. } => {}
        }
        // Guard 1: the adaptive detector clears it while it owns interruption.
        if !session.by_audio_activity {
            return Err(R::Disabled);
        }
        // Guard 2: both, since `remaining` is positive before the timer is ever armed.
        if !session.aec.remaining().is_zero() && session.aec.armed() {
            return Err(R::AecWarmup);
        }
        // Guard 3: realtime turn detection, read at the call.
        let stt_enabled = turn.stt_enabled();
        let realtime_turn_detection = self.activity.realtime().is_some_and(|s| {
            let caps = s.capabilities;
            caps.turn_detection
                && !(caps.can_disable_turn_detection
                    && (stt_enabled
                        || self.input.sinks.contains_key(&crate::audio::AudioChannel::Vad)))
        });
        if realtime_turn_detection {
            return Err(R::RealtimeTurnDetection);
        }
        // Guard 4: skipped with no transcript source; counts the final plus the live interim.
        let min_words = options.min_words() as usize;
        if stt_enabled && min_words > 0 && spoken_words(&turn.current_transcript()) < min_words {
            return Err(R::TooFewWords);
        }
        // Guard 5.
        let Some((id, current)) = self.activity.current_speech() else {
            return Err(R::NoCurrentSpeech);
        };
        if current.interrupted() {
            return Err(R::AlreadyInterrupted);
        }
        if !current.allow_interruptions() {
            return Err(R::NotAllowed);
        }
        Ok(id)
    }

    /// Cancel the resume timer (the pause survives), then pause the current speech if not
    /// already paused (an onset before the agent flipped to `speaking`).
    pub(crate) fn on_start_of_speech(&mut self) {
        self.cancel_false_interruption_timer();
        let agent_state = self.session.states.agent_state();
        if agent_state == AgentState::Speaking
            || self.activity.paused_speech.pause_timeout().is_none()
        {
            return;
        }
        let Some((id, current)) = self.activity.current_speech() else { return };
        if !current.interruptible() || self.activity.paused_speech.paused_handle() == Some(id) {
            return;
        }
        // EOS arms the resume; a final transcript or a replying commit interrupts the handle.
        self.activity.paused_speech.update_paused_speech(id, agent_state, Duration::ZERO);
        self.gate_op(id, GateOp::Pause, GateCause::Launch);
    }

    /// Uses the timeout captured at pause time (an onset pause stores `0` and resolves on the
    /// next tick).
    pub(crate) fn start_false_interruption_timer(&mut self, now: Ts) {
        let Some(timeout) = self.activity.paused_speech.paused_timeout() else { return };
        self.cancel_false_interruption_timer();
        self.activity.paused_speech.false_interruption_timer_armed();
        self.set_deadline(Timer::FalseInterruption, now.saturating_add(timeout));
    }

    /// The resume deadline passed. An open turn decision owns the paused speech: only once it
    /// commits or drops is the interruption known to be false.
    pub(crate) fn on_false_interruption_timeout(&mut self, now: Ts) {
        let turn_open = self.activity.audio_recognition.turn_open();
        if self.activity.paused_speech.false_interruption_timed_out(turn_open) {
            self.on_false_interruption(now);
        }
    }

    /// A no-op unless the timeout is already waiting on a turn. A turn torn down by the close does
    /// not resume (closing releases the pause itself); one the host cleared does.
    pub(crate) fn on_turn_settled(&mut self, now: Ts, settled: TurnSettled) {
        if self.activity.paused_speech.turn_settled(settled)
            && !(settled == TurnSettled::TornDown && self.session.closing.is_some())
        {
            self.on_false_interruption(now);
        }
    }

    /// The pause is resolved, resumed or not; a newer current speech makes it a real interruption.
    /// The pause is always taken here, or the next speech would play into a paused sink.
    fn on_false_interruption(&mut self, now: Ts) {
        let Some(paused) = self.activity.paused_speech.take_paused() else { return };
        // A pause the facts no longer permit was already released by `refresh_pause_facts`.
        let held = self.activity.paused_speech.pause_timeout().is_some();
        if self.activity.speech_scheduler.current_speech().is_some_and(|c| c != paused.handle) {
            if held {
                self.gate_op(paused.handle, GateOp::Resume, GateCause::Released);
            }
            return;
        }
        let done = self.activity.speech_scheduler.speech(paused.handle).is_none_or(|s| s.done());
        let resumed = held && !done;
        if held && done {
            self.gate_op(paused.handle, GateOp::Resume, GateCause::Released);
        }
        // Restored and resumed before the event.
        if resumed {
            self.set_agent_state(now, paused.agent_state);
            // Resuming starts a new active-speech interval.
            if paused.agent_state == AgentState::Speaking {
                let eff = self.activity.audio_recognition.on_start_of_agent_speech(now, now);
                self.apply_recognition_effects(eff);
            }
            let cause = GateCause::FalseInterruptionResolved;
            self.gate_op(paused.handle, GateOp::Resume, cause);
        }
        self.publish_event(SessionEvent::AgentFalseInterruption { resumed });
    }

    /// Without it a parked speech never finishes, holds the floor, and no later reply is
    /// authorized. With nothing paused the sink is not resumed: it would un-pause a playout this
    /// call knows nothing about.
    pub(crate) fn cancel_speech_pause(&mut self, now: Ts) {
        self.refresh_pause_facts(now);
        self.cancel_false_interruption_timer();
        let Some(paused) = self.activity.paused_speech.take_paused() else { return };
        self.interrupt_paused(now, paused.handle);
        // A pause the facts no longer permit was already released by `refresh_pause_facts`.
        if self.activity.paused_speech.pause_timeout().is_some() {
            self.gate_op(paused.handle, GateOp::Resume, GateCause::Cancelled);
        }
    }

    /// A change that no longer permits the pause releases it; the user speaking at the change
    /// interrupts the parked speech first, or resuming would play it into the user.
    pub(crate) fn refresh_pause_facts(&mut self, now: Ts) {
        let speech = self
            .activity
            .paused_speech
            .paused_handle()
            .or_else(|| self.activity.speech_scheduler.current_speech());
        let outputs =
            speech.map(|speech| self.speech_outputs(speech)).unwrap_or(&self.session.outputs);
        let (sink, audio_enabled) = outputs.pause_facts();
        let Some(paused) = self.activity.paused_speech.refresh_facts(sink, audio_enabled) else {
            return;
        };
        self.cancel_false_interruption_timer();
        if self.activity.audio_recognition.user_speaking() {
            self.interrupt_paused(now, paused);
        }
        self.gate_op(paused, GateOp::Resume, GateCause::Released);
    }

    /// The paused handle became current again, or finished. Asked at authorization and at
    /// release, since an onset pause can land between.
    pub(crate) fn release_paused_speech(&mut self, speech: SpeechId) {
        if self.activity.paused_speech.paused_handle() != Some(speech) {
            return;
        }
        self.cancel_false_interruption_timer();
        self.activity.paused_speech.take_paused();
        if self.activity.paused_speech.audio_enabled() {
            self.gate_op(speech, GateOp::Resume, GateCause::Released);
        }
    }

    /// The session-level `interrupt()`: the preemptive generation, the current speech (a protected
    /// one is refused), the next speech, the socket once, the speeches in their tools, then the
    /// queue in pop order until one refuses.
    pub(crate) fn interrupt_all(
        &mut self,
        now: Ts,
        force: bool,
        cause: InterruptCause,
    ) -> Result<Vec<SpeechId>, InterruptRefusal> {
        let abandoned =
            crate::agent_session::machine::activity::scheduling::PreemptiveDiscard::TurnAbandoned;
        self.cancel_preemptive_generation(now, abandoned);
        let generating = self.activity.realtime_generating();
        let mut reached = Vec::new();
        let mut socket_told = false;
        if let Some(id) = self.activity.speech_scheduler.current_speech() {
            self.cancel_false_interruption_timer();
            if self.interrupt_or_refuse(now, id, force, cause)? {
                reached.push(id);
                socket_told = self.activity.is_realtime_speech(id);
            }
        }
        // The next speech (AMD's hold, or the consecutive-speech gap) counts as current, so AMD's
        // forced interrupt cuts the reply made under the hold.
        let candidate = self.activity.speech_scheduler.next_speech();
        if let Some(id) = candidate
            && self.interrupt_or_refuse(now, id, force, cause)?
        {
            reached.push(id);
        }
        if !socket_told {
            self.realtime_interrupt(cause);
        }
        self.realtime_interrupt_session(generating);
        // Speeches whose tools run still end (outputs commit, no follow-up). One awaiting a
        // VoiceTask is the suspended parent's: the task's turns are not addressed to it.
        let working: Vec<SpeechId> = self
            .activity
            .speech_scheduler
            .speech_ids()
            .filter(|id| {
                let status = self.activity.speech_scheduler.status(*id);
                status == Some(Status::Working) && !self.activity.voice_tasks.awaits(*id)
            })
            .collect();
        for id in working {
            let result = if force {
                Ok(self.force_interrupt_speech(now, id, cause))
            } else {
                self.interrupt_speech(now, id, cause)
            };
            if let Ok(InterruptOutcome::Now) = result {
                reached.push(id);
            }
        }
        // In pop order: the first refusal keeps that speech and all behind it.
        for id in self.activity.speech_scheduler.speech_q().clone() {
            let result = if force {
                Ok(self.force_interrupt_speech(now, id, cause))
            } else {
                self.interrupt_speech(now, id, cause)
            };
            match result {
                Ok(InterruptOutcome::Now) => reached.push(id),
                Ok(InterruptOutcome::Already) => {}
                Err(_) => {
                    self.record_trace(InterruptStep::InterruptWalkStopped { speech: id });
                    break;
                }
            }
        }
        Ok(reached)
    }

    /// The whole speech stopped; a protected one journalled and returned as the caller's refusal.
    fn interrupt_or_refuse(
        &mut self,
        now: Ts,
        id: SpeechId,
        force: bool,
        cause: InterruptCause,
    ) -> Result<bool, InterruptRefusal> {
        let result = if force {
            Ok(self.force_interrupt_speech(now, id, cause))
        } else {
            self.interrupt_speech(now, id, cause)
        };
        match result {
            Ok(outcome) => Ok(matches!(outcome, InterruptOutcome::Now)),
            Err(_) => {
                let refusal = InterruptRefusal::NotAllowed;
                self.record_trace(InterruptStep::ByAudioActivityRefused(refusal));
                Err(refusal)
            }
        }
    }
}

impl VoiceDriver {
    fn cancel_false_interruption_timer(&mut self) {
        if self.activity.paused_speech.cancel_false_interruption_timer() {
            self.clear_deadline(Timer::FalseInterruption);
        }
    }
}

impl super::AgentActivity {
    pub(crate) fn take_settled_interrupt_waiters(&mut self) -> Vec<InterruptWaiter> {
        let mut settled = Vec::new();
        let mut waiters = std::mem::take(&mut self.interrupt_waiters);
        waiters.retain_mut(|w| {
            if w.speeches.iter().any(|id| self.speech_scheduler.contains(*id)) {
                return true;
            }
            settled
                .push(InterruptWaiter { call: w.call, speeches: std::mem::take(&mut w.speeches) });
            false
        });
        self.interrupt_waiters = waiters;
        settled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a pause whose verdict waits on an open turn left for ever when the host clears the
    // turn: the agent stays silent until the close.
    #[test]
    fn a_cleared_turn_resolves_the_pause_waiting_on_it() {
        let cfg = AgentSessionOptions::default();
        let mut paused = PausedSpeech::new(&cfg, AudioOutputCapabilities::default(), true);
        paused.update_paused_speech(SpeechId(1), AgentState::Speaking, Duration::ZERO);
        paused.false_interruption_timer_armed();
        assert!(!paused.false_interruption_timed_out(true), "the open turn owns the verdict");
        assert!(paused.turn_settled(TurnSettled::TornDown), "a torn-down turn resolves it");
    }
}
