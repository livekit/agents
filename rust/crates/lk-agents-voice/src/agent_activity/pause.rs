//! Interruption: interruption, `interrupt()`, and the false-interruption pause
//! ([`PausedSpeech`]).

use crate::agent_activity::options::SessionConfig;
use crate::agent_activity::scheduling::{SpeechId, Status};
use crate::agent_session::effects::Out;
use crate::agent_session::effects::{GateCause, GateOp, TurnTimer};
use crate::agent_session::{Answer, Driver, RequestId};
use crate::audio_recognition::TurnDetectionMode;
use crate::events::SessionEvent;
use crate::speech_handle::SpeechError;
use crate::trace::{InputStep, InterruptCause, InterruptStep, SpeechStep};
use crate::turn::InterruptionOptions;
use lk_agents_abi::Ts;
use lk_agents_abi::tokenize::{WordTokenizerOptions, count_words};
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::voice::rtc::AudioOutputCapabilities;
use lk_agents_voice::io::{AudioOutput, AudioOutputExt};
use std::time::Duration;

pub use paused_speech::PausedSpeech;

/// `_paused_speech` and the false-interruption timer, behind their methods.
mod paused_speech {
    use super::*;

    /// The false-interruption pause: `_paused_speech`, `_false_interruption_timer` and
    /// `_false_interruption_pending`, with the sink facts `_pause_enabled()` reads. Part of the
    /// [`AgentActivity`](crate::agent_activity::AgentActivity).
    #[derive(Debug, Clone)]
    pub struct PausedSpeech {
        /// The interruption options, read off the resolved configuration at construction.
        interruption: InterruptionOptions,
        /// The sink's capabilities as of the last `refresh_pause_facts`.
        sink: AudioOutputCapabilities,
        /// `session.output.audio_enabled and session.output.audio is not None`, likewise.
        audio_enabled: bool,
        paused_speech: Option<PausedSpeechInfo>,
        /// Armed, so a cancel is issued only when needed.
        false_interruption_timer: bool,
        /// The timeout elapsed with a turn decision open, so the turn owns the verdict.
        false_interruption_pending: bool,
    }

    impl PausedSpeech {
        /// A session with no sink passes `AudioOutputCapabilities::default()` and `false`.
        pub fn new(
            cfg: &SessionConfig,
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

        /// `_pause_enabled()`, the **one** predicate for pausing and resuming, with its timeout
        /// (which pause-enabled implies, so the reference's `assert` can never fire). `None` is
        /// pausing off.
        pub fn pause_timeout(&self) -> Option<Duration> {
            let i = &self.interruption;
            let enabled = i.resume_false_interruption && self.audio_enabled && self.sink.pause;
            i.false_interruption_timeout.filter(|_| enabled)
        }

        pub fn paused_handle(&self) -> Option<SpeechId> {
            self.paused_speech.map(|p| p.handle)
        }

        /// `_cancel_false_interruption_timer`: whether the timer was armed (its cancel is the
        /// caller's); the pending flag clears, the pause itself stays.
        pub fn cancel_false_interruption_timer(&mut self) -> bool {
            let armed = std::mem::take(&mut self.false_interruption_timer);
            self.false_interruption_pending = false;
            armed
        }

        /// The paused speech's resume timeout, captured at pause time.
        pub fn paused_timeout(&self) -> Option<Duration> {
            self.paused_speech.map(|p| p.timeout)
        }

        /// The caller armed `TurnTimer::FalseInterruption`.
        pub fn false_interruption_timer_armed(&mut self) {
            self.false_interruption_timer = true;
        }

        /// `_on_timeout`: `true` when the interruption is known to be false now; an open turn
        /// decision owns the verdict instead (`turn_open`).
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

        /// `_on_turn_settled`: `true` when the pending verdict resolves now. A turn torn down
        /// instead of decided does not resume ("closing releases the pause itself").
        pub fn turn_settled(&mut self, settled: TurnSettled) -> bool {
            if !self.false_interruption_pending {
                return false;
            }
            match settled {
                TurnSettled::Superseded => false,
                TurnSettled::TornDown => {
                    self.false_interruption_pending = false;
                    false
                }
                TurnSettled::Decided => {
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

        /// The sink's facts as they are now. A change that no longer permits the pause releases it:
        /// the released speech.
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

        /// `_update_paused_speech`: re-pausing the same handle updates only the timeout.
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

/// What a interruption did, journalled as `InterruptStep::ByAudioActivity`. A refusal is not a report: it is
/// journalled as `InterruptStep::ByAudioActivityRefused(reason)` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterruptReport {
    pub speech: SpeechId,
    pub trigger: AudioInterruptTrigger,
    /// `_pause_enabled()` held: the playout was paused, nothing was interrupted.
    pub paused: bool,
    /// The handle was told to stop, and whether it landed *now*; `None` for a pause or a refusal.
    pub interrupted: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum InterruptOutcome {
    /// Interrupted now. `position` is where its output had reached at the clear, if anywhere.
    Now { position: Option<Duration> },
    /// Already interrupted, already finished, or already released: nothing to do, nothing said.
    Already,
}

/// The pause facts off the driver's sink: its `can_pause` and whether there is one.
pub fn sink_facts(out: Option<&dyn AudioOutput>) -> (AudioOutputCapabilities, bool) {
    match out {
        Some(o) => (AudioOutputCapabilities { pause: o.can_pause() }, true),
        None => (AudioOutputCapabilities::default(), false),
    }
}

// The port is `lk-agents-abi`'s; the host supplies the object (`lk-agents-inference` cannot be
// linked here).
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
    /// The realtime server's own `input_speech_started`; exempt from guard 3.
    ServerOnset,
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

/// `len(split_words(text, split_character=True))`: on here (off in `user_turn_limit`) so a CJK
/// utterance can reach `min_words`. The one tokenizer for interruption, commit and transcript triggers.
pub fn count_min_words(text: &str) -> usize {
    count_words(text, WordTokenizerOptions { split_character: true, ..Default::default() })
}

/// `agent_activity._PausedSpeechInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PausedSpeechInfo {
    pub handle: SpeechId,
    /// The agent state at the **first** pause of this handle; a re-pause updates only the timeout.
    pub agent_state: AgentState,
    pub timeout: Duration,
}

/// `_on_turn_settled`, `._start_false_interruption_timer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnSettled {
    /// A newer decision superseded this one; keep waiting.
    Superseded,
    /// Torn down instead of decided. **Does not resume**: "closing releases the pause itself".
    TornDown,
    Decided,
}

impl Driver {
    /// **The only writer of a speech's interruption.** `Err(InterruptionsNotAllowed)` when
    /// forbidden and not forced; an ended speech is `Ok(Already)`. The handle's own `interrupt()`
    /// (a tool's `ctx.speech_handle.interrupt()`: the host, unforced) stops only the playing step,
    /// so the tool's follow-up is still spoken; every other cause stops the whole speech.
    pub(crate) fn interrupt_speech(
        &mut self,
        now: Ts,
        speech: SpeechId,
        cause: InterruptCause,
        force: bool,
        out: &mut Out,
    ) -> Result<InterruptOutcome, SpeechError> {
        let step_only = cause == InterruptCause::Host && !force;
        self.stop_speech(now, speech, cause, force, step_only, out)
    }

    /// `step_only` stops the current step and lets the tool reply follow; otherwise the whole
    /// speech stops, as the session-level `interrupt()` does: running tools end, outputs commit,
    /// no follow-up is spoken.
    fn stop_speech(
        &mut self,
        now: Ts,
        speech: SpeechId,
        cause: InterruptCause,
        force: bool,
        step_only: bool,
        out: &mut Out,
    ) -> Result<InterruptOutcome, SpeechError> {
        let Some(handle) = self.activity.speech_scheduler.handle(speech).cloned() else {
            return Ok(InterruptOutcome::Already);
        };
        let landed = if step_only { handle.interrupt_step()? } else { handle.interrupt(force)? };
        // A close's cut: the task reports its end during the drain; the hard edge releases what
        // is still open.
        if !landed {
            return Ok(InterruptOutcome::Already);
        }
        // Only a speech whose audio is in the sink has anything to clear; clearing for a queued
        // speech would cut the current one. The clear precedes any `Resume`; the position is read
        // after it. `cut`, never a bare `clear_buffer`: the flush closes the segment in the
        // trackers, or the next speech's wait resolves on this cut's event.
        let in_the_sink = self.activity.speech_scheduler.current_speech() == Some(speech)
            || self.activity.paused_speech.paused_handle() == Some(speech);
        let audio_out = in_the_sink.then(|| self.shared.output.audio()).flatten();
        let position = audio_out.as_deref().map(|out| {
            out.cut();
            out.tracker().position()
        });
        out.trace(InterruptStep::Interrupted { speech, cause, force, position });
        // A filler dies with the interruption of the speech that owns its tool.
        self.fillers_speech_ended(speech, out);
        // A tool's `foreground()` / `wait_for_idle()` ask is refused too when the cut lands during
        // playout (the reference cancels the tools), or it would wait on the speech itself. A
        // speech already in its tool phase, or a step interruption, keeps its tools running.
        if !step_only && !self.activity.tools_in_flight.contains(&speech) {
            self.withdraw_idle_asks(speech, out);
        }
        if audio_out.is_some() {
            let (op, cause) = (GateOp::Clear, GateCause::Interrupted);
            out.trace(InterruptStep::PlayoutGate { speech, op, cause });
        }
        let realtime = self.is_realtime_speech(speech);
        // Never authorized now; the watchdog is armed on the session clock. Nothing fires after
        // the close: every handle is interrupted there and the loop exits.
        let closing = self.session.closing_past_edge();
        if let Some(at) = self.activity.speech_scheduler.interrupted(speech, now, !closing) {
            out.trace(SpeechStep::WatchdogArmed { speech, at });
        }
        // A realtime server keeps generating (into its own mirror) until told not to.
        if realtime {
            self.realtime_interrupt(out);
        }
        Ok(InterruptOutcome::Now { position })
    }

    /// The one place the loop moves the playout sink, journalled as `PlayoutGate{speech, op,
    /// cause}`. No sink, no op, no step.
    fn gate_op(&self, out: &mut Out, speech: SpeechId, op: GateOp, cause: GateCause) {
        if self.shared.output.audio().is_some() {
            let gate = crate::agent_session::effects::AudioOp::Gate { speech, op, cause };
            out.machine().push(crate::agent_session::effects::Effect::Audio(gate));
        }
    }

    /// Interrupt a paused speech the way `_cancel_speech_pause(interrupt=True)` does: refused for a
    /// handle already interrupted or never interruptible, and a hold taken since is kept.
    fn interrupt_paused(&mut self, now: Ts, speech: SpeechId, out: &mut Out) {
        let Some(h) = self.activity.speech_scheduler.handle(speech) else { return };
        if !h.interruptible() {
            return;
        }
        if self.interrupt_speech(now, speech, InterruptCause::UserTurn, false, out).is_err() {
            out.trace(InterruptStep::ByAudioActivityRefused(InterruptRefusal::NotAllowed));
        }
    }

    /// `_restore_interruption_by_audio_activity()`: guard 1 back to the session's option, not
    /// `true` (a manual or realtime-detected session had it off).
    pub(crate) fn restore_audio_activity_guard(&mut self) {
        self.session.by_audio_activity =
            self.session.config.effective.turn.interruption.by_audio_activity;
    }

    /// `_on_overlap_speech_event` plus `on_interruption`; the decisions are
    /// [`AudioRecognition::on_overlap_verdict`]'s. A leg failure hands interruption back to the VAD and never
    /// errors; an unrecoverable one also stops the sentinels.
    pub(crate) fn on_overlap_verdict(
        &mut self,
        _now: Ts,
        ev: lk_agents_abi::turn::detector::LegEvent<crate::audio_recognition::OverlapVerdict>,
        out: &mut Out,
    ) {
        use crate::trace::DetectorLeg;
        use lk_agents_abi::turn::detector::LegEvent;
        let (req, v) = match ev {
            LegEvent::Verdict { req, verdict } => (req, verdict),
            LegEvent::Error { recoverable, .. } => {
                out.trace(InputStep::LegFailed { leg: DetectorLeg::Interruption, recoverable });
                self.restore_audio_activity_guard();
                if !recoverable {
                    self.activity.audio_recognition.set_interruption_active(false);
                }
                return;
            }
            LegEvent::Closed => {
                out.trace(InputStep::LegEnded {
                    leg: "interruption",
                    cause: crate::trace::LegEndCause::SourceEnded,
                });
                self.restore_audio_activity_guard();
                self.activity.audio_recognition.set_interruption_active(false);
                return;
            }
        };
        out.trace(InputStep::LegVerdict {
            leg: DetectorLeg::Interruption,
            req: req.map_or(0, |r| r.0),
        });
        out.trace(InterruptStep::OverlapVerdict {
            is_interruption: v.is_interruption,
            agent_ended: v.agent_ended,
        });
        // A verdict for an overlap no longer open changes nothing; one inside the boundary
        // cooldown is not trusted. The rest is on the effect list, in order.
        let (applied, eff) = self.activity.audio_recognition.on_overlap_verdict(req, &v);
        out.machine().append(eff);
        if let Err(refusal) = applied {
            out.trace(InterruptStep::OverlapRefused(refusal));
        }
    }

    /// `_interrupt_by_audio_activity`, from any path: the path's own gate, five guards, then pause
    /// (`_pause_enabled()`) or interrupt. Every refusal is journalled by name. Returns whether the
    /// realtime socket was told.
    pub(crate) fn interrupt_by_audio_activity(
        &mut self,
        now: Ts,
        out: &mut Out,
        trigger: AudioInterruptTrigger,
    ) -> bool {
        // The sink as it is *now*, not as it was at start.
        self.refresh_pause_facts(now, out);
        let speech = match self.audio_interrupt_guards(trigger) {
            Ok(speech) => speech,
            Err(refusal) => {
                out.trace(InterruptStep::ByAudioActivityRefused(refusal));
                return false;
            }
        };
        // The pause itself survives; only the resume timer goes.
        self.cancel_false_interruption_timer(out);
        let mut report = InterruptReport { speech, trigger, paused: false, interrupted: None };
        match self.activity.paused_speech.pause_timeout() {
            // Nothing is interrupted; the false-interruption timer decides later whether to resume.
            Some(timeout) => {
                report.paused = true;
                // `on_interruption` passes the overlap's start as the ignore window; others `now`.
                let ignore_user_transcript_until = match trigger {
                    AudioInterruptTrigger::DetectorVerdict { ignore_user_transcript_until } => {
                        ignore_user_transcript_until
                    }
                    AudioInterruptTrigger::VadThreshold { .. }
                    | AudioInterruptTrigger::SttInterim { .. }
                    | AudioInterruptTrigger::SttFinal { .. }
                    | AudioInterruptTrigger::ServerOnset => now,
                };
                // The record before the sink goes quiet, so the resume has something to resume.
                let agent_state = self.session.states.agent_state();
                self.activity.paused_speech.update_paused_speech(speech, agent_state, timeout);
                self.gate_op(out, speech, GateOp::Pause, GateCause::ByAudioActivity);
                // `listening` before the flush below re-enters `AudioRecognition`.
                self.set_agent_state(now, AgentState::Listening, out);
                let turn = &mut self.activity.audio_recognition;
                let eff = turn.on_end_of_agent_speech(now, ignore_user_transcript_until);
                out.machine().append(eff);
                // Hands interruption back to audio activity (undoes guard 1).
                if self.session.config.effective.turn.interruption.detection_enabled {
                    self.session.by_audio_activity = true;
                }
            }
            // The handle may have stopped allowing interruptions since guard 5 read it. A
            // interruption never overrides `allow_interruptions`; only the host may.
            None => match self.interrupt_speech(
                now,
                speech,
                InterruptCause::ByAudioActivity,
                false,
                out,
            ) {
                Ok(outcome) => report.interrupted = Some(outcome != InterruptOutcome::Already),
                Err(_) => {
                    out.trace(InterruptStep::ByAudioActivityRefused(InterruptRefusal::NotAllowed))
                }
            },
        }
        // A pause keeps the realtime generation so playout can resume.
        let socket_told = self.activity.rt_session.is_some() && report.interrupted == Some(true);
        out.trace(InterruptStep::ByAudioActivity(Box::new(report)));
        socket_told
    }

    /// The calling path's gate, then `_interrupt_by_audio_activity`'s guards in its order: the
    /// current speech to barge into, or the guard that refused.
    fn audio_interrupt_guards(
        &self,
        trigger: AudioInterruptTrigger,
    ) -> Result<SpeechId, InterruptRefusal> {
        use InterruptRefusal as R;
        let (turn, session) = (&self.activity.audio_recognition, &self.session);
        let options = &self.session.config.effective.turn.interruption;
        // `self._turn_detection not in ("manual", "realtime_llm")`.
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
                if duration < options.min_duration {
                    return Err(R::TooShort);
                }
                // "STT may send EOS before VAD EOS, we only interrupt if: turn detection is not
                // STT; or STT EOS hasn't been received yet; or VAD speech is still ongoing." In
                // `stt` mode "the user is not speaking" is exactly `_stt_eos_received`.
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
            AudioInterruptTrigger::ServerOnset | AudioInterruptTrigger::DetectorVerdict { .. } => {}
        }
        // Guard 1: the adaptive detector clears it while it owns interruption.
        if !session.by_audio_activity {
            return Err(R::Disabled);
        }
        // Guard 2: `_aec_warmup_remaining > 0 and _aec_warmup_timer is not None`: both, since
        // `remaining` is positive before the timer is ever armed.
        if !session.aec.remaining().is_zero() && session.aec.armed() {
            return Err(R::AecWarmup);
        }
        // Guard 3: `_rt_turn_detection_enabled`, read at the call; the server's own onset is exempt.
        let stt_enabled = turn.stt_enabled();
        let realtime_turn_detection = self.activity.rt_session.as_ref().is_some_and(|s| {
            let caps = s.capabilities();
            caps.turn_detection
                && !(caps.can_disable_turn_detection && (stt_enabled || self.input.has_vad()))
        });
        if realtime_turn_detection && trigger != AudioInterruptTrigger::ServerOnset {
            return Err(R::RealtimeTurnDetection);
        }
        // Guard 4: `min_words > 0` enables it; with no transcript source it is skipped. The
        // transcript is the accumulated final **plus** the live interim.
        let min_words = options.min_words as usize;
        if stt_enabled && min_words > 0 && count_min_words(&turn.current_transcript()) < min_words {
            return Err(R::TooFewWords);
        }
        // Guard 5, three refusals with names.
        let Some(current) = self.current_handle() else { return Err(R::NoCurrentSpeech) };
        if current.interrupted() {
            return Err(R::AlreadyInterrupted);
        }
        if !current.allow_interruptions() {
            return Err(R::NotAllowed);
        }
        Ok(current.id())
    }

    /// `_cancel_false_interruption_timer`: cancels the timer and clears the pending flag, leaving
    /// the pause itself in place.
    fn cancel_false_interruption_timer(&mut self, out: &mut Out) {
        if self.activity.paused_speech.cancel_false_interruption_timer() {
            out.machine().cancel(TurnTimer::FalseInterruption);
        }
    }

    /// `on_start_of_speech`'s pause: cancel the resume timer (the pause survives), then pause the
    /// current speech if not already paused (an onset before the agent flipped to `speaking`).
    pub(crate) fn on_start_of_speech(&mut self, out: &mut Out) {
        self.cancel_false_interruption_timer(out);
        let agent_state = self.session.states.agent_state();
        if agent_state == AgentState::Speaking
            || self.activity.paused_speech.pause_timeout().is_none()
        {
            return;
        }
        let Some(current) = self.current_handle() else { return };
        let id = current.id();
        if !current.interruptible() || self.activity.paused_speech.paused_handle() == Some(id) {
            return;
        }
        // EOS arms the resume; a final transcript or a replying commit interrupts the handle.
        self.activity.paused_speech.update_paused_speech(id, agent_state, Duration::ZERO);
        self.gate_op(out, id, GateOp::Pause, GateCause::Launch);
    }

    /// `on_end_of_speech`'s `_start_false_interruption_timer(timeout)`, with the timeout captured
    /// at pause time (an onset pause stores `0` and resolves on the next tick).
    pub(crate) fn start_false_interruption_timer(&mut self, now: Ts, out: &mut Out) {
        let Some(timeout) = self.activity.paused_speech.paused_timeout() else { return };
        self.cancel_false_interruption_timer(out);
        self.activity.paused_speech.false_interruption_timer_armed();
        out.machine().arm(TurnTimer::FalseInterruption, now.saturating_add(timeout));
    }

    /// `_on_timeout`: the resume deadline passed. An open turn decision owns the paused speech: it
    /// commits (interrupting) or drops it, and only then is the interruption known to be false.
    pub(crate) fn on_false_interruption_timeout(&mut self, now: Ts, out: &mut Out) {
        let turn_open = self.activity.audio_recognition.turn_open();
        if self.activity.paused_speech.false_interruption_timed_out(turn_open) {
            self.on_false_interruption(now, out);
        }
    }

    /// `_on_turn_settled`. A no-op unless the timeout is already waiting on a turn; a turn torn
    /// down instead of decided does not resume ("closing releases the pause itself").
    pub(crate) fn on_turn_settled(&mut self, now: Ts, settled: TurnSettled, out: &mut Out) {
        if self.activity.paused_speech.turn_settled(settled) {
            self.on_false_interruption(now, out);
        }
    }

    /// `_on_false_interruption`: the pause is resolved, resumed or not. A newer current speech
    /// makes it a real interruption: no event.
    fn on_false_interruption(&mut self, now: Ts, out: &mut Out) {
        let Some(paused) = self.activity.paused_speech.take_paused() else { return };
        if self.activity.speech_scheduler.current_speech().is_some_and(|c| c != paused.handle) {
            return;
        }
        // The one predicate (the reference drops `audio_enabled` here; this port does not).
        let done = self.activity.speech_scheduler.handle(paused.handle).is_none_or(|h| h.done());
        let resumed = self.activity.paused_speech.pause_timeout().is_some() && !done;
        out.machine().emit(SessionEvent::AgentFalseInterruption { resumed });
        if resumed {
            self.set_agent_state(now, paused.agent_state, out);
            // Resuming starts a new active-speech interval: the lifecycle follows audible playout.
            if paused.agent_state == AgentState::Speaking {
                let eff = self.activity.audio_recognition.on_start_of_agent_speech(now, now);
                out.machine().append(eff);
            }
            let cause = GateCause::FalseInterruptionResolved;
            self.gate_op(out, paused.handle, GateOp::Resume, cause);
        }
    }

    /// `_cancel_speech_pause(interrupt=True)`: without it a parked speech never finishes, holds
    /// `_current_speech`, and no later reply is authorized. The timer is cancelled before the
    /// nothing-paused return, but the sink is not resumed then: it would un-pause a playout this
    /// call knows nothing about. Teardown goes through `begin_closing`.
    pub(crate) fn cancel_speech_pause(&mut self, now: Ts, out: &mut Out) {
        self.refresh_pause_facts(now, out);
        self.cancel_false_interruption_timer(out);
        let Some(paused) = self.activity.paused_speech.take_paused() else { return };
        self.interrupt_paused(now, paused.handle, out);
        // A pause the facts no longer permit was already released by `refresh_pause_facts`.
        if self.activity.paused_speech.pause_timeout().is_some() {
            self.gate_op(out, paused.handle, GateOp::Resume, GateCause::Cancelled);
        }
    }

    /// Re-read the pause facts off the driver's own `audio_out`, so no pause decision reads a stale
    /// copy. A change that no longer permits the pause **releases** it; the user speaking at the
    /// change interrupts the parked speech first, or resuming would play it into the user.
    pub(crate) fn refresh_pause_facts(&mut self, now: Ts, out: &mut Out) {
        let (sink, present) = sink_facts(self.shared.output.audio().as_deref());
        let audio_enabled = present && self.shared.output.plays_audio();
        let Some(paused) = self.activity.paused_speech.refresh_facts(sink, audio_enabled) else {
            return;
        };
        self.cancel_false_interruption_timer(out);
        if self.activity.audio_recognition.user_speaking() {
            self.interrupt_paused(now, paused, out);
        }
        self.gate_op(out, paused, GateOp::Resume, GateCause::Released);
    }

    /// `_scheduling_task`'s post-await release: the paused handle became current again, or
    /// finished. Asked at authorization and at release, since an onset pause can land between.
    pub(crate) fn release_paused_speech(&mut self, speech: SpeechId, out: &mut Out) {
        if self.activity.paused_speech.paused_handle() != Some(speech) {
            return;
        }
        self.cancel_false_interruption_timer(out);
        self.activity.paused_speech.take_paused();
        if self.activity.paused_speech.audio_enabled() {
            self.gate_op(out, speech, GateOp::Resume, GateCause::Released);
        }
    }

    /// The session-level `interrupt()`: the preemptive generation, the current speech (a protected
    /// one is the caller's `RuntimeError`), the next speech, the socket once, the speeches in
    /// their tools, then the queue in pop order until one refuses.
    pub(crate) fn interrupt_all(
        &mut self,
        now: Ts,
        force: bool,
        cause: InterruptCause,
        out: &mut Out,
    ) -> Result<Vec<SpeechId>, InterruptRefusal> {
        // `_cancel_preemptive_generation()`: the user taking the floor back invalidates a
        // preemptive generation.
        let abandoned = crate::agent_activity::scheduling::PreemptiveDiscard::TurnAbandoned;
        self.cancel_preemptive_generation(now, abandoned, out);
        let mut reached = Vec::new();
        let mut socket_told = false;
        if let Some(id) = self.activity.speech_scheduler.current_speech() {
            self.cancel_false_interruption_timer(out);
            if self.interrupt_or_refuse(now, id, force, cause, out)? {
                reached.push(id);
                socket_told = self.is_realtime_speech(id);
            }
        }
        // The next speech (AMD's hold, or the consecutive-speech gap) is the reference's
        // `_current_speech` already, so AMD's forced interrupt cuts the reply made under the hold.
        let candidate = self.activity.speech_scheduler.next_speech();
        if let Some(id) = candidate
            && self.interrupt_or_refuse(now, id, force, cause, out)?
        {
            reached.push(id);
        }
        // `self._rt_session.interrupt()`, unconditional and once.
        if !socket_told {
            self.realtime_interrupt(out);
        }
        // Speeches whose step is spoken and tools running: still `_current_speech` to the
        // reference, so they end (outputs commit, no follow-up).
        let working: Vec<SpeechId> = self
            .activity
            .speech_scheduler
            .speech_ids()
            .filter(|id| {
                let status = self.activity.speech_scheduler.status(*id);
                matches!(status, Some(Status::Working | Status::Parked))
            })
            .collect();
        for id in working {
            if let Ok(InterruptOutcome::Now { .. }) =
                self.stop_speech(now, id, cause, force, false, out)
            {
                reached.push(id);
            }
        }
        // In pop order: the first refusal keeps that speech and all behind it (the reference breaks).
        for id in self.activity.speech_scheduler.speech_q().clone() {
            match self.stop_speech(now, id, cause, force, false, out) {
                Ok(InterruptOutcome::Now { .. }) => reached.push(id),
                Ok(InterruptOutcome::Already) => {}
                Err(_) => {
                    out.trace(InterruptStep::InterruptWalkStopped { speech: id });
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
        out: &mut Out,
    ) -> Result<bool, InterruptRefusal> {
        match self.stop_speech(now, id, cause, force, false, out) {
            Ok(outcome) => Ok(matches!(outcome, InterruptOutcome::Now { .. })),
            Err(_) => {
                let refusal = InterruptRefusal::NotAllowed;
                out.trace(InterruptStep::ByAudioActivityRefused(refusal));
                Err(refusal)
            }
        }
    }

    pub(crate) fn settle_interrupt_waiters(&mut self, out: &mut Out) {
        let mut waiters = std::mem::take(&mut self.activity.interrupt_waiters);
        waiters.retain_mut(|w| {
            // Done means released (items committed), not merely the handle's flag.
            if w.speeches.iter().any(|id| self.activity.speech_scheduler.contains(*id)) {
                return true;
            }
            out.trace(InterruptStep::InterruptSettled {
                speeches: std::mem::take(&mut w.speeches),
            });
            self.answer(w.call, Answer::Interrupt(Ok(())));
            false
        });
        self.activity.interrupt_waiters = waiters;
    }
}
