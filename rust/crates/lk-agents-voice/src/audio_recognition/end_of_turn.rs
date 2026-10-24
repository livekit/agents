//! The end of the user's turn: the phase, the detector's predictions and the countdown, the
//! commit, the user turn limit and the transcription timeout.

use super::overlap::In;
use super::{
    AudioRecognition, CommitDecision, CommitVerdict, Detector, EndOfTurnInfo, Phase, Prediction,
    Run, Start, TurnId, TurnSnapshot, UserTurnTracker, compute_end_of_turn_metrics,
    user_transcript,
};
use crate::agent_activity::pause::TurnSettled;
use crate::agent_session::effects::{
    Decision, Effect, Effects, FlushReason, ProviderCall, TurnTimer,
};
use crate::events::{
    AgentBackchannelOpportunityEvent, EotPredictionEvent, SessionEvent, TranscriptionTimeoutEvent,
    UserTurnExceededEvent,
};
use lk_agents_abi::tokenize::{WordTokenizerOptions, count_words};
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::{EotPrediction, SpeechEvent, SpeechEventType, Ts};
use std::time::Duration;

impl AudioRecognition {
    // --- the phase -----------------------------------------------------------------------------

    /// The one place the phase changes. Arms and cancels the phases' timers by diffing, so
    /// "armed => fired xor cancelled" holds here.
    pub(super) fn set_phase(&mut self, next: Phase) -> Effects {
        let mut eff = Effects::new();
        let before = self.phase.timer();
        let after = next.timer();
        if before != after {
            if let Some((timer, _)) = before
                && self.fired.take() != Some(timer)
            {
                eff.cancel(timer);
            }
            if let Some((timer, at)) = after {
                eff.arm(timer, at);
            }
        }
        self.phase = next;
        eff
    }

    /// `_user_turn_committed`: the owner declared this turn's end at least once. A running
    /// countdown carries its own flag (`Run::declared`); a countdown is not a declaration.
    pub(super) fn declared(&self) -> bool {
        match &self.phase {
            Phase::Idle | Phase::Committed { .. } => false,
            Phase::Speaking { declared, .. } | Phase::Ended { declared, .. } => *declared,
            Phase::Awaiting { run, .. } | Phase::Delaying { run, .. } => run.declared,
            Phase::ManualCommitted { .. } => true,
        }
    }

    fn prediction(&self) -> Prediction {
        match &self.phase {
            Phase::Speaking { prediction, .. } | Phase::Ended { prediction, .. } => *prediction,
            Phase::Awaiting { req, .. } => Prediction::Pending(*req),
            Phase::Delaying { prediction: Some(p), .. } => Prediction::Ready(*p),
            Phase::Delaying { prediction: None, .. }
            | Phase::Idle
            | Phase::ManualCommitted { .. }
            | Phase::Committed { .. } => Prediction::Absent,
        }
    }

    pub(super) fn mint_req(&mut self) -> ReqId {
        self.eot_seq += 1;
        ReqId(self.eot_seq)
    }

    /// User speech began: a running countdown and any prediction in flight are abandoned, except
    /// under `ManualCommitted` or `Committed`.
    pub(super) fn begin_speaking(&mut self, started_at: Ts) -> Effects {
        let mut eff = Effects::new();
        let next = match self.phase {
            Phase::ManualCommitted { run, fire_at, .. } => {
                Phase::ManualCommitted { run, fire_at, speaking: true }
            }
            Phase::Committed { turn_id, .. } => Phase::Committed { turn_id, speaking: true },
            _ => {
                let since = match self.phase {
                    Phase::Speaking { since, .. } => since,
                    _ => started_at,
                };
                let declared = self.declared();
                eff.append(self.abandon());
                Phase::Speaking { since, prediction: Prediction::Absent, declared }
            }
        };
        eff.append(self.set_phase(next));
        eff
    }

    /// User speech ended. A declaration landing on a running countdown is kept on it; the
    /// restart is the caller's.
    pub(super) fn end_speaking(&mut self, anchor: Ts, declared: bool) -> Effects {
        let mut eff = Effects::new();
        let next = match &mut self.phase {
            Phase::Idle => Phase::Ended { anchor, prediction: Prediction::Absent, declared },
            Phase::Speaking { prediction, declared: d, .. }
            | Phase::Ended { prediction, declared: d, .. } => {
                Phase::Ended { anchor, prediction: *prediction, declared: *d || declared }
            }
            Phase::Awaiting { run, .. } | Phase::Delaying { run, .. } => {
                run.declared |= declared;
                return eff;
            }
            Phase::ManualCommitted { run, fire_at, .. } => {
                Phase::ManualCommitted { run: *run, fire_at: *fire_at, speaking: false }
            }
            Phase::Committed { turn_id, .. } => {
                Phase::Committed { turn_id: *turn_id, speaking: false }
            }
        };
        eff.append(self.set_phase(next));
        eff
    }

    fn abandon(&self) -> Effects {
        let mut eff = Effects::new();
        match self.phase {
            Phase::Idle | Phase::Committed { .. } => {}
            Phase::Speaking { prediction, .. } | Phase::Ended { prediction, .. } => {
                eff.append(self.cancel_prediction(prediction));
            }
            Phase::Awaiting { req, .. } => {
                eff.cancel_eot(req, false);
                eff.decide(Decision::TurnSettled(TurnSettled::TornDown));
            }
            Phase::Delaying { .. } | Phase::ManualCommitted { .. } => {
                eff.decide(Decision::TurnSettled(TurnSettled::TornDown));
            }
        }
        eff
    }

    fn cancel_prediction(&self, prediction: Prediction) -> Effects {
        let mut eff = Effects::new();
        match prediction {
            // `timed_out=False`: the inference is being abandoned, not given up on.
            Prediction::Pending(req) => eff.cancel_eot(req, false),
            Prediction::Absent | Prediction::Ready(_) => {}
        }
        eff
    }

    /// Retire the running evaluation because the configuration changed. Unlike new speech this
    /// also leaves `Committed`, and forgets the declaration.
    pub(super) fn retire(&mut self, now: Ts) -> Effects {
        let mut eff = Effects::new();
        let speaking = self.user_speaking();
        self.deferred_opportunity = None;
        eff.append(self.abandon());
        let next = if speaking {
            Phase::Speaking {
                since: self.speech_start_time.unwrap_or(now),
                prediction: Prediction::Absent,
                declared: false,
            }
        } else {
            Phase::Idle
        };
        eff.append(self.set_phase(next));
        eff
    }

    // --- the turn detector --------------------------------------------------------------------

    /// The detector answered `req`, or failed. Only the request the phase holds applies; a
    /// failure takes the timeout's transition and serves `min_delay`.
    pub fn on_prediction(
        &mut self,
        now: Ts,
        req: ReqId,
        result: Result<EotPrediction, ()>,
    ) -> Effects {
        let mut eff = Effects::new();
        let held = match result {
            Ok(p) => Prediction::Ready(p),
            Err(()) => Prediction::Absent,
        };
        match self.phase {
            Phase::Speaking { since, prediction: Prediction::Pending(r), declared } if r == req => {
                eff.append(self.set_phase(Phase::Speaking { since, prediction: held, declared }));
            }
            Phase::Ended { anchor, prediction: Prediction::Pending(r), declared } if r == req => {
                eff.append(self.set_phase(Phase::Ended { anchor, prediction: held, declared }));
            }
            Phase::Awaiting { run, req: r, deadline: _ } if r == req => match result {
                Ok(p) => eff.append(self.resolve(now, run, p)),
                // "eot prediction timed out, committing without a prediction".
                Err(()) => eff.append(self.enter_delaying(now, run, run.min_delay, None, None)),
            },
            // Stale, or nothing was ever asked: the phase is untouched.
            Phase::Idle
            | Phase::Speaking { .. }
            | Phase::Ended { .. }
            | Phase::Awaiting { .. }
            | Phase::Delaying { .. }
            | Phase::ManualCommitted { .. }
            | Phase::Committed { .. } => {}
        }
        eff
    }

    // --- timers ---------------------------------------------------------------------------------

    /// A timer for a phase the machine has left (a fire that raced a cancel) is dropped.
    pub fn on_timer(&mut self, now: Ts, timer: TurnTimer) -> Effects {
        let mut eff = Effects::new();
        match timer {
            TurnTimer::EotPredictionTimeout => {
                if let Phase::Awaiting { run, req, deadline: Some(_) } = &self.phase {
                    let (run, req) = (*run, *req);
                    self.fired = Some(timer);
                    // `._bounce_eou_task`: the one site that cancels with `timed_out=True`.
                    eff.cancel_eot(req, true);
                    eff.append(self.enter_delaying(now, run, run.min_delay, None, None));
                    self.fired = None;
                }
            }
            TurnTimer::EndpointingDelay => {
                if let Some((run, speaking)) = self.phase.delay() {
                    self.fired = Some(timer);
                    eff.append(self.commit(now, run, speaking));
                    self.fired = None;
                }
            }
            TurnTimer::TranscriptionTimeout => {
                self.transcription_timeout_armed = false;
                // No open turn means nothing was expected; a landed transcript means the timeout
                // raced it.
                if let Some(turn_start) = self.user_turn_start
                    && !self.turn_transcript_received
                {
                    eff.emit(SessionEvent::UserTranscriptionTimeout(TranscriptionTimeoutEvent {
                        speech_duration: self.turn_speech_duration,
                        turn_start,
                    }));
                }
            }
            TurnTimer::BackchannelBoundary => {
                eff.append(self.overlap.step(In::Boundary { armed: false }).1);
            }
            TurnTimer::UserTurnDuration => {
                self.turn_limit_timer_armed = false;
                eff.append(self.fire_user_turn_limit(now));
            }
            // Owned by other machines; the driver routes them directly.
            TurnTimer::FalseInterruption
            | TurnTimer::AecWarmup
            | TurnTimer::UserAway
            | TurnTimer::Amd(_)
            | TurnTimer::IvrSilence
            | TurnTimer::RecorderFlush
            | TurnTimer::KeytermTimeout
            | TurnTimer::RealtimeRequest(_)
            | TurnTimer::Filler(_)
            | TurnTimer::DtmfDebounce => {}
        }
        eff
    }

    // --- lifecycle --------------------------------------------------------------------------------

    /// `_clear_user_turn`: drop the turn and everything accumulated for it. Unlike the reference
    /// this also retires a running countdown and a manual commit.
    pub fn clear_user_turn(&mut self, now: Ts) -> Effects {
        let mut eff = Effects::new();
        self.audio_transcript.clear();
        self.interim_transcript.clear();
        self.preflight_transcript.clear();
        self.final_confidences.clear();
        self.last_final_transcript_time = None;
        self.speech_start_time = None;
        self.last_speaking_time = None;
        self.vad_speech_started = false;
        self.last_emitted = None;
        self.deferred_opportunity = None;

        eff.append(self.retire(now));

        if matches!(self.cfg.detector, Detector::Streaming { .. }) {
            eff.flush_turn_detector(FlushReason::ClearUserTurn);
        }

        eff.append(self.reset_turn_limit());
        self.user_turn_start = None;
        eff.append(self.reset_transcription_timeout());

        // "reset stt to clear the buffer from previous user turn" (a stream swap here).
        eff.push(Effect::Provider(ProviderCall::FlushStt));
        eff
    }

    /// `._bounce_eou_task`. `TooFewWords` drops the words, `RealtimeBackchannel` keeps them.
    /// Anchors reset only `if committed:`. A verdict for a turn already left is dropped.
    pub fn on_commit_verdict(
        &mut self,
        now: Ts,
        info: &EndOfTurnInfo,
        decision: &CommitDecision,
    ) -> Effects {
        let mut eff = Effects::new();
        let Phase::Committed { turn_id, speaking } = &self.phase else { return eff };
        if *turn_id != info.turn_id {
            return eff;
        }
        let speaking = *speaking;

        let committed = decision.committed;
        let drops_transcript = match decision.verdict {
            CommitVerdict::Reply
            | CommitVerdict::LateTranscript
            | CommitVerdict::Dropped { .. }
            | CommitVerdict::Empty
            | CommitVerdict::TooFewWords => true,
            CommitVerdict::RealtimeBackchannel => false,
        };
        if committed || drops_transcript {
            self.audio_transcript.clear();
            self.final_confidences.clear();
        }
        // The opportunity held for this verdict: surfaced on a refusal or `Dropped` (no reply can
        // collide with it), dropped on a commit that replies.
        let deferred = self.deferred_opportunity.take();
        let no_reply = !committed || matches!(decision.verdict, CommitVerdict::Dropped { .. });
        if no_reply && let Some(opportunity) = deferred {
            eff.emit(SessionEvent::AgentBackchannelOpportunity(opportunity));
        }
        if committed {
            // "clear the transcript if the user turn was committed"; the final's instant goes too.
            self.last_final_transcript_time = None;
            self.user_turn_start = None;
            eff.append(self.reset_transcription_timeout());

            // "only reset if there is no new speech", against the snapshot the countdown took.
            if self.last_speaking_time == info.snapshot.last_speaking_time {
                self.speech_start_time = None;
                self.vad_speech_started = false;
                self.last_speaking_time = None;
            }

            if matches!(self.cfg.detector, Detector::Streaming { .. }) {
                eff.flush_turn_detector(FlushReason::TurnCommitted);
            }
        }

        // "reset turn-scoped interruption state once per logical turn (commit or drop)"
        eff.append(self.overlap.step(In::TurnClosed).1);

        // The turn is consumed (`_user_turn_committed = False`); the user may still be talking.
        let next = if speaking {
            Phase::Speaking {
                since: self.speech_start_time.unwrap_or(now),
                prediction: Prediction::Absent,
                declared: false,
            }
        } else {
            Phase::Idle
        };
        eff.append(self.set_phase(next));
        eff
    }

    /// `_commit_user_turn`'s tail, after the driver's wait. `text` stands in for STT and goes
    /// through `promote_interim`, never as a forged final (which double-reported the words).
    pub fn commit_user_turn(&mut self, now: Ts, text: Option<String>, skip_reply: bool) -> Effects {
        let mut eff = Effects::new();
        eff.append(self.promote_interim(text));
        // Past the close's edge nothing waits: commit what the turn holds, or nothing.
        if self.closed && self.audio_transcript.is_empty() {
            return eff;
        }
        eff.append(self.run_eou_detection(now, Start::Manual { skip_reply }));
        eff
    }

    /// Held interim words become the final words (emitted first, then appended; an empty interim
    /// does nothing). `Some(text)` replaces the interim; `Some("")` is `None`.
    fn promote_interim(&mut self, text: Option<String>) -> Effects {
        let mut eff = Effects::new();
        let held = std::mem::take(&mut self.interim_transcript);
        let promoted = match text.filter(|t| !t.is_empty()) {
            Some(text) => text,
            None => held,
        };
        if promoted.is_empty() {
            return eff;
        }
        let mut ev = SpeechEvent::new(SpeechEventType::FinalTranscript);
        ev.alternatives = vec![lk_agents_abi::SpeechData::new("", &promoted)];
        eff.emit(user_transcript(&ev, true, self.speaking_hint()));
        self.audio_transcript = format!("{} {}", self.audio_transcript, promoted).trim().to_owned();
        eff
    }

    /// Closing: cut a running endpointing delay short, or drop it when `commit` is off (an error
    /// close). Nothing happens in `Awaiting`: the reference's inference wait ignores `_closing`.
    pub fn on_closing(&mut self, now: Ts, commit: bool) -> Effects {
        let mut eff = Effects::new();
        self.closed = true;
        if self.turn_limit_timer_armed {
            self.turn_limit_timer_armed = false;
            eff.cancel(TurnTimer::UserTurnDuration);
        }
        match self.phase.delay() {
            Some((run, speaking)) if commit => eff.append(self.commit(now, run, speaking)),
            Some(_) => eff.append(self.retire(now)),
            None => {}
        }
        eff
    }

    // --- the user turn limit ------------------------------------------------------------------------

    /// `_check_user_turn_limit`. Words are counted with `split_character` off (#2366), unlike the
    /// interruption guard, so CJK text hits `max_duration` long before `max_words`.
    pub(super) fn check_user_turn_limit(&mut self, now: Ts, transcript: &str) -> Effects {
        let mut eff = Effects::new();
        let limit = self.cfg.user_turn_limit;
        if limit.max_words.is_none() && limit.max_duration.is_none() {
            return eff;
        }

        if self.turn_tracker.started_at.is_none() {
            let started_at = self.speech_start_time.unwrap_or(now);
            eff.append(self.arm_turn_limit(started_at));
        }
        self.turn_tracker.words += count_words(transcript, WordTokenizerOptions::default());
        self.turn_tracker.transcript =
            format!("{} {}", self.turn_tracker.transcript, transcript).trim().to_owned();

        let duration = now.saturating_duration_since(self.turn_tracker.started_at.unwrap_or(now));
        // `>=` on both, not `>`: a limit of 20 words fires *at* 20.
        let time_exceeded = limit.max_duration.is_some_and(|m| duration >= m);
        let words_exceeded = limit.max_words.is_some_and(|m| self.turn_tracker.words >= m);
        if time_exceeded || words_exceeded {
            eff.append(self.fire_user_turn_limit(now));
        }
        eff
    }

    /// Starts the tracker and its duration timer at the first onset since the agent spoke.
    pub(super) fn arm_turn_limit(&mut self, started_at: Ts) -> Effects {
        let mut eff = Effects::new();
        let limit = self.cfg.user_turn_limit;
        if limit.max_words.is_none() && limit.max_duration.is_none() {
            return eff;
        }
        if self.turn_tracker.started_at.is_some() {
            return eff;
        }
        self.turn_tracker.started_at = Some(started_at);
        if let Some(max) = limit.max_duration {
            self.turn_limit_timer_armed = true;
            eff.arm(TurnTimer::UserTurnDuration, started_at.saturating_add(max));
        }
        eff
    }

    /// Once per crossing: told again only after the tracker resets.
    fn fire_user_turn_limit(&mut self, now: Ts) -> Effects {
        let mut eff = Effects::new();
        if self.turn_tracker.exceeded {
            return eff;
        }
        let Some(started_at) = self.turn_tracker.started_at else { return eff };
        self.turn_tracker.exceeded = true;
        eff.emit(SessionEvent::UserTurnExceeded(UserTurnExceededEvent {
            transcript: self.current_transcript(),
            accumulated_transcript: self.turn_tracker.transcript.clone(),
            accumulated_word_count: self.turn_tracker.words,
            duration: now.saturating_duration_since(started_at),
        }));
        eff
    }

    pub(super) fn reset_turn_limit(&mut self) -> Effects {
        let mut eff = Effects::new();
        self.turn_tracker = UserTurnTracker::default();
        if self.turn_limit_timer_armed {
            self.turn_limit_timer_armed = false;
            eff.cancel(TurnTimer::UserTurnDuration);
        }
        eff
    }

    // --- the transcription timeout ------------------------------------------------------------

    /// `_arm_transcription_timeout`. `delay` (VAD silence already elapsed) is subtracted so the
    /// window runs from when the user stopped.
    pub(super) fn arm_transcription_timeout(
        &mut self,
        now: Ts,
        speech_duration: Duration,
        delay: Duration,
    ) -> Effects {
        let mut eff = Effects::new();
        let Some(timeout) = self.cfg.transcription_timeout else {
            return eff;
        };
        if self.turn_transcript_received {
            return eff;
        }
        // Accumulated across segments (#6182's `speech_duration < 0.7` filter needs the total).
        self.turn_speech_duration += speech_duration;
        eff.append(self.cancel_transcription_timeout());
        let remaining = timeout.saturating_sub(delay);
        self.transcription_timeout_armed = true;
        eff.arm(TurnTimer::TranscriptionTimeout, now.saturating_add(remaining));
        eff
    }

    pub(super) fn cancel_transcription_timeout(&mut self) -> Effects {
        let mut eff = Effects::new();
        if self.transcription_timeout_armed {
            self.transcription_timeout_armed = false;
            eff.cancel(TurnTimer::TranscriptionTimeout);
        }
        eff
    }

    fn reset_transcription_timeout(&mut self) -> Effects {
        let mut eff = Effects::new();
        eff.append(self.cancel_transcription_timeout());
        self.turn_speech_duration = Duration::ZERO;
        self.turn_transcript_received = false;
        eff
    }

    pub(super) fn mark_turn_transcribed(&mut self) -> Effects {
        let mut eff = Effects::new();
        self.turn_transcript_received = true;
        eff.append(self.cancel_transcription_timeout());
        eff
    }

    pub(super) fn cancel_backchannel_boundary(&mut self) -> Effects {
        let mut eff = Effects::new();
        if self.overlap.boundary_armed() {
            eff.append(self.overlap.step(In::Boundary { armed: false }).1);
            eff.cancel(TurnTimer::BackchannelBoundary);
        }
        eff
    }

    // --- the countdown ----------------------------------------------------------------------------

    /// `_run_eou_detection`: start or restart the countdown. Legal from every phase; a fresh turn
    /// is the only way out of `Committed`.
    pub(super) fn run_eou_detection(&mut self, now: Ts, start: Start) -> Effects {
        let mut eff = Effects::new();
        let manual = matches!(start, Start::Manual { .. });
        let declared = match start {
            Start::Manual { .. } => true,
            Start::Auto { declared } => declared || self.declared(),
        };
        // No transcript yet: the turn doesn't start, but the owner's declaration is remembered so
        // the words commit at once when they land.
        if self.cfg.stt_enabled && self.audio_transcript.is_empty() && !manual {
            let next = match self.phase {
                Phase::Idle => Some(Phase::Ended {
                    anchor: self.last_speaking_time.unwrap_or(now),
                    prediction: Prediction::Absent,
                    declared,
                }),
                Phase::Speaking { since, prediction, .. } => {
                    Some(Phase::Speaking { since, prediction, declared })
                }
                Phase::Ended { anchor, prediction, .. } => {
                    Some(Phase::Ended { anchor, prediction, declared })
                }
                // A rejected trigger leaves a running countdown alone.
                Phase::Awaiting { .. }
                | Phase::Delaying { .. }
                | Phase::ManualCommitted { .. }
                | Phase::Committed { .. } => None,
            };
            if let Some(next) = next {
                eff.append(self.set_phase(next));
            }
            return eff;
        }

        // Snapshot before `_ensure_user_turn_span()`: the reference passes the start as a task
        // argument, so a countdown started before any span snapshots `None`.
        let snapshot_turn_start = self.user_turn_start;
        self.ensure_user_turn(now, None);
        // The opportunity held for an unanswered `Committed` turn is stale.
        self.deferred_opportunity = None;
        let run = Run {
            snapshot: TurnSnapshot {
                last_speaking_time: self.last_speaking_time,
                last_final_transcript_time: self.last_final_transcript_time,
                // `_user_turn_start` (the turn's earliest onset), not `_speech_start_time`.
                speech_start_time: snapshot_turn_start,
            },
            skip_reply: match start {
                Start::Auto { .. } => false,
                Start::Manual { skip_reply } => skip_reply,
            },
            min_delay: self.endpointing.min_delay(),
            max_delay: self.endpointing.max_delay(),
            // A final with no onset since the previous commit: the turn it belongs to has ended.
            late: !manual && self.turn_seq > 0 && snapshot_turn_start.is_none(),
            declared,
        };

        // Whatever the phase held: the streaming detector's answer, and a countdown to supersede.
        let prediction = self.prediction();
        let speaking = self.user_speaking();
        if self.phase.is_running() {
            eff.decide(Decision::TurnSettled(TurnSettled::Superseded));
        }

        if manual {
            // The detector is disabled for manual turn detection.
            eff.append(self.cancel_prediction(prediction));
            let fire_at = anchored(now, run.snapshot.last_speaking_time, run.min_delay);
            // A closed turn arms nothing: the close's trailing commit is the words as they stand.
            if fire_at > now && !self.closed {
                eff.append(self.set_phase(Phase::ManualCommitted { run, fire_at, speaking }));
            } else {
                eff.append(self.commit(now, run, speaking && !self.closed));
            }
            return eff;
        }

        // A batch detector with nothing to score is not asked.
        let usable = (!self.audio_transcript.is_empty()
            || matches!(self.cfg.detector, Detector::Streaming { .. }))
            && self.thresholds.supports(self.last_language.as_ref());

        if !usable {
            eff.append(self.enter_delaying(now, run, run.min_delay, None, None));
            return eff;
        }

        match self.cfg.detector {
            Detector::None => eff.append(self.enter_delaying(now, run, run.min_delay, None, None)),
            Detector::Batch => {
                let req = self.mint_req();
                eff.request_eot(req);
                eff.append(self.set_phase(Phase::Awaiting { run, req, deadline: None }));
            }
            Detector::Streaming { prediction_timeout } => match prediction {
                // `._bounce_eou_task`: no request in flight, so commit on `min_delay` with no
                // prediction.
                Prediction::Absent => {
                    eff.append(self.enter_delaying(now, run, run.min_delay, None, None))
                }
                Prediction::Pending(req) => {
                    let deadline = now.saturating_add(prediction_timeout);
                    eff.append(self.set_phase(Phase::Awaiting {
                        run,
                        req,
                        deadline: Some(deadline),
                    }));
                }
                // `from_cache`: resolves without suspending, so no timer is ever armed.
                Prediction::Ready(p) => eff.append(self.resolve(now, run, p)),
            },
        }
        eff
    }

    /// The prediction path, shared by a live prediction and a cached one: `-> Delaying`.
    fn resolve(&mut self, now: Ts, run: Run, p: EotPrediction) -> Effects {
        let mut eff = Effects::new();
        // Evaluated with the language as it is now; `None` when the detector can't score it.
        let threshold = self.thresholds.lookup(self.last_language.as_ref());
        let probability = p.end_of_turn_probability;

        // Strictly less than: exactly on the threshold gets `min_delay`.
        let delay = match threshold {
            Some(t) if probability < t => run.max_delay,
            Some(_) | None => run.min_delay,
        };

        // Already due means `enter_delaying` commits this cycle, and an opportunity beside a
        // commit would collide: hold it for the verdict (a refused turn stays open).
        let committing_now = anchored(now, run.snapshot.last_speaking_time, delay) <= now;
        if let Some(t) = threshold {
            let announced_delay = run
                .snapshot
                .last_speaking_time
                .map(|t| now.saturating_duration_since(t))
                .unwrap_or_default();
            eff.append(self.announce(&p, t, announced_delay, committing_now));
        }

        eff.append(self.enter_delaying(now, run, delay, Some(p), threshold));
        eff
    }

    /// Surface the prediction and its backchannel opportunity (held when `committing_now`).
    fn announce(
        &mut self,
        p: &EotPrediction,
        threshold: f64,
        delay: Duration,
        committing_now: bool,
    ) -> Effects {
        let mut eff = Effects::new();
        let probability = p.end_of_turn_probability;
        // Only the streaming path can hand the same prediction over twice.
        if matches!(self.cfg.detector, Detector::Streaming { .. })
            && self.last_emitted.as_ref() == Some(p)
        {
            return eff;
        }
        self.last_emitted = Some(*p);

        eff.emit(SessionEvent::EotPrediction(EotPredictionEvent {
            probability,
            threshold,
            inference_duration: p.inference_duration.unwrap_or(0.0),
            delay,
        }));

        // The opportunity uses its own threshold (`>=`) and exists only while the turn is open.
        let language = self.last_language.as_ref();
        let bc_threshold = self.thresholds.lookup_backchannel(language);
        if let (Some(bp), Some(bt)) = (p.backchannel_probability, bc_threshold)
            && bp >= bt
            && probability < threshold
        {
            let opportunity = AgentBackchannelOpportunityEvent {
                probability: bp,
                threshold: bt,
                end_of_turn_probability: probability,
                end_of_turn_threshold: threshold,
                language: language.map(|l| l.as_str().to_owned()),
            };
            if committing_now {
                self.deferred_opportunity = Some(opportunity);
            } else {
                eff.emit(SessionEvent::AgentBackchannelOpportunity(opportunity));
            }
        }
        eff
    }

    /// `-> Delaying`, or straight to `Committed` when the deadline is already behind us.
    fn enter_delaying(
        &mut self,
        now: Ts,
        run: Run,
        delay: Duration,
        prediction: Option<EotPrediction>,
        threshold: Option<f64>,
    ) -> Effects {
        let mut eff = Effects::new();
        let fire_at = anchored(now, run.snapshot.last_speaking_time, delay);
        // `if extra_sleep > 0`: a passed deadline is not slept on.
        if fire_at > now {
            eff.append(self.set_phase(Phase::Delaying {
                run,
                fire_at,
                delay,
                prediction,
                threshold,
            }));
        } else {
            eff.append(self.commit(now, run, false));
        }
        eff
    }

    /// `-> Committed`: the turn goes to the driver as a decision, in order with the rest.
    fn commit(&mut self, now: Ts, run: Run, speaking: bool) -> Effects {
        let mut eff = Effects::new();
        self.turn_seq += 1;
        let turn_id = TurnId(self.turn_seq);
        let info = EndOfTurnInfo {
            turn_id,
            owner: self.cfg.turn_detection_mode,
            stt_enabled: self.cfg.stt_enabled,
            skip_reply: run.skip_reply,
            new_transcript: self.audio_transcript.clone(),
            transcript_confidence: self.average_confidence(),
            metrics: compute_end_of_turn_metrics(
                run.snapshot.speech_start_time,
                run.snapshot.last_speaking_time,
                run.snapshot.last_final_transcript_time,
                now,
            ),
            backchannel_over_agent: self.overlap.backchannel_over_agent(),
            late: run.late,
            snapshot: run.snapshot,
        };
        eff.append(self.set_phase(Phase::Committed { turn_id, speaking }));
        eff.decide(Decision::EndOfTurn(info));
        eff.decide(Decision::TurnSettled(TurnSettled::Decided));
        eff
    }
}

/// The deadline is anchored on when the user stopped speaking (`._bounce_eou_task`), so a slow
/// detector eats into the delay. `Ts::ZERO` is a real instant, so no falsy-zero test.
fn anchored(now: Ts, last_speaking_time: Option<Ts>, delay: Duration) -> Ts {
    match last_speaking_time {
        Some(t) => t.saturating_add(delay),
        None => now.saturating_add(delay),
    }
}
