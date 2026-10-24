//! The user's speech as `AudioRecognition` hears it: STT and VAD events, the realtime model's
//! turn taking, agent speech and overlap verdicts, and the hold-flush buffer.

use super::overlap::{In, OverlapRefusal};
use super::{
    AudioRecognition, Detector, InterruptionSignal, MIN_LANGUAGE_DETECTION_LENGTH,
    MIN_SILENCE_DURATION, NON_SPECIFIC_LANGUAGE_CODES, OverlapVerdict, Phase, Prediction,
    ServerSpeech, Start, TurnDetectionMode, user_transcript,
};
use crate::agent_session::cycle::{Decision, Effect, Effects, ProviderCall, TurnTimer};
use crate::events::{
    EndOfSpeechEvent, PreemptiveGenerationEvent, SessionEvent, StartOfSpeechEvent,
};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::{SpeechEvent, SpeechEventType, Ts, VadEvent, VadEventType};
use std::time::Duration;

impl AudioRecognition {
    // --- STT ----------------------------------------------------------------------------------

    /// `_on_stt_event`.
    pub fn on_stt_event(&mut self, now: Ts, ev: &SpeechEvent) -> Effects {
        let mut eff = Effects::new();
        if self.closed {
            eff.append(self.accumulate_closed(now, ev));
            return eff;
        }
        // In manual mode a committed turn stops accepting transcripts, unless a countdown still
        // runs and this is not an interim (the commit is waiting on that final).
        if self.cfg.turn_detection_mode == TurnDetectionMode::Manual {
            let dropped = match &self.phase {
                Phase::ManualCommitted { .. } => ev.kind == SpeechEventType::InterimTranscript,
                Phase::Committed { .. } => true,
                Phase::Idle
                | Phase::Speaking { .. }
                | Phase::Ended { .. }
                | Phase::Awaiting { .. }
                | Phase::Delaying { .. } => false,
            };
            if dropped {
                return eff;
            }
        }

        // "Keep the timeout armed for interim and preflight transcripts". An empty final does not
        // mark the turn transcribed.
        if ev.kind == SpeechEventType::FinalTranscript
            && ev.best_alternative().is_some_and(|a| !a.text.is_empty())
        {
            eff.append(self.mark_turn_transcribed());
        }

        // `RECOGNITION_USAGE` carries no transcript, so it bypasses the hold buffer.
        if ev.kind != SpeechEventType::RecognitionUsage && self.cfg.interruption_enabled {
            if self.should_hold_stt_event(now, ev) {
                self.transcript_buffer.push_back(ev.clone());
                return eff;
            }
            if !self.transcript_buffer.is_empty() {
                eff.append(self.flush_held_transcripts(now));
                // no return: the new event is still processed normally
            }
        }

        // `min(..., now)`: a fast provider clock must not place speech in the future.
        let stt_last_speaking_time = match (ev.best_alternative(), self.input_started_at) {
            (Some(a), Some(started)) if a.end_time > 0.0 => {
                std::cmp::min(started.saturating_add(secs(a.end_time)), now)
            }
            (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) | (None, None) => now,
        };

        match ev.kind {
            SpeechEventType::FinalTranscript => {
                eff.append(self.on_final_transcript(now, ev, stt_last_speaking_time))
            }
            SpeechEventType::PreflightTranscript => {
                eff.append(self.on_preflight_transcript(now, ev, stt_last_speaking_time))
            }
            SpeechEventType::InterimTranscript => {
                let speaking = self.speaking_hint();
                eff.emit(user_transcript(ev, false, speaking));
                if let Some(a) = ev.best_alternative() {
                    self.interim_transcript = a.text.clone();
                }
            }
            SpeechEventType::EndOfSpeech => {
                if self.cfg.turn_detection_mode == TurnDetectionMode::Stt {
                    eff.append(self.on_stt_end_of_speech(now, stt_last_speaking_time));
                }
            }
            SpeechEventType::StartOfSpeech => {
                if self.cfg.turn_detection_mode == TurnDetectionMode::Stt {
                    eff.append(self.on_stt_start_of_speech(now, ev, stt_last_speaking_time));
                }
            }
            SpeechEventType::RecognitionUsage => {}
        }
        eff
    }

    /// Words after the close's edge: shown and kept for the trailing commit, nothing else.
    fn accumulate_closed(&mut self, now: Ts, ev: &SpeechEvent) -> Effects {
        let mut eff = Effects::new();
        match ev.kind {
            SpeechEventType::FinalTranscript => {
                let Some(alt) = ev.best_alternative() else { return eff };
                if alt.text.is_empty() {
                    return eff;
                }
                eff.emit(user_transcript(ev, true, self.speaking_hint()));
                self.last_final_transcript_time = Some(now);
                self.append_final(&alt.text);
            }
            // Kept for the trailing commit, not shown: a late interim after the edge is dropped.
            SpeechEventType::InterimTranscript => {
                if let Some(a) = ev.best_alternative() {
                    self.interim_transcript = a.text.clone();
                }
            }
            SpeechEventType::PreflightTranscript
            | SpeechEventType::EndOfSpeech
            | SpeechEventType::StartOfSpeech
            | SpeechEventType::RecognitionUsage => {}
        }
        eff
    }

    fn on_final_transcript(
        &mut self,
        now: Ts,
        ev: &SpeechEvent,
        stt_last_speaking_time: Ts,
    ) -> Effects {
        let mut eff = Effects::new();
        let Some(alt) = ev.best_alternative() else { return eff };
        self.update_last_language(&alt.language, &alt.text);

        // Set before the empty-transcript bail-out so the manual commit stops waiting even on an
        // empty final (a driver-side event).
        if alt.text.is_empty() {
            return eff;
        }

        eff.emit(user_transcript(ev, true, self.speaking_hint()));

        self.last_final_transcript_time = Some(now);
        self.append_final(&alt.text);
        self.final_confidences.push(alt.confidence);

        // #6667: compared after folding the final in, so a formatting-only change doesn't
        // re-trigger.
        let transcript_changed = self.audio_transcript != self.preflight_transcript;
        self.preflight_transcript.clear();

        if self.stt_timing_wins() {
            self.last_speaking_time = Some(stt_last_speaking_time);
        }

        eff.append(self.check_user_turn_limit(now, &alt.text));

        if self.cfg.turn_detection_mode == TurnDetectionMode::Vad || self.declared() {
            if transcript_changed {
                let confidence = self.average_confidence();
                eff.emit(SessionEvent::PreemptiveGeneration(PreemptiveGenerationEvent {
                    new_transcript: self.audio_transcript.clone(),
                    transcript_confidence: confidence,
                    started_speaking_at: self.speech_start_time,
                }));
            }
            // Only once the user stopped. A manual commit waiting on this final restarts as the
            // same commit, `skip_reply` included.
            let start = match &self.phase {
                Phase::ManualCommitted { run, .. } => Start::Manual { skip_reply: run.skip_reply },
                Phase::Idle
                | Phase::Speaking { .. }
                | Phase::Ended { .. }
                | Phase::Awaiting { .. }
                | Phase::Delaying { .. }
                | Phase::Committed { .. } => Start::Auto { declared: false },
            };
            if !self.user_speaking() {
                eff.append(self.run_eou_detection(now, start));
            }
        }
        eff
    }

    fn on_preflight_transcript(
        &mut self,
        now: Ts,
        ev: &SpeechEvent,
        stt_last_speaking_time: Ts,
    ) -> Effects {
        let mut eff = Effects::new();
        // The interim hook fires before the empty-text bail-out, as in the reference.
        eff.emit(user_transcript(ev, false, self.speaking_hint()));

        let Some(alt) = ev.best_alternative() else { return eff };
        self.update_last_language(&alt.language, &alt.text);
        if alt.text.is_empty() {
            return eff;
        }

        // "still need to increment it as it's used for turn detection" (`._on_stt_event`).
        self.last_final_transcript_time = Some(now);
        self.preflight_transcript =
            format!("{} {}", self.audio_transcript, alt.text).trim_start().to_owned();
        self.interim_transcript = alt.text.clone();

        if self.stt_timing_wins() {
            self.last_speaking_time = Some(stt_last_speaking_time);
        }

        if self.cfg.turn_detection_mode != TurnDetectionMode::Manual || self.declared() {
            // The preflight's confidence joins the average without being committed.
            let sum: f64 = self.final_confidences.iter().sum::<f64>() + alt.confidence;
            let confidence = sum / (self.final_confidences.len() + 1) as f64;
            eff.emit(SessionEvent::PreemptiveGeneration(PreemptiveGenerationEvent {
                new_transcript: self.preflight_transcript.clone(),
                transcript_confidence: confidence,
                started_speaking_at: self.speech_start_time,
            }));
        }
        eff
    }

    fn on_stt_end_of_speech(&mut self, now: Ts, stt_last_speaking_time: Ts) -> Effects {
        let mut eff = Effects::new();
        self.ensure_user_turn(now, None);
        eff.append(self.emit_end_of_speech(EndOfSpeechEvent {
            at: now,
            from_vad: false,
            silence_duration: 0.0,
        }));
        // The provider declared the end even if a VAD closed the speech first.
        eff.decide(Decision::TurnDeclared { owner: TurnDetectionMode::Stt });
        eff.append(self.end_of_speech(now));

        // Reset the VAD so an incorrect STT end of turn can be corrected by a VAD interruption.
        if self.cfg.vad_enabled && self.vad_speech_started {
            eff.push(Effect::Provider(ProviderCall::FlushVad));
        }

        // The STT declared the end: `_user_turn_committed = True`.
        eff.append(self.end_speaking(now, true));
        // Under a host commit's window, the provider's end neither re-anchors nor restarts it.
        if matches!(self.phase, Phase::ManualCommitted { .. }) {
            return eff;
        }
        if self.stt_timing_wins() {
            self.last_speaking_time = Some(stt_last_speaking_time);
        }
        eff.append(self.run_eou_detection(now, Start::Auto { declared: true }));
        eff
    }

    fn on_stt_start_of_speech(
        &mut self,
        now: Ts,
        ev: &SpeechEvent,
        stt_last_speaking_time: Ts,
    ) -> Effects {
        let mut eff = Effects::new();
        // `ev.speech_start_time or time.time()`: a `0.0` falls back to arrival time.
        if self.speech_start_time.is_none() {
            let onset = ev
                .speech_start_time
                .filter(|w| *w != 0.0)
                .and_then(|w| self.wall_anchor.map(|a| a.ts(w)))
                .unwrap_or(now);
            self.speech_start_time = Some(onset);
        }
        let started_at = self.speech_start_time.unwrap_or(now);
        self.ensure_user_turn(now, Some(started_at));

        eff.append(self.emit_start_of_speech(StartOfSpeechEvent {
            at: started_at,
            speech_duration: Duration::ZERO,
            from_vad: false,
        }));
        eff.append(self.start_of_speech(started_at, Duration::ZERO));
        eff.append(self.arm_turn_limit(started_at));

        self.last_speaking_time = Some(stt_last_speaking_time);
        eff.append(self.begin_speaking(started_at));
        eff
    }

    // --- VAD ----------------------------------------------------------------------------------

    /// `_on_vad_event`.
    pub fn on_vad_event(&mut self, now: Ts, ev: &VadEvent) -> Effects {
        let mut eff = Effects::new();
        if self.closed {
            return eff;
        }
        match ev.kind {
            VadEventType::StartOfSpeech => {
                // VAD reports after the fact, so the onset is back-dated. Every endpointing delay
                // is measured from this anchor.
                let started_at =
                    now.saturating_sub(secs(ev.speech_duration + ev.inference_duration));
                eff.append(self.user_speech_started(
                    now,
                    started_at,
                    secs(ev.speech_duration),
                    true,
                ));
            }

            VadEventType::InferenceDone => {
                // No emit: this fires ~30x/s. The driver fans it out itself.
                if ev.raw_accumulated_speech > 0.0 {
                    // "for metrics, get the *earliest* signal of speech as possible"
                    self.last_speaking_time = Some(now);
                    if self.speech_start_time.is_none() {
                        self.speech_start_time =
                            Some(now.saturating_sub(secs(ev.raw_accumulated_speech)));
                    }
                    // A prediction pending while the user is still in a segment is worthless.
                    if let Phase::Speaking { since, prediction, declared } = &self.phase {
                        let (since, prediction, declared) = (*since, *prediction, *declared);
                        if let Prediction::Pending(req) = prediction {
                            eff.cancel_eot(req, false);
                            eff.append(self.set_phase(Phase::Speaking {
                                since,
                                prediction: Prediction::Absent,
                                declared,
                            }));
                        }
                    }
                }

                if secs(ev.raw_accumulated_silence) >= MIN_SILENCE_DURATION
                    && matches!(self.cfg.detector, Detector::Streaming { .. })
                    && let Phase::Speaking { since, prediction: Prediction::Absent, declared } =
                        &self.phase
                {
                    let (since, declared) = (*since, *declared);
                    let req = self.mint_req();
                    eff.request_eot(req);
                    eff.append(self.set_phase(Phase::Speaking {
                        since,
                        prediction: Prediction::Pending(req),
                        declared,
                    }));
                }
            }

            VadEventType::EndOfSpeech => {
                let vad_speech_started = self.vad_speech_started;
                self.ensure_user_turn(now, None);
                let ended_at =
                    now.saturating_sub(secs(ev.silence_duration + ev.inference_duration));
                // `last_speaking_time` is re-stamped unconditionally (`._on_vad_event`), even under
                // a host commit: the commit's run keeps its own snapshot.
                let silence_duration = ev.silence_duration + ev.inference_duration;
                let end = EndOfSpeechEvent { at: ended_at, from_vad: true, silence_duration };
                eff.append(self.user_speech_ended(end));
                // A host commit owns the countdown: the VAD's end restarts nothing, so the commit's
                // anchor, `skip_reply` and deadline are kept. The reference restarts under a VAD
                // owner.
                let manual_committed = matches!(self.phase, Phase::ManualCommitted { .. });

                // "A committed turn clears `_vad_speech_started` before its late VAD EOS arrives".
                // With no STT no transcript was ever expected.
                if self.cfg.stt_enabled && vad_speech_started {
                    eff.append(self.arm_transcription_timeout(
                        now,
                        secs(ev.speech_duration),
                        secs(ev.silence_duration + ev.inference_duration),
                    ));
                }

                if !manual_committed
                    && (self.cfg.turn_detection_mode == TurnDetectionMode::Vad
                        || (self.cfg.turn_detection_mode == TurnDetectionMode::Stt
                            && self.declared()))
                {
                    eff.append(self.run_eou_detection(now, Start::Auto { declared: false }));
                }
            }
        }
        eff
    }

    /// `on_end_of_speech(None)`: the mic was detached or muted under a speaking user. No countdown
    /// and no `_ensure_user_turn_span`, as in the reference; an open turn stays open.
    pub fn on_host_end_of_speech(&mut self, now: Ts) -> Effects {
        let mut eff = Effects::new();
        if self.closed || !self.user_speaking() {
            return eff;
        }
        eff.append(self.user_speech_ended(EndOfSpeechEvent {
            at: now,
            from_vad: false,
            silence_duration: 0.0,
        }));
        eff
    }

    fn user_speech_ended(&mut self, ev: EndOfSpeechEvent) -> Effects {
        let mut eff = Effects::new();
        let ended_at = ev.at;
        eff.append(self.emit_end_of_speech(ev));
        eff.append(self.end_of_speech(ended_at));
        self.vad_speech_started = false;
        eff.append(self.end_speaking(ended_at, false));
        self.last_speaking_time = Some(ended_at);
        eff
    }

    /// The local VAD's onset and the server's, one path.
    fn user_speech_started(
        &mut self,
        now: Ts,
        started_at: Ts,
        speech_duration: Duration,
        from_vad: bool,
    ) -> Effects {
        let mut eff = Effects::new();
        if !self.vad_speech_started {
            self.speech_start_time = Some(started_at);
            self.vad_speech_started = true;
        }

        eff.append(self.cancel_transcription_timeout());
        self.ensure_user_turn(now, Some(started_at));
        eff.append(self.emit_start_of_speech(StartOfSpeechEvent {
            at: started_at,
            speech_duration,
            from_vad,
        }));
        eff.append(self.start_of_speech(started_at, speech_duration));
        eff.append(self.arm_turn_limit(started_at));

        // Invalidates any prediction and countdown, unless a manual commit owns the turn.
        eff.append(self.begin_speaking(started_at));
        eff
    }

    // --- the realtime model's turn taking -------------------------------------------------------

    /// Server VAD and transcripts under [`TurnDetectionMode::RealtimeLlm`]: the same anchors, emits
    /// and `user_state` as local VAD, but no countdown.
    pub fn on_server(&mut self, now: Ts, ev: ServerSpeech) -> Effects {
        let mut eff = Effects::new();
        match ev {
            ServerSpeech::Started => {
                eff.append(self.user_speech_started(now, now, Duration::ZERO, true))
            }
            ServerSpeech::Stopped => {
                self.ensure_user_turn(now, None);
                let ev = EndOfSpeechEvent { at: now, from_vad: true, silence_duration: 0.0 };
                eff.append(self.user_speech_ended(ev));
            }
            ServerSpeech::Transcription { text, is_final } => {
                let mut ev = SpeechEvent::new(if is_final {
                    SpeechEventType::FinalTranscript
                } else {
                    SpeechEventType::InterimTranscript
                });
                ev.alternatives = vec![lk_agents_abi::SpeechData::new("", &text)];
                if is_final {
                    if text.is_empty() {
                        return eff;
                    }
                    eff.emit(user_transcript(&ev, true, self.speaking_hint()));
                    self.last_final_transcript_time = Some(now);
                    self.append_final(&text);
                } else {
                    eff.emit(user_transcript(&ev, false, self.speaking_hint()));
                    self.interim_transcript = text;
                }
            }
        }
        eff
    }

    // --- agent speech, written from outside -----------------------------------------------------

    /// `_on_start_of_agent_speech`: follows audible playout, so resuming paused playout starts a
    /// new interval (`PausedSpeech` calls this on resume).
    pub fn on_start_of_agent_speech(&mut self, now: Ts, started_at: Ts) -> Effects {
        let mut eff = Effects::new();
        self.agent_speaking = true;
        self.agent_speech_started_at = Some(started_at);
        self.endpointing.on_start_of_agent_speech(started_at);

        // The user-turn limit accumulates between agent turns, so agent speech resets it.
        eff.append(self.reset_turn_limit());

        // A zero cooldown arms nothing.
        if let Some((start_cooldown, _)) = self.cfg.backchannel_boundary
            && !start_cooldown.is_zero()
        {
            eff.append(self.cancel_backchannel_boundary());
            eff.append(self.overlap.step(In::Boundary { armed: true }).1);
            eff.arm(TurnTimer::BackchannelBoundary, now.saturating_add(start_cooldown));
        }

        if self.adaptive_interruption_active() {
            eff.signal(InterruptionSignal::AgentSpeechStarted);
        }
        // Agent speech starting inside the user's speech is an overlap: tell the detector now.
        if self.user_speaking() {
            eff.append(self.start_of_overlap_speech(started_at, Duration::ZERO));
        }
        eff
    }

    /// `_on_end_of_agent_speech`. Pausing playout ends audible speech, so this can run while the
    /// generation is alive.
    pub fn on_end_of_agent_speech(&mut self, now: Ts, ignore_user_transcript_until: Ts) -> Effects {
        let mut eff = Effects::new();
        eff.append(self.cancel_backchannel_boundary());

        let was_speaking = std::mem::take(&mut self.agent_speaking);
        if was_speaking {
            self.endpointing.on_end_of_agent_speech(now);
        }
        if !self.adaptive_interruption_active() {
            return eff;
        }
        if was_speaking {
            // close any unresolved overlap before resetting the detector
            eff.append(self.overlap.step(In::Ended { ended_at: now, agent_ended: true }).1);
        }
        // The overlap dies with the speech it overlapped: a later verdict must not stop what plays
        // next.
        eff.append(self.overlap.step(In::AgentSpeechEnded).1);
        eff.signal(InterruptionSignal::AgentSpeechEnded);

        if was_speaking {
            let end_cooldown = self.end_cooldown();
            // `min(passed, existing)`: a second end can only shrink the hold.
            let ignore_until = self
                .ignore_user_transcript_until
                .map_or(ignore_user_transcript_until, |e| e.min(ignore_user_transcript_until));
            self.ignore_user_transcript_until = Some(ignore_until.saturating_sub(end_cooldown));
            // `agent_speaking` is already clear, or every re-emitted event would go back into
            // the buffer via `should_hold_stt_event`.
            eff.append(self.flush_held_transcripts(now));
        }
        eff
    }

    // --- overlap verdicts -----------------------------------------------------------------------

    /// `_on_overlap_speech_event`. `Err` names why a verdict was not applied, for the journal.
    pub fn on_overlap_verdict(
        &mut self,
        req: Option<ReqId>,
        v: &OverlapVerdict,
    ) -> (Result<(), OverlapRefusal>, Effects) {
        let mut eff = Effects::new();
        let user_speaking = self.user_speaking();
        let verdict = In::Verdict { req, verdict: *v, user_speaking };
        let (refusal, stepped) = self.overlap.step(verdict);
        eff.append(stepped);
        let result = match refusal {
            Some(refusal) => Err(refusal),
            None => Ok(()),
        };
        (result, eff)
    }

    // --- the hold-flush buffer (calculators) ---------------------------------------------------

    /// `_should_hold_stt_event`. `&mut`: a `START_OF_SPEECH` clears the ignore window on the way
    /// through ("better than swallowing the transcript").
    pub fn should_hold_stt_event(&mut self, now: Ts, ev: &SpeechEvent) -> bool {
        if !self.cfg.interruption_enabled {
            return false;
        }
        if self.agent_speaking {
            return true;
        }
        if ev.kind == SpeechEventType::StartOfSpeech {
            self.ignore_user_transcript_until = None;
            return false;
        }
        if self.ignore_user_transcript_until.is_none() {
            return false;
        }
        // "sentinel events are always held until we have something concrete to release them".
        let Some(alt) = ev.best_alternative() else {
            return true;
        };
        let Some(started) = self.input_started_at else {
            // "most vendors don't set timestamps properly".
            return false;
        };
        if alt.start_time == 0.0 && alt.end_time == 0.0 {
            return false;
        }
        alt.start_time > 0.0
            && self.within_ignore_window(now, started.saturating_add(secs(alt.start_time)))
    }

    /// `_within_ignore_window`, strict on both sides; the upper bound is `min(now, ignore_until)`.
    pub fn within_ignore_window(&self, now: Ts, event_time: Ts) -> bool {
        let Some(until) = self.ignore_user_transcript_until else {
            return false;
        };
        let lower = self.agent_speech_started_at.unwrap_or(Ts::ZERO);
        let upper = std::cmp::min(now, until);
        lower < event_time && event_time < upper
    }

    /// The scan half of `_flush_held_transcripts`. Every in-window event resets the start, so a
    /// fully in-window buffer emits nothing (otherwise the agent's echo would be re-emitted).
    fn plan_flush(&self, now: Ts) -> Option<usize> {
        if !self.cfg.interruption_enabled || self.ignore_user_transcript_until.is_none() {
            return None;
        }
        let started = self.input_started_at?;

        let mut emit_from: Option<usize> = None;
        for (i, ev) in self.transcript_buffer.iter().enumerate() {
            let Some(alt) = ev.best_alternative() else {
                // "always try to emit from a sentinel event"
                emit_from = Some(emit_from.map_or(i, |j| j.min(i)));
                continue;
            };
            if alt.start_time == 0.0 && alt.end_time == 0.0 {
                // No timestamps at all: drop the buffer rather than guess.
                return None;
            }
            if alt.end_time > 0.0
                && self.within_ignore_window(now, started.saturating_add(secs(alt.end_time)))
            {
                emit_from = None;
            } else {
                return Some(emit_from.map_or(i, |j| j.min(i)));
            }
        }
        None
    }

    fn flush_held_transcripts(&mut self, now: Ts) -> Effects {
        let mut eff = Effects::new();
        let events: Vec<SpeechEvent> = match self.plan_flush(now) {
            Some(from) => self.transcript_buffer.iter().skip(from).cloned().collect(),
            None => Vec::new(),
        };

        // "reset before emitting to avoid recursive calls"
        self.reset_interruption_detection();
        for ev in events {
            eff.append(self.on_stt_event(now, &ev));
        }
        eff
    }

    /// `_reset_interruption_detection`.
    fn reset_interruption_detection(&mut self) {
        self.transcript_buffer.clear();
        self.ignore_user_transcript_until = None;
        // "keep the anchor while a newer agent-speech cycle is active".
        if !self.agent_speaking {
            self.agent_speech_started_at = None;
        }
    }

    // --- speech segments ------------------------------------------------------------------------

    /// `_on_start_of_speech`. An onset is announced only when the user wasn't already speaking;
    /// two legs may report one segment and the first announces it. Read before the phase moves.
    fn emit_start_of_speech(&self, ev: StartOfSpeechEvent) -> Effects {
        let mut eff = Effects::new();
        if !self.user_speaking() {
            eff.emit(SessionEvent::StartOfSpeech(ev));
        }
        eff
    }

    fn emit_end_of_speech(&self, ev: EndOfSpeechEvent) -> Effects {
        let mut eff = Effects::new();
        if self.user_speaking() {
            eff.emit(SessionEvent::EndOfSpeech(ev));
        }
        eff
    }

    fn start_of_speech(&mut self, started_at: Ts, speech_duration: Duration) -> Effects {
        let mut eff = Effects::new();
        self.endpointing.on_start_of_speech(started_at, self.agent_speaking);
        eff.append(self.overlap.step(In::UserOnset { agent_speaking: self.agent_speaking }).1);
        eff.append(self.start_of_overlap_speech(started_at, speech_duration));
        eff
    }

    /// `_on_start_of_overlap_speech`.
    fn start_of_overlap_speech(&mut self, started_at: Ts, speech_duration: Duration) -> Effects {
        let mut eff = Effects::new();
        if !self.adaptive_interruption_active() || !self.agent_speaking {
            return eff;
        }
        eff.append(self.overlap.step(In::Started { started_at, speech_duration }).1);
        eff
    }

    /// `_on_end_of_speech`.
    fn end_of_speech(&mut self, ended_at: Ts) -> Effects {
        let mut eff = Effects::new();
        // The detector said "not an interruption" while the agent talked: exclude the segment.
        let should_ignore = self.cfg.interruption_enabled
            && !self.overlap.interruption_detected()
            && self.agent_speaking;
        if self.user_speaking()
            && let Some(min_delay) = self.endpointing.on_end_of_speech(ended_at, should_ignore)
        {
            eff.decide(Decision::EndpointingDelayUpdated { min_delay });
        }
        if self.adaptive_interruption_active() {
            eff.append(self.overlap.step(In::Ended { ended_at, agent_ended: false }).1);
        }
        eff
    }

    // --- helpers -----------------------------------------------------------------------------

    fn adaptive_interruption_active(&self) -> bool {
        self.cfg.interruption_enabled && self.interruption_active
    }

    /// "vad disabled or missed a speech, use stt timestamp" (`._on_stt_event`).
    fn stt_timing_wins(&self) -> bool {
        !self.cfg.vad_enabled || self.cfg.using_default_vad || self.last_speaking_time.is_none()
    }

    /// The transcript hooks' `speaking=`. `None` means no trusted VAD; collapsing it to `false`
    /// would make every default-VAD session look silent.
    pub(super) fn speaking_hint(&self) -> Option<bool> {
        let trusted_vad = self.cfg.vad_enabled && !self.cfg.using_default_vad;
        (trusted_vad || self.cfg.turn_detection_mode == TurnDetectionMode::Stt)
            .then_some(self.user_speaking())
    }

    fn end_cooldown(&self) -> Duration {
        self.cfg.backchannel_boundary.map(|(_, end)| end).unwrap_or_default()
    }

    pub(super) fn average_confidence(&self) -> f64 {
        if self.final_confidences.is_empty() {
            0.0
        } else {
            self.final_confidences.iter().sum::<f64>() / self.final_confidences.len() as f64
        }
    }

    /// `_update_last_language`. A short transcript never overwrites an established language.
    fn update_last_language(&mut self, language: &LanguageCode, transcript: &str) {
        if language.as_str().is_empty()
            || NON_SPECIFIC_LANGUAGE_CODES.contains(&language.language())
        {
            return;
        }
        let unset = self.last_language.as_ref().is_none_or(|l| l.as_str().is_empty());
        if unset || transcript.chars().count() > MIN_LANGUAGE_DETECTION_LENGTH {
            self.last_language = Some(language.clone());
        }
    }

    /// A final's words join the turn (`+= f" {transcript}"` then `.lstrip()`); the interim goes.
    fn append_final(&mut self, text: &str) {
        self.audio_transcript =
            format!("{} {}", self.audio_transcript, text).trim_start().to_owned();
        self.interim_transcript.clear();
    }

    /// `_ensure_user_turn_span`, minus the span.
    pub(super) fn ensure_user_turn(&mut self, now: Ts, start_time: Option<Ts>) {
        if self.user_turn_start.is_none() {
            self.user_turn_start = Some(start_time.unwrap_or(now));
        }
    }
}

/// Seconds, saturating at zero (a negative provider value would panic `from_secs_f64`).
fn secs(v: f64) -> Duration {
    if v.is_finite() && v > 0.0 { Duration::from_secs_f64(v) } else { Duration::ZERO }
}
