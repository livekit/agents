//! The recognition hooks: STT and VAD input, the user turn and its commit, and the preemptive
//! generation.

use crate::agent_activity::ConfigDiff;
use crate::agent_activity::pause::{BargeTrigger, InterruptOutcome, InterruptRefusal};
use crate::agent_activity::scheduling::{PreemptRefusal, PreemptiveDiscard, SpeechKind, Status};
use crate::agent_activity::speech::SpeechSpec;
use crate::agent_session::cycle::Cycle;
use crate::agent_session::{Driver, TurnCommand};
use crate::audio::{LegId, Tagged};
use crate::audio_recognition::{
    CommitInputs, CommitSpeech, CommitVerdict, EndOfTurnInfo, PendingCommit, Phase, TurnEdits,
    decide_commit,
};
use crate::generation::ReplySource;
use crate::trace::{
    CloseAct, CloseStep, CoreStep, HandoffStep, InputStep, InterruptCause, InterruptStep, Step,
    TurnStep,
};
use lk_agents_abi::turn::detector::DetectorLeg as _;
use lk_agents_abi::voice::events::AgentState;
use lk_agents_abi::{
    ChatContext, ChatMessage, Role, SpeechEvent, SpeechEventType, Ts, VadEvent, VadEventType,
};
use std::time::Duration;

impl Driver {
    /// `input_started_at` (`_STTPipeline.input_started_at`) is stamped on the first recogniser
    /// event after a frame was seen; `wall_anchor` pairs wall and monotonic clocks so server onsets
    /// map without reading a clock.
    pub(crate) fn anchor_stt_stream(&mut self, now: Ts) {
        let turn = self.activity.audio_recognition_mut();
        if turn.wall_anchor().is_none() {
            turn.set_wall_anchor(crate::audio_recognition::WallAnchor {
                wall: self.shared.time.wall_secs(),
                at: now,
            });
        }
        if turn.input_started_at().is_none() {
            // When audio started flowing, not this event's instant.
            if let Some(first) = self.input.first_input_at() {
                turn.set_input_started_at(Some(first));
            }
        }
    }

    pub(crate) fn on_stt(&mut self, now: Ts, ev: Tagged<SpeechEvent>, cycle: &mut Cycle) {
        let Tagged { open, item: ev } = ev;
        self.anchor_stt_stream(now);
        // `RecognizeStream._metrics_monitor_task`: one `STTMetrics` per usage report.
        if ev.kind == SpeechEventType::RecognitionUsage {
            let usage = ev.recognition_usage.clone().unwrap_or_default();
            let id = self.session.stt().map(|s| s.id()).unwrap_or_default();
            let record = lk_agents_abi::metrics::base::SttMetrics {
                label: id.label.clone(),
                request_id: ev.request_id.clone(),
                timestamp: self.shared.time.wall_secs(),
                duration: 0.0,
                audio_duration: usage.audio_duration,
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                streamed: true,
                acquire_time: 0.0,
                connection_reused: false,
                metadata: Some(id.metadata()),
            };
            cycle.machine().emit(crate::events::SessionEvent::MetricsCollected {
                metrics: Box::new(lk_agents_abi::metrics::base::AgentMetrics::Stt(record)),
                speech: None,
            });
            return;
        }
        // An item from a stream the row retired is dropped: the reference's `_update_stt` cancels
        // the old consumer, so a swapped-out recogniser's final reaches nothing. Until the new
        // stream lands the old row is the installed one.
        if self.input.legs().installed_open(LegId::Stt) != Some(open) {
            cycle.trace(InputStep::LegItemDiscarded { leg: LegId::Stt.label(), open });
            return;
        }
        let Some(ev) = self.speaker_filter(open, ev, cycle) else { return };
        // The recogniser's item for the peers (the AMD reads the finals).
        if !self.watch.is_empty() {
            self.watch.publish(crate::agent_session::Watched::Stt(ev.clone()));
        }
        let eff = self.activity.audio_recognition_mut().on_stt_event(now, &ev);
        cycle.machine().append(eff);
        // `_commit_user_turn` waits on `_final_transcript_received`; the commit follows this final.
        if ev.kind == SpeechEventType::FinalTranscript
            && self.activity.audio_recognition().commit_pending()
        {
            self.perform_pending_commit(now, cycle);
        }

        // A final is the barge-in failsafe if VAD failed (`on_final_transcript`); with no VAD a
        // non-empty interim is the onset (`on_interim_transcript`). The gates are `decide_interrupt`'s.
        let words =
            ev.best_alternative().map_or(0, |a| crate::agent_activity::pause::count_min_words(&a.text));
        let has_vad = self.input.has_vad();
        match ev.kind {
            SpeechEventType::FinalTranscript => {
                self.interrupt_by_audio_activity(now, cycle, BargeTrigger::SttFinal { words });
            }
            SpeechEventType::InterimTranscript if !has_vad => {
                self.interrupt_by_audio_activity(now, cycle, BargeTrigger::SttInterim { words });
            }
            SpeechEventType::InterimTranscript
            | SpeechEventType::StartOfSpeech
            | SpeechEventType::PreflightTranscript
            | SpeechEventType::RecognitionUsage
            | SpeechEventType::EndOfSpeech => {}
        }
    }

    /// `StartOfSpeech` pauses playout at once; only `InferenceDone` may stop the agent, gated on
    /// `interruption.min_duration` (`on_vad_inference_done`), so a cough doesn't cut the agent off.
    pub(crate) fn on_vad(&mut self, now: Ts, ev: VadEvent, cycle: &mut Cycle) {
        self.anchor_stt_stream(now);
        // The VAD's edges for the peers (the AMD).
        if !self.watch.is_empty() {
            self.watch.publish(crate::agent_session::Watched::Vad(ev.clone()));
        }
        let eff = self.activity.audio_recognition_mut().on_vad_event(now, &ev);
        cycle.machine().append(eff);

        match ev.kind {
            VadEventType::StartOfSpeech => {
                // Cancel the resume timer (the pause survives), then pause if not already paused.
                self.on_start_of_speech(cycle);
            }
            VadEventType::InferenceDone => {
                // A negative or NaN measurement clamps to zero, which `decide_interrupt` refuses.
                let secs = |x: f64| Duration::try_from_secs_f64(x).unwrap_or(Duration::ZERO);
                self.interrupt_by_audio_activity(
                    now,
                    cycle,
                    BargeTrigger::VadThreshold {
                        duration: secs(ev.speech_duration),
                        raw_silence: secs(ev.raw_accumulated_silence),
                    },
                );
            }
            VadEventType::EndOfSpeech => self.start_false_interruption_timer(now, cycle),
        }
    }

    /// One `_detector` per wrapper stream. Called by `on_stt` before the turn machine.
    pub(crate) fn speaker_filter(
        &mut self,
        open: u32,
        ev: lk_agents_abi::SpeechEvent,
        cycle: &mut Cycle,
    ) -> Option<lk_agents_abi::SpeechEvent> {
        let decision = self.input.filter_speaker(open, ev);
        if let Some((speaker, class)) = decision.changed {
            cycle.trace(InputStep::SpeakerVerdict { speaker, class });
        }
        decision.kept
    }

    pub(crate) fn on_turn(&mut self, now: Ts, cmd: TurnCommand, cycle: &mut Cycle) {
        match cmd {
            TurnCommand::Commit { skip_reply } => {
                cycle.trace(TurnStep::CommitRequested { skip_reply });
                if self.activity.rt_session().is_some() {
                    // Nothing is pushed: the commit means "commit whatever STT already has";
                    // pushing a blank re-sent the whole context with a blank user turn.
                    if !skip_reply {
                        self.realtime_user_turn(now, cycle);
                    }
                    return;
                }
                // `_commit_user_turn`'s tail. The commit decides the turn in this cycle, so no
                // preemptive generation may start for it (it would be a parked speech with no
                // owner).
                cycle.trace(TurnStep::PreemptRefused(
                    crate::agent_activity::scheduling::PreemptRefusal::ManualTurn,
                ));
                let eff =
                    self.activity.audio_recognition_mut().commit_user_turn(now, None, skip_reply);
                cycle.machine().append(eff);
                // A preemptive generation the commit did not adopt has no turn left to wait for.
                self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, cycle);
            }
            TurnCommand::Clear => {
                // The reference's `_clear_user_turn` empties a pending end-of-turn task too, so a
                // turn parked for a handoff goes with the preemptive generation.
                self.activity.hooks_mut().clear_parked_turn();
                self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, cycle);
                // Clearing only unwinds, a running countdown included.
                let eff = self.activity.audio_recognition_mut().clear_user_turn(now);
                cycle.machine().append(eff);
                // The server holds its own input buffer; the reference calls
                // `_rt_session.clear_audio()`.
                if self.activity.rt_session().is_some() {
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::ClearAudio(
                            lk_agents_abi::llm::realtime::ClearAudio {},
                        ),
                        cycle,
                    );
                }
            }
            TurnCommand::Configure(patch) => {
                let diff = self.session.configure(&patch);
                self.apply_config(now, crate::agent_activity::options::ConfigLayer::Session, diff, cycle);
            }
        }
    }

    /// A changed endpointing replaces the model (`create_endpointing(opts)`); a changed turn
    /// detection mode reaches the turn machine.
    pub(crate) fn apply_config(
        &mut self,
        now: Ts,
        layer: crate::agent_activity::options::ConfigLayer,
        diff: ConfigDiff,
        cycle: &mut Cycle,
    ) {
        let ConfigDiff { endpointing, turn_detection_mode: mode, aec_warmup } = diff;
        // Refine the mode first: the journalled config carries the machine's mode.
        let mode = match self.activity.rt_session().map(|s| s.capabilities()) {
            Some(caps) => self.reconcile_realtime_turn_detection(&caps, cycle),
            None => mode,
        };
        cycle
            .trace(CoreStep::ConfigApplied { layer, effective: Box::new(*self.session.options()) });
        if let Some(e) = endpointing {
            self.activity.audio_recognition_mut().set_endpointing(&e);
        }
        if let Some(mode) = mode {
            let eff = self.activity.audio_recognition_mut().set_turn_detection_mode(now, mode);
            cycle.machine().append(eff);
        }
        if let Some(duration) = aec_warmup {
            cycle.machine().append(self.session.set_aec_warmup(duration));
        }
    }

    pub(crate) fn on_end_of_turn(&mut self, now: Ts, info: EndOfTurnInfo, cycle: &mut Cycle) {
        // Inside the transition window the leaving agent takes no new turn; it is kept for the
        // arriving one. Diverges from the reference, which drops it (`on_end_of_turn`).
        if self.activity.hooks().in_window() {
            self.park_turn(now, info, None, cycle);
            return;
        }
        let guard = self.speech_guard().map(|s| CommitSpeech {
            allow_interruptions: s.allow_interruptions,
            interrupted: s.interrupted,
        });
        // `self._rt_turn_detection_enabled`: the realtime model does its own turn taking.
        let rt_turn_detection =
            self.activity.rt_session().is_some_and(|s| s.capabilities().turn_detection);
        let decision = decide_commit(&CommitInputs {
            // `_scheduling_paused` holds through the hard edge (only a resume clears it, and the
            // close never resumes), so past the edge it is read off `closing`.
            scheduling_paused: self.activity.scheduling_task().is_paused()
                || self.session.closing_past_edge(),
            // The reference sets `_closing` before the drain: the requested close.
            closing: self.session.is_closing(),
            stt_enabled: self.activity.audio_recognition().stt_enabled(),
            manual_turn_detection: self.activity.audio_recognition().turn_detection_mode()
                == crate::audio_recognition::TurnDetectionMode::Manual,
            current_speech: guard,
            min_interruption_words: self.session.options().turn.interruption.min_words as usize,
            new_transcript: &info.new_transcript,
            realtime_llm: self.activity.rt_session().is_some(),
            rt_turn_detection,
            interruption_detection: self.session.options().turn.interruption.detection_enabled,
            backchannel_over_agent: info.backchannel_over_agent,
            skip_reply: info.skip_reply,
            late: info.late,
        });

        cycle.trace(TurnStep::EndOfTurn {
            transcript: info.new_transcript.clone(),
            verdict: decision.verdict,
            committed: decision.committed,
        });
        // Told to the host before the verdict and unconditionally, where the reference calls
        // `on_end_of_turn`: AMD's release gate waits on it. It cannot veto the reply.
        cycle.machine().emit(crate::events::SessionEvent::UserTurnCompleted(
            crate::events::UserTurnCompletedEvent {
                turn: info.turn_id,
                new_transcript: info.new_transcript.clone(),
                transcript_confidence: info.transcript_confidence,
            },
        ));

        let eff = self.activity.audio_recognition_mut().on_commit_verdict(now, &info, &decision);
        cycle.machine().append(eff);

        if decision.cancel_false_interruption {
            self.cancel_speech_pause(now, cycle);
        }
        if !decision.verdict.replies() {
            // Only the paused arm logs in the reference; the others stay silent.
            if matches!(decision.verdict, CommitVerdict::Dropped { .. }) {
                cycle.machine().emit(crate::events::SessionEvent::Diagnostic(
                    crate::events::DiagnosticEvent::warning(
                        crate::events::diagnostics::SCHEDULING_PAUSED,
                    )
                    .with(lk_agents_abi::telemetry::semconv::LK_PII_USER_INPUT, info.new_transcript.clone()),
                ));
            }
            // Under `_closing` the dropped turn's words still reach the record, with no reply.
            if let CommitVerdict::Dropped { append_user_message: true } = decision.verdict {
                let user = self.user_turn_item(&info, None);
                self.commit_user_message(
                    user,
                    crate::agent::Source::UserTurn { turn: info.turn_id },
                    cycle,
                );
            }
            // A turn decided without a reply: a preemptive generation made for it has no owner.
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, cycle);
            return;
        }
        // The hook's seam: the user item is committed and the reply planned only once the host
        // answers or the bound passes. A newer turn ends the older one's wait, with no edits.
        if let Some(previous) = self.activity.take_pending_turn() {
            if let Some(hook) = previous.hook {
                self.activity.hooks_mut().forget(hook);
                cycle.trace(HandoffStep::HookFinished {
                    hook,
                    kind: crate::agent_session::HookKind::OnUserTurnCompleted,
                    how: crate::trace::HookEnd::TimedOut,
                });
            }
            self.finish_turn(now, previous.finished(now), Vec::new(), cycle);
        }
        match self.activity.hook_bound() {
            Some(bound) => self.issue_turn_hook(now, info, bound, cycle),
            None => self.finish_turn(now, info, Vec::new(), cycle),
        }
    }

    /// `StopResponse` ends it with no item and no reply; `skip_reply` commits the item and asks for
    /// nothing.
    pub(crate) fn finish_turn(
        &mut self,
        now: Ts,
        info: EndOfTurnInfo,
        edits: Vec<crate::agent::TurnEdit>,
        cycle: &mut Cycle,
    ) {
        // The reference drops the reply here (`_new_turns_blocked`); the port keeps the turn for
        // the arriving frame.
        if self.activity.hooks().in_window() {
            self.park_turn(now, info, Some(edits), cycle);
            return;
        }
        let mut applied = TurnEdits::default();
        for edit in edits {
            match edit {
                crate::agent::TurnEdit::UpdateInstructions(text) => {
                    applied.instructions = Some(text)
                }
                crate::agent::TurnEdit::ReplaceItems(items) => {
                    applied.chat_ctx = Some(ChatContext::new(items));
                }
                crate::agent::TurnEdit::UpdateUserMessage(text) => applied.user_text = Some(text),
                crate::agent::TurnEdit::StopResponse => {
                    self.activity.set_last_turn(info.turn_id);
                    self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, cycle);
                    return;
                }
            }
        }
        if info.skip_reply {
            self.activity.set_last_turn(info.turn_id);
            self.cancel_preemptive_generation(now, PreemptiveDiscard::TurnAbandoned, cycle);
            let user = self.user_turn_item(&info, applied.user_text.as_deref());
            self.commit_user_message(
                user,
                crate::agent::Source::UserTurn { turn: info.turn_id },
                cycle,
            );
            return;
        }
        // Before the reply exists: supersede an unheard late reply, then interrupt the current
        // speech if it allows it. A realtime session interrupts in `generate_reply`.
        if self.activity.rt_session().is_none() {
            if info.late {
                self.supersede_unheard_reply(now, cycle);
            }
            self.interrupt_for_user_turn(now, cycle);
        }
        self.generate_reply(now, &info, applied, cycle);
    }

    /// Stops the current speech if it allows it; otherwise the reply queues behind it.
    pub(crate) fn interrupt_for_user_turn(&mut self, now: Ts, cycle: &mut Cycle) {
        let Some(current) = self.speech_guard() else { return };
        if current.interrupted || !current.allow_interruptions {
            return;
        }
        match self.interrupt_speech(now, current.id, InterruptCause::UserTurn, false, cycle) {
            Ok(InterruptOutcome::Now { .. } | InterruptOutcome::Already) => {}
            // A hold taken since the guard was read: the handle's to keep, said by name.
            Err(_) => cycle.trace(InterruptStep::BargeRefused(InterruptRefusal::NotAllowed)),
        }
    }

    /// A late final re-committed a turn whose reply is still unheard: that reply is superseded so
    /// one utterance never gets two replies. A reply already heard stays.
    pub(crate) fn supersede_unheard_reply(&mut self, now: Ts, cycle: &mut Cycle) {
        let paused = self.activity.paused_speech().paused_handle();
        let speaking = self.session.agent_state() == AgentState::Speaking;
        // The newest reply first: it is the one the turn being re-committed created.
        let unheard = self.activity.scheduling_task().handles().rev().find_map(|(id, handle)| {
            if handle.interrupted() || handle.done() {
                return None;
            }
            let scheduling = self.activity.scheduling_task();
            let reply = match scheduling.kind(id)? {
                SpeechKind::TurnReply | SpeechKind::Guess => true,
                SpeechKind::Say | SpeechKind::GenerateReply | SpeechKind::Realtime => false,
            };
            let unheard = match scheduling.status(id)? {
                Status::Queued => true,
                Status::Current => !speaking && paused != Some(id),
                Status::Preemptive | Status::Working | Status::Parked | Status::Discarded => false,
            };
            (reply && unheard).then_some(id)
        });
        let Some(speech) = unheard else { return };
        match self.interrupt_speech(now, speech, InterruptCause::Superseded, true, cycle) {
            Ok(InterruptOutcome::Now { .. } | InterruptOutcome::Already) => {}
            // Unreachable: `force` overrides the hold. Said rather than swallowed.
            Err(_) => cycle.trace(CoreStep::PortFailure("a forced supersede was refused")),
        }
    }

    /// Ask the end-of-turn model; the answer returns as `Input::Eot`. A signal, not an await: the
    /// driver is mid-borrow here, and the countdown's own timeout bounds a slow answer.
    pub(crate) fn request_eot_prediction(&mut self, req: lk_agents_abi::turn::detector::ReqId) {
        // Unreachable: the machine asks only when a detector was configured.
        let Some(leg) = self.activity.audio_recognition().eot() else { return };
        let chat_ctx = self.session.chat_ctx().agent_chat_ctx();
        let timeout = self.activity.audio_recognition().prediction_timeout();
        // Any outstanding request is superseded; its late answer is dropped.
        self.journal(Step::Input(InputStep::LegRequested {
            leg: crate::trace::DetectorLeg::Eot,
            req: req.0,
        }));
        leg.signal(Some(req), lk_agents_abi::EotRequest { chat_ctx, timeout });
    }

    /// A prediction arrived for `req`, or will not. Only the request the phase holds applies: a
    /// stale answer would shorten the next pause's delay. A failure is an input, not a forged timer.
    pub(crate) fn on_eot(
        &mut self,
        now: Ts,
        req: lk_agents_abi::turn::detector::ReqId,
        p: Result<lk_agents_abi::EotPrediction, lk_agents_abi::errors::ProviderError>,
        cycle: &mut Cycle,
    ) {
        use crate::trace::DetectorLeg;
        cycle.trace(InputStep::LegVerdict { leg: DetectorLeg::Eot, req: req.0 });
        // Recoverable: the next pause asks again. The machine serves `min_delay`.
        let result = p.map_err(|_| {
            cycle.trace(InputStep::LegFailed { leg: DetectorLeg::Eot, recoverable: true });
        });
        let eff = self.activity.audio_recognition_mut().on_prediction(now, req, result);
        cycle.machine().append(eff);
    }

    pub(crate) fn commit_user_turn(
        &mut self,
        now: Ts,
        transcript_timeout: Duration,
        stt_flush_duration: Duration,
        mut skip_reply: bool,
        done: tokio::sync::oneshot::Sender<String>,
        cycle: &mut Cycle,
    ) {
        // A realtime session commits its server buffer itself, and only when the server does not
        // segment audio (a client commit of a server-segmented buffer is refused).
        if let Some(rt) = self.activity.rt_session() {
            let server_segments = rt.capabilities().turn_detection
                && !self.activity.realtime_turn_detection_disabled();
            if skip_reply {
                if !server_segments {
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::CommitAudio(
                            lk_agents_abi::llm::realtime::CommitAudio {},
                        ),
                        cycle,
                    );
                }
            } else {
                self.realtime_user_turn(now, cycle);
            }
            skip_reply = true;
        }
        // "if not self._stt or self._closing.is_set(): fut.set_result('')". `_stt` is the
        // configured recogniser, so a dead stream still waits the timeout, as in the reference.
        if !self.activity.audio_recognition().stt_enabled() || self.session.is_closing() {
            cycle.trace(TurnStep::CommitAnswered { transcript: String::new() });
            let _ = done.send(String::new());
            return;
        }
        self.begin_commit(
            now,
            transcript_timeout,
            stt_flush_duration,
            skip_reply,
            false,
            done,
            cycle,
        );
    }

    pub(crate) fn begin_commit(
        &mut self,
        now: Ts,
        transcript_timeout: Duration,
        stt_flush_duration: Duration,
        skip_reply: bool,
        close: bool,
        done: tokio::sync::oneshot::Sender<String>,
        cycle: &mut Cycle,
    ) {
        // A newer commit cancels the waiting one (`_commit_user_turn_atask`).
        if let Some(previous) = self.activity.audio_recognition_mut().take_pending_commit() {
            cycle.trace(TurnStep::CommitSuperseded);
            drop(previous.done);
        }
        // A final within the last 0.5 s is fresh: commit at once, no flush, no wait.
        let fresh = self.activity.audio_recognition().final_is_fresh(now);
        self.activity.audio_recognition_mut().set_pending_commit(PendingCommit {
            skip_reply,
            done,
            deadline: now.saturating_add(transcript_timeout),
            close,
        });
        if fresh || transcript_timeout.is_zero() {
            self.perform_pending_commit(now, cycle);
            return;
        }
        // The reference flushes only when the mic is detached; that fact doesn't reach the driver,
        // so every awaited commit flushes (it is what makes the last final arrive).
        self.flush_stt_with_silence(stt_flush_duration.as_secs_f64());
    }

    pub(crate) fn perform_pending_commit(&mut self, now: Ts, cycle: &mut Cycle) {
        let Some(pending) = self.activity.audio_recognition_mut().take_pending_commit() else {
            return;
        };
        let transcript = self.activity.audio_recognition().current_transcript();
        // The final that ended the wait may have decided the turn already; don't commit twice.
        if !matches!(self.activity.audio_recognition().phase(), Phase::Committed { .. }) {
            self.on_turn(now, TurnCommand::Commit { skip_reply: pending.skip_reply }, cycle);
        }
        if pending.close {
            cycle.trace(CloseStep::Performed(CloseAct::TranscriptWaitEnded {
                transcript: transcript.clone(),
            }));
        } else {
            cycle.trace(TurnStep::CommitAnswered { transcript: transcript.clone() });
        }
        let _ = pending.done.send(transcript);
    }

    /// `on_preemptive_generation`: ask the model now, speak later. Guards in the reference's order;
    /// the policy is read off the resolved configuration, so agent overrides apply.
    pub(crate) fn on_preemptive_generation(
        &mut self,
        now: Ts,
        transcript: String,
        started_speaking_at: Option<Ts>,
        cycle: &mut Cycle,
    ) {
        let policy = self.session.options().turn.preemptive;
        let s = self.activity.scheduling_task();
        let speaking = s.current_speech_uninterrupted();
        let refusal = if !policy.enabled {
            Some(PreemptRefusal::Disabled)
        } else if self.activity.rt_session().is_some() {
            Some(PreemptRefusal::Realtime)
        // The transition window: the leaving agent starts no preemptive generation
        // (`_new_turns_blocked`), which covers the `on_exit` phase before the scheduling task's
        // pause.
        } else if self.activity.hooks().in_window()
            || s.authorization_paused()
            || s.is_paused()
            || self.session.closing_past_edge()
        {
            Some(PreemptRefusal::SchedulingPaused)
        // An agent still talking has no turn to be early for; an interrupted one does.
        } else if speaking {
            Some(PreemptRefusal::AgentSpeaking)
        // Strictly past `max_speech_duration`; a limit of zero refuses any timed onset.
        } else if started_speaking_at
            .is_some_and(|at| now.saturating_duration_since(at) > policy.max_speech_duration)
        {
            Some(PreemptRefusal::TooLong)
        } else if s.preemptive_generation_count() >= policy.max_retries {
            Some(PreemptRefusal::RetriesExhausted)
        } else {
            None
        };
        if let Some(refusal) = refusal {
            return cycle.trace(TurnStep::PreemptRefused(refusal));
        }
        self.activity.scheduling_task_mut().set_preflight(transcript.clone());
        // The preemptive generation belongs to the turn after the last commit; `generate_reply`
        // checks it.
        self.activity.key_preemptive_generation(self.session.record_version());
        self.start_preemptive_generation(now, transcript, cycle);
    }

    /// Create the preemptive generation: planned against the preflight words, not queued and not
    /// committed.
    fn start_preemptive_generation(&mut self, now: Ts, transcript: String, cycle: &mut Cycle) {
        let wall = self.shared.time.wall();
        let user = ChatMessage::text(self.session.mint("item"), Role::User, &transcript, wall);
        // Announced by `start_speech` when the handle exists, as the reference's is.
        self.start_speech(
            now,
            SpeechSpec {
                new_message: Some(user),
                ..SpeechSpec::new(
                    SpeechKind::Guess,
                    ReplySource::Llm,
                    self.session.options().turn.interruption.enabled,
                )
            },
            cycle,
        );
    }
}
