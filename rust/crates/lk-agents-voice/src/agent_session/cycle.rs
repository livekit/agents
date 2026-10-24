//! The driver's [`Cycle`]: the [`Effect`]s the components hand it, in order, the timers they arm,
//! and its performer, which runs them under the frame that asked.

use crate::agent_activity::pause::{BargeTrigger, TurnSettled};
use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{Driver, HostEvent};
use crate::audio::LegId;
use crate::audio_recognition::{EndOfTurnInfo, InterruptionSignal, TurnDetectionMode};
use crate::events::SessionEvent;
use crate::trace::{CoreStep, InputStep, InterruptStep, RealtimeStep, Step, Trace, TurnStep};
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::voice::events::UserState;
use lk_agents_abi::{FrameId, Ts};
use lk_agents_voice::io::AudioOutputExt;
use std::sync::{Arc, Mutex};

pub struct Cycle {
    frame: FrameId,
    /// What the machines asked for, not yet performed.
    todo: Vec<Effect>,
    /// What a machine writes into. Drained into `todo` by [`Cycle::absorb`].
    machine: Effects,
    trace: Arc<Mutex<Trace>>,
}

impl std::fmt::Debug for Cycle {
    /// Without the trace, which would bury the fields a failing test's reader wants.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cycle").field("frame", &self.frame).field("todo", &self.todo).finish()
    }
}

impl Cycle {
    /// `frame` is the driver's `current` at the moment the event was dequeued.
    pub fn new(frame: FrameId, trace: Arc<Mutex<Trace>>) -> Self {
        Cycle { frame, todo: Vec::new(), machine: Effects::new(), trace }
    }

    pub fn frame(&self) -> FrameId {
        self.frame
    }

    /// Where this cycle's effects are written; [`Cycle::take`] and [`Cycle::peek`] read them.
    pub fn machine(&mut self) -> &mut Effects {
        &mut self.machine
    }

    /// Move the machine's effects into the cycle's to-do list, recording each as a step at the
    /// instant it was decided. Called after every transition, so the journal reads in causal
    /// order; `perform` records nothing.
    pub fn absorb(&mut self) {
        use crate::agent_session::cycle::TimerOp;
        for effect in self.machine.take() {
            match &effect {
                // The journal's copy never pins a host exception (`SessionEvent::journal_form`).
                Effect::Emit(ev) => self.record(Step::Emitted(ev.journal_form())),
                Effect::Timer(TimerOp::Arm(t, at)) => {
                    self.record(Step::Core(CoreStep::TimerArmed { timer: *t, at: *at }))
                }
                Effect::Timer(TimerOp::Cancel(t)) => {
                    self.record(Step::Core(CoreStep::TimerCancelled(*t)))
                }
                Effect::Provider(call) => {
                    self.record(Step::Input(InputStep::ProviderCalled(*call)))
                }
                Effect::Detector(call) => {
                    self.record(Step::Input(InputStep::DetectorCalled(*call)))
                }
                Effect::Audio(crate::agent_session::cycle::AudioOp::Gate { speech, op, cause }) => {
                    self.record(Step::Interrupt(InterruptStep::PlayoutGate {
                        speech: *speech,
                        op: *op,
                        cause: *cause,
                    }))
                }
                // The source swap's own step (`AudioInputSet`, `VideoInputSet`) is the handler's.
                Effect::Audio(
                    crate::agent_session::cycle::AudioOp::SetInput
                    | crate::agent_session::cycle::AudioOp::SetVideoInput,
                ) => {}
                Effect::Decision(d) => self.record(Step::Decided(d.clone())),
                Effect::Realtime(send) => self.record(Step::Realtime(RealtimeStep::Sent {
                    kind: send.0.kind(),
                    req_id: send.0.req_id().map(str::to_owned),
                })),
            }
            self.todo.push(effect);
        }
    }

    /// Record a driver decision. Immediate: the journal is shared, so there is nothing to buffer.
    pub fn trace(&mut self, step: impl Into<Step>) {
        self.absorb();
        self.record(step.into());
    }

    fn record(&mut self, step: Step) {
        self.trace.lock().unwrap_or_else(|p| p.into_inner()).push(step);
    }

    pub fn take(&mut self) -> Vec<Effect> {
        self.absorb();
        std::mem::take(&mut self.todo)
    }

    /// Read the to-do list without draining it.
    pub fn peek(&mut self) -> &[Effect] {
        self.absorb();
        &self.todo
    }

    /// The timers a pending `Arm` or `Cancel` of this cycle touches, in order. Timer effects run
    /// after the loop's timer pass, so without this a deadline cancelled by an earlier fire in the
    /// same instant would still be popped (`Driver::on_timers`).
    pub fn timers_touched(&mut self) -> Vec<crate::agent_session::cycle::TurnTimer> {
        use crate::agent_session::cycle::TimerOp;
        self.absorb();
        self.todo
            .iter()
            .filter_map(|e| match e {
                Effect::Timer(TimerOp::Arm(t, _) | TimerOp::Cancel(t)) => Some(*t),
                Effect::Emit(_)
                | Effect::Provider(_)
                | Effect::Detector(_)
                | Effect::Audio(_)
                | Effect::Decision(_)
                | Effect::Realtime(_) => None,
            })
            .collect()
    }
}

/// One ordered list, not per-kind fields: order within and across kinds is inherited from the
/// reference, where these ran inline in exactly this order.
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    Timer(TimerOp),
    /// Buffered because reaching it needs `&mut` on the driver.
    Provider(ProviderCall),
    Emit(SessionEvent),
    Decision(Decision),
    /// Per-frame `PushAudio` never comes this way.
    Realtime(RealtimeSend),
    Audio(AudioOp),
    Detector(DetectorCall),
}

#[cfg(test)]
impl Effect {
    pub fn cancel(timer: TurnTimer) -> Effect {
        Effect::Timer(TimerOp::Cancel(timer))
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
#[must_use = "effects a transition asked for must reach the driver's cycle"]
pub struct Effects {
    effects: Vec<Effect>,
}

impl Effects {
    pub fn new() -> Self {
        Effects::default()
    }

    pub fn take(&mut self) -> Vec<Effect> {
        std::mem::take(&mut self.effects)
    }

    #[cfg(test)]
    pub fn as_slice(&self) -> &[Effect] {
        &self.effects
    }

    pub fn push(&mut self, e: Effect) {
        self.effects.push(e);
    }

    /// What a component returned, after what is already here, in its order.
    pub fn append(&mut self, returned: Effects) {
        self.effects.extend(returned.effects);
    }

    pub fn arm(&mut self, timer: TurnTimer, at: Ts) {
        self.push(Effect::Timer(TimerOp::Arm(timer, at)));
    }

    pub fn cancel(&mut self, timer: TurnTimer) {
        self.push(Effect::Timer(TimerOp::Cancel(timer)));
    }

    pub fn emit(&mut self, ev: SessionEvent) {
        self.push(Effect::Emit(ev));
    }

    pub fn request_eot(&mut self, req: ReqId) {
        self.push(Effect::Detector(DetectorCall::RequestEot { req }));
    }

    pub fn cancel_eot(&mut self, req: ReqId, timed_out: bool) {
        self.push(Effect::Detector(DetectorCall::CancelEot { req, timed_out }));
    }

    pub fn signal(&mut self, signal: InterruptionSignal) {
        self.push(Effect::Detector(DetectorCall::Signal(signal)));
    }

    pub fn decide(&mut self, decision: Decision) {
        self.push(Effect::Decision(decision));
    }

    pub fn flush_turn_detector(&mut self, reason: FlushReason) {
        self.push(Effect::Detector(DetectorCall::Flush(reason)));
    }

    /// The close's edge and its `CloseLegs` phase.
    pub fn close_turn_detector(&mut self) {
        self.push(Effect::Detector(DetectorCall::Close));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimerOp {
    /// Arming an armed timer *replaces* it ([`TimerSet`](lk_agents_abi::TimerSet)).
    Arm(TurnTimer, Ts),
    /// Idempotent: cancelling an unarmed timer is a no-op, so cleanup paths cancel unconditionally.
    Cancel(TurnTimer),
}

/// `arm` on an armed timer replaces it (the reference's `handle.cancel(); handle =
/// call_later(..)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TurnTimer {
    /// `audio_recognition._bounce_eou_task`.
    EotPredictionTimeout,
    /// `audio_recognition._bounce_eou_task`.
    EndpointingDelay,
    /// `_arm_transcription_timeout`.
    TranscriptionTimeout,
    /// `_backchannel_boundary_timer`: after agent speech onset, a backchannel verdict is ignored.
    BackchannelBoundary,
    /// `_false_interruption_timer`.
    FalseInterruption,
    /// Barge-in refused, STT hears silence.
    AecWarmup,
    /// Armed while both sides listen and no tool runs.
    UserAway,
    /// From the first speech onset since the agent last spoke; fires `UserTurnExceeded` once per
    /// crossing.
    UserTurnDuration,
    /// `IVRActivity._debounced_silence`.
    IvrSilence,
    /// `RecorderIO._write_task`'s 2.5 s.
    RecorderFlush,
    /// `keyterm_detection.timeout`.
    KeytermTimeout,
    Amd(crate::amd::classifier::AmdTimer),
    /// No `GenerationCreated` (or failure) back yet; on firing it fails with a timeout.
    RealtimeRequest(crate::agent_activity::realtime::RequestKey),
    /// `RunContext.with_filler`.
    Filler(crate::filler_scheduler::FillerId),
    /// Re-armed by every digit, cancelled while either side speaks.
    DtmfDebounce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderCall {
    FlushVad,
    FlushStt,
    /// Comes back as `Input::LegOpened{open}`: the loop never awaits `stream()`.
    OpenLeg {
        leg: LegKind,
        open: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LegKind {
    Stt,
    Vad,
}

impl LegKind {
    pub fn label(self) -> &'static str {
        match self {
            LegKind::Stt => "stt",
            LegKind::Vad => "vad",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorCall {
    /// `req` is the id the answer carries back.
    RequestEot { req: ReqId },
    /// `cancel_inference(timed_out=..)`: `true` only from the prediction timeout.
    CancelEot { req: ReqId, timed_out: bool },
    /// `AudioRecognition._clear_user_turn`, `._bounce_eou_task`.
    Flush(FlushReason),
    /// `_interruption_ch`.
    Signal(InterruptionSignal),
    /// A prediction still in flight answers a turn nobody will decide.
    Close,
}

/// The reference's free-form `reason`, closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushReason {
    /// `audio_recognition._bounce_eou_task`: the user turn was accepted.
    TurnCommitted,
    /// `AudioRecognition._clear_user_turn`: the turn was dropped and its state cleared.
    ClearUserTurn,
}

impl FlushReason {
    /// So the host logs the same words the reference does.
    pub fn as_str(self) -> &'static str {
        match self {
            FlushReason::TurnCommitted => "turn committed",
            FlushReason::ClearUserTurn => "clear_user_turn",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOp {
    /// Pushed only when a sink exists, so the journal's `PlayoutGate` is exactly the sink's
    /// history.
    Gate {
        speech: SpeechId,
        op: GateOp,
        cause: GateCause,
    },
    /// The old pump stops first.
    SetInput,
    SetVideoInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOp {
    Pause,
    Resume,
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateCause {
    /// The launch hold: `Pause` before the first frame, `Resume` when the user stopped.
    Launch,
    BargeIn,
    FalseInterruptionResolved,
    Interrupted,
    /// A hold that no longer applies: the parked speech became uninterruptible.
    Cancelled,
    /// Released a sink pause the facts no longer permit (sink swap, audio off).
    Released,
}

/// Compares equal by kind and request id, as the journal records it (payloads are not `PartialEq`).
#[derive(Debug, Clone)]
pub struct RealtimeSend(pub lk_agents_abi::RealtimeCommand);

impl PartialEq for RealtimeSend {
    fn eq(&self, other: &Self) -> bool {
        self.0.kind() == other.0.kind() && self.0.req_id() == other.0.req_id()
    }
}

#[derive(Debug, Clone, PartialEq, lk_agents_abi::Vocabulary)]
pub enum Decision {
    /// `decide_commit`, tell the host, answer the machine, and maybe reply.
    EndOfTurn(EndOfTurnInfo),
    TurnSettled(TurnSettled),
    BargeGuard(BargeGuard),
    /// Still subject to `decide_interrupt`'s guards.
    Barge(BargeTrigger),
    /// Journal-only.
    EndpointingDelayUpdated {
        min_delay: std::time::Duration,
    },
    /// Even when a VAD end already closed the user's speech. Journal-only.
    TurnDeclared {
        owner: TurnDetectionMode,
    },
}

/// Guard 1 of `decide_interrupt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BargeGuard {
    /// `false` while the detector classifies an overlap.
    Set(bool),
    /// The session's configured default: the option, not `true`.
    Restore,
}

impl Driver {
    /// Perform the buffered port calls, under the frame that produced them.
    pub(crate) fn execute(&mut self, mut cycle: Cycle) {
        let frames = Arc::clone(&self.shared.frames);
        let _scope = frames.enter(cycle.frame());
        let now = self.step.now;

        // A performed decision may ask for more, so drain until nothing is left, in order.
        loop {
            let effects = cycle.take();
            if effects.is_empty() {
                // The current speech is read last, so `FloorChanged` is the cycle's last emit;
                // announcing it may itself be an effect to perform, hence the `continue`.
                if self.publish_current_speech(&mut cycle) {
                    continue;
                }
                break;
            }
            for effect in effects {
                self.perform(now, effect, &mut cycle);
            }
        }
    }

    /// Speech boundaries move the user state, dated to the boundary's instant; a final transcript
    /// restores `away` or pushes the deadline out; every state edge reconciles the away timer.
    pub(crate) fn on_session_edge(&mut self, now: Ts, ev: &SessionEvent, cycle: &mut Cycle) {
        // A DTMF collection reads the same edges (and its read-back's end) off the same events.
        self.dtmf_on_event(now, ev, cycle);
        match ev {
            SessionEvent::StartOfSpeech(e) => {
                cycle
                    .machine()
                    .append(self.session.presence.update_user_state(e.at, UserState::Speaking));
            }
            SessionEvent::EndOfSpeech(e) => {
                cycle
                    .machine()
                    .append(self.session.presence.update_user_state(e.at, UserState::Listening));
            }
            SessionEvent::UserTranscript(t) => {
                // Any non-empty transcript is the STT recovering
                // (`AgentSession._user_input_transcribed`).
                if let Some(a) = t.ev.best_alternative() {
                    self.session.errors.on_transcript(&a.text);
                }
                if t.is_final {
                    let closing = self.session.closing_past_edge();
                    cycle.machine().append(self.session.presence.on_final_transcript(now, closing));
                }
            }
            // The IVR navigator reads the two state edges off the feed, as `IVRActivity` does.
            SessionEvent::AgentStateChanged { .. } | SessionEvent::UserStateChanged { .. } => {
                let closing = self.session.closing_past_edge();
                cycle.machine().append(self.session.presence.reconcile_away_timer(now, closing));
            }
            SessionEvent::MetricsCollected { .. }
            | SessionEvent::SpanStarted(_)
            | SessionEvent::SpanEnded(_)
            | SessionEvent::Error(_)
            | SessionEvent::Diagnostic(_)
            | SessionEvent::EotPrediction(_)
            | SessionEvent::AgentBackchannelOpportunity(_)
            | SessionEvent::PreemptiveGeneration(_)
            | SessionEvent::UserTurnExceeded(_)
            | SessionEvent::TranscriptionTimeout(_)
            | SessionEvent::BackchannelConfirmed
            | SessionEvent::OverlapInterruption(_)
            | SessionEvent::AgentFalseInterruption { .. }
            | SessionEvent::UserTurnCompleted(_)
            | SessionEvent::SpeechCreated { .. }
            | SessionEvent::ToolsExecuted(_)
            | SessionEvent::ToolExecutionUpdated(_)
            | SessionEvent::SpeechDone { .. }
            | SessionEvent::FloorChanged { .. }
            | SessionEvent::AmdResult { .. }
            | SessionEvent::RunStarted { .. }
            | SessionEvent::RunItemAdded { .. }
            | SessionEvent::RunRefused { .. }
            | SessionEvent::RunSettled { .. }
            | SessionEvent::ContextCarried { .. }
            | SessionEvent::Close { .. } => {}
            // `_on_conversation_item_added`: the keyterm pass's turn gate.
            SessionEvent::ItemAdded { item, .. } => {
                if self.session.keyterms.is_some() {
                    let item = (**item).clone();
                    self.keyterms_on_item(now, &item, cycle);
                }
            }
        }
    }

    /// `SessionEvent::FloorChanged` when `_current_speech` differs from the last announced one.
    /// Returns whether an emit was queued.
    pub(crate) fn publish_current_speech(&mut self, cycle: &mut Cycle) -> bool {
        let current = self.activity.scheduling_task.current_speech();
        let Some(holder) = self.activity.announce_current_speech(current) else { return false };
        cycle.machine().emit(SessionEvent::FloorChanged { holder });
        true
    }

    pub(crate) fn perform(
        &mut self,
        now: Ts,
        effect: crate::agent_session::cycle::Effect,
        cycle: &mut Cycle,
    ) {
        use crate::agent_session::cycle::{
            AudioOp, BargeGuard, Decision, DetectorCall as D, Effect as E, GateOp,
            ProviderCall as P, TimerOp,
        };
        // Journalled already: `Cycle::absorb` recorded every effect when the machine produced it.
        match effect {
            E::Timer(TimerOp::Arm(t, at)) => {
                self.timers.arm(t, at);
            }
            E::Timer(TimerOp::Cancel(t)) => {
                self.timers.cancel(t);
            }
            // Not traced here: `Cycle::absorb` journalled it where it was decided.
            E::Emit(mut ev) => {
                // `on_backchannel_confirmed`: clear the realtime model's buffered audio so a
                // "mm-hm" cannot prefix the next turn. The reference's `turn_detection not in
                // ("manual", "realtime_llm")` guard is unported: `TurnConfig::mode` is private.
                if matches!(ev, SessionEvent::BackchannelConfirmed)
                    && self.session.config.effective.turn.interruption.detection_enabled
                    && self.activity.rt_session.is_some()
                {
                    self.rt_send(
                        lk_agents_abi::RealtimeCommand::ClearAudio(
                            lk_agents_abi::llm::realtime::ClearAudio {},
                        ),
                        cycle,
                    );
                }
                // The turn machine cannot mint ids (pure tier), so the utterance's shared item id
                // is stamped here, where the emission leaves the driver.
                if let SessionEvent::UserTranscript(t) = &mut ev {
                    let id = self.session.ids.user_transcript_item(t.is_final);
                    let text = t.ev.best_alternative().map(|a| a.text.clone()).unwrap_or_default();
                    self.journal(Step::Turn(TurnStep::TranscriptSegment {
                        id: id.clone(),
                        text,
                        is_final: t.is_final,
                    }));
                    t.item_id = Some(id);
                }
                // The user state and away deadline are decided on the way to the host.
                self.on_session_edge(now, &ev, cycle);
                let closed = matches!(ev, SessionEvent::Close { .. });
                // The peers read what the host reads (`Watched::Session`), in the same order.
                if !self.watch.is_empty() {
                    self.watch.publish(crate::agent_session::Watched::Session(ev.clone()));
                }
                let _ = self.events.send(HostEvent::Session(ev));
                // `delete_room_on_close`: asked of the host on the `close` event, after it.
                if closed {
                    self.on_session_closed();
                }
            }
            E::Provider(call @ (P::FlushStt | P::FlushVad)) => {
                let leg = if matches!(call, P::FlushStt) { LegId::Stt } else { LegId::Vad };
                if let Some(s) = self.input.leg(leg) {
                    let _ = s.flush();
                }
            }
            // A session without a turn detector asks for no flush.
            E::Detector(D::Flush(reason)) => {
                if let Some(d) = self.activity.audio_recognition.eot() {
                    d.flush(reason.as_str());
                }
            }
            E::Detector(D::RequestEot { req }) => self.request_eot_prediction(req),
            // `_generate_reply` is the one `timed_out=True` site. A session without a detector
            // asks for none.
            E::Detector(D::CancelEot { req: _, timed_out }) => {
                if let Some(d) = self.activity.audio_recognition.eot() {
                    d.cancel(timed_out);
                }
            }
            // Not awaited, like the reference's `send_nowait` on `_interruption_ch`. A session
            // with no detector raises no signals.
            E::Detector(D::Signal(signal)) => {
                if let Some(d) = self.activity.audio_recognition.interruption().cloned() {
                    match signal {
                        crate::audio_recognition::InterruptionSignal::OverlapStarted { .. } => {
                            let overlap = self
                                .activity
                                .audio_recognition
                                .open_overlap()
                                .unwrap_or(lk_agents_abi::turn::detector::ReqId(0));
                            d.signal_overlap(overlap, signal);
                        }
                        crate::audio_recognition::InterruptionSignal::AgentSpeechStarted
                        | crate::audio_recognition::InterruptionSignal::AgentSpeechEnded
                        | crate::audio_recognition::InterruptionSignal::OverlapEnded { .. } => {
                            d.signal(signal)
                        }
                    }
                }
            }
            // The turn -> reply hop, in order with everything else the transition asked for.
            E::Decision(Decision::EndOfTurn(info)) => {
                self.on_end_of_turn(now, info, cycle);
                // Scheduled in the same cycle: a reply created here is authorised now.
                self.pump_schedule(now, cycle);
            }
            // `_on_turn_settled`: a pause left in `AwaitingTurn` resolves the way the
            // false-interruption timer would have.
            E::Decision(Decision::TurnSettled(settled)) => {
                self.on_turn_settled(now, settled, cycle)
            }
            E::Decision(Decision::BargeGuard(BargeGuard::Set(value))) => {
                self.session.by_audio_activity = value;
            }
            E::Decision(Decision::BargeGuard(BargeGuard::Restore)) => {
                self.restore_audio_activity_guard();
            }
            // Through `interrupt_by_audio_activity`, so all its guards still apply. The flush is
            // skipped when the pause already ran `on_end_of_agent_speech`: running it twice would
            // shrink the `min`-folded ignore window again.
            E::Decision(Decision::Barge(trigger)) => {
                self.interrupt_by_audio_activity(now, cycle, trigger);
                if let BargeTrigger::DetectorVerdict { ignore_user_transcript_until } = trigger
                    && self.activity.paused_speech.paused_handle().is_none()
                {
                    let eff = self
                        .activity
                        .audio_recognition
                        .on_end_of_agent_speech(now, ignore_user_transcript_until);
                    cycle.machine().append(eff);
                }
            }
            // Journal-only: the learn is the machine's. The open runs off the loop (the provider
            // awaits a connection); the stream comes back as `Input::LegOpened`.
            E::Provider(P::OpenLeg { leg, open }) => {
                let Some(pending) = self.input.take_pending_open(open) else { return };
                let tx = self.leg_opens.clone();
                lk_agents_abi::SpawnerExt::spawn_unboxed(&*self.shared.spawner, async move {
                    use crate::room_io::LegOpened;
                    use crate::room_io::Opened;
                    use crate::room_io::PendingOpen;
                    let outcome = match pending {
                        PendingOpen::Stt(stt) => match stt.stream(None, None).await {
                            Some(Ok((sink, source))) => Ok(Opened::Stt(sink, source)),
                            Some(Err(e)) => Err(e),
                            None => Err(lk_agents_abi::errors::ProviderError::Local(
                                lk_agents_abi::errors::ApiError::new(
                                    "the new agent's STT does not stream",
                                ),
                            )),
                        },
                        PendingOpen::Vad(vad) => {
                            vad.stream().await.map(|(sink, source)| Opened::Vad(sink, source))
                        }
                    };
                    let _ = tx.send(LegOpened { open, leg, outcome });
                });
            }
            E::Detector(D::Close) => {
                if let Some(leg) = self.activity.audio_recognition.eot() {
                    leg.close();
                }
            }
            E::Audio(AudioOp::Gate { speech: _, op, cause: _ }) => {
                if let Some(out) = self.shared.output.audio() {
                    match op {
                        GateOp::Pause => out.pause(),
                        GateOp::Resume => out.resume(),
                        GateOp::Clear => out.cut(),
                    }
                }
            }
            // Abort the old pump first: two sources on one table look like the user talking over
            // themselves.
            E::Audio(AudioOp::SetInput) => self.input.swap_audio_pump(&*self.shared.spawner),
            E::Audio(AudioOp::SetVideoInput) => self.input.swap_video_pump(&*self.shared.spawner),
            E::Realtime(crate::agent_session::cycle::RealtimeSend(cmd)) => {
                if let Some(session) = self.activity.rt_session.as_ref() {
                    session.send(cmd);
                }
            }
            E::Decision(Decision::EndpointingDelayUpdated { min_delay: _ }) => {}
            // Journal-only: the owner's declaration, read by the rules.
            E::Decision(Decision::TurnDeclared { owner: _ }) => {}
        }
    }
}
