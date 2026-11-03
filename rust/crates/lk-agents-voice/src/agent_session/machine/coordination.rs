//! Private ordered session coordination. Pure transitions and dispatch; no I/O.

use crate::agent_activity::drain::DrainEvent;
use crate::agent_activity::false_interruption::AudioInterruptTrigger;
use crate::agent_activity::handoff::{HandoffEvent, HandoffId};
use crate::agent_activity::scheduling::PauseCause;
use crate::agent_session::{
    AgentUpdate, Answer, HandoffRefused, HookCall, HookId, RequestId, VoiceDriver, VoiceHostEvent,
};
use crate::audio_recognition::InterruptionSignal;
use crate::events::SessionEvent;
use crate::trace::{CoreStep, InputStep, InterruptStep, RealtimeStep, Step, ToolStep, TurnStep};
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::UserState;
use lk_agents_core::tts::markup::ChatMessageExt;
use std::sync::Arc;

use crate::agent_session::effects::*;

fn effect_observation(effect: &VoiceEffect) -> Option<Step> {
    let step = match effect {
        // The journal's copy never pins a host exception (`SessionEvent::journal_form`).
        VoiceEffect::Emit(_) => return None,
        VoiceEffect::OpenSttStream { open, stt } => {
            ProviderCall::OpenLeg { leg: StreamKind::Stt, open: *open, resource: *stt }.into()
        }
        VoiceEffect::OpenVadStream { open, vad } => {
            ProviderCall::OpenLeg { leg: StreamKind::Vad, open: *open, resource: *vad }.into()
        }
        VoiceEffect::ProviderPrewarm(id) => ProviderCall::Prewarm(*id).into(),
        VoiceEffect::ProviderRelease(id) => ProviderCall::Release(*id).into(),
        VoiceEffect::CloseRealtimeSession(id) => ProviderCall::CloseRealtime(*id).into(),
        VoiceEffect::EotPredict { req, .. } => DetectorCall::RequestEot { req: *req }.into(),
        VoiceEffect::EotCancel { req, timed_out } => {
            DetectorCall::CancelEot { req: *req, timed_out: *timed_out }.into()
        }
        VoiceEffect::EotFlush(reason) => DetectorCall::Flush(*reason).into(),
        VoiceEffect::EotClose => DetectorCall::Close.into(),
        VoiceEffect::Audio(AudioOp::Gate { speech, op, cause, output: _ }) => {
            let (speech, op, cause) = (*speech, *op, *cause);
            Step::Interrupt(InterruptStep::PlayoutGate { speech, op, cause })
        }
        // The source swap's own step (`AudioInputSet`, `VideoInputSet`) is the handler's.
        VoiceEffect::Audio(
            AudioOp::SetVideoInput(_)
            | AudioOp::SetAudioInput(_)
            | AudioOp::SetAudioOutput { .. }
            | AudioOp::SetTranscriptionOutput { .. }
            | AudioOp::SetOutputEnabled { .. },
        ) => return None,
        VoiceEffect::Journal(_)
        | VoiceEffect::Answer(..)
        | VoiceEffect::Host(_)
        | VoiceEffect::ChatContext(_)
        | VoiceEffect::StorageCommit { .. }
        | VoiceEffect::StorageAbort(_)
        | VoiceEffect::Outputs(_)
        | VoiceEffect::Tap(_)
        | VoiceEffect::SessionView(_)
        | VoiceEffect::Closing
        | VoiceEffect::SttConversationItem { .. }
        | VoiceEffect::StartSpeech { .. }
        | VoiceEffect::PushKeyterms { .. }
        | VoiceEffect::RequestKeyterms { .. }
        | VoiceEffect::CancelKeyterms(_)
        | VoiceEffect::AttachRecorder { .. }
        | VoiceEffect::DetachRecorder(_)
        | VoiceEffect::FlushRecorder(_)
        | VoiceEffect::RealtimePlayer { .. }
        | VoiceEffect::CloseRealtimePlayer(_)
        | VoiceEffect::CloseRealtimePlayers
        | VoiceEffect::CancelSpeech(_)
        | VoiceEffect::Speech(..)
        | VoiceEffect::FlushStream(_)
        | VoiceEffect::InterruptionSignal { .. } => return None,
        // Journalled through the data plane's reports.
        VoiceEffect::UpdateAudioInputRouting(_)
        | VoiceEffect::AdoptOpenedInput { .. }
        | VoiceEffect::DiscardOpenedInput(_) => return None,
        VoiceEffect::Realtime { command: cmd, .. } => Step::Realtime(RealtimeStep::Sent {
            kind: cmd.kind(),
            req_id: cmd.req_id().map(str::to_owned),
        }),
        VoiceEffect::CancelTool { call_id, .. } => {
            Step::Tool(ToolStep::Cancelled { call_id: call_id.clone() })
        }
        VoiceEffect::ValidateOutput { .. }
        | VoiceEffect::CancelOutputValidation { .. }
        | VoiceEffect::RenderToolHandlingTemplate { .. }
        | VoiceEffect::CancelToolRefusal { .. }
        | VoiceEffect::AnswerExpert { .. } => return None,
    };
    Some(step)
}
/// Resolution delayed until the current domain transition has installed its sinks/overlap.
/// The insertion index preserves order among requests in the single runtime outbox.
#[derive(Debug)]
pub(crate) enum PendingInputOperation {
    Flush(StreamKind),
    NotifyInterruption(InterruptionSignal),
}

impl VoiceDriver {
    pub(crate) fn record_effect(&mut self, effect: VoiceEffect) {
        if self.journalled
            && let Some(observation) = effect_observation(&effect)
        {
            self.record_trace(observation);
        }
        self.outbox.push(effect);
    }

    pub(crate) fn record_trace(&mut self, observation: impl Into<Step>) {
        if self.journalled {
            self.outbox.push(VoiceEffect::Journal(Box::new(observation.into())));
        }
    }

    pub(crate) fn defer_callback(&mut self, callback: VoiceCallback) {
        self.callbacks.push_back(callback);
    }

    pub(crate) fn publish_event(&mut self, event: SessionEvent) {
        self.record_trace(Step::Emitted(event.journal_form()));
        self.defer_callback(VoiceCallback::SessionEvent(event));
    }

    /// Replaces this timer's deadline. Changes are visible immediately to domain queries;
    /// unlike the old staging query, reading a deadline never changes journal or output order.
    pub(crate) fn set_deadline(&mut self, timer: Timer, at: Ts) {
        self.record_trace(CoreStep::TimerArmed { timer: timer.into(), at });
        self.timers.set_deadline(timer, at);
        self.revised_deadlines.insert(timer);
    }

    pub(crate) fn clear_deadline(&mut self, timer: Timer) {
        self.record_trace(CoreStep::TimerCancelled(timer.into()));
        self.timers.clear_deadline(timer);
        self.revised_deadlines.insert(timer);
    }

    pub(crate) fn deadline(&self, timer: Timer) -> Option<Ts> {
        self.timers.deadline(timer)
    }

    pub(crate) fn flush_input(&mut self, leg: StreamKind) {
        self.record_trace(InputStep::ProviderCalled(match leg {
            StreamKind::Stt => ProviderCall::FlushStt,
            StreamKind::Vad => ProviderCall::FlushVad,
        }));
        self.pending_input.push((self.outbox.len(), PendingInputOperation::Flush(leg)));
    }

    pub(crate) fn notify_interruption(&mut self, signal: InterruptionSignal) {
        self.record_trace(InputStep::DetectorCalled(DetectorCall::Signal(signal)));
        self.pending_input
            .push((self.outbox.len(), PendingInputOperation::NotifyInterruption(signal)));
    }

    pub(crate) fn flush_turn_detector(&mut self, reason: FlushReason) {
        self.record_effect(VoiceEffect::EotFlush(reason));
    }

    /// Resolve operation identities before the next FIFO callback. A stream replacement in
    /// the same transition must be installed before its flush target is selected.
    fn resolve_input_operations(&mut self) {
        let mut inserted = 0;
        for (index, operation) in std::mem::take(&mut self.pending_input) {
            let effect = match operation {
                PendingInputOperation::Flush(leg) => {
                    let channel = match leg {
                        StreamKind::Stt => crate::audio::AudioChannel::Stt,
                        StreamKind::Vad => crate::audio::AudioChannel::Vad,
                    };
                    self.input.sinks.get(&channel).copied().map(VoiceEffect::FlushStream)
                }
                PendingInputOperation::NotifyInterruption(signal) => {
                    if let InterruptionSignal::OverlapEnded { .. } = signal {
                        let now = self.step.now;
                        self.session.spans.turns(now, |_, _, user, _| user.overlap_ended());
                    }
                    Some(VoiceEffect::InterruptionSignal {
                        overlap: self.activity.open_overlap(),
                        signal,
                    })
                }
            };
            if let Some(effect) = effect {
                self.outbox.insert(index + inserted, effect);
                inserted += 1;
            }
        }
    }

    /// Settle atomic domain transitions and their FIFO continuations. No domain handler
    /// has to request scheduling or close progression; these are coordinator guarantees.
    pub(crate) fn settle_internal(&mut self) {
        self.advance_transition();
        loop {
            self.resolve_input_operations();
            self.revised_deadlines.clear();
            if let Some(callback) = self.callbacks.pop_front() {
                self.dispatch(callback);
                self.advance_transition();
                continue;
            }
            if self.settle_reply_plans() || self.settle_deferred_calls() {
                self.advance_transition();
                continue;
            }
            if !self.publish_current_speech() {
                break;
            }
        }
        self.flush_activity();
        let speeches = self.activity.take_speech_updates();
        self.outbox
            .extend(speeches.into_iter().map(|(id, update)| VoiceEffect::Speech(id, update)));
        let launch_holds = self.activity.launch_hold();
        let view = &self.view;
        if view.launch_holds != launch_holds
            || view.config != self.session.config
            || view.agent != self.activity.agent_name()
            || view.agent_scope != self.activity.scope()
            || view.vad != self.session.vad
            || view.usage_revision != self.session.usage.revision()
        {
            self.view = Arc::new(crate::agent_session::SessionView {
                launch_holds,
                config: self.session.config.clone(),
                agent: self.activity.agent_name().to_owned(),
                agent_scope: self.activity.scope(),
                vad: self.session.vad.clone(),
                usage: self.session.usage.snapshot(),
                usage_revision: self.session.usage.revision(),
            });
            self.outbox.push(VoiceEffect::SessionView(Arc::clone(&self.view)));
        }
        let runs = &self.session.runs;
        let retired = self.activity.retire_settled_speeches(|speech| runs.names(speech));
        self.outbox.extend(
            retired.into_iter().map(|id| VoiceEffect::Host(VoiceHostEvent::SpeechRetired(id))),
        );
    }

    pub(crate) fn settle_step(&mut self) -> Vec<VoiceEffect> {
        self.settle_internal();
        self.take_outbox()
    }

    fn advance_transition(&mut self) {
        let now = self.step.now;
        loop {
            let before = self.session.close_phase();
            self.advance_close(now);
            if self.session.close_phase() == before {
                break;
            }
        }
        if !self.stopped {
            self.advance_scheduling(now);
        }
    }

    fn on_session_event(&mut self, now: Ts, ev: SessionEvent) {
        // `on_backchannel_confirmed`: clear the realtime model's buffered audio so a
        // "mm-hm" cannot prefix the next turn. The reference's `turn_detection not in
        // ("manual", "realtime_llm")` guard is unported: `TurnConfig::mode` is private.
        if matches!(ev, SessionEvent::BackchannelConfirmed)
            && self.session.config.effective.turn.interruption.detection_enabled()
            && self.activity.realtime().is_some()
        {
            self.rt_send(lk_agents_abi::RealtimeCommand::ClearAudio(
                lk_agents_abi::llm::realtime::ClearAudio {},
            ));
        }
        // The journal names the utterance's segments by one id: the realtime item's, else
        // one minted here (the pure tier cannot). The event carries only the provider's, as
        // the reference's STT path carries none.
        if let SessionEvent::UserInputTranscribed(t) = &ev {
            let minted = self.session.ids.user_transcript_item(t.is_final);
            let id = t.item_id.clone().unwrap_or(minted);
            let text = t.ev.best_alternative().map(|a| a.text.clone()).unwrap_or_default();
            self.record_trace(Step::Turn(TurnStep::TranscriptSegment {
                id: id.clone(),
                text,
                is_final: t.is_final,
            }));
        }
        if let SessionEvent::MetricsCollected { metrics, speech, counted: false } = &ev {
            // A request the session saw start is measured by its record; others count here.
            use lk_agents_abi::metrics::base::AgentMetrics;
            use lk_agents_core::usage::ModelKind;
            let s = &mut self.session;
            let started = match (&**metrics, speech) {
                (AgentMetrics::Llm(_), Some(speech)) => {
                    s.requests.get_mut(&(*speech, ModelKind::Llm))
                }
                (AgentMetrics::Tts(_), Some(speech)) => {
                    s.requests.get_mut(&(*speech, ModelKind::Tts))
                }
                _ => None,
            }
            .and_then(std::collections::VecDeque::pop_front);
            let started = started.or_else(|| match &**metrics {
                AgentMetrics::RealtimeModel(m) => s.realtime_requests.remove(&m.request_id),
                _ => None,
            });
            match started {
                Some(request) => self.session.usage.finish_measured(request, metrics),
                None => self.session.usage.observe(metrics),
            }
        }
        // The user state and away deadline are decided on the way to the host.
        self.on_session_edge(now, &ev);
        // The spans this event ended go out right before it.
        self.observe_spans(now, &ev);
        self.flush_spans();
        let closed = matches!(ev, SessionEvent::Close { .. });
        if let SessionEvent::PreemptiveGeneration(pg) = &ev {
            self.on_preemptive_generation(now, pg.new_transcript.clone(), pg.started_speaking_at);
        }
        self.record_effect(VoiceEffect::Emit(ev));
        if closed {
            self.on_session_closed();
        }
    }

    /// Speech boundaries move the user state, dated to the boundary's instant; a final transcript
    /// restores `away` or pushes the deadline out; every state edge reconciles the away timer.
    pub(crate) fn on_session_edge(&mut self, now: Ts, ev: &SessionEvent) {
        // A DTMF collection reads the same edges (and its read-back's end) off the same events.
        self.dtmf_on_event(ev);
        self.end_call_on_event(now, ev);
        match ev {
            SessionEvent::StartOfSpeech(e) => {
                {
                    let updates = self.session.states.update_user_state(e.at, UserState::Speaking);
                    self.apply_user_state_updates(updates);
                };
            }
            SessionEvent::EndOfSpeech(e) => {
                {
                    let updates = self.session.states.update_user_state(e.at, UserState::Listening);
                    self.apply_user_state_updates(updates);
                };
            }
            SessionEvent::UserInputTranscribed(t) => {
                // Any non-empty transcript is the STT recovering
                // (`AgentSession._user_input_transcribed`).
                if let Some(a) = t.ev.best_alternative() {
                    self.session.errors.on_transcript(&a.text);
                }
                if t.is_final {
                    let closing = self.session.closing_past_edge();
                    {
                        let updates = self.session.states.on_final_transcript(now, closing);
                        self.apply_user_state_updates(updates);
                    };
                }
            }
            // The IVR navigator reads the two state edges off the feed, as `IVRActivity` does.
            SessionEvent::AgentStateChanged { .. } | SessionEvent::UserStateChanged { .. } => {
                let speaking = lk_agents_abi::voice::events::AgentState::Speaking;
                if matches!(ev, SessionEvent::AgentStateChanged { new, .. } if *new == speaking)
                    && let Some(reason) = self.activity.turn_limit_agent_speaking()
                {
                    self.record_trace(crate::trace::TurnStep::TurnLimitSkipped(reason));
                }
                let closing = self.session.closing_past_edge();
                {
                    let updates = self.session.states.reconcile_away_timer(now, closing);
                    self.apply_user_state_updates(updates);
                };
            }
            SessionEvent::UserTurnExceeded(e) => self.on_user_turn_exceeded(e),
            SessionEvent::MetricsCollected { .. }
            | SessionEvent::Telemetry(_)
            | SessionEvent::Error(_)
            | SessionEvent::OverlappingSpeech(_)
            | SessionEvent::Diagnostic(_)
            | SessionEvent::EotPrediction(_)
            | SessionEvent::AgentBackchannelOpportunity(_)
            | SessionEvent::PreemptiveGeneration(_)
            | SessionEvent::UserTranscriptionTimeout(_)
            | SessionEvent::BackchannelConfirmed
            | SessionEvent::OverlapInterruption(_)
            | SessionEvent::AgentFalseInterruption { .. }
            | SessionEvent::UserTurnCompleted(_)
            | SessionEvent::SpeechCreated { .. }
            | SessionEvent::FunctionToolsExecuted(_)
            | SessionEvent::ToolExecutionUpdated(_)
            | SessionEvent::HistoryReloaded { .. }
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
            SessionEvent::ConversationItemAdded { item, .. } => {
                self.forward_chat_context(item);
                if self.session.keyterms.is_some() {
                    let item = (**item).clone();
                    self.keyterms_on_item(now, &item);
                }
            }
        }
    }

    /// `stt._push_conversation_item` on `conversation_item_added`: a message's role and text reach
    /// an STT that consumes conversation context, unless `forward_chat_context` is off.
    fn forward_chat_context(&mut self, item: &lk_agents_abi::ChatItem) {
        if !self.session.forward_chat_context {
            return;
        }
        let Some(message) = item.as_message() else { return };
        let Some(text) = message.text_content().filter(|t| !t.is_empty()) else { return };
        let Some(stt) = self.session.stt.as_ref() else { return };
        self.record_effect(VoiceEffect::SttConversationItem {
            stt: stt.id,
            role: message.role.as_str().to_owned(),
            text,
        });
    }

    /// `SessionEvent::FloorChanged` when `_current_speech` differs from the last announced one.
    /// Returns whether an emit was queued.
    pub(crate) fn publish_current_speech(&mut self) -> bool {
        let Some(holder) = self.activity.current_speech_changed() else { return false };
        self.publish_event(SessionEvent::FloorChanged { holder });
        true
    }
}

/// What one part of the session tells another within a step: queued privately by VoiceDriver, dispatched in
/// the order raised until none is left, never seen outside the step.
#[derive(Debug)]
pub(crate) enum VoiceCallback {
    EndOfTurn(crate::audio_recognition::EndOfTurnInfo),
    TurnSettled(crate::agent_activity::false_interruption::TurnSettled),
    AudioInterruptGuard(AudioInterruptGuard),
    InterruptByAudioActivity(AudioInterruptTrigger),
    SessionEvent(SessionEvent),
    /// In order with the session events whose spans it nests in.
    EouDetected(crate::audio_recognition::EouDetected),
    /// `_pause_scheduling_task` for `cause`, answered with `Paused`.
    PauseScheduling(PauseCause),
    SchedulingDrainFinished(PauseCause, PauseResult),
    /// The last speech held was released.
    SpeechesSettled,
    /// Dispatch an accepted hook completion after dependent caller state is installed.
    /// The hook owner retains its result until this callback; abandonment invalidates it.
    HookFinished(HookId),
    /// The switch's drain settled: the swap is due.
    ApplyHandoff,
    RefuseHandoff {
        id: Option<HandoffId>,
        update: Box<AgentUpdate>,
        cause: HandoffRefused,
    },
    /// The switch in progress ended without a swap.
    AgentStays,
    /// Outside a switch, a held user turn reaches the agent.
    DeliverHeldTurn,
    /// The switch ended without a swap: the host's held speech goes to the agent left driving.
    ReplayHeldSpeech,
    StartNextHandoff,
    InterruptForDtmf,
    /// The digits read back for the caller to confirm.
    ReadBackDtmf(String),
    DtmfFinished(crate::workflows::dtmf::DtmfOutcome),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PauseResult {
    Settled,
    Refused,
}

impl VoiceDriver {
    pub(crate) fn dispatch(&mut self, signal: VoiceCallback) {
        let now = self.step.now;
        match signal {
            VoiceCallback::EndOfTurn(info) => self.on_end_of_turn(now, info),
            VoiceCallback::TurnSettled(settled) => self.on_turn_settled(now, settled),
            VoiceCallback::EouDetected(detected) => self.observe_detection(now, &detected),
            VoiceCallback::AudioInterruptGuard(AudioInterruptGuard::Set(value)) => {
                self.session.by_audio_activity = value;
            }
            VoiceCallback::AudioInterruptGuard(AudioInterruptGuard::Restore) => {
                self.restore_audio_activity_guard();
            }
            VoiceCallback::InterruptByAudioActivity(trigger) => {
                self.interrupt_by_audio_activity(now, trigger);
                if let AudioInterruptTrigger::DetectorVerdict { ignore_user_transcript_until } =
                    trigger
                {
                    let effects = self
                        .activity
                        .end_agent_speech_after_overlap(now, ignore_user_transcript_until);
                    self.apply_recognition_effects(effects);
                }
            }
            VoiceCallback::SessionEvent(event) => self.on_session_event(now, event),
            VoiceCallback::PauseScheduling(cause) => self.pause_scheduling(now, cause),
            VoiceCallback::SchedulingDrainFinished(PauseCause::Handoff, result) => {
                self.handoff_step(HandoffEvent::Paused(result))
            }
            VoiceCallback::SchedulingDrainFinished(PauseCause::Drain, result) => {
                self.drain_step(DrainEvent::Paused(result))
            }
            VoiceCallback::SpeechesSettled => self.drain_step(DrainEvent::Quiet),
            VoiceCallback::HookFinished(hook) => self.dispatch_hook_completion(now, hook),
            VoiceCallback::ApplyHandoff => self.perform_handoff(now),
            VoiceCallback::RefuseHandoff { id, update, cause } => {
                self.refuse_handoff(id, update, cause)
            }
            VoiceCallback::AgentStays => self.drain_step(DrainEvent::AgentStays),
            VoiceCallback::DeliverHeldTurn => self.deliver_held_turn(now),
            VoiceCallback::ReplayHeldSpeech => self.replay_held_speech(now),
            VoiceCallback::StartNextHandoff => {
                self.handoff_step(HandoffEvent::Next);
            }
            VoiceCallback::InterruptForDtmf => self.interrupt_for_dtmf(now),
            VoiceCallback::ReadBackDtmf(digits) => self.read_back_dtmf(now, digits),
            VoiceCallback::DtmfFinished(outcome) => self.dtmf_task_finished(now, outcome),
        }
    }

    /// `drain()`: under a close, its callers wait on the close's own drain.
    pub(crate) fn drain(&mut self, waiters: Vec<RequestId>) {
        if self.session.closing.is_some() {
            return self.join_close_drain(waiters);
        }
        // A VoiceTask holds the floor: its parent's tool waits on it, so nothing would settle.
        if self.activity.has_voice_task() {
            let refused = crate::agent_session::DrainRefused::TaskRunning;
            self.record_trace(crate::trace::HandoffStep::Drained { refused: Some(refused) });
            for call in waiters {
                self.answer(Some(call), Answer::Drain(Err(refused)));
            }
            return;
        }
        self.drain_step(DrainEvent::Asked(waiters));
    }

    pub(crate) fn handoff_step(&mut self, ev: HandoffEvent) {
        for callback in self.activity.handoff_transition(ev, self.session.closing.is_some()) {
            self.defer_callback(callback);
        }
        self.issue_claimed_exits();
    }

    pub(crate) fn drain_step(&mut self, ev: DrainEvent) {
        let outcomes = self.activity.drain_transition(ev, self.session.closing.is_some());
        self.record_drain_outcomes(outcomes);
        self.issue_claimed_exits();
    }

    pub(crate) fn record_drain_outcomes(
        &mut self,
        outcomes: Vec<crate::agent_activity::drain::DrainOutcome>,
    ) {
        use crate::agent_activity::drain::DrainOutcome;
        for outcome in outcomes {
            match outcome {
                DrainOutcome::Started { exit_hook } => {
                    self.record_trace(crate::trace::HandoffStep::DrainStarted { exit_hook })
                }
                DrainOutcome::Finished { waiters, refused } => {
                    self.record_trace(crate::trace::HandoffStep::Drained { refused });
                    for call in waiters {
                        self.answer(Some(call), Answer::Drain(refused.map_or(Ok(()), Err)));
                    }
                }
                DrainOutcome::PauseScheduling => {
                    self.defer_callback(VoiceCallback::PauseScheduling(PauseCause::Drain))
                }
            }
        }
    }

    /// The `on_exit`s claimed in the last step, asked of the host; with none, over at once.
    pub(crate) fn issue_claimed_exits(&mut self) {
        let now = self.step.now;
        for (hook, agent, scope) in self.activity.take_claimed_exits() {
            let wait = self.activity.hook_wait();
            self.issue_hook(now, HookCall::OnExit { id: hook, agent, scope }, Some(wait));
        }
    }
}

impl From<ProviderCall> for Step {
    fn from(call: ProviderCall) -> Self {
        Step::Input(InputStep::ProviderCalled(call))
    }
}
impl From<DetectorCall> for Step {
    fn from(call: DetectorCall) -> Self {
        Step::Input(InputStep::DetectorCalled(call))
    }
}
