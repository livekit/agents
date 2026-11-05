//! The activity's span entities, opened and ended through `self.telemetry`: the user turn heard
//! from the recognizer's facts (with the STT, the endpointing and the conversation only the driver
//! knows), the agent turns its speeches own, and its lifecycle callbacks' spans with the
//! `session_start` or `update_agent` they run in.

use crate::agent_activity::scheduling::SpeechId;
use crate::agent_session::{HookKind, VoiceDriver};
use crate::audio_recognition::{EouDetected, TurnDetectionMode};
use crate::events::SessionEvent;
use crate::telemetry;
use lk_agents_abi::llm::chat_context::CopyOptions;
use lk_agents_abi::{ChatContext, ChatMessage, Role, ToDictOptions};
use lk_agents_abi::{SpeechEventType, Ts};
use lk_agents_core::llm::history::ChatContextExt;

/// `_EOU_MAX_HISTORY_TURNS`: the items an `eou_detection` span records.
const EOU_HISTORY: usize = 6;

impl VoiceDriver {
    /// The STT's plugin model and provider: `user_turn` records them as 1.8.10 does, and they
    /// label its measurements.
    pub(crate) fn span_stt(&self) -> telemetry::Stt {
        let id = self.stt_identity();
        telemetry::Stt {
            model: id.as_ref().map(|id| id.model.clone()),
            provider: id.map(|id| id.provider),
        }
    }

    fn span_endpointing(&self) -> telemetry::Endpointing {
        let recognition = &self.activity.audio_recognition;
        let source = match recognition.turn_detection_mode() {
            TurnDetectionMode::Vad => telemetry::EouSource::Vad,
            TurnDetectionMode::Stt => telemetry::EouSource::Stt,
            TurnDetectionMode::Manual => telemetry::EouSource::Manual,
            TurnDetectionMode::RealtimeLlm => telemetry::EouSource::RealtimeLlm,
        };
        telemetry::Endpointing { source, delay: recognition.min_delay().as_secs_f64() }
    }

    /// The heard user turn's evidence, as the session publishes the recognizer's facts: the turn
    /// opens on its first evidence and records its speech, its transcripts and its countdown.
    pub(crate) fn hear_user_turn(&mut self, now: Ts, ev: &SessionEvent) {
        let endpointing = self.span_endpointing();
        let recognition = &self.activity.audio_recognition;
        let mode = recognition.turn_detection_mode();
        let speaking = recognition.user_speaking();
        let wall = self.step.wall_secs_at(now);
        match ev {
            SessionEvent::StartOfSpeech(e) => {
                let origin = match (e.from_vad, mode == TurnDetectionMode::RealtimeLlm) {
                    (true, _) => telemetry::UserTurnOrigin::Speech,
                    (false, true) => telemetry::UserTurnOrigin::ServerSpeech,
                    (false, false) => telemetry::UserTurnOrigin::Transcript,
                };
                self.hear_turn(e.at, origin);
                let participant = self.session.room.linked_participant();
                if let Some(turn) = &mut self.activity.user_turn {
                    turn.speech_started(&mut self.telemetry, e.at, participant);
                }
            }
            SessionEvent::EndOfSpeech(e) => {
                if let Some(turn) = &mut self.activity.user_turn {
                    turn.speech_ended(&mut self.telemetry, e.at, endpointing);
                }
            }
            SessionEvent::UserInputTranscribed(t) => {
                self.hear_transcript(now, wall, t, endpointing, speaking, mode)
            }
            SessionEvent::OverlappingSpeech(e) => {
                if let Some(turn) = &mut self.activity.user_turn {
                    turn.overlap(e);
                }
            }
            SessionEvent::UserTranscriptionTimeout(_) => {
                if let Some(mut turn) = self.activity.user_turn.take() {
                    let outcome = telemetry::UserTurnOutcome::TranscriptionTimeout;
                    turn.end(&mut self.telemetry, outcome);
                }
            }
            _ => {}
        }
    }

    /// A transcript opens a turn no speech edge opened; a final heard after the speech re-arms
    /// the countdown (1.8.10's `stt` trigger).
    fn hear_transcript(
        &mut self,
        now: Ts,
        wall: f64,
        t: &crate::events::UserTranscriptEvent,
        endpointing: telemetry::Endpointing,
        speaking: bool,
        mode: TurnDetectionMode,
    ) {
        let text = t.ev.best_alternative().map_or("", |a| a.text.as_str());
        if text.is_empty() && self.activity.user_turn.is_none() {
            return;
        }
        let received = t.ev.created_at.unwrap_or(wall);
        let kind = match t.ev.kind {
            SpeechEventType::FinalTranscript => telemetry::SttEventType::FinalTranscript,
            SpeechEventType::PreflightTranscript => telemetry::SttEventType::PreflightTranscript,
            _ => telemetry::SttEventType::InterimTranscript,
        };
        self.hear_turn(now, telemetry::UserTurnOrigin::Transcript);
        let Some(turn) = &mut self.activity.user_turn else { return };
        turn.transcript(received, kind, text, t.ev.request_id.as_str());
        if t.is_final && !speaking && mode != TurnDetectionMode::Manual {
            let stt = telemetry::Endpointing { source: telemetry::EouSource::Stt, ..endpointing };
            turn.retriggered(&mut self.telemetry, stt);
        }
    }

    /// Evidence at `at` opens the user turn being heard, when none is.
    fn hear_turn(&mut self, at: Ts, origin: telemetry::UserTurnOrigin) {
        if self.activity.user_turn.is_none() {
            let stt = self.span_stt();
            let participant = self.session.room.linked_participant();
            let turn =
                telemetry::UserTurnSpan::open(&mut self.telemetry, at, origin, participant, &stt);
            self.activity.user_turn = Some(turn);
        }
    }

    /// The session closes: the heard turn and every decided one still waiting or parked on its
    /// hook end `closed`.
    pub(crate) fn close_user_turns(&mut self) {
        let trace = &mut self.telemetry;
        let closed = telemetry::UserTurnOutcome::Closed;
        if let Some(mut turn) = self.activity.user_turn.take() {
            turn.end(trace, closed);
        }
        for turn in &mut self.activity.waiting_turns {
            turn.span.end(trace, closed);
        }
        if let Some(pending) = &mut self.activity.pending_turn {
            pending.span.end(trace, closed);
        }
    }

    /// The countdown's detector: `eou_detection` records the conversation it read.
    pub(crate) fn observe_detection(&mut self, detected: &EouDetected) {
        let predicted = matches!(detected, EouDetected::Resolved { prediction: Some(_), .. });
        let chat_ctx = predicted.then(|| {
            let transcript = self.activity.audio_recognition.audio_transcript();
            let mut items = self.session.chat_ctx.agent_chat_ctx().items;
            if !transcript.is_empty() {
                let wall = std::time::SystemTime::UNIX_EPOCH;
                items
                    .push(ChatMessage::text("eou_transcript", Role::User, transcript, wall).into());
            }
            let from = items.len().saturating_sub(EOU_HISTORY);
            let recent = ChatContext::new(items.split_off(from)).copy(&CopyOptions {
                exclude_function_call: true,
                exclude_instructions: true,
                exclude_empty_message: true,
                exclude_handoff: true,
                exclude_config_update: true,
                tools: None,
            });
            let opts = ToDictOptions {
                exclude_image: true,
                exclude_audio: true,
                exclude_timestamp: true,
                exclude_function_call: true,
                exclude_metrics: true,
                exclude_config_update: true,
            };
            lk_agents_core::telemetry::gen_ai::chat_ctx(&recent.items, &opts)
        });
        if let Some(turn) = &mut self.activity.user_turn {
            turn.detection(&mut self.telemetry, detected, chat_ctx.unwrap_or_default());
        }
    }
}

impl super::AgentActivity {
    /// Speech `speech`'s turn, while its entry or the record of its running tools holds it.
    pub(crate) fn agent_turn(&mut self, speech: SpeechId) -> Option<&mut telemetry::AgentTurnSpan> {
        if self.speech_scheduler.agent_turn(speech).is_some() {
            return self.speech_scheduler.agent_turn(speech);
        }
        self.tools.turns.iter_mut().find(|t| t.speech() == speech)
    }

    /// A session event went out: an unclaimed agent turn handover lapses.
    pub(crate) fn lapse_agent_turn_handover(&mut self) {
        self.speech_scheduler.lapse_handover();
    }
}

/// The agent turns: the speech's scheduler entry owns its turn until the speech's release, then
/// the record of the tool calls it started that still run.
impl VoiceDriver {
    /// The span speech `speech`'s host callbacks run under: its turn, while it is held.
    pub(crate) fn speech_turn(&mut self, speech: SpeechId) -> Option<telemetry::SpanContext> {
        let trace = &self.telemetry;
        self.activity.agent_turn(speech).and_then(|turn| turn.context(trace))
    }

    /// A speech's first generation: its turn opens, or continues the discarded preemptive
    /// generation's. The runtime's work for it nests under the returned span.
    pub(crate) fn agent_turn_started(
        &mut self,
        speech: SpeechId,
        reply: telemetry::Reply,
    ) -> Option<telemetry::SpanContext> {
        let scheduler = &mut self.activity.speech_scheduler;
        if let Some(turn) = scheduler.agent_turn(speech) {
            return turn.context(&self.telemetry);
        }
        let session = self.session.ids.session();
        let participant = self.session.room.local_participant();
        let trace = &mut self.telemetry;
        let turn = match scheduler.take_handover() {
            Some(turn) => turn.adopt(trace, session, speech, reply, participant),
            None => telemetry::AgentTurnSpan::open(trace, session, speech, reply, participant),
        };
        let context = turn.context(trace);
        if let Some(stray) = scheduler.begin_agent_turn(speech, turn) {
            stray.end(trace);
        }
        if let Some(context) = context {
            self.session.item_turns.speech(speech, context);
        }
        context
    }

    /// Speech `speech`'s turn, while its entry or the record of its running tools holds it.
    pub(crate) fn agent_turn(&mut self, speech: SpeechId) -> Option<&mut telemetry::AgentTurnSpan> {
        self.activity.agent_turn(speech)
    }

    /// The speech's first audio frame: its `agent_speaking` opens, every other one ends.
    pub(crate) fn agent_turn_speaking(&mut self, speech: SpeechId, wall: f64) {
        self.agent_stopped_speaking();
        let participant = self.session.room.local_participant();
        if let Some(turn) = self.activity.agent_turn(speech) {
            turn.speaking(&mut self.telemetry, wall, participant);
        }
    }

    /// The agent left the speaking state: every `agent_speaking` ends.
    pub(crate) fn agent_stopped_speaking(&mut self) {
        let trace = &mut self.telemetry;
        for turn in self.activity.speech_scheduler.agent_turns() {
            turn.stopped(trace);
        }
        for turn in &mut self.activity.tools.turns {
            turn.stopped(trace);
        }
    }

    /// The speech is done: its turn ends with this outcome.
    pub(crate) fn agent_turn_done(
        &mut self,
        speech: SpeechId,
        outcome: crate::events::SpeechOutcome,
        error: Option<lk_agents_abi::errors::ProviderError>,
    ) {
        let trace = &mut self.telemetry;
        if let Some(turn) = self.activity.speech_scheduler.agent_turn(speech) {
            turn.done(trace, outcome, error);
        }
    }

    /// The speech's entry was released with its turn: the turn ends now, or moves to the record of
    /// the tool calls it started that still run.
    pub(crate) fn agent_turn_released(&mut self, turn: telemetry::AgentTurnSpan) {
        if turn.running_tools() {
            self.activity.tools.turns.push(turn);
        } else {
            turn.end(&mut self.telemetry);
        }
    }

    /// A tool call ended: a released speech's turn ends with its last one.
    pub(crate) fn agent_turn_tool_ended(&mut self, call_id: &str) {
        for turn in self.activity.speech_scheduler.agent_turns() {
            if turn.tool_ended(call_id) {
                return;
            }
        }
        let turns = &mut self.activity.tools.turns;
        let Some(i) = turns.iter_mut().position(|t| t.tool_ended(call_id)) else { return };
        if !turns[i].running_tools() {
            turns.remove(i).end(&mut self.telemetry);
        }
    }

    /// The close: every agent turn ends with what it has.
    fn close_agent_turns(&mut self) {
        let trace = &mut self.telemetry;
        let mut turns = self.activity.speech_scheduler.take_agent_turns();
        turns.append(&mut self.activity.tools.turns);
        turns.sort_by_key(telemetry::AgentTurnSpan::speech);
        for turn in turns {
            turn.end(trace);
        }
    }
}

/// The lifecycle callbacks' spans: `on_enter` in its `start_agent_activity`, under
/// `session_start` or the agent update's `update_agent`; `on_exit` in its `drain_agent_activity`,
/// under the update or the close's `session_close`.
impl VoiceDriver {
    /// A handoff to `agent` set its toolsets up under `activation`: the spans its drain and the
    /// arriving agent's `on_enter` open as.
    pub(crate) fn arriving(&mut self, agent: &str, activation: Option<telemetry::Activation>) {
        if let Some(a) = activation {
            self.activity.arriving = Some(telemetry::Arriving {
                agent: agent.to_owned(),
                update: Some(a.update),
                activity: a.activity,
            });
        }
    }

    /// The activity an agent's input streams open under (`start_agent_activity`, as the
    /// reference's STT pump): the arriving agent's, reserved, else the running one's.
    pub(crate) fn activity_span(&self) -> Option<telemetry::SpanContext> {
        (self.activity.arriving.as_ref().map(|a| a.activity.context))
            .or(self.activity.running_activity)
    }

    /// The agent update's `update_agent`, opened now unless one is open: as reserved when the
    /// arriving agent's toolsets were set up.
    fn agent_update(
        &mut self,
        reserved: Option<telemetry::Reserved<telemetry::UpdateAgent>>,
        decl: telemetry::UpdateAgent,
    ) -> Option<&telemetry::Holding<telemetry::UpdateAgent>> {
        if self.activity.update.is_none() {
            let root = self.session.span.as_ref()?;
            let span = match reserved {
                Some(r) => self.telemetry.open_in_reserved(root, r, decl),
                None => self.telemetry.open_in(root, decl),
            };
            self.activity.update = Some(telemetry::Holding { span, done: false });
        }
        self.activity.update.as_ref()
    }

    /// A lifecycle callback the host is asked to run: its span, in the span its activity runs in.
    /// `arriving` is the agent the switch in progress brings in, which an `on_exit` leaves for.
    pub(crate) fn hook_span(
        &mut self,
        kind: HookKind,
        arriving: Option<&str>,
    ) -> Option<telemetry::HookSpan> {
        self.session.span.as_ref()?;
        let label = self.activity.agent.clone();
        let reserved_update = self.activity.arriving.as_ref().and_then(|a| a.update);
        match kind {
            HookKind::OnEnter => self.on_enter_span(label, reserved_update),
            HookKind::OnExit => self.on_exit_span(label, arriving, reserved_update),
            HookKind::OnUserTurnCompleted
            | HookKind::OnUserTurnExceeded
            | HookKind::FillerSource
            | HookKind::FunctionToolsExecuted => None,
        }
    }

    /// `on_enter` runs in the agent's new activity: under `session_start` for the first agent,
    /// else under the switch's `update_agent`.
    fn on_enter_span(
        &mut self,
        label: String,
        reserved_update: Option<telemetry::Reserved<telemetry::UpdateAgent>>,
    ) -> Option<telemetry::HookSpan> {
        let llm = self.session.llm.as_ref().map(|l| &l.provider);
        let decl = telemetry::StartAgentActivity {
            agent_label: label.clone(),
            operation: telemetry::GenAiOperation::CreateAgent,
            agent_name: label.clone(),
            provider: llm
                .and_then(|l| lk_agents_core::telemetry::gen_ai::provider_name(&l.provider)),
            model: llm.map(|l| l.model.clone()),
        };
        let reserved =
            (self.activity.arriving.take()).filter(|a| a.agent == label).map(|a| a.activity);
        let (under, activity) = match &self.session.starting {
            Some(start) if self.activity.update.is_none() => (
                telemetry::Under::Start,
                open_activity(&mut self.telemetry, &start.span, reserved, decl),
            ),
            _ => {
                let update = telemetry::UpdateAgent {
                    previous_agent_label: String::new(),
                    agent_label: label.clone(),
                };
                self.agent_update(reserved_update, update)?;
                let update = &self.activity.update.as_ref()?.span;
                (
                    telemetry::Under::Update,
                    open_activity(&mut self.telemetry, update, reserved, decl),
                )
            }
        };
        self.activity.running_activity = self.telemetry.context(&activity);
        let hook = self.telemetry.open_in(&activity, telemetry::OnEnter { agent_label: label });
        Some(telemetry::HookSpan::Enter { under, activity, hook })
    }

    /// `on_exit` runs in the leaving agent's drain: under `session_close` when the session
    /// closes, else under the switch's `update_agent`.
    fn on_exit_span(
        &mut self,
        label: String,
        arriving: Option<&str>,
        reserved_update: Option<telemetry::Reserved<telemetry::UpdateAgent>>,
    ) -> Option<telemetry::HookSpan> {
        let decl = telemetry::DrainAgentActivity { agent_label: label.clone() };
        let (under, drain) = match &self.session.closing_span {
            Some(close) => (telemetry::Under::Close, self.telemetry.open_in(close, decl)),
            None => {
                let update = telemetry::UpdateAgent {
                    previous_agent_label: label.clone(),
                    agent_label: arriving.unwrap_or_default().to_owned(),
                };
                self.agent_update(reserved_update, update)?;
                let update = &self.activity.update.as_ref()?.span;
                (telemetry::Under::Update, self.telemetry.open_in(update, decl))
            }
        };
        let hook = self.telemetry.open_in(&drain, telemetry::OnExit { agent_label: label });
        Some(telemetry::HookSpan::Exit { under, drain, hook })
    }

    /// The callback returned, raised or timed out: its span ends, and the activity it ran in; the
    /// `session_start` or `update_agent` holding it ends once done and holding no other.
    pub(crate) fn hook_span_ended(
        &mut self,
        span: telemetry::HookSpan,
        failure: Option<&dyn telemetry::SpanError>,
    ) {
        let entered = matches!(span, telemetry::HookSpan::Enter { .. });
        let under = span.end(&mut self.telemetry, failure);
        match (entered, under) {
            (true, telemetry::Under::Start) => {
                if let Some(start) = &mut self.session.starting {
                    start.done = true;
                }
            }
            (true, telemetry::Under::Update) => {
                if let Some(update) = &mut self.activity.update {
                    update.done = true;
                }
            }
            _ => {}
        }
        self.settle_holdings();
    }

    /// `session_start` and `update_agent` end once done and holding no running activity.
    pub(crate) fn settle_holdings(&mut self) {
        let hooks = &self.activity.hooks;
        let busy = |under| hooks.running_under(under);
        if self.session.starting.as_ref().is_some_and(|s| s.done && !busy(telemetry::Under::Start))
            && let Some(start) = self.session.starting.take()
        {
            self.telemetry.end(start.span, ());
        }
        if self.activity.update.as_ref().is_some_and(|u| u.done && !busy(telemetry::Under::Update))
            && let Some(update) = self.activity.update.take()
        {
            self.telemetry.end(update.span, ());
        }
    }

    /// A `VoiceTask` suspends `previous` for `next`: `next`'s `on_enter` runs under this
    /// `update_agent`, which ends with it, or [`VoiceDriver::agent_updated`] without one.
    pub(crate) fn agent_suspended(&mut self, previous: &str, next: &str) {
        let reserved = self.activity.arriving.as_ref().and_then(|a| a.update);
        let decl = telemetry::UpdateAgent {
            previous_agent_label: previous.to_owned(),
            agent_label: next.to_owned(),
        };
        if self.agent_update(reserved, decl).is_none() {
            return;
        }
        let Some(update) = &self.activity.update else { return };
        let pause = telemetry::PauseAgentActivity { agent_label: previous.to_owned() };
        let span = self.telemetry.open_in(&update.span, pause);
        self.telemetry.end(span, ());
    }

    /// The task ended and its parent is back, without its hooks.
    pub(crate) fn agent_resumed(&mut self, previous: &str, next: &str) {
        self.agent_updated();
        let Some(root) = self.session.span.as_ref() else { return };
        let decl = telemetry::UpdateAgent {
            previous_agent_label: previous.to_owned(),
            agent_label: next.to_owned(),
        };
        let update = self.telemetry.open_in(root, decl);
        let resume = telemetry::ResumeAgentActivity { agent_label: next.to_owned() };
        let span = self.telemetry.open_in(&update, resume);
        self.telemetry.end(span, ());
        self.telemetry.end(update, ());
    }

    /// The update is over with no `on_enter` to end it.
    pub(crate) fn agent_updated(&mut self) {
        if let Some(update) = &mut self.activity.update {
            update.done = true;
            self.settle_holdings();
        }
    }

    /// The close: every callback's span still open ends, then the agent update's.
    fn close_activity_spans(&mut self) {
        for span in self.activity.hooks.take_spans() {
            span.end(&mut self.telemetry, None);
        }
        if let Some(update) = self.activity.update.take() {
            self.telemetry.end(update.span, ());
        }
    }

    /// The close of the activity's spans, in the order the session ends them: the user turns,
    /// the agent turns, the callbacks.
    pub(crate) fn close_activity(&mut self) {
        self.close_user_turns();
        self.close_agent_turns();
        self.close_activity_spans();
    }
}

#[cfg(test)]
#[path = "spans_tests.rs"]
mod tests;

/// An agent's activity, opened as the span its toolsets reserved when they did.
fn open_activity<P>(
    trace: &mut telemetry::Trace,
    parent: &telemetry::ManualSpan<P>,
    reserved: Option<telemetry::Reserved<telemetry::StartAgentActivity>>,
    decl: telemetry::StartAgentActivity,
) -> telemetry::ManualSpan<telemetry::StartAgentActivity>
where
    P: lk_agents_abi::telemetry::decl::SpanDecl
        + lk_agents_abi::telemetry::decl::ParentOf<telemetry::StartAgentActivity>,
{
    match reserved {
        Some(r) => trace.open_in_reserved(parent, r, decl),
        None => trace.open_in(parent, decl),
    }
}
