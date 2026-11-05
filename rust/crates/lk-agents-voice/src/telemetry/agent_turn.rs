//! The agent's turn: one `agent_turn` per speech, from its first generation to its end and the
//! end of the tools it started, with `agent_speaking` as a direct child and a `generation` event
//! at each step's start. The speech's scheduler entry owns its [`AgentTurnSpan`]; the runtime
//! opens the nodes, the realtime inference and the tool calls under the turn's [`SpanContext`]. A
//! preemptive generation discarded for the speech that answers the same user turn hands that
//! speech its turn, as 1.8.10 does.

use super::participant_attrs;
use crate::agent_activity::scheduling::SpeechId;
use crate::events::SpeechOutcome;
use crate::trace::InterruptCause;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::telemetry::decl::{
    Content, Failed, Json, Label, Measured, Number, Seconds, SpanContext,
};
use lk_agents_abi::telemetry::instruments::{
    InvokeAgentDuration, ModelLabels, TurnE2eLatency, TurnOnUserTurnCompletedDelay,
};
use lk_agents_abi::telemetry::spans::agent_turn::{
    AgentSpeaking, AgentTurn, AgentTurnEnd, Generation, InterruptionSource,
    PreemptiveGenerationDiscarded, ResponseFunctionCall,
};
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_abi::voice::room_io::ParticipantInfo;
use lk_agents_abi::{FunctionCall, MetricsReport, ProviderId};
use lk_agents_core::telemetry::span::{ManualSpan, Trace};
use std::collections::BTreeSet;
use std::time::Duration;

/// What a speech's turn starts with.
pub(crate) struct Reply {
    pub(crate) agent: String,
    /// The user turn replied to, for the latencies.
    pub(crate) user: Option<MetricsReport>,
    pub(crate) realtime: bool,
    /// A pipeline reply's `generate_reply(instructions=)`.
    pub(crate) instructions: Option<String>,
    /// A pipeline reply's user input.
    pub(crate) user_input: Option<String>,
    /// The LLM, for the turn's measurements.
    pub(crate) llm: Option<ProviderId>,
}

/// What the turn learns while it runs.
#[derive(Default)]
struct Progress {
    steps: u32,
    scheduled: Option<Ts>,
    authorized: Option<Ts>,
    interruption: Option<InterruptionSource>,
    /// Seconds played when the speech was cut.
    playout: Option<f64>,
    function_calls: Vec<ResponseFunctionCall>,
    e2e: Option<f64>,
    response: String,
    done: Option<(SpeechOutcome, Option<ProviderError>)>,
}

fn decl(id: &str, reply: &Reply, participant: Option<&ParticipantInfo>) -> AgentTurn {
    let pipeline = !reply.realtime;
    AgentTurn {
        speech_id: id.to_owned(),
        operation: GenAiOperation::InvokeAgent,
        agent_name: reply.agent.clone(),
        instructions: reply.instructions.clone().filter(|_| pipeline).map(Content),
        user_input: reply.user_input.clone().filter(|_| pipeline).map(Content),
        participant: participant.map(participant_attrs),
    }
}

/// `SpeechHandle._interrupt_source`: what a first interruption is attributed to.
pub(crate) fn interruption_source(cause: InterruptCause) -> InterruptionSource {
    match cause {
        InterruptCause::ByAudioActivity => InterruptionSource::AudioActivity,
        InterruptCause::UserTurn => InterruptionSource::UserTurn,
        InterruptCause::Host
        | InterruptCause::OutputReplaced
        | InterruptCause::Detector
        | InterruptCause::Superseded
        | InterruptCause::Closing
        | InterruptCause::ScheduleRejected
        | InterruptCause::ModelSwapped
        | InterruptCause::RunCancelled
        | InterruptCause::VoiceTask => InterruptionSource::Programmatic,
    }
}

/// One speech's `agent_turn` and its `agent_speaking`, with what the turn learns while it runs.
/// The speech's scheduler entry owns it; a speech that ends while tools it started still run
/// hands it to the record of those calls, and the last one to end ends it.
pub(crate) struct AgentTurnSpan {
    span: ManualSpan<AgentTurn>,
    /// The speech whose turn it is: the one that adopted it after a handover.
    speech: SpeechId,
    /// `lk.speech_id`, unique across sessions.
    id: String,
    started: Ts,
    reply: Reply,
    progress: Progress,
    speaking: Option<ManualSpan<AgentSpeaking>>,
    /// The tool calls it started that still run.
    calls: BTreeSet<String>,
}

impl std::fmt::Debug for AgentTurnSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTurnSpan").field("speech", &self.speech).field("id", &self.id).finish()
    }
}

impl AgentTurnSpan {
    /// The speech's first generation: its turn opens. `session` makes `lk.speech_id` unique.
    pub(crate) fn open(
        trace: &mut Trace,
        session: &str,
        speech: SpeechId,
        reply: Reply,
        participant: Option<&ParticipantInfo>,
    ) -> Self {
        let id = format!("{session}_{speech}");
        let span = trace.open(decl(&id, &reply, participant));
        trace.event(&span, generation(&id, 1));
        let started = trace.now();
        let progress = Progress { steps: 1, ..Progress::default() };
        AgentTurnSpan {
            span,
            speech,
            id,
            started,
            reply,
            progress,
            speaking: None,
            calls: BTreeSet::new(),
        }
    }

    /// A discarded preemptive generation's turn, continued by `speech`, the one answering the
    /// same user turn (as 1.8.10 does).
    pub(crate) fn adopt(
        mut self,
        trace: &mut Trace,
        session: &str,
        speech: SpeechId,
        reply: Reply,
        participant: Option<&ParticipantInfo>,
    ) -> Self {
        let id = format!("{session}_{speech}");
        let discarded = PreemptiveGenerationDiscarded { speech_id: std::mem::take(&mut self.id) };
        trace.event(&self.span, discarded);
        trace.set(&self.span, decl(&id, &reply, participant));
        trace.event(&self.span, generation(&id, 1));
        self.speech = speech;
        self.id = id;
        self.reply = reply;
        self.progress = Progress { steps: 1, ..Progress::default() };
        self
    }

    /// The speech whose turn it is.
    pub(crate) fn speech(&self) -> SpeechId {
        self.speech
    }

    /// What the runtime's work for the speech nests under.
    pub(crate) fn context(&self, trace: &Trace) -> Option<SpanContext> {
        trace.context(&self.span)
    }

    /// The speech entered the queue (`_mark_scheduled`), once.
    pub(crate) fn scheduled(&mut self, now: Ts) {
        self.progress.scheduled.get_or_insert(now);
    }

    /// The speech was first authorized to play (`_authorize_generation`).
    pub(crate) fn authorized(&mut self, now: Ts) {
        self.progress.authorized.get_or_insert(now);
    }

    /// The speech's first interruption landed.
    pub(crate) fn interrupted(&mut self, cause: InterruptCause) {
        self.progress.interruption.get_or_insert(interruption_source(cause));
    }

    /// One of the speech's segments finished playing at `position`.
    pub(crate) fn played(&mut self, position: Duration, interrupted: bool) {
        if !interrupted {
            return;
        }
        // A realtime response's messages play one after the other; a pipeline step is one.
        let base = self.progress.playout.filter(|_| self.reply.realtime).unwrap_or(0.0);
        self.progress.playout = Some(base + position.as_secs_f64());
    }

    /// A realtime response's function call.
    pub(crate) fn function_call(&mut self, call: &FunctionCall) {
        if !self.reply.realtime {
            return;
        }
        self.progress.function_calls.push(ResponseFunctionCall {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        });
    }

    /// The tool loop's next step starts.
    pub(crate) fn step(&mut self, trace: &mut Trace, step: u32) {
        self.progress.steps = self.progress.steps.max(step);
        trace.event(&self.span, generation(&self.id, step));
    }

    /// The speech's first audio frame: `wall` dates it for the end-to-end latency. The caller
    /// ended every other turn's `agent_speaking` first.
    pub(crate) fn speaking(
        &mut self,
        trace: &mut Trace,
        wall: f64,
        participant: Option<&ParticipantInfo>,
    ) {
        self.stopped(trace);
        if self.progress.e2e.is_none() {
            let stopped = self.reply.user.as_ref().and_then(|u| u.stopped_speaking_at);
            self.progress.e2e = stopped.map(|stopped| (wall - stopped).max(0.0));
        }
        let participant = participant.map(participant_attrs);
        self.speaking = Some(trace.open_in(&self.span, AgentSpeaking { participant }));
    }

    /// The agent left the speaking state, or another speech speaks.
    pub(crate) fn stopped(&mut self, trace: &mut Trace) {
        if let Some(span) = self.speaking.take() {
            trace.end(span, ());
        }
    }

    /// The speech's words, as committed to the conversation.
    pub(crate) fn response(&mut self, text: &str) {
        let response = &mut self.progress.response;
        if !response.is_empty() && !text.is_empty() {
            response.push(' ');
        }
        response.push_str(text);
    }

    /// A tool call the speech started.
    pub(crate) fn tool_started(&mut self, call_id: &str) {
        self.calls.insert(call_id.to_owned());
    }

    /// A tool call ended: `true` when it was one of this turn's.
    pub(crate) fn tool_ended(&mut self, call_id: &str) -> bool {
        self.calls.remove(call_id)
    }

    /// Tool calls the speech started still run.
    pub(crate) fn running_tools(&self) -> bool {
        !self.calls.is_empty()
    }

    /// The speech is done: its outcome, which the turn ends with.
    pub(crate) fn done(
        &mut self,
        trace: &mut Trace,
        outcome: SpeechOutcome,
        error: Option<ProviderError>,
    ) {
        self.stopped(trace);
        self.progress.done.get_or_insert((outcome, error));
    }

    /// The turn ends with what it has: at its speech's release, at its last tool's end, or at the
    /// close.
    pub(crate) fn end(mut self, trace: &mut Trace) {
        self.stopped(trace);
        end_turn(trace, self);
    }
}

fn generation(id: &str, step: u32) -> Generation {
    Generation {
        generation_id: format!("{id}_{step}"),
        parent_generation_id: (step > 1).then(|| format!("{id}_{}", step - 1)),
    }
}

fn end_turn(trace: &mut Trace, turn: AgentTurnSpan) {
    let AgentTurnSpan { span, id, started, reply, progress: p, .. } = turn;
    let now = trace.now();
    // A turn opened outside the session's life records nothing, its measurements included.
    let recording = trace.context(&span).is_some();
    let (outcome, error) = p.done.unwrap_or((SpeechOutcome::Completed, None));
    let interrupted = outcome == SpeechOutcome::Interrupted;
    let user = reply.user.as_ref().filter(|_| p.e2e.is_some());
    let queue_wait = p.scheduled.zip(p.authorized).map(|(scheduled, authorized)| {
        Seconds(authorized.saturating_duration_since(scheduled).as_secs_f64())
    });
    let end = AgentTurnEnd {
        generation_id: Some(format!("{id}_{}", p.steps)),
        generation_count: p.steps,
        queue_wait,
        interrupted,
        interruption_source: interrupted
            .then(|| p.interruption.unwrap_or(InterruptionSource::Programmatic)),
        playout_position: p.playout.filter(|_| interrupted).map(Seconds),
        response_text: (!p.response.is_empty()).then_some(Content(p.response)),
        function_calls: reply.realtime.then_some(Content(Json(p.function_calls))),
        e2e_latency: Measured(p.e2e.map(Seconds)),
        end_of_turn_delay: user.and_then(|u| u.end_of_turn_delay).map(Seconds),
        transcription_delay: user.and_then(|u| u.transcription_delay).map(Seconds),
        on_user_turn_completed_delay: user
            .and_then(|u| u.on_user_turn_completed_delay)
            .map(Seconds),
    };
    match error.as_ref().filter(|_| outcome == SpeechOutcome::Failed) {
        Some(error) => trace.end(span, Failed { end, error }),
        None => trace.end(span, end),
    }
    if !recording {
        return;
    }
    // Labeled as 1.8.10's report: the latency with the LLM, the hook's delay with the user turn's
    // STT.
    if let Some(e2e) = p.e2e {
        let llm = reply.llm.as_ref();
        let model = ModelLabels::of(llm.map(|l| &*l.provider), llm.map(|l| &*l.model));
        trace.measure(&TurnE2eLatency { model }, Number::Double(e2e));
    }
    if let Some(report) = &reply.user
        && let Some(delay) = report.on_user_turn_completed_delay
    {
        let stt = report.stt_metadata.as_ref();
        let model = ModelLabels::of(
            stt.and_then(|m| m.model_provider.as_deref()),
            stt.and_then(|m| m.model_name.as_deref()),
        );
        trace.measure(&TurnOnUserTurnCompletedDelay { model }, Number::Double(delay));
    }
    let duration = now.saturating_duration_since(started).as_secs_f64();
    let metric =
        InvokeAgentDuration { operation: GenAiOperation::InvokeAgent, agent: Label(reply.agent) };
    trace.measure(&metric, Number::Double(duration));
}
