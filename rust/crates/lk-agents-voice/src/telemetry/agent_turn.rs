//! The agent's turn: one `agent_turn` per speech, from its first generation to its end and the
//! end of the tools it started, with `agent_speaking` as a direct child and a `generation` event
//! at each step's start. The runtime opens the nodes, the realtime inference and the tool calls
//! under the turn's [`SpanRef`]. A preemptive generation discarded for the speech that answers
//! the same user turn hands that speech its turn, as 1.8.10 does.

use super::participant_attrs;
use crate::agent_activity::scheduling::SpeechId;
use crate::events::SpeechOutcome;
use crate::trace::InterruptCause;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::telemetry::decl::{Content, Json, Label, Measured, Number, Seconds, SpanRef};
use lk_agents_abi::telemetry::instruments::{
    InvokeAgentDuration, ModelLabels, TurnE2eLatency, TurnOnUserTurnCompletedDelay,
};
use lk_agents_abi::telemetry::spans::agent_turn::{
    AgentSpeaking, AgentTurn, AgentTurnEnd, Generation, InterruptionSource,
    PreemptiveGenerationDiscarded,
};
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_abi::telemetry::spans::session::AgentSession;
use lk_agents_abi::voice::room_io::ParticipantInfo;
use lk_agents_abi::{FunctionCall, MetricsReport, ProviderId};
use lk_agents_core::telemetry::span::{ManualSpan, Trace};
use std::collections::BTreeMap;
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
    function_calls: Vec<serde_json::Value>,
    e2e: Option<f64>,
    response: String,
    done: Option<(SpeechOutcome, Option<ProviderError>)>,
}

struct Turn {
    span: ManualSpan<AgentTurn>,
    /// `lk.speech_id`, unique across sessions.
    id: String,
    started: Ts,
    reply: Reply,
    progress: Progress,
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

#[derive(Default)]
pub(crate) struct AgentTurns {
    turns: BTreeMap<SpeechId, Turn>,
    /// Running tools and the speech that started them.
    tools: BTreeMap<String, SpeechId>,
    speaking: Option<(SpeechId, ManualSpan<AgentSpeaking>)>,
    /// A discarded preemptive generation whose turn the next speech of this step continues.
    handover: Option<SpeechId>,
    /// The agent's own participant, while the room is connected.
    pub(crate) participant: Option<ParticipantInfo>,
}

impl AgentTurns {
    /// The speech's first generation starts: its turn opens, or continues the discarded
    /// preemptive generation's.
    pub(crate) fn started(
        &mut self,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        now: Ts,
        session: &str,
        speech: SpeechId,
        reply: Reply,
    ) -> SpanRef {
        if let Some(turn) = self.turns.get(&speech) {
            return turn.span.context();
        }
        let id = format!("{session}_{speech}");
        let decl = decl(&id, &reply, self.participant.as_ref());
        let (span, started) = match self.handover.take().and_then(|old| self.turns.remove(&old)) {
            Some(old) => {
                let discarded = PreemptiveGenerationDiscarded { speech_id: old.id };
                old.span.event(trace, now, discarded);
                old.span.set(trace, decl);
                (old.span, old.started)
            }
            None => (root.child(trace, now, decl), now),
        };
        span.event(trace, now, generation(&id, 1));
        let context = span.context();
        let progress = Progress { steps: 1, ..Progress::default() };
        self.turns.insert(speech, Turn { span, id, started, reply, progress });
        context
    }

    /// A preemptive generation was discarded for a speech this step starts: it continues the
    /// discarded one's turn. Unclaimed by the end of the step, the turn stays the discarded one's.
    pub(crate) fn discarded(&mut self, speech: SpeechId) {
        if self.turns.contains_key(&speech) {
            self.handover = Some(speech);
        }
    }

    /// The step is over: an unclaimed handover lapses.
    pub(crate) fn settle(&mut self) {
        self.handover = None;
    }

    /// The speech entered the queue (`_mark_scheduled`), once.
    pub(crate) fn scheduled(&mut self, now: Ts, speech: SpeechId) {
        if let Some(turn) = self.turns.get_mut(&speech) {
            turn.progress.scheduled.get_or_insert(now);
        }
    }

    /// The speech was first authorized to play (`_authorize_generation`).
    pub(crate) fn authorized(&mut self, now: Ts, speech: SpeechId) {
        if let Some(turn) = self.turns.get_mut(&speech) {
            turn.progress.authorized.get_or_insert(now);
        }
    }

    /// The speech's first interruption landed.
    pub(crate) fn interrupted(&mut self, speech: SpeechId, cause: InterruptCause) {
        if let Some(turn) = self.turns.get_mut(&speech) {
            turn.progress.interruption.get_or_insert(interruption_source(cause));
        }
    }

    /// One of the speech's segments finished playing at `position`.
    pub(crate) fn played(&mut self, speech: SpeechId, position: Duration, interrupted: bool) {
        let Some(turn) = self.turns.get_mut(&speech) else { return };
        if !interrupted {
            return;
        }
        // A realtime response's messages play one after the other; a pipeline step is one.
        let base = turn.progress.playout.filter(|_| turn.reply.realtime).unwrap_or(0.0);
        turn.progress.playout = Some(base + position.as_secs_f64());
    }

    /// A realtime response's function call.
    pub(crate) fn function_call(&mut self, speech: SpeechId, call: &FunctionCall) {
        let Some(turn) = self.turns.get_mut(&speech).filter(|t| t.reply.realtime) else { return };
        let mut fields = serde_json::json!({
            "id": call.id,
            "call_id": call.call_id,
            "arguments": call.arguments,
            "name": call.name,
            "extra": call.extra,
        });
        if let Some(group) = &call.group_id {
            fields["group_id"] = serde_json::Value::String(group.clone());
        }
        turn.progress.function_calls.push(fields);
    }

    /// The tool loop's next step starts.
    pub(crate) fn step(&mut self, trace: &mut Trace, now: Ts, speech: SpeechId, step: u32) {
        let Some(turn) = self.turns.get_mut(&speech) else { return };
        turn.progress.steps = turn.progress.steps.max(step);
        turn.span.event(trace, now, generation(&turn.id, step));
    }

    /// The speech's first audio frame: `wall` dates it for the end-to-end latency.
    pub(crate) fn speaking(&mut self, trace: &mut Trace, now: Ts, wall: f64, speech: SpeechId) {
        self.stopped(trace, now);
        let Some(turn) = self.turns.get_mut(&speech) else { return };
        if turn.progress.e2e.is_none() {
            let stopped = turn.reply.user.as_ref().and_then(|u| u.stopped_speaking_at);
            turn.progress.e2e = stopped.map(|stopped| (wall - stopped).max(0.0));
        }
        let participant = self.participant.as_ref().map(participant_attrs);
        let span = turn.span.child(trace, now, AgentSpeaking { participant });
        self.speaking = Some((speech, span));
    }

    /// The agent left the speaking state.
    pub(crate) fn stopped(&mut self, trace: &mut Trace, now: Ts) {
        if let Some((_, span)) = self.speaking.take() {
            span.end(trace, now, ());
        }
    }

    /// The speech's words, as committed to the conversation.
    pub(crate) fn response(&mut self, speech: SpeechId, text: &str) {
        let Some(turn) = self.turns.get_mut(&speech) else { return };
        let response = &mut turn.progress.response;
        if !response.is_empty() && !text.is_empty() {
            response.push(' ');
        }
        response.push_str(text);
    }

    pub(crate) fn tool_started(&mut self, speech: SpeechId, call_id: &str) {
        if self.turns.contains_key(&speech) {
            self.tools.insert(call_id.to_owned(), speech);
        }
    }

    /// A tool ended; the turn ends with its last tool once the speech is done.
    pub(crate) fn tool_ended(&mut self, trace: &mut Trace, now: Ts, call_id: &str) {
        let Some(speech) = self.tools.remove(call_id) else { return };
        if self.turns.get(&speech).is_some_and(|t| t.progress.done.is_some()) {
            self.finish(trace, now, speech);
        }
    }

    /// The speech is done; its turn ends now, or with the last tool it still runs.
    pub(crate) fn done(
        &mut self,
        trace: &mut Trace,
        now: Ts,
        speech: SpeechId,
        outcome: SpeechOutcome,
        error: Option<ProviderError>,
    ) {
        if self.speaking.as_ref().is_some_and(|(s, _)| *s == speech) {
            self.stopped(trace, now);
        }
        let Some(turn) = self.turns.get_mut(&speech) else { return };
        turn.progress.done.get_or_insert((outcome, error));
        self.finish(trace, now, speech);
    }

    fn finish(&mut self, trace: &mut Trace, now: Ts, speech: SpeechId) {
        if self.tools.values().any(|s| *s == speech) {
            return;
        }
        if let Some(turn) = self.turns.remove(&speech) {
            end_turn(trace, now, turn);
        }
    }

    /// The session closed: every turn ends with what it has.
    pub(crate) fn close(&mut self, trace: &mut Trace, now: Ts) {
        self.stopped(trace, now);
        self.tools.clear();
        self.handover = None;
        for (_, turn) in std::mem::take(&mut self.turns) {
            end_turn(trace, now, turn);
        }
    }
}

fn generation(id: &str, step: u32) -> Generation {
    Generation {
        generation_id: format!("{id}_{step}"),
        parent_generation_id: (step > 1).then(|| format!("{id}_{}", step - 1)),
    }
}

fn end_turn(trace: &mut Trace, now: Ts, turn: Turn) {
    let Turn { span, id, started, reply, progress: p } = turn;
    let (outcome, error) = p.done.unwrap_or((SpeechOutcome::Completed, None));
    let interrupted = outcome == SpeechOutcome::Interrupted;
    let user = reply.user.as_ref().filter(|_| p.e2e.is_some());
    if let (SpeechOutcome::Failed, Some(error)) = (outcome, &error) {
        span.fail(trace, now, error);
    }
    let queue_wait = p.scheduled.zip(p.authorized).map(|(scheduled, authorized)| {
        Seconds(authorized.saturating_duration_since(scheduled).as_secs_f64())
    });
    let calls = serde_json::Value::Array(p.function_calls);
    let end = AgentTurnEnd {
        generation_id: Some(format!("{id}_{}", p.steps)),
        generation_count: p.steps,
        queue_wait,
        interrupted,
        interruption_source: interrupted
            .then(|| p.interruption.unwrap_or(InterruptionSource::Programmatic)),
        playout_position: p.playout.filter(|_| interrupted).map(Seconds),
        response_text: (!p.response.is_empty()).then_some(Content(p.response)),
        function_calls: reply.realtime.then_some(Content(Json(calls))),
        e2e_latency: Measured(p.e2e.map(Seconds)),
        end_of_turn_delay: user.and_then(|u| u.end_of_turn_delay).map(Seconds),
        transcription_delay: user.and_then(|u| u.transcription_delay).map(Seconds),
        on_user_turn_completed_delay: user
            .and_then(|u| u.on_user_turn_completed_delay)
            .map(Seconds),
    };
    span.end(trace, now, end);
    // Labeled as 1.8.10's report: the latency with the LLM, the hook's delay with the user turn's
    // STT.
    let labels = |provider: &str, model: &str| ModelLabels {
        provider: Label(provider.to_owned()),
        model: Label(model.to_owned()),
    };
    // Unknown metadata leaves the measurement unlabeled, as 1.8.10 records it.
    if let Some(e2e) = p.e2e {
        let value = Number::Double(e2e);
        match &reply.llm {
            Some(llm) => {
                let model = labels(&llm.provider, &llm.model);
                trace.measure(&TurnE2eLatency { model }, value);
            }
            None => trace.measure_unlabeled::<TurnE2eLatency>(value),
        }
    }
    if let Some(report) = &reply.user
        && let Some(delay) = report.on_user_turn_completed_delay
    {
        let value = Number::Double(delay);
        let stt = report.stt_metadata.as_ref();
        match stt.and_then(|m| m.model_provider.as_ref().zip(m.model_name.as_ref())) {
            Some((provider, model)) => {
                let model = labels(provider, model);
                trace.measure(&TurnOnUserTurnCompletedDelay { model }, value);
            }
            None => trace.measure_unlabeled::<TurnOnUserTurnCompletedDelay>(value),
        }
    }
    let duration = now.saturating_duration_since(started).as_secs_f64();
    let metric =
        InvokeAgentDuration { operation: GenAiOperation::InvokeAgent, agent: Label(reply.agent) };
    trace.measure(&metric, Number::Double(duration));
}
