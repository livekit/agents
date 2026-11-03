//! The agent's turn: one `agent_turn` per speech, from its first generation to its end and the
//! end of the tools it started, with `agent_speaking` as a direct child and a `generation` event
//! at each step's start. The runtime opens the nodes, the realtime inference and the tool calls
//! under the turn's [`SpanRef`].

use crate::agent_activity::scheduling::SpeechId;
use crate::events::SpeechOutcome;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::telemetry::decl::{Content, Label, Measured, Number, Seconds, SpanRef};
use lk_agents_abi::telemetry::instruments::InvokeAgentDuration;
use lk_agents_abi::telemetry::spans::agent_turn::{
    AgentSpeaking, AgentTurn, AgentTurnEnd, Generation,
};
use lk_agents_abi::telemetry::spans::common::GenAiOperation;
use lk_agents_abi::telemetry::spans::session::AgentSession;
use lk_agents_core::telemetry::span::{ManualSpan, Trace};
use std::collections::BTreeMap;

struct Turn {
    span: ManualSpan<AgentTurn>,
    /// `lk.speech_id`, unique across sessions.
    id: String,
    agent: String,
    started: Ts,
    steps: u32,
    /// The user turn replied to, for the latencies.
    user: Option<lk_agents_abi::MetricsReport>,
    e2e: Option<f64>,
    response: String,
    done: Option<(SpeechOutcome, Option<ProviderError>)>,
}

#[derive(Default)]
pub(crate) struct AgentTurns {
    turns: BTreeMap<SpeechId, Turn>,
    /// Running tools and the speech that started them.
    tools: BTreeMap<String, SpeechId>,
    speaking: Option<(SpeechId, ManualSpan<AgentSpeaking>)>,
}

impl AgentTurns {
    /// The speech's first generation starts: its turn opens.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn started(
        &mut self,
        trace: &mut Trace,
        root: &ManualSpan<AgentSession>,
        now: Ts,
        session: &str,
        speech: SpeechId,
        agent: &str,
        user: Option<lk_agents_abi::MetricsReport>,
    ) -> SpanRef {
        if let Some(turn) = self.turns.get(&speech) {
            return turn.span.context();
        }
        let id = format!("{session}_{speech}");
        let decl = AgentTurn {
            speech_id: id.clone(),
            operation: GenAiOperation::InvokeAgent,
            agent_name: agent.to_owned(),
            instructions: None,
            user_input: None,
            participant: None,
        };
        let span = root.child(trace, now, decl);
        span.event(trace, now, generation(&id, 1));
        let context = span.context();
        let turn = Turn {
            span,
            id,
            agent: agent.to_owned(),
            started: now,
            steps: 1,
            user,
            e2e: None,
            response: String::new(),
            done: None,
        };
        self.turns.insert(speech, turn);
        context
    }

    /// The tool loop's next step starts.
    pub(crate) fn step(&mut self, trace: &mut Trace, now: Ts, speech: SpeechId, step: u32) {
        let Some(turn) = self.turns.get_mut(&speech) else { return };
        turn.steps = turn.steps.max(step);
        turn.span.event(trace, now, generation(&turn.id, step));
    }

    /// The speech's first audio frame: `wall` dates it for the end-to-end latency.
    pub(crate) fn speaking(&mut self, trace: &mut Trace, now: Ts, wall: f64, speech: SpeechId) {
        self.stopped(trace, now);
        let Some(turn) = self.turns.get_mut(&speech) else { return };
        if turn.e2e.is_none() {
            let stopped = turn.user.as_ref().and_then(|u| u.stopped_speaking_at);
            turn.e2e = stopped.map(|stopped| (wall - stopped).max(0.0));
        }
        let span = turn.span.child(trace, now, AgentSpeaking { participant: None });
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
        if !turn.response.is_empty() && !text.is_empty() {
            turn.response.push(' ');
        }
        turn.response.push_str(text);
    }

    pub(crate) fn tool_started(&mut self, speech: SpeechId, call_id: &str) {
        if self.turns.contains_key(&speech) {
            self.tools.insert(call_id.to_owned(), speech);
        }
    }

    /// A tool ended; the turn ends with its last tool once the speech is done.
    pub(crate) fn tool_ended(&mut self, trace: &mut Trace, now: Ts, call_id: &str) {
        let Some(speech) = self.tools.remove(call_id) else { return };
        if self.turns.get(&speech).is_some_and(|t| t.done.is_some()) {
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
        turn.done.get_or_insert((outcome, error));
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
    let Turn { span, id, agent, started, steps, user, e2e, response, done } = turn;
    let (outcome, error) = done.unwrap_or((SpeechOutcome::Completed, None));
    let user = user.filter(|_| e2e.is_some());
    if let (SpeechOutcome::Failed, Some(error)) = (outcome, &error) {
        span.fail(trace, now, error);
    }
    let end = AgentTurnEnd {
        generation_id: Some(format!("{id}_{steps}")),
        generation_count: steps,
        queue_wait: None,
        interrupted: outcome == SpeechOutcome::Interrupted,
        interruption_source: None,
        playout_position: None,
        response_text: (!response.is_empty()).then_some(Content(response)),
        function_calls: None,
        e2e_latency: Measured(e2e.map(Seconds)),
        end_of_turn_delay: user.as_ref().and_then(|u| u.end_of_turn_delay).map(Seconds),
        transcription_delay: user.as_ref().and_then(|u| u.transcription_delay).map(Seconds),
        on_user_turn_completed_delay: user
            .as_ref()
            .and_then(|u| u.on_user_turn_completed_delay)
            .map(Seconds),
    };
    span.end(trace, now, end);
    let duration = now.saturating_duration_since(started).as_secs_f64();
    let metric =
        InvokeAgentDuration { operation: GenAiOperation::InvokeAgent, agent: Label(agent) };
    trace.measure(&metric, Number::Double(duration));
}
