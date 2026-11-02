//! A generation step's committed words, metrics and completion report.

use super::llm::LlmGenerationData;
use super::{
    FinishInputs, GenEvent, ReplyContext, ReplySource, TurnAccumulator, TurnOutcome, TurnStats,
};
use crate::agent_activity::scheduling::SpeechId;
use crate::generation::effects::ReplyStartupEffect;
use lk_agents_abi::{MetricsMetadata, MetricsReport};
use std::time::SystemTime;

pub struct Finished {
    pub outcome: TurnOutcome,
    pub interrupted: bool,
}

pub struct CompletionObservation {
    pub speech: SpeechId,
    pub step: u32,
    pub adopted_user_metrics: Option<MetricsReport>,
    pub provider_request_ids: Option<Vec<String>>,
    pub has_tool_output: bool,
    pub tools_running: bool,
}

/// Adoption updates a speculative reply's user turn before its metrics are finalized.
pub fn finish_step(
    mut accumulator: TurnAccumulator,
    inputs: FinishInputs,
    observed: CompletionObservation,
) -> (Finished, GenEvent) {
    if let Some(metrics) = observed.adopted_user_metrics {
        accumulator.set_user_metrics(metrics);
    }
    let interrupted = inputs.interrupted;
    let mut outcome = accumulator.finish(inputs);
    if let Some(message) = outcome.assistant_message.as_mut() {
        message.metrics.provider_request_ids = observed.provider_request_ids;
    }
    let spoken = GenEvent::StepSpoken {
        speech: observed.speech,
        step: observed.step,
        interrupted,
        has_tool_output: observed.has_tool_output,
        tools_running: observed.tools_running,
        item: outcome.assistant_message.clone().map(Box::new),
    };
    (Finished { outcome, interrupted }, spoken)
}

impl ReplyContext {
    /// A reply refused before it speaks has no generated words or provider timings.
    pub fn empty_outcome(
        &self,
        started_at: SystemTime,
        stopped_at: f64,
        interrupted: bool,
    ) -> TurnOutcome {
        TurnAccumulator::new(started_at, self.user_metrics.clone()).finish(FinishInputs {
            stopped_speaking_at: stopped_at,
            interrupted,
            stats: TurnStats::default(),
            message_id: self.message_id.clone(),
        })
    }
}

impl ReplySource {
    /// `say()` times its voice; only a model request contributes model statistics.
    pub fn turn_stats(
        &self,
        data: LlmGenerationData,
        llm_metadata: Option<MetricsMetadata>,
        tts_metadata: Option<MetricsMetadata>,
    ) -> TurnStats {
        match self {
            Self::Llm => TurnStats {
                llm_ttft: data.metrics.ttft(),
                llm_tps: data.tps,
                llm_started_at: Some(data.started_at),
                llm_metadata,
                tts_metadata,
                llm_extra: data.extra,
            },
            _ => TurnStats::default(),
        }
    }
}

pub enum ReplyStartupEvent {
    Begin { speech: SpeechId },
    Planned(Option<super::PlannedReply>),
}

pub enum Execution {
    Pipeline,
    Realtime,
}

impl ReplyContext {
    /// A speculative request starts only after its preceding speech is on the accepted record.
    pub fn startup(
        &mut self,
        event: ReplyStartupEvent,
        now: crate::agent_session::Now,
    ) -> ReplyStartupEffect {
        match event {
            ReplyStartupEvent::Begin { speech } => match self.replan.take() {
                Some(replan) => ReplyStartupEffect::RequestPlan(super::ReplyPlan {
                    speech,
                    after: replan.after,
                    spec: replan.spec,
                    tool_choice: self.tool_choice.clone(),
                }),
                None => ReplyStartupEffect::Run,
            },
            ReplyStartupEvent::Planned(Some(planned)) => {
                self.request = planned.request;
                ReplyStartupEffect::Run
            }
            ReplyStartupEvent::Planned(None) => ReplyStartupEffect::Finished(Finished {
                outcome: self.empty_outcome(now.wall, lk_agents_abi::unix_secs(now.wall), true),
                interrupted: true,
            }),
        }
    }
}

impl ReplySource {
    pub fn execution(&self) -> Execution {
        match self {
            Self::Realtime => Execution::Realtime,
            _ => Execution::Pipeline,
        }
    }
}
