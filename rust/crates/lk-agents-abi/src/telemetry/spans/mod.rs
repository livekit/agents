//! Every framework span, its start and end attributes and its events (`telemetry.md`, "Span
//! API"). The registry, the consumer schema and the trace rules are generated from these.

pub mod agent_turn;
pub mod common;
pub mod inference;
pub mod job;
pub mod session;

use crate::telemetry::decl::Declared;

/// Every `handle = Manual` declaration, the spans the voice core records, for the DST's span
/// rule. `lk-agents-codegen --check` rejects a manual span missing here.
pub const MANUAL: &[Declared] = &[
    Declared::of::<session::AgentSession>(),
    Declared::of::<session::SessionStart>(),
    Declared::of::<session::SessionClose>(),
    Declared::of::<session::UpdateAgent>(),
    Declared::of::<session::StartAgentActivity>(),
    Declared::of::<session::OnEnter>(),
    Declared::of::<session::DrainAgentActivity>(),
    Declared::of::<session::OnExit>(),
    Declared::of::<session::PauseAgentActivity>(),
    Declared::of::<session::ResumeAgentActivity>(),
    Declared::of::<session::Amd>(),
    Declared::of::<session::KeytermDetection>(),
    Declared::of::<agent_turn::AgentTurn>(),
    Declared::of::<agent_turn::AgentSpeaking>(),
    Declared::of::<user_turn::UserTurn>(),
    Declared::of::<user_turn::UserSpeaking>(),
    Declared::of::<user_turn::EouWait>(),
    Declared::of::<user_turn::EouDetection>(),
    Declared::of::<user_turn::OnUserTurnCompleted>(),
];

/// Text runs and evaluation judges.
pub mod text {
    use super::common::GenAiOperation;
    use crate::telemetry::decl::{Attrs, Current, External, GenAiContent, SpanValue, span};
    use crate::telemetry::semconv::otel;

    /// One A2A request; continues the caller's `traceparent`.
    #[span(name = "a2a_request", parent = External, kind = Server)]
    #[derive(Attrs)]
    pub struct A2aRequest {
        /// `invoke_agent`.
        #[attr(otel::GEN_AI_OPERATION_NAME)]
        pub operation: GenAiOperation,
        #[attr(otel::GEN_AI_AGENT_NAME)]
        pub agent_name: String,
        /// The A2A task the request runs.
        #[attr("lk.a2a.task_id")]
        pub task_id: String,
    }

    /// One `AgentRun`.
    #[span(name = "agent_run", parent = A2aRequest | Current, end = AgentRunEnd)]
    #[derive(Attrs)]
    pub struct AgentRun {
        /// `invoke_agent`.
        #[attr(otel::GEN_AI_OPERATION_NAME)]
        pub operation: GenAiOperation,
        #[attr(otel::GEN_AI_AGENT_NAME)]
        pub agent_name: String,
    }

    #[derive(Attrs)]
    pub struct AgentRunEnd {
        /// How the run ended.
        #[attr("lk.run.status")]
        pub status: RunStatus,
    }

    #[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RunStatus {
        Completed,
        InputRequired,
        Failed,
    }

    /// One evaluation judge; its judge model's `llm_request` is a child.
    #[span(name = "judge_evaluation", parent = Current, end = JudgeEvaluationEnd)]
    #[derive(Attrs)]
    pub struct JudgeEvaluation {
        #[attr(otel::GEN_AI_EVALUATION_NAME)]
        pub name: String,
    }

    #[derive(Attrs)]
    pub struct JudgeEvaluationEnd {
        /// The verdict.
        #[attr(otel::GEN_AI_EVALUATION_SCORE_LABEL)]
        pub verdict: Option<String>,
        #[attr(otel::GEN_AI_EVALUATION_EXPLANATION)]
        pub explanation: Option<GenAiContent<String>>,
    }
}

pub mod user_turn;
