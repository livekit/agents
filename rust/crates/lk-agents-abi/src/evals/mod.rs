//! `livekit.agents.evals`. `Judge` (`judge.py:166`) stays in Python, since `JudgeGroup` takes the
//! structural [`Evaluator`] Protocol.

mod error;
mod evaluator;
mod group;
mod judgment;
mod llm_judge;
pub mod prompts;
pub mod transcript;

#[cfg(test)]
mod tests;

pub use error::JudgeError;
pub use evaluator::{EvaluateRequest, Evaluator, JudgeLlm, reference_transcript};
pub use group::{EvaluationResult, EvaluationRun, JudgeFailure, JudgeGroup};
pub use judgment::{JudgmentResult, Verdict};
#[cfg(feature = "python")]
pub use judgment::{PyJudgmentResult, PyVerdict};
pub use llm_judge::{
    HandoffJudge, INTENT_SYSTEM_PROMPT, IntentVerdict, JUDGE_CONN_OPTIONS, JUDGE_INFERENCE_CLASS,
    LlmJudge, TaskCompletionJudge, check_intent_tool, judge_intent, parse_intent_arguments,
    accuracy_judge, coherence_judge, conciseness_judge, evaluate_with_llm, handoff_judge,
    judgment_chat_ctx, judgment_extra, parse_verdict_arguments, relevancy_judge, safety_judge,
    submit_verdict_tool, task_completion_judge, tool_use_judge,
};
pub use prompts::JudgePrompt;
