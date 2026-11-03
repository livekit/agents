//! Conversation evals (testing.md "Conversation evals"): the judges of one evaluation run
//! concurrently on one conversation, the built-in ones natively and the application's through the
//! host. Their prompts, transcripts and scoring are runtime support's.

use std::sync::Arc;

use crate::errors::HostError;
use crate::usage::TokenUsage;
use crate::utils::host_ref::HostRef;
use crate::{ChatItem, Llm};

/// One judge's decision; `verdict` is `"pass"`, `"fail"` or `"maybe"`.
#[crate::host(record(
    module = "evals",
    name = "JudgmentResult",
    path = "livekit.agents.evals",
    dataclass,
    emit,
    extract
))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[python(property(
    name = "passed",
    ret = "bool",
    hook = "judgment_result_passed",
    doc = "Whether the evaluation passed. Maybe is treated as not passed."
))]
#[python(property(
    name = "failed",
    ret = "bool",
    hook = "judgment_result_failed",
    doc = "Whether the evaluation failed. Maybe is treated as not failed."
))]
#[python(property(
    name = "uncertain",
    ret = "bool",
    hook = "judgment_result_uncertain",
    doc = "Whether the judge was uncertain about the verdict."
))]
pub struct Judgment {
    #[host(annotation = "typing.Literal['pass', 'fail', 'maybe']")]
    pub verdict: String,
    pub reasoning: String,
    /// The criteria the judge evaluated against.
    #[host(default = "")]
    pub instructions: String,
}

/// One judge of an evaluation.
#[crate::host(record(module = "evals", extract))]
#[derive(Debug, Clone)]
pub enum JudgeSpec {
    /// A built-in judge, by its name (`"accuracy"`, `"task_completion"`, ...).
    BuiltIn { name: String },
    /// An application evaluator, which the evaluation's host runs.
    Host { name: String, evaluator: HostRef },
}

/// What one judge of an evaluation came to.
#[crate::host(record(module = "evals", emit))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JudgeOutcome {
    Judged {
        name: String,
        judgment: Judgment,
    },
    /// The judge raised or the model gave no usable verdict: no judgment, never a pass.
    Failed {
        name: String,
        error: String,
    },
}

/// The intent judge's decision on one message (`ChatMessageAssert.judge`).
#[crate::host(record(module = "evals", emit))]
#[derive(Debug, Clone, PartialEq)]
pub struct IntentJudgment {
    pub success: bool,
    pub reason: String,
    /// The judge request's usage, when the model reported it.
    pub usage: Option<TokenUsage>,
}

/// The aggregates of a set of judgments: pass scores 1, maybe 0.5 and fail 0.
#[crate::host(record(module = "evals", emit))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluationScore {
    /// The mean score; 0 for no judgments.
    pub score: f64,
    pub all_passed: bool,
    pub any_passed: bool,
    /// More than half passed; true for no judgments.
    pub majority_passed: bool,
    /// Nothing failed; maybe is allowed.
    pub none_failed: bool,
}

/// A judge, verdict or judged message the evaluation cannot use.
#[crate::host(error(kind = "ValueError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvalidJudgment(pub String);

/// The host side of one evaluation: runs an application evaluator on the conversation under
/// evaluation, with the evaluation's model.
#[crate::host(proxy(host = "livekit.agents.evals._Evaluation"))]
pub trait EvaluationHost: Send + Sync {
    /// Runs `evaluator` with its judge's span current.
    #[host(await)]
    fn evaluate<'a>(
        &'a self,
        evaluator: &'a HostRef,
        #[host(context)] context: &'a crate::invoke::CallContext,
    ) -> crate::BoxFuture<'a, Result<Judgment, HostError>>;
}

/// Runs judges on one model. A model the host implements is offered `tools`, the host's live
/// judge tools (`judge_tool_schemas`).
#[crate::host(class(name = "_core_JudgeRunner"))]
pub trait JudgeRunner: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(new)]
    fn new(
        env: &Self::Env,
        llm: Option<Arc<dyn Llm>>,
        tools: HostRef,
        host: Arc<dyn EvaluationHost>,
    ) -> Self;

    /// Runs `judges` concurrently on `chat_ctx`, judged against `reference` when given: their
    /// outcomes in judge order. An unknown built-in judge is refused before any runs.
    #[host(await)]
    async fn evaluate(
        &self,
        judges: Vec<JudgeSpec>,
        chat_ctx: Vec<ChatItem>,
        reference: Option<Vec<ChatItem>>,
        #[host(context)] context: crate::invoke::CallContext,
    ) -> Result<Vec<JudgeOutcome>, InvalidJudgment>;

    /// Whether `message` fulfills `intent`. A provider failure raises the provider's error.
    #[host(await)]
    async fn judge_intent(
        &self,
        intent: String,
        message: String,
        #[host(context)] context: crate::invoke::CallContext,
    ) -> Result<IntentJudgment, IntentFailure>;
}

/// Why the intent judge has no decision: the judged message or the model's answer, else the
/// provider's own failure.
#[derive(Debug, thiserror::Error)]
#[crate::host(error(kind = "AssertionError"))]
pub enum IntentFailure {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    #[host(delegate)]
    Provider(#[from] crate::errors::ProviderError),
}

/// The pure eval rules the host's types read.
#[crate::host(functions)]
pub trait Evaluations {
    /// The aggregates of `verdicts`; a verdict other than pass, fail or maybe is refused.
    #[host(function, name = "_core_evaluation_score")]
    fn score(verdicts: Vec<String>) -> Result<EvaluationScore, InvalidJudgment>;

    /// The judge tools' raw schemas, which the host offers a model it implements.
    #[host(function, name = "_core_judge_tool_schemas")]
    fn tool_schemas() -> serde_json::Value;
}

/// Result of evaluating a conversation with a group of judges.
#[crate::host(record(module = "evals", path = "livekit.agents.evals", dataclass))]
#[python(property(
    name = "score",
    ret = "float",
    hook = "evaluation_result_score",
    doc = "Score from 0.0 to 1.0. Pass=1, maybe=0.5, fail=0."
))]
#[python(property(
    name = "all_passed",
    ret = "bool",
    hook = "evaluation_result_all_passed",
    doc = "True if all judgments passed. Maybes count as not passed."
))]
#[python(property(
    name = "any_passed",
    ret = "bool",
    hook = "evaluation_result_any_passed",
    doc = "True if at least one judgment passed."
))]
#[python(property(
    name = "majority_passed",
    ret = "bool",
    hook = "evaluation_result_majority_passed",
    doc = "True if more than half of the judgments passed."
))]
#[python(property(
    name = "none_failed",
    ret = "bool",
    hook = "evaluation_result_none_failed",
    doc = "True if no judgments explicitly failed. Maybes are allowed."
))]
#[python(method(
    name = "_score",
    ret = ".evals.EvaluationScore",
    hook = "evaluation_result_core_score",
    doc = ""
))]
pub struct EvaluationResult {
    /// Individual judgment results keyed by judge name.
    #[host(default_factory = "dict")]
    pub judgments: std::collections::BTreeMap<String, Judgment>,
}
