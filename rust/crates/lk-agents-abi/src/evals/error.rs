use crate::errors::ProviderError;

/// A value that travels: the group returns failures rather than dropping them.
#[derive(Debug, thiserror::Error)]
pub enum JudgeError {
    /// `judge.py:224` / `:277` / `:355`, verbatim: no LLM from the judge or the group.
    #[error(
        "No LLM provided for judge '{name}'. Pass llm to JudgeGroup or to the judge constructor."
    )]
    NoLlm { name: String },

    /// `judge.py:154`, verbatim: the model answered with prose despite `tool_choice="required"`.
    #[error("LLM did not return verdict arguments")]
    NoVerdict,

    /// Strict JSON on purpose (the reference would try recovery first).
    #[error("submit_verdict returned unusable arguments: {reason}")]
    BadArguments { reason: String },

    /// `ChatMessageAssert.judge`'s preconditions and missing verdict, verbatim (`run_result.py`).
    #[error("The chat message is empty.")]
    EmptyMessage,
    #[error("Intent is required to judge the message.")]
    EmptyIntent,
    #[error("LLM did not return any arguments for evaluation.")]
    NoArguments,
    #[error("check_intent returned unusable arguments: {reason}")]
    BadIntentArguments { reason: String },

    #[error(transparent)]
    Provider(#[from] ProviderError),

    /// For adapters with no host exception to capture.
    #[error("judge '{name}' failed: {message}")]
    Evaluator { name: String, message: String },
}

impl JudgeError {
    /// `provider_host` carries a host exception (`exc_info=`), `provider_local` does not.
    pub fn kind(&self) -> &'static str {
        match self {
            JudgeError::NoLlm { .. } => "no_llm",
            JudgeError::NoVerdict => "no_verdict",
            JudgeError::BadArguments { .. } | JudgeError::BadIntentArguments { .. } => {
                "bad_arguments"
            }
            JudgeError::EmptyMessage | JudgeError::EmptyIntent => "precondition",
            JudgeError::NoArguments => "no_verdict",
            JudgeError::Provider(ProviderError::Host { .. }) => "provider_host",
            JudgeError::Provider(ProviderError::Local(_)) => "provider_local",
            JudgeError::Evaluator { .. } => "evaluator",
        }
    }
}
