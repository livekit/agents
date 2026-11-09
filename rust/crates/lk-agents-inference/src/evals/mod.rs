//! Conversation evals (testing.md "Conversation evals"): the application's judges of one
//! evaluation run concurrently in the host, each in its span; the intent judge makes one forced
//! tool call on the evaluation's model. Its prompt and the scoring are [`prompts`]: pure support.

pub mod prompts;

#[cfg(test)]
#[path = "../evals_tests.rs"]
mod tests;

use std::sync::Arc;
use std::time::SystemTime;

use futures_util::future::join_all;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::evals::{
    EvaluationHost, EvaluationResult, EvaluationScore, Evaluations, IntentFailure, IntentJudgment,
    JudgeOutcome, JudgeRunner, JudgeSpec, JudgeVerdict, Judgment,
};
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::{
    ChatContext, ChatMessage, ChatRequest, Clock, DEFAULT_API_CONNECT_OPTIONS,
    HOST_TEARDOWN_TIMEOUT, Llm, LlmChunk, Role, StreamOutput, ToolChoice,
};

use crate::telemetry;
use lk_agents_abi::invoke::CallContext;
use lk_agents_net::env::ProcessEnv;

/// Judging is batch load after the conversation: it never competes with live traffic.
const INFERENCE_CLASS: &str = "low";

/// The judges of evaluations on one model, and the host that runs the application's.
pub struct Judges {
    llm: Option<Arc<dyn Llm>>,
    host: Arc<dyn EvaluationHost>,
    clock: Arc<dyn Clock>,
}

impl JudgeRunner for Judges {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv, llm: Option<Arc<dyn Llm>>, host: Arc<dyn EvaluationHost>) -> Self {
        Judges { llm, host, clock: env.clock.clone() }
    }

    async fn evaluate(
        &self,
        judges: Vec<JudgeSpec>,
        context: CallContext,
        cx: telemetry::Cx,
    ) -> Vec<JudgeOutcome> {
        join_all(judges.into_iter().map(|judge| self.judge(&cx, &context, judge))).await
    }

    async fn judge_intent(
        &self,
        intent: String,
        message: String,
        cx: telemetry::Cx,
    ) -> Result<IntentJudgment, IntentFailure> {
        let span = telemetry::start(&cx, telemetry::JudgeEvaluation { name: "intent".into() });
        let judged = self.intent(span.cx(), intent, message).await;
        match &judged {
            Ok(judgment) => {
                telemetry::end(
                    span,
                    telemetry::JudgeEvaluationEnd {
                        verdict: Some(if judgment.success {
                            JudgeVerdict::Pass
                        } else {
                            JudgeVerdict::Fail
                        }),
                        explanation: cx
                            .capture_content()
                            .then(|| telemetry::GenAiContent(judgment.reason.clone())),
                    },
                );
            }
            Err(error) => {
                telemetry::end(
                    span,
                    telemetry::Failure {
                        kind: telemetry::ErrorType::JudgeError,
                        message: error.to_string(),
                    },
                );
            }
        }
        judged
    }
}

impl Judges {
    /// Whether `message` fulfills `intent`, judged in `cx`'s `judge_evaluation`.
    async fn intent(
        &self,
        cx: &telemetry::Cx,
        intent: String,
        message: String,
    ) -> Result<IntentJudgment, IntentFailure> {
        let invalid = |reason: &str| IntentFailure::Invalid(reason.to_owned());
        if message.is_empty() {
            return Err(invalid("The chat message is empty."));
        }
        if intent.is_empty() {
            return Err(invalid("Intent is required to judge the message."));
        }
        let llm =
            self.llm.as_ref().ok_or_else(|| invalid("No LLM provided to judge the message."))?;
        let prompt = prompts::intent_prompt(&intent, &message);
        let (arguments, usage) = self.call(cx, llm, prompts::INTENT_SYSTEM_PROMPT, prompt).await?;
        let arguments = arguments
            .filter(|a| !a.is_empty())
            .ok_or_else(|| invalid("LLM did not return any arguments for evaluation."))?;
        let (success, reason) = prompts::parse_intent(&arguments).map_err(|reason| {
            IntentFailure::Invalid(format!("check_intent returned unusable arguments: {reason}"))
        })?;
        Ok(IntentJudgment { success, reason, usage })
    }

    /// One judge in its `judge_evaluation` span.
    async fn judge(
        &self,
        cx: &telemetry::Cx,
        context: &CallContext,
        JudgeSpec { name, evaluator }: JudgeSpec,
    ) -> JudgeOutcome {
        let span = telemetry::start(cx, telemetry::JudgeEvaluation { name: name.clone() });
        match self.host.evaluate(span.cx(), &evaluator, context).await {
            Ok(judgment) => {
                let capture = cx.capture_content();
                telemetry::end(
                    span,
                    telemetry::JudgeEvaluationEnd {
                        verdict: Some(judgment.verdict),
                        explanation: capture
                            .then(|| telemetry::GenAiContent(judgment.reasoning.clone())),
                    },
                );
                JudgeOutcome::Judged { name, judgment }
            }
            Err(error) => {
                telemetry::end(
                    span,
                    telemetry::Failure {
                        kind: telemetry::ErrorType::JudgeError,
                        message: error.message.clone(),
                    },
                );
                JudgeOutcome::Failed { name, error: error.message }
            }
        }
    }

    /// One completion that must call `check_intent`: the first call's arguments and the reported
    /// usage. The stream is closed whatever it ended with.
    async fn call(
        &self,
        cx: &telemetry::Cx,
        llm: &Arc<dyn Llm>,
        system: &str,
        prompt: String,
    ) -> Result<(Option<String>, Option<TokenUsage>), ProviderError> {
        // Sent once and never kept: fixed ids and creation times.
        let at = SystemTime::UNIX_EPOCH;
        let req = ChatRequest {
            chat_ctx: ChatContext::new(vec![
                ChatMessage::text("lk.judge.system", Role::System, system, at).into(),
                ChatMessage::text("lk.judge.prompt", Role::User, prompt, at).into(),
            ]),
            chat_ctx_host: None,
            // a host model gets a placeholder for it from the request's function schemas
            tools: vec![prompts::check_intent_tool()],
            tools_host: None,
            tools_host_extra: Vec::new(),
            conn_options: DEFAULT_API_CONNECT_OPTIONS,
            parallel_tool_calls: None,
            tool_choice: Some(ToolChoice::Required),
            extra: Some(prompts::request_extra(&llm.model())),
            inference_class: Some(INFERENCE_CLASS.to_owned()),
        };
        let mut trace = telemetry::LlmTrace::start(cx, llm.as_ref(), &req);
        let attempt = trace.attempt();
        let read = match crate::llm::chat(llm.clone(), trace.cx(), &req, self.clock.clone(), None)
            .await
        {
            Ok(mut stream) => {
                let read = first_call(stream.as_mut(), &mut trace).await;
                let _ = stream.close(self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
                read
            }
            Err(error) => Err(error),
        };
        trace.end_attempt(attempt, read.as_ref().err());
        if let Err(error) = &read {
            trace.fail(error);
        }
        trace.end(Default::default());
        read
    }
}

async fn first_call(
    stream: &mut dyn StreamOutput<LlmChunk>,
    trace: &mut telemetry::LlmTrace,
) -> Result<(Option<String>, Option<TokenUsage>), ProviderError> {
    let (mut arguments, mut usage) = (None, None);
    while let Some(chunk) = stream.next().await? {
        trace.observe(&chunk);
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
        if arguments.is_none()
            && let Some(call) = chunk.delta.and_then(|d| d.tool_calls.into_iter().next())
        {
            arguments = Some(call.arguments);
        }
    }
    Ok((arguments, usage.map(|usage| *usage)))
}

/// The pure rules the host's evaluation types read.
pub struct Rules;

impl Evaluations for Rules {
    fn score(result: EvaluationResult) -> EvaluationScore {
        let verdicts: Vec<JudgeVerdict> =
            result.judgments.into_values().map(|j| j.verdict).collect();
        prompts::score(&verdicts)
    }

    fn judgment_passed(judgment: Judgment) -> bool {
        judgment.verdict == JudgeVerdict::Pass
    }

    fn judgment_failed(judgment: Judgment) -> bool {
        judgment.verdict == JudgeVerdict::Fail
    }

    fn judgment_uncertain(judgment: Judgment) -> bool {
        judgment.verdict == JudgeVerdict::Maybe
    }
}
