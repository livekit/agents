//! Conversation evals (testing.md "Conversation evals"): the judges of one evaluation run
//! concurrently on one conversation. Built-in judges make one forced tool call each on the
//! evaluation's model; application evaluators run in the host. Prompts and scoring are
//! [`prompts`], the transcript [`transcript`]: pure support.

pub mod prompts;
pub mod transcript;

#[cfg(test)]
#[path = "../evals_tests.rs"]
mod tests;

use std::sync::Arc;
use std::time::SystemTime;

use futures_util::future::join_all;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::evals::{
    EvaluationHost, EvaluationScore, Evaluations, IntentFailure, IntentJudgment, InvalidJudgment,
    JudgeOutcome, JudgeRunner, JudgeSpec, Judgment,
};
use lk_agents_abi::usage::TokenUsage;
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{
    ApiConnectOptions, ChatContext, ChatItem, ChatMessage, ChatRequest, Clock,
    DEFAULT_API_CONNECT_OPTIONS, HOST_TEARDOWN_TIMEOUT, Llm, LlmChunk, Role, StreamOutput,
    ToolChoice, ToolSpec,
};

use prompts::BuiltIn;

use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::telemetry::decl::GenAiContent;
use lk_agents_abi::telemetry::spans::text::{JudgeEvaluation, JudgeEvaluationEnd};
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::telemetry::{Cx, Tracer};

use crate::telemetry::LlmTrace;

/// A judgment is one long completion over a whole transcript.
const JUDGE_CONN_OPTIONS: ApiConnectOptions =
    ApiConnectOptions { read_timeout: Some(90.0), ..DEFAULT_API_CONNECT_OPTIONS };

/// Judging is batch load after the conversation: it never competes with live traffic.
const INFERENCE_CLASS: &str = "low";

/// The judges of evaluations on one model, and the host that runs the application's.
pub struct Judges {
    llm: Option<Arc<dyn Llm>>,
    /// The host's live judge tools, offered to a model the host implements.
    tools: HostRef,
    host: Arc<dyn EvaluationHost>,
    clock: Arc<dyn Clock>,
    tracer: Tracer,
}

enum Judge {
    BuiltIn(&'static BuiltIn),
    Host { name: String, evaluator: HostRef },
}

impl JudgeRunner for Judges {
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        llm: Option<Arc<dyn Llm>>,
        tools: HostRef,
        host: Arc<dyn EvaluationHost>,
    ) -> Self {
        Judges { llm, tools, host, clock: env.clock.clone(), tracer: env.tracer.clone() }
    }

    async fn evaluate(
        &self,
        judges: Vec<JudgeSpec>,
        chat_ctx: Vec<ChatItem>,
        reference: Option<Vec<ChatItem>>,
        context: CallContext,
    ) -> Result<Vec<JudgeOutcome>, InvalidJudgment> {
        let cx = self.tracer.root().continuing(context.span);
        let judges = judges
            .into_iter()
            .map(|spec| match spec {
                JudgeSpec::BuiltIn { name } => prompts::built_in(&name)
                    .map(Judge::BuiltIn)
                    .ok_or_else(|| InvalidJudgment(format!("no built-in judge {name:?}"))),
                JudgeSpec::Host { name, evaluator } => Ok(Judge::Host { name, evaluator }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reference = reference.as_deref();
        let judged =
            judges.iter().map(|judge| self.judge(&cx, &context, judge, &chat_ctx, reference));
        Ok(join_all(judged).await)
    }

    async fn judge_intent(
        &self,
        intent: String,
        message: String,
        context: CallContext,
    ) -> Result<IntentJudgment, IntentFailure> {
        let cx = self.tracer.root().continuing(context.span);
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
        let (arguments, usage) = self
            .call(
                &cx,
                llm,
                prompts::INTENT_SYSTEM_PROMPT,
                prompt,
                prompts::check_intent_tool(),
                DEFAULT_API_CONNECT_OPTIONS,
            )
            .await?;
        let arguments = arguments
            .filter(|a| !a.is_empty())
            .ok_or_else(|| invalid("LLM did not return any arguments for evaluation."))?;
        let (success, reason) = prompts::parse_intent(&arguments).map_err(|reason| {
            IntentFailure::Invalid(format!("check_intent returned unusable arguments: {reason}"))
        })?;
        Ok(IntentJudgment { success, reason, usage })
    }
}

impl Judges {
    /// One judge in its `judge_evaluation` span.
    async fn judge(
        &self,
        cx: &Cx,
        context: &CallContext,
        judge: &Judge,
        chat_ctx: &[ChatItem],
        reference: Option<&[ChatItem]>,
    ) -> JudgeOutcome {
        let name = match judge {
            Judge::BuiltIn(judge) => judge.name.to_owned(),
            Judge::Host { name, .. } => name.clone(),
        };
        let span = cx.start_span(JudgeEvaluation { name: name.clone() });
        let judged = match judge {
            Judge::BuiltIn(judge) => self.built_in(span.cx(), judge, chat_ctx, reference).await,
            Judge::Host { evaluator, .. } => {
                let context = CallContext { span: span.cx().span(), ..context.clone() };
                let judged = self.host.evaluate(evaluator, &context).await.map_err(|e| e.message);
                judged.and_then(prompts::checked)
            }
        };
        match judged {
            Ok(judgment) => {
                let capture = cx.tracer().capture_content();
                span.end(JudgeEvaluationEnd {
                    verdict: Some(judgment.verdict.clone()),
                    explanation: capture.then(|| GenAiContent(judgment.reasoning.clone())),
                });
                JudgeOutcome::Judged { name, judgment }
            }
            Err(error) => {
                span.fail(&lk_agents_net::telemetry::Failure {
                    kind: "judge_error",
                    message: error.clone(),
                });
                JudgeOutcome::Failed { name, error }
            }
        }
    }

    async fn built_in(
        &self,
        cx: &Cx,
        judge: &BuiltIn,
        chat_ctx: &[ChatItem],
        reference: Option<&[ChatItem]>,
    ) -> Result<Judgment, String> {
        if let Some(judgment) = judge.without_model(chat_ctx) {
            return Ok(judgment);
        }
        let llm = self.llm.as_ref().ok_or_else(|| {
            format!(
                "No LLM provided for judge '{}'. Pass llm to JudgeGroup or to the judge constructor.",
                judge.name
            )
        })?;
        let prompt = judge.prompt(chat_ctx, reference);
        let (arguments, _) = self
            .call(
                cx,
                llm,
                prompts::JUDGE_SYSTEM_PROMPT,
                prompt,
                prompts::submit_verdict_tool(),
                JUDGE_CONN_OPTIONS,
            )
            .await
            .map_err(|e| e.to_string())?;
        let arguments = arguments.ok_or("LLM did not return verdict arguments")?;
        let (verdict, reasoning) = prompts::parse_verdict(&arguments)
            .map_err(|reason| format!("submit_verdict returned unusable arguments: {reason}"))?;
        Ok(Judgment { verdict, reasoning, instructions: judge.criteria.to_owned() })
    }

    /// One completion that must call `tool`: the first call's arguments and the reported usage.
    /// The stream is closed whatever it ended with.
    async fn call(
        &self,
        cx: &Cx,
        llm: &Arc<dyn Llm>,
        system: &str,
        prompt: String,
        tool: ToolSpec,
        conn_options: ApiConnectOptions,
    ) -> Result<(Option<String>, Option<TokenUsage>), ProviderError> {
        // Sent once and never kept: fixed ids and creation times.
        let at = SystemTime::UNIX_EPOCH;
        let mut req = ChatRequest {
            chat_ctx: ChatContext::new(vec![
                ChatMessage::text("lk.judge.system", Role::System, system, at).into(),
                ChatMessage::text("lk.judge.prompt", Role::User, prompt, at).into(),
            ]),
            chat_ctx_host: None,
            tools: vec![tool],
            tools_host: Some(self.tools.clone()),
            tools_host_extra: Vec::new(),
            conn_options,
            parallel_tool_calls: None,
            tool_choice: Some(ToolChoice::Required),
            extra: Some(prompts::request_extra(&llm.model())),
            inference_class: Some(INFERENCE_CLASS.to_owned()),
            trace: None,
        };
        let mut trace = LlmTrace::start(cx, llm.as_ref(), &mut req);
        let attempt = trace.attempt();
        let read = match llm.chat(&req).await {
            Ok(mut stream) => {
                let read = first_call(stream.as_mut(), &mut trace).await;
                let _ = stream.close(self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
                read
            }
            Err(error) => Err(error),
        };
        trace.end_attempt(attempt, read.as_ref().err());
        match &read {
            Ok(_) => trace.end(false, None),
            Err(error) => trace.fail(error),
        }
        read
    }
}

async fn first_call(
    stream: &mut dyn StreamOutput<LlmChunk>,
    trace: &mut LlmTrace,
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
    fn score(verdicts: Vec<String>) -> Result<EvaluationScore, InvalidJudgment> {
        prompts::score(&verdicts)
    }

    fn tool_schemas() -> serde_json::Value {
        [prompts::submit_verdict_tool(), prompts::check_intent_tool()]
            .into_iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.json_schema,
                })
            })
            .collect()
    }
}
