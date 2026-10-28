//! Conversation evals (testing.md "Conversation evals"): the judges of one evaluation run
//! concurrently on one conversation. Built-in judges make one forced tool call each on the
//! evaluation's model; application evaluators run in the host. Prompts and scoring are
//! [`prompts`], the transcript [`transcript`]: pure support.

pub mod prompts;
pub mod transcript;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use futures_util::future::join_all;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::evals::{
    EvaluationHost, EvaluationScore, Evaluations, IntentFailure, IntentJudgment, InvalidJudgment,
    JudgeOutcome, JudgeRunner, JudgeSpec, Judgment,
};
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{
    ApiConnectOptions, ChatContext, ChatItem, ChatMessage, ChatRequest, Clock, CompletionUsage,
    DEFAULT_API_CONNECT_OPTIONS, HOST_TEARDOWN_TIMEOUT, Llm, LlmChunk, Role, StreamSource,
    ToolChoice, ToolSpec,
};

use prompts::BuiltIn;

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
}

enum Judge {
    BuiltIn(&'static BuiltIn),
    Host { name: String, evaluator: HostRef },
}

#[async_trait]
impl JudgeRunner for Judges {
    fn new(llm: Option<Arc<dyn Llm>>, tools: HostRef, host: Arc<dyn EvaluationHost>) -> Self {
        Judges { llm, tools, host, clock: crate::process::host_clock() }
    }

    async fn evaluate(
        &self,
        judges: Vec<JudgeSpec>,
        chat_ctx: Vec<ChatItem>,
        reference: Option<Vec<ChatItem>>,
    ) -> Result<Vec<JudgeOutcome>, InvalidJudgment> {
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
        Ok(join_all(judges.iter().map(|judge| self.judge(judge, &chat_ctx, reference))).await)
    }

    async fn judge_intent(
        &self,
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
        let (arguments, usage) = self
            .call(
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
    async fn judge(
        &self,
        judge: &Judge,
        chat_ctx: &[ChatItem],
        reference: Option<&[ChatItem]>,
    ) -> JudgeOutcome {
        let (name, judged) = match judge {
            Judge::BuiltIn(judge) => {
                (judge.name.to_owned(), self.built_in(judge, chat_ctx, reference).await)
            }
            Judge::Host { name, evaluator } => {
                let judged = self.host.evaluate(evaluator).await.map_err(|e| e.message);
                (name.clone(), judged.and_then(prompts::checked))
            }
        };
        match judged {
            Ok(judgment) => JudgeOutcome::Judged { name, judgment },
            Err(error) => JudgeOutcome::Failed { name, error },
        }
    }

    async fn built_in(
        &self,
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
        llm: &Arc<dyn Llm>,
        system: &str,
        prompt: String,
        tool: ToolSpec,
        conn_options: ApiConnectOptions,
    ) -> Result<(Option<String>, Option<CompletionUsage>), ProviderError> {
        // Sent once and never kept: fixed ids and creation times.
        let at = SystemTime::UNIX_EPOCH;
        let req = ChatRequest {
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
        };
        let mut stream = llm.chat(&req).await?;
        let read = first_call(stream.as_mut()).await;
        let _ = stream.close(self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT)).await;
        read
    }
}

async fn first_call(
    stream: &mut dyn StreamSource<LlmChunk>,
) -> Result<(Option<String>, Option<CompletionUsage>), ProviderError> {
    let (mut arguments, mut usage) = (None, None);
    while let Some(chunk) = stream.next().await? {
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
        if arguments.is_none()
            && let Some(call) = chunk.delta.and_then(|d| d.tool_calls.into_iter().next())
        {
            arguments = Some(call.arguments);
        }
    }
    Ok((arguments, usage))
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
