//! The judges and their shared LLM call. Six of the reference's eight judges (`judge.py:396`) are
//! one `_LLMJudge` with different strings: rows in `prompts`.

use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use lk_agents_abi::{ChatRequest, ToolChoice, ToolFlags, ToolSpec};
use lk_agents_abi::{ApiConnectOptions, ChatContext, ChatMessage, Extra, LlmChunk, Role};
use serde_json::{Value, json};

use crate::evals::error::JudgeError;
use crate::evals::evaluator::{EvaluateRequest, Evaluator, JudgeLlm, reference_transcript};
use crate::evals::judgment::{JudgmentResult, Verdict};
use crate::evals::prompts;
use crate::evals::transcript::{format_chat_ctx, has_handoffs, latest_instructions};

/// `judge.py:11`: a judgment is one long completion over a whole transcript.
pub const JUDGE_CONN_OPTIONS: ApiConnectOptions =
    ApiConnectOptions { max_retry: 3, retry_interval: 2.0, timeout: 90.0 };

/// The inference class every judgment must run under (`_judge_chat_kwargs`, `judge.py:19`).
// Not applied here: `inference_class` is a per-instance header in Rust, so whoever constructs the
// judge LLM must set it (see `JudgeGroup`).
#[crate::host(export(pid = false, name = "judge_inference_class"))]
pub const JUDGE_INFERENCE_CLASS: &str = "low";

/// Fixed (and `created_at` the epoch): this context is sent once and never enters a history.
const SYSTEM_MSG_ID: &str = "lk.judge.system";
const PROMPT_MSG_ID: &str = "lk.judge.prompt";

/// `judge.py:110`. pydantic v2's schema: keys sorted, `properties` in declaration order.
pub fn submit_verdict_tool() -> ToolSpec {
    ToolSpec {
        name: "submit_verdict".to_string(),
        description: prompts::SUBMIT_VERDICT_DESCRIPTION.to_string(),
        json_schema: json!({
            "properties": {
                "verdict": {
                    "description": prompts::SUBMIT_VERDICT_ARG_VERDICT,
                    "enum": ["pass", "fail", "maybe"],
                    "title": "Verdict",
                    "type": "string",
                },
                "reasoning": {
                    "description": prompts::SUBMIT_VERDICT_ARG_REASONING,
                    "title": "Reasoning",
                    "type": "string",
                },
            },
            "required": ["verdict", "reasoning"],
            "title": "SubmitVerdictArgs",
            "type": "object",
        }),
        flags: ToolFlags::default(),
    }
}

pub fn judgment_chat_ctx(prompt: &str) -> ChatContext {
    let at = SystemTime::UNIX_EPOCH;
    ChatContext::new(vec![
        ChatMessage::text(SYSTEM_MSG_ID, Role::System, prompts::JUDGE_SYSTEM_PROMPT, at).into(),
        ChatMessage::text(PROMPT_MSG_ID, Role::User, prompt, at).into(),
    ])
}

/// `judge.py:136`: always given, sometimes empty. No `inference_class` (`test_judge.py` pins
/// `{"temperature": 0.0}` exactly).
pub fn judgment_extra(llm: &JudgeLlm) -> Extra {
    let mut extra = Extra::new();
    if llm.supports_temperature() {
        extra.insert("temperature".to_string(), json!(0.0));
    }
    if let Some(effort) = llm.reasoning_effort() {
        extra.insert("reasoning_effort".to_string(), json!(effort));
    }
    extra
}

pub fn parse_verdict_arguments(arguments: &str) -> Result<JudgmentResult, JudgeError> {
    let value: Value = serde_json::from_str(arguments)
        .map_err(|e| JudgeError::BadArguments { reason: format!("not JSON: {e}") })?;
    let obj = value
        .as_object()
        .ok_or_else(|| JudgeError::BadArguments { reason: "expected a JSON object".to_string() })?;

    let verdict = match obj.get("verdict") {
        Some(Value::String(s)) => {
            Verdict::from_literal(s).ok_or_else(|| JudgeError::BadArguments {
                reason: format!("verdict must be 'pass', 'fail' or 'maybe', got {s:?}"),
            })?
        }
        Some(other) => {
            return Err(JudgeError::BadArguments {
                reason: format!("verdict must be a string, got {other}"),
            });
        }
        None => {
            return Err(JudgeError::BadArguments { reason: "missing 'verdict'".to_string() });
        }
    };

    let reasoning = match obj.get("reasoning") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(JudgeError::BadArguments {
                reason: format!("reasoning must be a string, got {other}"),
            });
        }
        None => {
            return Err(JudgeError::BadArguments { reason: "missing 'reasoning'".to_string() });
        }
    };

    Ok(JudgmentResult::new(verdict, reasoning))
}

/// `judge.py:110`. `instructions` is left `""` for each judge to fill in.
pub async fn evaluate_with_llm(llm: &JudgeLlm, prompt: &str) -> Result<JudgmentResult, JudgeError> {
    let req = ChatRequest {
        chat_ctx: judgment_chat_ctx(prompt),
        chat_ctx_host: None,
        tools: vec![submit_verdict_tool()],
        // `None` for a native provider; a host-backed one needs the live callable.
        tools_host: llm.submit_verdict_host().cloned(),
        tools_host_extra: Vec::new(),
        conn_options: JUDGE_CONN_OPTIONS,
        parallel_tool_calls: None,
        tool_choice: Some(ToolChoice::Required),
        extra: Some(judgment_extra(llm)),
        inference_class: llm.inference_class().map(str::to_owned),
    };

    let mut stream = llm.llm().chat(&req).await?;
    let first_call = collect_first_tool_call(stream.as_mut()).await;
    // `collect()` is `async with self`: close the stream, or a host-backed one leaks its task.
    let _closed: lk_agents_abi::Closed = stream.close(llm.teardown_deadline()).await;

    let arguments = first_call?.ok_or(JudgeError::NoVerdict)?;
    parse_verdict_arguments(&arguments)
}

// --- `ChatMessageAssert.judge`: the test DSL's intent judge ------------------------------------

/// `run_result.py`, verbatim.
pub const INTENT_SYSTEM_PROMPT: &str = "You are a test evaluator for conversational agents.\n\
You will be shown a message and a target intent. Determine whether the message accomplishes the intent.\n\
Only respond by calling the `check_intent(success: bool, reason: str)` function with your final judgment.\n\
Be strict: if the message does not clearly fulfill the intent, return `success = False` and explain why.";

pub fn check_intent_tool() -> ToolSpec {
    ToolSpec {
        name: "check_intent".to_string(),
        description: "Determines whether the message correctly fulfills the given intent.".to_string(),
        json_schema: json!({
            "properties": {
                "success": {
                    "description": "Whether the message satisfies the intent.",
                    "title": "Success",
                    "type": "boolean",
                },
                "reason": {
                    "description": "A concise explanation justifying the result.",
                    "title": "Reason",
                    "type": "string",
                },
            },
            "required": ["success", "reason"],
            "title": "CheckIntentArgs",
            "type": "object",
        }),
        flags: ToolFlags::default(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IntentVerdict {
    pub success: bool,
    pub reason: String,
    /// For the span's `gen_ai.usage.*` attributes.
    pub usage: Option<lk_agents_abi::CompletionUsage>,
}

/// Matches the reference's request; the last chunk's call wins.
pub async fn judge_intent(
    llm: &JudgeLlm,
    intent: &str,
    message: &str,
) -> Result<IntentVerdict, JudgeError> {
    if message.is_empty() {
        return Err(JudgeError::EmptyMessage);
    }
    if intent.is_empty() {
        return Err(JudgeError::EmptyIntent);
    }
    let at = SystemTime::UNIX_EPOCH;
    let user = format!(
        "Check if the following message fulfills the given intent.\n\nIntent:\n{intent}\n\nMessage:\n{message}"
    );
    let req = ChatRequest {
        chat_ctx: ChatContext::new(vec![
            ChatMessage::text(SYSTEM_MSG_ID, Role::System, INTENT_SYSTEM_PROMPT, at).into(),
            ChatMessage::text(PROMPT_MSG_ID, Role::User, user, at).into(),
        ]),
        chat_ctx_host: None,
        tools: vec![check_intent_tool()],
        tools_host: llm.submit_verdict_host().cloned(),
        tools_host_extra: Vec::new(),
        conn_options: ApiConnectOptions::default(),
        parallel_tool_calls: None,
        tool_choice: Some(ToolChoice::Required),
        extra: Some(judgment_extra(llm)),
        inference_class: llm.inference_class().map(str::to_owned),
    };
    let mut stream = llm.llm().chat(&req).await?;
    let collected = collect_last_call_and_usage(stream.as_mut()).await;
    let _closed: lk_agents_abi::Closed = stream.close(llm.teardown_deadline()).await;
    let (arguments, usage) = collected?;
    let arguments = arguments.ok_or(JudgeError::NoArguments)?;
    let (success, reason) = parse_intent_arguments(&arguments)?;
    Ok(IntentVerdict { success, reason, usage })
}

async fn collect_last_call_and_usage(
    stream: &mut dyn lk_agents_abi::StreamSource<LlmChunk>,
) -> Result<(Option<String>, Option<lk_agents_abi::CompletionUsage>), JudgeError> {
    let (mut arguments, mut usage) = (None, None);
    while let Some(chunk) = stream.next().await? {
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
        let Some(delta) = chunk.delta else { continue };
        if let Some(call) = delta.tool_calls.into_iter().next() {
            arguments = Some(call.arguments);
        }
    }
    Ok((arguments, usage))
}

/// Repaired JSON, then pydantic's lax `bool`.
pub fn parse_intent_arguments(arguments: &str) -> Result<(bool, String), JudgeError> {
    let bad = |reason: String| JudgeError::BadIntentArguments { reason };
    let value = crate::llm::json_repair::repair(arguments).map_err(bad)?;
    let obj = value.as_object().ok_or_else(|| bad("expected a JSON object".to_string()))?;
    let success = match obj.get("success") {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) if n.as_i64() == Some(1) => true,
        Some(Value::Number(n)) if n.as_i64() == Some(0) => false,
        Some(Value::String(s)) => match s.to_ascii_lowercase().as_str() {
            "t" | "true" | "on" | "y" | "yes" | "1" => true,
            "f" | "false" | "off" | "n" | "no" | "0" => false,
            _ => return Err(bad(format!("success is not a boolean: {s:?}"))),
        },
        Some(other) => return Err(bad(format!("success is not a boolean: {other}"))),
        None => return Err(bad("missing 'success'".to_string())),
    };
    let reason = match obj.get("reason") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => return Err(bad(format!("reason must be a string, got {other}"))),
        None => return Err(bad("missing 'reason'".to_string())),
    };
    Ok((success, reason))
}

/// Narrowed to the tool calls (which arrive whole).
async fn collect_first_tool_call(
    stream: &mut dyn lk_agents_abi::StreamSource<LlmChunk>,
) -> Result<Option<String>, JudgeError> {
    let mut first: Option<String> = None;
    while let Some(chunk) = stream.next().await? {
        let LlmChunk { delta: Some(delta), .. } = chunk else { continue };
        if first.is_none()
            && let Some(call) = delta.tool_calls.into_iter().next()
        {
            first = Some(call.arguments);
        }
    }
    Ok(first)
}

/// `effective_llm = llm or self._llm`.
fn effective_llm<'a>(
    name: &str,
    own: Option<&'a Arc<JudgeLlm>>,
    req: &'a EvaluateRequest,
) -> Result<&'a Arc<JudgeLlm>, JudgeError> {
    req.llm.as_ref().or(own).ok_or_else(|| JudgeError::NoLlm { name: name.to_string() })
}

// --- `_LLMJudge` -------------------------------------------------------------------------------

/// `judge.py:207`. Public so a custom prompt judge can be built.
pub struct LlmJudge {
    llm: Option<Arc<JudgeLlm>>,
    instructions: String,
    name: String,
}

impl LlmJudge {
    pub fn new(
        name: impl Into<String>,
        instructions: impl Into<String>,
        llm: Option<Arc<JudgeLlm>>,
    ) -> Self {
        LlmJudge { llm, instructions: instructions.into(), name: name.into() }
    }

    pub fn built_in(prompt: prompts::JudgePrompt, llm: Option<Arc<JudgeLlm>>) -> Self {
        LlmJudge::new(prompt.name, prompt.instructions, llm)
    }

    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    /// `judge.py:229`, split out so it can be asserted on without an LLM.
    pub fn prompt(&self, req: &EvaluateRequest) -> String {
        let mut parts = vec![
            format!("Criteria: {}", self.instructions),
            String::new(),
            format!("Conversation:\n{}", format_chat_ctx(&req.chat_ctx)),
        ];
        if let Some(reference) = &req.reference {
            parts.push(String::new());
            parts.push(format!("Reference:\n{}", reference_transcript(reference)));
        }
        parts.push(String::new());
        parts.push(prompts::EVALUATE_TRAILER.to_string());
        parts.join("\n")
    }
}

#[async_trait]
impl Evaluator for LlmJudge {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn evaluate(&self, req: &EvaluateRequest) -> Result<JudgmentResult, JudgeError> {
        let llm = effective_llm(&self.name, self.llm.as_ref(), req)?;
        let result = evaluate_with_llm(llm, &self.prompt(req)).await?;
        Ok(result.with_instructions(&self.instructions))
    }
}

pub fn accuracy_judge(llm: Option<Arc<JudgeLlm>>) -> LlmJudge {
    LlmJudge::built_in(prompts::ACCURACY, llm)
}

pub fn tool_use_judge(llm: Option<Arc<JudgeLlm>>) -> LlmJudge {
    LlmJudge::built_in(prompts::TOOL_USE, llm)
}

pub fn safety_judge(llm: Option<Arc<JudgeLlm>>) -> LlmJudge {
    LlmJudge::built_in(prompts::SAFETY, llm)
}

pub fn relevancy_judge(llm: Option<Arc<JudgeLlm>>) -> LlmJudge {
    LlmJudge::built_in(prompts::RELEVANCY, llm)
}

pub fn coherence_judge(llm: Option<Arc<JudgeLlm>>) -> LlmJudge {
    LlmJudge::built_in(prompts::COHERENCE, llm)
}

pub fn conciseness_judge(llm: Option<Arc<JudgeLlm>>) -> LlmJudge {
    LlmJudge::built_in(prompts::CONCISENESS, llm)
}

// --- `_TaskCompletionJudge` --------------------------------------------------------------------

/// `judge.py:254`: judged against the latest `agent_config_update` instructions.
pub struct TaskCompletionJudge {
    llm: Option<Arc<JudgeLlm>>,
}

impl TaskCompletionJudge {
    pub fn new(llm: Option<Arc<JudgeLlm>>) -> Self {
        TaskCompletionJudge { llm }
    }

    /// `judge.py:290`. Also whether instructions were found (the reference warns when not).
    pub fn prompt(&self, req: &EvaluateRequest) -> (String, bool) {
        let instructions = latest_instructions(&req.chat_ctx);
        let mut parts = vec![prompts::TASK_COMPLETION_CRITERIA.to_string(), String::new()];
        if let Some(instructions) = instructions {
            parts.push(format!("Agent Instructions:\n{instructions}"));
            parts.push(String::new());
        }
        parts.push(format!("Conversation:\n{}", format_chat_ctx(&req.chat_ctx)));
        if let Some(reference) = &req.reference {
            parts.push(String::new());
            parts.push(format!("Reference:\n{}", reference_transcript(reference)));
        }
        (parts.join("\n"), instructions.is_some())
    }
}

#[async_trait]
impl Evaluator for TaskCompletionJudge {
    fn name(&self) -> String {
        "task_completion".to_string()
    }

    async fn evaluate(&self, req: &EvaluateRequest) -> Result<JudgmentResult, JudgeError> {
        let llm = effective_llm("task_completion", self.llm.as_ref(), req)?;
        let (prompt, _had_instructions) = self.prompt(req);
        let result = evaluate_with_llm(llm, &prompt).await?;
        Ok(result.with_instructions(prompts::TASK_COMPLETION_CRITERIA))
    }
}

pub fn task_completion_judge(llm: Option<Arc<JudgeLlm>>) -> TaskCompletionJudge {
    TaskCompletionJudge::new(llm)
}

// --- `_HandoffJudge` ---------------------------------------------------------------------------

/// `judge.py:312`. With no real handoff it passes without calling (or needing) an LLM.
pub struct HandoffJudge {
    llm: Option<Arc<JudgeLlm>>,
}

impl HandoffJudge {
    pub fn new(llm: Option<Arc<JudgeLlm>>) -> Self {
        HandoffJudge { llm }
    }

    /// `judge.py:358`. Does not embed the agent's instructions.
    pub fn prompt(&self, req: &EvaluateRequest) -> String {
        let mut parts = vec![
            prompts::HANDOFF_CRITERIA.to_string(),
            String::new(),
            format!("Conversation:\n{}", format_chat_ctx(&req.chat_ctx)),
        ];
        if let Some(reference) = &req.reference {
            parts.push(String::new());
            parts.push(format!("Reference:\n{}", reference_transcript(reference)));
        }
        parts.join("\n")
    }
}

#[async_trait]
impl Evaluator for HandoffJudge {
    fn name(&self) -> String {
        "handoff".to_string()
    }

    async fn evaluate(&self, req: &EvaluateRequest) -> Result<JudgmentResult, JudgeError> {
        if !has_handoffs(&req.chat_ctx) {
            // `instructions` stays `""`, as in the reference's short circuit.
            return Ok(JudgmentResult::new(Verdict::Pass, prompts::HANDOFF_NO_HANDOFFS_REASONING));
        }
        let llm = effective_llm("handoff", self.llm.as_ref(), req)?;
        let result = evaluate_with_llm(llm, &self.prompt(req)).await?;
        Ok(result.with_instructions(prompts::HANDOFF_CRITERIA))
    }
}

pub fn handoff_judge(llm: Option<Arc<JudgeLlm>>) -> HandoffJudge {
    HandoffJudge::new(llm)
}

#[cfg(feature = "python")]
pub use python::PyJudge;

#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::*;
    use crate::evals::evaluator::PyJudgeLlm;
    use crate::evals::judgment::PyJudgmentResult;
    use crate::python::check_pid;

    /// One of the eight shipped judges.
    #[pyclass(name = "Judge", frozen, module = "livekit.agents._core")]
    pub struct PyJudge {
        pub(crate) inner: Arc<dyn Evaluator>,
    }

    #[pymethods]
    impl PyJudge {
        #[getter]
        fn name(&self) -> PyResult<String> {
            check_pid()?;
            Ok(self.inner.name())
        }
    }

    crate::register_host!(class PyJudge);

    /// The group's LLM, when set, wins over a per-judge one (the reference's precedence).
    macro_rules! judge_factory {
        ($py_name:ident, $core:path, $doc:literal) => {
            #[doc = $doc]
            #[pyfunction]
            #[pyo3(signature = (llm=None))]
            pub fn $py_name(llm: Option<&PyJudgeLlm>) -> PyResult<PyJudge> {
                check_pid()?;
                let llm = llm.map(|l| l.inner.clone());
                Ok(PyJudge { inner: Arc::new($core(llm)) })
            }
            crate::register_host!(fn $py_name);
        };
    }

    judge_factory!(accuracy, accuracy_judge, "`accuracy`: is what the agent said true?");
    judge_factory!(tool_use, tool_use_judge, "`tool_use`: were the right tools called, with the right arguments?");
    judge_factory!(safety, safety_judge, "`safety`: did the agent stay inside its guardrails?");
    judge_factory!(relevancy, relevancy_judge, "`relevancy`: did the agent answer what was asked?");
    judge_factory!(coherence, coherence_judge, "`coherence`: does the conversation hold together?");
    judge_factory!(conciseness, conciseness_judge, "`conciseness`: did the agent say it without padding?");
    judge_factory!(
        task_completion,
        task_completion_judge,
        "`task_completion`: judged against the *instructions* in the context, so it needs one."
    );
    judge_factory!(
        handoff,
        handoff_judge,
        "`handoff`: were the transfers between agents correct? Passes vacuously when there were none."
    );

    /// `ChatMessageAssert.judge(llm, intent=)`: `(success, reason, usage)`, `usage` being
    /// `(prompt_tokens, completion_tokens, prompt_cached_tokens)` or `None`. Provider failures raise
    /// the host's exception; others raise `ValueError` with the reference's text.
    #[pyfunction]
    pub fn judge_intent<'py>(py: Python<'py>, judge: &PyJudgeLlm, intent: String, message: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let judge = Arc::clone(&judge.inner);
        crate::python::future_into_py(py, async move {
            match super::judge_intent(&judge, &intent, &message).await {
                Ok(v) => Ok((
                    v.success,
                    v.reason,
                    v.usage.map(|u| (u.prompt_tokens, u.completion_tokens, u.prompt_cached_tokens)),
                )),
                Err(e) => Err(judge_error_to_py(e)),
            }
        })
    }

    crate::register_host!(fn self::judge_intent);

    /// `_evaluate_with_llm(llm, prompt)`: one forced-tool-call judgment.
    #[pyfunction]
    pub fn evaluate_with_llm<'py>(py: Python<'py>, judge: &PyJudgeLlm, prompt: String) -> PyResult<Bound<'py, PyAny>> {
        check_pid()?;
        let judge = Arc::clone(&judge.inner);
        crate::python::future_into_py(py, async move {
            match super::evaluate_with_llm(&judge, &prompt).await {
                Ok(r) => Ok(PyJudgmentResult::of(&r)),
                Err(e) => Err(judge_error_to_py(e)),
            }
        })
    }

    crate::register_host!(fn self::evaluate_with_llm);

    /// A provider's failure raises the host's own exception, any other a `ValueError`.
    fn judge_error_to_py(e: JudgeError) -> PyErr {
        match e {
            JudgeError::Provider(e) => Python::attach(|py| match crate::python::to_py_exception(py, &e) {
                Ok(exc) => PyErr::from_value(exc.into_bound(py)),
                Err(err) => err,
            }),
            other => PyValueError::new_err(other.to_string()),
        }
    }
}
