//! AMD's classifier and menu legs: one model call each, rendered as Python's `_inference.py`
//! renders it, and its outcome mapped onto the machine's inputs.
//!
//! An LLM answers through a `record_result` tool with `tool_choice="required"` and must call it
//! exactly once. A decision model answers one `category` choice. A provider failure, an invalid
//! answer or a timeout is an `inference_error` prediction; any other failure ends the run.

use super::prompts::{CLASSIFY_PROMPT, MENU_PROMPT};
use crate::telemetry;
use crate::voice::session::Timebase;
use lk_agents_abi::errors::{ApiErrorKind, ErrorKind, ProviderError};
use lk_agents_abi::types::provider::NativeLabel;
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, ChatContext, ChatItem, ChatMessage, ChatRequest, Closed, Llm,
    Role, ToolChoice, ToolSpec,
};
use lk_agents_voice::amd::effects::{ClassifierItem, ClassifyRequest, MenuRequest};
use lk_agents_voice::amd::events::{Category, IvrMenuOption};
use lk_agents_voice::amd::machine::{Classification, ClassifyError, Menu};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio_util::sync::CancellationToken;

/// The one tool a structured call offers; its schema is the answer's.
pub const RECORD_RESULT: &str = "record_result";
/// The header LiveKit Inference reads a request's feature from.
pub const HEADER_CLIENT_FEATURE: &str = "X-LiveKit-Client-Feature";
/// AMD's feature tag on Inference requests.
pub const CLIENT_FEATURE: &str = "AMD";
/// The choice a decision model answers.
pub const CATEGORY_CHOICE: &str = "category";

/// A model that answers one named choice: `llm.DecisionModel.evaluate` with one `Choice`. The
/// host's decision models implement it; it never reads a menu.
pub trait DecisionModel: Send + Sync {
    /// The class path logs and metrics name it by.
    fn label(&self) -> String;

    /// Every request reaches only LiveKit Inference, so AMD tags it with its client feature.
    fn inference(&self) -> bool {
        false
    }

    /// The option chosen for `choice`, or `None` when the model gave no result for it.
    fn choose<'a>(
        &'a self,
        cx: &telemetry::Cx,
        choice: Choice,
    ) -> BoxFuture<'a, Result<Option<String>, ProviderError>>;
}

/// One choice for a [`DecisionModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// What the model decides on: the request's state and the transcript.
    pub input: String,
    /// The decision's name; its result is read under it.
    pub name: String,
    pub prompt: String,
    /// The options, with no descriptions: a short definition next to a choice outweighed the
    /// prompt's conditions.
    pub options: Vec<String>,
    pub conn_options: ApiConnectOptions,
    /// Extra request headers: AMD's client feature on an Inference model.
    pub headers: BTreeMap<String, String>,
}

/// What classifies AMD's turns.
#[derive(Clone)]
pub enum Classifier {
    Llm(Arc<dyn Llm>),
    Decision(Arc<dyn DecisionModel>),
}

impl Classifier {
    /// Only an LLM reads menus.
    pub fn extracts_menus(&self) -> bool {
        matches!(self, Classifier::Llm(_))
    }
}

impl std::fmt::Debug for Classifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Classifier::Llm(llm) => f.debug_tuple("Llm").field(&llm.label()).finish(),
            Classifier::Decision(model) => f.debug_tuple("Decision").field(&model.label()).finish(),
        }
    }
}

/// How a classifier call ended, as the machine reads it.
#[derive(Debug)]
pub enum Classified {
    /// A category, or an `inference_error`.
    Result(Result<Classification, ClassifyError>),
    /// Neither a provider failure nor an invalid answer: the run ends with `internal_error`.
    Failed,
}

/// A structured call's end.
enum Called {
    /// The arguments of the one `record_result` call.
    Answer(String),
    /// No call, several, or another tool.
    Invalid,
    Failed(ProviderError),
}

/// `client_feature_kwargs`: only a model that reaches only Inference is tagged.
pub fn inference_llm(llm: &dyn Llm) -> bool {
    llm.label() == NativeLabel::InferenceLlm.as_str()
}

fn feature_headers(inference: bool) -> BTreeMap<String, String> {
    let mut headers = BTreeMap::new();
    if inference {
        headers.insert(HEADER_CLIENT_FEATURE.to_owned(), CLIENT_FEATURE.to_owned());
    }
    headers
}

/// The `record_result` tool for a pydantic model's schema.
fn record_result(parameters: serde_json::Value) -> ToolSpec {
    ToolSpec {
        name: RECORD_RESULT.into(),
        description: "Record the result using the supplied schema.".into(),
        json_schema: parameters,
        flags: Default::default(),
        provider_tool: None,
    }
}

/// `AMDResponse.model_json_schema()`.
fn category_schema() -> serde_json::Value {
    let categories: Vec<&str> = Category::ALL.iter().map(|c| c.as_str()).collect();
    serde_json::json!({
        "$defs": {
            "AMDCategory": { "enum": categories, "title": "AMDCategory", "type": "string" }
        },
        "properties": { "category": { "$ref": "#/$defs/AMDCategory" } },
        "required": ["category"],
        "title": "AMDResponse",
        "type": "object",
    })
}

/// `AMDIVRMenuResponse.model_json_schema()`.
fn menu_schema() -> serde_json::Value {
    serde_json::json!({
        "$defs": {
            "IVRMenuOption": {
                "properties": {
                    "label": { "title": "Label", "type": "string" },
                    "dtmf": { "default": "", "title": "Dtmf", "type": "string" },
                    "spoken_response": {
                        "default": "", "title": "Spoken Response", "type": "string"
                    },
                },
                "required": ["label"],
                "title": "IVRMenuOption",
                "type": "object",
            }
        },
        "properties": {
            "menu": { "title": "Menu", "type": "string" },
            "options": {
                "items": { "$ref": "#/$defs/IVRMenuOption" },
                "maxItems": lk_agents_voice::amd::machine::MAX_MENU_OPTIONS,
                "title": "Options",
                "type": "array",
            },
        },
        "required": ["menu"],
        "title": "AMDIVRMenuResponse",
        "type": "object",
    })
}

/// A structured call's request: the messages, the one tool, required, never parallel.
fn structured_request(
    chat_ctx: ChatContext,
    schema: serde_json::Value,
    inference: bool,
    conn_options: ApiConnectOptions,
) -> ChatRequest {
    let headers = feature_headers(inference);
    let extra = (!headers.is_empty()).then(|| {
        let mut extra = lk_agents_abi::Extra::new();
        extra.insert("extra_headers".into(), serde_json::json!(headers));
        extra
    });
    ChatRequest {
        chat_ctx,
        tools: vec![record_result(schema)],
        tool_choice: Some(ToolChoice::Required),
        parallel_tool_calls: Some(false),
        conn_options,
        extra,
        ..Default::default()
    }
}

/// AMD's own history as chat items: user turns, agent replies, DTMF calls and outputs.
pub fn history_items(history: &[ClassifierItem], wall: SystemTime) -> Vec<ChatItem> {
    history
        .iter()
        .enumerate()
        .map(|(n, item)| match item {
            ClassifierItem::User { turn_id, transcript } => ChatItem::Message(ChatMessage::text(
                format!("amd_turn_{turn_id}"),
                Role::User,
                transcript.for_classifier(),
                wall,
            )),
            ClassifierItem::Agent { text } => ChatItem::Message(ChatMessage::text(
                format!("amd_agent_{n}"),
                Role::Assistant,
                text.clone(),
                wall,
            )),
            ClassifierItem::DtmfCall(call) => ChatItem::FunctionCall(call.clone()),
            ClassifierItem::DtmfOutput(output) => ChatItem::FunctionCallOutput(output.clone()),
        })
        .collect()
}

/// `classify`'s LLM request: the prompt, the request's state, then AMD's history.
pub fn classify_request(
    request: &ClassifyRequest,
    inference: bool,
    conn_options: ApiConnectOptions,
    wall: SystemTime,
) -> ChatRequest {
    let mut items = vec![
        ChatItem::Message(ChatMessage::text("amd_prompt", Role::System, CLASSIFY_PROMPT, wall)),
        ChatItem::Message(ChatMessage::text("amd_state", Role::System, request.state_json(), wall)),
    ];
    items.extend(history_items(&request.history, wall));
    structured_request(ChatContext::new(items), category_schema(), inference, conn_options)
}

/// `extract_ivr_menu`'s request: the prompt and the turn's transcript.
pub fn menu_request(
    request: &MenuRequest,
    inference: bool,
    conn_options: ApiConnectOptions,
    wall: SystemTime,
) -> ChatRequest {
    let items = vec![
        ChatItem::Message(ChatMessage::text("amd_menu_prompt", Role::System, MENU_PROMPT, wall)),
        ChatItem::Message(ChatMessage::text("amd_menu", Role::User, request.user_json(), wall)),
    ];
    structured_request(ChatContext::new(items), menu_schema(), inference, conn_options)
}

/// `_format_transcript(chat_ctx, include_context_events=True)` over AMD's history.
pub fn history_transcript(history: &[ClassifierItem]) -> String {
    let lines = history.iter().filter_map(|item| match item {
        ClassifierItem::User { transcript, .. } => {
            let text = transcript.for_classifier();
            (!text.is_empty()).then(|| format!("user: {text}"))
        }
        ClassifierItem::Agent { text } => (!text.is_empty()).then(|| format!("assistant: {text}")),
        ClassifierItem::DtmfCall(call) => {
            Some(format!("[function call: {}({})]", call.name, call.arguments))
        }
        ClassifierItem::DtmfOutput(output) => {
            let label = if output.is_error { "function error" } else { "function output" };
            Some(format!("[{label}: {}]", output.output))
        }
    });
    lines.collect::<Vec<_>>().join("\n")
}

/// The decision model's choice for a request.
pub fn decision_choice(
    request: &ClassifyRequest,
    inference: bool,
    conn_options: ApiConnectOptions,
) -> Choice {
    let transcript = history_transcript(&request.history);
    Choice {
        input: format!("Request: {}\nTranscript:\n{transcript}", request.state_json()),
        name: CATEGORY_CHOICE.into(),
        prompt: CLASSIFY_PROMPT.into(),
        options: Category::ALL.iter().map(|c| c.as_str().to_owned()).collect(),
        conn_options,
        headers: feature_headers(inference),
    }
}

/// `APIError`, its timeout and invalid output give an `inference_error` prediction; any other
/// failure (a host exception outside the API taxonomy) ends the run.
fn classify_error(error: &ProviderError) -> Option<ClassifyError> {
    let timeout = match error {
        ProviderError::Local(error) => {
            error.kind == ApiErrorKind::Timeout || error.error_kind == ErrorKind::Timeout
        }
        ProviderError::Host { api_kind: Some(kind), error_kind, .. } => {
            *kind == ApiErrorKind::Timeout || *error_kind == Some(ErrorKind::Timeout)
        }
        ProviderError::Host { api_kind: None, .. } => return None,
    };
    Some(if timeout { ClassifyError::Timeout } else { ClassifyError::Api })
}

#[derive(serde::Deserialize)]
struct CategoryArgs {
    category: String,
}

fn parse_category(arguments: &str) -> Option<Category> {
    let args: CategoryArgs = serde_json::from_str(arguments).ok()?;
    Category::from_wire(&args.category)
}

#[derive(serde::Deserialize)]
struct MenuArgs {
    menu: String,
    #[serde(default)]
    options: Vec<MenuOptionArgs>,
}

#[derive(serde::Deserialize)]
struct MenuOptionArgs {
    label: String,
    #[serde(default)]
    dtmf: String,
    #[serde(default)]
    spoken_response: String,
}

/// The menu as answered; the machine refuses one with too many options, as Python's validation
/// does.
fn parse_menu(arguments: &str) -> Option<Menu> {
    let args: MenuArgs = serde_json::from_str(arguments).ok()?;
    let options = (args.options.into_iter())
        .map(|o| IvrMenuOption { label: o.label, dtmf: o.dtmf, spoken_response: o.spoken_response })
        .collect();
    Some(Menu { menu: args.menu, options })
}

/// One classification. Model time is measured around the model call alone. `None` once
/// cancelled: a cancelled request reports nothing.
pub async fn classify(
    model: Classifier,
    cx: telemetry::Cx,
    request: ClassifyRequest,
    conn_options: ApiConnectOptions,
    time: Timebase,
    cancel: CancellationToken,
) -> Option<(Classified, Option<lk_agents_abi::metrics::base::LlmMetrics>)> {
    let started = time.now();
    let (outcome, metrics) = match model {
        Classifier::Llm(llm) => {
            let inference = inference_llm(llm.as_ref());
            let chat = classify_request(&request, inference, conn_options, time.wall());
            let (called, metrics) = structured(llm, &cx, &chat, &time, &cancel).await?;
            let took = time.now().saturating_duration_since(started);
            let outcome = match called {
                Called::Answer(arguments) => match parse_category(&arguments) {
                    Some(category) => Classified::Result(Ok(Classification {
                        category,
                        inference_duration: took,
                    })),
                    None => Classified::Result(Err(ClassifyError::InvalidOutput)),
                },
                Called::Invalid => Classified::Result(Err(ClassifyError::InvalidOutput)),
                Called::Failed(error) => failed(&error),
            };
            (outcome, metrics)
        }
        Classifier::Decision(model) => {
            let choice = decision_choice(&request, model.inference(), conn_options);
            let answer = tokio::select! {
                biased;
                () = cancel.cancelled() => return None,
                answer = model.choose(&cx, choice) => answer,
            };
            let took = time.now().saturating_duration_since(started);
            let outcome = match answer {
                Ok(Some(value)) => match Category::from_wire(&value) {
                    Some(category) => Classified::Result(Ok(Classification {
                        category,
                        inference_duration: took,
                    })),
                    None => Classified::Result(Err(ClassifyError::InvalidOutput)),
                },
                Ok(None) => Classified::Result(Err(ClassifyError::InvalidOutput)),
                Err(error) => failed(&error),
            };
            (outcome, None)
        }
    };
    Some((outcome, metrics))
}

fn failed(error: &ProviderError) -> Classified {
    match classify_error(error) {
        Some(error) => Classified::Result(Err(error)),
        None => Classified::Failed,
    }
}

/// One menu extraction; any failure is silent (`None`). `None` outside once cancelled.
pub async fn extract_menu(
    llm: Arc<dyn Llm>,
    cx: telemetry::Cx,
    request: MenuRequest,
    conn_options: ApiConnectOptions,
    time: Timebase,
    cancel: CancellationToken,
) -> Option<(Option<Menu>, Option<lk_agents_abi::metrics::base::LlmMetrics>)> {
    let inference = inference_llm(llm.as_ref());
    let chat = menu_request(&request, inference, conn_options, time.wall());
    let (called, metrics) = structured(llm, &cx, &chat, &time, &cancel).await?;
    let menu = match called {
        Called::Answer(arguments) => parse_menu(&arguments),
        Called::Invalid | Called::Failed(_) => None,
    };
    Some((menu, metrics))
}

/// `_structured_response`: the call's tool calls, its stream closed on every path. `None` once
/// cancelled.
async fn structured(
    llm: Arc<dyn Llm>,
    cx: &telemetry::Cx,
    request: &ChatRequest,
    time: &Timebase,
    cancel: &CancellationToken,
) -> Option<(Called, Option<lk_agents_abi::metrics::base::LlmMetrics>)> {
    let mut trace = telemetry::LlmTrace::start(cx, llm.as_ref(), request);
    let mut attempt = Some(trace.attempt());
    // Abandoning an open is safe: a host open dropped unsettled is cancelled, and a stream it
    // returned unread is closed.
    let open = lk_agents_inference::llm::chat(llm.clone(), trace.cx(), request, time.clock(), None);
    let opened = tokio::select! {
        biased;
        () = cancel.cancelled() => return None,
        opened = open => opened,
    };
    let mut stream = match opened {
        Ok(stream) => stream,
        Err(error) => {
            trace.end_attempt(attempt.take().flatten(), Some(&error));
            trace.fail(&error);
            return Some((Called::Failed(error), None));
        }
    };
    let mut calls: Vec<lk_agents_abi::FunctionToolCall> = Vec::new();
    let mut failure = None;
    let mut aborted = false;
    loop {
        // Cancelling the whole request may drop a pull: the stream is closed below.
        let next = tokio::select! {
            biased;
            () = cancel.cancelled() => {
                aborted = true;
                break;
            }
            item = stream.next() => item,
        };
        match next {
            Ok(Some(chunk)) => {
                trace.observe(&chunk);
                if let Some(delta) = &chunk.delta {
                    calls.extend(delta.tool_calls.iter().cloned());
                }
            }
            Ok(None) => break,
            Err(error) => {
                trace.end_attempt(attempt.take().flatten(), Some(&error));
                trace.fail(&error);
                failure = Some(error);
                break;
            }
        }
    }
    // Dropping does not cancel a host stream; `close` runs on every path.
    let deadline = time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    let closed = stream.close(deadline).await;
    if closed != Closed::Clean {
        log::warn!(target: lk_agents_abi::LOG_TARGET, outcome:? = closed; "AMD model stream cleanup incomplete");
    }
    if aborted {
        return None;
    }
    if let Some(error) = failure {
        return Some((Called::Failed(error), None));
    }
    if let Some(attempt) = attempt.take() {
        trace.end_attempt(attempt, None);
    }
    let metrics = trace.end(Default::default());
    let called = match calls.as_slice() {
        [call] if call.name == RECORD_RESULT => Called::Answer(call.arguments.clone()),
        _ => Called::Invalid,
    };
    Some((called, metrics))
}

#[cfg(test)]
#[path = "classifier_tests.rs"]
mod tests;
