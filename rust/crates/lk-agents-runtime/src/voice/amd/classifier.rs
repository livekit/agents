//! AMD's classifier and menu legs: one model call each, rendered as Python's `_inference.py`
//! renders it, and its outcome mapped onto the machine's inputs.
//!
//! An LLM answers through a `record_result` tool with `tool_choice="required"` and must call it
//! exactly once. A decision model answers one `category` choice. A provider failure, an invalid
//! answer or a timeout is an `inference_error` prediction; any other failure ends the run.

pub use super::prompts::{CLASSIFY_PROMPT, MENU_PROMPT};
use super::{AmdModel, ChoiceRequest, Classifier};
use crate::telemetry;
use lk_agents_abi::errors::{ApiErrorKind, ErrorKind, HostErrorKind, ProviderError};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::{
    ApiConnectOptions, ChatContext, ChatItem, ChatMessage, ChatRequest, Clock, Closed, Llm, Role,
    ToolChoice, ToolSpec,
};
use lk_agents_voice::amd::effects::{ClassifierItem, ClassifyRequest, MenuRequest};
use lk_agents_voice::amd::events::{Category, IvrMenuOption};
use lk_agents_voice::amd::machine::{Classification, ClassifyError, In, MAX_MENU_OPTIONS, Menu};
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

/// A structured call's end.
enum Called {
    /// The arguments of the one `record_result` call.
    Answer(String),
    /// No call, several, or another tool.
    Invalid,
    Failed(ProviderError),
}

pub(super) fn feature_headers(inference: bool) -> BTreeMap<String, String> {
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
                "maxItems": MAX_MENU_OPTIONS,
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
) -> ChoiceRequest {
    let transcript = history_transcript(&request.history);
    ChoiceRequest {
        input: format!("Request: {}\nTranscript:\n{transcript}", request.state_json()),
        name: CATEGORY_CHOICE.into(),
        prompt: CLASSIFY_PROMPT.into(),
        options: Category::ALL.iter().map(|c| c.as_str().to_owned()).collect(),
        conn_options,
        extra_headers: feature_headers(inference),
    }
}

/// `APIError`, its timeout, a model's `ValueError` or `TimeoutError` and invalid output give an
/// `inference_error` prediction (`detector.py` `_classify`); any other failure (a host exception
/// outside these) ends the run.
fn classify_error(error: &ProviderError) -> Option<ClassifyError> {
    let timeout = match error {
        ProviderError::Local(error) => {
            error.kind == ApiErrorKind::Timeout || error.error_kind == ErrorKind::Timeout
        }
        ProviderError::Host { api_kind: Some(kind), error_kind, .. } => {
            *kind == ApiErrorKind::Timeout || *error_kind == Some(ErrorKind::Timeout)
        }
        ProviderError::Host { api_kind: None, error, .. } => match error.kind {
            HostErrorKind::Invalid => false,
            HostErrorKind::Timeout => true,
            _ => return None,
        },
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

/// The menu as answered, or `None` when it fails `AMDIVRMenuResponse` validation: more than
/// [`MAX_MENU_OPTIONS`] options (`max_length=20`) is invalid output, so it is never observed.
fn parse_menu(arguments: &str) -> Option<Menu> {
    let args: MenuArgs = serde_json::from_str(arguments).ok()?;
    if args.options.len() > MAX_MENU_OPTIONS {
        return None;
    }
    let options = (args.options.into_iter())
        .map(|o| IvrMenuOption { label: o.label, dtmf: o.dtmf, spoken_response: o.spoken_response })
        .collect();
    Some(Menu { menu: args.menu, options })
}

/// Runs `request` on the classifier and returns what the machine is told: `In::Classified` with
/// the category and the model time measured around the call, or an `inference_error` category of
/// failure; `In::LegFailed` for a failure outside the provider taxonomy. `None` once `cancel`
/// fired: a cancelled request reports nothing, and its stream is closed.
pub async fn classify(
    classifier: &Classifier,
    request: &ClassifyRequest,
    conn_options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    cx: &Cx,
    cancel: &CancellationToken,
) -> Option<In> {
    let req = request.req;
    let started = clock.now();
    let answered = |answer: Option<Category>, clock: &dyn Clock| match answer {
        Some(category) => {
            let inference_duration = clock.now().saturating_duration_since(started);
            In::Classified { req, result: Ok(Classification { category, inference_duration }) }
        }
        None => In::Classified { req, result: Err(ClassifyError::InvalidOutput) },
    };
    let got = match classifier {
        Classifier::Llm(llm) => {
            let chat = classify_request(request, llm.inference, conn_options, clock.wall());
            match structured(llm.model.clone(), cx, &chat, clock.clone(), cancel).await? {
                Called::Answer(arguments) => answered(parse_category(&arguments), clock.as_ref()),
                Called::Invalid => answered(None, clock.as_ref()),
                Called::Failed(error) => failed(req, &error),
            }
        }
        Classifier::Decision(model) => {
            let choice = decision_choice(request, model.inference, conn_options);
            let answer = tokio::select! {
                biased;
                () = cancel.cancelled() => return None,
                answer = model.model.choose(cx, &choice) => answer,
            };
            match answer {
                Ok(value) => {
                    answered(value.as_deref().and_then(Category::from_wire), clock.as_ref())
                }
                Err(error) => failed(req, &error),
            }
        }
    };
    Some(got)
}

fn failed(req: lk_agents_voice::amd::effects::ReqId, error: &ProviderError) -> In {
    match classify_error(error) {
        Some(error) => In::Classified { req, result: Err(error) },
        None => In::LegFailed,
    }
}

/// Extracts the menu `request` names (spec §11.2) and returns `In::MenuExtracted`: the menu, or
/// `None` when the model failed or answered something that does not validate. A failure is
/// silent. `None` outside once `cancel` fired.
pub async fn extract_menu(
    model: &AmdModel<dyn Llm>,
    request: &MenuRequest,
    conn_options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    cx: &Cx,
    cancel: &CancellationToken,
) -> Option<In> {
    let chat = menu_request(request, model.inference, conn_options, clock.wall());
    let menu = match structured(model.model.clone(), cx, &chat, clock, cancel).await? {
        Called::Answer(arguments) => parse_menu(&arguments),
        Called::Invalid | Called::Failed(_) => None,
    };
    Some(In::MenuExtracted { req: request.req, menu })
}

/// `_structured_response`: the call's tool calls, its stream closed on every path. `None` once
/// cancelled.
async fn structured(
    llm: Arc<dyn Llm>,
    cx: &Cx,
    request: &ChatRequest,
    clock: Arc<dyn Clock>,
    cancel: &CancellationToken,
) -> Option<Called> {
    let mut trace = telemetry::LlmTrace::start(cx, llm.as_ref(), request);
    let mut attempt = Some(trace.attempt());
    // Abandoning an open is safe: a host open dropped unsettled is cancelled, and a stream it
    // returned unread is closed.
    let open =
        lk_agents_inference::llm::chat(llm.clone(), trace.cx(), request, clock.clone(), None);
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
            return Some(Called::Failed(error));
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
    let deadline = clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    let closed = stream.close(deadline).await;
    if closed != Closed::Clean {
        log::warn!(target: lk_agents_abi::LOG_TARGET, outcome:? = closed; "AMD model stream cleanup incomplete");
    }
    if aborted {
        return None;
    }
    if let Some(error) = failure {
        return Some(Called::Failed(error));
    }
    if let Some(attempt) = attempt.take() {
        trace.end_attempt(attempt, None);
    }
    // AMD's own model usage is not the session's, as in Python.
    let _ = trace.end(Default::default());
    let called = match calls.as_slice() {
        [call] if call.name == RECORD_RESULT => Called::Answer(call.arguments.clone()),
        _ => Called::Invalid,
    };
    Some(called)
}

#[cfg(test)]
#[path = "classifier_tests.rs"]
mod tests;
