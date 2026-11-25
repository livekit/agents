//! The classifier and menu legs against fake models that record what they were asked. Expected
//! values come from `voice/amd/_inference.py` and `tests/test_amd.py` on `main`.

use super::*;
use crate::voice::amd::{ChoiceRequest, DecisionModel};
use lk_agents_abi::ChoiceDelta;
use lk_agents_abi::errors::{ApiError, ApiErrorKind, ErrorKind, HostError, ProviderError};
use lk_agents_abi::utils::stream::{Closed, ProviderStream, StreamOutput};
use lk_agents_abi::{
    ChatItem, ChatRequest, FunctionCall, FunctionCallOutput, FunctionToolCall, LlmChunk, LlmStream,
    Role, ToolChoice, Ts,
};
use lk_agents_voice::amd::effects::{
    AGENT_REPLY, ClassifierItem, PreviousPrediction, ReqId, TurnTranscript,
};
use lk_agents_voice::amd::events::{AmdReason, Category, IvrMenuOption, Stage};
use lk_agents_voice::amd::machine::{Classification, ClassifyError, Menu};
use lk_agents_voice::audio_recognition::TurnId;
use parking_lot::Mutex;
use serde_json::json;
use std::collections::VecDeque;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// One attempt of a fake model: how long it takes and how it ends.
enum Attempt {
    /// The open fails.
    Refuse(ProviderError),
    /// The stream yields these, each after its delay, then ends.
    Stream(Vec<(Duration, Result<LlmChunk, ProviderError>)>),
}

/// An LLM that records each request and plays its attempts in order.
struct FakeLlm {
    requests: Mutex<Vec<ChatRequest>>,
    attempts: Mutex<VecDeque<Attempt>>,
}

impl FakeLlm {
    fn new(attempts: Vec<Attempt>) -> Arc<Self> {
        Arc::new(FakeLlm { requests: Mutex::default(), attempts: Mutex::new(attempts.into()) })
    }

    fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().clone()
    }
}

struct Chunks(VecDeque<(Duration, Result<LlmChunk, ProviderError>)>);

impl StreamOutput<LlmChunk> for Chunks {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            let Some((delay, chunk)) = self.0.pop_front() else { return Ok(None) };
            tokio::time::sleep(delay).await;
            chunk.map(Some)
        })
    }

    fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl ProviderStream<LlmChunk> for Chunks {}

use lk_agents_abi::BoxFuture;

impl Llm for FakeLlm {
    fn label(&self) -> String {
        "fake.LLM".into()
    }

    fn chat<'a>(
        &'a self,
        _cx: &Cx,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, ProviderError>> {
        self.requests.lock().push(request.clone());
        let attempt = self.attempts.lock().pop_front();
        Box::pin(async move {
            match attempt {
                Some(Attempt::Refuse(error)) => Err(error),
                Some(Attempt::Stream(chunks)) => Ok(Box::new(Chunks(chunks.into())) as LlmStream),
                None => Ok(Box::new(Chunks(VecDeque::new())) as LlmStream),
            }
        })
    }
}

/// A decision model's answer, after its delay.
type Choice = (Duration, Result<Option<String>, ProviderError>);

/// A decision model that records each request and answers in order.
struct FakeDecision {
    requests: Mutex<Vec<ChoiceRequest>>,
    answers: Mutex<VecDeque<Choice>>,
}

impl FakeDecision {
    fn new(answers: Vec<Choice>) -> Arc<Self> {
        Arc::new(FakeDecision { requests: Mutex::default(), answers: Mutex::new(answers.into()) })
    }
}

impl DecisionModel for FakeDecision {
    fn label(&self) -> String {
        "fake.DecisionModel".into()
    }

    fn choose<'a>(
        &'a self,
        _cx: &Cx,
        request: &'a ChoiceRequest,
    ) -> BoxFuture<'a, Result<Option<String>, ProviderError>> {
        self.requests.lock().push(request.clone());
        let answer = self.answers.lock().pop_front();
        Box::pin(async move {
            let (delay, answer) = answer.unwrap_or((Duration::ZERO, Ok(None)));
            tokio::time::sleep(delay).await;
            answer
        })
    }
}

/// One chunk carrying these tool calls, after `delay`.
fn calls(delay: Duration, calls: &[(&str, &str)]) -> (Duration, Result<LlmChunk, ProviderError>) {
    let tool_calls = (calls.iter().enumerate())
        .map(|(i, (name, arguments))| FunctionToolCall {
            name: (*name).to_owned(),
            arguments: (*arguments).to_owned(),
            call_id: format!("call_{i}"),
            extra: None,
        })
        .collect();
    let delta = ChoiceDelta { role: Some(Role::Assistant), tool_calls, ..Default::default() };
    (delay, Ok(LlmChunk { id: "resp".into(), delta: Some(delta), ..Default::default() }))
}

/// A model that answers with one `record_result` call holding `arguments`.
fn answering(arguments: &str) -> Arc<FakeLlm> {
    FakeLlm::new(vec![Attempt::Stream(vec![calls(Duration::ZERO, &[(RECORD_RESULT, arguments)])])])
}

fn category(category: &str) -> String {
    json!({ "category": category }).to_string()
}

fn llm(model: Arc<FakeLlm>, inference: bool) -> Classifier {
    Classifier::Llm(AmdModel::supplied(model as Arc<dyn Llm>, inference))
}

fn decision(model: Arc<FakeDecision>, inference: bool) -> Classifier {
    Classifier::Decision(AmdModel::supplied(model as Arc<dyn DecisionModel>, inference))
}

/// No retries: one attempt per request.
fn once() -> ApiConnectOptions {
    ApiConnectOptions { max_retry: 0, retry_interval: 0.0, ..ApiConnectOptions::default() }
}

fn clock() -> Arc<dyn lk_agents_abi::Clock> {
    Arc::new(lk_agents_net::clock::RealClock::new())
}

fn dtmf_call() -> FunctionCall {
    FunctionCall {
        id: "item_dtmf".into(),
        call_id: "call_dtmf".into(),
        arguments: r#"{"events": ["1"]}"#.into(),
        name: "send_dtmf_events".into(),
        created_at: std::time::SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    }
}

fn dtmf_output() -> FunctionCallOutput {
    FunctionCallOutput {
        id: "item_dtmf_output".into(),
        name: "send_dtmf_events".into(),
        call_id: "call_dtmf".into(),
        output: "sent 1".into(),
        is_error: false,
        created_at: std::time::SystemTime::UNIX_EPOCH,
        reply_required: true,
    }
}

fn user(turn_id: u64, session: &str) -> ClassifierItem {
    let transcript = TurnTranscript { session: Some(session.into()), amd: None };
    ClassifierItem::User { turn_id, transcript }
}

/// `REQUEST` of `test_amd.py`, as the machine builds it for a first turn.
fn request() -> ClassifyRequest {
    ClassifyRequest {
        req: ReqId(7),
        turn: TurnId(3),
        turn_id: 1,
        stage: Stage::Uncertain,
        previous_prediction: None,
        likely_human: false,
        recommended: vec![Category::Human],
        speech_duration: Duration::from_millis(500),
        history: vec![user(1, "Hello?")],
    }
}

/// A later request in `machine-ivr`: an agent reply, a sent digit, both transcripts.
fn ivr_request() -> ClassifyRequest {
    ClassifyRequest {
        req: ReqId(9),
        turn: TurnId(5),
        turn_id: 2,
        stage: Stage::MachineIvr,
        previous_prediction: Some(PreviousPrediction {
            turn_id: 1,
            category: Category::MachineIvr,
            reason: AmdReason::Prediction,
        }),
        likely_human: false,
        recommended: lk_agents_voice::amd::machine::stage::recommended(Stage::MachineIvr),
        speech_duration: Duration::from_millis(1200),
        history: vec![
            user(1, "Thanks for calling. For billing, press 1."),
            ClassifierItem::Agent { text: AGENT_REPLY.into() },
            ClassifierItem::DtmfCall(dtmf_call()),
            ClassifierItem::DtmfOutput(dtmf_output()),
            ClassifierItem::User {
                turn_id: 2,
                transcript: TurnTranscript {
                    session: Some("For billing".into()),
                    amd: Some("For billing, press".into()),
                },
            },
        ],
    }
}

async fn classify_with(classifier: &Classifier, request: &ClassifyRequest) -> In {
    classify_on(classifier, request, once()).await
}

/// One classification no one cancels.
async fn classify_on(
    classifier: &Classifier,
    request: &ClassifyRequest,
    conn: ApiConnectOptions,
) -> In {
    let never = CancellationToken::new();
    let got = classify(classifier, request, conn, clock(), &Cx::NONE, &never).await;
    got.expect("a request no one cancelled reports")
}

fn classified(req: ReqId, category: Category, took: Duration) -> In {
    In::Classified { req, result: Ok(Classification { category, inference_duration: took }) }
}

fn failed(req: ReqId, error: ClassifyError) -> In {
    In::Classified { req, result: Err(error) }
}

/// Each item of a request as `(role or kind, text)`.
fn items(request: &ChatRequest) -> Vec<(String, String)> {
    (request.chat_ctx.items.iter())
        .map(|item| match item {
            ChatItem::Message(m) => {
                (format!("{:?}", m.role).to_lowercase(), m.raw_text_content().unwrap_or_default())
            }
            ChatItem::FunctionCall(call) => {
                ("function_call".into(), format!("{}({})", call.name, call.arguments))
            }
            ChatItem::FunctionCallOutput(output) => {
                ("function_call_output".into(), output.output.clone())
            }
            other => ("other".into(), format!("{other:?}")),
        })
        .collect()
}

fn pair(role: &str, text: &str) -> (String, String) {
    (role.to_owned(), text.to_owned())
}

/// `AMDResponse.model_json_schema()`, as pydantic 2 renders it on `main`.
fn category_schema() -> serde_json::Value {
    json!({
        "$defs": {"AMDCategory": {
            "enum": ["human", "machine-ivr", "machine-screening", "machine-vm",
                     "machine-unavailable", "wait", "uncertain"],
            "title": "AMDCategory", "type": "string"}},
        "properties": {"category": {"$ref": "#/$defs/AMDCategory"}},
        "required": ["category"], "title": "AMDResponse", "type": "object"
    })
}

/// `AMDIVRMenuResponse.model_json_schema()`, as pydantic 2 renders it on `main`.
fn menu_schema() -> serde_json::Value {
    json!({
        "$defs": {"IVRMenuOption": {
            "properties": {
                "label": {"title": "Label", "type": "string"},
                "dtmf": {"default": "", "title": "Dtmf", "type": "string"},
                "spoken_response": {"default": "", "title": "Spoken Response", "type": "string"}},
            "required": ["label"], "title": "IVRMenuOption", "type": "object"}},
        "properties": {
            "menu": {"title": "Menu", "type": "string"},
            "options": {"items": {"$ref": "#/$defs/IVRMenuOption"}, "maxItems": 20,
                        "title": "Options", "type": "array"}},
        "required": ["menu"], "title": "AMDIVRMenuResponse", "type": "object"
    })
}

fn feature_header() -> lk_agents_abi::llm::chat_context::Extra {
    let headers = json!({ HEADER_CLIENT_FEATURE: CLIENT_FEATURE });
    lk_agents_abi::llm::chat_context::Extra::from_iter([("extra_headers".to_owned(), headers)])
}

// Classifier requests (`_inference.classify`, spec §8.1).

// Catches: a request without the prompt and state as system messages ahead of AMD's own history,
// or a history that drops the agent's reply marker or the sent digit. Python: `classify` adds
// `CLASSIFY_PROMPT`, then the state JSON, then extends with the request's items.
#[tokio::test(start_paused = true)]
async fn an_llm_request_is_the_prompt_the_state_and_amds_history() {
    let model = answering(&category("machine-ivr"));
    let request = ivr_request();
    classify_with(&llm(model.clone(), false), &request).await;

    let requests = model.requests();
    assert_eq!(requests.len(), 1, "one model call per classification");
    let dual = r#"{"transcript_1": "For billing", "transcript_2": "For billing, press"}"#;
    assert_eq!(
        items(&requests[0]),
        vec![
            pair("system", CLASSIFY_PROMPT),
            pair("system", &request.state_json()),
            pair("user", "Thanks for calling. For billing, press 1."),
            pair("assistant", AGENT_REPLY),
            pair("function_call", r#"send_dtmf_events({"events": ["1"]})"#),
            pair("function_call_output", "sent 1"),
            pair("user", dual),
        ]
    );
}

// Catches: a state message that is not Python's `json.dumps` of the request state.
#[tokio::test(start_paused = true)]
async fn the_state_message_is_pythons_json() {
    let model = answering(&category("human"));
    classify_with(&llm(model.clone(), false), &request()).await;
    let state = r#"{"stage": "uncertain", "previous_prediction": null, "recommended_next_categories": ["human"], "speech_duration": 0.5}"#;
    assert_eq!(items(&model.requests()[0])[1], pair("system", state));
}

// Catches: a classifier that lets the model answer in text, call other tools or call several.
// Python: `test_amd_uses_a_required_structured_tool`.
#[tokio::test(start_paused = true)]
async fn the_request_requires_one_record_result_call() {
    let model = answering(&category("human"));
    let conn =
        ApiConnectOptions { max_retry: 0, retry_interval: 0.0, ..ApiConnectOptions::default() };
    classify_on(&llm(model.clone(), false), &request(), conn).await;

    let request = &model.requests()[0];
    assert_eq!(request.tools.len(), 1, "{:?}", request.tools);
    let tool = &request.tools[0];
    assert_eq!(tool.name, RECORD_RESULT);
    assert_eq!(tool.description, "Record the result using the supplied schema.");
    assert_eq!(tool.json_schema, category_schema());
    assert_eq!(request.tool_choice, Some(ToolChoice::Required));
    assert_eq!(request.parallel_tool_calls, Some(false));
}

// Catches: a recommended list used as a restriction. Python:
// `test_classifier_recommends_next_states_without_restricting_predictions`.
#[tokio::test(start_paused = true)]
async fn any_category_is_accepted_whatever_the_recommendations() {
    for answer in Category::ALL {
        let model = answering(&category(answer.as_str()));
        let request = ivr_request();
        let got = classify_with(&llm(model, false), &request).await;
        assert_eq!(got, classified(request.req, answer, Duration::ZERO), "{answer:?}");
    }
}

// Catches: arguments accepted that pydantic's `model_validate_json` rejects. Python:
// `test_classifier_rejects_invalid_tool_arguments`.
#[tokio::test(start_paused = true)]
async fn invalid_arguments_are_an_inference_error() {
    for arguments in [
        "not JSON",
        "[]",
        r#"{"category":"unknown"}"#,
        "```json\n{\"category\":\"human\"}\n```",
        "{}",
    ] {
        let model = answering(arguments);
        let got = classify_with(&llm(model.clone(), false), &request()).await;
        assert_eq!(got, failed(ReqId(7), ClassifyError::InvalidOutput), "{arguments}");
        assert_eq!(model.requests().len(), 1);
    }
}

// Catches: a field pydantic ignores treated as invalid output. `AMDResponse` keeps pydantic's
// default `extra="ignore"`.
#[tokio::test(start_paused = true)]
async fn extra_fields_in_the_arguments_are_ignored() {
    let model = answering(r#"{"category": "machine-vm", "reason": "a beep"}"#);
    let got = classify_with(&llm(model, false), &request()).await;
    assert_eq!(got, classified(ReqId(7), Category::MachineVm, Duration::ZERO));
}

// Catches: a classification taken from the wrong call, or from text. Python:
// `test_classifier_requires_exactly_one_result_tool`.
#[tokio::test(start_paused = true)]
async fn anything_but_exactly_one_record_result_call_is_an_inference_error() {
    let human = category("human");
    let cases: Vec<Vec<(&str, &str)>> = vec![
        vec![],
        vec![("unexpected", &human)],
        vec![(RECORD_RESULT, &human), (RECORD_RESULT, &human)],
        vec![(RECORD_RESULT, &human), ("unexpected", &human)],
    ];
    for names in cases {
        let model = FakeLlm::new(vec![Attempt::Stream(vec![calls(Duration::ZERO, &names)])]);
        let got = classify_with(&llm(model, false), &request()).await;
        assert_eq!(got, failed(ReqId(7), ClassifyError::InvalidOutput), "{names:?}");
    }
    let text = ChoiceDelta { content: Some(human.clone()), ..Default::default() };
    let chunk = LlmChunk { id: "resp".into(), delta: Some(text), ..Default::default() };
    let model = FakeLlm::new(vec![Attempt::Stream(vec![(Duration::ZERO, Ok(chunk))])]);
    let got = classify_with(&llm(model, false), &request()).await;
    assert_eq!(got, failed(ReqId(7), ClassifyError::InvalidOutput), "a text answer");
}

// Catches: model time measured from the commit (the late-final wait included) or not to the
// end of the stream. Python times `classify` alone: `started = time.monotonic()` just before the
// call, `inference_duration = time.monotonic() - started` after it.
#[tokio::test(start_paused = true)]
async fn model_time_is_measured_around_the_call() {
    let model = FakeLlm::new(vec![Attempt::Stream(vec![
        (Duration::from_millis(300), Ok(LlmChunk { id: "resp".into(), ..Default::default() })),
        calls(Duration::from_millis(150), &[(RECORD_RESULT, &category("machine-vm"))]),
    ])]);
    tokio::time::sleep(Duration::from_secs(2)).await;
    let got = classify_with(&llm(model, false), &request()).await;
    assert_eq!(got, classified(ReqId(7), Category::MachineVm, Duration::from_millis(450)));
}

// Catches: a provider failure that ends the run instead of giving an `inference_error`
// prediction (spec §8.4: an `APIError` or a timeout).
#[tokio::test(start_paused = true)]
async fn provider_failures_are_inference_errors() {
    let server = || ProviderError::Local(ApiError::new(ErrorKind::Server, "down"));
    let timeout = || ProviderError::Local(ApiError::new(ErrorKind::Timeout, "slow"));
    let host_api = || ProviderError::Host {
        error: HostError::new("APIStatusError"),
        api_kind: Some(ApiErrorKind::Status),
        error_kind: Some(ErrorKind::Server),
        retryable: false,
        status_code: Some(500),
        said: None,
    };
    let cases: Vec<(Attempt, ClassifyError)> = vec![
        (Attempt::Refuse(server()), ClassifyError::Api),
        (Attempt::Refuse(host_api()), ClassifyError::Api),
        (Attempt::Refuse(timeout()), ClassifyError::Timeout),
        (
            Attempt::Stream(vec![
                (Duration::ZERO, Ok(LlmChunk { id: "resp".into(), ..Default::default() })),
                (Duration::ZERO, Err(server())),
            ]),
            ClassifyError::Api,
        ),
    ];
    for (attempt, expected) in cases {
        let model = FakeLlm::new(vec![attempt]);
        let got = classify_with(&llm(model.clone(), false), &request()).await;
        assert_eq!(got, failed(ReqId(7), expected));
        assert_eq!(model.requests().len(), 1);
    }
}

// Catches: a failure outside the provider taxonomy turned into an `inference_error`. Python's
// `_classify` re-raises anything but `APIError`, `ValueError` and `TimeoutError`, and the failed
// task ends the run with `internal_error`: `test_unexpected_inference_errors_finish_with_internal_error`.
#[tokio::test(start_paused = true)]
async fn a_failure_outside_the_provider_taxonomy_fails_the_leg() {
    let crash = ProviderError::from(HostError::new("RuntimeError: plugin bug"));
    let model = FakeLlm::new(vec![Attempt::Refuse(crash)]);
    let got = classify_with(&llm(model.clone(), false), &request()).await;
    assert_eq!(got, In::LegFailed);
    assert_eq!(model.requests().len(), 1);
}

// Catches: a classifier that ignores the session's `llm_conn_options`. Python:
// `test_llm_calls_use_session_connection_options` (retries within the session's budget).
#[tokio::test(start_paused = true)]
async fn the_session_connection_options_bound_the_retries() {
    let lost =
        || Attempt::Refuse(ProviderError::Local(ApiError::new(ErrorKind::Connection, "reset")));
    let answer =
        || Attempt::Stream(vec![calls(Duration::ZERO, &[(RECORD_RESULT, &category("human"))])]);

    let retried = ApiConnectOptions { max_retry: 1, retry_interval: 0.0, ..once() };
    let model = FakeLlm::new(vec![lost(), answer()]);
    let got = classify_on(&llm(model.clone(), false), &request(), retried).await;
    assert!(
        matches!(
            got,
            In::Classified {
                req: ReqId(7),
                result: Ok(Classification { category: Category::Human, .. })
            }
        ),
        "{got:?}"
    );
    assert_eq!(model.requests().len(), 2, "one retry");

    let model = FakeLlm::new(vec![lost(), answer()]);
    let got = classify_on(&llm(model.clone(), false), &request(), once()).await;
    assert_eq!(got, failed(ReqId(7), ClassifyError::Api));
    assert_eq!(model.requests().len(), 1, "no retry");
}

// Catches: AMD's client feature header sent to a plugin model, or missing on an Inference one.
// Python: `test_plugin_llm_gets_no_header` (`extra_kwargs` is `NOT_GIVEN`) and
// `client_feature_kwargs` (`{"extra_headers": {HEADER_CLIENT_FEATURE: "AMD"}}`).
#[tokio::test(start_paused = true)]
async fn the_feature_header_goes_only_to_an_inference_model() {
    let model = answering(&category("human"));
    classify_with(&llm(model.clone(), true), &request()).await;
    assert_eq!(model.requests()[0].extra, Some(feature_header()));

    let model = answering(&category("human"));
    classify_with(&llm(model.clone(), false), &request()).await;
    assert_eq!(model.requests()[0].extra, None);
}

// Catches: a cancelled request still reporting its result, which the machine would read as the
// newer request's. Python cancels `_classify_atask` and its result never lands.
#[tokio::test(start_paused = true)]
async fn a_cancelled_classification_reports_nothing() {
    let model = answering(&category("human"));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let got = classify(&llm(model, false), &request(), once(), clock(), &Cx::NONE, &cancel).await;
    assert_eq!(got, None);
}

// Decision models (spec §8.1, §14 invariant 14).

// Catches: a decision model asked anything but one `category` choice over every category with
// the classifier prompt, or given another input format. Python:
// `test_decision_model_classifies_with_one_choice_and_skips_menus` and `_format_transcript(...,
// include_context_events=True)`.
#[tokio::test(start_paused = true)]
async fn a_decision_model_gets_one_choice_with_the_state_and_transcript() {
    let model = FakeDecision::new(vec![(Duration::ZERO, Ok(Some("machine-ivr".into())))]);
    let request = ivr_request();
    let conn = ApiConnectOptions { max_retry: 2, retry_interval: 0.0, ..once() };
    let got = classify_on(&decision(model.clone(), false), &request, conn).await;
    assert_eq!(got, classified(request.req, Category::MachineIvr, Duration::ZERO));

    let requests = model.requests.lock().clone();
    assert_eq!(requests.len(), 1, "one choice per classification");
    let choice = &requests[0];
    assert_eq!(choice.name, "category");
    assert_eq!(choice.prompt, CLASSIFY_PROMPT);
    let every: Vec<String> = Category::ALL.iter().map(|c| c.as_str().to_owned()).collect();
    assert_eq!(choice.options, every, "`AMDCategory` in declaration order");
    let transcript = [
        "user: Thanks for calling. For billing, press 1.",
        &format!("assistant: {AGENT_REPLY}"),
        r#"[function call: send_dtmf_events({"events": ["1"]})]"#,
        "[function output: sent 1]",
        r#"user: {"transcript_1": "For billing", "transcript_2": "For billing, press"}"#,
    ]
    .join("\n");
    assert_eq!(
        choice.input,
        format!("Request: {}\nTranscript:\n{transcript}", request.state_json())
    );
    assert_eq!(choice.conn_options, conn);
    assert!(choice.extra_headers.is_empty(), "no header for a plugin model");
}

// Catches: a decision model's missing or unknown choice, or its failure, ending the run. Python:
// `test_decision_model_error_is_an_inference_error` (no `ChoiceResult` raises `ValueError`).
#[tokio::test(start_paused = true)]
async fn a_decision_model_without_a_valid_choice_is_an_inference_error() {
    let server = ProviderError::Local(ApiError::new(ErrorKind::Server, "down"));
    let cases = [
        (Ok(None), ClassifyError::InvalidOutput),
        (Ok(Some("voicemail".to_owned())), ClassifyError::InvalidOutput),
        (Err(server), ClassifyError::Api),
    ];
    for (answer, expected) in cases {
        let model = FakeDecision::new(vec![(Duration::ZERO, answer)]);
        let got = classify_with(&decision(model.clone(), false), &request()).await;
        assert_eq!(got, failed(ReqId(7), expected));
        assert_eq!(model.requests.lock().len(), 1);
    }
}

// Catches: a decision model's time not measured, or the header missing on an Inference model.
// Python: `test_inference_decision_model_is_tagged`.
#[tokio::test(start_paused = true)]
async fn an_inference_decision_model_is_tagged_and_timed() {
    let model = FakeDecision::new(vec![(Duration::from_millis(250), Ok(Some("human".into())))]);
    let got = classify_with(&decision(model.clone(), true), &request()).await;
    assert_eq!(got, classified(ReqId(7), Category::Human, Duration::from_millis(250)));
    let headers = model.requests.lock()[0].extra_headers.clone();
    assert_eq!(
        headers,
        BTreeMap::from([(HEADER_CLIENT_FEATURE.to_owned(), CLIENT_FEATURE.to_owned())])
    );
}

use std::collections::BTreeMap;

// Menu extraction (`_inference.extract_ivr_menu`, spec §11.2).

fn menu_request(transcript: &str) -> MenuRequest {
    MenuRequest { req: ReqId(11), turn_id: 2, transcript: transcript.into() }
}

async fn extract_with(model: Arc<FakeLlm>, inference: bool, request: &MenuRequest) -> In {
    let model = AmdModel::supplied(model as Arc<dyn Llm>, inference);
    let never = CancellationToken::new();
    let got = extract_menu(&model, request, once(), clock(), &Cx::NONE, &never).await;
    got.expect("a request no one cancelled reports")
}

fn menu_answer(arguments: serde_json::Value) -> Arc<FakeLlm> {
    answering(&arguments.to_string())
}

fn option(label: &str, dtmf: &str, spoken: &str) -> IvrMenuOption {
    IvrMenuOption { label: label.into(), dtmf: dtmf.into(), spoken_response: spoken.into() }
}

// Catches: a menu request that is not the menu prompt and the transcript as Python's JSON, or
// not one required `record_result` call. Python: `extract_ivr_menu`, `_structured_response`.
#[tokio::test(start_paused = true)]
async fn a_menu_request_is_the_prompt_and_the_transcript() {
    let model =
        menu_answer(json!({"menu": "Main menu", "options": [{"label": "Sales", "dtmf": "1"}]}));
    extract_with(model.clone(), false, &menu_request("Press 1 for sales, or say café.")).await;

    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(
        items(request),
        vec![
            pair("system", MENU_PROMPT),
            // `json.dumps` escapes non-ASCII.
            pair("user", "{\"transcript\": \"Press 1 for sales, or say caf\\u00e9.\"}"),
        ]
    );
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, RECORD_RESULT);
    assert_eq!(request.tools[0].json_schema, menu_schema());
    assert_eq!(request.tool_choice, Some(ToolChoice::Required));
    assert_eq!(request.parallel_tool_calls, Some(false));
    assert_eq!(request.extra, None);
}

// Catches: the menu leg dropping the feature header an Inference model's requests carry.
#[tokio::test(start_paused = true)]
async fn an_inference_menu_request_is_tagged() {
    let model = menu_answer(json!({"menu": "Main menu", "options": []}));
    extract_with(model.clone(), true, &menu_request("Press 1.")).await;
    assert_eq!(model.requests()[0].extra, Some(feature_header()));
}

// Catches: a menu that loses its options or their defaults (`dtmf` and `spoken_response` default
// to `""`).
#[tokio::test(start_paused = true)]
async fn a_menu_answer_is_the_menu_and_its_options() {
    let model = menu_answer(json!({
        "menu": "Choose a department",
        "options": [{"label": "Sales", "dtmf": "1"}, {"label": "Support", "spoken_response": "support"}]
    }));
    let got = extract_with(model, false, &menu_request("Press 1 for sales, or say support.")).await;
    let menu = Menu {
        menu: "Choose a department".into(),
        options: vec![option("Sales", "1", ""), option("Support", "", "support")],
    };
    assert_eq!(got, In::MenuExtracted { req: ReqId(11), menu: Some(menu) });
}

// Catches: an answer that fails `AMDIVRMenuResponse` validation observed anyway. At most 20
// options (`max_length=20`); `label` and `menu` are required.
#[tokio::test(start_paused = true)]
async fn a_menu_answer_that_does_not_validate_is_dropped() {
    let twenty: Vec<_> = (0..20).map(|i| json!({"label": format!("o{i}"), "dtmf": "1"})).collect();
    let model = menu_answer(json!({"menu": "Long", "options": twenty}));
    let got = extract_with(model, false, &menu_request("A long menu.")).await;
    assert!(
        matches!(&got, In::MenuExtracted { menu: Some(menu), .. } if menu.options.len() == 20),
        "twenty options validate: {got:?}"
    );

    let twenty_one: Vec<_> = (0..21).map(|i| json!({"label": format!("o{i}")})).collect();
    for arguments in [
        json!({"menu": "Long", "options": twenty_one}),
        json!({"menu": "No label", "options": [{"dtmf": "1"}]}),
        json!({"options": []}),
    ] {
        let model = menu_answer(arguments.clone());
        let got = extract_with(model.clone(), false, &menu_request("Press 1.")).await;
        assert_eq!(got, In::MenuExtracted { req: ReqId(11), menu: None }, "{arguments}");
        assert_eq!(model.requests().len(), 1);
    }
}

// Catches: a menu failure that ends the run. Python's `_extract_menu` catches every exception and
// returns: `test_unexpected_menu_error_does_not_end_detection`.
#[tokio::test(start_paused = true)]
async fn a_menu_failure_is_silent() {
    let crash = ProviderError::from(HostError::new("RuntimeError: menu bug"));
    let server = ProviderError::Local(ApiError::new(ErrorKind::Server, "down"));
    for attempt in [
        Attempt::Refuse(crash),
        Attempt::Refuse(server),
        Attempt::Stream(vec![calls(Duration::ZERO, &[])]),
        Attempt::Stream(vec![calls(Duration::ZERO, &[(RECORD_RESULT, "not JSON")])]),
    ] {
        let model = FakeLlm::new(vec![attempt]);
        let got = extract_with(model.clone(), false, &menu_request("Press 1.")).await;
        assert_eq!(got, In::MenuExtracted { req: ReqId(11), menu: None });
        assert_eq!(model.requests().len(), 1);
    }
}
