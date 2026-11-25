#![allow(clippy::disallowed_types, reason = "tests drive tokio's sync primitives directly")]

use super::*;
use lk_agents_abi::errors::{ApiError, ErrorKind, HostError};
use lk_agents_abi::{FunctionToolCall, LlmChunk, Ts};
use lk_agents_core::markup::ChatMessageExt;
use lk_agents_voice::amd::effects::{ReqId, TurnTranscript};
use lk_agents_voice::amd::events::Stage;
use lk_agents_voice::audio_recognition::TurnId;
use parking_lot::Mutex;
use std::time::Duration;

/// What a scripted model call does.
#[derive(Clone)]
enum Script {
    /// The stream answers with these tool calls, then ends.
    Calls(Vec<(&'static str, &'static str)>),
    /// The open fails.
    Refuse(ProviderError),
}

/// A model that answers every request with its script and records what it was asked.
struct ScriptedLlm {
    label: String,
    script: Script,
    requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptedLlm {
    fn new(script: Script) -> Arc<Self> {
        Arc::new(ScriptedLlm { label: "openai.LLM".into(), script, requests: Mutex::default() })
    }

    fn inference(script: Script) -> Arc<Self> {
        Arc::new(ScriptedLlm {
            label: NativeLabel::InferenceLlm.as_str().into(),
            script,
            requests: Mutex::default(),
        })
    }

    fn request(&self) -> ChatRequest {
        self.requests.lock().last().cloned().expect("the model was asked")
    }
}

struct Answer(Vec<LlmChunk>);

impl lk_agents_abi::StreamOutput<LlmChunk> for Answer {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        let next = (!self.0.is_empty()).then(|| self.0.remove(0));
        Box::pin(async move { Ok(next) })
    }

    fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Answer {}

impl Llm for ScriptedLlm {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn chat<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<lk_agents_abi::LlmStream, ProviderError>> {
        self.requests.lock().push(request.clone());
        let script = self.script.clone();
        Box::pin(async move {
            match script {
                Script::Refuse(error) => Err(error),
                Script::Calls(calls) => {
                    let chunks = (calls.into_iter().enumerate())
                        .map(|(n, (name, arguments))| LlmChunk {
                            id: format!("chunk_{n}"),
                            delta: Some(lk_agents_abi::ChoiceDelta {
                                tool_calls: vec![FunctionToolCall {
                                    name: name.into(),
                                    arguments: arguments.into(),
                                    call_id: format!("call_{n}"),
                                    extra: None,
                                }],
                                ..Default::default()
                            }),
                            ..Default::default()
                        })
                        .collect();
                    Ok(Box::new(Answer(chunks)) as lk_agents_abi::LlmStream)
                }
            }
        })
    }
}

fn time() -> Timebase {
    Timebase::from_virtual(Arc::new(lk_agents_abi::VirtualClock::new()))
}

/// One attempt, so a failure is the call's outcome.
fn once() -> ApiConnectOptions {
    ApiConnectOptions { max_retry: 0, ..Default::default() }
}

fn request(history: Vec<ClassifierItem>) -> ClassifyRequest {
    ClassifyRequest {
        req: ReqId(1),
        turn: TurnId(1),
        turn_id: 1,
        stage: Stage::Uncertain,
        previous_prediction: None,
        likely_human: false,
        recommended: vec![Category::Human, Category::MachineVm],
        speech_duration: Duration::from_millis(1500),
        history,
    }
}

fn user(text: &str) -> ClassifierItem {
    let transcript = TurnTranscript { session: Some(text.into()), amd: None };
    ClassifierItem::User { turn_id: 1, transcript }
}

async fn classified(llm: Arc<ScriptedLlm>) -> Classified {
    let model = Classifier::Llm(llm);
    let cancel = CancellationToken::new();
    let call =
        classify(model, telemetry::Cx::NONE, request(vec![user("Hello?")]), once(), time(), cancel);
    call.await.expect("not cancelled").0
}

fn category(outcome: &Classified) -> Option<Category> {
    match outcome {
        Classified::Result(Ok(classification)) => Some(classification.category),
        _ => None,
    }
}

// Catches: a request that lets the model answer in text or call tools in parallel, a lost state
// message, or AMD's history placed before the prompt.
#[tokio::test(start_paused = true)]
async fn a_classification_offers_one_required_record_result_after_the_prompt_and_state() {
    let llm =
        ScriptedLlm::new(Script::Calls(vec![(RECORD_RESULT, r#"{"category": "machine-vm"}"#)]));
    let outcome = classified(llm.clone()).await;
    assert_eq!(category(&outcome), Some(Category::MachineVm));
    let request = llm.request();
    assert_eq!(request.tool_choice, Some(ToolChoice::Required));
    assert_eq!(request.parallel_tool_calls, Some(false));
    let tools: Vec<&str> = request.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tools, [RECORD_RESULT]);
    let texts: Vec<(Role, String)> = (request.chat_ctx.items.iter())
        .filter_map(|item| {
            item.as_message().map(|m| (m.role, m.text_content().unwrap_or_default()))
        })
        .collect();
    assert_eq!(texts[0], (Role::System, CLASSIFY_PROMPT.to_owned()));
    assert!(texts[1].1.starts_with(r#"{"stage": "uncertain", "previous_prediction": null"#));
    assert_eq!(texts[2], (Role::User, "Hello?".to_owned()));
    assert_eq!(request.extra, None, "only an Inference model is tagged");
}

// Catches: the AMD client feature sent to a provider that is not LiveKit Inference, or missing
// on one that is.
#[tokio::test(start_paused = true)]
async fn only_an_inference_model_carries_the_amd_client_feature() {
    let llm =
        ScriptedLlm::inference(Script::Calls(vec![(RECORD_RESULT, r#"{"category": "human"}"#)]));
    let outcome = classified(llm.clone()).await;
    assert_eq!(category(&outcome), Some(Category::Human));
    let extra = llm.request().extra.expect("tagged");
    assert_eq!(extra["extra_headers"][HEADER_CLIENT_FEATURE], CLIENT_FEATURE);
}

// Catches: a second call, another tool, or a value outside the categories read as a prediction
// instead of an `inference_error`.
#[tokio::test(start_paused = true)]
async fn an_answer_other_than_one_valid_record_result_is_invalid_output() {
    let invalid = [
        vec![],
        vec![
            (RECORD_RESULT, r#"{"category": "human"}"#),
            (RECORD_RESULT, r#"{"category": "human"}"#),
        ],
        vec![("send_dtmf_events", r#"{"category": "human"}"#)],
        vec![(RECORD_RESULT, r#"{"category": "person"}"#)],
        vec![(RECORD_RESULT, "not json")],
    ];
    for calls in invalid {
        let outcome = classified(ScriptedLlm::new(Script::Calls(calls.clone()))).await;
        assert!(
            matches!(outcome, Classified::Result(Err(ClassifyError::InvalidOutput))),
            "{calls:?}: {outcome:?}"
        );
    }
}

// Catches: a provider timeout or refusal ending the run instead of predicting `inference_error`,
// or a host exception outside the API taxonomy kept as a prediction.
#[tokio::test(start_paused = true)]
async fn provider_failures_are_inference_errors_and_other_failures_end_the_run() {
    let cases = [
        (
            ProviderError::Local(ApiError::new(ErrorKind::Timeout, "slow")),
            Some(ClassifyError::Timeout),
        ),
        (
            ProviderError::Local(ApiError::new(ErrorKind::InvalidRequest, "no")),
            Some(ClassifyError::Api),
        ),
        (ProviderError::from(HostError::new("boom")), None),
    ];
    for (error, expected) in cases {
        let outcome = classified(ScriptedLlm::new(Script::Refuse(error.clone()))).await;
        match (outcome, expected) {
            (Classified::Result(Err(got)), Some(want)) => assert_eq!(got, want, "{error}"),
            (Classified::Failed, None) => {}
            (outcome, _) => panic!("{error}: {outcome:?}"),
        }
    }
}

// Catches: a cancelled request still reporting its result, which the machine would read as the
// newer request's.
#[tokio::test(start_paused = true)]
async fn a_cancelled_classification_reports_nothing() {
    let llm = ScriptedLlm::new(Script::Calls(vec![(RECORD_RESULT, r#"{"category": "human"}"#)]));
    let cancel = CancellationToken::new();
    cancel.cancel();
    let call = classify(
        Classifier::Llm(llm),
        telemetry::Cx::NONE,
        request(vec![]),
        once(),
        time(),
        cancel,
    );
    assert!(call.await.is_none());
}

struct ScriptedDecision {
    inference: bool,
    answer: Option<&'static str>,
    choices: Mutex<Vec<Choice>>,
}

impl DecisionModel for ScriptedDecision {
    fn label(&self) -> String {
        "decision".into()
    }

    fn inference(&self) -> bool {
        self.inference
    }

    fn choose<'a>(
        &'a self,
        _cx: &telemetry::Cx,
        choice: Choice,
    ) -> BoxFuture<'a, Result<Option<String>, ProviderError>> {
        self.choices.lock().push(choice);
        let answer = self.answer.map(str::to_owned);
        Box::pin(async move { Ok(answer) })
    }
}

// Catches: a decision model given a chat request instead of one `category` choice over every
// category, its transcript lines in another format, or a missing answer read as a prediction.
#[tokio::test(start_paused = true)]
async fn a_decision_model_answers_one_category_choice() {
    for (answer, expected) in
        [(Some("wait"), Some(Category::Wait)), (None, None), (Some("bot"), None)]
    {
        let model =
            Arc::new(ScriptedDecision { inference: true, answer, choices: Mutex::default() });
        let history =
            vec![user("Please hold."), ClassifierItem::Agent { text: "(the agent spoke)".into() }];
        let call = classify(
            Classifier::Decision(model.clone()),
            telemetry::Cx::NONE,
            request(history),
            once(),
            time(),
            CancellationToken::new(),
        );
        let outcome = call.await.expect("not cancelled").0;
        assert_eq!(category(&outcome), expected, "{answer:?}");
        if expected.is_none() {
            assert!(matches!(outcome, Classified::Result(Err(ClassifyError::InvalidOutput))));
        }
        let choice = model.choices.lock().pop().expect("asked");
        assert_eq!(choice.name, CATEGORY_CHOICE);
        assert_eq!(choice.options.len(), Category::ALL.len());
        assert!(choice.input.starts_with("Request: {\"stage\": \"uncertain\""), "{}", choice.input);
        assert!(
            choice.input.ends_with("Transcript:\nuser: Please hold.\nassistant: (the agent spoke)"),
            "{}",
            choice.input
        );
        assert_eq!(
            choice.headers.get(HEADER_CLIENT_FEATURE).map(String::as_str),
            Some(CLIENT_FEATURE)
        );
    }
}

// Catches: a menu extraction failure surfacing, or options with absent keys dropped instead of
// read as empty.
#[tokio::test(start_paused = true)]
async fn a_menu_reads_its_options_and_a_failure_is_silent() {
    let request =
        MenuRequest { req: ReqId(2), turn_id: 1, transcript: "Press 1 for sales.".into() };
    let menu = r#"{"menu": "main", "options": [{"label": "sales", "dtmf": "1"}]}"#;
    let llm = ScriptedLlm::new(Script::Calls(vec![(RECORD_RESULT, menu)]));
    let call = extract_menu(
        llm.clone(),
        telemetry::Cx::NONE,
        request.clone(),
        once(),
        time(),
        CancellationToken::new(),
    );
    let (menu, _) = call.await.expect("not cancelled");
    let menu = menu.expect("a menu");
    assert_eq!(menu.menu, "main");
    assert_eq!(menu.options[0].dtmf, "1");
    assert_eq!(menu.options[0].spoken_response, "");
    let user = llm.request().chat_ctx.items[1].as_message().and_then(|m| m.text_content());
    assert_eq!(user.as_deref(), Some(r#"{"transcript": "Press 1 for sales."}"#));

    let refused = ProviderError::Local(ApiError::new(ErrorKind::Server, "down"));
    let llm = ScriptedLlm::new(Script::Refuse(refused));
    let call =
        extract_menu(llm, telemetry::Cx::NONE, request, once(), time(), CancellationToken::new());
    assert_eq!(call.await.expect("not cancelled").0, None);
}
