//! The pure transcript and verdict rules, and judges run on a scripted model and a fake host.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use lk_agents_abi::errors::{HostError, ProviderError, RefRegistry};
use lk_agents_abi::{
    AgentConfigUpdate, AgentHandoffItem, ChoiceDelta, Closed, FunctionCall, FunctionCallOutput,
    FunctionToolCall, LlmStream, Ts,
};
use parking_lot::Mutex;
use tokio::sync::Barrier;

use super::*;
use lk_agents_net::env::ProcessEnv;

const T0: SystemTime = SystemTime::UNIX_EPOCH;

fn msg(id: &str, role: Role, text: &str) -> ChatItem {
    ChatMessage::text(id, role, text, T0).into()
}

fn handoff(id: &str, old: Option<&str>, new: &str) -> ChatItem {
    AgentHandoffItem {
        id: id.into(),
        old_agent_id: old.map(str::to_owned),
        new_agent_id: new.into(),
        created_at: T0,
    }
    .into()
}

// Catches: an item kind dropped from the judge transcript, the `[interrupted]` marker lost,
// multi-line instructions splitting the line, or the first assignment skipped.
#[test]
fn every_item_kind_reaches_the_transcript_in_order() {
    let mut cut = ChatMessage::text("m3", Role::Assistant, "your balance is", T0);
    cut.interrupted = true;
    let items = vec![
        msg("m1", Role::User, "book me a table"),
        FunctionCall {
            id: "f1".into(),
            call_id: "c1".into(),
            arguments: r#"{"party":2}"#.into(),
            name: "find_table".into(),
            created_at: T0,
            extra: Default::default(),
            group_id: None,
        }
        .into(),
        FunctionCallOutput::new("o1", "c1", "table 7 at 8pm", false, T0).into(),
        handoff("h1", Some("host"), "concierge"),
        FunctionCallOutput::new("o2", "c2", "no such table", true, T0).into(),
        AgentConfigUpdate {
            id: "u1".into(),
            instructions: Some("Be brief.\nDon't guess.".into()),
            tools_added: Some(vec!["cancel".into()]),
            tools_removed: None,
            created_at: T0,
            tools: None,
        }
        .into(),
        cut.into(),
        handoff("h0", None, "greeter"),
    ];
    assert_eq!(
        transcript::format_items(&items),
        "user: book me a table\n\
         [function call: find_table({\"party\":2})]\n\
         [function output: table 7 at 8pm]\n\
         [agent handoff: host -> concierge]\n\
         [function error: no such table]\n\
         [agent config: instructions=\"Be brief.\\nDon't guess.\", tools_added=['cancel']]\n\
         assistant: your balance is [interrupted]\n\
         [agent handoff: None -> greeter]"
    );
}

// Catches: a malformed `submit_verdict` payload coerced into a verdict.
#[test]
fn a_malformed_verdict_payload_is_no_verdict() {
    let good = prompts::parse_verdict(r#"{"verdict": "maybe", "reasoning": "unclear"}"#);
    assert_eq!(good, Ok(("maybe".into(), "unclear".into())));
    for bad in [
        r#"{"verdict": "pass", "reasoning": "#,
        r#"{"verdict": "PASS", "reasoning": "ok"}"#,
        r#"{"verdict": true, "reasoning": "ok"}"#,
        r#"{"verdict": "pass"}"#,
        r#"["pass", "ok"]"#,
    ] {
        assert!(prompts::parse_verdict(bad).is_err(), "{bad} must not parse");
    }
}

// Catches: the aggregates written as variations of one another, maybe counted as a pass, or
// an empty evaluation scoring as a success.
#[test]
fn the_aggregates_disagree_correctly_on_a_mixed_set() {
    let verdicts = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    let mixed = prompts::score(&verdicts(&["pass", "pass", "fail", "maybe"])).unwrap();
    assert_eq!(mixed.score, 0.625);
    assert!(!mixed.all_passed && mixed.any_passed && !mixed.majority_passed && !mixed.none_failed);
    let empty = prompts::score(&[]).unwrap();
    assert_eq!(empty.score, 0.0);
    assert!(empty.all_passed && !empty.any_passed && empty.majority_passed && empty.none_failed);
    assert!(prompts::score(&verdicts(&["PASS"])).is_err());
}

struct Registry;

impl RefRegistry for Registry {
    fn release(&self, _id: u64) {}
}

/// Answers every request with `arguments` as its `submit_verdict` call, once `barrier` lets
/// every concurrent request through; `None` answers in prose.
struct Verdicts {
    arguments: Option<&'static str>,
    barrier: Barrier,
    prompts: Mutex<Vec<String>>,
}

struct Chunks(Vec<LlmChunk>);

impl StreamOutput<LlmChunk> for Chunks {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move { Ok(self.0.pop()) })
    }

    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for Chunks {}

impl Llm for Verdicts {
    fn label(&self) -> String {
        "verdicts".into()
    }

    fn chat<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            let ChatItem::Message(prompt) = &req.chat_ctx.items[1] else { unreachable!() };
            self.prompts.lock().push(prompt.raw_text_content().unwrap_or_default());
            self.barrier.wait().await;
            let tool_calls = self.arguments.map_or_else(Vec::new, |arguments| {
                vec![FunctionToolCall {
                    name: "submit_verdict".into(),
                    arguments: arguments.into(),
                    call_id: "v".into(),
                    extra: None,
                }]
            });
            let delta =
                ChoiceDelta { content: Some("ok".into()), tool_calls, ..Default::default() };
            let chunk = LlmChunk { id: "c".into(), delta: Some(delta), ..Default::default() };
            Ok(Box::new(Chunks(vec![chunk])) as LlmStream)
        })
    }
}

/// The application's evaluator: a fixed judgment, or a failure.
struct Host(Result<Judgment, &'static str>);

impl EvaluationHost for Host {
    fn evaluate<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _evaluator: &'a HostRef,
        _context: &'a lk_agents_abi::invoke::CallContext,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Judgment, HostError>> {
        Box::pin(async move { self.0.clone().map_err(HostError::new) })
    }
}

fn judges(arguments: Option<&'static str>, requests: usize, host: Host) -> (Judges, Arc<Verdicts>) {
    let llm = Arc::new(Verdicts {
        arguments,
        barrier: Barrier::new(requests),
        prompts: Mutex::default(),
    });
    let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
    let runner = Judges::new(
        &ProcessEnv::current_runtime(),
        Some(llm.clone()),
        HostRef::new(0, &registry),
        Arc::new(host),
    );
    (runner, llm)
}

fn built_in(name: &str) -> JudgeSpec {
    JudgeSpec::BuiltIn { name: name.into() }
}

fn host_judge(name: &str) -> JudgeSpec {
    let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
    JudgeSpec::Host { name: name.into(), evaluator: HostRef::new(1, &registry) }
}

// Catches: judges run one after another (the barrier holds both model requests until both are
// open), outcomes reordered by completion, or the handoff judge calling the model without a
// handoff to judge.
#[tokio::test]
async fn judges_run_concurrently_and_answer_in_judge_order() {
    let booked =
        Judgment { verdict: "maybe".into(), reasoning: "pending".into(), instructions: "".into() };
    let (runner, llm) =
        judges(Some(r#"{"verdict":"pass","reasoning":"grounded"}"#), 2, Host(Ok(booked.clone())));
    let chat_ctx = vec![msg("m1", Role::User, "hi"), handoff("h0", None, "greeter")];
    let judges =
        vec![built_in("accuracy"), host_judge("booking"), built_in("handoff"), built_in("safety")];
    let outcomes = tokio::time::timeout(
        Duration::from_secs(5),
        runner.evaluate(
            judges,
            chat_ctx,
            Some(vec![msg("r1", Role::Assistant, "hello")]),
            Default::default(),
            lk_agents_abi::telemetry::Cx::NONE,
        ),
    )
    .await
    .expect("the two model requests run concurrently")
    .unwrap();
    let names: Vec<&str> = outcomes
        .iter()
        .map(|o| match o {
            JudgeOutcome::Judged { name, .. } | JudgeOutcome::Failed { name, .. } => name.as_str(),
        })
        .collect();
    assert_eq!(names, ["accuracy", "booking", "handoff", "safety"]);
    assert_eq!(outcomes[1], JudgeOutcome::Judged { name: "booking".into(), judgment: booked });
    let JudgeOutcome::Judged { judgment, .. } = &outcomes[2] else { panic!("{outcomes:?}") };
    assert_eq!(judgment.reasoning, prompts::NO_HANDOFFS);
    let prompts = llm.prompts.lock();
    assert_eq!(prompts.len(), 2);
    assert!(prompts.iter().all(|p| p.contains("Reference:\nassistant: hello")), "{prompts:?}");
}

// Catches: a judge whose model answered in prose, or whose evaluator raised or returned an
// unknown verdict, reported as a judgment.
#[tokio::test]
async fn a_judge_without_a_usable_verdict_fails_by_name() {
    let shouted =
        Judgment { verdict: "PASS".into(), reasoning: "".into(), instructions: "".into() };
    for (host, model) in [(Host(Err("unreachable")), None), (Host(Ok(shouted)), None)] {
        let (runner, _) = judges(model, 1, host);
        let outcomes = runner
            .evaluate(
                vec![built_in("safety"), host_judge("booking")],
                vec![],
                None,
                Default::default(),
                lk_agents_abi::telemetry::Cx::NONE,
            )
            .await
            .unwrap();
        assert!(outcomes.iter().all(|o| matches!(o, JudgeOutcome::Failed { .. })), "{outcomes:?}");
    }
}

#[derive(Default)]
struct Spans(Mutex<Vec<lk_agents_abi::telemetry::record::SpanRecord>>);

impl lk_agents_abi::telemetry::record::RecordSink for Spans {
    fn record(&self, record: lk_agents_abi::telemetry::record::Record) {
        if let lk_agents_abi::telemetry::record::Record::Span(span) = record {
            self.0.lock().push(span);
        }
    }
}

// Catches: judges sharing one span or running outside the caller's trace, a judge model request
// not nested under its judge, or a failed judge recorded as a success.
#[tokio::test]
async fn each_judge_is_one_span_under_the_caller() {
    use lk_agents_abi::telemetry::record::{SpanContext, Status};
    let spans = Arc::new(Spans::default());
    let env = ProcessEnv {
        tracer: lk_agents_net::telemetry::Tracer::new(
            spans.clone(),
            Arc::new(lk_agents_abi::VirtualClock::new()),
            3,
        ),
        ..ProcessEnv::current_runtime()
    };
    let llm = Arc::new(Verdicts {
        arguments: Some(r#"{"verdict":"pass","reasoning":"grounded"}"#),
        barrier: Barrier::new(1),
        prompts: Mutex::default(),
    });
    let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
    let runner =
        Judges::new(&env, Some(llm), HostRef::new(0, &registry), Arc::new(Host(Err("down"))));
    let caller = SpanContext { trace_id: 5, span_id: 6, sampled: true };
    runner
        .evaluate(
            vec![built_in("safety"), host_judge("booking")],
            vec![],
            None,
            Default::default(),
            env.tracer.root().continuing(Some(caller)),
        )
        .await
        .unwrap();
    let spans = spans.0.lock().clone();
    let judges: Vec<_> = spans.iter().filter(|s| s.name == "judge_evaluation").collect();
    assert_eq!(judges.len(), 2, "{spans:?}");
    assert!(judges.iter().all(|j| j.parent == Some(6) && j.context.trace_id == 5));
    let request = spans.iter().find(|s| s.name == "llm_request").expect("the judge model request");
    let safety = judges.iter().find(|j| j.status == Status::Unset).expect("the built-in judged");
    assert_eq!(request.parent, Some(safety.context.span_id));
    assert!(judges.iter().any(|j| matches!(j.status, Status::Error(_))), "the host judge failed");
}

/// Answers with `judgment` and keeps the span the core made current for it.
struct Seen(Judgment, Mutex<Option<lk_agents_abi::telemetry::record::SpanContext>>);

impl EvaluationHost for Seen {
    fn evaluate<'a>(
        &'a self,
        cx: &lk_agents_abi::telemetry::Cx,
        _evaluator: &'a HostRef,
        _context: &'a lk_agents_abi::invoke::CallContext,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Judgment, HostError>> {
        *self.1.lock() = cx.span();
        Box::pin(async move { Ok(self.0.clone()) })
    }
}

// Catches: an application judge running outside its `judge_evaluation`, or its explanation
// recorded without GenAI message capture.
#[tokio::test]
async fn a_host_judge_runs_in_its_span_and_its_explanation_follows_capture() {
    for capture in [true, false] {
        let spans = Arc::new(Spans::default());
        let tracer = lk_agents_net::telemetry::Tracer::new(
            spans.clone(),
            Arc::new(lk_agents_abi::VirtualClock::new()),
            3,
        );
        tracer.set_capture_content(capture);
        let env = ProcessEnv { tracer, ..ProcessEnv::current_runtime() };
        let judgment = Judgment {
            verdict: "pass".into(),
            reasoning: "private".into(),
            instructions: String::new(),
        };
        let host = Arc::new(Seen(judgment, Mutex::default()));
        let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
        let runner = Judges::new(&env, None, HostRef::new(0, &registry), host.clone());
        runner
            .evaluate(
                vec![host_judge("private")],
                vec![],
                None,
                Default::default(),
                env.tracer.root(),
            )
            .await
            .unwrap();
        let spans = spans.0.lock().clone();
        let judge = spans.iter().find(|s| s.name == "judge_evaluation").expect("{spans:?}");
        assert_eq!(*host.1.lock(), Some(judge.context));
        let explained = judge.attributes.iter().any(|(k, _)| *k == "gen_ai.evaluation.explanation");
        assert_eq!(explained, capture);
    }
}
