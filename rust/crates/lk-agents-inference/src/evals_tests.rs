//! The scoring rules, and judges run on a fake host.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{HostError, RefRegistry};
use parking_lot::Mutex;
use tokio::sync::Barrier;

use super::*;
use lk_agents_net::env::ProcessEnv;

// Catches: the aggregates written as variations of one another, maybe counted as a pass, or
// an empty evaluation scoring as a success.
#[test]
fn the_aggregates_disagree_correctly_on_a_mixed_set() {
    use JudgeVerdict::{Fail, Maybe, Pass};
    let mixed = prompts::score(&[Pass, Pass, Fail, Maybe]);
    assert_eq!(mixed.score, 0.625);
    assert!(!mixed.all_passed && mixed.any_passed && !mixed.majority_passed && !mixed.none_failed);
    let empty = prompts::score(&[]);
    assert_eq!(empty.score, 0.0);
    assert!(empty.all_passed && !empty.any_passed && empty.majority_passed && empty.none_failed);
}

struct Registry;

impl RefRegistry for Registry {
    fn release(&self, _id: u64) {}
}

/// The application's evaluators: each answers once `barrier` lets every concurrent one through,
/// the evaluator `HostRef` 1 with `judgment` and any other with a failure.
struct Host {
    judgment: Judgment,
    barrier: Barrier,
}

impl EvaluationHost for Host {
    fn evaluate<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        evaluator: &'a HostRef,
        _context: &'a lk_agents_abi::invoke::CallContext,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Judgment, HostError>> {
        Box::pin(async move {
            self.barrier.wait().await;
            if evaluator.id() == 1 {
                Ok(self.judgment.clone())
            } else {
                Err(HostError::new("down"))
            }
        })
    }
}

fn booked() -> Judgment {
    Judgment { verdict: JudgeVerdict::Maybe, reasoning: "pending".into(), instructions: "".into() }
}

fn judges(env: &ProcessEnv, requests: usize) -> Judges {
    let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
    let host = Host { judgment: booked(), barrier: Barrier::new(requests) };
    Judges::new(env, None, HostRef::new(0, &registry), Arc::new(host))
}

/// A judge whose evaluator answers (`HostRef` 1), or raises (any other).
fn judge(name: &str, answers: bool) -> JudgeSpec {
    let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
    JudgeSpec { name: name.into(), evaluator: HostRef::new(if answers { 1 } else { 2 }, &registry) }
}

fn name(outcome: &JudgeOutcome) -> &str {
    match outcome {
        JudgeOutcome::Judged { name, .. } | JudgeOutcome::Failed { name, .. } => name,
    }
}

// Catches: judges run one after another (the barrier holds every evaluator until all are open),
// outcomes reordered by completion, or an evaluator that raised reported as a judgment.
#[tokio::test]
async fn judges_run_concurrently_and_answer_in_judge_order() {
    let runner = judges(&ProcessEnv::current_runtime(), 3);
    let outcomes = tokio::time::timeout(
        Duration::from_secs(5),
        runner.evaluate(
            vec![judge("accuracy", false), judge("booking", true), judge("safety", false)],
            Default::default(),
            lk_agents_abi::telemetry::Cx::NONE,
        ),
    )
    .await
    .expect("the evaluators run concurrently");
    assert_eq!(outcomes.iter().map(name).collect::<Vec<_>>(), ["accuracy", "booking", "safety"]);
    assert_eq!(outcomes[1], JudgeOutcome::Judged { name: "booking".into(), judgment: booked() });
    assert_eq!(outcomes[0], JudgeOutcome::Failed { name: "accuracy".into(), error: "down".into() });
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

// Catches: judges sharing one span or running outside the caller's trace, or a failed judge
// recorded as a success.
#[tokio::test]
async fn each_judge_is_one_span_under_the_caller() {
    use lk_agents_abi::telemetry::record::{SpanContext, Status};
    let spans = Arc::new(Spans::default());
    let env = ProcessEnv {
        tracer: lk_agents_telemetry::Tracer::new(
            spans.clone(),
            Arc::new(lk_agents_abi::VirtualClock::new()),
            3,
        ),
        ..ProcessEnv::current_runtime()
    };
    let caller = SpanContext { trace_id: 5, span_id: 6, sampled: true };
    judges(&env, 2)
        .evaluate(
            vec![judge("booking", true), judge("safety", false)],
            Default::default(),
            env.tracer.root().continuing(Some(caller)),
        )
        .await;
    let spans = spans.0.lock().clone();
    let judges: Vec<_> = spans.iter().filter(|s| s.name == "judge_evaluation").collect();
    assert_eq!(judges.len(), 2, "{spans:?}");
    assert!(judges.iter().all(|j| j.parent == Some(6) && j.context.trace_id == 5));
    assert!(judges.iter().any(|j| j.status == Status::Unset), "the booking judge judged");
    assert!(judges.iter().any(|j| matches!(j.status, Status::Error(_))), "the safety judge failed");
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
        let tracer = lk_agents_telemetry::Tracer::new(
            spans.clone(),
            Arc::new(lk_agents_abi::VirtualClock::new()),
            3,
        );
        tracer.set_capture_content(capture);
        let env = ProcessEnv { tracer, ..ProcessEnv::current_runtime() };
        let judgment = Judgment {
            verdict: JudgeVerdict::Pass,
            reasoning: "private".into(),
            instructions: String::new(),
        };
        let host = Arc::new(Seen(judgment, Mutex::default()));
        let registry: Arc<dyn RefRegistry> = Arc::new(Registry);
        let runner = Judges::new(&env, None, HostRef::new(0, &registry), host.clone());
        runner.evaluate(vec![judge("private", true)], Default::default(), env.tracer.root()).await;
        let spans = spans.0.lock().clone();
        let judge = spans.iter().find(|s| s.name == "judge_evaluation").expect("{spans:?}");
        assert_eq!(*host.1.lock(), Some(judge.context));
        let explained = judge.attributes.iter().any(|(k, _)| *k == "gen_ai.evaluation.explanation");
        assert_eq!(explained, capture);
    }
}
