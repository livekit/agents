#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's paused clock directly"
)]
use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::ipc::ipc_channel::memory;
use crate::ipc::wire::Token;

fn issue() -> (Arc<WorkerIssue>, Channel<ToChild, ToParent>) {
    let rt = crate::test_rt();
    let (parent, child) = memory(&rt);
    (Arc::new(WorkerIssue::new(Arc::new(child), rt.clock.clone())), parent)
}

async fn asked(parent: &Channel<ToChild, ToParent>) -> (u64, String) {
    match parent.recv().await.unwrap() {
        ToParent::CreateJobToken { request_id, claims } => (request_id, claims),
        other => panic!("expected a job token request, got {other:?}"),
    }
}

// Two requests, and the worker answers the second one first with a refusal. Catches an answer
// that goes to the wrong request, a refusal that is read as a token or loses its reason, and a
// token that loses its expiry.
#[tokio::test(start_paused = true)]
async fn the_workers_answers_reach_their_requests() {
    let (issue, parent) = issue();
    let (first, second) = (issue.clone(), issue.clone());
    let first = tokio::spawn(async move { first.issue("a").await });
    let second = tokio::spawn(async move { second.issue("b").await });
    let (one, two) = (asked(&parent).await, asked(&parent).await);
    let ((a, _), (b, _)) = if one.1 == "a" { (one, two) } else { (two, one) };
    issue.answer(b, JobTokenAnswer::Refused("not this one".into()));
    issue.answer(a, JobTokenAnswer::Issued { token: Token("signed".into()), expires_at: 42 });
    let issued = first.await.unwrap().unwrap();
    assert_eq!((issued.token.as_str(), issued.expires_at), ("signed", 42));
    let error = second.await.unwrap().unwrap_err();
    assert!(error.message.contains("not this one"), "{error}");
}

// The worker never answers, and then answers late. Catches a job that waits forever, or a request
// that stays open after the timeout. The late answer must be dropped.
#[tokio::test(start_paused = true)]
async fn an_unanswered_request_times_out_after_the_workers_own_limit() {
    let (issue, parent) = issue();
    let started = tokio::time::Instant::now();
    let asking = issue.clone();
    let asking = tokio::spawn(async move { asking.issue("a").await });
    let (request_id, _) = asked(&parent).await;
    let error = asking.await.unwrap().unwrap_err();
    assert!(error.message.contains("did not answer"), "{error}");
    assert_eq!(started.elapsed(), ANSWER_TIMEOUT);
    assert!(ANSWER_TIMEOUT > crate::JOB_TOKEN_TIMEOUT);
    assert!(issue.pending.lock().is_empty());
    issue.answer(request_id, JobTokenAnswer::Refused("late".into()));
    tokio::time::sleep(Duration::from_millis(1)).await;
}
