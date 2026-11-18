use super::*;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn poll<F: std::future::Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    future.poll(&mut Context::from_waker(Waker::noop()))
}

fn channel() -> (Arc<Driver>, mpsc::Receiver<Command>) {
    let background = Background::new(
        Arc::default(),
        Arc::new(lk_agents_net::task::TokioSpawner::current()),
        Arc::new(lk_agents_net::clock::RealClock::new()),
    );
    Driver::channel(
        &background,
        super::super::configuration::Configuration::new(
            "https://booking.test",
            &super::super::Project {
                url: String::new(),
                api_key: String::new(),
                api_secret: String::new(),
                endpoints: String::new(),
            },
            None,
            None,
        )
        .unwrap(),
    )
}

fn input() -> DelegationInput {
    DelegationInput {
        arguments: serde_json::json!({"request": "Saturday"}),
        structured: false,
        input_request: None,
    }
}

#[tokio::test]
async fn dropping_a_frontend_wait_keeps_the_admitted_task_message() {
    let (driver, mut commands) = channel();
    let origin = ToolCallRef { agent_id: "desk".into(), call_id: "canonical-call".into() };
    let claim = Arc::new(RemoteResponseClaim);
    let tracked = Arc::downgrade(&claim);
    let mut wait = Box::pin(driver.input(input(), Some(claim), Some(origin.clone())));
    assert!(poll(wait.as_mut()).is_pending());
    drop(wait);
    let command = commands.try_recv().unwrap();
    assert!(tracked.upgrade().is_some(), "the queued command owns its response after caller loss");
    let Message::Input(input) = command.message else { panic!("an ordinary task message") };
    assert_eq!(input.arguments, serde_json::json!({"request": "Saturday"}));
    assert_eq!(command.origin, Some(origin));
    command.done.resolve(Ok(()));
    drop(command.response_claim);
    assert!(tracked.upgrade().is_none());
}

#[tokio::test]
async fn canceling_before_queue_admission_releases_only_that_response_claim() {
    let (driver, commands) = channel();
    let mut waiting = Vec::new();
    for _ in 0..16 {
        let mut wait = Box::pin(driver.input(input(), None, None));
        assert!(poll(wait.as_mut()).is_pending());
        waiting.push(wait);
    }
    assert_eq!(commands.capacity(), 0);
    let claim = Arc::new(RemoteResponseClaim);
    let tracked = Arc::downgrade(&claim);
    let mut not_admitted = Box::pin(driver.input(input(), Some(claim), None));
    assert!(poll(not_admitted.as_mut()).is_pending());
    drop(not_admitted);
    assert!(tracked.upgrade().is_none());
    assert!(driver.is_live(), "the incumbent still owns its other work");
    drop(commands);
}

#[tokio::test]
async fn ending_the_native_driver_settles_every_queued_message_wait() {
    let (driver, commands) = channel();
    let mut first = pin!(driver.input(input(), None, None));
    let mut second = pin!(driver.input(input(), None, None));
    assert!(poll(first.as_mut()).is_pending());
    assert!(poll(second.as_mut()).is_pending());
    drop(commands);
    assert!(matches!(poll(first.as_mut()), Poll::Ready(Err(AgentClientError::InvalidInput(_)))));
    assert!(matches!(poll(second.as_mut()), Poll::Ready(Err(AgentClientError::InvalidInput(_)))));
}

#[tokio::test]
async fn message_admission_waits_for_the_peer_receipt() {
    let (driver, mut commands) = channel();
    let mut wait = pin!(driver.input(input(), None, None));
    assert!(poll(wait.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    assert!(poll(wait.as_mut()).is_pending());
    command.done.resolve(Ok(()));
    assert!(matches!(poll(wait.as_mut()), Poll::Ready(Ok(()))));
}

#[tokio::test]
async fn losing_an_in_flight_message_resolves_its_waiter() {
    let (driver, mut commands) = channel();
    let mut wait = pin!(driver.input(input(), None, None));
    assert!(poll(wait.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    drop(command);
    assert!(matches!(poll(wait.as_mut()), Poll::Ready(Err(AgentClientError::InvalidInput(_)))));
}

#[tokio::test]
async fn resuming_prepared_work_keeps_the_original_request_and_message_identities() {
    let (driver, mut commands) = channel();
    let request: Box<lk_agents_abi::a2a::protocol::SendMessageRequest> = Box::new(
        serde_json::from_value(serde_json::json!({
            "message": {"messageId": "original-message", "role": "ROLE_USER",
                "taskId": "booking", "contextId": "trip", "parts": [{"text": "Saturday"}]}
        }))
        .unwrap(),
    );
    let reference = lk_agents_abi::session::InputRequestRef::RemoteTask {
        remote_id: "booking".into(),
        task: lk_agents_abi::delegation::remote::RemoteTaskRef {
            context_id: "trip".into(),
            task_id: "booking".into(),
        },
        request_id: "original-question".into(),
    };
    let mut wait = pin!(driver.resume(
        "prepared-request".into(),
        request.clone(),
        true,
        Some(reference.clone()),
        None
    ));
    assert!(poll(wait.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    let Message::Resume { request_id, request: saved, streaming, input_request } = command.message
    else {
        panic!("a prepared operation resumes without being rebuilt")
    };
    assert_eq!(request_id, "prepared-request");
    assert_eq!(saved, request);
    assert!(streaming);
    assert_eq!(input_request, Some(reference));
    command.done.resolve(Ok(()));
    assert!(matches!(poll(wait.as_mut()), Poll::Ready(Ok(()))));
}

#[tokio::test]
async fn reconciliation_failure_does_not_fail_already_admitted_work() {
    let (driver, mut commands) = channel();
    let mut wait = pin!(driver.reconcile());
    assert!(poll(wait.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    assert!(matches!(command.message, Message::Reconcile));
    command.done.resolve(Err(AgentClientError::Protocol("observation failed".into())));
    assert_eq!(poll(wait.as_mut()), Poll::Ready(()));
}

#[tokio::test]
async fn reconciliation_wait_ends_when_its_original_observer_dies() {
    let (driver, commands) = channel();
    let mut wait = pin!(driver.reconcile());
    assert!(poll(wait.as_mut()).is_pending());
    drop(commands);
    assert_eq!(poll(wait.as_mut()), Poll::Ready(()));
}

fn task() -> lk_agents_abi::delegation::remote::RemoteTaskRef {
    lk_agents_abi::delegation::remote::RemoteTaskRef {
        task_id: "booking".into(),
        context_id: "trip".into(),
    }
}

#[tokio::test]
async fn overlapping_cancellations_share_the_same_refusal_without_another_command() {
    let (driver, mut commands) = channel();
    let first = driver.cancellation(task(), None);
    let command = commands.recv().await.unwrap();
    let second = driver.cancellation(task(), Some("not-yet-usable-refusal".into()));
    assert!(Arc::ptr_eq(&first, &second));
    assert!(matches!(command.message, Message::Cancel { task: reference, retry_after: None }
        if reference == task()));
    assert!(commands.try_recv().is_err(), "the second waiter created no transport work");
    let mut wait = pin!(second.wait());
    assert!(poll(wait.as_mut()).is_pending(), "enqueue is not confirmed cancellation");
    command.done.resolve(Err(AgentClientError::Protocol("A2A error -32002: protected".into())));
    assert!(matches!(first.wait().await, Err(AgentClientError::Protocol(error))
        if error == "A2A error -32002: protected"));
    assert!(matches!(poll(wait.as_mut()), Poll::Ready(Err(AgentClientError::Protocol(_)))));
    assert!(driver.is_live(), "refusal leaves the original observer alive");
    drop(commands);
    driver.close().await;
}

#[tokio::test]
async fn losing_every_cancel_waiter_keeps_the_concrete_enqueued_attempt_eligible() {
    let (driver, mut commands) = channel();
    let first = driver.cancellation(task(), None);
    let identity = Arc::downgrade(&first);
    drop(first);
    let command = commands.recv().await.unwrap();
    let duplicate = driver.cancellation(task(), None);
    assert!(
        Arc::ptr_eq(&identity.upgrade().unwrap(), &duplicate),
        "the queued command retains its attempt independently of application waiters"
    );
    command.done.resolve(Ok(()));
    duplicate.wait().await.unwrap();
    drop(commands);
    driver.close().await;
}

#[tokio::test]
async fn driver_death_settles_every_waiter_of_an_in_flight_cancel_attempt() {
    let (driver, mut commands) = channel();
    let first = driver.cancellation(task(), None);
    let command = commands.recv().await.unwrap();
    let second = driver.cancellation(task(), None);
    assert!(Arc::ptr_eq(&first, &second));
    drop(commands);
    drop(command);
    assert!(matches!(first.wait().await, Err(AgentClientError::InvalidInput(_))));
    assert!(matches!(second.wait().await, Err(AgentClientError::InvalidInput(_))));
    driver.close().await;
}

#[tokio::test]
async fn only_a_later_explicit_cancel_gets_a_new_attempt_after_the_previous_result() {
    let (driver, mut commands) = channel();
    let first = driver.cancellation(task(), None);
    let command = commands.recv().await.unwrap();
    command.done.resolve(Err(AgentClientError::Protocol("the definite refusal".into())));
    assert!(!first.pending());
    let later = driver.cancellation(task(), Some("refused-request".into()));
    assert!(!Arc::ptr_eq(&first, &later));
    let command = commands.recv().await.unwrap();
    assert!(matches!(command.message, Message::Cancel { retry_after: Some(id), .. }
        if id == "refused-request"));
    command.done.resolve(Ok(()));
    later.wait().await.unwrap();
    assert!(
        matches!(first.wait().await, Err(AgentClientError::Protocol(_))),
        "a later attempt never rewrites the old immutable result"
    );
    drop(commands);
    driver.close().await;
}

#[tokio::test]
async fn owner_death_releases_cancel_enqueue_waiting_for_queue_capacity() {
    let (driver, commands) = channel();
    let mut messages = Vec::new();
    for _ in 0..16 {
        let mut wait = Box::pin(driver.input(input(), None, None));
        assert!(poll(wait.as_mut()).is_pending());
        messages.push(wait);
    }
    let cancel = driver.cancellation(task(), None);
    drop(commands);
    driver.close().await;
    assert!(matches!(cancel.wait().await, Err(AgentClientError::InvalidInput(_))));
    assert!(driver.enqueues.tasks.is_empty(), "the driver joins its own enqueue work");
}
