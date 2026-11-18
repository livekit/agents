use super::*;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

fn poll<F: std::future::Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    future.poll(&mut Context::from_waker(Waker::noop()))
}

fn input() -> DelegationInput {
    DelegationInput {
        arguments: serde_json::json!({"request": "Saturday"}),
        structured: false,
        input_request: None,
    }
}

#[test]
fn dropping_a_frontend_wait_keeps_the_admitted_task_message() {
    let (driver, mut commands) = Driver::channel();
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

#[test]
fn canceling_before_queue_admission_releases_only_that_response_claim() {
    let (driver, commands) = Driver::channel();
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

#[test]
fn ending_the_native_driver_settles_every_queued_message_wait() {
    let (driver, commands) = Driver::channel();
    let mut first = pin!(driver.input(input(), None, None));
    let mut second = pin!(driver.input(input(), None, None));
    assert!(poll(first.as_mut()).is_pending());
    assert!(poll(second.as_mut()).is_pending());
    drop(commands);
    assert!(matches!(poll(first.as_mut()), Poll::Ready(Err(AgentClientError::InvalidInput(_)))));
    assert!(matches!(poll(second.as_mut()), Poll::Ready(Err(AgentClientError::InvalidInput(_)))));
}

#[test]
fn message_admission_waits_for_the_peer_receipt() {
    let (driver, mut commands) = Driver::channel();
    let mut wait = pin!(driver.input(input(), None, None));
    assert!(poll(wait.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    assert!(poll(wait.as_mut()).is_pending());
    command.done.resolve(Ok(()));
    assert!(matches!(poll(wait.as_mut()), Poll::Ready(Ok(()))));
}

#[test]
fn losing_an_in_flight_message_resolves_its_waiter() {
    let (driver, mut commands) = Driver::channel();
    let mut wait = pin!(driver.input(input(), None, None));
    assert!(poll(wait.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    drop(command);
    assert!(matches!(poll(wait.as_mut()), Poll::Ready(Err(AgentClientError::InvalidInput(_)))));
}

#[test]
fn resuming_prepared_work_keeps_the_original_request_and_message_identities() {
    let (driver, mut commands) = Driver::channel();
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

#[test]
fn reconciliation_failure_does_not_fail_already_admitted_work() {
    let (driver, mut commands) = Driver::channel();
    let mut wait = pin!(driver.reconcile());
    assert!(poll(wait.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    assert!(matches!(command.message, Message::Reconcile));
    command.done.resolve(Err(AgentClientError::Protocol("observation failed".into())));
    assert_eq!(poll(wait.as_mut()), Poll::Ready(()));
}

#[test]
fn reconciliation_wait_ends_when_its_original_observer_dies() {
    let (driver, commands) = Driver::channel();
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

#[test]
fn cancellation_waits_for_its_owner_and_preserves_a_refusal() {
    let (driver, mut commands) = Driver::channel();
    let mut wait = pin!(driver.cancel(task()));
    assert!(poll(wait.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    assert!(matches!(command.message, Message::Cancel(reference) if reference == task()));
    assert!(poll(wait.as_mut()).is_pending(), "queue admission does not confirm cancellation");
    command.done.resolve(Err(AgentClientError::Protocol("the peer refused cancellation".into())));
    assert!(matches!(poll(wait.as_mut()), Poll::Ready(Err(AgentClientError::Protocol(_)))));
    assert!(driver.is_live(), "a refused control does not end the original observer");
}

#[test]
fn dropping_a_cancel_wait_does_not_retract_its_admitted_control() {
    let (driver, mut commands) = Driver::channel();
    let mut wait = Box::pin(driver.cancel(task()));
    assert!(poll(wait.as_mut()).is_pending());
    drop(wait);
    let command = commands.try_recv().unwrap();
    assert!(matches!(command.message, Message::Cancel(reference) if reference == task()));
    command.done.resolve(Ok(()));
}

#[test]
fn driver_death_settles_queued_and_in_flight_cancellation_waits() {
    let (driver, mut commands) = Driver::channel();
    let mut in_flight = pin!(driver.cancel(task()));
    let mut queued = pin!(driver.cancel(task()));
    assert!(poll(in_flight.as_mut()).is_pending());
    let command = commands.try_recv().unwrap();
    assert!(poll(queued.as_mut()).is_pending());
    drop(commands);
    assert!(matches!(poll(queued.as_mut()), Poll::Ready(Err(AgentClientError::InvalidInput(_)))));
    drop(command);
    assert!(matches!(
        poll(in_flight.as_mut()),
        Poll::Ready(Err(AgentClientError::InvalidInput(_)))
    ));
}
