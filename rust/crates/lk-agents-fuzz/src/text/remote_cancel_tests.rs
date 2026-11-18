//! Bound task controls use the real native task owner, client, journal and SQLite binding.

use super::*;
use futures_util::FutureExt as _;
use lk_agents_abi::a2a::client::AgentClientError;
use lk_agents_abi::a2a::remote::{DelegationInput, RemoteEvent, RemoteInteraction as _};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::session::remote::{RemoteAgentView as _, RemoteTaskView as _};
use lk_agents_runtime::delegation::remote::RemoteInvocation;
use lk_agents_runtime::net::signal;

struct WaitingPresenter(Mutex<Option<signal::Producer<Option<()>>>>);

impl lk_agents_abi::a2a::remote::DelegationHost for WaitingPresenter {
    fn progress<'a>(
        &'a self,
        _: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::invoke::CallContext,
        _: Option<String>,
        _: serde_json::Map<String, Value>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }

    fn delegated<'a>(
        &'a self,
        _: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::invoke::CallContext,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }

    fn question<'a>(
        &'a self,
        _: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::invoke::CallContext,
        _: String,
        _: bool,
        _: Option<lk_agents_abi::session::InputRequestRef>,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move {
            if let Some(ready) = self.0.lock().take() {
                ready.resolve(());
            }
            Ok(())
        })
    }

    fn ask<'a>(
        &'a self,
        _: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::invoke::CallContext,
        _: String,
        _: bool,
        _: Option<lk_agents_abi::session::InputRequestRef>,
    ) -> BoxFuture<'a, Result<DelegationInput, HostError>> {
        Box::pin(async { panic!("remote questions return through the native task driver") })
    }
}

fn invocation(
    env: &Env,
    owner: Arc<dyn SessionCommands>,
    ready: signal::Producer<Option<()>>,
    options: lk_agents_abi::ApiConnectOptions,
) -> RemoteInvocation {
    RemoteInvocation {
        arguments: json!({"request":"Book a seat"}),
        structured: false,
        history: Vec::new(),
        shared: None,
        live: true,
        host: Arc::new(WaitingPresenter(Mutex::new(Some(ready)))),
        context: Default::default(),
        cx: env.tracer.root(),
        owner: Some(owner),
        origin: Some(origin()),
        conn_options: options,
    }
}

#[tokio::test(start_paused = true)]
async fn bound_cancel_waits_for_persisted_cancellation_and_discards_queued_continuations() {
    bound_cancel(CancelReply::Confirmed, true).await;
}

#[tokio::test(start_paused = true)]
async fn a_bound_cancel_reconciles_a_committed_lost_ack_without_another_control_write() {
    bound_cancel(CancelReply::Lost, false).await;
}

#[tokio::test(start_paused = true)]
async fn a_refused_bound_cancel_preserves_the_live_task_and_does_not_retry_its_control() {
    bound_cancel(CancelReply::Refused, false).await;
}

async fn bound_cancel(cancel_reply: CancelReply, queue_continuation: bool) {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(83, false, clock.clone()).await;
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (owner, updates, binding, view) = owner(&env, store.clone()).await;
    let (cancel_started, cancellation_started) = signal::once();
    let (release_cancel, cancellation_released) = signal::once();
    let service = Arc::new(Service {
        cancel_reply,
        gated_cancel: Mutex::new(
            queue_continuation
                .then_some(CancelGate { started: cancel_started, release: cancellation_released }),
        ),
        ..Default::default()
    });
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        Some("current-configured-credential".into()),
        Default::default(),
        service.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    let interaction = interaction();
    let (ready, question) = signal::once();
    let running = interaction.run_with_client(
        &env.process,
        client,
        invocation(&env, owner.clone(), ready, Default::default()),
    );
    tokio::pin!(running);
    tokio::select! {
        biased;
        ready = question.wait() => ready.unwrap(),
        result = &mut running => panic!("the original call must be waiting for its task: {result:?}"),
    }
    let task =
        view.remote_agents().remove(ENDPOINT).unwrap().tasks().unwrap().remove("booking").unwrap();
    let detached = SessionState::new(Some(view.snapshot())).unwrap();
    let detached_task = detached
        .remote_agents()
        .remove(ENDPOINT)
        .unwrap()
        .tasks()
        .unwrap()
        .remove("booking")
        .unwrap();
    assert!(matches!(detached_task.cancel().await, Err(AgentClientError::InvalidInput(_))));
    assert_eq!(*service.cancellations.lock(), 0, "snapshot inspection cannot acquire authority");
    let cancellation = task.cancel();
    tokio::pin!(cancellation);
    let duplicate_driver = if queue_continuation {
        tokio::select! {
            biased;
            started = cancellation_started.wait() => started.unwrap(),
            result = &mut cancellation => panic!("a sent control is not observed cancellation: {result:?}"),
        }
        assert!(task.cancel_requested().unwrap());
        assert_ne!(task.task().unwrap().status.state, a2a::TaskState::Canceled);
        Some(
            owner
                .remote_driver(
                    ENDPOINT,
                    ENDPOINT,
                    &RemoteTaskRef { task_id: "booking".into(), context_id: "trip".into() },
                )
                .await
                .unwrap()
                .unwrap(),
        )
    } else {
        None
    };
    let mut duplicate = queue_continuation.then(|| Box::pin(task.cancel()));
    if let Some(wait) = duplicate.as_mut() {
        assert!(wait.as_mut().now_or_never().is_none());
        // This owner receipt follows the second cancellation's admission command. Polling
        // its waiter again transfers that admitted control into the blocked native driver.
        owner.checkpoint().await.unwrap();
        assert!(
            wait.as_mut().now_or_never().is_none(),
            "the second native cancel is queued while the first peer receipt is held"
        );
    }
    if let Some(driver) = &duplicate_driver {
        assert!(
            driver
                .input(
                    DelegationInput {
                        arguments: json!({"request":"Actually Saturday"}),
                        structured: false,
                        input_request: None,
                    },
                    None,
                    Some(origin())
                )
                .now_or_never()
                .is_none()
        );
    }
    release_cancel.resolve(());
    let outcome = cancellation.await;
    if let Some(duplicate) = duplicate {
        duplicate.await.expect("confirmed cancellation settles the queued same-task control");
    }
    let remote = view.remote_agents().remove(ENDPOINT).unwrap();
    assert!(task.cancel_requested().unwrap());
    match cancel_reply {
        CancelReply::Confirmed | CancelReply::Lost => {
            outcome.unwrap();
            assert_eq!(task.task().unwrap().status.state, a2a::TaskState::Canceled);
            assert!(matches!(
                running.await.unwrap(),
                RemoteEvent::Outcome { outcome: ToolOutcome::Cancelled }
            ));
            task.cancel().await.expect("the active owner already observed cancellation");
        }
        CancelReply::Refused => {
            assert!(outcome.is_err());
            assert_eq!(task.task().unwrap().status.state, a2a::TaskState::InputRequired);
            assert!(task.cancel().await.is_err(), "a read cannot turn refusal into success");
            assert!(remote.events().unwrap().iter().any(|event| matches!(&event.payload,
                lk_agents_abi::delegation::remote::A2AJournalPayload::Error(
                    lk_agents_abi::delegation::remote::RemoteFailure::Rpc { error }
                ) if error.code == a2a::error_code::TASK_NOT_CANCELABLE
                    && error.data == Some(json!({"stage":"booking"}))
            )));
            interaction.cancelled().unwrap();
            assert!(matches!(
                running.await.unwrap(),
                RemoteEvent::Outcome { outcome: ToolOutcome::Cancelled }
            ));
        }
    }
    interaction.settled().await;
    assert_eq!(*service.cancellations.lock(), 1, "the saved control was never sent twice");
    assert_eq!(service.messages.lock().len(), 1, "a terminal task cannot send queued corrections");
    assert!(service.bookings.lock().is_empty());
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    assert!(matches!(task.cancel().await, Err(AgentClientError::InvalidInput(_))));
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}

#[tokio::test(start_paused = true)]
async fn an_active_binding_cannot_cancel_a_saved_task_without_a_configured_live_driver() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(89, false, clock.clone()).await;
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (owner, updates, binding, view) = owner(&env, store.clone()).await;
    let service = Arc::new(Service::default());
    assert!(matches!(
        resume(&env, service.clone(), owner).await.unwrap(),
        RemoteEvent::InputRequired { .. }
    ));
    let task =
        view.remote_agents().remove(ENDPOINT).unwrap().tasks().unwrap().remove("booking").unwrap();
    assert!(matches!(task.cancel().await, Err(AgentClientError::InvalidInput(_))));
    assert!(
        !task.cancel_requested().unwrap(),
        "unavailable authority cannot record dispatch intent"
    );
    assert_eq!(*service.cancellations.lock(), 0);
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}

#[tokio::test(start_paused = true)]
async fn bound_cancel_settles_when_its_owner_ends_during_lost_ack_reconciliation() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(97, false, clock.clone()).await;
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (owner, updates, binding, view) = owner(&env, store.clone()).await;
    let (started, read_started) = signal::once();
    let service = Arc::new(Service {
        cancel_reply: CancelReply::Lost,
        stalled_reconciliation: Mutex::new(Some(started)),
        ..Default::default()
    });
    let options = lk_agents_abi::ApiConnectOptions { read_timeout: None, ..Default::default() };
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        None,
        options,
        service.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    let interaction = interaction();
    let (ready, question) = signal::once();
    let running =
        interaction.run_with_client(&env.process, client, invocation(&env, owner, ready, options));
    tokio::pin!(running);
    tokio::select! {
        biased;
        ready = question.wait() => ready.unwrap(),
        result = &mut running => panic!("the original call must remain live: {result:?}"),
    }
    let task =
        view.remote_agents().remove(ENDPOINT).unwrap().tasks().unwrap().remove("booking").unwrap();
    let cancellation = task.cancel();
    tokio::pin!(cancellation);
    tokio::select! {
        biased;
        started = read_started.wait() => started.unwrap(),
        result = &mut cancellation => panic!("the reconciliation read is still pending: {result:?}"),
    }
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    assert!(
        matches!(cancellation.now_or_never(), Some(Err(AgentClientError::InvalidInput(_)))),
        "ending the command receiver settles the public control wait immediately"
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    interaction.cancelled().unwrap();
    assert!(running.await.is_err());
    interaction.settled().await;
    assert_eq!(*service.cancellations.lock(), 1);
    assert_eq!(service.messages.lock().len(), 1);
    assert!(
        env.process
            .tasks
            .wait_empty(&*clock, Some(clock.now().saturating_add(Duration::from_secs(15))))
            .await
    );
    assert!(!database.held(PARTITION_KEY).await);
}
