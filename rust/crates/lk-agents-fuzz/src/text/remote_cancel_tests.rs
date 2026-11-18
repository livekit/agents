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
        _: bool,
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

pub(super) fn invocation(
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
        configured_token: None,
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
async fn a_later_explicit_bound_cancel_can_retry_a_definitely_refused_control() {
    bound_cancel(CancelReply::Refused, false).await;
}

#[tokio::test(start_paused = true)]
async fn a_cancel_answered_by_completion_delivers_the_result_and_keeps_the_intent() {
    bound_cancel(CancelReply::Completed, false).await;
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
    let mut request = invocation(&env, owner.clone(), ready, Default::default());
    request.configured_token = Some("current-configured-credential".into());
    let running = interaction.run_with_client(&env.process, client, request);
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
    assert!(matches!(
        detached_task.cancel(&env.process).await,
        Err(AgentClientError::InvalidInput(_))
    ));
    assert_eq!(*service.cancellations.lock(), 0, "snapshot inspection cannot acquire authority");
    let cancellation = task.cancel(&env.process);
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
    let mut duplicate = queue_continuation.then(|| Box::pin(task.cancel(&env.process)));
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
            task.cancel(&env.process)
                .await
                .expect("the active owner already observed cancellation");
        }
        CancelReply::Refused => {
            assert!(outcome.is_err());
            assert_eq!(task.task().unwrap().status.state, a2a::TaskState::InputRequired);
            assert!(
                task.cancel(&env.process).await.is_err(),
                "another definite refusal remains a refusal"
            );
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
        CancelReply::Completed => {
            assert!(outcome.is_err(), "completion is never reported as a confirmed cancel");
            assert_eq!(task.task().unwrap().status.state, a2a::TaskState::Completed);
            let RemoteEvent::Outcome { outcome: ToolOutcome::Value { text, .. } } =
                running.await.unwrap()
            else {
                panic!("the original call receives the real late result")
            };
            assert!(text.contains("Booked Friday"), "{text}");
        }
    }
    interaction.settled().await;
    assert_eq!(
        *service.cancellations.lock(),
        if matches!(cancel_reply, CancelReply::Refused) { 2 } else { 1 },
        "only a later explicit intent after a definite refusal can send a fresh control"
    );
    assert_eq!(service.messages.lock().len(), 1, "a terminal task cannot send queued corrections");
    assert!(service.bookings.lock().is_empty());
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    assert!(matches!(task.cancel(&env.process).await, Err(AgentClientError::InvalidInput(_))));
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
    assert!(matches!(task.cancel(&env.process).await, Err(AgentClientError::InvalidInput(_))));
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
    let cancellation = task.cancel(&env.process);
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

static CONTROL_CREDENTIALS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
thread_local! {
    static CONTROL_HTTP: std::cell::RefCell<Option<Arc<dyn HttpExchangePort>>> =
        const { std::cell::RefCell::new(None) };
}

fn control_executor(
    env: lk_agents_runtime::net::env::ProcessEnv,
    bound: Arc<dyn lk_agents_abi::session::control::BoundSessionControl>,
    target: lk_agents_abi::session::control::RemoteControlRef,
    evidence: Option<lk_agents_abi::delegation::remote::RemoteCancellationEvidence>,
) -> Arc<dyn lk_agents_abi::session::control::RemoteCancellation> {
    let http = CONTROL_HTTP.with(|http| http.borrow().clone()).unwrap();
    lk_agents_runtime::delegation::remote::control::cancel_with_transport(
        env, bound, target, evidence, http,
    )
}

const CONTROL_KEY: &str = "worker-key";
const CONTROL_SECRET: &str = "worker-secret-not-saved";

fn control_credentials() -> lk_agents_runtime::net::env::ServerSettings {
    lk_agents_runtime::net::env::serve_settings(vec![
        ("LIVEKIT_URL", ENDPOINT.into()),
        ("LIVEKIT_API_KEY", CONTROL_KEY.into()),
        ("LIVEKIT_API_SECRET", CONTROL_SECRET.into()),
    ])
}

fn control_environment(clock: Arc<dyn Clock>, service: Arc<Service>) -> Arc<Env> {
    let mut env = environment(clock);
    Arc::get_mut(&mut env).unwrap().process.remote_control = Some(control_executor);
    CONTROL_HTTP.with(|http| *http.borrow_mut() = Some(service));
    env
}

async fn verified_delegation(
    env: &Arc<Env>,
    database: &Store,
    service: Arc<Service>,
    owner: Arc<dyn SessionCommands>,
) {
    use lk_agents_abi::a2a::remote::RemoteConfiguration as _;
    let token = lk_agents_runtime::delegation::remote::RemoteInteraction::_core_remote_token(
        ENDPOINT.into(),
        ENDPOINT.into(),
        CONTROL_KEY.into(),
        CONTROL_SECRET.into(),
        None,
        Some(database.thread_id.clone()),
    )
    .unwrap()
    .unwrap();
    let interaction = lk_agents_runtime::delegation::remote::RemoteInteraction::new(
        ENDPOINT.into(),
        call(),
        ENDPOINT.into(),
        CONTROL_KEY.into(),
        CONTROL_SECRET.into(),
        None,
    )
    .unwrap();
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        Some(token.clone()),
        Default::default(),
        service,
        (env.process.spawner)().unwrap(),
        env.clock.clone(),
    )
    .unwrap();
    let (ready, _) = signal::once();
    let mut request = invocation(env, owner, ready, Default::default());
    request.live = false;
    request.shared = Some((database.thread_id.clone(), Some(token)));
    assert!(matches!(
        interaction.run_with_client(&env.process, client, request).await.unwrap(),
        RemoteEvent::InputRequired { .. }
    ));
    interaction.settled().await;
}

#[tokio::test(start_paused = true)]
async fn a_bound_idle_control_uses_current_worker_credentials_after_restart_without_replaying_work()
{
    let _serial = CONTROL_CREDENTIALS.lock().await;
    let _credentials = control_credentials();
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let database = Store::open(113, false, clock.clone()).await;
    let service =
        Arc::new(Service { context_id: Some(database.thread_id.clone()), ..Default::default() });
    let env = control_environment(clock.clone(), service.clone());
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (first, updates, binding, view) = owner(&env, store.clone()).await;
    verified_delegation(&env, &database, service.clone(), first).await;
    let before = view.snapshot();
    assert!(!before.contains(CONTROL_SECRET), "no signing secret enters a snapshot");
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    store.close();
    assert!(!database.held(PARTITION_KEY).await);

    let fresh = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let restored = SessionState::new(Some(fresh.load_session_snapshot().await.unwrap())).unwrap();
    let task = restored
        .remote_agents()
        .remove(ENDPOINT)
        .unwrap()
        .tasks()
        .unwrap()
        .remove("booking")
        .unwrap();
    assert!(
        matches!(task.cancel(&env.process).await, Err(AgentClientError::InvalidInput(_))),
        "loading alone installs no control authority"
    );
    assert_eq!(*service.cancellations.lock(), 0);
    fresh
        .declare_binding(
            HostRef::new(0, &env.refs),
            HostRef::new(0, &env.refs),
            Some(restored.clone()),
        )
        .unwrap();
    task.cancel(&env.process).await.unwrap();
    assert_eq!(task.task().unwrap().status.state, a2a::TaskState::Canceled);
    assert!(task.cancel_requested().unwrap());
    assert_eq!(*service.cancellations.lock(), 1);
    assert_eq!(service.messages.lock().len(), 1, "restoration never replays the business send");
    assert_eq!(*service.gets.lock(), 0, "an acknowledged cancellation needs no fallback read");
    assert!(!database.held(PARTITION_KEY).await, "the short fenced owner has released its lease");
    let committed = SessionState::new(Some(fresh.load_session_snapshot().await.unwrap())).unwrap();
    assert_eq!(
        committed
            .remote_agents()
            .remove(ENDPOINT)
            .unwrap()
            .tasks()
            .unwrap()
            .remove("booking")
            .unwrap()
            .task()
            .unwrap()
            .status
            .state,
        a2a::TaskState::Canceled
    );
    fresh.close();
    assert!(matches!(task.cancel(&env.process).await, Err(AgentClientError::InvalidInput(_))));
    CONTROL_HTTP.with(|http| http.borrow_mut().take());
}

#[tokio::test(start_paused = true)]
async fn a_restarted_bound_cancel_reconciles_its_lost_ack_without_another_control_write() {
    restarted_bound_cancel(CancelReply::Lost).await;
}

#[tokio::test(start_paused = true)]
async fn a_restarted_bound_cancel_keeps_a_refusal_until_a_later_explicit_intent() {
    restarted_bound_cancel(CancelReply::Refused).await;
}

/// The bound `task.cancel()` fails (refused, or applied with its reply and read lost), the
/// owner restarts, and a fresh owner's `task.cancel()` continues from the saved journal.
async fn restarted_bound_cancel(reply: CancelReply) {
    let _serial = CONTROL_CREDENTIALS.lock().await;
    let _credentials = control_credentials();
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let database = Store::open(167, false, clock.clone()).await;
    let lost = matches!(reply, CancelReply::Lost);
    let service = Arc::new(Service {
        context_id: Some(database.thread_id.clone()),
        cancel_reply: reply,
        fail_reconciliation: Mutex::new(lost),
        ..Default::default()
    });
    let env = control_environment(clock.clone(), service.clone());
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (first, updates, binding, _) = owner(&env, store.clone()).await;
    verified_delegation(&env, &database, service.clone(), first).await;
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    store.close();

    let mut cancels = Vec::new();
    for _restart in 0..2 {
        let fresh = ContextStore::bound(
            database.backend.clone(),
            database.thread_id.clone(),
            PARTITION_KEY.into(),
            env.tracer.root(),
        )
        .await
        .unwrap();
        let restored =
            SessionState::new(Some(fresh.load_session_snapshot().await.unwrap())).unwrap();
        let task = restored
            .remote_agents()
            .remove(ENDPOINT)
            .unwrap()
            .tasks()
            .unwrap()
            .remove("booking")
            .unwrap();
        fresh
            .declare_binding(
                HostRef::new(0, &env.refs),
                HostRef::new(0, &env.refs),
                Some(restored.clone()),
            )
            .unwrap();
        let result = task.cancel(&env.process).await;
        assert!(task.cancel_requested().unwrap(), "the person's intent survives the restart");
        cancels.push((result, task.task().unwrap().status.state, *service.cancellations.lock()));
        assert!(!database.held(PARTITION_KEY).await, "each short fenced owner releases its lease");
        fresh.close();
        // The peer becomes readable again only after the first owner is gone.
        *service.fail_reconciliation.lock() = false;
    }
    let [(first, first_state, first_writes), (second, second_state, second_writes)] =
        <[_; 2]>::try_from(cancels).unwrap();
    assert!(first.is_err(), "neither a refusal nor an unknown outcome is success");
    assert_eq!(first_state, a2a::TaskState::InputRequired);
    assert_eq!(first_writes, 1);
    if lost {
        second.expect("the fresh owner reconciles the applied cancel");
        assert_eq!(second_state, a2a::TaskState::Canceled);
        assert_eq!(second_writes, 1, "a lost ack is read back, never sent again");
    } else {
        assert!(second.is_err(), "the peer still refuses");
        assert_eq!(second_state, a2a::TaskState::InputRequired);
        assert_eq!(second_writes, 2, "a later explicit intent after a definite refusal retries");
    }
    assert_eq!(service.messages.lock().len(), 1, "restoration never replays the business send");
    CONTROL_HTTP.with(|http| http.borrow_mut().take());
}

#[tokio::test(start_paused = true)]
async fn a_restored_task_uses_the_active_conversation_journal_without_another_fenced_owner() {
    let _serial = CONTROL_CREDENTIALS.lock().await;
    let _credentials = control_credentials();
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let database = Store::open(127, false, clock.clone()).await;
    let service =
        Arc::new(Service { context_id: Some(database.thread_id.clone()), ..Default::default() });
    let env = control_environment(clock.clone(), service.clone());
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (active, updates, binding, view) = owner(&env, store.clone()).await;
    verified_delegation(&env, &database, service.clone(), active.clone()).await;
    let task =
        view.remote_agents().remove(ENDPOINT).unwrap().tasks().unwrap().remove("booking").unwrap();
    assert!(
        active
            .remote_driver(
                ENDPOINT,
                ENDPOINT,
                &RemoteTaskRef {
                    context_id: database.thread_id.clone(),
                    task_id: "booking".into(),
                }
            )
            .await
            .unwrap()
            .is_none(),
        "the original observation ended after asking its question"
    );
    task.cancel(&env.process).await.unwrap();
    assert_eq!(task.task().unwrap().status.state, a2a::TaskState::Canceled);
    assert!(database.held(PARTITION_KEY).await, "the existing execution still owns its lease");
    active.checkpoint().await.unwrap();
    assert_eq!(*service.cancellations.lock(), 1);
    assert_eq!(service.messages.lock().len(), 1);
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
    CONTROL_HTTP.with(|http| http.borrow_mut().take());
}

#[tokio::test(start_paused = true)]
async fn overlapping_refusal_waiters_share_one_attempt_but_a_later_intent_can_retry() {
    control_attempts(CancelReply::Refused, false, 2).await;
}

#[tokio::test(start_paused = true)]
async fn overlapping_unknown_waiters_and_later_intents_reconcile_the_original_control() {
    control_attempts(CancelReply::Lost, true, 1).await;
}

async fn control_attempts(reply: CancelReply, unknown: bool, expected_cancels: u32) {
    let _serial = CONTROL_CREDENTIALS.lock().await;
    let _credentials = control_credentials();
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let database = Store::open(131, false, clock.clone()).await;
    let (read, reading) = signal::once();
    let (release, released) = signal::once();
    let service = Arc::new(Service {
        context_id: Some(database.thread_id.clone()),
        cancel_reply: reply,
        unknown_cancel: unknown,
        gated_reconciliation: Mutex::new(Some(CancelGate { started: read, release: released })),
        ..Default::default()
    });
    let env = control_environment(clock.clone(), service.clone());
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (active, updates, binding, view) = owner(&env, store.clone()).await;
    verified_delegation(&env, &database, service.clone(), active.clone()).await;
    let task =
        view.remote_agents().remove(ENDPOINT).unwrap().tasks().unwrap().remove("booking").unwrap();
    let first = task.cancel(&env.process);
    tokio::pin!(first);
    tokio::select! {
        biased;
        ready = reading.wait() => ready.unwrap(),
        result = &mut first => panic!("the first control still owns its gated reconciliation: {result:?}"),
    }
    assert_eq!(*service.cancellations.lock(), 1);
    let mut overlapping = Box::pin(task.cancel(&env.process));
    assert!(overlapping.as_mut().now_or_never().is_none());
    // The read gate proves that the original refusal/lost reply has already been journaled.
    // Joining at this boundary cannot become a second explicit control write.
    assert_eq!(*service.cancellations.lock(), 1);
    release.resolve(());
    let first = first.await.unwrap_err();
    let overlapping = overlapping.await.unwrap_err();
    assert_eq!(first.to_string(), overlapping.to_string(), "one immutable attempt result");
    assert!(task.cancel(&env.process).await.is_err());
    assert_eq!(*service.cancellations.lock(), expected_cancels);
    assert_eq!(*service.gets.lock(), 2);
    assert_eq!(service.messages.lock().len(), 1);
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
    CONTROL_HTTP.with(|http| http.borrow_mut().take());
}

#[tokio::test(start_paused = true)]
async fn closing_an_idle_bound_store_settles_stalled_reconciliation_and_releases_its_fence() {
    let _serial = CONTROL_CREDENTIALS.lock().await;
    let _credentials = control_credentials();
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let database = Store::open(137, false, clock.clone()).await;
    let (read, reading) = signal::once();
    let service = Arc::new(Service {
        context_id: Some(database.thread_id.clone()),
        cancel_reply: CancelReply::Lost,
        unknown_cancel: true,
        stalled_reconciliation: Mutex::new(Some(read)),
        ..Default::default()
    });
    let env = control_environment(clock.clone(), service.clone());
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (active, updates, binding, view) = owner(&env, store.clone()).await;
    verified_delegation(&env, &database, service.clone(), active).await;
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    let task =
        view.remote_agents().remove(ENDPOINT).unwrap().tasks().unwrap().remove("booking").unwrap();
    let cancellation = task.cancel(&env.process);
    tokio::pin!(cancellation);
    tokio::select! {
        biased;
        ready = reading.wait() => ready.unwrap(),
        result = &mut cancellation => panic!("the peer is holding its read: {result:?}"),
    }
    store.close();
    assert!(cancellation.await.is_err());
    assert_eq!(*service.cancellations.lock(), 1);
    assert_eq!(*service.gets.lock(), 1);
    assert!(
        env.process
            .tasks
            .wait_empty(&*clock, Some(clock.now().saturating_add(Duration::from_secs(15))))
            .await
    );
    assert!(!database.held(PARTITION_KEY).await);
    CONTROL_HTTP.with(|http| http.borrow_mut().take());
}

/// The second activation borrows the real serialized owner through its managed handler port.
/// This fake only models attachment; it makes no control or recovery decisions.
struct AttachedOwner(Arc<dyn SessionCommands>);
impl lk_agents_abi::a2a::turn::HandlerPort for AttachedOwner {
    fn session_commands(
        &self,
    ) -> Result<Option<Arc<dyn SessionCommands>>, lk_agents_abi::storage::StorageError> {
        Ok(Some(self.0.clone()))
    }
    fn checkpoint(&self) -> BoxFuture<'_, bool> {
        Box::pin(async { false })
    }
    fn progress(&self, _: String) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
    fn token(&self, _: String) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

thread_local! {
    static CONTROL_SPAWNER: std::cell::RefCell<Option<Arc<dyn lk_agents_abi::Spawner>>> =
        const { std::cell::RefCell::new(None) };
}

/// Observes scheduling at the injected boundary. In the attached-owner case, the first poll
/// reaches the real CancelRemote receipt wait, proving its command is queued before the FIFO
/// checkpoint barrier. No sleep, fake controller or admission decision is introduced.
struct FirstControlPoll {
    inner: Arc<dyn lk_agents_abi::Spawner>,
    posted: Mutex<Option<signal::Producer<Option<()>>>>,
}
impl lk_agents_abi::Spawner for FirstControlPoll {
    fn launch(
        &self,
        mut task: lk_agents_abi::Launch,
    ) -> lk_agents_abi::OwnedTask<Box<dyn std::any::Any + Send>> {
        let posted = self.posted.lock().take();
        let Some(posted) = posted else { return self.inner.launch(task) };
        let mut work = task.work;
        task.work = Box::pin(async move {
            let mut posted = Some(posted);
            std::future::poll_fn(|cx| {
                let result = std::future::Future::poll(work.as_mut(), cx);
                if let Some(posted) = posted.take() {
                    posted.resolve(());
                }
                result
            })
            .await
        });
        self.inner.launch(task)
    }
}
fn controlled_spawner()
-> Result<Arc<dyn lk_agents_abi::Spawner>, lk_agents_runtime::net::env::ProcessError> {
    CONTROL_SPAWNER.with(|spawner| spawner.borrow().clone()).ok_or_else(|| {
        lk_agents_runtime::net::env::ProcessError::NoRuntime("no control fixture scheduler".into())
    })
}

#[tokio::test(start_paused = true)]
async fn separate_managed_activations_join_one_retained_cancel_before_either_prepare() {
    separate_controls(true, false, CancelReply::Refused).await;
}

#[tokio::test(start_paused = true)]
async fn losing_a_retained_cancel_waiter_keeps_its_attempt_for_another_managed_activation() {
    separate_controls(true, true, CancelReply::Refused).await;
}

#[tokio::test(start_paused = true)]
async fn independent_idle_stores_reconcile_an_earlier_intent_before_a_later_explicit_retry() {
    separate_controls(false, false, CancelReply::Refused).await;
}

#[tokio::test(start_paused = true)]
async fn separate_managed_activations_share_an_unknown_retained_control_until_reconciliation() {
    separate_controls(true, false, CancelReply::Lost).await;
}

async fn separate_controls(managed: bool, lose_first_waiter: bool, reply: CancelReply) {
    let _serial = CONTROL_CREDENTIALS.lock().await;
    let _credentials = control_credentials();
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let database = Store::open(149, false, clock.clone()).await;
    let (started, reading) = signal::once();
    let (release, released) = signal::once();
    let service = Arc::new(Service {
        context_id: Some(database.thread_id.clone()),
        cancel_reply: reply,
        unknown_cancel: matches!(reply, CancelReply::Lost),
        ..Default::default()
    });
    let env = control_environment(clock.clone(), service.clone());
    let first_store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (active, updates, binding, first_view) = owner(&env, first_store.clone()).await;
    verified_delegation(&env, &database, service.clone(), active.clone()).await;
    let initial = first_store.load_session().await.unwrap().unwrap();
    let second_store = if managed {
        ContextStore::turn(
            database.backend.clone(),
            database.thread_id.clone(),
            PARTITION_KEY.into(),
            Vec::new(),
            Some(initial.clone()),
            false,
            env.tracer.root(),
        )
        .await
        .unwrap()
    } else {
        ContextStore::bound(
            database.backend.clone(),
            database.thread_id.clone(),
            PARTITION_KEY.into(),
            env.tracer.root(),
        )
        .await
        .unwrap()
    };
    let attachment = managed.then(|| {
        second_store.attach_session_sink(Arc::new(AttachedOwner(active.clone()))).unwrap()
    });
    let second_view = SessionState::new(Some(serde_json::to_string(&initial).unwrap())).unwrap();
    second_store
        .declare_binding(
            HostRef::new(0, &env.refs),
            HostRef::new(0, &env.refs),
            Some(second_view.clone()),
        )
        .unwrap();
    let mut updates = Some(updates);
    if !managed {
        assert_eq!(
            Box::new(updates.take().unwrap())
                .close(clock.now().saturating_add(Duration::from_secs(15)))
                .await,
            Closed::Clean
        );
        binding.close();
        first_store.finish_session(false).await.unwrap();
    }
    let first_task = first_view
        .remote_agents()
        .remove(ENDPOINT)
        .unwrap()
        .tasks()
        .unwrap()
        .remove("booking")
        .unwrap();
    let second_task = second_view
        .remote_agents()
        .remove(ENDPOINT)
        .unwrap()
        .tasks()
        .unwrap()
        .remove("booking")
        .unwrap();
    let gate = CancelGate { started, release: released };
    if managed {
        *service.gated_card.lock() = Some(gate);
    } else {
        *service.gated_reconciliation.lock() = Some(gate);
    }
    let mut first = Some(Box::pin(first_task.cancel(&env.process)));
    tokio::select! {
        biased;
        started = reading.wait() => started.unwrap(),
        result = first.as_mut().expect("the first waiter is alive") => panic!("the original control still owns its read: {result:?}"),
    }
    if managed {
        assert!(!first_task.cancel_requested().unwrap(), "neither control has prepared yet");
    }
    let (posted, queued) = signal::once();
    let mut second_env = control_environment(clock.clone(), service.clone());
    CONTROL_SPAWNER.with(|slot| {
        *slot.borrow_mut() = Some(Arc::new(FirstControlPoll {
            inner: (second_env.process.spawner)().unwrap(),
            posted: Mutex::new(Some(posted)),
        }))
    });
    Arc::get_mut(&mut second_env).unwrap().process.spawner = controlled_spawner;
    let mut second = Box::pin(second_task.cancel(&second_env.process));
    assert!(second.as_mut().now_or_never().is_none());
    queued.wait().await.unwrap();
    if managed {
        active.checkpoint().await.unwrap();
    }
    assert_eq!(*service.cancellations.lock(), u32::from(!managed));
    if lose_first_waiter {
        first.take();
    }
    release.resolve(());
    let first_error = match first {
        Some(first) => Some(first.await.unwrap_err().to_string()),
        None => None,
    };
    let second_error = second.await.unwrap_err();
    if managed && let Some(first_error) = first_error {
        assert_eq!(first_error, second_error.to_string(), "one immutable control result");
    }
    assert_eq!(
        *service.cancellations.lock(),
        1,
        "earlier overlapping intent cannot send a fresh control"
    );
    assert_eq!(
        *service.gets.lock(),
        if managed { 1 } else { 2 },
        "managed activations join one receipt; a later fence only reconciles it"
    );
    assert!(second_task.cancel_requested().unwrap(), "the joined view adopted committed evidence");
    let events = second_view.remote_agents().remove(ENDPOINT).unwrap().events().unwrap();
    assert!(events.iter().any(|event| match (&event.payload, reply) {
        (
            lk_agents_abi::delegation::remote::A2AJournalPayload::Error(
                lk_agents_abi::delegation::remote::RemoteFailure::Rpc { error },
            ),
            CancelReply::Refused,
        ) => error.code == a2a::error_code::TASK_NOT_CANCELABLE,
        (
            lk_agents_abi::delegation::remote::A2AJournalPayload::Error(
                lk_agents_abi::delegation::remote::RemoteFailure::Transport { .. },
            ),
            CancelReply::Lost,
        ) => true,
        _ => false,
    }));
    assert!(second_task.cancel(&second_env.process).await.is_err());
    assert_eq!(
        *service.cancellations.lock(),
        if matches!(reply, CancelReply::Refused) { 2 } else { 1 },
        "only a later explicit call naming observed refusal may start another control"
    );
    if let Some(updates) = updates {
        assert_eq!(
            Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
            Closed::Clean
        );
        binding.close();
        first_store.finish_session(false).await.unwrap();
    }
    drop(attachment);
    first_store.close();
    second_store.close();
    for process in [&env.process, &second_env.process] {
        assert!(
            process
                .tasks
                .wait_empty(&*clock, Some(clock.now().saturating_add(Duration::from_secs(15))))
                .await
        );
    }
    assert!(!database.held(PARTITION_KEY).await);
    CONTROL_SPAWNER.with(|slot| slot.borrow_mut().take());
    CONTROL_HTTP.with(|http| http.borrow_mut().take());
}

#[tokio::test(start_paused = true)]
async fn recovery_cannot_borrow_a_live_task_driver_with_another_custom_credential() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = environment(clock.clone());
    let database = Store::open(157, false, clock.clone()).await;
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    let (owner, updates, binding, _) = owner(&env, store.clone()).await;
    let service = Arc::new(Service::default());
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        Some("credential-a".into()),
        Default::default(),
        service.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    let first = interaction();
    let (ready, waiting) = signal::once();
    let mut request = invocation(&env, owner.clone(), ready, Default::default());
    request.configured_token = Some("credential-a".into());
    let running = first.run_with_client(&env.process, client, request);
    tokio::pin!(running);
    tokio::select! {
        biased;
        ready = waiting.wait() => ready.unwrap(),
        result = &mut running => panic!("the original task must still have its question: {result:?}"),
    }
    let second = interaction();
    let client = AgentClient::with_transport(
        ENDPOINT.into(),
        Some("credential-b".into()),
        Default::default(),
        service.clone(),
        (env.process.spawner)().unwrap(),
        clock.clone(),
    )
    .unwrap();
    let (ready, _) = signal::once();
    // The exact same canonical origin selects saved recovery before ordinary call routing.
    // That path must enforce the same native configuration invariant before borrowing A.
    let mut request = invocation(&env, owner, ready, Default::default());
    request.configured_token = Some("credential-b".into());
    assert!(matches!(
        second.run_with_client(&env.process, client, request).await,
        Err(AgentClientError::InvalidInput(_))
    ));
    second.settled().await;
    assert_eq!(service.messages.lock().len(), 1);
    assert_eq!(*service.gets.lock(), 0);
    assert_eq!(*service.cancellations.lock(), 0);
    first.cancelled().unwrap();
    let ended = running.await;
    assert!(matches!(ended, Ok(RemoteEvent::Outcome { outcome: ToolOutcome::Cancelled })));
    first.settled().await;
    assert_eq!(
        Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
        Closed::Clean
    );
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await);
}
