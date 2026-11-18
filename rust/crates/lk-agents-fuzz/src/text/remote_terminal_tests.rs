//! A stale poll returns only after another observation checkpointed the final result.

use super::bound_cancellation::invocation;
use super::*;
use lk_agents_abi::a2a::remote::{RemoteEvent, RemoteInteraction as _};
use lk_agents_abi::delegation::remote::{A2AJournalPayload, A2ARequest, A2AResponse};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::session::remote::{RemoteAgentView as _, RemoteTaskView as _};
use lk_agents_runtime::net::signal;

#[tokio::test(start_paused = true)]
async fn a_stale_poll_cannot_reopen_the_original_call_after_accepted_terminal_completion() {
    for stale in ["TASK_STATE_WORKING", "TASK_STATE_INPUT_REQUIRED"] {
        let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
        let env = environment(clock.clone());
        let database = Store::open(163, false, clock.clone()).await;
        let store = ContextStore::bound(
            database.backend.clone(),
            database.thread_id.clone(),
            PARTITION_KEY.into(),
            env.tracer.root(),
        )
        .await
        .unwrap();
        let (owner, updates, binding, view) = owner(&env, store.clone()).await;
        let (read, reading) = signal::once();
        let (release, released) = signal::once();
        let service = Arc::new(Service {
            gated_reconciliation: Mutex::new(Some(CancelGate { started: read, release: released })),
            task_snapshot: Mutex::new(Some(
                serde_json::from_value(json!({
                    "id":"booking","contextId":"trip","status":{"state":stale}
                }))
                .unwrap(),
            )),
            ..Default::default()
        });
        let client = AgentClient::with_transport(
            ENDPOINT.into(),
            None,
            Default::default(),
            service.clone(),
            (env.process.spawner)().unwrap(),
            clock.clone(),
        )
        .unwrap();
        let original = interaction();
        let (ready, question) = signal::once();
        let running = original.run_with_client(
            &env.process,
            client,
            invocation(&env, owner.clone(), ready, Default::default()),
        );
        tokio::pin!(running);
        tokio::select! {
            biased;
            ready = question.wait() => ready.unwrap(),
            result = &mut running => panic!("the original body still owns its input question: {result:?}"),
        }
        let target = RemoteTaskRef { task_id: "booking".into(), context_id: "trip".into() };
        let driver = owner.remote_driver(ENDPOINT, ENDPOINT, &target).await.unwrap().unwrap();
        let reconcile = driver.reconcile();
        tokio::pin!(reconcile);
        tokio::select! {
            biased;
            ready = reading.wait() => ready.unwrap(),
            () = &mut reconcile => panic!("the SDK GetTask read must remain gated"),
            result = &mut running => panic!("the original result must still be pending: {result:?}"),
        }
        // Another real producer applies a conclusive SDK observation through the same
        // serialized owner while the already-sent poll remains in flight. No business body
        // is replayed and the fixture supplies no routing or recovery policy.
        for input in [
            RemoteInput::Prepare {
                request_id: "accepted-final".into(),
                request: A2ARequest::Get(a2a::GetTaskRequest {
                    id: "booking".into(),
                    history_length: None,
                    tenant: None,
                }),
                streaming: false,
                origin: None,
                input_request: None,
                control_scope: None,
            },
            RemoteInput::DispatchStarted { request_id: "accepted-final".into() },
            RemoteInput::Response {
                request_id: "accepted-final".into(),
                response: Box::new(A2AResponse::Task(
                    serde_json::from_value(json!({"id":"booking","contextId":"trip",
                        "status":{"state":"TASK_STATE_COMPLETED"},
                        "artifacts":[{"artifactId":"answer","parts":[{"text":"Booked"}]}]
                    }))
                    .unwrap(),
                )),
            },
            RemoteInput::Ended { request_id: "accepted-final".into() },
        ] {
            owner
                .remote(RemoteCommand {
                    remote_id: ENDPOINT.into(),
                    endpoint: ENDPOINT.into(),
                    input,
                })
                .await
                .unwrap();
        }
        release.resolve(());
        reconcile.await;
        let outcome = running.await.unwrap();
        assert!(
            matches!(outcome, RemoteEvent::Outcome { outcome: ToolOutcome::Value { text, .. } } if text == "Booked")
        );
        original.settled().await;
        assert_eq!(
            service.messages.lock().len(),
            1,
            "the original business request was not replayed"
        );
        assert_eq!(
            *service.gets.lock(),
            1,
            "a stale poll did not start another observation/subscription"
        );
        assert_eq!(*service.cancellations.lock(), 0);
        let remote = view.remote_agents().remove(ENDPOINT).unwrap();
        let task = remote.tasks().unwrap().remove("booking").unwrap();
        assert_eq!(task.task().unwrap().status.state, a2a::TaskState::Completed);
        assert!(
            task.input_requests()
                .unwrap()
                .values()
                .all(|input| input.state == lk_agents_abi::session::InputRequestState::Closed)
        );
        assert!(remote.events().unwrap().iter().any(|event| matches!(&event.payload,
            A2AJournalPayload::Response(response) if matches!(response.as_ref(),
                A2AResponse::Task(task) if !task.status.state.is_terminal()))));
        assert_eq!(
            Box::new(updates).close(clock.now().saturating_add(Duration::from_secs(15))).await,
            Closed::Clean
        );
        binding.close();
        store.finish_session(false).await.unwrap();
        assert!(!database.held(PARTITION_KEY).await);
    }
}
