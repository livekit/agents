//! A later managed request keeps the same leased owner after the initial binding closes.

use super::*;

struct HoldingOpaque {
    inner: ManagedHandler,
    entered: signal::Producer<bool>,
    resumed: signal::Cell<bool>,
    initial_closed: signal::Producer<bool>,
}

impl TurnHost for HoldingOpaque {
    fn handle<'a>(
        &'a self,
        cx: &Cx,
        ctx: &'a TurnContext,
        port: Arc<dyn lk_agents_abi::a2a::turn::HandlerPort>,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'a, Result<HandlerOutput, ProviderError>> {
        let cx = cx.clone();
        Box::pin(async move {
            let words = ctx.message.as_message().unwrap().raw_text_content().unwrap();
            if words == "Hold the context." {
                self.entered.set(true);
                self.resumed
                    .wait_for(|ready| *ready)
                    .await
                    .expect("the story releases the held arrival");
                return Ok(HandlerOutput {
                    output: Some(Value::String("The context is retained.".into())),
                    state: ctx.state.clone(),
                    ..Default::default()
                });
            }
            let result = self.inner.handle(&cx, ctx, port, cancel).await;
            if words == "Reserve a room." {
                // ManagedHandler explicitly closes its SessionBinding before returning.
                self.initial_closed.set(true);
            }
            result
        })
    }
}

#[tokio::test(start_paused = true)]
async fn the_endpoint_checkpoints_a_later_turn_after_its_initial_handler_binding_closed() {
    let clock: Arc<dyn Clock> = Arc::new(SimClock::new());
    let env = environment(clock.clone(), Vec::new());
    let database = Arc::new(Store::open(97, false, clock.clone()).await);
    let (release, released) = signal::once();
    let (entered, started) = signal::once();
    let (closed, body_closed) = signal::once();
    let service = Arc::new(Service {
        release: released,
        entered: Mutex::new(Some(entered)),
        closed: Mutex::new(Some(closed)),
        starts: Mutex::default(),
        effects: Mutex::default(),
        cancellations: Mutex::default(),
        owners: Mutex::default(),
    });
    let (opaque_started, opaque_entered) = signal::signal(false);
    let (resume, resumed) = signal::signal(false);
    let (initial_closed, binding_closed) = signal::signal(false);
    let models = Arc::<Models>::default();
    let host: Arc<dyn TurnHost> = Arc::new(HoldingOpaque {
        inner: ManagedHandler {
            opaque_side: false,
            env: env.clone(),
            database: database.clone(),
            service: service.clone(),
            models: models.clone(),
        },
        entered: opaque_started,
        resumed,
        initial_closed,
    });
    let options = TurnOptions { policy: PolicySpec::All(Policy::Live), ..Default::default() };
    let callers = Arc::<TaskSet>::default();
    let spawner = (env.process.spawner)().unwrap();
    let original_thread = database.thread().await.unwrap();
    let (sender, mut observer) = outbox::outbox(16);
    let original = Turn::new(
        original_thread.clone(),
        "original",
        PARTITION_KEY,
        host.clone(),
        clock.clone(),
        env.process.spawner,
        options.clone(),
    )
    .with_stream_observer(lk_agents_abi::a2a_protocol::JsonRpcId::Number(1), sender);
    let (done, completed) = signal::once();
    let context = database.thread_id.clone();
    let _first = callers.scoped(&*spawner, "initial managed arrival", async move {
        done.resolve(original.run(json!({"messageId":"initial", "contextId":context, "role":"ROLE_USER", "parts":[{"text":"Reserve a room."}]}), None).await);
    });
    let original_task =
        boundary(&*clock, "initial managed progress", progress(&mut observer)).await;
    boundary(&*clock, "initial body entered", started.wait()).await.unwrap();
    drop(observer);

    let opaque_thread = database.thread().await.unwrap();
    let opaque = Turn::new(
        opaque_thread.clone(),
        "opaque",
        PARTITION_KEY,
        host.clone(),
        clock.clone(),
        env.process.spawner,
        options.clone(),
    );
    let (done, opaque_result) = signal::once();
    let context = database.thread_id.clone();
    let _opaque = callers.scoped(&*spawner, "held opaque arrival", async move {
        done.resolve(opaque.run(json!({"messageId":"held", "contextId":context, "role":"ROLE_USER", "parts":[{"text":"Hold the context."}]}), None).await);
    });
    boundary(&*clock, "opaque handler entered", opaque_entered.wait_for(|entered| *entered))
        .await
        .expect("the opaque handler enters");
    release.resolve(());
    boundary(
        &*clock,
        "initial binding explicitly closed",
        binding_closed.wait_for(|closed| *closed),
    )
    .await
    .expect("the initial binding closes");
    boundary(&*clock, "initial body cleanup", body_closed.wait()).await.unwrap();
    assert!(opaque_result.get().is_none());
    assert!(database.held(PARTITION_KEY).await);

    let later_thread = database.thread().await.unwrap();
    let later = Turn::new(
        later_thread.clone(),
        "later",
        PARTITION_KEY,
        host,
        clock.clone(),
        env.process.spawner,
        options,
    );
    let (done, later_result) = signal::once();
    let context = database.thread_id.clone();
    let _later = callers.scoped(&*spawner, "later managed arrival", async move {
        done.resolve(later.run(json!({"messageId":"later", "contextId":context, "role":"ROLE_USER", "parts":[{"text":"When is check-in?"}]}), None).await);
    });
    let later =
        boundary(&*clock, "later native result after first binding closed", later_result.wait())
            .await
            .unwrap()
            .unwrap();
    assert_eq!(later.task.status, TaskState::Completed);
    assert_eq!(later.output, Some(json!("Check-in is at 3pm.")));
    assert!(opaque_result.get().is_none(), "the same owner remains while opaque work is held");
    {
        let owners = service.owners.lock();
        assert_eq!(owners.len(), 2);
        assert!(
            Weak::ptr_eq(&owners[0], &owners[1]),
            "a later request cannot replace the native owner"
        );
    }
    resume.set(true);
    let original =
        boundary(&*clock, "all endpoint work settled", completed.wait()).await.unwrap().unwrap();
    assert_eq!(original.task.task_id, original_task);
    assert_eq!(original.task.status, TaskState::Completed);
    let opaque =
        boundary(&*clock, "opaque result committed", opaque_result.wait()).await.unwrap().unwrap();
    assert_eq!(opaque.task.status, TaskState::Completed);
    assert_eq!(service.starts.lock().len(), 1);
    assert_eq!(service.effects.lock().len(), 1);
    assert!(
        callers.wait_empty(&*clock, Some(clock.now().saturating_add(Duration::from_secs(5)))).await
    );
    assert!(!database.held(PARTITION_KEY).await);
    assert!(service.owners.lock().iter().all(|owner| owner.upgrade().is_none()));
    assert_eq!(*models.opened.lock(), *models.closed.lock());
    later_thread.close().await;
    opaque_thread.close().await;
    original_thread.close().await;
    database.close().await;
}
