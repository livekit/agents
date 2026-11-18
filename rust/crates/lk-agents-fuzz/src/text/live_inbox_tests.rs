//! Real leased A2A + bound TextSession execution, with only model/tool/transport ports faked.

use super::*;
use lk_agents_abi::a2a::door::{DoorError, DoorFrame};
use lk_agents_abi::a2a::turn::{HandlerOutput, TurnContext, TurnHost};
use lk_agents_abi::agent::run::{RunOutcome, StreamEvent};
use lk_agents_abi::agent::{
    AgentCallbacks, FunctionToolInfo, OutputCandidate, PreparedArguments, ReadBackAnswer,
    ToolArgumentError, ToolCompletion, ValidatedOutput,
};
use lk_agents_abi::errors::{HostError, HostErrorKind, ProviderError};
use lk_agents_abi::llm::tool_executor::ToolOutcome;
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::storage::session::{PersistedSessionState as _, SessionBinding as _};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::text_session::SessionStateSink as _;
use lk_agents_abi::{BoxFuture, ChatRequest, ChoiceDelta, Closed, LlmChunk, LlmStream};
use lk_agents_runtime::net::{clock::SimClock, outbox, signal, task::TaskSet};
use lk_agents_runtime::storage::context::store::ContextStore;
use lk_agents_runtime::storage::session_binding::SessionBinding;
use lk_agents_runtime::storage::session_state::SessionState;
use serde_json::{Map, Value};
use std::sync::Weak;

#[path = "live_checkpoint_tests.rs"]
mod checkpoint_lifetime;

type Runner = lk_agents_runtime::text_session::Runner;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Started {
    task_id: String,
    call_id: String,
    output_id: String,
}

struct Service {
    release: signal::Cell<Option<()>>,
    entered: Mutex<Option<signal::Producer<Option<Started>>>>,
    closed: Mutex<Option<signal::Producer<Option<()>>>>,
    starts: Mutex<Vec<Started>>,
    effects: Mutex<Vec<Started>>,
    cancellations: Mutex<Vec<String>>,
    owners: Mutex<Vec<Weak<dyn lk_agents_abi::session::SessionCommands>>>,
}

struct BodyClosed(Option<signal::Producer<Option<()>>>);
impl Drop for BodyClosed {
    fn drop(&mut self) {
        if let Some(closed) = self.0.take() {
            closed.resolve(());
        }
    }
}

#[derive(Default)]
struct Models {
    opened: Mutex<usize>,
    closed: Mutex<usize>,
    requests: Mutex<Vec<(String, Vec<ChatItem>)>>,
}

struct Model {
    source: String,
    base: FakeLlm,
    observations: Arc<Models>,
}

impl Llm for Model {
    fn label(&self) -> String {
        "live-story-model".into()
    }

    fn chat<'a>(
        &'a self,
        cx: &Cx,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, ProviderError>> {
        let cx = cx.clone();
        Box::pin(async move {
            self.observations
                .requests
                .lock()
                .push((self.source.clone(), request.chat_ctx.items.clone()));
            let stream = self.base.chat(&cx, request).await?;
            *self.observations.opened.lock() += 1;
            Ok(Box::new(ModelStream { inner: stream, observations: self.observations.clone() })
                as LlmStream)
        })
    }
}

struct ModelStream {
    inner: LlmStream,
    observations: Arc<Models>,
}
impl StreamOutput<LlmChunk> for ModelStream {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        self.inner.next()
    }

    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move {
            let closed = self.inner.close(deadline).await;
            *self.observations.closed.lock() += 1;
            closed
        })
    }
}
impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for ModelStream {}

struct Host {
    base: FakeHost,
    runner: Mutex<Option<Weak<Runner>>>,
    service: Arc<Service>,
    canceled: signal::Producer<bool>,
    cancellation: signal::Cell<bool>,
}

impl AgentCallbacks for Host {
    fn checkpoint<'a>(
        &'a self,
        cx: &Cx,
        invocation: &'a Invocation,
    ) -> BoxFuture<'a, Result<bool, HostError>> {
        self.base.checkpoint(cx, invocation)
    }

    fn render_instructions<'a>(
        &'a self,
        cx: &Cx,
        invocation: &'a Invocation,
    ) -> BoxFuture<'a, Result<String, HostError>> {
        self.base.render_instructions(cx, invocation)
    }

    fn list_tools<'a>(
        &'a self,
        cx: &Cx,
        invocation: &'a Invocation,
    ) -> BoxFuture<'a, Result<Vec<FunctionToolInfo>, HostError>> {
        self.base.list_tools(cx, invocation)
    }

    fn prepare_tool<'a>(
        &'a self,
        cx: &Cx,
        invocation: &'a Invocation,
        arguments: Map<String, Value>,
    ) -> BoxFuture<'a, Result<Result<PreparedArguments, ToolArgumentError>, HostError>> {
        self.base.prepare_tool(cx, invocation, arguments)
    }

    fn read_back<'a>(
        &'a self,
        cx: &Cx,
        invocation: &'a Invocation,
        prepared: &'a HostRef,
    ) -> BoxFuture<'a, Result<ReadBackAnswer, HostError>> {
        self.base.read_back(cx, invocation, prepared)
    }

    fn run_tool<'a>(
        &'a self,
        _cx: &Cx,
        invocation: &'a Invocation,
        _prepared: &'a HostRef,
    ) -> BoxFuture<'a, Result<ToolCompletion, HostError>> {
        Box::pin(async move {
            let _closed = BodyClosed(self.service.closed.lock().take());
            let call = invocation
                .function_call
                .as_ref()
                .expect("the real executor supplies its canonical call");
            let started = Started {
                task_id: invocation
                    .execution_id
                    .clone()
                    .expect("the transport task owns this body"),
                call_id: call.id.clone(),
                output_id: call.output_id(),
            };
            self.service.starts.lock().push(started.clone());
            let runner = self
                .runner
                .lock()
                .as_ref()
                .and_then(Weak::upgrade)
                .expect("the live handler still owns its runner");
            runner
                .progress(
                    &invocation.invocation_id,
                    invocation.operation,
                    "Checking the booking service.".into(),
                    false,
                )
                .await
                .map_err(|error| HostError::new(error.to_string()))?;
            if let Some(entered) = self.service.entered.lock().take() {
                entered.resolve(started.clone());
            }
            let released = tokio::select! {
                biased;
                released = self.service.release.wait() => released,
                _ = self.cancellation.wait_for(|canceled| *canceled) => {
                    return Err(HostError { kind: HostErrorKind::Cancelled, ..HostError::new("the body was canceled") });
                }
            };
            released
                .map_err(|_| HostError::new("the fake service gate ended without a response"))?;
            self.service.effects.lock().push(started);
            Ok(ToolCompletion {
                report: ToolOutcome::Value {
                    text: "Reservation R1".into(),
                    reply_required: true,
                    verbatim: false,
                    more: None,
                }
                .into(),
                extra: Default::default(),
                usage: None,
            })
        })
    }

    fn validate_output<'a>(
        &'a self,
        cx: &Cx,
        invocation: &'a Invocation,
        candidate: OutputCandidate,
    ) -> BoxFuture<'a, Result<ValidatedOutput, HostError>> {
        self.base.validate_output(cx, invocation, candidate)
    }

    fn cancel(&self, invocation: &Invocation) {
        if let Some(call) = &invocation.function_call {
            self.service.cancellations.lock().push(call.id.clone());
        }
        self.canceled.set(true);
        self.base.cancel(invocation);
    }
}

struct ManagedHandler {
    opaque_side: bool,
    env: Arc<Env>,
    database: Arc<Store>,
    service: Arc<Service>,
    models: Arc<Models>,
}

fn text(words: &str) -> ChoiceDelta {
    ChoiceDelta { role: Some(Role::Assistant), content: Some(words.into()), ..Default::default() }
}

impl TurnHost for ManagedHandler {
    fn handle<'a>(
        &'a self,
        _cx: &Cx,
        ctx: &'a TurnContext,
        port: Arc<dyn HandlerPort>,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'a, Result<HandlerOutput, ProviderError>> {
        Box::pin(async move {
            let human = ctx.message.as_message().unwrap().raw_text_content().unwrap();
            let booking = human == "Reserve a room.";
            assert!(
                booking || human == "When is check-in?",
                "the story admits only its two human messages"
            );
            if !booking && self.opaque_side {
                return Ok(HandlerOutput {
                    output: Some(Value::String("Check-in is at 3pm.".into())),
                    // This older handler view cannot replace the active native owner.
                    state: ctx.state.clone(),
                    ..Default::default()
                });
            }
            let id = self.env.add_run(RunKind::Handler);
            let fakes = RunFakes::new(self.env.clone(), id, Some(port.clone()));
            fakes.script(
                if booking {
                    vec![
                        ChoiceDelta {
                            role: Some(Role::Assistant),
                            tool_calls: vec![lk_agents_abi::FunctionToolCall {
                                name: "lookup".into(),
                                arguments: json!({"q":"room"}).to_string(),
                                call_id: "wire_booking".into(),
                                extra: None,
                            }],
                            ..Default::default()
                        },
                        text("I'm checking availability."),
                        text("Booked reservation R1."),
                    ]
                } else {
                    vec![text("Check-in is at 3pm.")]
                },
                Vec::new(),
            );
            let store = ContextStore::turn(
                self.database.backend.clone(),
                ctx.thread_id.clone(),
                ctx.partition_key.clone(),
                ctx.context_scopes.clone(),
                ctx.state.clone(),
                ctx.read_only,
                self.env.tracer.root(),
            )
            .await
            .unwrap();
            store.set_session_sink(port.clone()).unwrap();
            let view = SessionState::new(None).unwrap();
            let binding = Arc::new(SessionBinding::new(store.clone(), view).unwrap());
            let (canceled, cancellation) = signal::signal(false);
            let host = Arc::new(Host {
                base: FakeHost(fakes.clone()),
                runner: Mutex::new(None),
                service: self.service.clone(),
                canceled,
                cancellation,
            });
            let model = Arc::new(Model {
                source: human.clone(),
                base: FakeLlm(fakes),
                observations: self.models.clone(),
            });
            let runner = Arc::new(
                Runner::new(
                    &self.env.process,
                    "desk".into(),
                    vec![TextSessionAgent {
                        definition: Env::agent(),
                        llm: model,
                        host: host.clone(),
                        output_type_explicit: false,
                        agent_tools: BTreeMap::new(),
                        host_context: HostRef::new(id.into(), &self.env.refs),
                        tools_host: None,
                    }],
                    4,
                    None,
                )
                .unwrap(),
            );
            *host.runner.lock() = Some(Arc::downgrade(&runner));
            let mut updates = runner
                .run(TextSessionRequest {
                    output: None,
                    session_sink: Some(binding.clone()),
                    session_id: format!("session_{}", ctx.thread_id),
                    id_seed: Some(id.into()),
                    execution_id: ctx.task_id.clone(),
                    task_id: Some(ctx.task_id.clone()),
                    message: ctx.message.clone(),
                    delegated: ctx.delegated,
                    conn_options: Default::default(),
                })
                .unwrap();
            let owner =
                binding.commands().unwrap().expect("the real binding lends its active owner");
            self.service.owners.lock().push(Arc::downgrade(&owner));
            drop(owner);
            let mut delivered = BTreeMap::<String, String>::new();
            let finished = loop {
                let update = tokio::select! {
                    biased;
                    update = updates.next() => Some(update?),
                    () = cancel.cancelled() => None,
                };
                let Some(update) = update else {
                    let closed = Box::new(updates)
                        .close(
                            self.env
                                .clock
                                .now()
                                .saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
                        )
                        .await;
                    assert_eq!(closed, Closed::Clean);
                    return Err(ProviderError::Host {
                        error: HostError {
                            kind: HostErrorKind::Cancelled,
                            ..HostError::new("the handler was canceled")
                        },
                        api_kind: None,
                        error_kind: None,
                        retryable: false,
                        status_code: None,
                        said: None,
                    });
                };
                match update {
                    Some(TextSessionUpdate::Stream {
                        event: StreamEvent::TextDelta { message_id, delta },
                        ..
                    }) => {
                        delivered.entry(message_id).or_default().push_str(&delta);
                        port.token(delta).await;
                    }
                    Some(TextSessionUpdate::Stream {
                        event: StreamEvent::Progress { message, .. },
                        ..
                    }) => port.progress(message).await,
                    Some(
                        TextSessionUpdate::InvocationSettled(invocation_id)
                        | TextSessionUpdate::Parked(invocation_id),
                    ) => {
                        runner
                            .release(TextSessionDelivery {
                                invocation_id,
                                messages: std::mem::take(&mut delivered),
                            })
                            .unwrap();
                    }
                    Some(TextSessionUpdate::Finished(finished)) => break finished,
                    Some(TextSessionUpdate::Failed { error, .. }) => {
                        assert!(
                            matches!(
                                error,
                                lk_agents_abi::text_session::TextSessionError::Run(
                                    lk_agents_abi::agent::run::RunError::Cancelled
                                )
                            ),
                            "the managed story failed: {error}"
                        );
                        assert_eq!(
                            Box::new(updates)
                                .close(
                                    self.env
                                        .clock
                                        .now()
                                        .saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT,)
                                )
                                .await,
                            Closed::Clean
                        );
                        binding.close();
                        return Err(ProviderError::Host {
                            error: HostError {
                                kind: HostErrorKind::Cancelled,
                                ..HostError::new("the admitted task was canceled")
                            },
                            api_kind: None,
                            error_kind: None,
                            retryable: false,
                            status_code: None,
                            said: None,
                        });
                    }
                    Some(_) => {}
                    None => panic!("a managed handler ended without its native result"),
                }
            };
            assert_eq!(
                Box::new(updates)
                    .close(
                        self.env.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT)
                    )
                    .await,
                Closed::Clean
            );
            let RunOutcome::Completed { output: lk_agents_abi::agent::run::Output::Text(answer) } =
                &finished.turn.result.outcome
            else {
                panic!("the story's typed framework result must complete");
            };
            let result = HandlerOutput {
                output: Some(Value::String(answer.clone())),
                new_items: finished.turn.result.new_messages.clone(),
                state: store.session_state(),
                scope_writes: store.writes(),
                app_writes: store.statements(),
                usage: vec![finished.turn.result.usage.clone()],
                input_required: None,
                pending_call: None,
                interrupted: false,
                metadata: Default::default(),
                error: None,
                tool_executions: Vec::new(),
                status: None,
            };
            binding.close();
            Ok(result)
        })
    }
}

async fn boundary<T>(
    clock: &dyn Clock,
    name: &str,
    future: impl std::future::Future<Output = T>,
) -> T {
    eprintln!("live inbox story: waiting for {name}");
    let result = lk_agents_abi::timeout(clock, std::time::Duration::from_secs(5), future)
        .await
        .unwrap_or_else(|_| panic!("live inbox story stalled at {name}"));
    eprintln!("live inbox story: observed {name}");
    result
}

async fn progress(receiver: &mut mpsc::Receiver<Result<DoorFrame, DoorError>>) -> String {
    loop {
        let frame = receiver.recv().await.expect("the response observer stays open").unwrap();
        assert!(!frame.is_error);
        let frame: Value = serde_json::from_str(&frame.json).unwrap();
        eprintln!("live inbox story: observer frame {frame}");
        if frame["result"]["statusUpdate"]["status"]["message"]["parts"][0]["text"]
            == "Checking the booking service."
        {
            assert_eq!(frame["result"]["statusUpdate"]["status"]["state"], "TASK_STATE_WORKING");
            return frame["result"]["statusUpdate"]["taskId"].as_str().unwrap().into();
        }
    }
}

#[tokio::test(start_paused = true)]
async fn a_live_a2a_owner_answers_an_independent_task_before_its_held_tool_finishes() {
    live_story(false, false).await;
}

#[tokio::test(start_paused = true)]
async fn an_opaque_side_reply_reaches_the_same_owner_before_original_result_delivery() {
    live_story(true, false).await;
}

#[tokio::test(start_paused = true)]
async fn explicit_task_cancellation_stops_only_its_held_body_after_an_independent_reply() {
    live_story(false, true).await;
}

async fn live_story(opaque_side: bool, cancel_original: bool) {
    let clock: Arc<dyn Clock> = Arc::new(SimClock::new());
    let env = environment(clock.clone(), Vec::new());
    let database = Arc::new(Store::open(91, false, clock.clone()).await);
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
    let models = Arc::<Models>::default();
    let host: Arc<dyn TurnHost> = Arc::new(ManagedHandler {
        opaque_side,
        env: env.clone(),
        database: database.clone(),
        service: service.clone(),
        models: models.clone(),
    });
    let options = TurnOptions { policy: PolicySpec::All(Policy::Live), ..Default::default() };
    let callers = Arc::<TaskSet>::default();
    let spawner = (env.process.spawner)().unwrap();
    let first_thread = database.thread().await.unwrap();
    let (sender, mut observer) = outbox::outbox(16);
    let first = Turn::new(
        first_thread.clone(),
        "worker-A",
        PARTITION_KEY,
        host.clone(),
        clock.clone(),
        env.process.spawner,
        options.clone(),
    )
    .with_stream_observer(lk_agents_abi::a2a_protocol::JsonRpcId::Number(1), sender);
    let (done, original_result) = signal::once();
    let context = database.thread_id.clone();
    let _original = callers.scoped(&*spawner, "original booking arrival", async move {
        done.resolve(first.run(json!({"messageId":"booking", "contextId":context, "role":"ROLE_USER", "parts":[{"text":"Reserve a room."}]}), None).await);
    });
    let original_task =
        boundary(&*clock, "native Working progress publication", progress(&mut observer)).await;
    let started = boundary(&*clock, "held service start", started.wait()).await.unwrap();
    assert_eq!(started.task_id, original_task);
    drop(observer); // A real response observer can leave after durable admission and progress.

    let second_thread = database.thread().await.unwrap();
    let second = Turn::new(
        second_thread.clone(),
        "worker-B",
        PARTITION_KEY,
        host,
        clock.clone(),
        env.process.spawner,
        options,
    );
    let (side_done, side_result) = signal::once();
    let side_context = database.thread_id.clone();
    let _side = callers.scoped(&*spawner, "independent check-in arrival", async move {
        side_done.resolve(second.run(json!({"messageId":"side", "contextId":side_context, "role":"ROLE_USER", "parts":[{"text":"When is check-in?"}]}), None).await);
    });
    let side = boundary(
        &*clock,
        "independent task completion while original service is held",
        side_result.wait(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(side.task.status, TaskState::Completed);
    assert_eq!(side.output, Some(json!("Check-in is at 3pm.")));
    assert_ne!(side.task.task_id, original_task);
    assert!(
        original_result.get().is_none(),
        "the acknowledgement did not complete the original task"
    );
    assert!(
        service.effects.lock().is_empty(),
        "the service is still held when the independent task completes"
    );
    assert_eq!(service.starts.lock().len(), 1);
    {
        let owners = service.owners.lock();
        assert_eq!(owners.len(), if opaque_side { 1 } else { 2 });
        if !opaque_side {
            assert!(
                Weak::ptr_eq(&owners[0], &owners[1]),
                "both managed turns attach to the same native CommandPort"
            );
        }
    }
    let observed = database.clean_thread().await;
    let partition = observed.partition(PARTITION_KEY);
    assert_eq!(partition.task(&original_task).await.unwrap().unwrap().status, TaskState::Working);
    assert!(database.held(PARTITION_KEY).await, "one leased owner retains the held work");

    if cancel_original {
        // External CancelTask's actual storage boundary, while the owner keeps its lease.
        partition.cancel_task(&original_task, Default::default()).await.unwrap();
    } else {
        release.resolve(());
    }
    let original = boundary(
        &*clock,
        "original task settlement after release or explicit cancellation",
        original_result.wait(),
    )
    .await
    .unwrap()
    .unwrap();
    boundary(&*clock, "original tool body cleanup", body_closed.wait()).await.unwrap();
    assert_eq!(original.task.task_id, original_task);
    if cancel_original {
        assert_eq!(original.task.status, TaskState::Canceled);
        assert!(original.output.is_none());
        assert!(service.effects.lock().is_empty(), "cancellation cannot release the business gate");
        assert_eq!(&*service.cancellations.lock(), std::slice::from_ref(&started.call_id));
    } else {
        assert_eq!(original.task.status, TaskState::Completed);
        assert_eq!(original.output, Some(json!("Booked reservation R1.")));
        assert_eq!(&*service.effects.lock(), std::slice::from_ref(&started));
        assert!(service.cancellations.lock().is_empty(), "observer loss is not task cancellation");
    }
    assert_eq!(service.starts.lock().len(), 1);
    let history = partition.history().await.unwrap();
    assert_eq!(history.iter().filter(|item| item.id == started.call_id).count(), 1);
    let outputs: Vec<_> = history.iter().filter(|item| item.id == started.output_id).collect();
    assert_eq!(outputs.len(), 1);
    let native =
        lk_agents_abi::llm::chat_context::protocol::decode_chat_item(outputs[0].payload.clone())
            .unwrap();
    assert!(matches!(native, ChatItem::FunctionCallOutput(output)
        if output.output == if cancel_original {""} else {"Reservation R1"}));
    if !cancel_original {
        let requests = models.requests.lock();
        let (_, delivery) =
            requests.iter().rev().find(|(source, _)| source == "Reserve a room.").unwrap();
        assert!(delivery.iter().any(|item| item.as_message().is_some_and(|message| {
            message.raw_text_content().as_deref() == Some("Check-in is at 3pm.")
        })));
    }
    assert_eq!(*models.opened.lock(), 3 + usize::from(!opaque_side) - usize::from(cancel_original));
    assert_eq!(
        *models.opened.lock(),
        *models.closed.lock(),
        "every actual provider stream is closed"
    );
    let tasks = partition.tasks(None, 100).await.unwrap();
    assert_eq!(tasks.len(), 2, "independent no-taskId messages retain their two task identities");
    assert_eq!(
        tasks.iter().filter(|task| task.status == TaskState::Completed).count(),
        if cancel_original { 1 } else { 2 }
    );
    assert_eq!(
        tasks.iter().filter(|task| task.status == TaskState::Canceled).count(),
        usize::from(cancel_original)
    );
    assert!(
        callers.wait_empty(&*clock, Some(clock.now().saturating_add(Duration::from_secs(5)))).await,
        "arrival owners did not settle"
    );
    assert!(
        service.owners.lock().iter().all(|owner| owner.upgrade().is_none()),
        "ending the owner drops its unsaved command capability"
    );
    assert!(partition.runs(100).await.unwrap().iter().all(|run| !run.interrupted()));
    assert!(!database.held(PARTITION_KEY).await);
    observed.close().await;
    second_thread.close().await;
    first_thread.close().await;
    database.close().await;
}
