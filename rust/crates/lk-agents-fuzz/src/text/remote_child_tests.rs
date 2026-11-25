//! A crash after child acceptance retains its value until the original remote call reconciles.

use super::*;
use lk_agents_abi::a2a::remote::{RemoteEvent, RemoteInteraction as _};
use lk_agents_abi::agent::run::{RunOutcome, StreamEvent};
use lk_agents_abi::agent::{
    AgentCallbacks, FunctionToolInfo, OutputCandidate, PreparedArguments, ReadBackAnswer,
    ToolArgumentError, ToolCompletion, ValidatedOutput,
};
use lk_agents_abi::delegation::remote::{A2ARequest, A2AResponse};
use lk_agents_abi::llm::tool_executor::{REMOTE_REPORT_KEY, RemoteReport, ToolReport};
use lk_agents_abi::session::{ToolCallRef, snapshot::*};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::text_session::{ChatHistory, LocalAgentTool, SessionStateSink};
use lk_agents_abi::{ChatRequest, ChoiceDelta, FunctionCall, FunctionCallOutput, Llm, LlmStream};
use lk_agents_runtime::delegation::remote::{RemoteInteraction, RemoteInvocation};
use lk_agents_runtime::storage::{CommitOptions, Holder, RunOptions};
use serde_json::Map;

const REMOTE_ID: &str = "booking-service";
const CANONICAL_CALL: &str = "original-canonical-remote-call";

fn remote_call() -> FunctionCall {
    FunctionCall {
        id: CANONICAL_CALL.into(),
        call_id: "provider-id-may-repeat".into(),
        name: "remote_booking".into(),
        arguments: r#"{"request":"Book a seat"}"#.into(),
        created_at: std::time::UNIX_EPOCH,
        extra: [(REMOTE_REPORT_KEY.into(), serde_json::to_value(RemoteReport::Pending).unwrap())]
            .into_iter()
            .collect(),
        group_id: None,
    }
}

fn configured_tools() -> (FunctionToolInfo, FunctionToolInfo) {
    let mut child = tool();
    child.spec.name = "book_child".into();
    child.local_agent = Some("booking".into());
    let mut remote = tool();
    remote.spec.name = "remote_booking".into();
    remote.spec.json_schema =
        json!({"type":"object","properties":{"request":{"type":"string"}},"required":["request"]});
    remote.remote = Some(REMOTE_ID.into());
    (child, remote)
}

fn observed_task(state: a2a::TaskState) -> a2a::Task {
    let message = (state == a2a::TaskState::InputRequired).then(|| {
        json!({
            "messageId":"question", "role":"ROLE_AGENT", "parts":[{"text":"Which day?"}]
        })
    });
    let artifacts = (state == a2a::TaskState::Completed).then(|| {
        json!([
            {"artifactId":"receipt", "parts":[{"text":"Booked"}]}
        ])
    });
    serde_json::from_value(json!({
        "id":"booking", "contextId":"trip", "status":{"state":state,"message":message},
        "artifacts":artifacts
    }))
    .unwrap()
}

fn accepted_child() -> SessionSnapshot {
    let mut state =
        SessionSnapshot::new("conversation".into(), "original-execution".into(), "desk".into());
    let child_id = FrameId { value: "conversation/frame_child".into() };
    let parent =
        ParentCall { frame_id: state.root_frame_id.clone(), call_id: "parent-canonical".into() };
    let parent_item = FunctionCall {
        id: parent.call_id.clone(),
        call_id: "parent-provider-id".into(),
        name: "book_child".into(),
        arguments: "{}".into(),
        created_at: std::time::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    let root = state.frames.get_mut(&state.root_frame_id).unwrap();
    root.history.push(parent_item.into());
    root.pending_calls.push(PendingChildCall {
        parent: parent.clone(),
        invocation_id: "original-parent-invocation".into(),
        execution_id: state.execution_id.clone(),
        tool_name: "book_child".into(),
        agent_name: "booking".into(),
        validated_arguments: json!({}),
        identity: "{}".into(),
        status: ChildCallStatus::Running { frame_id: child_id.clone() },
    });
    let call = remote_call();
    state.frames.insert(
        child_id.clone(),
        SessionFrame {
            id: child_id.clone(),
            agent_name: "booking".into(),
            history: vec![
                call.clone().into(),
                FunctionCallOutput::new(
                    call.output_id(),
                    &call.call_id,
                    "Checking",
                    false,
                    std::time::UNIX_EPOCH,
                )
                .into(),
            ],
            parent: Some(parent),
            completed: true,
            output: Some(json!({"reference":"accepted-R1"})),
            error: None,
            input_requests: Default::default(),
            pending_calls: Vec::new(),
            visible_dialogue: Vec::new(),
            starting_history_ids: Default::default(),
            options: FrameOptions::default(),
        },
    );
    state.ancestors.push(state.root_frame_id.clone());
    state.active_frame_id = child_id.clone();
    state.next_id = 20;
    let request = serde_json::from_value(json!({
        "message": {
            "messageId": "original-message", "role": "ROLE_USER",
            "parts": [{"text": "Book a seat"}]
        }
    }))
    .unwrap();
    for input in [
        RemoteInput::Prepare {
            request_id: "original-request".into(),
            request: A2ARequest::Send(Box::new(request)),
            streaming: false,
            origin: Some(ToolCallRef { agent_id: child_id.value, call_id: CANONICAL_CALL.into() }),
            input_request: None,
            control_scope: None,
        },
        RemoteInput::DispatchStarted { request_id: "original-request".into() },
        RemoteInput::Response {
            request_id: "original-request".into(),
            response: Box::new(A2AResponse::Send(a2a::SendMessageResponse::Task(observed_task(
                a2a::TaskState::Working,
            )))),
        },
        RemoteInput::Ended { request_id: "original-request".into() },
    ] {
        lk_agents_core::session::remote::apply(
            &mut state,
            RemoteCommand { remote_id: REMOTE_ID.into(), endpoint: ENDPOINT.into(), input },
        )
        .unwrap();
    }
    state
}

struct Host {
    base: FakeHost,
    env: Arc<Env>,
    binding: Arc<SessionBinding>,
    service: Arc<Service>,
    calls: Arc<Mutex<Vec<FunctionCall>>>,
}

impl AgentCallbacks for Host {
    fn render_instructions<'a>(
        &'a self,
        cx: &Cx,
        invocation: &'a Invocation,
    ) -> BoxFuture<'a, Result<String, HostError>> {
        self.base.render_instructions(cx, invocation)
    }
    fn list_tools<'a>(
        &'a self,
        _cx: &Cx,
        invocation: &'a Invocation,
    ) -> BoxFuture<'a, Result<Vec<FunctionToolInfo>, HostError>> {
        let (child, remote) = configured_tools();
        let tools = if invocation.agent_name == "desk" { vec![child] } else { vec![remote] };
        Box::pin(async move { Ok(tools) })
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
        cx: &Cx,
        invocation: &'a Invocation,
        _prepared: &'a HostRef,
    ) -> BoxFuture<'a, Result<ToolCompletion, HostError>> {
        let cx = cx.clone();
        Box::pin(async move {
            let call = invocation.function_call.as_ref().unwrap();
            assert_eq!(call.id, CANONICAL_CALL, "recovery cannot invent another canonical call");
            assert_eq!(call.arguments, remote_call().arguments);
            self.calls.lock().push(call.clone());
            let owner = self
                .binding
                .commands()
                .unwrap()
                .expect("the real Runner attached its serialized owner");
            let interaction = RemoteInteraction::new(
                ENDPOINT.into(),
                call.clone().into(),
                String::new(),
                String::new(),
                String::new(),
                Some(REMOTE_ID.into()),
            )
            .unwrap();
            let options = lk_agents_abi::ApiConnectOptions { max_retry: 0, ..Default::default() };
            let client = AgentClient::with_transport(
                ENDPOINT.into(),
                None,
                options,
                self.service.clone(),
                (self.env.process.spawner)().unwrap(),
                self.env.clock.clone(),
            )
            .unwrap();
            let result = interaction
                .run_with_client(
                    &self.env.process,
                    client,
                    RemoteInvocation {
                        arguments: serde_json::from_str(&call.arguments).unwrap(),
                        structured: false,
                        history: invocation.history.clone(),
                        shared: None,
                        configured_token: None,
                        live: false,
                        host: Arc::new(Presenter::default()),
                        context: Default::default(),
                        cx: cx.clone(),
                        owner: Some(owner),
                        origin: Some(ToolCallRef {
                            agent_id: invocation.frame_id.as_ref().unwrap().value.clone(),
                            call_id: call.id.clone(),
                        }),
                        conn_options: options,
                    },
                )
                .await;
            interaction.settled().await;
            let report = match result.map_err(|error| HostError::new(error.to_string()))? {
                RemoteEvent::Outcome { outcome } => ToolReport::from(outcome),
                RemoteEvent::InputRequired { input_request, outcome, .. } => ToolReport {
                    outcome,
                    input_request: input_request.map(Box::new),
                    transition: None,
                },
                other => panic!("the off-call original recovery must settle or ask: {other:?}"),
            };
            Ok(ToolCompletion { report, extra: interaction.extra().unwrap(), usage: None })
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
        self.base.cancel(invocation);
    }
}

struct ObservedLlm {
    inner: FakeLlm,
    requests: Arc<Mutex<Vec<ChatRequest>>>,
}

impl Llm for ObservedLlm {
    fn label(&self) -> String {
        self.inner.label()
    }

    fn chat<'a>(
        &'a self,
        cx: &Cx,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, lk_agents_abi::errors::ProviderError>> {
        self.requests.lock().push(request.clone());
        self.inner.chat(cx, request)
    }
}

struct Restart {
    result: Option<RunOutcome>,
    failure: Option<String>,
    state: SessionSnapshot,
    recovered: Vec<FunctionCallOutput>,
    calls: Vec<FunctionCall>,
    model_requests: Vec<ChatRequest>,
}

async fn restart(database: &Store, env: &Arc<Env>, service: Arc<Service>) -> Restart {
    restart_with_input(database, env, service, "Thanks").await
}

async fn restart_with_input(
    database: &Store,
    env: &Arc<Env>,
    service: Arc<Service>,
    human: &str,
) -> Restart {
    let store = ContextStore::bound(
        database.backend.clone(),
        database.thread_id.clone(),
        PARTITION_KEY.into(),
        env.tracer.root(),
    )
    .await
    .unwrap();
    store.begin_session().await.unwrap();
    let view = SessionState::new(Some(store.load_session_snapshot().await.unwrap())).unwrap();
    let binding = Arc::new(SessionBinding::new(store.clone(), view.clone()).unwrap());
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (child_tool, remote_tool) = configured_tools();
    let model_requests = Arc::new(Mutex::new(Vec::new()));
    let mut agents = Vec::new();
    for (name, tools, replies, child) in [
        ("desk", vec![child_tool], vec!["Booked R1."], false),
        (
            "booking",
            vec![remote_tool],
            vec!["Which day?", "I will check whether cancellation is still possible."],
            true,
        ),
    ] {
        let fakes = RunFakes::new(env.clone(), env.add_run(RunKind::Session), None);
        fakes.script(
            replies
                .into_iter()
                .map(|text| ChoiceDelta {
                    role: Some(Role::Assistant),
                    content: Some(text.into()),
                    ..Default::default()
                })
                .collect(),
            vec![],
        );
        let output = if child {
            OutputSpec::Structured {
                schema: json!({
                    "type":"object", "properties":{"reference":{"type":"string"}},
                    "required":["reference"]
                }),
            }
        } else {
            OutputSpec::Text { validated: false }
        };
        let agent_tools = if child {
            BTreeMap::new()
        } else {
            [(
                "book_child".into(),
                LocalAgentTool {
                    agent_name: "booking".into(),
                    chat_history: ChatHistory::Fresh,
                    merge_chat_ctx: false,
                },
            )]
            .into()
        };
        agents.push(TextSessionAgent {
            definition: AgentDefinition {
                mcp: false,
                name: name.into(),
                instructions: String::new(),
                dynamic_instructions: false,
                tools,
                output,
                retries: Retries::default(),
            },
            llm: Arc::new(ObservedLlm {
                inner: FakeLlm(fakes.clone()),
                requests: model_requests.clone(),
            }),
            host: Arc::new(Host {
                base: FakeHost(fakes),
                env: env.clone(),
                binding: binding.clone(),
                service: service.clone(),
                calls: calls.clone(),
            }),
            output_type_explicit: child,
            agent_tools,
            host_context: HostRef::new(0, &env.refs),
            tools_host: None,
        });
    }
    let runner =
        lk_agents_runtime::text_session::Runner::new(&env.process, "desk".into(), agents, 4, None)
            .unwrap();
    let mut updates = runner
        .run(TextSessionRequest {
            output: None,
            session_sink: Some(binding.clone()),
            session_id: "conversation".into(),
            id_seed: Some(23),
            execution_id: "restart-execution".into(),
            task_id: None,
            message: ChatMessage::text("new-human", Role::User, human, env.wall()).into(),
            delegated: false,
            conn_options: Default::default(),
        })
        .unwrap();
    let mut delivered = BTreeMap::<String, String>::new();
    let mut result = None;
    let mut failure = None;
    let mut recovered = Vec::new();
    while let Some(update) = updates.next().await.unwrap() {
        match update {
            TextSessionUpdate::Stream {
                event: StreamEvent::TextDelta { message_id, delta },
                ..
            } => {
                delivered.entry(message_id).or_default().push_str(&delta);
            }
            TextSessionUpdate::InvocationSettled(invocation_id)
            | TextSessionUpdate::Parked(invocation_id) => {
                let state = view._snapshot_data().unwrap().unwrap();
                for frame in state.frames.values() {
                    for item in &frame.history {
                        if let ChatItem::FunctionCallOutput(output) = item
                            && output.id == format!("{CANONICAL_CALL}_out")
                        {
                            recovered.push(output.clone());
                        }
                    }
                }
                runner
                    .release(TextSessionDelivery {
                        invocation_id,
                        messages: std::mem::take(&mut delivered),
                    })
                    .unwrap();
            }
            TextSessionUpdate::Finished(finished) => result = Some(finished.turn.result.outcome),
            TextSessionUpdate::Failed { error, .. } => failure = Some(error.to_string()),
            _ => {}
        }
    }
    let state = view._snapshot_data().unwrap().unwrap();
    binding.close();
    store.finish_session(false).await.unwrap();
    assert!(!database.held(PARTITION_KEY).await, "restart cleanup releases the partition lease");
    let calls = calls.lock().clone();
    let model_requests = model_requests.lock().clone();
    Restart { result, failure, state, recovered, calls, model_requests }
}

async fn seeded(clock: Arc<dyn Clock>, seed: u64) -> Store {
    let database = Store::open(seed, false, clock).await;
    let thread = database.clean_thread().await;
    let holder = Holder::new("saved-child-fixture");
    let mut run = thread
        .partition(PARTITION_KEY)
        .open(RunOptions::new("saved-child", ""), &holder)
        .await
        .unwrap();
    run.set_state(
        lk_agents_runtime::storage::session_state::snapshot_value(accepted_child()).unwrap(),
    );
    run.commit(CommitOptions::default()).await.unwrap();
    thread.close().await;
    database
}

#[tokio::test(start_paused = true)]
async fn a_saved_completed_child_reconciles_its_original_remote_with_get_before_returning_once() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = super::super::environment(clock.clone(), vec![]);
    let database = seeded(clock, 181).await;
    let service = Arc::new(Service {
        task_snapshot: Mutex::new(Some(observed_task(a2a::TaskState::Completed))),
        ..Default::default()
    });
    let turn = restart(&database, &env, service.clone()).await;
    assert!(turn.failure.is_none(), "{:?}", turn.failure);
    assert!(matches!(turn.result, Some(RunOutcome::Completed { .. })));
    assert_eq!(*service.gets.lock(), 1);
    assert!(
        service.messages.lock().is_empty(),
        "the accepted business request cannot be sent again"
    );
    assert_eq!(turn.calls.len(), 1);
    assert_eq!(turn.calls[0].id, CANONICAL_CALL);
    assert_eq!(turn.recovered.len(), 1);
    assert_eq!(turn.recovered[0].output, "Booked");
    assert_eq!(turn.recovered[0].call_id, "provider-id-may-repeat");
    assert_eq!(turn.state.frames.len(), 1);
    let parent = &turn.state.frames[&turn.state.root_frame_id];
    let child_results: Vec<_> = parent
        .history
        .iter()
        .filter_map(|item| match item {
            ChatItem::FunctionCallOutput(output) if output.id == "parent-canonical_out" => {
                Some(output)
            }
            _ => None,
        })
        .collect();
    assert_eq!(child_results.len(), 1);
    assert_eq!(child_results[0].output, r#"{"reference":"accepted-R1"}"#);
}

#[tokio::test(start_paused = true)]
async fn a_recovered_remote_question_invalidates_the_older_accepted_child_output() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = super::super::environment(clock.clone(), vec![]);
    let database = seeded(clock, 191).await;
    let service = Arc::new(Service {
        task_snapshot: Mutex::new(Some(observed_task(a2a::TaskState::InputRequired))),
        ..Default::default()
    });
    let turn = restart_with_input(&database, &env, service.clone(), "Actually, do not book").await;
    assert!(turn.failure.is_none(), "{:?}", turn.failure);
    assert!(matches!(turn.result, Some(RunOutcome::InputRequired(_))));
    assert_eq!(*service.gets.lock(), 1);
    assert!(service.messages.lock().is_empty());
    let child = &turn.state.frames[&turn.state.active_frame_id];
    assert!(!child.completed);
    assert!(turn.model_requests.iter().any(|request|
        request.chat_ctx.items.iter().any(|item| matches!(item, ChatItem::Message(message)
            if message.id == "new-human" && message.raw_text_content().as_deref() == Some("Actually, do not book")))),
        "the arriving correction must reach a real model request after reconciliation");
    assert!(child.history.iter().any(|item| matches!(item, ChatItem::Message(message)
        if message.id == "new-human" && message.raw_text_content().as_deref() == Some("Actually, do not book"))),
        "recovery must let the child handle the already accepted human correction");
    let task = &turn.state.remote_agents[REMOTE_ID].tasks["booking"];
    assert!(
        task.input_requests
            .values()
            .all(|request| request.state == lk_agents_abi::session::InputRequestState::Open
                && request.response_message_id.is_none()),
        "a newly discovered question cannot consume the earlier human correction"
    );
    assert_eq!(
        child.output,
        Some(json!({"reference":"accepted-R1"})),
        "the previous value remains inspectable but is not propagated"
    );
    assert!(
        !turn.state.frames[&turn.state.root_frame_id]
            .history
            .iter()
            .any(|item| item.id() == "parent-canonical_out")
    );
}

#[tokio::test(start_paused = true)]
async fn an_unknown_remote_recovery_keeps_the_accepted_child_visible_without_a_business_resend() {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_runtime::net::clock::SimClock::new());
    let env = super::super::environment(clock.clone(), vec![]);
    let database = seeded(clock, 193).await;
    let service = Arc::new(Service { fail_reconciliation: Mutex::new(true), ..Default::default() });
    let turn = restart(&database, &env, service.clone()).await;
    assert!(turn.failure.is_some());
    assert_eq!(*service.gets.lock(), 1);
    assert!(service.messages.lock().is_empty());
    let child = &turn.state.frames[&turn.state.active_frame_id];
    assert!(child.completed);
    assert_eq!(child.output, Some(json!({"reference":"accepted-R1"})));
    assert!(
        child.history.iter().any(|item| matches!(item, ChatItem::Message(message)
        if message.id == "new-human" && message.raw_text_content().as_deref() == Some("Thanks"))),
        "an unprovable outcome refuses visibly without losing the accepted arriving input"
    );
    assert!(
        !turn.state.frames[&turn.state.root_frame_id]
            .history
            .iter()
            .any(|item| item.id() == "parent-canonical_out")
    );
}
