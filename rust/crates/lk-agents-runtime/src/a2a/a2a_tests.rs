#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
use std::sync::Arc;
use std::time::Duration;

use crate::a2a::Turn;
use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{ChatMessage, ChatRole, FunctionCall, FunctionCallOutput};
use lk_agents_abi::a2a::turn::HandlerPort;
use lk_agents_abi::a2a::turn::TurnContext;
use lk_agents_abi::errors::{HostError, HostErrorKind, ProviderError};
use lk_agents_abi::llm::chat_context::protocol::{decode_chat_item, encode_chat_item};
use lk_agents_abi::{Clock, Spawner};
use lk_agents_core::a2a::{
    DebounceSpec, HandlerOutput, PendingCall, Policy, PolicySpec, TurnHost, TurnOptions, TurnResult,
};
use lk_agents_net::clock::RealClock;
use lk_agents_net::env::{ProcessEnv, ProcessError};
use lk_agents_storage::TaskWrite;
use lk_agents_storage::testing::Backend;
use lk_agents_storage::thread::db::Receive;
use lk_agents_storage::thread::run::{ItemRow, MERGED_INTO_KEY, RunOutcome, TaskState};
use lk_agents_storage::thread::schema::{Ask, TaskData};
use lk_agents_storage::timestamp;
use lk_agents_storage::{RunOptions, Thread, shortuuid};
use parking_lot::Mutex;

use lk_agents_storage::StorageError;
use serde_json::{Value, json};
use tokio::sync::watch;

const PARTITION_KEY: &str = "front-desk";

/// A user's A2A v1 message, as protobuf JSON.
fn a2a(text: &str, message_id: &str, channel: Option<&str>, task_id: Option<&str>) -> Value {
    let mut msg = json!({
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    });
    if let Some(c) = channel {
        msg["metadata"] = json!({"https://livekit.io/a2a/ext/agents/v1/channel": c});
    }
    if let Some(t) = task_id {
        msg["taskId"] = json!(t);
    }
    msg
}

/// [`a2a`], typed.
fn user_message(
    text: &str,
    message_id: &str,
    channel: Option<&str>,
    task_id: Option<&str>,
) -> a2a::Message {
    serde_json::from_value(a2a(text, message_id, channel, task_id)).unwrap()
}

fn json_of(v: &impl serde::Serialize) -> Value {
    serde_json::to_value(v).unwrap()
}

fn text_of(item: &ChatItem) -> String {
    match &item.item {
        Some(chat_item::Item::Message(m)) => m
            .content
            .iter()
            .filter_map(|c| c.payload.clone().map(|Payload::Text(t)| t))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn kind_of(item: &ItemRow) -> &'static str {
    match item.kind() {
        Some(chat_item::Item::Message(_)) => "message",
        Some(chat_item::Item::FunctionCall(_)) => "function_call",
        Some(chat_item::Item::FunctionCallOutput(_)) => "function_call_output",
        Some(chat_item::Item::AgentHandoff(_)) => "agent_handoff",
        Some(chat_item::Item::AgentConfigUpdate(_)) => "agent_config_update",
        None => "",
    }
}

fn message(role: ChatRole, text: &str, created_at: f64) -> ChatItem {
    ChatItem {
        item: Some(chat_item::Item::Message(ChatMessage {
            id: shortuuid("item_"),
            role: role as i32,
            content: vec![ChatContent { payload: Some(Payload::Text(text.into())) }],
            created_at: Some(timestamp(created_at)),
            ..Default::default()
        })),
    }
}

fn user_item(text: &str, created_at: f64) -> ChatItem {
    message(ChatRole::User, text, created_at)
}

fn call_item(name: &str, call_id: &str, created_at: f64) -> ChatItem {
    ChatItem {
        item: Some(chat_item::Item::FunctionCall(FunctionCall {
            id: shortuuid("item_"),
            call_id: call_id.into(),
            arguments: "{}".into(),
            name: name.into(),
            created_at: Some(timestamp(created_at)),
            ..Default::default()
        })),
    }
}

/// A flag two tasks share: `set` once, `wait` until then.
#[derive(Clone)]
struct Gate(watch::Sender<bool>);

impl Gate {
    fn new() -> Self {
        Self(watch::channel(false).0)
    }

    fn set(&self) {
        let _ = self.0.send(true);
    }

    async fn wait(&self) {
        let mut rx = self.0.subscribe();
        while !*rx.borrow() {
            rx.changed().await.expect("the gate outlives the waiter");
        }
    }
}

/// What the scripted host does with a turn.
#[derive(Clone)]
enum Behavior {
    /// `"reply to: " + text`.
    Echo,
    Text(&'static str),
    /// `state.verified += 1`, `"seen <n>"`.
    Counter,
    /// The handler raised.
    Raise,
    /// The host met a storage error: the run aborts.
    Unavailable,
    /// Sets `entered`, reports an update, waits for `gate`, returns the text.
    Slow {
        entered: Gate,
        gate: Gate,
        output: &'static str,
    },
    /// Opens `gate`, returns the text.
    Opener {
        gate: Gate,
        output: &'static str,
    },
    /// `InputRequired("Could you send a photo?")` from `request_photo`, call `p1`.
    Ask,
}

#[derive(Debug, Clone, Default)]
struct Call {
    batch: Vec<String>,
    history: Vec<String>,
    message: Vec<String>,
}

#[derive(Clone)]
struct Fake {
    behavior: Behavior,
    calls: Arc<Mutex<Vec<Call>>>,
    updates: Arc<Mutex<Vec<(String, String)>>>,
}

impl Fake {
    fn new(behavior: Behavior) -> Arc<Self> {
        Arc::new(Self {
            behavior,
            calls: Arc::new(Mutex::new(Vec::new())),
            updates: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().clone()
    }

    fn batches(&self) -> Vec<Vec<String>> {
        self.calls().into_iter().map(|c| c.batch).collect()
    }
}

fn text(output: &str) -> HandlerOutput {
    HandlerOutput { output: Some(json!(output)), ..HandlerOutput::default() }
}

impl TurnHost for Fake {
    fn handle<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        ctx: &'a TurnContext,
        _turn: Arc<dyn HandlerPort>,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<lk_agents_abi::a2a::turn::HandlerOutput, ProviderError>>
    {
        Box::pin(async move {
            // cancelled as the host cancels its task: the run stops where it waits
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    Err(HostError { kind: HostErrorKind::Cancelled, ..HostError::new("") }.into())
                }
                out = self.run(ctx.clone()) => out.map(host_output),
            }
        })
    }
}

/// The fake's output as the host returns it: chat items as records.
fn host_output(out: HandlerOutput) -> lk_agents_abi::a2a::turn::HandlerOutput {
    lk_agents_abi::a2a::turn::HandlerOutput {
        output: out.output,
        new_items: out
            .new_items
            .into_iter()
            .map(|item| decode_chat_item(item).expect("chat item"))
            .collect(),
        state: out.state,
        scope_writes: out.scope_writes,
        app_writes: out.app_writes,
        usage: vec![out.usage],
        input_required: out.input_required,
        pending_call: out.pending_call,
        interrupted: out.interrupted,
        metadata: out.metadata,
        error: out.error,
        tool_executions: out.tool_executions,
        status: out.status,
    }
}

fn batch_message_text(message: &lk_agents_abi::a2a::turn::Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            lk_agents_abi::a2a::client::MessagePart::TextPart { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

impl Fake {
    async fn run(&self, ctx: TurnContext) -> Result<HandlerOutput, ProviderError> {
        let message_text = ctx.batch.last().map(batch_message_text).unwrap_or_default();
        self.calls.lock().push(Call {
            batch: ctx.batch.iter().map(batch_message_text).collect(),
            history: ctx.history.iter().map(|i| text_of(&encode_chat_item(&i.payload))).collect(),
            message: text_of(&encode_chat_item(&ctx.message))
                .split('\n')
                .map(str::to_owned)
                .collect(),
        });
        match &self.behavior {
            Behavior::Echo => Ok(text(&format!("reply to: {message_text}"))),
            Behavior::Text(t) => Ok(text(t)),
            Behavior::Counter => {
                let n = ctx.state.as_ref().and_then(|s| s["verified"].as_i64()).unwrap_or(0) + 1;
                Ok(HandlerOutput {
                    output: Some(json!(format!("seen {n}"))),
                    state: Some(json!({"verified": n})),
                    ..HandlerOutput::default()
                })
            }
            Behavior::Raise => Ok(HandlerOutput {
                error: Some("RuntimeError: boom".into()),
                ..HandlerOutput::default()
            }),
            Behavior::Unavailable => Err(ProviderError::Host {
                error: HostError {
                    kind: HostErrorKind::Unavailable,
                    class: "StorageUnavailable".into(),
                    ..HostError::new("the model is down")
                },
                api_kind: None,
                retryable: true,
                status_code: None,
                said: None,
            }),
            Behavior::Slow { entered, gate, output } => {
                entered.set();
                self.updates.lock().push((ctx.task_id.clone(), "working on it".into()));
                gate.wait().await;
                Ok(text(output))
            }
            Behavior::Opener { gate, output } => {
                gate.set();
                Ok(text(output))
            }
            Behavior::Ask => {
                let t = lk_agents_net::clock::unix_now();
                Ok(HandlerOutput {
                    new_items: vec![
                        user_item(&message_text, t),
                        call_item("request_photo", "p1", t + 1e-6),
                    ],
                    input_required: Some("Could you send a photo?".into()),
                    pending_call: Some(PendingCall {
                        name: "request_photo".into(),
                        call_id: "p1".into(),
                        answer_tool: None,
                    }),
                    ..HandlerOutput::default()
                })
            }
        }
    }
}

fn clock() -> Arc<dyn Clock> {
    Arc::new(RealClock::new())
}

fn spawner() -> fn() -> Result<Arc<dyn Spawner>, ProcessError> {
    ProcessEnv::current_runtime().spawner
}

fn options(policy: Option<Policy>) -> TurnOptions {
    TurnOptions {
        policy: policy.map_or(PolicySpec::Default, PolicySpec::All),
        debounce: DebounceSpec::All(Duration::ZERO),
        ..TurnOptions::default()
    }
}

fn turn(s: &Thread, host: Arc<Fake>, policy: Option<Policy>) -> Turn {
    Turn::new(s.clone(), "worker", PARTITION_KEY, host, clock(), spawner(), options(policy))
}

/// `run_turn(storage, thread, PARTITION_KEY, message, policy, handler)` of the Python tests.
async fn run_turn(
    s: &Thread,
    message: Value,
    policy: Option<Policy>,
    host: Arc<Fake>,
) -> Result<TurnResult, StorageError> {
    turn(s, host, policy).run(message, None).await
}

async fn history_texts(s: &Thread) -> Vec<String> {
    s.partition(PARTITION_KEY)
        .history()
        .await
        .unwrap()
        .iter()
        .map(|i| text_of(&i.payload))
        .collect()
}

fn placeholder_id(v: &mut Value, key: &str, prefix: &str) {
    let id = v[key].as_str().expect("an id").to_owned();
    assert!(id.starts_with(prefix) && id.len() > prefix.len(), "{id}");
    v[key] = Value::String(format!("{prefix}X"));
}

async fn two_workers_share_one_session(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread().await;
    let opts = TurnOptions { policy: PolicySpec::All(Policy::Queue), ..TurnOptions::default() };
    let host = Fake::new(Behavior::Counter);
    let first = Turn::new(
        a.clone(),
        "worker",
        PARTITION_KEY,
        host.clone(),
        clock(),
        spawner(),
        opts.clone(),
    )
    .run(a2a("hello", "m1", None, None), None)
    .await
    .unwrap();
    let b = fx.thread().await;
    let second =
        Turn::new(b.clone(), "worker", PARTITION_KEY, host.clone(), clock(), spawner(), opts)
            .run(a2a("again", "m2", None, None), None)
            .await
            .unwrap();

    assert_eq!(first.output, Some(json!("seen 1")));
    assert_eq!(second.output, Some(json!("seen 2")));
    let seen: Vec<Vec<String>> = host.calls().into_iter().map(|c| c.history).collect();
    assert_eq!(seen, vec![vec![], vec!["hello".to_owned(), "seen 1".to_owned()]]);
    assert_eq!(first.task.status, TaskState::Completed);
    assert!(first.task.data.artifacts.as_ref().is_some_and(|a| !a.is_empty()));
    assert_eq!(history_texts(&b).await, vec!["hello", "seen 1", "again", "seen 2"]);
    let runs = b.partition(PARTITION_KEY).runs(10).await.unwrap();
    assert_eq!(
        runs.iter().map(|r| r.outcome).collect::<Vec<_>>(),
        vec![Some(RunOutcome::Completed), Some(RunOutcome::Completed)]
    );
    assert_eq!(runs[0].task_id.as_deref(), Some(second.task.task_id.as_str()));
    assert_eq!(b.partition(PARTITION_KEY).info().await.unwrap().unwrap().version, 2);
    // The state was written at the first commit and loaded by the second worker.
    let run = b
        .partition(PARTITION_KEY)
        .open(RunOptions::new("peek", "AST_peek"), &lk_agents_storage::Holder::new("test"))
        .await
        .unwrap();
    assert_eq!(run.state(), Some(&json!({"verified": 2})));
    drop(run);
    fx.cleanup(&[&a, &b]).await;
}

async fn a_failed_storage_run_preserves_its_original_task(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread().await;
    // The failed run saves no staged history/state; its original accepted task remains
    // queryable with the observed storage failure rather than becoming a worker-loss orphan.
    let failed = run_turn(
        &s,
        a2a("lost", "m1", None, None),
        Some(Policy::Queue),
        Fake::new(Behavior::Unavailable),
    )
    .await
    .expect("the accepted task reports its durable failure");
    assert_eq!(failed.task.status, TaskState::Failed);
    let partition_key = s.partition(PARTITION_KEY);
    assert!(partition_key.history().await.unwrap().is_empty());
    let events = s.journal(0, 100).await.unwrap();
    assert_eq!(
        events.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(),
        vec!["run_started", "run_ended"]
    );
    assert_eq!(
        events[1].payload,
        json!({"outcome": "failed", "reason": "StorageUnavailable: the model is down"})
    );
    let rows = partition_key.tasks(None, 10).await.unwrap();
    assert_eq!((rows.len(), rows[0].status.as_str()), (1, "failed"));
    let info = partition_key.info().await.unwrap().unwrap();
    assert_eq!((info.version, info.lease_expires_at_ms), (0, 0), "released, not bumped");

    // An application error fails its own task; the prior terminal task stays immutable.
    let host = Fake::new(Behavior::Raise);
    let r = run_turn(&s, a2a("boom", "m2", None, None), Some(Policy::Queue), host.clone())
        .await
        .unwrap();
    assert_eq!(r.task.status, TaskState::Failed);
    let mut sm = json_of(&r.task.data.status_message);
    placeholder_id(&mut sm, "messageId", "msg_");
    assert_eq!(
        sm,
        json!({"messageId": "msg_X", "role": "ROLE_AGENT", "parts": [{"text": "RuntimeError: boom"}],
            "taskId": r.task.task_id})
    );
    assert_eq!(r.output, None);
    assert_eq!(host.batches(), vec![vec!["boom".to_owned()]], "the orphan was not run again");
    let orphan = partition_key.task(&rows[0].task_id).await.unwrap().unwrap();
    assert_eq!(orphan.status, TaskState::Failed);
    assert_eq!(
        json_of(&orphan.data.status_message)["metadata"]["https://livekit.io/a2a/ext/agents/v1/reason"],
        json!("storage-unavailable")
    );
    let runs = partition_key.runs(10).await.unwrap();
    assert_eq!(runs[0].reason.as_deref(), Some("RuntimeError: boom"));
    assert_eq!(history_texts(&s).await, vec!["boom"], "the message, no reply, nothing re-run");
    fx.cleanup(&[&s]).await;
}

async fn the_inbox_batches_a_burst_into_one_reply(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread().await;
    let b = fx.thread().await;
    let host = Fake::new(Behavior::Text("one reply"));
    let opts = TurnOptions {
        debounce: DebounceSpec::All(Duration::from_millis(300)),
        ..TurnOptions::default()
    };
    let first = {
        let (a, host, opts) = (a.clone(), host.clone(), opts.clone());
        tokio::spawn(async move {
            Turn::new(a, "worker", PARTITION_KEY, host, clock(), spawner(), opts)
                .run(a2a("where", "m1", Some("sms"), None), None)
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    let second = {
        let (b, host, opts) = (b.clone(), host.clone(), opts);
        tokio::spawn(async move {
            Turn::new(b, "worker", PARTITION_KEY, host, clock(), spawner(), opts)
                .run(a2a("are you", "m2", Some("sms"), None), None)
                .await
        })
    };
    let r1 = first.await.unwrap().unwrap();
    let r2 = second.await.unwrap().unwrap();

    assert_eq!(host.batches(), vec![vec!["where".to_owned(), "are you".to_owned()]]);
    assert_eq!(host.calls()[0].message, vec!["where", "are you"], "ctx.message.content");
    assert_eq!(r1.task.status, TaskState::Completed);
    assert!(r1.task.data.artifacts.as_ref().is_none_or(|a| a.is_empty()));
    assert_eq!(r1.task.data.metadata.get(MERGED_INTO_KEY), Some(&json!(r2.task.task_id)));
    assert_eq!(r1.answered_by, r2.task.task_id);
    assert_eq!(r2.task.status, TaskState::Completed);
    assert_eq!(r2.output, Some(json!("one reply")));
    let items = a.partition(PARTITION_KEY).history().await.unwrap();
    let roles: Vec<ChatRole> = items
        .iter()
        .filter_map(|i| match i.kind() {
            Some(chat_item::Item::Message(m)) => Some(m.role()),
            _ => None,
        })
        .collect();
    assert_eq!(roles, vec![ChatRole::User, ChatRole::Assistant]);
    assert_eq!(text_of(&items[0].payload), "where\nare you");
    let info = a.partition(PARTITION_KEY).info().await.unwrap().unwrap();
    assert_eq!((info.drained_id, info.lease_expires_at_ms), (2, 0));
    fx.cleanup(&[&a, &b]).await;
}

async fn a_message_id_resend_returns_the_existing_task(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread().await;
    let echo = Fake::new(Behavior::Echo);
    let first = run_turn(&s, a2a("hi", "m1", Some("sms"), None), None, echo.clone()).await.unwrap();
    let again = run_turn(&s, a2a("hi", "m1", Some("sms"), None), None, echo.clone()).await.unwrap();
    assert_eq!(again.task.task_id, first.task.task_id);
    assert_eq!(again.output, first.output);
    assert_eq!(again.output, Some(json!("reply to: hi")));
    // under queue the task row holds the message id
    let first =
        run_turn(&s, a2a("yo", "m2", None, None), Some(Policy::Queue), echo.clone()).await.unwrap();
    let again =
        run_turn(&s, a2a("yo", "m2", None, None), Some(Policy::Queue), echo.clone()).await.unwrap();
    assert_eq!(again.task.task_id, first.task.task_id);
    assert_eq!(again.run_id, None, "nothing ran");
    assert_eq!(s.partition(PARTITION_KEY).history().await.unwrap().len(), 4);
    assert_eq!(echo.calls().len(), 2);
    fx.cleanup(&[&s]).await;
}

async fn a_lapsed_lease_fails_the_orphan_worker_lost(backend: Backend) {
    let lease = Some((Duration::from_secs(1), Duration::from_millis(300)));
    let fx = backend.fixture(lease).await;
    let a = fx.thread().await;
    let partition_a = a.partition(PARTITION_KEY);
    let db = &partition_a;
    let message = user_message("first", "m1", Some("sms"), None);
    let data = TaskData { history: vec![message.clone()], ..TaskData::default() };

    // worker A receives, picks the task up, and dies before its commit
    let (received, lease) = db
        .receive(
            &lk_agents_storage::Holder::new("worker-a"),
            lk_agents_storage::thread::db::ReceiveInput {
                task_id: "AST_1",
                message: &message,
                data: &data,
                internal: None,
            },
        )
        .await
        .unwrap();
    assert_eq!(received, Receive::Received(Some(0)));
    assert_eq!(db.pickup("AST_1").await.unwrap(), Some(TaskState::Submitted));
    drop(lease); // A dead worker also drops its renewal task.
    // the lease is one second here; it lapses once unixepoch() moves past lease_expires_at_ms
    tokio::time::sleep(Duration::from_millis(2100)).await;

    // worker B takes the partition for the next message: the orphan fails worker-lost, not run again
    let b = fx.thread().await;
    let host = Fake::new(Behavior::Text("answered"));
    let r2 = tokio::time::timeout(
        Duration::from_secs(10),
        run_turn(&b, a2a("second", "m2", Some("sms"), None), None, host.clone()),
    )
    .await
    .expect("the expired owner's inbox was not recovered")
    .unwrap();
    assert_eq!(host.batches(), vec![vec!["second".to_owned()]]);
    assert_eq!(r2.task.status, TaskState::Completed);

    // the inbox drain picked the row up once more and found it working: a dead holder had it
    let lost = b.partition(PARTITION_KEY).task("AST_1").await.unwrap().unwrap();
    assert_eq!(lost.status, TaskState::Failed);
    let mut sm = json_of(&lost.data.status_message);
    placeholder_id(&mut sm, "messageId", "msg_");
    assert_eq!(
        sm,
        json!({"messageId": "msg_X", "role": "ROLE_AGENT",
            "parts": [{"text": "the task failed: worker-lost"}], "taskId": "AST_1",
            "metadata": {"https://livekit.io/a2a/ext/agents/v1/reason": "worker-lost"}})
    );
    assert_eq!(history_texts(&b).await, vec!["second", "answered"], "the dead turn left nothing");
    fx.cleanup(&[&a, &b]).await;
}

async fn input_required_pending_call_is_the_next_messages_output(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread().await;
    let first = run_turn(
        &s,
        a2a("my car is damaged", "m1", None, None),
        Some(Policy::Queue),
        Fake::new(Behavior::Ask),
    )
    .await
    .unwrap();
    assert_eq!(first.task.status, TaskState::InputRequired);
    assert_eq!(
        first.task.internal.as_ref().and_then(|internal| internal.ask.as_ref()),
        Some(&Ask {
            question: "Could you send a photo?".into(),
            tool: Some("request_photo".into()),
            call_id: Some("p1".into()),
            answer_tool: None,
        })
    );
    let sm = json_of(&first.task.data.status_message);
    assert_eq!(sm["parts"][0]["text"], json!("Could you send a photo?"));
    assert_eq!(first.task.data.history, vec![user_message("my car is damaged", "m1", None, None)]);
    assert_eq!(first.task.message_id.as_deref(), Some("m1"));
    assert_eq!(first.output, None);

    let host = Fake::new(Behavior::Text("thanks"));
    let second =
        run_turn(&s, a2a("here it is", "m2", None, None), Some(Policy::Queue), host.clone())
            .await
            .unwrap();
    assert_eq!(second.output, Some(json!("thanks")));
    let items = s.partition(PARTITION_KEY).history().await.unwrap();
    let outputs: Vec<&FunctionCallOutput> = items
        .iter()
        .filter_map(|i| match i.kind() {
            Some(chat_item::Item::FunctionCallOutput(o)) if o.call_id == "p1" => Some(o),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        (outputs[0].output.as_str(), outputs[0].name.as_str(), outputs[0].is_error),
        ("here it is", "request_photo", false)
    );
    assert_eq!(
        items.iter().map(kind_of).collect::<Vec<_>>(),
        vec!["message", "function_call", "function_call_output", "message", "message"]
    );
    assert_eq!(
        host.calls()[0].history.len(),
        3,
        "the handler sees the question, the call and the answer"
    );
    let answered = s.partition(PARTITION_KEY).task(&first.task.task_id).await.unwrap().unwrap();
    assert_eq!(answered.status, TaskState::Completed);
    assert_eq!(answered.data.metadata.get(MERGED_INTO_KEY), Some(&json!(second.task.task_id)));
    assert!(answered.internal.is_none(), "a completed row holds no pending transition");
    fx.cleanup(&[&s]).await;
}

async fn concurrent_runs_append_without_a_lease(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread().await;
    let b = fx.thread().await;
    let entered = Gate::new();
    let gate = Gate::new();
    let slow =
        Fake::new(Behavior::Slow { entered: entered.clone(), gate: gate.clone(), output: "slow" });
    let fast = Fake::new(Behavior::Opener { gate: gate.clone(), output: "fast" });
    let first = {
        let a = a.clone();
        tokio::spawn(async move {
            run_turn(&a, a2a("one", "m1", None, None), Some(Policy::Concurrent), slow).await
        })
    };
    entered.wait().await;
    let second =
        run_turn(&b, a2a("two", "m2", None, None), Some(Policy::Concurrent), fast).await.unwrap();
    let r1 = first.await.unwrap().unwrap();
    assert_eq!(r1.output, Some(json!("slow")));
    assert_eq!(second.output, Some(json!("fast")));
    let mut texts = history_texts(&a).await;
    texts.sort();
    assert_eq!(texts, vec!["fast", "one", "slow", "two"]);
    let info = a.partition(PARTITION_KEY).info().await.unwrap().unwrap();
    assert_eq!(info.lease_expires_at_ms, 0, "no lease was taken");
    fx.cleanup(&[&a, &b]).await;
}

async fn reject_refuses_while_a_turn_runs(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread().await;
    let b = fx.thread().await;
    let entered = Gate::new();
    let gate = Gate::new();
    let slow =
        Fake::new(Behavior::Slow { entered: entered.clone(), gate: gate.clone(), output: "slow" });
    let first = {
        let (a, slow) = (a.clone(), slow.clone());
        tokio::spawn(async move {
            run_turn(&a, a2a("one", "m1", None, None), Some(Policy::Reject), slow).await
        })
    };
    entered.wait().await;
    let err =
        run_turn(&b, a2a("two", "m2", None, None), Some(Policy::Reject), Fake::new(Behavior::Echo))
            .await
            .expect_err("rejected");
    gate.set();
    let r1 = first.await.unwrap().unwrap();
    match err {
        StorageError::Rejected { running_task_id } => {
            assert_eq!(
                running_task_id.as_deref(),
                Some(r1.task.task_id.as_str()),
                "the error names the running task"
            );
        }
        other => panic!("{other}"),
    }
    assert_eq!(r1.output, Some(json!("slow")));
    assert_eq!(
        slow.updates.lock().clone(),
        vec![(r1.task.task_id.clone(), "working on it".to_owned())],
        "the host's update reached the wire hook"
    );
    fx.cleanup(&[&a, &b]).await;
}

async fn a_submitted_row_exists_while_the_turn_runs(backend: Backend) {
    for policy in [Policy::Queue, Policy::Concurrent, Policy::Reject] {
        let fx = backend.fixture(None).await;
        let a = fx.thread().await;
        let b = fx.thread().await;
        let entered = Gate::new();
        let gate = Gate::new();
        let slow = Fake::new(Behavior::Slow {
            entered: entered.clone(),
            gate: gate.clone(),
            output: "done",
        });
        let running = {
            let a = a.clone();
            tokio::spawn(async move {
                turn(&a, slow, Some(policy))
                    .run(a2a("hello", "m1", None, None), Some("AST_early".into()))
                    .await
            })
        };
        entered.wait().await;
        // another worker, before the commit: the row is there, with the message
        let partition_key = b.partition(PARTITION_KEY);
        let during = partition_key.task("AST_early").await.unwrap().expect("the early row");
        let status =
            if policy == Policy::Concurrent { TaskState::Submitted } else { TaskState::Working };
        assert_eq!(during.status, status, "{policy}");
        assert_eq!(during.message_id.as_deref(), Some("m1"));
        assert_eq!(during.data.history, vec![user_message("hello", "m1", None, None)]);
        assert_eq!(during.pos, 1);
        assert_eq!((during.name.as_ref(), during.internal.as_ref()), (None, None));
        assert!(during.data.metadata.is_empty());

        gate.set();
        let result = running.await.unwrap().unwrap();
        let after = partition_key.task("AST_early").await.unwrap().unwrap();
        assert_eq!((after.status, after.pos), (TaskState::Completed, 1), "{policy}");
        assert_eq!(result.task.task_id, "AST_early");
        assert_eq!(result.output, Some(json!("done")));
        let mut artifacts = json_of(&after.data.artifacts);
        placeholder_id(&mut artifacts[0], "artifactId", "art_");
        assert_eq!(artifacts, json!([{"artifactId": "art_X", "parts": [{"text": "done"}]}]));
        let mut sm = json_of(&after.data.status_message);
        placeholder_id(&mut sm, "messageId", "msg_");
        assert_eq!(
            sm,
            json!({"messageId": "msg_X", "role": "ROLE_AGENT", "parts": [{"text": "done"}],
                "taskId": "AST_early"})
        );
        assert_eq!(
            partition_key.tasks(None, 50).await.unwrap().len(),
            1,
            "the commit moved the early row"
        );
        // a resend of the message inserts no second row
        let inserted = b
            .partition(PARTITION_KEY)
            .submit(&TaskWrite {
                task_id: "AST_other".into(),
                status: TaskState::Submitted,
                message_id: Some("m1".into()),
                name: None,
                data: TaskData::default(),
                internal: None,
            })
            .await
            .unwrap();
        assert!(!inserted);
        fx.cleanup(&[&a, &b]).await;
    }
}

async fn a_lease_timeout_fails_the_early_row(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread().await;
    let b = fx.thread().await;
    let entered = Gate::new();
    let gate = Gate::new();
    let slow =
        Fake::new(Behavior::Slow { entered: entered.clone(), gate: gate.clone(), output: "slow" });
    let first = {
        let a = a.clone();
        tokio::spawn(async move {
            run_turn(&a, a2a("one", "m1", None, None), Some(Policy::Queue), slow).await
        })
    };
    entered.wait().await;
    let short = Turn::new(
        b.clone(),
        "worker",
        PARTITION_KEY,
        Fake::new(Behavior::Echo),
        clock(),
        spawner(),
        TurnOptions {
            policy: PolicySpec::All(Policy::Queue),
            wait_timeout: Some(Duration::from_millis(50)),
            ..TurnOptions::default()
        },
    );
    let second = short.run(a2a("two", "m2", None, None), Some("AST_2".into())).await.unwrap();
    assert_eq!(second.task.status, TaskState::Failed);
    assert_eq!(
        json_of(&second.task.data.status_message)["metadata"]["https://livekit.io/a2a/ext/agents/v1/reason"],
        json!("lease-timeout")
    );
    assert_eq!(second.run_id, None);
    let row = b
        .partition(PARTITION_KEY)
        .task("AST_2")
        .await
        .unwrap()
        .expect("persisted, not the unwritten stand-in");
    assert_eq!((row.status, row.pos), (TaskState::Failed, 2));
    assert_eq!(row.data.history, vec![user_message("two", "m2", None, None)]);
    gate.set();
    first.await.unwrap().unwrap();
    let mut statuses: Vec<&str> = b
        .partition(PARTITION_KEY)
        .tasks(None, 50)
        .await
        .unwrap()
        .iter()
        .map(|t| t.status.as_str())
        .collect();
    statuses.sort();
    assert_eq!(statuses, vec!["completed", "failed"]);
    fx.cleanup(&[&a, &b]).await;
}

async fn a_rejected_message_leaves_no_submitted_row(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread().await;
    let b = fx.thread().await;
    let entered = Gate::new();
    let gate = Gate::new();
    let slow =
        Fake::new(Behavior::Slow { entered: entered.clone(), gate: gate.clone(), output: "slow" });
    let first = {
        let a = a.clone();
        tokio::spawn(async move {
            run_turn(&a, a2a("one", "m1", None, None), Some(Policy::Reject), slow).await
        })
    };
    entered.wait().await;
    // the partition reads as leased, so the second is refused before any row is written
    let err = turn(&b, Fake::new(Behavior::Echo), Some(Policy::Reject))
        .run(a2a("two", "m2", None, None), Some("AST_2".into()))
        .await
        .expect_err("rejected");
    assert!(matches!(err, StorageError::Rejected { .. }), "{err}");
    assert!(b.partition(PARTITION_KEY).task("AST_2").await.unwrap().is_none());
    gate.set();
    first.await.unwrap().unwrap();
    let statuses: Vec<TaskState> = b
        .partition(PARTITION_KEY)
        .tasks(None, 50)
        .await
        .unwrap()
        .into_iter()
        .map(|t| t.status)
        .collect();
    assert_eq!(statuses, vec![TaskState::Completed]);
    fx.cleanup(&[&a, &b]).await;
}

lk_agents_storage::contract!(
    two_workers_share_one_session,
    a_failed_storage_run_preserves_its_original_task,
    the_inbox_batches_a_burst_into_one_reply,
    a_message_id_resend_returns_the_existing_task,
    a_lapsed_lease_fails_the_orphan_worker_lost,
    input_required_pending_call_is_the_next_messages_output,
    concurrent_runs_append_without_a_lease,
    reject_refuses_while_a_turn_runs,
    a_submitted_row_exists_while_the_turn_runs,
    a_lease_timeout_fails_the_early_row,
    a_rejected_message_leaves_no_submitted_row,
);

/// A refused save fails the original task `storage-unavailable`, its staged changes absent: no
/// new task, no answer in the history. A save refused as too large does too: it reached the
/// client as the raw `TooLarge`, the task left `working`.
#[tokio::test]
async fn a_refused_commit_fails_the_original_task_storage_unavailable() {
    use lk_agents_storage::testing::WriteFault;
    for fault in [WriteFault::Refuse, WriteFault::RefuseTooLarge] {
        let fx = Backend::Sqlite.fixture(None).await;
        let (s, faults) = fx.faulty_thread().await;
        faults.fail_next_commit(fault);
        let host = Fake::new(Behavior::Echo);
        let r = turn(&s, host, Some(Policy::Queue))
            .run(a2a("hello", "m1", None, None), Some("AST_1".into()))
            .await
            .unwrap();
        assert_eq!((r.task.task_id.as_str(), r.task.status), ("AST_1", TaskState::Failed));
        assert_eq!(
            json_of(&r.task.data.status_message)["metadata"]["https://livekit.io/a2a/ext/agents/v1/reason"],
            json!("storage-unavailable")
        );
        assert!(history_texts(&s).await.is_empty(), "the staged turn was not committed");
        assert_eq!(
            s.partition(PARTITION_KEY).tasks(None, 10).await.unwrap().len(),
            1,
            "no second task"
        );
        fx.cleanup(&[&s]).await;
    }
}

/// An acknowledgment lost with the database unreachable through the whole wait
/// answers the original task still `working`, never a failure; the resolver later finds the
/// commit and the task reads completed.
#[tokio::test]
async fn an_unresolved_commit_answers_working_and_resolves_to_the_original_run() {
    let fx = Backend::Sqlite.fixture(None).await;
    let (s, faults) = fx.faulty_thread().await;
    faults.fail_next_commit(lk_agents_storage::testing::WriteFault::LoseAckAfter);
    faults.unreachable_after_commit_for(Duration::from_secs(12));
    let host = Fake::new(Behavior::Echo);
    let r = turn(&s, host, Some(Policy::Queue))
        .run(a2a("hello", "m1", None, None), Some("AST_1".into()))
        .await
        .unwrap();
    assert_eq!((r.task.task_id.as_str(), r.task.status), ("AST_1", TaskState::Working));

    let reader = fx.thread().await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let row = loop {
        let row = reader.partition(PARTITION_KEY).task("AST_1").await.unwrap().unwrap();
        if row.status != TaskState::Working || tokio::time::Instant::now() > deadline {
            break row;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    assert_eq!(row.status, TaskState::Completed, "the original run's commit");
    assert_eq!(history_texts(&reader).await, vec!["hello", "reply to: hello"]);
    assert_eq!(
        reader.partition(PARTITION_KEY).tasks(None, 10).await.unwrap().len(),
        1,
        "no second task"
    );
    fx.cleanup(&[&s, &reader]).await;
}

/// I-8: the message's `submitted` row is durable before the debounce, not after it.
#[tokio::test]
async fn the_submitted_row_is_durable_before_the_debounce() {
    let fx = Backend::Sqlite.fixture(None).await;
    let s = fx.thread().await;
    let opts = TurnOptions {
        policy: PolicySpec::All(Policy::Queue),
        debounce: DebounceSpec::All(Duration::from_secs(1)),
        ..TurnOptions::default()
    };
    let turn = Turn::new(
        s.clone(),
        "worker",
        PARTITION_KEY,
        Fake::new(Behavior::Echo),
        clock(),
        spawner(),
        opts,
    );
    let running = tokio::spawn(async move {
        turn.run(a2a("hello", "m1", Some("sms"), None), Some("AST_1".into())).await
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let row = s
        .partition(PARTITION_KEY)
        .task("AST_1")
        .await
        .unwrap()
        .expect("durable before the debounce");
    assert_eq!(row.status, TaskState::Submitted);
    let done = running.await.unwrap().unwrap();
    assert_eq!(done.task.status, TaskState::Completed);
    fx.cleanup(&[&s]).await;
}

mod attachments {
    // Attachments cross the managed turn as application references, never inline upload bytes.
    use std::sync::Arc;

    use crate::a2a::Turn;
    use lk_agents_abi::a2a::client::MessagePart;
    use lk_agents_abi::a2a::turn::{HandlerOutput, HandlerPort, TurnContext, TurnHost};
    use lk_agents_abi::errors::ProviderError;
    use lk_agents_core::a2a::{Policy, PolicySpec, TurnOptions};
    use lk_agents_net::clock::RealClock;
    use lk_agents_storage::StorageError;
    use lk_agents_storage::backends::sqlite::SqliteStorage;
    use lk_agents_storage::{Storage, shortuuid};
    use serde_json::json;

    struct Host;

    impl TurnHost for Host {
        fn handle<'a>(
            &'a self,
            _cx: &lk_agents_abi::telemetry::Cx,
            ctx: &'a TurnContext,
            _turn: Arc<dyn HandlerPort>,
            _cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
        ) -> lk_agents_abi::BoxFuture<'a, Result<HandlerOutput, ProviderError>> {
            Box::pin(async move {
                let parts = &ctx.batch[0].parts;
                assert!(
                    matches!(&parts[0], MessagePart::TextPart { text, .. } if text == "review")
                );
                for (part, expected) in
                    parts[1..].iter().zip(["private photo reference", "private document reference"])
                {
                    assert!(
                        matches!(part, MessagePart::FilePart { url: Some(url), data: None, .. } if url == expected)
                    );
                }
                Ok(HandlerOutput {
                    output: Some(json!("received references")),
                    ..Default::default()
                })
            })
        }
    }

    #[tokio::test]
    async fn sqlite_restart_keeps_references_but_no_inline_upload_in_task_or_inbox() {
        let dir = std::env::temp_dir().join(shortuuid("lk-attachment-boundary-"));
        let storage =
            SqliteStorage::open(&dir, Default::default(), lk_agents_storage::testing::threads())
                .await
                .unwrap();
        let created = storage.create_thread(Default::default()).await.unwrap();
        let thread = storage.open(&created.thread_id).await.unwrap();
        let turn = Turn::new(
            thread.clone(),
            "worker",
            "desk",
            Arc::new(Host),
            Arc::new(RealClock::new()),
            lk_agents_net::env::ProcessEnv::current_runtime().spawner,
            TurnOptions { policy: PolicySpec::All(Policy::Queue), ..Default::default() },
        );
        for mime in ["image/png", "audio/wav", "application/pdf"] {
            let result = turn.run(json!({"messageId": "inline", "role": "ROLE_USER", "parts": [{"raw": "dXBsb2FkZWQgYnl0ZXM=", "mediaType": mime}]}), None).await;
            assert!(matches!(result, Err(StorageError::Invalid(_))));
        }
        assert!(thread.partition("desk").tasks(None, 10).await.unwrap().is_empty());
        assert!(thread.partition("desk").inbox(false).await.unwrap().is_empty());
        let parts = json!([
            {"text": "review"},
            {"url": "private photo reference", "mediaType": "image/png", "filename": "photo.png"},
            {"url": "private document reference", "mediaType": "application/pdf", "filename": "receipt.pdf"},
        ]);
        let result = turn
            .run(json!({"messageId": "references", "role": "ROLE_USER", "parts": parts}), None)
            .await
            .unwrap();
        let task_id = result.task.task_id;
        drop(turn);
        thread.close().await;
        storage.close().await;
        drop(storage);

        let reopened =
            SqliteStorage::open(&dir, Default::default(), lk_agents_storage::testing::threads())
                .await
                .unwrap();
        let thread = reopened.open(&created.thread_id).await.unwrap();
        let task = thread.task(&task_id).await.unwrap().unwrap();
        assert_eq!(serde_json::to_value(&task.data.history[0]).unwrap()["parts"], parts);
        let inbox = thread.partition("desk").inbox(false).await.unwrap();
        assert!(inbox.is_empty(), "the successful turn drained its inbox rows");
        let durable =
            serde_json::to_string(&task).unwrap() + &serde_json::to_string(&inbox).unwrap();
        assert!(!durable.contains("dXBsb2FkZWQgYnl0ZXM="));
        thread.close().await;
        reopened.delete_thread(&created.thread_id).await.unwrap();
        reopened.close().await;
        std::fs::remove_dir_all(dir).unwrap();
    }
}
