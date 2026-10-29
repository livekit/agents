//! The A2A door's turns against both storage backends (`storage_common`). The host is a scripted fake: a handler that returns
//! a text, raises, asks input-required with a pending call, or sleeps until told, and records what
//! it was handed. The cases are those of `tests/test_storage.py`, with its assertions.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use livekit_protocol::agent::{ChatMessage, ChatRole, FunctionCall, FunctionCallOutput};
use lk_agents_abi::a2a::turn::TurnContext;
use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::{Clock, Closed, RealClock, StreamSink, StreamSource, Ts};
use lk_agents_core::a2a::{
    DebounceSpec, HandlerOutput, PendingCall, Policy, PolicySpec, TurnEvent, TurnHost, TurnOptions,
    TurnResult, Verdict,
};
use lk_agents_runtime::a2a::Turn;
use lk_agents_storage::TaskWrite;
use lk_agents_storage::db::Receive;
use lk_agents_storage::run::{
    INTERRUPTED_KEY, INTERRUPTION_REASON_KEY, ItemRow, MERGED_INTO_KEY, RunOutcome, TaskState,
};
use lk_agents_storage::schema::{Ask, TaskData};
use lk_agents_storage::testing::Backend;
use lk_agents_storage::timestamp;
use lk_agents_storage::{RunOptions, Thread, shortuuid};

use lk_agents_storage::StorageError;
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};

const LANE: &str = "front-desk";

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

fn assistant_item(text: &str, created_at: f64) -> ChatItem {
    message(ChatRole::Assistant, text, created_at)
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

fn output_item(name: &str, call_id: &str, output: &str, created_at: f64) -> ChatItem {
    ChatItem {
        item: Some(chat_item::Item::FunctionCallOutput(FunctionCallOutput {
            id: shortuuid("item_"),
            name: name.into(),
            call_id: call_id.into(),
            output: output.into(),
            is_error: false,
            created_at: Some(timestamp(created_at)),
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
    /// A tool call that waits for `gate`, then the boundary: yields with what ran, or answers.
    Tool {
        entered: Gate,
        gate: Gate,
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
        self.calls.lock().unwrap().clone()
    }

    fn batches(&self) -> Vec<Vec<String>> {
        self.calls().into_iter().map(|c| c.batch).collect()
    }
}

fn text(output: &str) -> HandlerOutput {
    HandlerOutput { output: Some(json!(output)), ..HandlerOutput::default() }
}

/// The turn's verdicts, as the host's sink.
struct Verdicts(mpsc::UnboundedSender<Verdict>);

impl StreamSink<Verdict> for Verdicts {
    fn push(&self, item: Verdict) -> Result<(), ProviderError> {
        let _ = self.0.send(item);
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

/// The handler's events, as the host's source.
struct Events(mpsc::UnboundedReceiver<Result<TurnEvent, ProviderError>>);

impl StreamSource<TurnEvent> for Events {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<TurnEvent>, ProviderError>> {
        Box::pin(async move { self.0.recv().await.transpose() })
    }
    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

/// A checkpoint of the fake's run: `true` when the turn answered `yield`.
struct Checkpoints {
    events: mpsc::UnboundedSender<Result<TurnEvent, ProviderError>>,
    verdicts: mpsc::UnboundedReceiver<Verdict>,
}

impl Checkpoints {
    async fn check(&mut self) -> bool {
        let _ = self.events.send(Ok(TurnEvent::Checkpoint));
        self.verdicts.recv().await == Some(Verdict::Yield)
    }
}

impl TurnHost for Fake {
    #[allow(clippy::type_complexity)]
    fn handle<'a>(
        &'a self,
        ctx: &'a TurnContext,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>,
    > {
        Box::pin(async move {
            let (verdicts, verdicts_rx) = mpsc::unbounded_channel();
            let (events, events_rx) = mpsc::unbounded_channel();
            let fake = self.clone();
            let ctx = ctx.clone();
            tokio::spawn(async move {
                let mut checkpoints = Checkpoints { events: events.clone(), verdicts: verdicts_rx };
                let out = fake
                    .run(ctx, &mut checkpoints)
                    .await
                    .map(|out| TurnEvent::Output { output: out.into() });
                let _ = events.send(out);
            });
            Ok((
                Box::new(Verdicts(verdicts)) as Box<dyn StreamSink<_>>,
                Box::new(Events(events_rx)) as Box<dyn StreamSource<_>>,
            ))
        })
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
    async fn run(
        &self,
        ctx: TurnContext,
        checkpoints: &mut Checkpoints,
    ) -> Result<HandlerOutput, ProviderError> {
        let message_text = ctx.batch.last().map(batch_message_text).unwrap_or_default();
        self.calls.lock().unwrap().push(Call {
            batch: ctx.batch.iter().map(batch_message_text).collect(),
            history: ctx
                .history
                .iter()
                .map(|i| text_of(&serde_json::from_value(i.payload.clone()).expect("chat item")))
                .collect(),
            message: text_of(
                &serde_json::from_value(serde_json::Value::Object(ctx.message.clone()))
                    .expect("chat item"),
            )
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
                    kind: "StorageUnavailable".into(),
                    message: "the model is down".into(),
                    host_ref: None,
                },
                api_kind: None,
                retryable: true,
                status_code: None,
            }),
            Behavior::Slow { entered, gate, output } => {
                entered.set();
                self.updates.lock().unwrap().push((ctx.task_id.clone(), "working on it".into()));
                gate.wait().await;
                Ok(text(output))
            }
            Behavior::Opener { gate, output } => {
                gate.set();
                Ok(text(output))
            }
            Behavior::Tool { entered, gate } => {
                let t = lk_agents_abi::unix_now();
                let mut items =
                    vec![user_item(&message_text, t), call_item("lookup", "c1", t + 1e-6)];
                entered.set();
                gate.wait().await;
                items.push(output_item("lookup", "c1", "found", t + 2e-6));
                if checkpoints.check().await {
                    return Ok(HandlerOutput {
                        new_items: items,
                        interrupted: true,
                        ..HandlerOutput::default()
                    });
                }
                items.push(assistant_item("never sent", t + 3e-6));
                Ok(HandlerOutput {
                    output: Some(json!("never sent")),
                    new_items: items,
                    ..HandlerOutput::default()
                })
            }
            Behavior::Ask => {
                let t = lk_agents_abi::unix_now();
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

fn options(policy: Option<Policy>) -> TurnOptions {
    TurnOptions {
        policy: policy.map_or(PolicySpec::Default, PolicySpec::All),
        debounce: DebounceSpec::All(Duration::ZERO),
        ..TurnOptions::default()
    }
}

fn turn(s: &Thread, host: Arc<Fake>, policy: Option<Policy>) -> Turn {
    Turn::new(s.clone(), LANE, host, clock(), options(policy))
}

/// `run_turn(storage, thread, LANE, message, policy, handler)` of the Python tests.
async fn run_turn(
    s: &Thread,
    message: Value,
    policy: Option<Policy>,
    host: Arc<Fake>,
) -> Result<TurnResult, StorageError> {
    turn(s, host, policy).run(message, None).await
}

async fn history_texts(s: &Thread) -> Vec<String> {
    s.lane(LANE).history().await.unwrap().iter().map(|i| text_of(&i.payload)).collect()
}

fn placeholder_id(v: &mut Value, key: &str, prefix: &str) {
    let id = v[key].as_str().expect("an id").to_owned();
    assert!(id.starts_with(prefix) && id.len() > prefix.len(), "{id}");
    v[key] = Value::String(format!("{prefix}X"));
}

// -- the thin slice --

async fn two_workers_share_one_session(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread("worker-a").await;
    let opts = TurnOptions { policy: PolicySpec::All(Policy::Queue), ..TurnOptions::default() };
    let host = Fake::new(Behavior::Counter);
    let first = Turn::new(a.clone(), LANE, host.clone(), clock(), opts.clone())
        .run(a2a("hello", "m1", None, None), None)
        .await
        .unwrap();
    let b = fx.thread("worker-b").await;
    let second = Turn::new(b.clone(), LANE, host.clone(), clock(), opts)
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
    let runs = b.lane(LANE).runs(10).await.unwrap();
    assert_eq!(
        runs.iter().map(|r| r.outcome).collect::<Vec<_>>(),
        vec![Some(RunOutcome::Completed), Some(RunOutcome::Completed)]
    );
    assert_eq!(runs[0].task_id.as_deref(), Some(second.task.task_id.as_str()));
    assert_eq!(b.lane(LANE).info().await.unwrap().unwrap().version, 2);
    // The state was written at the first commit and loaded by the second worker.
    let run = b.lane(LANE).open(RunOptions::new("peek", "AST_peek")).await.unwrap();
    assert_eq!(run.state, Some(json!({"verified": 2})));
    drop(run);
    fx.cleanup(&[&a, &b]).await;
}

async fn a_failed_storage_run_preserves_its_original_task(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("worker-a").await;
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
    let lane = s.lane(LANE);
    assert!(lane.history().await.unwrap().is_empty());
    let events = s.journal(0, 100).await.unwrap();
    assert_eq!(
        events.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(),
        vec!["run_started", "run_ended"]
    );
    assert_eq!(
        events[1].payload,
        json!({"outcome": "failed", "reason": "StorageUnavailable: the model is down"})
    );
    let rows = lane.tasks(None, 10).await.unwrap();
    assert_eq!((rows.len(), rows[0].status.as_str()), (1, "failed"));
    let info = lane.info().await.unwrap().unwrap();
    assert_eq!((info.version, info.lease_until), (0, 0), "released, not bumped");

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
    let orphan = lane.task(&rows[0].task_id).await.unwrap().unwrap();
    assert_eq!(orphan.status, TaskState::Failed);
    assert_eq!(
        json_of(&orphan.data.status_message)["metadata"]["https://livekit.io/a2a/ext/agents/v1/reason"],
        json!("storage-unavailable")
    );
    let runs = lane.runs(10).await.unwrap();
    assert_eq!(runs[0].reason.as_deref(), Some("RuntimeError: boom"));
    assert_eq!(history_texts(&s).await, vec!["boom"], "the message, no reply, nothing re-run");
    fx.cleanup(&[&s]).await;
}

// -- interrupt --

async fn interrupt_yields_and_the_newer_task_answers_both(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread("worker-a").await;
    let b = fx.thread("worker-b").await;
    let tool_entered = Gate::new();
    let waiter_set = Gate::new();
    let host_a =
        Fake::new(Behavior::Tool { entered: tool_entered.clone(), gate: waiter_set.clone() });
    let host_b = Fake::new(Behavior::Text("both answered"));

    let first = {
        let (a, host_a) = (a.clone(), host_a.clone());
        tokio::spawn(async move {
            run_turn(&a, a2a("first", "m1", None, None), Some(Policy::Interrupt), host_a).await
        })
    };
    tool_entered.wait().await;
    let second = {
        let (b, host_b) = (b.clone(), host_b.clone());
        tokio::spawn(async move {
            run_turn(&b, a2a("second", "m2", None, None), Some(Policy::Interrupt), host_b).await
        })
    };
    // B fails the lease and records itself as waiting; A sees it at its next boundary
    let mut found = false;
    for _ in 0..200 {
        let info = b.lane(LANE).info().await.unwrap();
        if info.is_some_and(|i| i.waiter.as_deref().is_some_and(|w| w.starts_with("worker-b/"))) {
            found = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(found, "the waiter was never set");
    waiter_set.set();

    let r1 = first.await.unwrap().unwrap();
    let r2 = second.await.unwrap().unwrap();
    assert_eq!(r1.task.status, TaskState::Canceled);
    assert_eq!(r1.task.data.metadata.get(INTERRUPTED_KEY), Some(&json!(true)));
    assert_eq!(r1.task.data.metadata.get(INTERRUPTION_REASON_KEY), Some(&json!("new-message")));
    assert_eq!(
        serde_json::to_string(&r1.task.data.metadata).unwrap(),
        r#"{"https://livekit.io/a2a/ext/agents/v1/interrupted":true,"https://livekit.io/a2a/ext/agents/v1/interruptionReason":"new-message"}"#
    );
    assert!(r1.task.data.artifacts.is_none() && r1.output.is_none());
    assert_eq!(r1.task.data.history, vec![user_message("first", "m1", None, None)]);
    assert_eq!(r2.task.status, TaskState::Completed);
    assert_eq!(r2.output, Some(json!("both answered")));
    // what ran is in the history, with no reply from the first turn
    let items = a.lane(LANE).history().await.unwrap();
    assert_eq!(
        items.iter().map(kind_of).collect::<Vec<_>>(),
        vec!["message", "function_call", "function_call_output", "message", "message"]
    );
    assert_eq!(text_of(&items[4].payload), "both answered");
    assert_eq!(host_b.calls()[0].history.len(), 3, "B loaded what A committed");
    let runs = a.lane(LANE).runs(10).await.unwrap();
    let outcomes: Vec<_> = runs.iter().filter_map(|r| r.outcome).collect();
    assert_eq!(outcomes.len(), 2);
    assert!(outcomes.contains(&RunOutcome::Canceled) && outcomes.contains(&RunOutcome::Completed));
    fx.cleanup(&[&a, &b]).await;
}

// -- inbox --

async fn the_inbox_batches_a_burst_into_one_reply(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread("worker-a").await;
    let b = fx.thread("worker-b").await;
    let host = Fake::new(Behavior::Text("one reply"));
    let opts = TurnOptions {
        debounce: DebounceSpec::All(Duration::from_millis(300)),
        ..TurnOptions::default()
    };
    let first = {
        let (a, host, opts) = (a.clone(), host.clone(), opts.clone());
        tokio::spawn(async move {
            Turn::new(a, LANE, host, clock(), opts)
                .run(a2a("where", "m1", Some("sms"), None), None)
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    let second = {
        let (b, host, opts) = (b.clone(), host.clone(), opts);
        tokio::spawn(async move {
            Turn::new(b, LANE, host, clock(), opts)
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
    let items = a.lane(LANE).history().await.unwrap();
    let roles: Vec<ChatRole> = items
        .iter()
        .filter_map(|i| match i.kind() {
            Some(chat_item::Item::Message(m)) => Some(m.role()),
            _ => None,
        })
        .collect();
    assert_eq!(roles, vec![ChatRole::User, ChatRole::Assistant]);
    assert_eq!(text_of(&items[0].payload), "where\nare you");
    let info = a.lane(LANE).info().await.unwrap().unwrap();
    assert_eq!((info.drained_id, info.lease_until), (2, 0));
    fx.cleanup(&[&a, &b]).await;
}

async fn a_message_id_resend_returns_the_existing_task(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("worker-a").await;
    let echo = Fake::new(Behavior::Echo);
    let first = run_turn(&s, a2a("hi", "m1", Some("sms"), None), None, echo.clone()).await.unwrap();
    let again = run_turn(&s, a2a("hi", "m1", Some("sms"), None), None, echo.clone()).await.unwrap();
    assert_eq!(again.task.task_id, first.task.task_id);
    assert_eq!(again.output, first.output);
    assert_eq!(again.output, Some(json!("reply to: hi")));
    // under interrupt the task row holds the message id
    let first = run_turn(&s, a2a("yo", "m2", None, None), Some(Policy::Interrupt), echo.clone())
        .await
        .unwrap();
    let again = run_turn(&s, a2a("yo", "m2", None, None), Some(Policy::Interrupt), echo.clone())
        .await
        .unwrap();
    assert_eq!(again.task.task_id, first.task.task_id);
    assert_eq!(again.run_id, None, "nothing ran");
    assert_eq!(s.lane(LANE).history().await.unwrap().len(), 4);
    assert_eq!(echo.calls().len(), 2);
    fx.cleanup(&[&s]).await;
}

// -- the lease --

async fn a_lapsed_lease_fails_the_orphan_worker_lost(backend: Backend) {
    let lease = Some((Duration::from_secs(1), Duration::from_millis(300)));
    let fx = backend.fixture(lease).await;
    let a = fx.thread("worker-a").await;
    let lane_a = a.lane(LANE);
    let db = &lane_a;
    let message = user_message("first", "m1", Some("sms"), None);
    let data = TaskData { history: vec![message.clone()], ..TaskData::default() };

    // worker A receives, picks the task up, and dies before its commit
    let (received, _lease) = db
        .receive(
            "worker-a",
            lk_agents_storage::db::ReceiveInput {
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
    // the lease is one second here; it lapses once unixepoch() moves past lease_until
    tokio::time::sleep(Duration::from_millis(2100)).await;

    // worker B takes the lane for the next message: the orphan fails worker-lost, not run again
    let b = fx.thread("worker-b").await;
    let host = Fake::new(Behavior::Text("answered"));
    let r2 =
        run_turn(&b, a2a("second", "m2", Some("sms"), None), None, host.clone()).await.unwrap();
    assert_eq!(host.batches(), vec![vec!["second".to_owned()]]);
    assert_eq!(r2.task.status, TaskState::Completed);

    // the inbox drain picked the row up once more and found it working: a dead holder had it
    let lost = b.lane(LANE).task("AST_1").await.unwrap().unwrap();
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

// -- input required --

async fn input_required_pending_call_is_the_next_messages_output(backend: Backend) {
    let fx = backend.fixture(None).await;
    let s = fx.thread("worker-a").await;
    let first = run_turn(
        &s,
        a2a("my car is damaged", "m1", None, None),
        Some(Policy::Interrupt),
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
        run_turn(&s, a2a("here it is", "m2", None, None), Some(Policy::Interrupt), host.clone())
            .await
            .unwrap();
    assert_eq!(second.output, Some(json!("thanks")));
    let items = s.lane(LANE).history().await.unwrap();
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
    let answered = s.lane(LANE).task(&first.task.task_id).await.unwrap().unwrap();
    assert_eq!(answered.status, TaskState::Completed);
    assert_eq!(answered.data.metadata.get(MERGED_INTO_KEY), Some(&json!(second.task.task_id)));
    assert!(answered.internal.is_none(), "a completed row holds no pending transition");
    fx.cleanup(&[&s]).await;
}

// -- concurrent and reject --

async fn concurrent_runs_append_without_a_lease(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread("worker-a").await;
    let b = fx.thread("worker-b").await;
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
    let info = a.lane(LANE).info().await.unwrap().unwrap();
    assert_eq!(info.lease_until, 0, "no lease was taken");
    fx.cleanup(&[&a, &b]).await;
}

async fn reject_refuses_while_a_turn_runs(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread("worker-a").await;
    let b = fx.thread("worker-b").await;
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
        slow.updates.lock().unwrap().clone(),
        vec![(r1.task.task_id.clone(), "working on it".to_owned())],
        "the host's update reached the wire hook"
    );
    fx.cleanup(&[&a, &b]).await;
}

async fn a_submitted_row_exists_while_the_turn_runs(backend: Backend) {
    for policy in [Policy::Interrupt, Policy::Queue, Policy::Concurrent, Policy::Reject] {
        let fx = backend.fixture(None).await;
        let a = fx.thread("worker-a").await;
        let b = fx.thread("worker-b").await;
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
        let lane = b.lane(LANE);
        let during = lane.task("AST_early").await.unwrap().expect("the early row");
        let status =
            if policy == Policy::Concurrent { TaskState::Submitted } else { TaskState::Working };
        assert_eq!(during.status, status, "{policy}");
        assert_eq!(during.message_id.as_deref(), Some("m1"));
        assert_eq!(during.data.history, vec![user_message("hello", "m1", None, None)]);
        assert_eq!(during.seq, 1);
        assert_eq!((during.name.as_ref(), during.internal.as_ref()), (None, None));
        assert!(during.data.metadata.is_empty());

        gate.set();
        let result = running.await.unwrap().unwrap();
        let after = lane.task("AST_early").await.unwrap().unwrap();
        assert_eq!((after.status, after.seq), (TaskState::Completed, 1), "{policy}");
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
        assert_eq!(lane.tasks(None, 50).await.unwrap().len(), 1, "the commit moved the early row");
        // a resend of the message inserts no second row
        let inserted = b
            .lane(LANE)
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
    let a = fx.thread("worker-a").await;
    let b = fx.thread("worker-b").await;
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
        LANE,
        Fake::new(Behavior::Echo),
        clock(),
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
    let row =
        b.lane(LANE).task("AST_2").await.unwrap().expect("persisted, not the unwritten stand-in");
    assert_eq!((row.status, row.seq), (TaskState::Failed, 2));
    assert_eq!(row.data.history, vec![user_message("two", "m2", None, None)]);
    gate.set();
    first.await.unwrap().unwrap();
    let mut statuses: Vec<&str> =
        b.lane(LANE).tasks(None, 50).await.unwrap().iter().map(|t| t.status.as_str()).collect();
    statuses.sort();
    assert_eq!(statuses, vec!["completed", "failed"]);
    fx.cleanup(&[&a, &b]).await;
}

async fn a_rejected_message_leaves_no_submitted_row(backend: Backend) {
    let fx = backend.fixture(None).await;
    let a = fx.thread("worker-a").await;
    let b = fx.thread("worker-b").await;
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
    // the lane reads as leased, so the second is refused before any row is written
    let err = turn(&b, Fake::new(Behavior::Echo), Some(Policy::Reject))
        .run(a2a("two", "m2", None, None), Some("AST_2".into()))
        .await
        .expect_err("rejected");
    assert!(matches!(err, StorageError::Rejected { .. }), "{err}");
    assert!(b.lane(LANE).task("AST_2").await.unwrap().is_none());
    gate.set();
    first.await.unwrap().unwrap();
    let statuses: Vec<TaskState> =
        b.lane(LANE).tasks(None, 50).await.unwrap().into_iter().map(|t| t.status).collect();
    assert_eq!(statuses, vec![TaskState::Completed]);
    fx.cleanup(&[&a, &b]).await;
}

lk_agents_storage::contract!(
    two_workers_share_one_session,
    a_failed_storage_run_preserves_its_original_task,
    interrupt_yields_and_the_newer_task_answers_both,
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

// -- I-7: commit outcomes reach the caller as they are (SQLite, injected faults) --

/// A refused save fails the original task `storage-unavailable`, its staged changes absent: no
/// new task, no answer in the history. A save refused as too large does too: it reached the
/// client as the raw `TooLarge`, the task left `working`.
#[tokio::test]
async fn a_refused_commit_fails_the_original_task_storage_unavailable() {
    use lk_agents_storage::testing::CommitFault;
    for fault in [CommitFault::Refuse, CommitFault::RefuseTooLarge] {
        let fx = Backend::Sqlite.fixture(None).await;
        let (s, faults) = fx.faulty_thread("worker-a").await;
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
        assert_eq!(s.lane(LANE).tasks(None, 10).await.unwrap().len(), 1, "no second task");
        fx.cleanup(&[&s]).await;
    }
}

/// Decision 39: an acknowledgment lost with the database unreachable through the whole wait
/// answers the original task still `working`, never a failure; the resolver later finds the
/// commit and the task reads completed.
#[tokio::test]
async fn an_unresolved_commit_answers_working_and_resolves_to_the_original_run() {
    let fx = Backend::Sqlite.fixture(None).await;
    let (s, faults) = fx.faulty_thread("worker-a").await;
    faults.fail_next_commit(lk_agents_storage::testing::CommitFault::LoseAckAfter);
    faults.unreachable_after_commit_for(Duration::from_secs(12));
    let host = Fake::new(Behavior::Echo);
    let r = turn(&s, host, Some(Policy::Queue))
        .run(a2a("hello", "m1", None, None), Some("AST_1".into()))
        .await
        .unwrap();
    assert_eq!((r.task.task_id.as_str(), r.task.status), ("AST_1", TaskState::Working));

    let reader = fx.thread("reader").await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let row = loop {
        let row = reader.lane(LANE).task("AST_1").await.unwrap().unwrap();
        if row.status != TaskState::Working || tokio::time::Instant::now() > deadline {
            break row;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    assert_eq!(row.status, TaskState::Completed, "the original run's commit");
    assert_eq!(history_texts(&reader).await, vec!["hello", "reply to: hello"]);
    assert_eq!(reader.lane(LANE).tasks(None, 10).await.unwrap().len(), 1, "no second task");
    fx.cleanup(&[&s, &reader]).await;
}

/// I-8: the message's `submitted` row is durable before the debounce, not after it.
#[tokio::test]
async fn the_submitted_row_is_durable_before_the_debounce() {
    let fx = Backend::Sqlite.fixture(None).await;
    let s = fx.thread("worker-a").await;
    let opts = TurnOptions {
        policy: PolicySpec::All(Policy::Queue),
        debounce: DebounceSpec::All(Duration::from_secs(1)),
        ..TurnOptions::default()
    };
    let turn = Turn::new(s.clone(), LANE, Fake::new(Behavior::Echo), clock(), opts);
    let running = tokio::spawn(async move {
        turn.run(a2a("hello", "m1", Some("sms"), None), Some("AST_1".into())).await
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let row = s.lane(LANE).task("AST_1").await.unwrap().expect("durable before the debounce");
    assert_eq!(row.status, TaskState::Submitted);
    let done = running.await.unwrap().unwrap();
    assert_eq!(done.task.status, TaskState::Completed);
    fx.cleanup(&[&s]).await;
}
