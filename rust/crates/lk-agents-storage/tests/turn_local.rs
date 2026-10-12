//! The turn state machine (`turn` module) against a local agent-db, the same way `run_local.rs`
//! drives the run: `LK_AGENTDB_TEST_URL` (default `ws://localhost:7781/v1/db`) and
//! `LK_AGENTDB_TEST_MGMT` (default `http://localhost:7783`, the `AgentDB` gRPC service); skipped when nothing listens there or
//! when what listens cannot create a database. The host is a scripted fake: a handler that returns
//! a text, raises, asks input-required with a pending call, or sleeps until told, and records what
//! it was handed. The cases are those of `tests/test_storage.py`, with its assertions.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_storage::run::{INTERRUPTED_KEY, INTERRUPTION_REASON_KEY, MERGED_INTO_KEY, Thread};
use lk_agents_storage::schema::TaskData;
use lk_agents_storage::statements::{ReceiveRequest, SubmittedTask};
use lk_agents_storage::turn::{
    DebounceSpec, HandlerOutput, HostError, PendingCall, Policy, PolicySpec, Turn, TurnContext,
    TurnHost, TurnOptions, TurnResult,
};
use lk_agents_storage::{Storage, StorageError, StorageOptions, TokenMinter};
use serde_json::{Value, json};
use tokio::sync::watch;

const LANE: &str = "front-desk";

/// The dev pair, or `LK_AGENTDB_TEST_SECRET` when the local agent-db runs on another key.
fn minter() -> TokenMinter {
    let secret = std::env::var("LK_AGENTDB_TEST_SECRET").unwrap_or_else(|_| "secret".into());
    TokenMinter::new("devkey", &secret).unwrap()
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

async fn reachable(url: &str) -> bool {
    let u = url::Url::parse(url).expect("a url");
    let addr = format!(
        "{}:{}",
        u.host_str().unwrap_or("localhost"),
        u.port_or_known_default().unwrap_or(80)
    );
    tokio::time::timeout(Duration::from_millis(500), tokio::net::TcpStream::connect(addr))
        .await
        .is_ok_and(|r| r.is_ok())
}

/// The storage and a fresh database, or `None` when there is no agent-db to test against.
async fn fresh_database_with(lease: Option<(Duration, Duration)>) -> Option<(Storage, String)> {
    let url = env_or("LK_AGENTDB_TEST_URL", "ws://localhost:7781/v1/db");
    let mgmt = env_or("LK_AGENTDB_TEST_MGMT", "http://localhost:7783");
    if !reachable(&url).await {
        eprintln!("skipped: no agent-db at {url}");
        return None;
    }
    let mut opts = StorageOptions::new(&url);
    opts.management_url = Some(mgmt.clone());
    if let Some((ttl, interval)) = lease {
        opts.lease_ttl = ttl;
        opts.renew_interval = interval;
    }
    let storage = Storage::new(opts, minter()).unwrap();
    let database_id = match storage.admin().unwrap().create_database(3600).await {
        Ok(c) => c.database_id,
        Err(e) => {
            eprintln!("skipped: agent-db at {mgmt} cannot create a database: {e}");
            return None;
        }
    };
    // Two workers dial the same fresh database at once here; the schema is applied by the first
    // acquire, so one bootstrap handle writes it before the workers open (as `run_local.rs` dials
    // its second handle after the first run).
    let boot = storage.open(&database_id).await.unwrap();
    boot.ensure_schema().await.unwrap();
    boot.close();
    Some((storage, database_id))
}

async fn fresh_database() -> Option<(Storage, String)> {
    fresh_database_with(None).await
}

/// One handle over the database, as one worker would hold it.
async fn thread(storage: &Storage, database_id: &str, holder: &str) -> Thread {
    Thread::new(storage.open(database_id).await.unwrap(), holder)
}

async fn cleanup(storage: &Storage, database_id: &str, sessions: &[&Thread]) {
    for s in sessions {
        s.close();
    }
    storage.admin().unwrap().delete_database(database_id).await.unwrap();
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// `a2a(text, message_id, channel=, task_id=)` of the Python tests.
fn a2a(text: &str, message_id: &str, channel: Option<&str>, task_id: Option<&str>) -> Value {
    let mut msg = json!({
        "messageId": message_id,
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
    });
    if let Some(c) = channel {
        msg["metadata"] = json!({"livekit/channel": c});
    }
    if let Some(t) = task_id {
        msg["taskId"] = json!(t);
    }
    msg
}

fn text_of(item: &Value) -> String {
    item["content"]
        .as_array()
        .map(|c| c.iter().filter_map(Value::as_str).collect::<Vec<_>>().join("\n"))
        .unwrap_or_default()
}

fn user_item(text: &str, created_at: f64) -> Value {
    json!({"id": lk_agents_storage::run::shortuuid("item_"), "type": "message", "role": "user",
        "content": [text], "interrupted": false, "extra": {}, "created_at": created_at})
}

fn assistant_item(text: &str, created_at: f64) -> Value {
    json!({"id": lk_agents_storage::run::shortuuid("item_"), "type": "message", "role": "assistant",
        "content": [text], "interrupted": false, "extra": {}, "created_at": created_at})
}

fn call_item(name: &str, call_id: &str, created_at: f64) -> Value {
    json!({"id": lk_agents_storage::run::shortuuid("item_"), "type": "function_call",
        "call_id": call_id, "arguments": "{}", "name": name, "created_at": created_at, "extra": {}})
}

fn output_item(name: &str, call_id: &str, output: &str, created_at: f64) -> Value {
    json!({"id": lk_agents_storage::run::shortuuid("item_"), "type": "function_call_output",
        "name": name, "call_id": call_id, "output": output, "is_error": false,
        "created_at": created_at, "reply_required": true})
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

#[async_trait::async_trait]
impl TurnHost for Fake {
    async fn handle(&self, ctx: TurnContext) -> Result<HandlerOutput, HostError> {
        let message_text = ctx.batch.last().map(|m| m.text()).unwrap_or_default();
        self.calls.lock().unwrap().push(Call {
            batch: ctx.batch.iter().map(|m| m.text()).collect(),
            history: ctx.history.iter().map(|i| text_of(&i.payload)).collect(),
            message: ctx.message["content"]
                .as_array()
                .map(|c| c.iter().filter_map(Value::as_str).map(str::to_owned).collect())
                .unwrap_or_default(),
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
            Behavior::Unavailable => {
                Err(HostError::Storage(StorageError::Unavailable("the model is down".into())))
            }
            Behavior::Slow { entered, gate, output } => {
                entered.set();
                self.on_update(&ctx.task_id, "working on it").await;
                gate.wait().await;
                Ok(text(output))
            }
            Behavior::Opener { gate, output } => {
                gate.set();
                Ok(text(output))
            }
            Behavior::Tool { entered, gate } => {
                let t = now_secs();
                let mut items =
                    vec![user_item(&message_text, t), call_item("lookup", "c1", t + 1e-6)];
                entered.set();
                gate.wait().await;
                items.push(output_item("lookup", "c1", "found", t + 2e-6));
                if ctx.boundary.check().await? {
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
                let t = now_secs();
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

    async fn on_update(&self, task_id: &str, text: &str) {
        self.updates.lock().unwrap().push((task_id.to_owned(), text.to_owned()));
    }

    async fn on_token(&self, _task_id: &str, _delta: &str) {}
}

fn options(policy: Option<Policy>) -> TurnOptions {
    TurnOptions {
        policy: policy.map_or(PolicySpec::Default, PolicySpec::All),
        debounce: DebounceSpec::All(Duration::ZERO),
        ..TurnOptions::default()
    }
}

fn turn(s: &Thread, host: Arc<Fake>, policy: Option<Policy>) -> Turn {
    Turn::new(s.clone(), LANE, host, options(policy))
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
    assert!(id.starts_with(prefix) && id.len() == prefix.len() + 12, "{id}");
    v[key] = Value::String(format!("{prefix}X"));
}

// -- the thin slice --

#[tokio::test]
async fn two_workers_share_one_session() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let a = thread(&storage, &database_id, "worker-a").await;
    let opts = TurnOptions {
        policy: PolicySpec::All(Policy::Queue),
        state_default: Some(json!({"verified": 0})),
        ..TurnOptions::default()
    };
    let host = Fake::new(Behavior::Counter);
    let first = Turn::new(a.clone(), LANE, host.clone(), opts.clone())
        .run(a2a("hello", "m1", None, None), None)
        .await
        .unwrap();
    let b = thread(&storage, &database_id, "worker-b").await;
    let second = Turn::new(b.clone(), LANE, host.clone(), opts)
        .run(a2a("again", "m2", None, None), None)
        .await
        .unwrap();

    assert_eq!(first.output, Some(json!("seen 1")));
    assert_eq!(second.output, Some(json!("seen 2")));
    let seen: Vec<Vec<String>> = host.calls().into_iter().map(|c| c.history).collect();
    assert_eq!(seen, vec![vec![], vec!["hello".to_owned(), "seen 1".to_owned()]]);
    assert_eq!(first.task.status, "completed");
    assert!(first.task.data.artifacts.as_ref().is_some_and(|a| !a.is_empty()));
    assert_eq!(history_texts(&b).await, vec!["hello", "seen 1", "again", "seen 2"]);
    let runs = b.lane(LANE).runs(10).await.unwrap();
    assert_eq!(
        runs.iter().map(|r| r.outcome.as_deref()).collect::<Vec<_>>(),
        vec![Some("completed"), Some("completed")]
    );
    assert_eq!(runs[0].task_id.as_deref(), Some(second.task.id.as_str()));
    assert_eq!(b.lane(LANE).info().await.unwrap().unwrap().version, 2);
    // The state was written at the first commit and loaded by the second worker.
    let run = b
        .lane(LANE)
        .open(lk_agents_storage::run::RunOptions::new("peek", "AST_peek"))
        .await
        .unwrap();
    assert_eq!(run.state, Some(json!({"verified": 2})));
    drop(run);
    cleanup(&storage, &database_id, &[&a, &b]).await;
}

#[tokio::test]
async fn a_run_that_raises_writes_run_ended_and_nothing_else() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s = thread(&storage, &database_id, "worker-a").await;
    // The host met a storage error: the run aborts, nothing but `run_ended` is written, and the
    // early row stays `working`, the orphan the next holder fails worker-lost.
    let err = run_turn(
        &s,
        a2a("lost", "m1", None, None),
        Some(Policy::Queue),
        Fake::new(Behavior::Unavailable),
    )
    .await
    .expect_err("the host's storage error propagates");
    assert!(matches!(err, StorageError::Unavailable(_)), "{err}");
    let lane = s.lane(LANE);
    assert!(lane.history().await.unwrap().is_empty());
    let events = s.events(0, 100).await.unwrap();
    assert_eq!(
        events.iter().map(|e| e.r#type.as_str()).collect::<Vec<_>>(),
        vec!["run_started", "run_ended"]
    );
    assert_eq!(
        events[1].payload,
        json!({"outcome": "failed", "reason": "StorageUnavailable: the model is down"})
    );
    let rows = lane.tasks(None, 10).await.unwrap();
    assert_eq!((rows.len(), rows[0].status.as_str()), (1, "working"));
    let info = lane.info().await.unwrap().unwrap();
    assert_eq!((info.version, info.lease_until), (0, 0), "released, not bumped");

    // The handler raised: the task fails with the error as its last message, and the turn
    // commits, the orphan above failed worker-lost first, not run again.
    let host = Fake::new(Behavior::Raise);
    let r = run_turn(&s, a2a("boom", "m2", None, None), Some(Policy::Queue), host.clone())
        .await
        .unwrap();
    assert_eq!(r.task.status, "failed");
    let mut sm = r.task.status_message.clone().unwrap();
    placeholder_id(&mut sm, "messageId", "msg_");
    assert_eq!(
        sm.to_string(),
        format!(
            r#"{{"messageId":"msg_X","role":"agent","parts":[{{"kind":"text","text":"RuntimeError: boom"}}],"taskId":"{}"}}"#,
            r.task.id
        )
    );
    assert_eq!(r.output, None);
    assert_eq!(host.batches(), vec![vec!["boom".to_owned()]], "the orphan was not run again");
    let orphan = lane.task(&rows[0].id).await.unwrap().unwrap();
    assert_eq!(orphan.status, "failed");
    assert_eq!(orphan.status_message.unwrap()["metadata"]["livekit/reason"], json!("worker-lost"));
    let runs = lane.runs(10).await.unwrap();
    assert_eq!(runs[0].reason.as_deref(), Some("RuntimeError: boom"));
    assert_eq!(history_texts(&s).await, vec!["boom"], "the message, no reply, nothing re-run");
    cleanup(&storage, &database_id, &[&s]).await;
}

// -- interrupt --

#[tokio::test]
async fn interrupt_yields_and_the_newer_task_answers_both() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let a = thread(&storage, &database_id, "worker-a").await;
    let b = thread(&storage, &database_id, "worker-b").await;
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
    assert_eq!(r1.task.status, "canceled");
    assert_eq!(r1.task.data.metadata.get(INTERRUPTED_KEY), Some(&json!(true)));
    assert_eq!(r1.task.data.metadata.get(INTERRUPTION_REASON_KEY), Some(&json!("new-message")));
    assert_eq!(
        serde_json::to_string(&r1.task.data.metadata).unwrap(),
        r#"{"livekit/interrupted":true,"livekit/interruptionReason":"new-message"}"#
    );
    assert!(r1.task.data.artifacts.is_none() && r1.output.is_none());
    assert_eq!(r1.task.data.history, vec![a2a("first", "m1", None, None)]);
    assert_eq!(r2.task.status, "completed");
    assert_eq!(r2.output, Some(json!("both answered")));
    // what ran is in the history, with no reply from the first turn
    let items = a.lane(LANE).history().await.unwrap();
    assert_eq!(
        items.iter().map(|i| i.kind()).collect::<Vec<_>>(),
        vec!["message", "function_call", "function_call_output", "message", "message"]
    );
    assert_eq!(text_of(&items[4].payload), "both answered");
    assert_eq!(host_b.calls()[0].history.len(), 3, "B loaded what A committed");
    let runs = a.lane(LANE).runs(10).await.unwrap();
    let mut outcomes: Vec<_> = runs.iter().filter_map(|r| r.outcome.clone()).collect();
    outcomes.sort();
    assert_eq!(outcomes, vec!["canceled", "completed"]);
    cleanup(&storage, &database_id, &[&a, &b]).await;
}

// -- inbox --

#[tokio::test]
async fn the_inbox_batches_a_burst_into_one_reply() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let a = thread(&storage, &database_id, "worker-a").await;
    let b = thread(&storage, &database_id, "worker-b").await;
    let host = Fake::new(Behavior::Text("one reply"));
    let opts = TurnOptions {
        debounce: DebounceSpec::All(Duration::from_millis(300)),
        ..TurnOptions::default()
    };
    let first = {
        let (a, host, opts) = (a.clone(), host.clone(), opts.clone());
        tokio::spawn(async move {
            Turn::new(a, LANE, host, opts).run(a2a("where", "m1", Some("sms"), None), None).await
        })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    let second = {
        let (b, host, opts) = (b.clone(), host.clone(), opts);
        tokio::spawn(async move {
            Turn::new(b, LANE, host, opts).run(a2a("are you", "m2", Some("sms"), None), None).await
        })
    };
    let r1 = first.await.unwrap().unwrap();
    let r2 = second.await.unwrap().unwrap();

    assert_eq!(host.batches(), vec![vec!["where".to_owned(), "are you".to_owned()]]);
    assert_eq!(host.calls()[0].message, vec!["where", "are you"], "ctx.message.content");
    assert_eq!(r1.task.status, "completed");
    assert!(r1.task.data.artifacts.as_ref().is_none_or(|a| a.is_empty()));
    assert_eq!(r1.task.data.metadata.get(MERGED_INTO_KEY), Some(&json!(r2.task.id)));
    assert_eq!(r1.answered_by, r2.task.id);
    assert_eq!(r2.task.status, "completed");
    assert_eq!(r2.output, Some(json!("one reply")));
    let items = a.lane(LANE).history().await.unwrap();
    assert_eq!(
        items
            .iter()
            .filter(|i| i.kind() == "message")
            .map(|i| i.payload["role"].clone())
            .collect::<Vec<_>>(),
        vec![json!("user"), json!("assistant")]
    );
    assert_eq!(items[0].payload["content"], json!(["where", "are you"]));
    assert_eq!(items[0].payload["extra"], json!({}), "the merged ids are not stored");
    let info = a.lane(LANE).info().await.unwrap().unwrap();
    assert_eq!((info.drained_id, info.lease_until), (2, 0));
    cleanup(&storage, &database_id, &[&a, &b]).await;
}

#[tokio::test]
async fn a_message_id_resend_returns_the_existing_task() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s = thread(&storage, &database_id, "worker-a").await;
    let echo = Fake::new(Behavior::Echo);
    let first = run_turn(&s, a2a("hi", "m1", Some("sms"), None), None, echo.clone()).await.unwrap();
    let again = run_turn(&s, a2a("hi", "m1", Some("sms"), None), None, echo.clone()).await.unwrap();
    assert_eq!(again.task.id, first.task.id);
    assert_eq!(again.output, first.output);
    assert_eq!(again.output, Some(json!("reply to: hi")));
    // under interrupt the task row holds the message id
    let first = run_turn(&s, a2a("yo", "m2", None, None), Some(Policy::Interrupt), echo.clone())
        .await
        .unwrap();
    let again = run_turn(&s, a2a("yo", "m2", None, None), Some(Policy::Interrupt), echo.clone())
        .await
        .unwrap();
    assert_eq!(again.task.id, first.task.id);
    assert_eq!(again.run_id, None, "nothing ran");
    assert_eq!(s.lane(LANE).history().await.unwrap().len(), 4);
    assert_eq!(echo.calls().len(), 2);
    cleanup(&storage, &database_id, &[&s]).await;
}

// -- the lease --

#[tokio::test]
async fn a_lapsed_lease_fails_the_orphan_worker_lost() {
    let lease = Some((Duration::from_secs(1), Duration::from_millis(300)));
    let Some((storage, database_id)) = fresh_database_with(lease).await else { return };
    let a = thread(&storage, &database_id, "worker-a").await;
    let db = a.db();
    let message = a2a("first", "m1", Some("sms"), None);
    let receive = ReceiveRequest {
        lane: LANE.into(),
        holder: "worker-a".into(),
        lease_token: "ta".into(),
        task_id: "AST_1".into(),
        message_id: "m1".into(),
        message: message.clone(),
        name: None,
        data: TaskData { history: vec![message.clone()], ..TaskData::default() },
        internal: None,
    };

    // worker A receives, picks the task up, and dies before its commit
    let received = db.receive(&receive).await.unwrap();
    assert!(matches!(
        received,
        lk_agents_storage::statements::Receive::Received(lk_agents_storage::statements::Received {
            acquired: Some(0),
            ..
        })
    ));
    assert_eq!(db.pickup_task(LANE, "AST_1").await.unwrap().as_deref(), Some("submitted"));
    // the lease is one second here; it lapses once unixepoch() moves past lease_until
    tokio::time::sleep(Duration::from_millis(2100)).await;

    // worker B takes the lane for the next message: the orphan fails worker-lost, not run again
    let b = thread(&storage, &database_id, "worker-b").await;
    let host = Fake::new(Behavior::Text("answered"));
    let r2 =
        run_turn(&b, a2a("second", "m2", Some("sms"), None), None, host.clone()).await.unwrap();
    assert_eq!(host.batches(), vec![vec!["second".to_owned()]]);
    assert_eq!(r2.task.status, "completed");

    // the inbox drain picked the row up once more and found it working: a dead holder had it
    let lost = b.lane(LANE).task("AST_1").await.unwrap().unwrap();
    assert_eq!(lost.status, "failed");
    let mut sm = lost.status_message.clone().unwrap();
    placeholder_id(&mut sm, "messageId", "msg_");
    assert_eq!(
        sm.to_string(),
        r#"{"messageId":"msg_X","role":"agent","parts":[{"kind":"text","text":"the task failed: worker-lost"}],"taskId":"AST_1","metadata":{"livekit/reason":"worker-lost"}}"#
    );
    assert_eq!(history_texts(&b).await, vec!["second", "answered"], "the dead turn left nothing");
    cleanup(&storage, &database_id, &[&a, &b]).await;
}

// -- input required --

#[tokio::test]
async fn input_required_pending_call_is_the_next_messages_output() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let s = thread(&storage, &database_id, "worker-a").await;
    let first = run_turn(
        &s,
        a2a("my car is damaged", "m1", None, None),
        Some(Policy::Interrupt),
        Fake::new(Behavior::Ask),
    )
    .await
    .unwrap();
    assert_eq!(first.task.status, "input-required");
    assert_eq!(
        first.task.ask().unwrap().to_string(),
        r#"{"question":"Could you send a photo?","tool":"request_photo","call_id":"p1"}"#
    );
    let sm = first.task.status_message.clone().unwrap();
    assert_eq!(sm["parts"][0]["text"], json!("Could you send a photo?"));
    assert_eq!(first.task.data.history, vec![a2a("my car is damaged", "m1", None, None)]);
    assert_eq!(first.task.message_id.as_deref(), Some("m1"));
    assert_eq!(first.output, None);

    let host = Fake::new(Behavior::Text("thanks"));
    let second =
        run_turn(&s, a2a("here it is", "m2", None, None), Some(Policy::Interrupt), host.clone())
            .await
            .unwrap();
    assert_eq!(second.output, Some(json!("thanks")));
    let items = s.lane(LANE).history().await.unwrap();
    let outputs: Vec<&Value> = items
        .iter()
        .filter(|i| i.kind() == "function_call_output" && i.payload["call_id"] == json!("p1"))
        .map(|i| &i.payload)
        .collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0]["output"], json!("here it is"));
    assert_eq!(outputs[0]["name"], json!("request_photo"));
    assert_eq!(outputs[0]["is_error"], json!(false));
    assert_eq!(
        items.iter().map(|i| i.kind()).collect::<Vec<_>>(),
        vec!["message", "function_call", "function_call_output", "message", "message"]
    );
    assert_eq!(
        host.calls()[0].history.len(),
        3,
        "the handler sees the question, the call and the answer"
    );
    let answered = s.lane(LANE).task(&first.task.id).await.unwrap().unwrap();
    assert_eq!(answered.status, "completed");
    assert_eq!(answered.data.metadata.get(MERGED_INTO_KEY), Some(&json!(second.task.id)));
    assert!(answered.ask().is_none(), "a completed row holds no pending transition");
    cleanup(&storage, &database_id, &[&s]).await;
}

// -- concurrent and reject --

#[tokio::test]
async fn concurrent_runs_append_without_a_lease() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let a = thread(&storage, &database_id, "worker-a").await;
    let b = thread(&storage, &database_id, "worker-b").await;
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
    // the concurrent policy requires a door with no state
    let with_state = Turn::new(
        a.clone(),
        LANE,
        Fake::new(Behavior::Echo),
        TurnOptions {
            policy: PolicySpec::All(Policy::Concurrent),
            state_default: Some(json!({"verified": 0})),
            ..TurnOptions::default()
        },
    );
    let err = with_state.run(a2a("x", "m3", None, None), None).await.expect_err("no state");
    assert!(matches!(err, StorageError::Invalid(_)), "{err}");
    cleanup(&storage, &database_id, &[&a, &b]).await;
}

#[tokio::test]
async fn reject_refuses_while_a_turn_runs() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let a = thread(&storage, &database_id, "worker-a").await;
    let b = thread(&storage, &database_id, "worker-b").await;
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
        StorageError::RunRejected { running_task_id } => {
            assert_eq!(
                running_task_id.as_deref(),
                Some(r1.task.id.as_str()),
                "the error names the running task"
            );
        }
        other => panic!("{other}"),
    }
    assert_eq!(r1.output, Some(json!("slow")));
    assert_eq!(
        slow.updates.lock().unwrap().clone(),
        vec![(r1.task.id.clone(), "working on it".to_owned())],
        "the host's update reached the wire hook"
    );
    cleanup(&storage, &database_id, &[&a, &b]).await;
}

#[tokio::test]
async fn a_submitted_row_exists_while_the_turn_runs() {
    for policy in [Policy::Interrupt, Policy::Queue, Policy::Concurrent, Policy::Reject] {
        let Some((storage, database_id)) = fresh_database().await else { return };
        let a = thread(&storage, &database_id, "worker-a").await;
        let b = thread(&storage, &database_id, "worker-b").await;
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
        let status = if policy == Policy::Concurrent { "submitted" } else { "working" };
        assert_eq!(during.status, status, "{policy}");
        assert_eq!(during.message_id.as_deref(), Some("m1"));
        assert_eq!(during.data.history, vec![a2a("hello", "m1", None, None)]);
        assert_eq!(during.seq, 1);
        assert_eq!((during.name.as_ref(), during.internal.as_ref()), (None, None));
        assert!(during.data.metadata.is_empty());

        gate.set();
        let result = running.await.unwrap().unwrap();
        let after = lane.task("AST_early").await.unwrap().unwrap();
        assert_eq!((after.status.as_str(), after.seq), ("completed", 1), "{policy}");
        assert_eq!(result.task.id, "AST_early");
        assert_eq!(result.output, Some(json!("done")));
        let mut artifacts = after.data.artifacts.clone().unwrap();
        placeholder_id(&mut artifacts[0], "artifactId", "art_");
        assert_eq!(
            Value::Array(artifacts).to_string(),
            r#"[{"artifactId":"art_X","parts":[{"kind":"text","text":"done"}]}]"#
        );
        let mut sm = after.status_message.clone().unwrap();
        placeholder_id(&mut sm, "messageId", "msg_");
        assert_eq!(
            sm.to_string(),
            r#"{"messageId":"msg_X","role":"agent","parts":[{"kind":"text","text":"done"}],"taskId":"AST_early"}"#
        );
        assert_eq!(lane.tasks(None, 50).await.unwrap().len(), 1, "the commit moved the early row");
        // a resend of the message inserts no second row
        let inserted = b
            .db()
            .insert_submitted_task(&SubmittedTask {
                lane: LANE.into(),
                task_id: "AST_other".into(),
                message_id: Some("m1".into()),
                name: None,
                data: TaskData::default(),
                internal: None,
            })
            .await
            .unwrap();
        assert!(!inserted);
        cleanup(&storage, &database_id, &[&a, &b]).await;
    }
}

#[tokio::test]
async fn a_lease_timeout_fails_the_early_row() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let a = thread(&storage, &database_id, "worker-a").await;
    let b = thread(&storage, &database_id, "worker-b").await;
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
        TurnOptions {
            policy: PolicySpec::All(Policy::Queue),
            wait_timeout: Some(Duration::from_millis(50)),
            ..TurnOptions::default()
        },
    );
    let second = short.run(a2a("two", "m2", None, None), Some("AST_2".into())).await.unwrap();
    assert_eq!(second.task.status, "failed");
    assert_eq!(
        second.task.status_message.as_ref().unwrap()["metadata"]["livekit/reason"],
        json!("lease-timeout")
    );
    assert_eq!(second.run_id, None);
    let row =
        b.lane(LANE).task("AST_2").await.unwrap().expect("persisted, not the unwritten stand-in");
    assert_eq!((row.status.as_str(), row.seq), ("failed", 2));
    assert_eq!(row.data.history, vec![a2a("two", "m2", None, None)]);
    gate.set();
    first.await.unwrap().unwrap();
    let mut statuses: Vec<String> =
        b.lane(LANE).tasks(None, 50).await.unwrap().into_iter().map(|t| t.status).collect();
    statuses.sort();
    assert_eq!(statuses, vec!["completed", "failed"]);
    cleanup(&storage, &database_id, &[&a, &b]).await;
}

#[tokio::test]
async fn a_rejected_message_leaves_no_submitted_row() {
    let Some((storage, database_id)) = fresh_database().await else { return };
    let a = thread(&storage, &database_id, "worker-a").await;
    let b = thread(&storage, &database_id, "worker-b").await;
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
    assert!(matches!(err, StorageError::RunRejected { .. }), "{err}");
    assert!(b.lane(LANE).task("AST_2").await.unwrap().is_none());
    gate.set();
    first.await.unwrap().unwrap();
    let statuses: Vec<String> =
        b.lane(LANE).tasks(None, 50).await.unwrap().into_iter().map(|t| t.status).collect();
    assert_eq!(statuses, vec!["completed"]);
    cleanup(&storage, &database_id, &[&a, &b]).await;
}
