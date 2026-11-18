use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use bytes::Bytes;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::{BoxFuture, Clock, Ts};
use lk_agents_core::request::AttemptFailure;
use lk_agents_runtime::net::http::exchange::ResponsePolicy;
use lk_agents_runtime::net::http::port::{HttpExchangePort, HttpResponse};
use lk_agents_runtime::net::http::{HeaderMap, Method, Reporter, Sent, StatusCode};
use lk_agents_runtime::net::transport::HttpRequest;
use parking_lot::Mutex;
use serde_json::{Value, json};

use crate::record::RemoteStep;

/// Where the peer reports what it observed: the voice registry, or a text story's own log.
pub type Observer = Arc<dyn Fn(RemoteStep) + Send + Sync>;

// A scripted A2A peer at the HTTP socket boundary: the real client, exchange and delegation
// executors run over it. Its work is committed when a message arrives and becomes visible on
// the clock (`GetTask` reads what is due), so a lost acknowledgement or a dropped stream leaves
// the business effect applied, as a real peer would.

/// The peer's endpoint, also its remote id: the canonical URL a `RemoteAgent` targets.
pub const ENDPOINT: &str = "https://desk.test";

/// What the peer does with each message, and how its transport misbehaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerScript {
    /// How long each event of a work phase takes to become visible.
    pub work_ms: u64,
    /// Progress updates before each question or the result.
    pub progress: u8,
    /// Questions a task asks, one per work phase, before its result.
    pub asks: u8,
    pub end: PeerEnd,
    /// The first answer to a question is applied, then its acknowledgement is lost.
    pub lost_ack: bool,
    /// The first response stream drops after this many events, its work still committed.
    pub drop_after: Option<u8>,
    pub cancel: CancelReply,
}

impl Default for PeerScript {
    fn default() -> Self {
        Self {
            work_ms: 200,
            progress: 1,
            asks: 1,
            end: PeerEnd::Completed,
            lost_ack: false,
            drop_after: None,
            cancel: CancelReply::Confirmed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerEnd {
    Completed,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelReply {
    Confirmed,
    /// `TaskNotCancelable`.
    Refused,
    /// Applied, then its reply lost.
    Lost,
}

/// The question a task asks the `k`-th time (from 1): unique in the run, so a presentation names it.
pub fn question(task: &str, k: u8) -> String {
    format!("For {task}, which day suits you (question {k})?")
}

/// What a completed task returns: unique in the run, so a delivery names it.
pub fn result(task: &str) -> String {
    format!("{task} is booked")
}

#[derive(Clone)]
enum Event {
    Status { state: &'static str, message: Option<Value> },
    Artifact(Value),
}

struct Task {
    context: String,
    asked: u8,
    /// The question message the task waits on.
    waiting: Option<String>,
    /// Every event, with when it becomes visible.
    log: Vec<(Ts, Event)>,
    history: Vec<Value>,
}

impl Task {
    fn visible(&self, now: Ts) -> impl Iterator<Item = &Event> {
        self.log.iter().filter(move |(due, _)| *due <= now).map(|(_, event)| event)
    }

    fn snapshot(&self, id: &str, now: Ts) -> Value {
        let mut status = json!({"state": "TASK_STATE_SUBMITTED"});
        let mut artifacts = Vec::new();
        let mut history = self.history.clone();
        for event in self.visible(now) {
            match event {
                Event::Status { state, message } => {
                    status = json!({"state": state});
                    if let Some(message) = message {
                        status["message"] = message.clone();
                        history.push(message.clone());
                    }
                }
                Event::Artifact(artifact) => artifacts.push(artifact.clone()),
            }
        }
        json!({"id": id, "contextId": self.context, "status": status, "history": history,
            "artifacts": artifacts})
    }

    fn stream_event(&self, id: &str, event: &Event) -> Value {
        match event {
            Event::Status { state, message } => {
                let mut status = json!({"state": state});
                if let Some(message) = message {
                    status["message"] = message.clone();
                }
                json!({"statusUpdate": {"taskId": id, "contextId": self.context, "status": status}})
            }
            Event::Artifact(artifact) => json!({"artifactUpdate": {"taskId": id,
                "contextId": self.context, "artifact": artifact, "lastChunk": true}}),
        }
    }
}

#[derive(Default)]
struct State {
    tasks: BTreeMap<String, Task>,
    /// One answer's acknowledgement is lost; one stream drops.
    ack_lost: bool,
    dropped: bool,
}

pub struct RemotePeer {
    script: PeerScript,
    observe: Observer,
    clock: Arc<dyn Clock>,
    state: Mutex<State>,
    streams: AtomicU32,
}

impl RemotePeer {
    pub fn new(script: PeerScript, observe: Observer, clock: Arc<dyn Clock>) -> Arc<Self> {
        Arc::new(Self {
            script,
            observe,
            clock,
            state: Mutex::default(),
            streams: AtomicU32::new(0),
        })
    }

    /// Commit a message's work phase: its events from now, each `work_ms` after the last.
    fn work(&self, state: &mut State, task_id: &str, now: Ts) -> Vec<(Ts, Event)> {
        let script = self.script;
        let task = state.tasks.get_mut(task_id).expect("work on a known task");
        let mut at = now;
        let mut next = || {
            at = at.saturating_add(Duration::from_millis(script.work_ms));
            at
        };
        let mut events = vec![(now, Event::Status { state: "TASK_STATE_WORKING", message: None })];
        for step in 1..=script.progress {
            let text = format!("Working on {task_id}, step {step}.");
            let message = agent_message(&format!("{task_id}-p{}-{step}", task.asked), &text);
            events.push((next(), Event::Status { state: "TASK_STATE_WORKING", message: Some(message) }));
        }
        if task.asked < script.asks {
            task.asked += 1;
            let id = format!("{task_id}-q{}", task.asked);
            let text = question(task_id, task.asked);
            task.waiting = Some(id.clone());
            (self.observe)(RemoteStep::Asked {
                task: task_id.into(),
                question: id.clone(),
                text: text.clone(),
            });
            let message = agent_message(&id, &text);
            events.push((
                next(),
                Event::Status { state: "TASK_STATE_INPUT_REQUIRED", message: Some(message) },
            ));
        } else {
            match script.end {
                PeerEnd::Completed => {
                    let text = result(task_id);
                    (self.observe)(RemoteStep::Finished { task: task_id.into(), result: text.clone() });
                    let at = next();
                    events.push((
                        at,
                        Event::Artifact(json!({"artifactId": format!("{task_id}-result"),
                            "parts": [{"text": text}]})),
                    ));
                    events.push((at, Event::Status { state: "TASK_STATE_COMPLETED", message: None }));
                }
                PeerEnd::Failed => {
                    let message = agent_message(&format!("{task_id}-failed"), "The desk is closed.");
                    events.push((
                        next(),
                        Event::Status { state: "TASK_STATE_FAILED", message: Some(message) },
                    ));
                }
            }
        }
        task.log.extend(events.iter().cloned());
        events
    }

    fn send_message(&self, rpc_id: &Value, params: Value, streaming: bool) -> Reply {
        let send: a2a::SendMessageRequest = match serde_json::from_value(params) {
            Ok(send) => send,
            Err(error) => return Reply::Error(rpc_error(rpc_id, -32602, &error.to_string())),
        };
        let message = send.message;
        let now = self.clock.now();
        let mut state = self.state.lock();
        let message_id = message.message_id.clone();
        let wire = serde_json::to_value(&message).unwrap_or(Value::Null);
        let text = message
            .parts
            .iter()
            .filter_map(|part| match &part.content {
                a2a::PartContent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        let (task_id, answering) = match message.task_id.clone() {
            None => {
                let n = state.tasks.len() + 1;
                let id = format!("task-{n}");
                let context = message.context_id.clone().unwrap_or_else(|| format!("ctx-{n}"));
                state.tasks.insert(
                    id.clone(),
                    Task { context, asked: 0, waiting: None, log: Vec::new(), history: vec![] },
                );
                (id, None)
            }
            Some(id) => {
                let Some(task) = state.tasks.get_mut(&id) else {
                    return Reply::Error(rpc_error(rpc_id, a2a::error_code::TASK_NOT_FOUND, "no task"));
                };
                let visible_waiting = task.visible(now).last().is_some_and(|event| {
                    matches!(event, Event::Status { state: "TASK_STATE_INPUT_REQUIRED", .. })
                });
                (id, visible_waiting.then(|| task.waiting.take()).flatten())
            }
        };
        let replying = answering.is_some();
        (self.observe)(RemoteStep::Received {
            task: task_id.clone(),
            message: message_id.clone(),
            answering,
            text,
        });
        state.tasks.get_mut(&task_id).expect("task exists").history.push(wire);
        let events = self.work(&mut state, &task_id, now);
        if replying && self.script.lost_ack && !state.ack_lost {
            state.ack_lost = true;
            (self.observe)(RemoteStep::AckLost { task: task_id, message: message_id });
            return Reply::Lost;
        }
        let task = &state.tasks[&task_id];
        let first = task.snapshot(&task_id, now);
        if !streaming {
            return Reply::Json(json!({"jsonrpc": "2.0", "id": rpc_id, "result": {"task": first}}));
        }
        let mut frames = vec![(now, json!({"task": first}))];
        frames.extend(events.iter().skip(1).map(|(due, e)| (*due, task.stream_event(&task_id, e))));
        let drop_after = self.script.drop_after.filter(|_| !state.dropped).map(usize::from);
        if drop_after.is_some_and(|n| n < frames.len()) {
            state.dropped = true;
        }
        Reply::Stream { rpc_id: rpc_id.clone(), frames, drop_after }
    }

    fn subscribe(&self, rpc_id: &Value, params: &Value) -> Reply {
        let now = self.clock.now();
        let state = self.state.lock();
        let id = params["id"].as_str().unwrap_or_default();
        let Some(task) = state.tasks.get(id) else {
            return Reply::Error(rpc_error(rpc_id, a2a::error_code::TASK_NOT_FOUND, "no task"));
        };
        let mut frames = vec![(now, json!({"task": task.snapshot(id, now)}))];
        frames.extend(
            task.log
                .iter()
                .filter(|(due, _)| *due > now)
                .map(|(due, event)| (*due, task.stream_event(id, event))),
        );
        Reply::Stream { rpc_id: rpc_id.clone(), frames, drop_after: None }
    }

    fn cancel(&self, rpc_id: &Value, params: &Value) -> Reply {
        let now = self.clock.now();
        let mut state = self.state.lock();
        let id = params["id"].as_str().unwrap_or_default().to_owned();
        (self.observe)(RemoteStep::CancelReceived { task: id.clone() });
        let Some(task) = state.tasks.get_mut(&id) else {
            return Reply::Error(rpc_error(rpc_id, a2a::error_code::TASK_NOT_FOUND, "no task"));
        };
        if self.script.cancel == CancelReply::Refused {
            return Reply::Error(rpc_error(
                rpc_id,
                a2a::error_code::TASK_NOT_CANCELABLE,
                "the booking is protected",
            ));
        }
        task.log.retain(|(due, _)| *due <= now);
        task.log.push((now, Event::Status { state: "TASK_STATE_CANCELED", message: None }));
        task.waiting = None;
        if self.script.cancel == CancelReply::Lost {
            return Reply::Lost;
        }
        Reply::Json(json!({"jsonrpc": "2.0", "id": rpc_id, "result": task.snapshot(&id, now)}))
    }

    fn card(&self) -> Value {
        json!({"name": "desk", "description": "Book with the desk", "version": "1",
            "supportedInterfaces": [{"url": format!("{ENDPOINT}/rpc"), "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0"}],
            "capabilities": {"streaming": true}, "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"], "skills": []})
    }

    fn reply(&self, method: &Method, request: &HttpRequest) -> Reply {
        if *method == Method::GET {
            return Reply::Json(self.card());
        }
        let rpc: a2a::JsonRpcRequest = match serde_json::from_slice(&request.body) {
            Ok(rpc) => rpc,
            Err(error) => return Reply::Error(rpc_error(&Value::Null, -32700, &error.to_string())),
        };
        let id = serde_json::to_value(&rpc.id).unwrap_or(Value::Null);
        let params = rpc.params.unwrap_or(Value::Null);
        match rpc.method.as_str() {
            a2a::methods::SEND_MESSAGE => self.send_message(&id, params, false),
            a2a::methods::SEND_STREAMING_MESSAGE => self.send_message(&id, params, true),
            a2a::methods::SUBSCRIBE_TO_TASK => self.subscribe(&id, &params),
            a2a::methods::CANCEL_TASK => self.cancel(&id, &params),
            a2a::methods::GET_TASK => {
                let now = self.clock.now();
                let state = self.state.lock();
                let task_id = params["id"].as_str().unwrap_or_default();
                match state.tasks.get(task_id) {
                    Some(task) => Reply::Json(
                        json!({"jsonrpc": "2.0", "id": id, "result": task.snapshot(task_id, now)}),
                    ),
                    None => Reply::Error(rpc_error(&id, a2a::error_code::TASK_NOT_FOUND, "no task")),
                }
            }
            other => Reply::Error(rpc_error(&id, -32601, &format!("no method {other}"))),
        }
    }
}

enum Reply {
    Json(Value),
    Error(Value),
    /// The work was applied; the connection died before any acknowledgement.
    Lost,
    /// Server-sent events, each written when due; `drop_after` cuts the connection after that many.
    Stream { rpc_id: Value, frames: Vec<(Ts, Value)>, drop_after: Option<usize> },
}

fn agent_message(id: &str, text: &str) -> Value {
    json!({"messageId": id, "role": "ROLE_AGENT", "parts": [{"text": text}]})
}

fn rpc_error(id: &Value, code: i32, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Closes its stream's record when the client drops the body, however the read ended.
struct StreamGuard {
    observe: Observer,
    stream: u32,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        (self.observe)(RemoteStep::StreamClosed { stream: self.stream });
    }
}

impl HttpExchangePort for RemotePeer {
    fn send<'a>(
        &'a self,
        method: Method,
        request: HttpRequest,
        _policy: ResponsePolicy,
        reporter: Reporter<(u32, Sent)>,
        attempt: u32,
    ) -> BoxFuture<'a, Result<HttpResponse, AttemptFailure>> {
        Box::pin(async move {
            reporter.report((attempt, Sent::Connected)).await;
            reporter.report((attempt, Sent::RequestWritten)).await;
            let reply = self.reply(&method, &request);
            let mut headers = HeaderMap::new();
            let json_body = |value: Value| -> lk_agents_runtime::net::http::port::ResponseBody {
                let bytes = Bytes::from(serde_json::to_vec(&value).unwrap_or_default());
                Box::pin(futures_util::stream::iter([Ok(bytes)]))
            };
            let body = match reply {
                Reply::Lost => {
                    return Err(AttemptFailure::Connection(
                        "the peer applied the message, then the connection died".into(),
                    ));
                }
                Reply::Json(value) | Reply::Error(value) => {
                    headers.insert("content-type", "application/json".parse().expect("header"));
                    json_body(value)
                }
                Reply::Stream { rpc_id, frames, drop_after } => {
                    headers.insert("content-type", "text/event-stream".parse().expect("header"));
                    let stream = self.streams.fetch_add(1, Ordering::SeqCst);
                    (self.observe)(RemoteStep::StreamOpened { stream });
                    let guard = StreamGuard { observe: self.observe.clone(), stream };
                    let clock = self.clock.clone();
                    let state = (frames.into_iter().enumerate(), Some(guard));
                    Box::pin(futures_util::stream::unfold(state, move |(mut frames, guard)| {
                        let clock = clock.clone();
                        let rpc_id = rpc_id.clone();
                        async move {
                            let guard = guard?;
                            let Some((index, (due, frame))) = frames.next() else {
                                drop(guard);
                                return None;
                            };
                            if drop_after.is_some_and(|n| index >= n) {
                                let error = AttemptFailure::Connection("the stream dropped".into());
                                return Some((Err(error), (frames, None)));
                            }
                            clock.wait_until(due).await;
                            let rpc = json!({"jsonrpc": "2.0", "id": rpc_id, "result": frame});
                            let line = format!("data: {rpc}\n\n");
                            Some((Ok(Bytes::from(line)), (frames, Some(guard))))
                        }
                    }))
                }
            };
            Ok(HttpResponse { status: StatusCode::OK, headers, body })
        })
    }
}
