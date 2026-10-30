//! The fake handler: an application's `a2a_session` handler at the turn's host port
//! (`TurnHost`). It works a while before each step, reaches a few checkpoints, yields once told
//! to, and ends with a text answer, a conversation tool's question (its call and placeholder
//! output in the new items, as an agent run records them), a transport `Rejected` or `Working`
//! transition, or a raise; while faults are on it may lose the lane, be cancelled by its host,
//! lose the database, or have its `save_context` refused: a stale save (`VersionConflict`), a
//! field that is not JSON (`NotSerializable`) or a state too large (`TooLarge`), raised as the
//! storage errors they are. Its output carries a marker
//! item naming the invocation, so the history shows which invocations' commits applied.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use lk_agents_abi::a2a::turn::{TurnContext, TurnEvent, TurnHost, Verdict};
use lk_agents_abi::errors::{HostError, ProviderError};
use lk_agents_abi::livekit_protocol::agent::chat_context::{ChatItem, chat_item};
use lk_agents_abi::livekit_protocol::agent::{ChatRole, FunctionCall, FunctionCallOutput};
use lk_agents_abi::storage::TaskState;
use lk_agents_abi::storage::thread::timestamp;
use lk_agents_abi::{BoxFuture, Closed, StreamSink, StreamSource, Ts};
use lk_agents_core::a2a::{HandlerOutput, PendingCall, chat_message};

use super::Faults;
use super::database::World;

/// The prefix of a marker item's id; the invocation's number follows.
pub const MARKER: &str = "exec_";

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ending {
    Text,
    Ask,
    Raise,
    /// A transport transition: `Rejected` or `Working`.
    Status(TaskState),
}

/// The handlers of every request: which run now, and what each invocation handled.
pub struct Handlers {
    world: Arc<World>,
    faults: Faults,
    /// Requests whose handler is running: a worker that dies dies mid-turn.
    pub open: Mutex<BTreeSet<usize>>,
    /// By invocation: the `messageId`s of the batch it handled.
    pub invocations: Mutex<Vec<Vec<String>>>,
    /// The invocations that answered `Working`, and the task each answered.
    pub working: Mutex<BTreeMap<usize, String>>,
}

impl Handlers {
    pub fn new(world: Arc<World>, faults: Faults) -> Arc<Self> {
        Arc::new(Handlers {
            world,
            faults,
            open: Mutex::default(),
            invocations: Mutex::default(),
            working: Mutex::default(),
        })
    }
}

/// One request's host.
pub struct Host {
    pub request: usize,
    pub handlers: Arc<Handlers>,
}

impl TurnHost for Host {
    fn handle<'a>(
        &'a self,
        ctx: &'a TurnContext,
    ) -> BoxFuture<
        'a,
        Result<(Box<dyn StreamSink<Verdict>>, Box<dyn StreamSource<TurnEvent>>), ProviderError>,
    > {
        Box::pin(async move {
            let handlers = &self.handlers;
            let batch: Vec<String> =
                ctx.batch.iter().filter_map(|m| m.message_id.clone()).collect();
            let invocation = {
                let mut invocations = handlers.invocations.lock().unwrap();
                invocations.push(batch);
                invocations.len() - 1
            };
            handlers.open.lock().unwrap().insert(self.request);
            let verdict = Arc::new(Mutex::new(None));
            let world = &handlers.world;
            let ending = match world.draw(|r| r.u8(0..20)) {
                0..=10 => Ending::Text,
                11..=16 => Ending::Ask,
                17 => Ending::Status(TaskState::Rejected),
                18 => Ending::Status(TaskState::Working),
                _ => Ending::Raise,
            };
            let handler = Handler {
                handlers: handlers.clone(),
                request: self.request,
                invocation,
                task_id: ctx.task_id.clone(),
                message: serde_json::Value::Object(ctx.message.clone()),
                checkpoints: world.draw(|r| r.u8(0..4)),
                ending,
                verdict: verdict.clone(),
                yielded: false,
                done: false,
            };
            Ok((
                Box::new(Verdicts(verdict)) as Box<dyn StreamSink<Verdict>>,
                Box::new(handler) as Box<dyn StreamSource<TurnEvent>>,
            ))
        })
    }
}

/// The turn's answers to the handler's checkpoints.
struct Verdicts(Arc<Mutex<Option<Verdict>>>);

impl StreamSink<Verdict> for Verdicts {
    fn push(&self, verdict: Verdict) -> Result<(), ProviderError> {
        *self.0.lock().unwrap() = Some(verdict);
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {}
}

/// One handler invocation.
struct Handler {
    handlers: Arc<Handlers>,
    request: usize,
    invocation: usize,
    task_id: String,
    /// The batch as one user chat item.
    message: serde_json::Value,
    checkpoints: u8,
    ending: Ending,
    verdict: Arc<Mutex<Option<Verdict>>>,
    yielded: bool,
    done: bool,
}

/// An exception the host's handler raised, as the host stream reports it.
fn raised(kind: &str, message: &str) -> ProviderError {
    let error = HostError { kind: kind.into(), message: message.into(), host_ref: None };
    ProviderError::Host { error, api_kind: None, retryable: false, status_code: None }
}

impl StreamSource<TurnEvent> for Handler {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<TurnEvent>, ProviderError>> {
        Box::pin(async move {
            let world = self.handlers.world.clone();
            tokio::time::sleep(world.ms(0..300)).await;
            let f = self.handlers.faults;
            if world.inject(f.lease_lost, "LeaseLost") {
                return Err(raised("LeaseLost", "the handler lost the lane"));
            }
            if world.inject(f.cancelled, "Cancelled") {
                return Err(raised("CancelledError", "the host cancelled"));
            }
            if world.inject(f.unavailable, "HandlerUnavailable") {
                return Err(raised("StorageUnavailable", "the handler lost the database"));
            }
            let kind =
                ["VersionConflict", "NotSerializable", "TooLarge"][world.draw(|r| r.usize(0..3))];
            if world.inject(f.save_refused, kind) {
                return Err(raised(kind, "save_context refused"));
            }
            if self.verdict.lock().unwrap().take() == Some(Verdict::Yield) {
                self.yielded = true;
            }
            if !self.yielded && self.checkpoints > 0 {
                self.checkpoints -= 1;
                return Ok(Some(TurnEvent::Checkpoint));
            }
            if std::mem::replace(&mut self.done, true) {
                return Ok(None);
            }
            let output = self.output(world.wall_secs());
            let reached = match (&output.input_required, output.status) {
                (Some(_), _) => Some("Question"),
                (None, Some(TaskState::Working)) => Some("HandlerWorking"),
                (None, Some(_)) => Some("HandlerRejected"),
                _ => None,
            };
            if let Some(kind) = reached {
                *world.ledger.lock().unwrap().injected.entry(kind).or_default() += 1;
            }
            if output.status == Some(TaskState::Working) {
                let mut working = self.handlers.working.lock().unwrap();
                working.insert(self.invocation, self.task_id.clone());
            }
            Ok(Some(TurnEvent::Output { output: output.into() }))
        })
    }

    fn close(self: Box<Self>, _deadline: Ts) -> BoxFuture<'static, Closed> {
        self.handlers.open.lock().unwrap().remove(&self.request);
        Box::pin(std::future::ready(Closed::Clean))
    }
}

impl Handler {
    fn output(&self, at: f64) -> HandlerOutput {
        let mut out = HandlerOutput { interrupted: self.yielded, ..HandlerOutput::default() };
        let said = match (self.yielded, self.ending) {
            (false, Ending::Text) => format!("answer for {}", self.task_id),
            _ => "executed".to_owned(),
        };
        let marker = format!("{MARKER}{}", self.invocation);
        let marker = chat_message(ChatRole::Assistant, vec![said], at + 3e-6, marker).payload;
        let user = serde_json::from_value(self.message.clone()).expect("the turn's chat item");
        out.new_items = vec![user, marker];
        if self.yielded {
            return out;
        }
        match self.ending {
            Ending::Text => out.output = Some(format!("answer for {}", self.task_id).into()),
            Ending::Raise => out.error = Some("RuntimeError: the tool failed".into()),
            Ending::Status(status) => {
                out.output = Some(format!("{status:?}: {}", self.task_id).into());
                out.status = Some(status);
            }
            Ending::Ask => {
                let call_id = format!("call_{}", self.invocation);
                let item = |kind| ChatItem { item: Some(kind) };
                out.new_items.extend([
                    item(chat_item::Item::FunctionCall(FunctionCall {
                        id: format!("item_{}_call", self.invocation),
                        call_id: call_id.clone(),
                        name: "ask_day".into(),
                        arguments: "{}".into(),
                        created_at: Some(timestamp(at + 1e-6)),
                        ..Default::default()
                    })),
                    item(chat_item::Item::FunctionCallOutput(FunctionCallOutput {
                        id: format!("item_{}_wait", self.invocation),
                        call_id: call_id.clone(),
                        name: "ask_day".into(),
                        output: "waiting for the person's answer to: which day?".into(),
                        is_error: false,
                        created_at: Some(timestamp(at + 2e-6)),
                    })),
                ]);
                out.input_required = Some("which day?".into());
                out.pending_call =
                    Some(PendingCall { name: "ask_day".into(), call_id, answer_tool: None });
            }
        }
        out
    }
}
