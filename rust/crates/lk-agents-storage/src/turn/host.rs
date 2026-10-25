//! The host boundary of plan 13.8: [`TurnHost`], the one trait a host implements (the Python
//! `a2a_session` handler or `Agent.run()` behind `_core`, a Node package), what the turn hands it
//! ([`TurnContext`]) and what it reads back ([`HandlerOutput`]). Everything crosses as JSON values
//! or strings; nothing here names a host type.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::Message;
use crate::StorageError;
use crate::run::{CANCEL_REQUESTED_KEY, INTERRUPT_KEY, Item, Ref, RunUsage, TaskRecord, TaskState};
use crate::statements::{Fence, ItemWrite, Lease, ThreadDb};

/// The host's three calls (plan 13.8): run the handler on one turn, and report progress to the
/// wire. The turn calls `handle`; `on_update` and `on_token` are the host's own reporting hooks,
/// named on the trait so that one object is both the handler and the wire.
#[async_trait::async_trait]
pub trait TurnHost: Send + Sync {
    /// Runs the handler on one turn: the Python `a2a_session` handler or `Agent.run()`.
    async fn handle(&self, ctx: TurnContext) -> Result<HandlerOutput, HostError>;
    /// A `working` status frame to the wire.
    async fn on_update(&self, task_id: &str, text: &str);
    /// Streamed answer text to the wire.
    async fn on_token(&self, task_id: &str, delta: &str);
    /// A task stopped running in a turn of the `live` holder: it ended, waits on a question,
    /// or works in the background. Its last frames go out on its own answer stream; `output`
    /// is the reply when the task carried it.
    async fn on_end(&self, _task: &TaskRecord, _output: Option<&Value>) {}
    /// The work this host left running in the background, waiting up to `wait` for some of it
    /// to finish: how much still runs, and the tasks whose work finished. The work of the
    /// `cancel` tasks, which clients canceled, is stopped first.
    async fn background(&self, wait: Duration, _cancel: Vec<String>) -> Background {
        tokio::time::sleep(wait).await;
        Background::default()
    }
    /// The task ended before its background work: what the work left is dropped.
    async fn discard(&self, _task_id: &str) {}
    /// The turn found which task the message is, before it waits on anything: its own, the
    /// question it continues, or, when `resent`, the task the same message id already made,
    /// whose answer is this request's too. `holds`: under `live`, this request took the lane
    /// and is its holder from now on.
    async fn accepted(&self, _task_id: &str, _resent: bool, _holds: bool) {}
    /// Under `live`: this request's message is in the lane's inbox, and another holds the lane.
    /// A holder in the same process, or in a job of the same worker, is told at once.
    async fn arrived(&self, _task_id: &str) {}
    /// Waits up to `wait` for a turn of the task to end where this host can hear it: `true`
    /// when it was told, `false` when the wait ran out and the caller reads the row anyway. A
    /// host that hears a holder in its own process may wait longer than `wait`.
    async fn settled(&self, _task_id: &str, wait: Duration) -> bool {
        tokio::time::sleep(wait).await;
        false
    }
}

/// What [`TurnHost::background`] reports.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn")]
pub struct Background {
    /// Tools still running in the background.
    #[serde(default)]
    #[python(default = "0")]
    pub running: i64,
    /// The tasks whose background work finished: they get one turn together.
    #[serde(default)]
    #[python(default_factory = "list")]
    pub ready: Vec<String>,
    /// A message landed in the inbox, or a client asked to cancel a task, and the host was told
    /// in its process.
    #[serde(default)]
    #[python(default = "False")]
    pub arrived: bool,
}

/// What `handle` cannot report as a handler error: a storage error the host met, which fails the
/// run, and a cancellation, which ends it `canceled`. A handler that raised is
/// [`HandlerOutput::error`], and the turn fails the task.
#[derive(Debug, Clone, thiserror::Error)]
pub enum HostError {
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// The Python `asyncio.CancelledError`; the text is the reason the run records.
    #[error("cancelled: {0}")]
    Cancelled(String),
}

/// What `A2ASessionContext` exposes today, the storage half: the message, the batch, the history,
/// the state, the task, the lane's tasks, and the boundary the host polls. The host record is
/// everything but the boundary, which the binding hands over beside it as a core object.
#[derive(Debug, Clone, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct TurnContext {
    /// The database id of the session this turn runs in.
    pub thread_id: String,
    /// The lane: the registered agent this turn belongs to.
    pub lane: String,
    /// The batch as one user chat message, in the stored item form (`batch_to_chat_message`).
    pub message: Value,
    /// The individual messages with their channel and time; `batch.last()` is the one answered.
    pub batch: Vec<Message>,
    /// The lane's history as loaded, plus the pending input-required answer when there was one.
    pub history: Vec<Item>,
    /// The state as loaded, or the door's default for a fresh lane; `None` for a door with none.
    pub state: Option<Value>,
    /// The task that carries the reply.
    pub task_id: String,
    /// The lane's task rows, newest first, up to 50.
    pub tasks: Vec<TaskRecord>,
    /// Who speaks, bottom first: empty while the door's root agent does; an `AgentTask` on top
    /// holds the conversation until it completes (docs/agent-tasks.md).
    pub agent_stack: Vec<Ref>,
    /// A delegating caller's conversation when the message carried a `chat_ctx` data part (plan
    /// 7.5), as the part's `data`.
    pub caller_history: Option<Value>,
    /// The tool that relays the answer of the question this message answers (a delegate's): it
    /// takes the message as said before the model runs.
    pub answer_tool: Option<String>,
    /// The pending call of the task this message continues: the call that asked.
    pub answer_call_id: Option<String>,
    /// Another question of the lane, in front of the person: a question this turn asks waits
    /// behind it.
    pub front_question: Option<String>,
    /// The oldest question waiting behind none: the turn puts it to the person when its own
    /// task stops waiting, as `{task_id, question, verbatim}`.
    pub relay: Option<Value>,
    /// The question this message answers was cut off by it: `{question, heard, verbatim}`,
    /// `heard` being what the person saw or heard of it, often nothing.
    pub cut_off: Option<Value>,
    /// The turn runs under `live`: a tool may go on in the background after its first
    /// update, and a turn of its own answers the task when it ends.
    pub live: bool,
    /// Asks whether to yield: before each model call and before the answer streams.
    #[python(skip)]
    pub boundary: Boundary,
}

/// A conversation tool's pending call, the one an `InputRequired` came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct PendingCall {
    pub name: String,
    pub call_id: String,
    /// The tool that relays the answer, when a call of the question is a delegate's.
    #[serde(default)]
    #[python(default = "None")]
    pub answer_tool: Option<String>,
}

/// A question of a resumed task that waits behind the one in front of the person.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct WaitingAsk {
    pub task_id: String,
    pub question: String,
    pub pending_call: PendingCall,
    #[serde(default)]
    #[python(default = "False")]
    pub verbatim: bool,
}

/// A journal event the handler recorded (docs/journal.md): `tool_execution_updated` for a
/// call's start, its progress and its end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct HostEvent {
    #[serde(rename = "type")]
    #[host(rename = "type")]
    pub r#type: String,
    pub payload: Value,
}

/// What the turn reads from the handler's results: the `RunResult` fields and the context's
/// metadata, as JSON. Every field has a default: a handler reports only what it has.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn")]
pub struct HandlerOutput {
    /// The answer: text, or a typed value as JSON.
    #[serde(default)]
    #[python(default = "None")]
    pub output: Option<Value>,
    /// The chat items the runs produced, in order, each with its `id` and `type`; empty when no
    /// run was made, in which case the turn stores the message and a text answer itself.
    #[serde(default)]
    #[python(default_factory = "list")]
    pub new_items: Vec<Value>,
    /// The state after the handler; written at commit when it differs from what was loaded.
    #[serde(default)]
    #[python(default = "None")]
    pub state: Option<Value>,
    #[serde(default)]
    #[python(default_factory = "RunUsage")]
    pub usage: RunUsage,
    /// A conversation tool's `InputRequired(question)`.
    #[serde(default)]
    #[python(default = "None")]
    pub input_required: Option<String>,
    /// The call that asked, when `input_required` is set.
    #[serde(default)]
    #[python(default = "None")]
    pub pending_call: Option<PendingCall>,
    /// Of tasks resumed together, the one whose work asked `input_required`: it waits on the
    /// person, and `output` is the reply that answers the others.
    #[serde(default)]
    #[python(default = "None")]
    pub asked_task: Option<String>,
    /// The agent stack after the handler, when a run moved it or a task on it changed state;
    /// `None` leaves the stack as loaded.
    #[serde(default)]
    #[python(default = "None")]
    pub agent_stack: Option<Vec<Ref>>,
    /// The handler yielded at a boundary.
    #[serde(default)]
    #[python(default = "False")]
    pub interrupted: bool,
    /// A tool of the task still runs in the background: the task stays `working` and gets a
    /// turn of its own when the tool ends.
    #[serde(default)]
    #[python(default = "False")]
    pub background: bool,
    /// `input_required` waits behind the question in front of the person; it was not asked.
    #[serde(default)]
    #[python(default = "False")]
    pub queued: bool,
    /// The task whose waiting question the reply put to the person (`TurnContext.relay`).
    #[serde(default)]
    #[python(default = "None")]
    pub relayed: Option<String>,
    /// A newer message cut `input_required` off: what the person saw or heard of it, often
    /// nothing. The question waits, and the next message is its answer.
    #[serde(default)]
    #[python(default = "None")]
    pub heard: Option<String>,
    /// Of tasks resumed together, the others whose work asked too: each waits behind the
    /// question in front of the person.
    #[serde(default)]
    #[python(default_factory = "list")]
    pub waiting: Vec<WaitingAsk>,
    /// What the handler put on the task's terminal row, under short keys.
    #[serde(default)]
    #[python(ty = "dict[str, Any]", import = "typing:Any", default_factory = "dict")]
    pub metadata: Map<String, Value>,
    /// The handler raised: `"<TypeName>: <text>"`; the task fails with it.
    #[serde(default)]
    #[python(default = "None")]
    pub error: Option<String>,
    /// Journal events the handler recorded, in order, written with the commit.
    #[serde(default)]
    #[python(default_factory = "list")]
    pub events: Vec<HostEvent>,
}

struct BoundaryInner {
    db: ThreadDb,
    lane: String,
    holder: String,
    task_id: String,
    interrupt: bool,
    lease: Option<Lease>,
    tripped: AtomicBool,
    client_canceled: AtomicBool,
    /// When the acquire that made this boundary cleared the holder's waiter and picked the
    /// task up: a check within [`FRESH`] of it is answered from it, without a read.
    acquired: Instant,
    /// Under `live`: the last inbox row the turn took. A newer row that interrupts yields.
    arrivals_after: Option<i64>,
    /// The version the run loaded: the fence of an early write.
    version: i64,
}

/// How long the acquire answers the boundary: an interrupt that lands inside this window
/// waits for the next boundary.
const FRESH: Duration = Duration::from_millis(100);

/// The `boundary()` of `_turn`: the host asks before each model call and once more before the
/// answer streams. `true` once means `true` from then on. The lease lost means the run cannot
/// commit anyway, so nothing yields; under `interrupt` a waiter yields the lane; a row a client
/// canceled with `tasks/cancel` yields too, and nothing is written over it (plan 5.5).
#[derive(Clone)]
pub struct Boundary {
    inner: Arc<BoundaryInner>,
}

impl std::fmt::Debug for Boundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Boundary")
            .field("lane", &self.inner.lane)
            .field("task_id", &self.inner.task_id)
            .field("interrupt", &self.inner.interrupt)
            .field("tripped", &self.tripped())
            .finish()
    }
}

impl Boundary {
    pub(crate) fn new(
        db: ThreadDb,
        lane: &str,
        holder: &str,
        task_id: &str,
        interrupt: bool,
        lease: Option<Lease>,
        version: i64,
    ) -> Self {
        Self {
            inner: Arc::new(BoundaryInner {
                db,
                lane: lane.to_owned(),
                holder: holder.to_owned(),
                task_id: task_id.to_owned(),
                interrupt,
                lease,
                tripped: AtomicBool::new(false),
                client_canceled: AtomicBool::new(false),
                acquired: Instant::now(),
                arrivals_after: None,
                version,
            }),
        }
    }

    /// The boundary of a `live` turn that took the inbox up to row `last`: a newer message
    /// that interrupts yields it, one that does not waits for the floor.
    pub(crate) fn live(mut self, last: i64) -> Self {
        Arc::get_mut(&mut self.inner).expect("a fresh boundary").arrivals_after = Some(last);
        self
    }

    pub fn task_id(&self) -> &str {
        &self.inner.task_id
    }

    /// Whether to yield now. With `read` it reads even within [`FRESH`] of the acquire: the
    /// caller was told that something landed.
    pub async fn check(&self, read: bool) -> Result<bool, StorageError> {
        let i = &self.inner;
        if self.tripped() {
            return Ok(true);
        }
        if i.lease.as_ref().is_some_and(Lease::is_lost) {
            return Ok(false);
        }
        if !read && i.acquired.elapsed() < FRESH {
            return Ok(false);
        }
        if let Some(last) = i.arrivals_after {
            let interrupts = i.db.inbox_pending(&i.lane).await?.into_iter().any(|row| {
                row.id > last
                    && row.message.get("metadata").and_then(|m| m.get(INTERRUPT_KEY))
                        != Some(&Value::Bool(false))
            });
            if interrupts {
                i.tripped.store(true, Ordering::Release);
                log::info!(target: "livekit.agents", lane = i.lane.as_str(), taskId = i.task_id.as_str(); "a newer message interrupts the reply");
                return Ok(true);
            }
        }
        let (waiter, row) = i.db.boundary(&i.lane, &i.task_id).await?;
        let status = row.as_ref().map(|r| r.status.clone());
        let cancel_requested = row
            .as_ref()
            .and_then(|r| r.data.get("metadata"))
            .and_then(|m| m.get(CANCEL_REQUESTED_KEY))
            == Some(&Value::Bool(true));
        if i.interrupt && waiter.is_some_and(|w| !w.is_empty() && w != i.holder) {
            i.tripped.store(true, Ordering::Release);
            log::info!(target: "livekit.agents", lane = i.lane.as_str(), taskId = i.task_id.as_str(); "yielding the lane to a newer message");
            return Ok(true);
        }
        if status.as_deref() == Some(TaskState::Canceled.as_str()) || cancel_requested {
            i.tripped.store(true, Ordering::Release);
            i.client_canceled.store(true, Ordering::Release);
            log::info!(target: "livekit.agents", lane = i.lane.as_str(), taskId = i.task_id.as_str(); "the task was canceled by a client; yielding");
        }
        Ok(self.tripped())
    }

    /// Writes the turn's message item now, ahead of the commit and under the run's fence: a
    /// delegation is about to go out, and the expert may read the caller's lane.
    pub async fn record(
        &self,
        item: Value,
        message_id: Option<String>,
    ) -> Result<(), StorageError> {
        let i = &self.inner;
        let Some(lease) = &i.lease else {
            return Ok(()); // a run with no lease commits with no fence: nothing to write ahead
        };
        let item = Item::new(item)?;
        let fence = Fence { version: i.version, lease_token: lease.token().to_owned() };
        i.db.write_items(
            &i.lane,
            &fence,
            &[ItemWrite { id: item.id, message_id, payload: item.payload }],
        )
        .await
    }

    /// A check answered `true` at some point.
    pub fn tripped(&self) -> bool {
        self.inner.tripped.load(Ordering::Acquire)
    }

    /// The yield was a client's `tasks/cancel`.
    pub fn client_canceled(&self) -> bool {
        self.inner.client_canceled.load(Ordering::Acquire)
    }
}

/// What `Turn::run` returns: the message's task after the turn, and the reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn", config = "frozen=True")]
pub struct TurnResult {
    pub task: TaskRecord,
    /// The answer, as the artifact holds it: text, or the data of a typed value.
    pub output: Option<Value>,
    /// The database id of the session the turn ran in.
    pub thread_id: String,
    /// The task that carries the reply: this one, or the later one it was merged into.
    pub answered_by: String,
    pub run_id: Option<String>,
}
