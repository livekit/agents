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
use super::message::interrupts;
use crate::StorageError;
use crate::run::{Item, Ref, RunUsage, TaskRecord, TaskState};
use crate::statements::{Lease, ThreadDb};

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
    /// Under `live`, once the message is in the inbox: its task `task_id` (its own, the one it
    /// continues, or a resend's), whose stream `local` says this worker writes.
    async fn accepted(&self, _task_id: &str, _local: bool) {}
    /// Under `live`: a turn this host ran committed `task`, which goes on its own answer
    /// stream; `output` is the reply when the task carried it.
    async fn on_end(&self, _task: &TaskRecord, _output: Option<&Value>) {}
    /// Under `live`: what the work this host left running in the background queued, after
    /// waiting up to `wait` for some.
    async fn background(&self, wait: Duration) -> Background {
        tokio::time::sleep(wait).await;
        Background::default()
    }
}

/// What [`TurnHost::background`] reports.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage_turn")]
pub struct Background {
    /// The tasks whose work runs on, or whose queue waits: the holder keeps the lane for them.
    #[serde(default)]
    #[python(default_factory = "list")]
    pub running: Vec<String>,
    /// The tasks whose progress and results are delivered now, in one turn with one reply.
    #[serde(default)]
    #[python(default_factory = "list")]
    pub ready: Vec<String>,
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
    /// The tool the message is handed to before the model runs: the last turn's question came
    /// from it with a reference, and this message is its answer.
    pub answer_tool: Option<String>,
    /// What `answer_tool` continues with the answer, such as the expert's task.
    pub answer_ref: Option<String>,
    /// The turn runs under `live`: a tool may go on in the background after its first update.
    #[python(default = "False")]
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
    /// The tool that takes the answer, when the question carries a reference.
    #[serde(default)]
    #[python(default = "None")]
    pub answer_tool: Option<String>,
    /// What the tool that takes the answer continues with it, such as the expert's task.
    #[serde(default)]
    #[python(default = "None")]
    pub answer_ref: Option<String>,
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
    /// The agent stack after the handler, when a run moved it or a task on it changed state;
    /// `None` leaves the stack as loaded.
    #[serde(default)]
    #[python(default = "None")]
    pub agent_stack: Option<Vec<Ref>>,
    /// The handler yielded at a boundary.
    #[serde(default)]
    #[python(default = "False")]
    pub interrupted: bool,
    /// A tool of the task still runs in the background: the task stays `working`, and a turn of
    /// its own answers it when the tool ends.
    #[serde(default)]
    #[python(default = "False")]
    pub background: bool,
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
    /// Under `live`: the last inbox row the turn took. A newer one that interrupts yields.
    arrivals_after: Option<i64>,
}

/// How long the acquire answers the boundary: an interrupt that lands inside this window
/// waits for the next boundary.
const FRESH: Duration = Duration::from_millis(100);

/// The `boundary()` of `_turn`: the host asks before each model call and once more before the
/// answer streams. `true` once means `true` from then on. The lease lost means the run cannot
/// commit anyway, so nothing yields; under `interrupt` a waiter yields the lane, under `live` a
/// newer message that interrupts; a row a client canceled with `tasks/cancel` yields too, and
/// nothing is written over it (plan 5.5).
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
            }),
        }
    }

    /// The boundary of a `live` turn that took the inbox up to row `last`.
    pub(crate) fn live(mut self, last: i64) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.arrivals_after = Some(last);
        }
        self
    }

    pub fn task_id(&self) -> &str {
        &self.inner.task_id
    }

    /// Whether to yield now.
    pub async fn check(&self) -> Result<bool, StorageError> {
        let i = &self.inner;
        if self.tripped() {
            return Ok(true);
        }
        if i.lease.as_ref().is_some_and(Lease::is_lost) {
            return Ok(false);
        }
        if i.acquired.elapsed() < FRESH {
            return Ok(false);
        }
        if let Some(last) = i.arrivals_after
            && i.db
                .inbox_pending(&i.lane)
                .await?
                .iter()
                .any(|r| r.id > last && interrupts(&r.message))
        {
            i.tripped.store(true, Ordering::Release);
            log::info!(target: "livekit.agents", lane = i.lane.as_str(), taskId = i.task_id.as_str(); "a newer message interrupts the reply");
            return Ok(true);
        }
        let (waiter, status) = i.db.boundary(&i.lane, &i.task_id).await?;
        if i.interrupt && waiter.is_some_and(|w| !w.is_empty() && w != i.holder) {
            i.tripped.store(true, Ordering::Release);
            log::info!(target: "livekit.agents", lane = i.lane.as_str(), taskId = i.task_id.as_str(); "yielding the lane to a newer message");
            return Ok(true);
        }
        if status.as_deref() == Some(TaskState::Canceled.as_str()) {
            i.tripped.store(true, Ordering::Release);
            i.client_canceled.store(true, Ordering::Release);
            log::info!(target: "livekit.agents", lane = i.lane.as_str(), taskId = i.task_id.as_str(); "the task was canceled by a client; yielding");
        }
        Ok(self.tripped())
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
