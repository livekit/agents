//! Settling the result does not imply that entry/exit handoffs have completed.

use crate::workflows::ToolError;
use lk_agents_abi::Monotone;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A task's identity, `task_`-prefixed, minted by its creator and scoped to it; never globally ordered.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(pub Arc<str>);

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for TaskId {
    fn from(s: &str) -> Self {
        TaskId(Arc::from(s))
    }
}

impl From<String> for TaskId {
    fn from(s: String) -> Self {
        TaskId(Arc::from(s))
    }
}

/// The lifecycle: `__started` and `__inactive_ev` folded into one ordered, [`Monotone`] value
/// (`Created` and `Exited` are both inactive; the event can't tell them apart). Runs once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPhase {
    /// `__started == False`, `__inactive_ev` set. Awaitable.
    Created,
    /// Past "await only once"; taking the inline slot and pausing the parent.
    Entering,
    /// The task's activity is live and `__fut` is being awaited.
    Running,
    /// Settled, with the handoff back in flight (the reference's `finally`). Still active.
    Exiting,
    /// The handoff is complete. `__inactive_ev` set again.
    Exited,
}

impl TaskPhase {
    /// `not __inactive_ev.is_set()`.
    pub fn is_active(self) -> bool {
        matches!(self, TaskPhase::Entering | TaskPhase::Running | TaskPhase::Exiting)
    }
}

/// Every way an operation on a [`TaskHandle`] can refuse; `Display` is the reference message (public API).
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TaskError {
    /// `RuntimeError(f"{cls} is already done")` (`agent.AgentTask.cancel`).
    #[error("{task} is already done")]
    AlreadyDone { task: TaskId },
    /// `RuntimeError(f"{cls} is not re-entrant, await only once")` (`agent.AgentTask.__await_impl`).
    #[error("{task} is not re-entrant, await only once")]
    NotReentrant { task: TaskId },
    /// Already taken by an earlier [`joined`](TaskHandle::joined): a driver bug, not the user's double await.
    #[error("{task} has already been joined")]
    AlreadyJoined { task: TaskId },
}

struct State<T> {
    id: TaskId,
    /// `__fut.done()`, independent of the phase: a result can settle while `Entering` or `Running`.
    settled: Monotone<bool>,
    phase: Monotone<TaskPhase>,
    /// `cancel()` asked for the parent activity to be interrupted; read by the supervisor.
    cancel_requested: Monotone<bool>,
    /// The `__started` half of [`begin`](TaskHandle::begin), a compare-and-set so two `begin`s
    /// can't both proceed.
    started: AtomicBool,
    outcome: Mutex<Option<Result<T, ToolError>>>,
}

/// The caller-facing half of an `AgentTask`; cheap to clone.
pub struct TaskHandle<T> {
    state: Arc<State<T>>,
}

impl<T> Clone for TaskHandle<T> {
    fn clone(&self) -> Self {
        TaskHandle { state: Arc::clone(&self.state) }
    }
}

impl<T> std::fmt::Debug for TaskHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskHandle")
            .field("id", &self.state.id)
            .field("phase", &self.phase())
            .field("done", &self.done())
            .finish()
    }
}

impl<T> TaskHandle<T> {
    pub fn new(id: TaskId) -> Self {
        TaskHandle {
            state: Arc::new(State {
                id,
                settled: Monotone::new(false),
                phase: Monotone::new(TaskPhase::Created),
                cancel_requested: Monotone::new(false),
                started: AtomicBool::new(false),
                outcome: Mutex::new(None),
            }),
        }
    }

    pub fn id(&self) -> TaskId {
        self.state.id.clone()
    }

    /// `self.done()`: the future's, not "the handoff finished".
    pub fn done(&self) -> bool {
        self.state.settled.get()
    }

    pub fn phase(&self) -> TaskPhase {
        self.state.phase.get()
    }

    pub fn cancel_requested(&self) -> bool {
        self.state.cancel_requested.get()
    }

    /// `self.complete(result)` (`agent.AgentTask.cancel`).
    pub fn complete(&self, value: T) -> Result<(), TaskError> {
        self.settle(Ok(value))
    }

    /// `self.complete(ToolError(...))` -- the `isinstance(result, Exception)` arm.
    pub fn fail(&self, error: ToolError) -> Result<(), TaskError> {
        self.settle(Err(error))
    }

    fn settle(&self, outcome: Result<T, ToolError>) -> Result<(), TaskError> {
        {
            let mut slot = self.state.outcome.lock().unwrap();
            if slot.is_some() || self.state.settled.get() {
                return Err(TaskError::AlreadyDone { task: self.state.id.clone() });
            }
            *slot = Some(outcome);
        }
        // After the value is in the slot: an observer woken by `settled` must find it.
        self.state.settled.set(true);
        Ok(())
    }

    /// `cancel()`: requests the interrupt (the driver performs it on the task's speeches before
    /// resuming the parent), then completes with `ToolError`. Idempotent. `name` is what the message
    /// calls the task.
    pub fn cancel(&self, name: &str) {
        self.state.cancel_requested.set(true);
        let _ = self.fail(ToolError::plain(format!("AgentTask {name} is cancelled")));
    }

    /// Only request the interrupt, for a host that settled the handle itself.
    pub fn request_cancel(&self) {
        self.state.cancel_requested.set(true);
    }

    /// The awaiter went away without the task completing (`_handle_task_done`). Must be called
    /// explicitly: a cancelled Python awaiter wakes nothing on the Rust side, and the supervisor
    /// would park forever holding the parent's interruption hold. Returns whether it landed.
    pub fn abandon(&self) -> bool {
        self.fail(ToolError::plain(format!(
            "The asyncio.Task finished before {} was completed.",
            self.state.id
        )))
        .is_ok()
    }

    /// "await only once" as a state transition; called by the supervisor.
    pub fn begin(&self) -> Result<(), TaskError> {
        if self.state.started.swap(true, Ordering::SeqCst) {
            return Err(TaskError::NotReentrant { task: self.state.id.clone() });
        }
        self.state.phase.set(TaskPhase::Entering);
        Ok(())
    }

    /// Advance the lifecycle. Monotone: a backwards move is a debug panic, not a state change.
    pub fn advance(&self, to: TaskPhase) {
        self.state.phase.set(to);
    }

    /// `await self.__fut` plus the reference's `finally`: resolves on [`TaskPhase::Exited`], so the
    /// tool body can't resume before the parent activity is live again. Takes the outcome; a second
    /// call is [`TaskError::AlreadyJoined`].
    pub async fn joined(&self) -> Result<Result<T, ToolError>, TaskError> {
        self.state.phase.wait_for(|p| *p >= TaskPhase::Exited).await;
        self.take_outcome().ok_or(TaskError::AlreadyJoined { task: self.state.id.clone() })
    }

    pub fn take_outcome(&self) -> Option<Result<T, ToolError>> {
        self.state.outcome.lock().unwrap().take()
    }

    /// The settled value without taking it (a consuming peek would empty the awaiter's handle).
    pub fn peek_outcome(&self) -> Option<Result<T, ToolError>>
    where
        T: Clone,
    {
        self.state.outcome.lock().unwrap().clone()
    }

    /// Settlement without handoff ordering: what the supervisor waits on.
    pub async fn wait_settled(&self) {
        self.state.settled.wait_for(|s| *s).await;
    }

    /// `_wait_for_inactive()`: true at `Created` too (`__inactive_ev` is set in `__init__`).
    pub async fn wait_for_inactive(&self) {
        self.state.phase.wait_for(|p| !p.is_active()).await;
    }
}

/// A host task's handle: its result is an arbitrary host object, carried as a side-table key the
/// core never inspects (the handle holds it, so a result outlives an awaiter that went away).
pub type HostTaskHandle = TaskHandle<lk_agents_abi::HostRef>;

/// What awaiting a host's task can raise: the handle's refusal, or the tool error it settled with
/// -- the reference's `raise ToolError(..)`, which user code catches as `ToolError`.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JoinError {
    #[error(transparent)]
    Task(#[from] TaskError),
    #[error("{}", .0.message)]
    #[host(kind = "livekit.agents.llm.tool_context.ToolError")]
    Tool(ToolError),
}

/// A cancelled Python awaiter drops its future and wakes nothing here, so the host calls
/// [`abandon`](TaskHandle::abandon) in its `finally` for the driver to hand the parent frame back.
#[lk_agents_abi::host(object(name = "TaskHandle", cell = "arc"))]
impl HostTaskHandle {
    /// `id` is the host's (`utils.shortuuid("task_")`); omitted, a process-unique one.
    #[host(new)]
    pub fn for_host(id: Option<String>) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = id.unwrap_or_else(|| format!("task_{:012x}", NEXT.fetch_add(1, Ordering::Relaxed)));
        TaskHandle::new(id.into())
    }

    /// `task_<shortuuid>`. Stable for the life of the handle and used in every message the core produces.
    #[host(getter(name = "id"))]
    pub fn host_id(&self) -> String {
        self.state.id.to_string()
    }

    /// `self.done()` -- **`__fut.done()`, not "the handoff finished"**. The reference's `done()`
    /// is the future's, and `warm_transfer.py` guards on exactly that.
    #[host(getter(name = "done"))]
    pub fn host_done(&self) -> bool {
        self.done()
    }

    /// Where the task is in its lifecycle: `created`, `entering`, `running`, `exiting`, `exited`.
    #[host(getter(name = "phase"))]
    pub fn host_phase(&self) -> &'static str {
        match self.phase() {
            TaskPhase::Created => "created",
            TaskPhase::Entering => "entering",
            TaskPhase::Running => "running",
            TaskPhase::Exiting => "exiting",
            TaskPhase::Exited => "exited",
        }
    }

    /// `self.complete(result)`. Raises `RuntimeError` on a second call, as the reference does
    /// (`agent.py:874`).
    #[host(method(name = "complete"))]
    pub fn host_complete(&self, result: lk_agents_abi::HostRef) -> Result<(), TaskError> {
        self.complete(result)
    }

    /// `self.complete(ToolError(...))` -- the `isinstance(result, Exception)` arm.
    #[host(method(name = "fail", kw_only = "class_name"))]
    pub fn host_fail(&self, message: String, class_name: Option<String>, reason: Option<String>) -> Result<(), TaskError> {
        self.fail(match class_name {
            Some(c) => ToolError::subclass(c, message, reason.unwrap_or_default()),
            None => ToolError::plain(message),
        })
    }

    /// `cancel()` (`agent.py:871`): settles with `ToolError("AgentTask {name} is cancelled")` unless
    /// already settled, and requests an interrupt of the task's speeches. Idempotent.
    #[host(method(name = "cancel"))]
    pub fn host_cancel(&self, name: Option<String>) {
        let name = name.unwrap_or_else(|| self.state.id.to_string());
        self.cancel(&name);
    }

    /// The interrupt half of `cancel()` alone, for a host that settled the handle itself (with the
    /// exception object it wants the awaiter to see) and still wants the task's speeches stopped.
    #[host(method(name = "request_cancel"))]
    pub fn host_request_cancel(&self) {
        self.request_cancel();
    }

    /// The awaiter went away: call this in a `finally`, always (`_handle_task_done`, `agent.py:928`).
    ///
    /// A dropped awaiter cancels nothing on the Rust side, so without this the task's frame stays
    /// installed. Returns whether it terminated something; `False` on a completed task.
    #[host(method(name = "abandon"))]
    pub fn host_abandon(&self) -> bool {
        self.abandon()
    }

    /// `await self.__fut` plus the reference's `finally`: resolves on `exited`, not on settlement.
    /// The side table hands back the *original* object, so `await task` is the value the tool
    /// passed to `complete()` and not a copy of it.
    #[host(method(name = "joined"))]
    pub async fn host_joined(&self) -> Result<lk_agents_abi::HostRef, JoinError> {
        self.joined().await?.map_err(JoinError::Tool)
    }

    /// The settled value without the handoff (`__fut.result()`, not `await task`), leaving it for a
    /// real awaiter. `None` while unsettled; a failure raises.
    #[host(method(name = "result"))]
    pub fn host_result(&self) -> Result<Option<lk_agents_abi::HostRef>, JoinError> {
        self.peek_outcome().transpose().map_err(JoinError::Tool)
    }

    /// `_wait_for_inactive()` (`agent.py:1061`). True at `created` as well as at `exited`, because
    /// the reference sets `__inactive_ev` in `__init__`.
    #[host(method(name = "wait_for_inactive"))]
    pub async fn host_wait_for_inactive(&self) {
        self.wait_for_inactive().await;
    }

    #[host(method(name = "__repr__", pid = false))]
    pub fn host_repr(&self) -> String {
        format!("TaskHandle(id={}, done={})", self.state.id, self.done())
    }
}
