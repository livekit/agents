use std::collections::BTreeMap;
use std::future::Future;
use std::panic::{AssertUnwindSafe, Location};
use std::pin::Pin;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use tokio::sync::{Notify, oneshot, watch};

use crate::{Clock, Ts};

/// Defined here to keep dependencies to `{tokio, async-trait}`.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Which way a task ended; the first of completion and cancellation to happen wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOutcome {
    Completed,
    Cancelled,
    Panicked,
}

/// Why an [`OwnedTask`] produced no value.
#[derive(Debug)]
pub enum TaskError {
    /// Cancelled by its owner, or dropped unfinished by its executor.
    Cancelled,
    /// The work panicked, with the panic's message. The panic itself still unwinds into the
    /// executor once the outcome is reported, so a harness judging its tasks sees it.
    Panicked(String),
}

impl TaskError {
    pub fn outcome(&self) -> TaskOutcome {
        match self {
            TaskError::Cancelled => TaskOutcome::Cancelled,
            TaskError::Panicked(_) => TaskOutcome::Panicked,
        }
    }
}

impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskError::Cancelled => f.write_str("task cancelled"),
            TaskError::Panicked(message) => write!(f, "task panicked: {message}"),
        }
    }
}

impl std::error::Error for TaskError {}

/// The one way to own spawned work: dropping it requests cancellation, never detaches; awaiting
/// it yields the work's value or why there is none, only after the work and everything it owned
/// were dropped and its settle step ran.
#[must_use = "dropping an OwnedTask cancels it"]
pub struct OwnedTask<T = ()> {
    /// `None` only once handed to an [`AbortHandle`].
    cancel: Option<watch::Sender<bool>>,
    done: oneshot::Receiver<Result<T, TaskError>>,
}

impl<T> OwnedTask<T> {
    /// Requests cancellation; idempotent. The outcome is reported by the task, not implied here.
    pub fn cancel(&self) {
        if let Some(cancel) = &self.cancel {
            cancel.send_replace(true);
        }
    }

    /// Settled: the work ended or was dropped and the outcome was reported.
    pub fn is_finished(&self) -> bool {
        self.cancel.as_ref().is_none_or(watch::Sender::is_closed)
    }

    /// Gives up ownership for a cancel-only handle; the work then ends on its own.
    pub fn into_abort_handle(mut self) -> AbortHandle {
        AbortHandle(self.cancel.take().expect("an owned task has its cancel sender"))
    }
}

impl<T> Future for OwnedTask<T> {
    type Output = Result<T, TaskError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.done)
            .poll(cx)
            .map(|reported| reported.unwrap_or(Err(TaskError::Cancelled)))
    }
}

impl<T> Drop for OwnedTask<T> {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl<T> std::fmt::Debug for OwnedTask<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedTask").field("finished", &self.is_finished()).finish()
    }
}

/// A cancel-only handle on work nobody owns. Only the voice crate still spawns such work.
#[derive(Clone)]
pub struct AbortHandle(watch::Sender<bool>);

impl AbortHandle {
    pub fn abort(&self) {
        self.0.send_replace(true);
    }

    pub fn is_finished(&self) -> bool {
        self.0.is_closed()
    }
}

impl std::fmt::Debug for AbortHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AbortHandle").field("finished", &self.is_finished()).finish()
    }
}

/// Spawned work as its executor sees it: a future to run to completion. Only [`SpawnerExt`]
/// builds one, so every task an executor runs has an [`OwnedTask`] owner.
pub struct Spawned {
    work: BoxFuture<'static, ()>,
    site: &'static Location<'static>,
}

impl Spawned {
    /// The line that spawned it, so an executor can place a task it reports as stuck.
    pub fn site(&self) -> &'static Location<'static> {
        self.site
    }
}

impl Future for Spawned {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.work.as_mut().poll(cx)
    }
}

/// `tokio::spawn` panics on a host thread with no reactor, so the executor is injected. It runs
/// what it is handed to completion; cancellation and outcomes are [`OwnedTask`]'s, the same on
/// every executor (Tokio, the host loop's virtual time, the DST's scheduler).
pub trait Spawner: Send + Sync {
    fn execute(&self, task: Spawned);
}

/// Blanket-implemented, including for `dyn Spawner`.
pub trait SpawnerExt: Spawner {
    #[track_caller]
    fn spawn<T, F>(&self, work: F) -> OwnedTask<T>
    where
        T: Send + 'static,
        F: Future<Output = T> + Send + 'static,
    {
        launch(self, Location::caller(), work, |_| async {})
    }

    /// Runs `settled` with the outcome once `work` is dropped, also when the owner cancelled it.
    #[track_caller]
    fn spawn_settled<W, S, F>(&self, work: W, settled: S) -> OwnedTask
    where
        W: Future<Output = ()> + Send + 'static,
        S: FnOnce(TaskOutcome) -> F + Send + 'static,
        F: Future<Output = ()> + Send + 'static,
    {
        launch(self, Location::caller(), work, settled)
    }

    /// Unowned work with a cancel-only handle. Only the voice crate still uses it.
    #[track_caller]
    fn spawn_unboxed<F>(&self, work: F) -> AbortHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        launch(self, Location::caller(), work, |_| async {}).into_abort_handle()
    }
}

impl<S: Spawner + ?Sized> SpawnerExt for S {}

fn launch<T, W, S, F, X>(
    spawner: &X,
    site: &'static Location<'static>,
    work: W,
    settled: S,
) -> OwnedTask<T>
where
    X: Spawner + ?Sized,
    T: Send + 'static,
    W: Future<Output = T> + Send + 'static,
    S: FnOnce(TaskOutcome) -> F + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let (cancel, mut cancelled) = watch::channel(false);
    let (report, done) = oneshot::channel();
    let task = async move {
        let mut panic = None;
        let result = {
            let mut work = std::pin::pin!(work);
            // A dropped sender without a request is an [`AbortHandle`] gone: the work runs on.
            let mut requested = std::pin::pin!(async {
                if cancelled.wait_for(|requested| *requested).await.is_err() {
                    std::future::pending::<()>().await;
                }
            });
            std::future::poll_fn(|cx| {
                if requested.as_mut().poll(cx).is_ready() {
                    return Poll::Ready(Err(TaskError::Cancelled));
                }
                match std::panic::catch_unwind(AssertUnwindSafe(|| work.as_mut().poll(cx))) {
                    Ok(Poll::Pending) => Poll::Pending,
                    Ok(Poll::Ready(value)) => Poll::Ready(Ok(value)),
                    Err(payload) => {
                        let message = payload
                            .downcast_ref::<&str>()
                            .map(|m| (*m).to_owned())
                            .or_else(|| payload.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "non-string payload".to_owned());
                        panic = Some(payload);
                        Poll::Ready(Err(TaskError::Panicked(message)))
                    }
                }
            })
            .await
        };
        let outcome = result.as_ref().map_or_else(TaskError::outcome, |_| TaskOutcome::Completed);
        settled(outcome).await;
        let _ = report.send(result);
        drop(cancelled);
        if let Some(payload) = panic {
            std::panic::resume_unwind(payload);
        }
    };
    spawner.execute(Spawned { work: Box::pin(task), site });
    OwnedTask { cancel: Some(cancel), done }
}

/// Named [`OwnedTask`]s and a way to wait for them to be gone (`utils.aio.TaskSet`). Each task
/// holds the set until it ends, so dropping the owner's handle cancels nothing: `close`,
/// `abort_all` and `drain` do.
#[derive(Default)]
pub struct TaskSet {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    inner: Mutex<TaskEntries>,
    changed: Notify,
}

/// A task's key in its [`TaskSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TaskId(u64);

#[derive(Default)]
struct TaskEntries {
    closed: bool,
    next: u64,
    tasks: BTreeMap<u64, (String, Option<OwnedTask>)>,
}

/// Owned by the work: leaves the set when the work is dropped, however it ended.
struct Leave {
    table: Arc<TaskSet>,
    id: u64,
}

impl Drop for Leave {
    fn drop(&mut self) {
        let left = self.table.inner.lock().expect("task table").tasks.remove(&self.id);
        drop(left);
        self.table.changed.notify_waiters();
    }
}

impl TaskSet {
    #[track_caller]
    pub fn spawn(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) -> TaskId {
        self.try_spawn(spawner, name, fut).expect("task admission is closed")
    }

    /// Admit a task unless its owner has permanently closed this set.
    #[track_caller]
    pub fn try_spawn(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) -> Option<TaskId> {
        let id = {
            let mut g = self.inner.lock().expect("task table");
            if g.closed {
                return None;
            }
            g.next += 1;
            let id = g.next;
            g.tasks.insert(id, (name.to_owned(), None));
            id
        };
        let leave = Leave { table: Arc::clone(self), id };
        let task = spawner.spawn(async move {
            let _leave = leave;
            fut.await;
        });
        let mut g = self.inner.lock().expect("task table");
        if g.closed {
            task.cancel();
        }
        // Absent when the work already ended; the task is then dropped finished.
        if let Some(entry) = g.tasks.get_mut(&id) {
            entry.1 = Some(task);
        }
        Some(TaskId(id))
    }

    /// Cancels one task; a finished one is ignored.
    pub fn cancel(&self, id: TaskId) {
        if let Some((_, Some(task))) = self.inner.lock().expect("task table").tasks.get(&id.0) {
            task.cancel();
        }
    }

    pub fn names(&self) -> Vec<String> {
        self.inner.lock().expect("task table").tasks.values().map(|(n, _)| n.clone()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().expect("task table").tasks.is_empty()
    }

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    /// `true` when the table emptied.
    pub async fn wait_empty(&self, clock: &dyn Clock, deadline: Option<Ts>) -> bool {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_empty() {
                return true;
            }
            match deadline {
                None => notified.await,
                Some(at) => {
                    tokio::select! {
                        _ = &mut notified => {}
                        _ = clock.wait_until(at) => return self.is_empty(),
                    }
                }
            }
        }
    }

    /// Permanently refuse new work and abort admitted tasks.
    pub fn close(&self) {
        self.inner.lock().expect("task table").closed = true;
        self.abort_all();
    }

    /// Abort every task without waiting: an owner going away.
    pub fn abort_all(&self) {
        for task in
            self.inner.lock().expect("task table").tasks.values().filter_map(|(_, t)| t.as_ref())
        {
            task.cancel();
        }
    }

    /// `cancel_and_wait(*tasks)`; returns the names aborted.
    pub async fn cancel_and_wait(&self, clock: &dyn Clock) -> Vec<String> {
        let names = self.names();
        self.abort_all();
        self.wait_empty(clock, None).await;
        names
    }

    /// `_drain(tasks, deadline)`: wait until `deadline`, then cancel the stragglers and name them.
    pub async fn drain(&self, clock: &dyn Clock, deadline: Ts) -> Vec<String> {
        if self.wait_empty(clock, Some(deadline)).await {
            return Vec::new();
        }
        self.cancel_and_wait(clock).await
    }
}
