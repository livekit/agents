//! Task ownership's implementation: [`launch`] turns a [`Launch`] into the future an executor
//! runs and the [`OwnedTask`] its owner holds, the same on every executor (Tokio here, the host
//! loop's virtual time, the DST's scheduler); [`TaskSet`] groups owned tasks by name.

use std::any::Any;
use std::collections::BTreeMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};
use std::task::Poll;

use lk_agents_abi::{
    BoxFuture, Clock, Launch, OwnedTask, Spawner, SpawnerExt, TaskControl, TaskError, TaskOutcome,
    Ts,
};
use tokio::runtime::Handle;
use tokio::sync::{Notify, oneshot, watch};

struct Control(watch::Sender<bool>);

impl TaskControl for Control {
    fn cancel(&self) {
        self.0.send_replace(true);
    }

    fn is_finished(&self) -> bool {
        self.0.is_closed()
    }
}

/// The future to run to completion on the executor, and its owner's handle. Cancellation wins
/// over a work that is ready in the same poll; `settled` runs after the work is dropped; the
/// outcome is reported last; a panic is reported, then resumed into the executor.
pub fn launch(task: Launch) -> (BoxFuture<'static, ()>, OwnedTask<Box<dyn Any + Send>>) {
    let Launch { work, settled, site: _ } = task;
    let (cancel, mut cancelled) = watch::channel(false);
    let (report, done) = oneshot::channel();
    let run = async move {
        let mut panic = None;
        let result = {
            let mut work = work;
            // A dropped control without a request is detached work: it runs on.
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
        if let Some(settled) = settled {
            let outcome =
                result.as_ref().map_or_else(TaskError::outcome, |_| TaskOutcome::Completed);
            settled(outcome).await;
        }
        let _ = report.send(result);
        drop(cancelled);
        if let Some(payload) = panic {
            std::panic::resume_unwind(payload);
        }
    };
    let output = Box::pin(async move { done.await.unwrap_or(Err(TaskError::Cancelled)) });
    (Box::pin(run), OwnedTask::new(Arc::new(Control(cancel)), output))
}

/// Never builds a runtime: an import-time thread would not survive the forkserver's `fork()`.
#[derive(Clone, Debug)]
pub struct TokioSpawner {
    handle: Handle,
}

impl TokioSpawner {
    pub fn new(handle: Handle) -> Self {
        TokioSpawner { handle }
    }

    /// Call it inside the runtime; panics otherwise.
    pub fn current() -> Self {
        TokioSpawner::new(Handle::current())
    }

    pub fn handle(&self) -> &Handle {
        &self.handle
    }
}

impl Spawner for TokioSpawner {
    fn launch(&self, task: Launch) -> OwnedTask<Box<dyn Any + Send>> {
        let (run, owned) = launch(task);
        drop(self.handle.spawn(run));
        owned
    }
}

/// Named [`OwnedTask`]s and a way to wait for them to be gone (`utils.aio.TaskSet`). Each task
/// holds the set until it ends, so dropping the owner's handle cancels nothing: `close`,
/// `abort_all` and `drain` do.
#[derive(Default)]
pub struct TaskSet {
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

/// Owned by the work: leaves the set when the work is dropped, however it ended. Also handed
/// out by [`TaskSet::track`] for work another set runs.
pub struct Leave {
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
    /// Counts work run elsewhere (another set's task) until the guard drops, so this set's
    /// waits include it; it cannot be cancelled from here.
    pub fn track(self: &Arc<Self>, name: &str) -> Leave {
        let mut g = self.inner.lock().expect("task table");
        g.next += 1;
        let id = g.next;
        g.tasks.insert(id, (name.to_owned(), None));
        Leave { table: Arc::clone(self), id }
    }

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

    /// Whether the task has not left the set yet.
    pub fn contains(&self, id: TaskId) -> bool {
        self.inner.lock().expect("task table").tasks.contains_key(&id.0)
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
                        biased;
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

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{SpawnerExt, TaskOutcome};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn closing_task_admission_refuses_future_work_and_settles_admitted_work() {
        let tasks = Arc::new(TaskSet::default());
        let spawner = TokioSpawner::current();
        tasks.spawn(&spawner, "pending", std::future::pending());
        tasks.close();
        assert!(tasks.try_spawn(&spawner, "late", async {}).is_none());
        assert!(tasks.wait_empty(&crate::clock::RealClock::new(), None).await);
    }

    async fn outcome_of(
        work: impl Future<Output = ()> + Send + 'static,
        cancel: bool,
    ) -> (TaskOutcome, bool) {
        let (tx, rx) = oneshot::channel();
        let (dropped_tx, mut dropped_rx) = oneshot::channel::<()>();
        struct Owned(Option<oneshot::Sender<()>>);
        impl Drop for Owned {
            fn drop(&mut self) {
                let _ = self.0.take().map(|tx| tx.send(()));
            }
        }
        let owned = Owned(Some(dropped_tx));
        let task = TokioSpawner::current().spawn_settled(
            async move {
                let _owned = owned;
                work.await
            },
            move |outcome| async move {
                // The work's resources are released before the outcome is reported.
                let _ = tx.send((outcome, dropped_rx.try_recv().is_ok()));
            },
        );
        if cancel {
            task.cancel();
            task.cancel();
        }
        let reported = rx.await.unwrap();
        let awaited = task.await.map_or_else(|e| e.outcome(), |()| TaskOutcome::Completed);
        assert_eq!(awaited, reported.0, "the owner sees the outcome the settle step saw");
        reported
    }

    // Catches: an outcome reported while the work still holds its resources, a repeated cancel
    // misbehaving, or a panic escaping as a lost task.
    #[tokio::test]
    async fn each_task_reports_one_outcome_after_its_work_is_dropped() {
        assert_eq!(outcome_of(async {}, false).await, (TaskOutcome::Completed, true));
        let blocked = std::future::pending::<()>();
        assert_eq!(outcome_of(blocked, true).await, (TaskOutcome::Cancelled, true));
        assert_eq!(
            outcome_of(async { panic!("boom") }, false).await,
            (TaskOutcome::Panicked, true)
        );
    }

    // Catches: a dropped handle detaching its task instead of cancelling it.
    #[tokio::test]
    async fn dropping_the_handle_cancels_the_task() {
        let (tx, rx) = oneshot::channel();
        drop(TokioSpawner::current().spawn_settled(
            tokio::time::sleep(Duration::from_secs(3600)),
            move |outcome| async move {
                let _ = tx.send(outcome);
            },
        ));
        assert_eq!(rx.await.unwrap(), TaskOutcome::Cancelled);
    }
}
