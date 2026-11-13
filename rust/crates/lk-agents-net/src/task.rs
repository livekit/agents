use std::any::Any;
use std::collections::BTreeMap;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use parking_lot::Mutex;
use std::task::Poll;

use lk_agents_abi::{
    BoxFuture, Clock, HostRef, Launch, OwnedTask, Settle, Spawner, SpawnerExt, TaskControl,
    TaskError, TaskOutcome, Ts,
};
use tokio::runtime::Handle;
use tokio::sync::oneshot;

use crate::signal;
use tokio_util::sync::CancellationToken;

struct Control {
    cancel: CancellationToken,
    finished: CancellationToken,
}

impl TaskControl for Control {
    fn cancel(&self) {
        self.cancel.cancel();
    }

    fn is_finished(&self) -> bool {
        self.finished.is_cancelled()
    }
}

/// The future to run to completion on the executor, and its owner's handle. Cancellation wins
/// over a work that is ready in the same poll; `settled` runs after the work is dropped; the
/// outcome is reported last; a panic is reported, then resumed into the executor.
pub fn launch(task: Launch) -> (BoxFuture<'static, ()>, OwnedTask<Box<dyn Any + Send>>) {
    let Launch { work, settled, site: _ } = task;
    let control = Control { cancel: CancellationToken::new(), finished: CancellationToken::new() };
    let (report, done) = oneshot::channel();
    let run = run_launched(
        work,
        settled,
        control.cancel.clone(),
        control.finished.clone().drop_guard(),
        report,
    );
    let output = Box::pin(async move { done.await.unwrap_or(Err(TaskError::Cancelled)) });
    (Box::pin(run), OwnedTask::new(Arc::new(control), output))
}

type Launched = Box<dyn Any + Send>;

async fn run_launched(
    work: BoxFuture<'static, Launched>,
    settled: Option<Settle>,
    cancelled: CancellationToken,
    finished: tokio_util::sync::DropGuard,
    report: oneshot::Sender<Result<Launched, TaskError>>,
) {
    let mut panic = None;
    let result = {
        // The work is dropped at the end of this block, before `settled` runs.
        let mut work = work;
        // A dropped control without a request is detached work: it runs on.
        let mut requested = std::pin::pin!(cancelled.cancelled());
        std::future::poll_fn(|cx| {
            if requested.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Err(TaskError::Cancelled));
            }
            match std::panic::catch_unwind(AssertUnwindSafe(|| work.as_mut().poll(cx))) {
                Ok(Poll::Pending) => Poll::Pending,
                Ok(Poll::Ready(value)) => Poll::Ready(Ok(value)),
                Err(payload) => {
                    let message = panic_message(&*payload);
                    panic = Some(payload);
                    Poll::Ready(Err(TaskError::Panicked(message)))
                }
            }
        })
        .await
    };
    if let Some(settled) = settled {
        let outcome = result.as_ref().map_or_else(TaskError::outcome, |_| TaskOutcome::Completed);
        settled(outcome).await;
    }
    let _ = report.send(result);
    drop(finished);
    if let Some(payload) = panic {
        std::panic::resume_unwind(payload);
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|m| (*m).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string payload".to_owned())
}

tokio::task_local! {
    /// The host context of the binding entry whose native future this task runs, or descends
    /// from: what a proxy taken with no running loop calls in (`HostContext for Option<HostRef>`).
    static HOST_CONTEXT: HostRef;
}

/// The host context of the operation driving this task, if a binding entry set one.
pub fn host_context() -> Option<HostRef> {
    HOST_CONTEXT.try_with(Clone::clone).ok()
}

/// `fut` driven by the operation whose host context is `context`, and every task it spawns on a
/// [`TokioSpawner`] (or through [`inherit`]).
pub fn in_host_context<F: Future + Send + 'static>(
    context: HostRef,
    fut: F,
) -> BoxFuture<'static, F::Output> {
    Box::pin(HOST_CONTEXT.scope(context, fut))
}

/// `fut` in the current task's host context, for a task spawned with it.
pub fn inherit<F: Future + Send + 'static>(fut: F) -> BoxFuture<'static, F::Output> {
    match host_context() {
        Some(context) => in_host_context(context, fut),
        None => Box::pin(fut),
    }
}

/// Never builds a runtime: an import-time thread would not survive the forkserver's `fork()`.
/// A task it spawns inherits the spawning task's host context.
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

    /// Runs `work` on this runtime, wherever the caller polls: tasks it spawns land there too.
    /// Dropping the returned future cancels the work; a panic reaches the caller.
    pub async fn run<T: Send + 'static>(
        &self,
        work: impl Future<Output = T> + Send + 'static,
    ) -> T {
        match lk_agents_abi::SpawnerExt::spawn(self, work).await {
            Ok(value) => value,
            Err(error) => std::panic::resume_unwind(Box::new(error.to_string())),
        }
    }
}

impl Spawner for TokioSpawner {
    fn launch(&self, task: Launch) -> OwnedTask<Box<dyn Any + Send>> {
        let (run, owned) = launch(task);
        #[allow(
            clippy::disallowed_methods,
            reason = "the Spawner itself: the returned OwnedTask owns the work"
        )]
        drop(self.handle.spawn(inherit(run)));
        owned
    }
}

/// Each task on a thread of its own, for work that blocks between its awaits (a SQLite worker):
/// the thread parks while the task waits and ends with it.
#[derive(Clone, Copy, Debug)]
pub struct DedicatedThreads;

impl Spawner for DedicatedThreads {
    fn launch(&self, task: Launch) -> OwnedTask<Box<dyn Any + Send>> {
        struct Unpark(std::thread::Thread);
        impl std::task::Wake for Unpark {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
        }
        let (run, owned) = launch(task);
        let thread = std::thread::Builder::new().name("lk-agents-dedicated".into()).spawn(|| {
            let waker = std::task::Waker::from(Arc::new(Unpark(std::thread::current())));
            let mut cx = std::task::Context::from_waker(&waker);
            let mut run = run;
            while run.as_mut().poll(&mut cx).is_pending() {
                std::thread::park();
            }
        });
        // A thread refused is a task dropped unrun: its owner sees it cancelled.
        drop(thread);
        owned
    }
}

/// Named [`OwnedTask`]s and a way to wait for them to be gone. Each task
/// holds the set until it ends, so dropping the owner's handle cancels nothing: `close`,
/// `cancel_all` and `drain` do. A task is never addressed by id: work the set joins is spawned
/// with `spawn`, work an owner may cancel early with `scoped`, whose handle cancels on drop.
pub struct TaskSet {
    inner: Mutex<TaskEntries>,
    /// Bumped as a task leaves, so a wait for the set to empty re-checks it.
    left: signal::Producer<u64>,
}

impl Default for TaskSet {
    fn default() -> Self {
        TaskSet { inner: Mutex::default(), left: signal::signal(0).0 }
    }
}

#[derive(Default)]
struct TaskEntries {
    closed: bool,
    next: u64,
    tasks: BTreeMap<u64, Entry>,
}

impl TaskEntries {
    fn admit(&mut self, name: &str, cleanup: bool) -> u64 {
        self.next += 1;
        self.tasks.insert(self.next, Entry { name: name.to_owned(), task: None, cleanup });
        self.next
    }
}

struct Entry {
    name: String,
    task: Option<OwnedTask>,
    /// A bounded teardown: admitted after `close` and never cancelled by the set.
    cleanup: bool,
}

/// Owned by the work: leaves the set when the work is dropped, however it ended. Also handed
/// out by [`TaskSet::track`] for work another set runs.
pub struct Leave {
    table: Arc<TaskSet>,
    id: u64,
}

impl Drop for Leave {
    fn drop(&mut self) {
        let left = self.table.inner.lock().tasks.remove(&self.id);
        drop(left);
        self.table.left.modify(|n| *n += 1);
    }
}

/// A task of a [`TaskSet`] its owner may end early: dropping the handle requests its
/// cancellation, and the set still joins the task until it settled.
#[must_use = "dropping a ScopedTask cancels it"]
pub struct ScopedTask {
    /// `None` once released to the set.
    table: Option<Arc<TaskSet>>,
    id: u64,
}

impl ScopedTask {
    /// Requests cancellation now; the set joins the task until it settled.
    pub fn cancel(&self) {
        if let Some(table) = &self.table {
            table.cancel(self.id);
        }
    }

    /// Leaves the task to its set, which still joins and may cancel it: for work that ends on
    /// its own once its owner is gone (a filler that closes its member on stop).
    pub fn release(mut self) {
        self.table = None;
    }
}

impl Drop for ScopedTask {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl std::fmt::Debug for ScopedTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ScopedTask").field(&self.id).finish()
    }
}

/// Scoped tasks of one set its owner ends together: dropping it cancels them all. A push first
/// forgets the tasks that ended, so it holds only running ones.
#[must_use = "dropping ScopedTasks cancels them"]
pub struct ScopedTasks {
    table: Arc<TaskSet>,
    ids: Vec<u64>,
}

impl ScopedTasks {
    pub fn new(table: Arc<TaskSet>) -> Self {
        ScopedTasks { table, ids: Vec::new() }
    }

    /// Adds `task`, a task of this set.
    pub fn push(&mut self, mut task: ScopedTask) {
        let table = task.table.take().expect("a scoped task is pushed once");
        debug_assert!(Arc::ptr_eq(&table, &self.table), "a task of another set");
        self.prune();
        self.ids.push(task.id);
    }

    /// Forgets the tasks that ended.
    fn prune(&mut self) {
        let entries = self.table.inner.lock();
        self.ids.retain(|id| entries.tasks.contains_key(id));
    }

    /// Whether every task ended.
    pub fn is_empty(&mut self) -> bool {
        self.prune();
        self.ids.is_empty()
    }

    /// Cancels every task; the set joins them until they settled.
    pub fn clear(&mut self) {
        let entries = self.table.inner.lock();
        for id in self.ids.drain(..) {
            if let Some(Entry { task: Some(task), .. }) = entries.tasks.get(&id) {
                task.cancel();
            }
        }
    }
}

impl Drop for ScopedTasks {
    fn drop(&mut self) {
        self.clear();
    }
}

impl TaskSet {
    /// Counts work run elsewhere (another set's task) until the guard drops, so this set's
    /// waits include it; it cannot be cancelled from here.
    pub fn track(self: &Arc<Self>, name: &str) -> Leave {
        let id = self.inner.lock().admit(name, false);
        Leave { table: Arc::clone(self), id }
    }

    /// Work the set owns until it ends or the set cancels it.
    #[track_caller]
    pub fn spawn(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) {
        assert!(self.try_spawn(spawner, name, fut), "task admission is closed");
    }

    /// Admit a task unless its owner has permanently closed this set.
    #[track_caller]
    pub fn try_spawn(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) -> bool {
        self.launch(spawner, name, false, fut).is_some()
    }

    /// [`spawn`](Self::spawn), with a handle that cancels the task when dropped.
    #[track_caller]
    pub fn scoped(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) -> ScopedTask {
        self.try_scoped(spawner, name, fut).expect("task admission is closed")
    }

    /// [`try_spawn`](Self::try_spawn), with a handle that cancels the task when dropped.
    #[track_caller]
    pub fn try_scoped(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) -> Option<ScopedTask> {
        let id = self.launch(spawner, name, false, fut)?;
        Some(ScopedTask { table: Some(Arc::clone(self)), id })
    }

    /// A teardown bounded by its own deadline (a resource's close): admitted even after `close`
    /// and never cancelled by the set, whose waits join it.
    #[track_caller]
    pub fn cleanup(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) {
        self.launch(spawner, name, true, fut);
    }

    /// Work whose result its waiters await on `producer`'s cell: the producer moves into the
    /// task, so a cancel, a panic or a dropped runtime ends the signal and every waiter resolves.
    /// `false`, with the signal ended, once the set closed.
    #[track_caller]
    pub fn spawn_settling<T: Send + Sync + 'static>(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        producer: signal::Producer<Option<T>>,
        fut: impl Future<Output = T> + Send + 'static,
    ) -> bool {
        self.try_spawn(spawner, name, async move { producer.resolve(fut.await) })
    }

    /// A scoped task feeding a machine: if the work panics, `lost` reports its death as the
    /// machine's input, so the machine never waits on it. A cancel is its owner's, who knows.
    #[track_caller]
    pub fn scoped_feeding(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
        lost: impl FnOnce() + Send + 'static,
    ) -> ScopedTask {
        let settled: Settle = Box::new(move |outcome| {
            if outcome == TaskOutcome::Panicked {
                lost();
            }
            Box::pin(async {})
        });
        let id = self.launch_with(spawner, name, false, fut, Some(settled));
        ScopedTask { table: Some(Arc::clone(self)), id: id.expect("task admission is closed") }
    }

    #[track_caller]
    fn launch(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        cleanup: bool,
        fut: impl Future<Output = ()> + Send + 'static,
    ) -> Option<u64> {
        self.launch_with(spawner, name, cleanup, fut, None)
    }

    #[track_caller]
    fn launch_with(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        cleanup: bool,
        fut: impl Future<Output = ()> + Send + 'static,
        settled: Option<Settle>,
    ) -> Option<u64> {
        let id = {
            let mut g = self.inner.lock();
            if g.closed && !cleanup {
                return None;
            }
            g.admit(name, cleanup)
        };
        let leave = Leave { table: Arc::clone(self), id };
        let work = async move {
            let _leave = leave;
            fut.await;
        };
        let task = match settled {
            Some(settled) => spawner.spawn_settled(work, settled),
            None => spawner.spawn(work),
        };
        let mut g = self.inner.lock();
        if g.closed && !cleanup {
            task.cancel();
        }
        // Absent when the work already ended; the task is then dropped finished.
        if let Some(entry) = g.tasks.get_mut(&id) {
            entry.task = Some(task);
        }
        Some(id)
    }

    fn cancel(&self, id: u64) {
        if let Some(Entry { task: Some(task), .. }) = self.inner.lock().tasks.get(&id) {
            task.cancel();
        }
    }

    pub fn names(&self) -> Vec<String> {
        self.inner.lock().tasks.values().map(|entry| entry.name.clone()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().tasks.is_empty()
    }

    /// `true` when the table emptied.
    pub async fn wait_empty(&self, clock: &dyn Clock, deadline: Option<Ts>) -> bool {
        // Subscribed before the check, so a task leaving after it wakes the wait.
        let mut left = self.left.cell();
        loop {
            if self.is_empty() {
                return true;
            }
            let Some(at) = deadline else {
                // The set is borrowed for the wait, so its producer outlives it.
                let _ = left.changed().await;
                continue;
            };
            tokio::select! {
                biased;
                _ = left.changed() => {}
                _ = clock.wait_until(at) => return self.is_empty(),
            }
        }
    }

    /// Permanently refuse new work and request cancellation of this set's admitted tasks.
    /// Returns without waiting for their cleanup; `wait_empty` observes settlement.
    pub fn close(&self) {
        self.inner.lock().closed = true;
        self.cancel_all();
    }

    /// Request cancellation of every task currently owned by this set but its cleanups, without
    /// waiting. This does not close admission or imply that tasks and their cleanup have finished.
    pub fn cancel_all(&self) {
        let tasks = self.inner.lock();
        for entry in tasks.tasks.values().filter(|entry| !entry.cleanup) {
            if let Some(task) = &entry.task {
                task.cancel();
            }
        }
    }

    /// Request cancellation and wait for this set to empty; returns the task names captured
    /// before cancellation. Call `close` first when new task admission must also stop.
    pub async fn cancel_and_wait(&self, clock: &dyn Clock) -> Vec<String> {
        let names = self.names();
        self.cancel_all();
        self.wait_empty(clock, None).await;
        names
    }

    /// Waits until `deadline`, then cancels the stragglers and names them.
    pub async fn drain(&self, clock: &dyn Clock, deadline: Ts) -> Vec<String> {
        if self.wait_empty(clock, Some(deadline)).await {
            return Vec::new();
        }
        self.cancel_and_wait(clock).await
    }
}

/// Explicit cancellation of an owned host operation, acknowledged by its normal result.
#[derive(Default)]
pub struct Cancellation(CancellationToken);
impl Cancellation {
    pub fn cancel(&self) {
        self.0.cancel();
    }
}
impl lk_agents_abi::invoke::Cancellation for Cancellation {
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
    fn cancelled(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(self.0.cancelled())
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
        assert!(!tasks.try_spawn(&spawner, "late", async {}));
        assert!(tasks.wait_empty(&crate::clock::RealClock::new(), None).await);
    }

    // Catches: a replaced or forgotten handle leaving its task running, or a dropped handle no
    // longer joined by its set.
    #[tokio::test]
    async fn dropping_a_scoped_handle_cancels_its_task_and_the_set_still_joins_it() {
        let tasks = Arc::new(TaskSet::default());
        let spawner = TokioSpawner::current();
        let (tx, rx) = oneshot::channel::<()>();
        let mut handle = Some(tasks.scoped(&spawner, "first", async move {
            let _tx = tx;
            std::future::pending::<()>().await
        }));
        assert!(handle.is_some() && !tasks.is_empty());
        // Overwriting the slot drops the previous handle, which cancels its task.
        handle = Some(tasks.scoped(&spawner, "second", std::future::pending()));
        assert!(rx.await.is_err(), "the replaced task was cancelled");
        drop(handle);
        assert!(tasks.wait_empty(&crate::clock::RealClock::new(), None).await);
    }

    // Catches: a collection of handles growing with every ended task, or one dropped without
    // cancelling the tasks still running.
    #[tokio::test]
    async fn scoped_tasks_forget_ended_tasks_and_cancel_the_rest() {
        let tasks = Arc::new(TaskSet::default());
        let spawner = TokioSpawner::current();
        let mut scoped = ScopedTasks::new(tasks.clone());
        let (done, ended) = oneshot::channel::<()>();
        scoped.push(tasks.scoped(&spawner, "ends", async move { drop(done) }));
        let _ = ended.await;
        assert!(tasks.wait_empty(&crate::clock::RealClock::new(), None).await);
        let (tx, rx) = oneshot::channel::<()>();
        scoped.push(tasks.scoped(&spawner, "runs", async move {
            let _tx = tx;
            std::future::pending::<()>().await
        }));
        assert_eq!(scoped.ids.len(), 1, "the ended task was forgotten");
        drop(scoped);
        assert!(rx.await.is_err(), "the running task was cancelled");
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

    // Catches: a task feeding a machine dying by panic without the machine hearing of it, or a
    // cancel by its owner reported as lost.
    #[tokio::test]
    async fn a_feeding_task_reports_its_panic_and_not_its_cancel() {
        let tasks = Arc::new(TaskSet::default());
        let spawner = TokioSpawner::current();
        let (lost, heard) = oneshot::channel();
        let _panics = tasks.scoped_feeding(&spawner, "panics", async { panic!("boom") }, || {
            let _ = lost.send(());
        });
        assert!(heard.await.is_ok());
        let (lost, mut heard) = oneshot::channel::<()>();
        let cancelled = tasks.scoped_feeding(&spawner, "runs", std::future::pending(), move || {
            let _ = lost.send(());
        });
        drop(cancelled);
        assert!(tasks.wait_empty(&crate::clock::RealClock::new(), None).await);
        assert!(heard.try_recv().is_err(), "a cancel is not a loss");
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
