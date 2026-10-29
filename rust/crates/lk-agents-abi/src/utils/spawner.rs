use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

use crate::{Clock, Ts};

use tokio::runtime::Handle;

/// Defined here to keep dependencies to `{tokio, async-trait}`.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What a [`Spawner`] hands back: stops the task it spawned. Not Tokio's own, because a task may
/// run on another executor (a virtual-time host loop, `python::loop_time`).
#[derive(Clone)]
pub struct AbortHandle(std::sync::Arc<dyn Abort>);

/// An executor's side of an [`AbortHandle`].
pub trait Abort: Send + Sync + 'static {
    /// Drops the task's future soon; a finished task ignores it.
    fn abort(&self);
    /// Completed or aborted.
    fn is_finished(&self) -> bool;
}

impl AbortHandle {
    pub fn new(inner: impl Abort) -> Self {
        AbortHandle(std::sync::Arc::new(inner))
    }

    pub fn abort(&self) {
        self.0.abort()
    }

    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }
}

impl std::fmt::Debug for AbortHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AbortHandle").field("finished", &self.is_finished()).finish()
    }
}

impl Abort for tokio::task::AbortHandle {
    fn abort(&self) {
        tokio::task::AbortHandle::abort(self)
    }

    fn is_finished(&self) -> bool {
        tokio::task::AbortHandle::is_finished(self)
    }
}

impl From<tokio::task::AbortHandle> for AbortHandle {
    fn from(handle: tokio::task::AbortHandle) -> Self {
        AbortHandle::new(handle)
    }
}

/// `tokio::spawn` panics on a host thread with no reactor, so the runtime handle is injected.
/// Returns an [`AbortHandle`], not a `JoinHandle`, which would detach on drop.
///
/// `#[track_caller]`, so an executor can name the line that spawned a task it reports as stuck.
pub trait Spawner: Send + Sync {
    #[track_caller]
    fn spawn(&self, f: BoxFuture<'static, ()>) -> AbortHandle;
}

/// Blanket-implemented, including for `dyn Spawner`.
pub trait SpawnerExt: Spawner {
    #[track_caller]
    fn spawn_unboxed<F>(&self, f: F) -> AbortHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.spawn(Box::pin(f))
    }
}

impl<S: Spawner + ?Sized> SpawnerExt for S {}

/// Lets the executor's other tasks run before continuing: one `Pending` that wakes itself.
/// Unlike `tokio::task::yield_now`, it assumes no Tokio runtime, so a deterministic scheduler
/// runs it as it is.
pub async fn yield_now() {
    let mut yielded = false;
    std::future::poll_fn(|cx| {
        if std::mem::replace(&mut yielded, true) {
            return std::task::Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    })
    .await
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
    #[allow(clippy::disallowed_methods, reason = "S0: runtime migration")]
    fn spawn(&self, f: BoxFuture<'static, ()>) -> AbortHandle {
        self.handle.spawn(f).abort_handle().into()
    }
}

/// Named tasks and a way to wait for them to be gone (`utils.aio.TaskSet`). Entries are inserted
/// before spawning and removed by a guard the task owns.
#[derive(Default)]
pub struct TaskSet {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    inner: Mutex<TaskEntries>,
    changed: Notify,
}

#[derive(Default)]
struct TaskEntries {
    closed: bool,
    next: u64,
    tasks: BTreeMap<u64, (String, Option<AbortHandle>)>,
}

struct Leave {
    table: Arc<TaskSet>,
    id: u64,
}

impl Drop for Leave {
    fn drop(&mut self) {
        self.table.inner.lock().expect("task table").tasks.remove(&self.id);
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
    ) -> AbortHandle {
        self.try_spawn(spawner, name, fut).expect("task admission is closed")
    }

    /// Admit a task unless its owner has permanently closed this set.
    #[track_caller]
    pub fn try_spawn(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) -> Option<AbortHandle> {
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
        let handle = spawner.spawn_unboxed(async move {
            let _leave = leave;
            fut.await;
        });
        let closed = {
            let mut g = self.inner.lock().expect("task table");
            if let Some(entry) = g.tasks.get_mut(&id) {
                entry.1 = Some(handle.clone());
            }
            g.closed
        };
        if closed {
            handle.abort();
        }
        Some(handle)
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

    /// Permanently refuse new work and abort admitted tasks without holding the table lock.
    pub fn close(&self) {
        self.inner.lock().expect("task table").closed = true;
        self.abort_all();
    }

    /// Abort every task without waiting: an owner going away.
    pub fn abort_all(&self) {
        let handles: Vec<_> = self
            .inner
            .lock()
            .expect("task table")
            .tasks
            .values()
            .filter_map(|(_, handle)| handle.clone())
            .collect();
        for handle in handles {
            handle.abort();
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
    use std::sync::Arc;
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// A runtime on its own thread, standing in for the driver's.
    struct Background {
        handle: Handle,
        stop: Option<tokio::sync::oneshot::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Background {
        #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
        fn start() -> Self {
            let (ready_tx, ready_rx) = std::sync::mpsc::channel();
            let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
            // Test-only thread: the calling thread must genuinely have no runtime.
            let thread = std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                    .expect("build the background runtime");
                ready_tx.send(rt.handle().clone()).expect("hand the handle back");
                rt.block_on(async {
                    let _ = stop_rx.await;
                });
            });
            Background {
                handle: ready_rx.recv().expect("the runtime thread started"),
                stop: Some(stop_tx),
                thread: Some(thread),
            }
        }
    }

    impl Drop for Background {
        fn drop(&mut self) {
            let _ = self.stop.take().expect("stop sender").send(());
            let _ = self.thread.take().expect("thread").join();
        }
    }

    // Catches: bare `tokio::spawn` on a host-reachable path (the calling thread has no reactor,
    // asserted first).
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[test]
    fn spawning_works_from_a_thread_with_no_runtime() {
        let rt = Background::start();
        let spawner: Arc<dyn Spawner> = Arc::new(TokioSpawner::new(rt.handle.clone()));

        assert!(
            Handle::try_current().is_err(),
            "this thread must have no reactor, or the test proves nothing"
        );

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        spawner.spawn_unboxed(async move {
            done_tx.send("ran").expect("the test is still waiting");
        });

        assert_eq!(
            done_rx.recv_timeout(Duration::from_secs(2)),
            Ok("ran"),
            "the injected handle did not run the task"
        );
    }

    // Catches: handing back something that cannot stop the task (a `JoinHandle`, or a unit).
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    #[tokio::test]
    async fn the_returned_handle_aborts_what_it_spawned() {
        /// Set when the spawned future is dropped, which is what an abort does.
        #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
        struct Cancelled(Arc<AtomicBool>);
        impl Drop for Cancelled {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let spawner = TokioSpawner::current();
        let cancelled = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));

        let handle = {
            let marker = Cancelled(Arc::clone(&cancelled));
            let finished = Arc::clone(&finished);
            spawner.spawn_unboxed(async move {
                let _marker = marker;
                // Never resolves, like a pump waiting on the next chunk from a provider.
                std::future::pending::<()>().await;
                finished.store(true, Ordering::SeqCst);
            })
        };

        tokio::task::yield_now().await; // let it start
        assert!(!cancelled.load(Ordering::SeqCst), "it must be running before the abort");
        handle.abort();

        for _ in 0..100 {
            if cancelled.load(Ordering::SeqCst) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(cancelled.load(Ordering::SeqCst), "abort() did not stop the task it was handed");
        assert!(!finished.load(Ordering::SeqCst), "the task ran past the abort point");
    }
    #[tokio::test]
    async fn closing_task_admission_refuses_future_work_and_settles_admitted_work() {
        let tasks = Arc::new(TaskSet::default());
        let spawner = TokioSpawner::current();
        tasks.spawn(&spawner, "pending", std::future::pending());
        tasks.close();
        assert!(tasks.try_spawn(&spawner, "late", async {}).is_none());
        assert!(tasks.wait_empty(&crate::RealClock::new(), None).await);
    }
}
