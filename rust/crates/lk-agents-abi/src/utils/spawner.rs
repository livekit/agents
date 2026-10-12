use std::future::Future;
use std::pin::Pin;

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
pub trait Spawner: Send + Sync {
    fn spawn(&self, f: BoxFuture<'static, ()>) -> AbortHandle;
}

/// Blanket-implemented, including for `dyn Spawner`.
pub trait SpawnerExt: Spawner {
    fn spawn_unboxed<F>(&self, f: F) -> AbortHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.spawn(Box::pin(f))
    }
}

impl<S: Spawner + ?Sized> SpawnerExt for S {}

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
    fn spawn(&self, f: BoxFuture<'static, ()>) -> AbortHandle {
        self.handle.spawn(f).abort_handle().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// A runtime on its own thread, standing in for the driver's.
    struct Background {
        handle: Handle,
        stop: Option<tokio::sync::oneshot::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Background {
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
    #[tokio::test]
    async fn the_returned_handle_aborts_what_it_spawned() {
        /// Set when the spawned future is dropped, which is what an abort does.
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
}
