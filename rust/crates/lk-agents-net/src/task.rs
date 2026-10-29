//! The Tokio executor behind the injected [`Spawner`]: the process runtime the runtime lends
//! through `ProcessEnv`, or a caller's own. Ownership, cancellation and outcomes are
//! [`lk_agents_abi::OwnedTask`]'s, the same on every executor.

use lk_agents_abi::{Spawned, Spawner};
use tokio::runtime::Handle;

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
    fn execute(&self, task: Spawned) {
        drop(self.handle.spawn(task));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{SpawnerExt, TaskOutcome, TaskSet};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;
    use tokio::sync::oneshot;

    /// A runtime on its own thread, standing in for the driver's.
    struct Background {
        handle: Handle,
        stop: Option<oneshot::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Background {
        fn start() -> Self {
            let (ready_tx, ready_rx) = std::sync::mpsc::channel();
            let (stop_tx, stop_rx) = oneshot::channel();
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
        let _task = spawner.spawn(async move {
            done_tx.send("ran").expect("the test is still waiting");
        });

        assert_eq!(
            done_rx.recv_timeout(Duration::from_secs(2)),
            Ok("ran"),
            "the injected handle did not run the task"
        );
    }

    // Catches: handing back something that cannot stop the task.
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
            spawner.spawn(async move {
                let _marker = marker;
                // Never resolves, like a pump waiting on the next chunk from a provider.
                std::future::pending::<()>().await;
                finished.store(true, Ordering::SeqCst);
            })
        };

        tokio::task::yield_now().await; // let it start
        assert!(!cancelled.load(Ordering::SeqCst), "it must be running before the abort");
        handle.cancel();

        for _ in 0..100 {
            if cancelled.load(Ordering::SeqCst) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(cancelled.load(Ordering::SeqCst), "cancel() did not stop the task it was handed");
        assert!(!finished.load(Ordering::SeqCst), "the task ran past the abort point");
    }

    #[tokio::test]
    async fn closing_task_admission_refuses_future_work_and_settles_admitted_work() {
        let tasks = Arc::new(TaskSet::default());
        let spawner = TokioSpawner::current();
        tasks.spawn(&spawner, "pending", std::future::pending());
        tasks.close();
        assert!(tasks.try_spawn(&spawner, "late", async {}).is_none());
        assert!(tasks.wait_empty(&lk_agents_abi::RealClock::new(), None).await);
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
