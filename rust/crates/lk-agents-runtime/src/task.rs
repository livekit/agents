//! Owned runtime tasks: every spawned task has an owner holding its [`OwnedTask`], a correlated
//! cancellation request and one observed terminal outcome, reported only after the work and
//! everything it owned were dropped. Dropping the handle requests cancellation; it never detaches.

use std::panic::AssertUnwindSafe;

use futures_util::FutureExt;
use tokio::sync::oneshot;

/// Which way a task ended; the first of completion and cancellation to happen wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOutcome {
    Completed,
    Cancelled,
    Panicked,
}

#[derive(Debug)]
pub struct OwnedTask {
    cancel: Option<oneshot::Sender<()>>,
}

impl OwnedTask {
    /// Runs `work` on `runtime`, then `settled` with the outcome once `work` is dropped.
    pub fn spawn<W, S, F>(runtime: &tokio::runtime::Handle, work: W, settled: S) -> Self
    where
        W: Future<Output = ()> + Send + 'static,
        S: FnOnce(TaskOutcome) -> F + Send + 'static,
        F: Future<Output = ()> + Send + 'static,
    {
        let (cancel, cancelled) = oneshot::channel::<()>();
        runtime.spawn(async move {
            let outcome = {
                let work = AssertUnwindSafe(work).catch_unwind();
                tokio::select! {
                    biased;
                    // An `Err` is the handle dropped: cancellation as well.
                    _ = cancelled => TaskOutcome::Cancelled,
                    done = work => match done {
                        Ok(()) => TaskOutcome::Completed,
                        Err(_) => TaskOutcome::Panicked,
                    },
                }
            };
            settled(outcome).await;
        });
        OwnedTask { cancel: Some(cancel) }
    }

    /// Requests cancellation; idempotent. Completion is reported by the task, not implied here.
    pub fn cancel(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
    }
}

impl Drop for OwnedTask {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    async fn outcome_of(work: impl Future<Output = ()> + Send + 'static, cancel: bool) -> (TaskOutcome, bool) {
        let (tx, rx) = oneshot::channel();
        let (dropped_tx, dropped_rx) = std::sync::mpsc::channel::<()>();
        struct Owned(std::sync::mpsc::Sender<()>);
        impl Drop for Owned {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let owned = Owned(dropped_tx);
        let mut task = OwnedTask::spawn(
            &tokio::runtime::Handle::current(),
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
        rx.await.unwrap()
    }

    // Catches: an outcome reported while the work still holds its resources, a repeated cancel
    // misbehaving, or a panic escaping as a lost task.
    #[tokio::test]
    async fn each_task_reports_one_outcome_after_its_work_is_dropped() {
        assert_eq!(outcome_of(async {}, false).await, (TaskOutcome::Completed, true));
        let blocked = std::future::pending::<()>();
        assert_eq!(outcome_of(blocked, true).await, (TaskOutcome::Cancelled, true));
        assert_eq!(outcome_of(async { panic!("boom") }, false).await, (TaskOutcome::Panicked, true));
    }

    // Catches: a dropped handle detaching its task instead of cancelling it.
    #[tokio::test]
    async fn dropping_the_handle_cancels_the_task() {
        let (tx, rx) = oneshot::channel();
        drop(OwnedTask::spawn(
            &tokio::runtime::Handle::current(),
            tokio::time::sleep(Duration::from_secs(3600)),
            move |outcome| async move {
                let _ = tx.send(outcome);
            },
        ));
        assert_eq!(rx.await.unwrap(), TaskOutcome::Cancelled);
    }
}
