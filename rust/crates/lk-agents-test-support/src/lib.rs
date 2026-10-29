//! Supplied-clock RTC peers (the SDK-port fakes) and probe ports for native and deterministic
//! tests.

pub mod playback;
pub mod recorder;
pub mod rtc;
pub use rtc as test_support_rtc;

pub mod text;

/// The test's Tokio runtime as the injected spawner, for crates with no net edge (the process
/// executor is `lk_agents_net::task::TokioSpawner`). A cancel aborts the whole task, so a settle
/// step runs only on completion: enough for the sans-I/O tests, which use none.
#[derive(Clone, Debug)]
pub struct TokioSpawner(tokio::runtime::Handle);

impl TokioSpawner {
    pub fn new(handle: tokio::runtime::Handle) -> Self {
        TokioSpawner(handle)
    }

    pub fn current() -> Self {
        TokioSpawner(tokio::runtime::Handle::current())
    }
}

struct Aborting(tokio::task::AbortHandle);

impl lk_agents_abi::TaskControl for Aborting {
    fn cancel(&self) {
        self.0.abort();
    }

    fn is_finished(&self) -> bool {
        self.0.is_finished()
    }
}

impl lk_agents_abi::Spawner for TokioSpawner {
    fn launch(
        &self,
        task: lk_agents_abi::Launch,
    ) -> lk_agents_abi::OwnedTask<Box<dyn std::any::Any + Send>> {
        use lk_agents_abi::{TaskError, TaskOutcome};
        let lk_agents_abi::Launch { work, settled, .. } = task;
        let join = self.0.spawn(async move {
            let value = work.await;
            if let Some(settled) = settled {
                settled(TaskOutcome::Completed).await;
            }
            value
        });
        let control = std::sync::Arc::new(Aborting(join.abort_handle()));
        lk_agents_abi::OwnedTask::new(
            control,
            Box::pin(async move {
                join.await.map_err(|error| match error.try_into_panic() {
                    Ok(_) => TaskError::Panicked("panicked".to_owned()),
                    Err(_) => TaskError::Cancelled,
                })
            }),
        )
    }
}
