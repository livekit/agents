//! Supplied-clock RTC peers (the SDK-port fakes) and probe ports for native and deterministic
//! tests.

pub mod corpus;
pub mod playback;
pub mod recorder;
pub mod rtc;
pub use rtc as test_support_rtc;

pub mod text;

/// The test's Tokio runtime as the injected spawner, for crates with no net edge (the process
/// executor is `lk_agents_net::task::TokioSpawner`).
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

impl lk_agents_abi::Spawner for TokioSpawner {
    fn execute(&self, task: lk_agents_abi::Spawned) {
        drop(self.0.spawn(task));
    }
}
