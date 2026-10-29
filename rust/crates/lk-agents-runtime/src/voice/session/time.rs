//! The session clock.

use lk_agents_abi::{Clock, Elapsed, Ts, VirtualClock};
use lk_agents_voice::agent_session::Now;
use std::sync::Arc;
use std::time::Duration;

/// A clock plus the only legal way to wait on it. Cheap to clone; `FakeRtc` shares the same
/// [`Clock`] so its playout model and the driver's timers cannot drift.
#[derive(Clone)]
pub struct Timebase {
    clock: Arc<dyn Clock>,
    /// `Some` only for a virtual clock: moving time is a test capability, not on [`Clock`].
    virt: Option<Arc<VirtualClock>>,
}

impl Timebase {
    pub fn of_clock(clock: Arc<dyn Clock>) -> Self {
        Timebase { clock, virt: None }
    }

    /// Shares an existing clock so the playout model and the driver agree.
    pub fn from_virtual(clock: Arc<VirtualClock>) -> Self {
        Timebase { clock: clock.clone(), virt: Some(clock) }
    }

    pub fn virtual_clock(&self) -> Option<Arc<VirtualClock>> {
        self.virt.clone()
    }

    pub fn clock(&self) -> Arc<dyn Clock> {
        self.clock.clone()
    }

    pub fn now(&self) -> Ts {
        self.clock.now()
    }

    pub fn wall(&self) -> std::time::SystemTime {
        self.clock.wall()
    }

    pub fn wall_secs(&self) -> f64 {
        lk_agents_abi::unix_secs(self.clock.wall())
    }

    /// Both clocks, read together: the instant a step is handled at.
    pub fn read(&self) -> Now {
        Now { at: self.now(), wall: self.wall() }
    }

    /// Panics on a real clock.
    pub fn advance(&self, by: Duration) {
        let clock = self.virt.as_ref().expect("Timebase::advance requires a virtual clock");
        clock.advance(by);
    }

    /// Every bounded wait goes through here, never `tokio::time::timeout`, which a virtual clock
    /// does not move.
    pub async fn timeout<F: std::future::Future>(
        &self,
        limit: Duration,
        fut: F,
    ) -> Result<F::Output, Elapsed> {
        lk_agents_abi::timeout(&*self.clock, limit, fut).await
    }

    pub fn is_virtual(&self) -> bool {
        self.virt.is_some()
    }

    /// On a virtual clock only moving time releases it.
    pub async fn wait_until(&self, at: Ts) {
        self.clock.wait_until(at).await
    }

    pub async fn wait_until_opt(&self, at: Option<Ts>) {
        match at {
            Some(at) => self.wait_until(at).await,
            None => std::future::pending::<()>().await,
        }
    }
}

impl std::fmt::Debug for Timebase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timebase")
            .field("now", &self.clock.now())
            .field("virtual", &self.virt.is_some())
            .finish()
    }
}
