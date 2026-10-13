//! The session clock.

use lk_agents_abi::{Clock, Elapsed, RealClock, Ts, VirtualClock};
use std::sync::Arc;
use std::time::Duration;

/// A clock plus the only legal way to wait on it. Cheap to clone; `FakeRtc` shares the same
/// [`Clock`] so its playout model and the driver's timers cannot drift.
///
/// As `_core.SessionClock` it is the handle a session hands the native clients a host provider is
/// built on (`obj._bind_session_clock(clock)`): a session on a virtual clock times a stalled gateway
/// out in its own seconds, not wall ones.
#[lk_agents_abi::class(name = "SessionClock")]
#[derive(Clone)]
pub struct Timebase {
    clock: Arc<dyn Clock>,
    /// `Some` only for a virtual clock: moving time is a test capability, not on [`Clock`].
    virt: Option<Arc<VirtualClock>>,
}

#[lk_agents_abi::host(object(name = "SessionClock", pid = false, as_dyn(Clock = clock)))]
impl Timebase {
    #[host(method(name = "__repr__"))]
    pub fn host_repr(&self) -> String {
        format!("SessionClock(now={:?})", self.clock.now())
    }
}

impl Timebase {
    pub fn of_clock(clock: Arc<dyn Clock>) -> Self {
        Timebase { clock, virt: None }
    }

    pub fn real() -> Self {
        Timebase { clock: Arc::new(RealClock::new()), virt: None }
    }

    /// Tokio's clock for scheduling, a synthetic epoch for wall stamps (so saved cases replay
    /// exactly).
    pub fn simulated() -> Self {
        Timebase { clock: Arc::new(lk_agents_abi::SimClock::new()), virt: None }
    }

    /// Shares an existing clock so the playout model and the driver agree.
    pub fn from_virtual(clock: Arc<VirtualClock>) -> Self {
        Timebase { clock: clock.clone(), virt: Some(clock) }
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
        self.clock
            .wall()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// The wall seconds a session instant fell (or will fall) at.
    pub fn wall_secs_at(&self, at: Ts) -> f64 {
        let now = self.now();
        self.wall_secs() - now.saturating_duration_since(at).as_secs_f64()
            + at.saturating_duration_since(now).as_secs_f64()
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

    /// For a host that stopped moving time (awaiting `aclose()`). Paced on real time (1 ms per
    /// step) so host tasks can run between steps. Panics on a real clock.
    pub async fn advance_until(
        &self,
        done: impl std::future::Future<Output = ()>,
        step: Duration,
        budget: u32,
    ) {
        let pacer = RealClock::new();
        let mut done = std::pin::pin!(done);
        let mut steps = 0u32;
        loop {
            let tick = pacer.wait_until(pacer.now().saturating_add(Duration::from_millis(1)));
            tokio::select! {
                biased;
                () = &mut done => break,
                () = tick => {
                    self.advance(step);
                    steps += 1;
                    if steps > budget {
                        break;
                    }
                }
            }
        }
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

// `_core.SessionClock` is the handle `lk_agents_abi::python::bind_host_provider` hands a host
// provider (`_bind_session_clock(clock)`).
#[cfg(feature = "python")]
lk_agents_abi::python::inventory::submit! {
    lk_agents_abi::python::SessionClockClass(|py, clock| {
        Ok(pyo3::Bound::new(py, Timebase::of_clock(clock))?.into_any())
    })
}
