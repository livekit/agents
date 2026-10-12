//! Native inference clients are built before any session, so the session binds its clock inside
//! this one; otherwise a virtual-clock test would time out in wall seconds.

use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use async_trait::async_trait;

use super::clock::{Clock, RealClock, Ts};

/// [`RealClock`] at construction; [`bind`](Self::bind) replaces it. A wait already parked finishes
/// on the previous clock.
pub struct BoundClock {
    inner: RwLock<Arc<dyn Clock>>,
}

impl BoundClock {
    pub fn new() -> Arc<Self> {
        Arc::new(BoundClock { inner: RwLock::new(Arc::new(RealClock::new())) })
    }

    pub fn bind(&self, clock: Arc<dyn Clock>) {
        *self.inner.write().unwrap_or_else(|p| p.into_inner()) = clock;
    }

    fn current(&self) -> Arc<dyn Clock> {
        Arc::clone(&self.inner.read().unwrap_or_else(|p| p.into_inner()))
    }
}

#[async_trait]
impl Clock for BoundClock {
    fn now(&self) -> Ts {
        self.current().now()
    }

    async fn wait_until(&self, at: Ts) {
        self.current().wait_until(at).await
    }

    fn wall(&self) -> SystemTime {
        self.current().wall()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualClock;
    use std::time::Duration;

    // Catches: a `BoundClock` that snapshots its construction clock, ignoring a later bind.
    #[tokio::test]
    async fn a_bound_clock_reads_the_clock_bound_last() {
        let bound = BoundClock::new();
        let virt = Arc::new(VirtualClock::new());
        bound.bind(Arc::clone(&virt) as Arc<dyn Clock>);
        assert_eq!(bound.now(), virt.now());
        virt.advance(Duration::from_secs(3));
        assert_eq!(bound.now(), virt.now(), "reads go to the bound clock, not a copy");

        let wait = bound.wait_until(virt.now().saturating_add(Duration::from_secs(1)));
        let mut wait = Box::pin(wait);
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        assert!(wait.as_mut().poll(&mut cx).is_pending(), "a virtual clock parks until moved");
        virt.advance(Duration::from_secs(1));
        assert!(wait.as_mut().poll(&mut cx).is_ready(), "and returns once time has moved");
    }
}
