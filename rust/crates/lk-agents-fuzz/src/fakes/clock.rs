use lk_agents_abi::{BoxFuture, Clock, Ts};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// A host clock whose wall time can be corrected without changing elapsed audio time.
pub(crate) struct SkewClock {
    inner: Arc<dyn Clock>,
    offset: Mutex<(bool, Duration)>,
}

impl SkewClock {
    pub(crate) fn new(inner: Arc<dyn Clock>) -> Self {
        Self { inner, offset: Mutex::new((true, Duration::ZERO)) }
    }

    pub(crate) fn shift(&self, forward: bool, by: Duration) {
        *self.offset.lock().unwrap() = (forward, by);
    }
}

impl Clock for SkewClock {
    fn now(&self) -> Ts {
        self.inner.now()
    }

    fn wall(&self) -> SystemTime {
        let wall = self.inner.wall();
        let (forward, by) = *self.offset.lock().unwrap();
        if forward { wall + by } else { wall.checked_sub(by).unwrap() }
    }

    fn wait_until(&self, at: Ts) -> BoxFuture<'_, ()> {
        self.inner.wait_until(at)
    }
}
