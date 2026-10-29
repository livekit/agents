//! What the decision core knows of time: the instant of the step it handles.

use lk_agents_abi::Ts;
use std::time::Duration;

/// An instant of the session clock and the wall time it fell at, read together by the queue owner.
#[derive(Debug, Clone, Copy)]
pub struct Now {
    pub at: Ts,
    pub wall: std::time::SystemTime,
}

/// What the core knows of time: the instant of the step it is handling, on both clocks.
#[derive(Debug)]
pub(crate) struct StepClock {
    pub(crate) now: Ts,
    wall: std::time::SystemTime,
    /// The last wall stamp handed out. Each is later than the one before: `ChatContext` orders
    /// items by it.
    stamped: std::cell::Cell<std::time::SystemTime>,
}

impl StepClock {
    pub(crate) fn new() -> Self {
        let epoch = std::time::UNIX_EPOCH;
        StepClock { now: Ts::ZERO, wall: epoch, stamped: std::cell::Cell::new(epoch) }
    }

    pub(crate) fn set(&mut self, now: Now) {
        self.now = now.at;
        self.wall = now.wall;
    }

    /// A wall stamp at the step's instant, later than every earlier one.
    pub(crate) fn wall(&self) -> std::time::SystemTime {
        let next = (self.stamped.get() + Duration::from_micros(1)).max(self.wall);
        self.stamped.set(next);
        next
    }

    pub(crate) fn wall_secs(&self) -> f64 {
        lk_agents_abi::unix_secs(self.wall())
    }

    /// The step's instant on both clocks, to stamp from.
    pub(crate) fn anchor(&self) -> lk_agents_abi::WallAnchor {
        lk_agents_abi::WallAnchor { wall: lk_agents_abi::unix_secs(self.wall), at: self.now }
    }

    /// The wall time a session instant fell (or will fall) at.
    pub(crate) fn wall_at(&self, at: Ts) -> std::time::SystemTime {
        let wall = self.wall;
        if at <= self.now {
            wall.checked_sub(self.now.saturating_duration_since(at)).unwrap_or(wall)
        } else {
            wall + at.saturating_duration_since(self.now)
        }
    }

    pub(crate) fn wall_secs_at(&self, at: Ts) -> f64 {
        lk_agents_abi::unix_secs(self.wall_at(at))
    }
}
