//! `RealtimeModelFallbackAdapter`'s decisions for a host that owns the models and the swap task.

use std::time::Duration;

use lk_agents_abi::Ts;

pub use lk_agents_abi::llm::fallback::{AvailabilityChange, ErrorPlan};


/// `self._available` + `self._cooldown_deadline` (`:189-190`), pure over `now`.
#[derive(Debug, Clone)]
struct AvailabilityTable {
    available: Vec<bool>,
    /// `None`: no cooldown pending (the reference's `0.0` sentinel only works on a Unix clock).
    deadline: Vec<Option<Ts>>,
}

impl AvailabilityTable {
    pub fn new(n: usize) -> Self {
        AvailabilityTable { available: vec![true; n], deadline: vec![None; n] }
    }

    pub fn is_available(&self, index: usize) -> bool {
        self.available.get(index).copied().unwrap_or(false)
    }

    /// `_set_available` (`:243-256`) minus the emit. A no-op write does not re-arm or re-announce.
    pub fn set(&mut self, index: usize, available: bool, now: Ts, cooldown: Duration) -> bool {
        if self.available.get(index).copied() != Some(!available) {
            return false;
        }
        self.available[index] = available;
        self.deadline[index] =
            if available { None } else { Some(now.saturating_add(cooldown)) };
        true
    }

    /// The caller re-enables them through [`set`](AvailabilityTable::set), which emits (`:262-264`).
    pub fn expired(&self, now: Ts) -> Vec<usize> {
        (0..self.available.len())
            .filter(|&i| !self.available[i] && self.deadline[i].is_some_and(|d| d <= now))
            .collect()
    }

    /// `_next_available_index`'s scan (`:266-270`); call [`expired`](AvailabilityTable::expired) first.
    pub fn find(&self, exclude_current: bool, current: usize) -> Option<usize> {
        (0..self.available.len())
            .find(|&i| !(exclude_current && i == current) && self.available[i])
    }
}

/// Per session, as in the reference: two sessions of one adapter may disagree about a model.
pub struct RealtimePolicy {
    table: AvailabilityTable,
    active_index: usize,
    cooldown: Duration,
}

#[lk_agents_abi::host(object(name = "RealtimeFallback", cell = "mutex"))]
impl RealtimePolicy {
    pub fn new(model_count: usize, cooldown: Duration) -> Self {
        RealtimePolicy { table: AvailabilityTable::new(model_count), active_index: 0, cooldown }
    }

    /// Only a count: the models stay in the host list the events index into. `cooldown` in seconds.
    #[host(new)]
    pub fn for_host(model_count: usize, cooldown: f64) -> Result<Self, String> {
        if model_count == 0 {
            return Err("at least one RealtimeModel instance must be provided.".to_owned());
        }
        Ok(RealtimePolicy::new(model_count, Duration::from_secs_f64(cooldown.max(0.0))))
    }

    #[host(getter)]
    pub fn active_index(&self) -> usize {
        self.active_index
    }

    #[host(setter)]
    pub fn set_active_index(&mut self, index: usize) {
        self.active_index = index;
    }

    #[host(method)]
    pub fn is_available(&self, index: usize) -> bool {
        self.table.is_available(index)
    }

    /// `_set_available`. The event to emit, or `None` when nothing changed (no re-armed cooldown).
    #[host(method)]
    pub fn set_available(&mut self, index: usize, available: bool, now: f64) -> Option<AvailabilityChange> {
        let now = ts(now);
        self.table
            .set(index, available, now, self.cooldown)
            .then_some(AvailabilityChange { index, available })
    }

    /// `_next_available_index`: expire cooldowns, then first available, primary preferred. Returns
    /// the target and the sweep's re-enable events.
    #[host(method(defaults(exclude_current = false), kw_only = "exclude_current"))]
    pub fn next_available_index(&mut self, now: f64, exclude_current: bool) -> (Option<usize>, Vec<AvailabilityChange>) {
        let changes = self.sweep(ts(now));
        (self.table.find(exclude_current, self.active_index), changes)
    }

    /// `_on_child_error`. `swap_busy` is the host's "swap task running".
    #[host(method(kw_only))]
    pub fn on_child_error(&mut self, recoverable: bool, swap_busy: bool, now: f64) -> ErrorPlan {
        if recoverable {
            // The plugin's own reconnect owns this one.
            return ErrorPlan { changes: vec![], rewrite_recoverable: false, swap_to: None };
        }

        let active = self.active_index;
        let mut changes = Vec::new();
        if let Some(change) = self.set_available(active, false, now) {
            changes.push(change);
        }
        changes.extend(self.sweep(ts(now)));

        let Some(target) = self.table.find(false, active) else {
            // Exhausted: escalate unchanged so `AgentSession` can close.
            return ErrorPlan { changes, rewrite_recoverable: false, swap_to: None };
        };
        ErrorPlan { changes, rewrite_recoverable: true, swap_to: (!swap_busy).then_some(target) }
    }

    /// `restart(switch_model=)`: the next available model, or the current one when there is no
    /// other. Never `None`.
    #[host(method(kw_only))]
    pub fn restart_target(&mut self, switch_model: bool, now: f64) -> (usize, Vec<AvailabilityChange>) {
        if !switch_model {
            return (self.active_index, vec![]);
        }
        let changes = self.sweep(ts(now));
        let current = self.active_index;
        (self.table.find(true, current).unwrap_or(current), changes)
    }

}

/// Host `time.time()` stamps are only compared with each other, so only monotonicity matters.
fn ts(now: f64) -> Ts {
    Ts::from_nanos((now.max(0.0) * 1e9) as u64)
}

impl RealtimePolicy {
    fn sweep(&mut self, now: Ts) -> Vec<AvailabilityChange> {
        self.table
            .expired(now)
            .into_iter()
            .filter(|&i| self.table.set(i, true, now, self.cooldown))
            .map(|index| AvailabilityChange { index, available: true })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: u64) -> f64 {
        secs as f64
    }

    // Catches: the rewrite missing, or applied when nothing is left to swap to.
    #[test]
    fn an_unrecoverable_error_is_rewritten_only_while_a_fallback_exists() {
        let mut p = RealtimePolicy::new(2, Duration::from_secs(60));
        assert_eq!(
            p.on_child_error(true, false, at(0)),
            ErrorPlan { changes: vec![], rewrite_recoverable: false, swap_to: None },
            "a recoverable error belongs to the plugin's own reconnect"
        );

        let plan = p.on_child_error(false, false, at(1));
        assert_eq!(plan.changes, vec![AvailabilityChange { index: 0, available: false }]);
        assert!(plan.rewrite_recoverable);
        assert_eq!(plan.swap_to, Some(1));

        p.set_active_index(1);
        let plan = p.on_child_error(false, true, at(2));
        assert_eq!(plan.changes, vec![AvailabilityChange { index: 1, available: false }]);
        assert!(!plan.rewrite_recoverable, "every model is down: escalate unchanged");
        assert_eq!(plan.swap_to, None);
    }

    // Catches: swapping while a swap is in flight, or a re-enable the sweep did not report.
    #[test]
    fn the_sweep_reports_each_expired_cooldown_and_a_busy_swap_holds() {
        let mut p = RealtimePolicy::new(2, Duration::from_secs(10));
        assert!(p.set_available(0, false, at(0)).is_some());
        assert!(p.set_available(0, false, at(5)).is_none(), "the no-op guard: no re-arm");
        let (target, changes) = p.next_available_index(at(9), false);
        assert_eq!((target, changes), (Some(1), vec![]));
        let (target, changes) = p.next_available_index(at(10), false);
        assert_eq!(target, Some(0), "the primary is preferred once it is back");
        assert_eq!(changes, vec![AvailabilityChange { index: 0, available: true }]);

        let plan = p.on_child_error(false, true, at(11));
        assert!(plan.rewrite_recoverable);
        assert_eq!(plan.swap_to, None, "a swap already in flight is not doubled");
        assert_eq!(p.restart_target(false, at(11)), (0, vec![]));
        assert_eq!(p.restart_target(true, at(11)).0, 1);
    }
}
