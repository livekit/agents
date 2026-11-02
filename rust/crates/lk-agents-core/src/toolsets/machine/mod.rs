//! A shared toolset's leases. IDs and operation outcomes are inputs; there are no host objects.

use super::effects::ToolsetEffect;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Opening,
    Aborting,
    Ready,
    Closing,
}

pub struct Lifetime {
    phase: Phase,
    pending: BTreeSet<u64>,
    owned: BTreeSet<u64>,
    releasing: BTreeSet<u64>,
}
impl Default for Lifetime {
    fn default() -> Self {
        Self {
            phase: Phase::Idle,
            pending: BTreeSet::new(),
            owned: BTreeSet::new(),
            releasing: BTreeSet::new(),
        }
    }
}
impl Lifetime {
    pub fn acquire(&mut self, id: u64) -> Vec<ToolsetEffect> {
        if self.phase == Phase::Ready {
            self.owned.insert(id);
            return vec![ToolsetEffect::Grant(id)];
        }
        self.pending.insert(id);
        self.open()
    }
    fn open(&mut self) -> Vec<ToolsetEffect> {
        if self.phase == Phase::Idle
            && let Some(id) = self.pending.first().copied()
        {
            self.phase = Phase::Opening;
            return vec![ToolsetEffect::Setup(id)];
        }
        Vec::new()
    }
    pub fn release(&mut self, id: u64) -> Vec<ToolsetEffect> {
        if self.pending.remove(&id) {
            if self.pending.is_empty() && self.phase == Phase::Opening {
                self.phase = Phase::Aborting;
                self.releasing.insert(id);
                return vec![ToolsetEffect::CancelSetup];
            }
            return vec![ToolsetEffect::Released(id)];
        }
        if self.owned.remove(&id) {
            if self.owned.is_empty() && self.phase == Phase::Ready {
                self.phase = Phase::Closing;
                self.releasing.insert(id);
                return vec![ToolsetEffect::Close];
            }
            return vec![ToolsetEffect::Released(id)];
        }
        if self.releasing.contains(&id) { Vec::new() } else { vec![ToolsetEffect::Released(id)] }
    }
    /// Called after setup and any rollback of framework-owned children have settled.
    pub fn opened(&mut self, success: bool) -> Vec<ToolsetEffect> {
        let mut effects = Vec::new();
        if self.phase == Phase::Aborting {
            if success {
                self.phase = Phase::Closing;
                return vec![ToolsetEffect::Close];
            }
            self.phase = Phase::Idle;
            effects.extend(
                std::mem::take(&mut self.releasing).into_iter().map(ToolsetEffect::Released),
            );
            effects.extend(self.open());
            return effects;
        }
        if self.phase != Phase::Opening {
            return effects;
        }
        self.phase = if success { Phase::Ready } else { Phase::Idle };
        for id in std::mem::take(&mut self.pending) {
            if success {
                self.owned.insert(id);
                effects.push(ToolsetEffect::Grant(id));
            } else {
                effects.push(ToolsetEffect::Fail(id));
            }
        }
        effects
    }
    pub fn closed(&mut self) -> Vec<ToolsetEffect> {
        if self.phase != Phase::Closing {
            return Vec::new();
        }
        self.phase = Phase::Idle;
        let mut effects: Vec<_> =
            std::mem::take(&mut self.releasing).into_iter().map(ToolsetEffect::Released).collect();
        effects.extend(self.open());
        effects
    }
    pub fn cancelling_setup(&self) -> bool {
        self.phase == Phase::Aborting
    }
}

/// One scope acquires roots in declaration order and releases every successful acquisition
/// before reporting initialization failure. Shared child identity is handled by each lifetime.
pub struct Group {
    total: usize,
    acquired: usize,
    failed: bool,
    closing: bool,
}
impl Group {
    pub fn new(total: usize) -> Self {
        Self { total, acquired: 0, failed: false, closing: false }
    }
    pub fn next(&self) -> Option<usize> {
        (!self.failed && !self.closing && self.acquired < self.total).then_some(self.acquired)
    }
    pub fn acquired(&mut self, success: bool) {
        if success {
            self.acquired += 1;
        } else {
            self.failed = true;
        }
    }
    pub fn request_close(&mut self) {
        self.closing = true;
    }
    pub fn ready(&self) -> bool {
        !self.failed && !self.closing && self.acquired == self.total
    }
    pub fn releases(&self) -> std::ops::Range<usize> {
        0..self.acquired
    }
}

#[derive(Default)]
pub struct Scopes {
    session_roots: BTreeSet<u64>,
    groups: std::collections::BTreeMap<u64, std::collections::VecDeque<u64>>,
}
impl Scopes {
    pub fn admit(&mut self, scope: u64, group: u64, roots: &[u64]) -> Vec<usize> {
        self.groups.entry(scope).or_default().push_back(group);
        if scope == 0 {
            self.session_roots.extend(roots);
        }
        let mut seen = BTreeSet::new();
        roots
            .iter()
            .enumerate()
            .filter_map(|(index, root)| {
                (seen.insert(*root) && (scope == 0 || !self.session_roots.contains(root)))
                    .then_some(index)
            })
            .collect()
    }
    /// Roll back one attempted activation without touching an older use of the same agent.
    pub fn discard(&mut self, group: u64) {
        self.groups.retain(|_, groups| {
            groups.retain(|id| *id != group);
            !groups.is_empty()
        });
    }
    pub fn release(&mut self, scope: Option<u64>) -> Vec<u64> {
        match scope {
            Some(scope) => self
                .groups
                .get_mut(&scope)
                .and_then(|groups| groups.pop_front())
                .into_iter()
                .collect(),
            None => {
                self.session_roots.clear();
                std::mem::take(&mut self.groups).into_values().flatten().collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scopes_preserve_session_roots_and_release_each_activation_separately() {
        let mut scopes = Scopes::default();
        assert_eq!(scopes.admit(0, 1, &[10, 10, 20]), vec![0, 2]);
        assert_eq!(scopes.admit(99, 2, &[20, 30, 30]), vec![1]);
        assert_eq!(scopes.admit(99, 3, &[30]), vec![0]);
        assert_eq!(scopes.release(Some(99)), vec![2]);
        assert_eq!(scopes.release(Some(99)), vec![3]);
        assert_eq!(scopes.release(None), vec![1]);
    }

    #[test]
    fn concurrent_users_share_setup_and_last_release_closes() {
        let mut state = Lifetime::default();
        assert_eq!(state.acquire(1), vec![ToolsetEffect::Setup(1)]);
        assert!(state.acquire(2).is_empty());
        assert_eq!(state.opened(true), vec![ToolsetEffect::Grant(1), ToolsetEffect::Grant(2)]);
        assert_eq!(state.release(1), vec![ToolsetEffect::Released(1)]);
        assert_eq!(state.release(2), vec![ToolsetEffect::Close]);
        assert!(state.acquire(3).is_empty());
        assert_eq!(state.closed(), vec![ToolsetEffect::Released(2), ToolsetEffect::Setup(3)]);
    }
    #[test]
    fn cancelling_one_waiter_keeps_shared_setup_but_last_waiter_cancels_it() {
        let mut state = Lifetime::default();
        state.acquire(1);
        state.acquire(2);
        assert_eq!(state.release(1), vec![ToolsetEffect::Released(1)]);
        assert_eq!(state.release(2), vec![ToolsetEffect::CancelSetup]);
        assert!(state.acquire(3).is_empty());
        assert_eq!(state.opened(false), vec![ToolsetEffect::Released(2), ToolsetEffect::Setup(3)]);
        assert_eq!(state.opened(false), vec![ToolsetEffect::Fail(3)]);
        assert_eq!(state.acquire(4), vec![ToolsetEffect::Setup(4)]);
    }
    #[test]
    fn setup_success_racing_last_waiter_cancellation_still_closes() {
        let mut state = Lifetime::default();
        state.acquire(1);
        state.release(1);
        assert_eq!(state.opened(true), vec![ToolsetEffect::Close]);
        assert_eq!(state.closed(), vec![ToolsetEffect::Released(1)]);
    }
}
