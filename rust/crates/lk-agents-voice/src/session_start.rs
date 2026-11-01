//! Voice-session startup policy over supplied configuration facts.

pub struct StorageStart {
    configured_lane: bool,
}

pub enum StorageSource {
    Lane,
    Ephemeral,
}

pub use lk_agents_core::resource_lifecycle::{CloseOwner, OwnerLifecycle, StartOwner};

/// The latest requested agent is exposed until that handoff settles. An older handoff cannot
/// clear a newer selection, even when both requests name the same agent.
#[derive(Default)]
pub struct AgentSelection {
    next: u64,
    pending: Option<u64>,
}

impl AgentSelection {
    pub fn begin(&mut self) -> u64 {
        self.next += 1;
        self.pending = Some(self.next);
        self.next
    }
    pub fn settled(&mut self, id: u64) -> bool {
        if self.pending == Some(id) {
            self.pending = None;
            true
        } else {
            false
        }
    }
}

impl StorageStart {
    pub fn new(configured_lane: bool) -> Self {
        Self { configured_lane }
    }
    pub fn load_lane(&self) -> bool {
        self.configured_lane
    }
    pub fn after_load(&self, loaded: bool) -> StorageSource {
        if self.configured_lane && loaded { StorageSource::Lane } else { StorageSource::Ephemeral }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_older_handoff_cannot_clear_the_latest_selection() {
        let mut selection = AgentSelection::default();
        let first = selection.begin();
        let latest = selection.begin();
        assert!(!selection.settled(first));
        assert!(selection.settled(latest));
        assert!(!selection.settled(latest));
    }

    #[test]
    fn only_a_successfully_loaded_configured_lane_is_durable() {
        assert!(!StorageStart::new(false).load_lane());
        assert!(StorageStart::new(true).load_lane());
        assert!(matches!(StorageStart::new(true).after_load(true), StorageSource::Lane));
        assert!(matches!(StorageStart::new(true).after_load(false), StorageSource::Ephemeral));
        assert!(matches!(StorageStart::new(false).after_load(false), StorageSource::Ephemeral));
    }
}
