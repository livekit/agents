//! Voice-session startup policy over supplied configuration facts.

pub struct StorageStart {
    configured_lane: bool,
}

pub enum StorageSource<T> {
    Lane(T),
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
    /// Refine the supplied load result without retaining its payload in decision state.
    /// A Lane outcome therefore always carries the successful result it selected.
    pub fn after_load<T>(&self, loaded: Option<T>) -> StorageSource<T> {
        match loaded {
            Some(lane) if self.configured_lane => StorageSource::Lane(lane),
            Some(_) | None => StorageSource::Ephemeral,
        }
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
        assert!(matches!(StorageStart::new(true).after_load(Some(())), StorageSource::Lane(())));
        assert!(matches!(StorageStart::new(true).after_load::<()>(None), StorageSource::Ephemeral));
        assert!(matches!(
            StorageStart::new(false).after_load::<()>(None),
            StorageSource::Ephemeral
        ));
        assert!(matches!(StorageStart::new(false).after_load(Some(())), StorageSource::Ephemeral));
        assert!(matches!(
            StorageStart::new(true).after_load(Some("loaded lane")),
            StorageSource::Lane("loaded lane")
        ));
    }
}
