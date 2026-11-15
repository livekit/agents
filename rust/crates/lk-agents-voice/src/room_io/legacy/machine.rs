use super::effects::LegacyPrimaryEffect;
use super::{LegacyProjection, LegacyTargetUnavailable};
use crate::room_io::bindings::BindingId;

#[derive(Default)]
enum Selection {
    #[default]
    Automatic,
    Pinned(BindingId),
}

/// Only successfully activated, legacy-compatible bindings enter this machine.
/// The shared room owner serializes its inputs with binding lifecycle transitions.
pub struct LegacyPrimary {
    enabled: bool,
    selection: Selection,
    /// Successful activation order, independent of declaration and task scheduling order.
    eligible: Vec<BindingId>,
    current: Option<LegacyProjection>,
    generation: u64,
}

impl Default for LegacyPrimary {
    fn default() -> Self {
        Self::new(true)
    }
}

impl LegacyPrimary {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            selection: Selection::Automatic,
            eligible: Vec::new(),
            current: None,
            generation: 0,
        }
    }

    pub fn started(&mut self, id: BindingId) -> Option<LegacyPrimaryEffect> {
        if !self.enabled || self.eligible.contains(&id) {
            return None;
        }
        self.eligible.push(id);
        match self.selection {
            Selection::Automatic => self.select(Some(id)),
            Selection::Pinned(_) => None,
        }
    }

    /// Losing the primary ends the projection. It never silently promotes a different session.
    pub fn closing(&mut self, id: BindingId) -> Option<LegacyPrimaryEffect> {
        self.eligible.retain(|&candidate| candidate != id);
        if self.current.is_some_and(|current| current.binding == id) {
            self.select(None)
        } else {
            None
        }
    }

    pub fn pin(
        &mut self,
        id: BindingId,
    ) -> Result<Option<LegacyPrimaryEffect>, LegacyTargetUnavailable> {
        if !self.eligible.contains(&id) {
            return Err(LegacyTargetUnavailable);
        }
        self.selection = Selection::Pinned(id);
        Ok(self.select(Some(id)))
    }

    /// Explicitly returning to automatic policy selects the latest still-active
    /// compatible binding. Future successful activations may replace it.
    pub fn automatic(&mut self) -> Option<LegacyPrimaryEffect> {
        self.selection = Selection::Automatic;
        self.select(self.eligible.last().copied())
    }

    pub fn current(&self) -> Option<LegacyProjection> {
        self.current
    }

    pub fn pinned(&self) -> Option<BindingId> {
        match self.selection {
            Selection::Automatic => None,
            Selection::Pinned(id) => Some(id),
        }
    }

    /// Checks projection-scoped publication, not a response to an earlier admitted request.
    pub fn accepts_event(&self, captured: LegacyProjection) -> bool {
        self.current == Some(captured)
    }

    fn select(&mut self, target: Option<BindingId>) -> Option<LegacyPrimaryEffect> {
        if self.current.map(|projection| projection.binding) == target {
            return None;
        }
        self.generation = self.generation.checked_add(1).expect("legacy generations cannot wrap");
        let current =
            target.map(|binding| LegacyProjection { binding, generation: self.generation });
        let previous = std::mem::replace(&mut self.current, current);
        Some(LegacyPrimaryEffect::Cutover { previous, current })
    }
}

#[cfg(test)]
#[path = "machine_tests.rs"]
mod tests;
