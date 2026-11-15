use super::effects::RoomBindingEffect;
use super::{BindingId, BindingKey, BindingPhase, BindingRefused};
use std::collections::BTreeMap;

struct Binding {
    key: BindingKey,
    phase: BindingPhase,
}

/// One room connection's binding registry. IDs arrive from its runtime owner;
/// all asynchronous completions name that incarnation, never only its readable key.
#[derive(Default)]
pub struct RoomBindings {
    closing: bool,
    last_id: Option<BindingId>,
    bindings: BTreeMap<BindingId, Binding>,
    keys: BTreeMap<BindingKey, BindingId>,
}

impl RoomBindings {
    pub fn declare(&mut self, id: BindingId, key: BindingKey) -> Result<(), BindingRefused> {
        if self.closing {
            return Err(BindingRefused::OwnerClosed);
        }
        if self.last_id.is_some_and(|last| id <= last) {
            return Err(BindingRefused::Unavailable);
        }
        if self.keys.contains_key(&key) {
            return Err(BindingRefused::KeyInUse(key));
        }
        self.keys.insert(key.clone(), id);
        self.bindings.insert(id, Binding { key, phase: BindingPhase::Declared });
        self.last_id = Some(id);
        Ok(())
    }

    /// Acquire the binding's single activation authority before acquiring resources.
    pub fn begin_start(&mut self, id: BindingId) -> Result<(), BindingRefused> {
        if self.closing {
            return Err(BindingRefused::OwnerClosed);
        }
        let binding = self.bindings.get_mut(&id).ok_or(BindingRefused::Unavailable)?;
        if binding.phase != BindingPhase::Declared {
            return Err(BindingRefused::Unavailable);
        }
        binding.phase = BindingPhase::Starting;
        Ok(())
    }

    /// Successful activation may arrive after cancellation. Only an accepted
    /// completion may become eligible for the legacy bridge's primary selection.
    pub fn started(&mut self, id: BindingId) -> bool {
        let Some(binding) = self.bindings.get_mut(&id) else { return false };
        if binding.phase != BindingPhase::Starting {
            return false;
        }
        binding.phase = BindingPhase::Active;
        true
    }

    /// The cleanup owner keeps the incarnation reserved until it acknowledges retirement.
    pub fn close_binding(&mut self, id: BindingId) -> Option<RoomBindingEffect> {
        let binding = self.bindings.get_mut(&id)?;
        if binding.phase == BindingPhase::Closing {
            return None;
        }
        binding.phase = BindingPhase::Closing;
        Some(RoomBindingEffect::Close(id))
    }

    /// Release only the incarnation whose cleanup has completed. An old receipt
    /// cannot remove a replacement which happens to use the same key.
    pub fn retired(&mut self, id: BindingId) -> bool {
        if self.phase(id) != Some(BindingPhase::Closing) {
            return false;
        }
        let binding = self.bindings.remove(&id).expect("the closing binding is present");
        self.keys.remove(&binding.key);
        true
    }

    /// Stop admission before asking every live binding to close, in stable ID order.
    pub fn close(&mut self) -> Vec<RoomBindingEffect> {
        self.closing = true;
        let mut effects = Vec::new();
        for (&id, binding) in &mut self.bindings {
            if binding.phase != BindingPhase::Closing {
                binding.phase = BindingPhase::Closing;
                effects.push(RoomBindingEffect::Close(id));
            }
        }
        effects
    }

    pub fn phase(&self, id: BindingId) -> Option<BindingPhase> {
        self.bindings.get(&id).map(|binding| binding.phase)
    }

    pub fn settled(&self) -> bool {
        self.closing && self.bindings.is_empty()
    }
}

#[cfg(test)]
#[path = "machine_tests.rs"]
mod tests;
