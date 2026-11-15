use super::BindingCapabilities;
use super::effects::SharedRoomEffect;
use crate::room_io::bindings::machine::RoomBindings;
use crate::room_io::bindings::{BindingId, BindingKey, BindingPhase, BindingRefused};
use crate::room_io::legacy::machine::LegacyPrimary;
use crate::room_io::legacy::{LegacyProjection, LegacyTargetUnavailable};
use std::collections::BTreeMap;

/// The registry and legacy projection transition together. The runtime cannot
/// accidentally promote a cancelled start or retain a closing binding as primary.
#[derive(Default)]
pub struct SharedRoom {
    bindings: RoomBindings,
    capabilities: BTreeMap<BindingId, BindingCapabilities>,
    legacy: LegacyPrimary,
}

impl SharedRoom {
    pub fn new(legacy_enabled: bool) -> Self {
        Self {
            bindings: RoomBindings::default(),
            capabilities: BTreeMap::new(),
            legacy: LegacyPrimary::new(legacy_enabled),
        }
    }

    pub fn declare(
        &mut self,
        id: BindingId,
        key: BindingKey,
        capabilities: BindingCapabilities,
    ) -> Result<(), BindingRefused> {
        self.bindings.declare(id, key)?;
        self.capabilities.insert(id, capabilities);
        Ok(())
    }

    pub fn begin_start(&mut self, id: BindingId) -> Result<(), BindingRefused> {
        self.bindings.begin_start(id)
    }

    pub fn started(&mut self, id: BindingId) -> Result<Vec<SharedRoomEffect>, BindingRefused> {
        if !self.bindings.started(id) {
            return Err(BindingRefused::Unavailable);
        }
        let mut effects = Vec::new();
        if self.capabilities.get(&id).is_some_and(|caps| caps.legacy_voice)
            && let Some(effect) = self.legacy.started(id)
        {
            effects.push(SharedRoomEffect::Legacy(effect));
        }
        Ok(effects)
    }

    pub fn close_binding(&mut self, id: BindingId) -> Vec<SharedRoomEffect> {
        let Some(close) = self.bindings.close_binding(id) else { return Vec::new() };
        let mut effects = Vec::new();
        if let Some(effect) = self.legacy.closing(id) {
            effects.push(SharedRoomEffect::Legacy(effect));
        }
        effects.push(SharedRoomEffect::Binding(close));
        effects
    }

    pub fn retired(&mut self, id: BindingId) -> bool {
        if !self.bindings.retired(id) {
            return false;
        }
        self.capabilities.remove(&id);
        true
    }

    pub fn close(&mut self) -> Vec<SharedRoomEffect> {
        let closing = self.bindings.close();
        let mut effects = Vec::new();
        // Withdraw the projection before starting any binding cleanup.
        if let Some(current) = self.legacy.current()
            && let Some(effect) = self.legacy.closing(current.binding)
        {
            effects.push(SharedRoomEffect::Legacy(effect));
        }
        for id in self.capabilities.keys() {
            effects.extend(self.legacy.closing(*id).map(SharedRoomEffect::Legacy));
        }
        effects.extend(closing.into_iter().map(SharedRoomEffect::Binding));
        effects
    }

    pub fn pin_legacy(
        &mut self,
        id: BindingId,
    ) -> Result<Vec<SharedRoomEffect>, LegacyTargetUnavailable> {
        self.legacy.pin(id).map(|effect| effect.into_iter().map(SharedRoomEffect::Legacy).collect())
    }

    pub fn automatic_legacy(&mut self) -> Vec<SharedRoomEffect> {
        self.legacy.automatic().into_iter().map(SharedRoomEffect::Legacy).collect()
    }

    pub fn legacy_projection(&self) -> Option<LegacyProjection> {
        self.legacy.current()
    }

    pub fn accepts_legacy_event(&self, captured: LegacyProjection) -> bool {
        self.legacy.accepts_event(captured)
    }

    pub fn phase(&self, id: BindingId) -> Option<BindingPhase> {
        self.bindings.phase(id)
    }

    pub fn settled(&self) -> bool {
        self.bindings.settled()
    }
}

#[cfg(test)]
#[path = "machine_tests.rs"]
mod tests;
