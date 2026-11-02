//! VoiceRuntime retention of host provider objects selected by the voice core.

use lk_agents_abi::HostRef;
use lk_agents_voice::provider_layers::{
    ProviderLayers, ProviderObjectId, ProviderOverrideId, ProviderSlot,
};
use std::collections::BTreeMap;

pub(super) type Providers = BTreeMap<String, Option<HostRef>>;

#[derive(Default)]
pub(super) struct ProviderRegistry {
    layers: ProviderLayers,
    objects: BTreeMap<ProviderObjectId, Option<HostRef>>,
    next: u64,
}

impl ProviderRegistry {
    fn register(&mut self, providers: Providers) -> BTreeMap<ProviderSlot, ProviderObjectId> {
        providers
            .into_iter()
            .map(|(leg, provider)| {
                self.next += 1;
                self.objects.insert(ProviderObjectId(self.next), provider);
                (ProviderSlot::from(leg), ProviderObjectId(self.next))
            })
            .collect()
    }

    pub fn update(&mut self, providers: Providers) -> (Providers, Vec<Option<HostRef>>) {
        let ids = self.register(providers);
        let changed = self.layers.update(ids);
        self.collect(changed)
    }

    pub fn push(&mut self, layer: u64, providers: Providers) -> (Providers, Vec<Option<HostRef>>) {
        let ids = self.register(providers);
        let changed = self.layers.push(ProviderOverrideId(layer), ids);
        self.collect(changed)
    }

    pub fn release(&mut self, layer: u64) -> (Providers, Vec<Option<HostRef>>) {
        let changed = self.layers.release(ProviderOverrideId(layer));
        self.collect(changed)
    }

    fn collect(
        &mut self,
        changed: BTreeMap<ProviderSlot, ProviderObjectId>,
    ) -> (Providers, Vec<Option<HostRef>>) {
        let providers = changed
            .into_iter()
            .map(|(leg, id)| (leg.into_name(), self.objects[&id].clone()))
            .collect();
        let retained = self.layers.retained();
        let retired: Vec<_> =
            self.objects.keys().filter(|id| !retained.contains(id)).copied().collect();
        // Return retired host references to the caller: their destructors run after its lock.
        let retired = retired.into_iter().filter_map(|id| self.objects.remove(&id)).collect();
        (providers, retired)
    }
}
