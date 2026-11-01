//! Session provider precedence over runtime-owned resource identities.

use std::collections::{BTreeMap, BTreeSet};

type Providers = BTreeMap<String, u64>;

/// Base providers updated explicitly, with temporary overrides held in insertion order.
/// Values identify runtime resources; no provider or host object enters the core.
#[derive(Default)]
pub struct ProviderLayers {
    base: Providers,
    layers: Vec<(u64, Providers)>,
}

impl ProviderLayers {
    pub fn update(&mut self, providers: Providers) -> Providers {
        let before = self.resolved();
        self.base.extend(providers);
        self.changed(before)
    }

    pub fn push(&mut self, layer: u64, providers: Providers) -> Providers {
        let before = self.resolved();
        self.layers.push((layer, providers));
        self.changed(before)
    }

    pub fn release(&mut self, layer: u64) -> Providers {
        let before = self.resolved();
        self.layers.retain(|(id, _)| *id != layer);
        self.changed(before)
    }

    /// Resources still needed by the base or any held layer, including covered providers.
    pub fn retained(&self) -> BTreeSet<u64> {
        self.base
            .values()
            .chain(self.layers.iter().flat_map(|(_, p)| p.values()))
            .copied()
            .collect()
    }

    fn resolved(&self) -> Providers {
        let mut providers = self.base.clone();
        for (_, layer) in &self.layers {
            providers.extend(layer.iter().map(|(leg, id)| (leg.clone(), *id)));
        }
        providers
    }

    fn changed(&self, before: Providers) -> Providers {
        self.resolved().into_iter().filter(|(leg, id)| before.get(leg) != Some(id)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn providers(entries: &[(&str, u64)]) -> Providers {
        entries.iter().map(|(leg, id)| ((*leg).to_owned(), *id)).collect()
    }

    #[test]
    fn covered_base_updates_and_out_of_order_release_preserve_the_topmost_provider() {
        let mut layers = ProviderLayers::default();
        layers.update(providers(&[("tts", 1), ("stt", 2)]));
        assert_eq!(layers.push(10, providers(&[("tts", 3)])), providers(&[("tts", 3)]));
        assert_eq!(layers.push(11, providers(&[("tts", 4)])), providers(&[("tts", 4)]));
        assert_eq!(layers.update(providers(&[("tts", 5), ("stt", 6)])), providers(&[("stt", 6)]));
        assert!(layers.release(10).is_empty());
        assert_eq!(layers.retained(), BTreeSet::from([4, 5, 6]));
        assert_eq!(layers.release(11), providers(&[("tts", 5)]));
        assert!(layers.release(11).is_empty());
        assert_eq!(layers.retained(), BTreeSet::from([5, 6]));
    }

    #[test]
    fn releasing_a_layer_restores_only_its_uncovered_legs() {
        let mut layers = ProviderLayers::default();
        layers.update(providers(&[("tts", 1), ("vad", 2)]));
        layers.push(10, providers(&[("tts", 3), ("vad", 4)]));
        layers.push(11, providers(&[("tts", 5)]));
        assert_eq!(layers.release(10), providers(&[("vad", 2)]));
        assert_eq!(layers.release(11), providers(&[("tts", 1)]));
    }
}
