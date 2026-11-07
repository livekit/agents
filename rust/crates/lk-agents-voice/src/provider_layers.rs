//! Session provider precedence over runtime-owned resource identities.

use std::collections::{BTreeMap, BTreeSet};

/// The model slots the session understands. Extension names preserve the existing internal
/// host-map round trip without treating arbitrary strings as one of the built-in models.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProviderSlot {
    Llm,
    Stt,
    Tts,
    Vad,
    Extension(String),
}
impl From<String> for ProviderSlot {
    fn from(name: String) -> Self {
        match name.as_str() {
            "llm" => Self::Llm,
            "stt" => Self::Stt,
            "tts" => Self::Tts,
            "vad" => Self::Vad,
            _ => Self::Extension(name),
        }
    }
}
impl ProviderSlot {
    pub fn into_name(self) -> String {
        match self {
            Self::Llm => "llm".into(),
            Self::Stt => "stt".into(),
            Self::Tts => "tts".into(),
            Self::Vad => "vad".into(),
            Self::Extension(name) => name,
        }
    }
}

/// Identity in the host-object retention table, distinct from an executable provider ResourceId.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProviderObjectId(pub u64);

/// Shared with turn-setting override ownership, converted once at the host boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderOverrideId(pub u64);

pub type Providers = BTreeMap<ProviderSlot, ProviderObjectId>;

/// Base providers updated explicitly, with temporary overrides held in insertion order.
/// Values identify runtime resources; no provider or host object enters the core.
#[derive(Default)]
pub struct ProviderLayers {
    base: Providers,
    layers: Vec<(ProviderOverrideId, Providers)>,
}

impl ProviderLayers {
    pub fn update(&mut self, providers: Providers) -> Providers {
        let before = self.resolved();
        self.base.extend(providers);
        self.changed(before)
    }

    pub fn push(&mut self, layer: ProviderOverrideId, providers: Providers) -> Providers {
        let before = self.resolved();
        self.layers.push((layer, providers));
        self.changed(before)
    }

    pub fn release(&mut self, layer: ProviderOverrideId) -> Providers {
        let before = self.resolved();
        self.layers.retain(|(id, _)| *id != layer);
        self.changed(before)
    }

    /// Resources still needed by the base or any held layer, including covered providers.
    pub fn retained(&self) -> BTreeSet<ProviderObjectId> {
        self.base
            .values()
            .chain(self.layers.iter().flat_map(|(_, p)| p.values()))
            .copied()
            .collect()
    }

    /// What `slot` runs on now: the topmost held layer's, else the base's.
    pub fn effective(&self, slot: &ProviderSlot) -> Option<ProviderObjectId> {
        let layers = self.layers.iter().rev().map(|(_, layer)| layer);
        layers.chain([&self.base]).find_map(|layer| layer.get(slot)).copied()
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
        entries
            .iter()
            .map(|(leg, id)| (ProviderSlot::from((*leg).to_owned()), ProviderObjectId(*id)))
            .collect()
    }

    #[test]
    fn covered_base_updates_and_out_of_order_release_preserve_the_topmost_provider() {
        let mut layers = ProviderLayers::default();
        layers.update(providers(&[("tts", 1), ("stt", 2)]));
        assert_eq!(
            layers.push(ProviderOverrideId(10), providers(&[("tts", 3)])),
            providers(&[("tts", 3)])
        );
        assert_eq!(
            layers.push(ProviderOverrideId(11), providers(&[("tts", 4)])),
            providers(&[("tts", 4)])
        );
        assert_eq!(layers.update(providers(&[("tts", 5), ("stt", 6)])), providers(&[("stt", 6)]));
        assert!(layers.release(ProviderOverrideId(10)).is_empty());
        assert_eq!(layers.retained(), BTreeSet::from([4, 5, 6].map(ProviderObjectId)));
        assert_eq!(layers.release(ProviderOverrideId(11)), providers(&[("tts", 5)]));
        assert!(layers.release(ProviderOverrideId(11)).is_empty());
        assert_eq!(layers.retained(), BTreeSet::from([5, 6].map(ProviderObjectId)));
    }

    // Catches: the effective provider read from the base under a held layer, or from the first
    // layer instead of the topmost.
    #[test]
    fn the_effective_provider_is_the_topmost_layers_else_the_bases() {
        let mut layers = ProviderLayers::default();
        let tts = ProviderSlot::Tts;
        assert_eq!(layers.effective(&tts), None);
        layers.update(providers(&[("tts", 1), ("stt", 2)]));
        layers.push(ProviderOverrideId(10), providers(&[("tts", 3)]));
        layers.push(ProviderOverrideId(11), providers(&[("tts", 4)]));
        layers.update(providers(&[("tts", 5)]));
        assert_eq!(layers.effective(&tts), Some(ProviderObjectId(4)));
        assert_eq!(layers.effective(&ProviderSlot::Stt), Some(ProviderObjectId(2)));
        layers.release(ProviderOverrideId(11));
        assert_eq!(layers.effective(&tts), Some(ProviderObjectId(3)));
        layers.release(ProviderOverrideId(10));
        assert_eq!(layers.effective(&tts), Some(ProviderObjectId(5)));
    }

    #[test]
    fn releasing_a_layer_restores_only_its_uncovered_legs() {
        let mut layers = ProviderLayers::default();
        layers.update(providers(&[("tts", 1), ("vad", 2)]));
        layers.push(ProviderOverrideId(10), providers(&[("tts", 3), ("vad", 4)]));
        layers.push(ProviderOverrideId(11), providers(&[("tts", 5)]));
        assert_eq!(layers.release(ProviderOverrideId(10)), providers(&[("vad", 2)]));
        assert_eq!(layers.release(ProviderOverrideId(11)), providers(&[("tts", 1)]));
    }
}
