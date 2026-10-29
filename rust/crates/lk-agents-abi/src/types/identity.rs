//! Who a provider is, as metrics and errors report it; defaults to `"unknown"` as the reference.

use lk_agents_host_derive::Host;

/// `label`, `model`, `provider`: the identity the reference's `Metadata` and `label` fields carry.
///
/// Read off a host provider's `_core_id()`, a dict with the base classes' `"unknown"` defaults.
#[derive(Debug, Clone, PartialEq, Eq, Default, Host)]
#[binding(extract, from_items)]
pub struct ProviderId {
    /// The name logs and metrics give a provider: a plugin's class path, whatever a native
    /// provider calls itself.
    pub label: String,
    /// The model name; `"unknown"` until a plugin overrides it.
    // Never a stand-in for a name.
    pub model: String,
    /// The provider name; `"unknown"` until a plugin overrides it.
    pub provider: String,
}

impl ProviderId {
    /// The reference's base-class defaults for a provider that only knows its label.
    pub fn of_label(label: impl Into<String>) -> Self {
        ProviderId {
            label: label.into(),
            model: "unknown".to_owned(),
            provider: "unknown".to_owned(),
        }
    }

    pub fn metadata(&self) -> crate::metrics::base::Metadata {
        crate::metrics::base::Metadata {
            model_name: Some(self.model.clone()),
            model_provider: Some(self.provider.clone()),
        }
    }

    /// `metrics_metadata`: the turn metrics' `{stt,llm,tts}_metadata` on a chat item.
    pub fn metrics_metadata(&self) -> crate::MetricsMetadata {
        crate::MetricsMetadata {
            model_name: Some(self.model.clone()),
            model_provider: Some(self.provider.clone()),
        }
    }
}
