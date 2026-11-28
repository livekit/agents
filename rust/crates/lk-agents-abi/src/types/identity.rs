/// `label`, `model`, `provider`: the identity the reference's `Metadata` and `label` fields carry.
///
/// Read off a host provider's `_core_id()`, a dict with the base classes' `"unknown"` defaults.
#[crate::host(record(module = "types", typed_dict, extract))]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProviderId {
    /// The name logs and metrics give a provider: a plugin's class path, whatever a native
    /// provider calls itself.
    pub label: String,
    /// The model name; `"unknown"` until a plugin overrides it.
    // Never a stand-in for a name.
    pub model: String,
    /// The provider name; `"unknown"` until a plugin overrides it.
    pub provider: String,
    /// The id of the plugin whose code runs the model, which every host's base class declares
    /// from its own packaging and no plugin writes: the last part of the package name
    /// (`livekit.plugins.deepgram` in Python, `@livekit/agents-plugin-deepgram` in Node, are
    /// both `deepgram`), `livekit-inference` for the LiveKit Inference models, `custom` for a
    /// class outside a LiveKit plugin. A Rust-native provider declares its own.
    #[host(default = "custom")]
    pub plugin: String,
    /// Behind a gateway (LiveKit Inference), the vendor whose model runs; blank when `provider`
    /// is that vendor. `provider` stays what the customer's metrics and usage name.
    #[host(default = "")]
    pub upstream_provider: String,
    /// Behind a gateway, the vendor's own id for the model, without the gateway's vendor prefix
    /// or options; blank when `model` is that id.
    #[host(default = "")]
    pub upstream_model: String,
}

impl ProviderId {
    /// The reference's base-class defaults for a provider that only knows its label.
    pub fn of_label(label: impl Into<String>) -> Self {
        ProviderId {
            label: label.into(),
            model: "unknown".to_owned(),
            provider: "unknown".to_owned(),
            plugin: "custom".to_owned(),
            ..ProviderId::default()
        }
    }

    /// The provider telemetry carries: the vendor id as declared, absent when blank.
    pub fn vendor(&self) -> Option<&str> {
        Some(self.provider.trim()).filter(|vendor| !vendor.is_empty())
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

#[cfg(test)]
mod tests {
    use super::ProviderId;

    // Catches: a declared provider respelled on its way to telemetry, or a blank one sent.
    #[test]
    fn the_vendor_is_the_declared_provider_and_absent_when_blank() {
        let id = |provider: &str| ProviderId { provider: provider.into(), ..Default::default() };
        assert_eq!(id("xai").vendor(), Some("xai"));
        assert_eq!(id(" aws ").vendor(), Some("aws"));
        assert_eq!(id("  ").vendor(), None);
    }
}
