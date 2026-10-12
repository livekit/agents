//! `Vad`: `VAD.stream()` is abstract (`vad.py:94`), so it returns `Result`, not `Option<Result>`.

use crate::VadEvent;
use lk_agents_host_derive::Host;
use async_trait::async_trait;
use crate::errors::ProviderError;

use crate::AudioFrame;
use crate::utils::stream::{StreamSink, StreamSource};

// `vad.py:71`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Host)]
#[python(dataclass, module = "vad", path = "livekit.agents.vad", name = "VADCapabilities")]
#[node(interface)]
#[binding(extract)]
pub struct VadCapabilities {
    // Seconds between `INFERENCE_DONE` events; the driver guards against zero (`vad.py:127`).
    pub update_interval: f64,
}

/// May drop.
pub type VadSink = Box<dyn StreamSink<AudioFrame>>;
pub type VadSource = Box<dyn StreamSource<VadEvent>>;

/// A voice activity detector. A host `vad.VAD` is this trait through its bare `#[host(proxy)]`
/// (`PyVad`): each method calls the base class's `_core_<method>`.
#[crate::host(proxy(host = "livekit.agents.vad.VAD", clock))]
#[async_trait]
pub trait Vad: Send + Sync {
    /// `VAD._label`: the reference has no public `label`.
    fn label(&self) -> String;
    fn capabilities(&self) -> VadCapabilities;

    /// `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// `prewarm()`: open the connection pool early; a provider with nothing to warm ignores it.
    fn prewarm(&self) {}

    /// `Err` is a construction failure. `async` as on [`Tts`](crate::Tts).
    async fn stream(&self) -> Result<(VadSink, VadSource), ProviderError>;
}
