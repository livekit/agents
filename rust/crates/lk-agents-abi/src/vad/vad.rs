//! `Vad`: `VAD.stream()` is abstract (`vad.py:94`), so it returns `Result`, not `Option<Result>`.

use crate::VadEvent;
use lk_agents_host_derive::Host;
use async_trait::async_trait;
use crate::errors::ProviderError;

use crate::AudioFrame;
use crate::utils::stream::{StreamSink, StreamSource};

// `vad.py:71`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Host)]
#[python(dataclass, module = "vad", path = "livekit.agents", name = "VADCapabilities")]
#[node(interface)]
#[binding(extract)]
pub struct VadCapabilities {
    // Seconds between `INFERENCE_DONE` events; the driver guards against zero (`vad.py:127`).
    pub update_interval: f64,
}

/// May drop.
pub type VadSink = Box<dyn StreamSink<AudioFrame>>;
pub type VadSource = Box<dyn StreamSource<VadEvent>>;

// A host `vad.VAD` is this trait through its bare `#[host(proxy)]` (`PyVad`), over the base class
// `lk-agents-codegen` writes from the `#[python]` markers.
/// A voice activity detector.
///
/// Plugins implement `stream()`, returning a `VADStream` whose `_main_task` turns the pushed
/// audio into `VADEvent`s.
#[crate::host(proxy(host = "livekit.agents.vad.VAD", clock))]
#[python(provider(module = "vad", path = "livekit.agents", events = "metrics_collected"))]
#[async_trait]
pub trait Vad: Send + Sync {
    /// The name logs and metrics give this instance: its class path.
    #[python(label)]
    fn label(&self) -> String;

    /// What this instance supports, fixed at construction.
    #[python(init)]
    fn capabilities(&self) -> VadCapabilities;

    /// The model and provider labelling this instance's metrics.
    // `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    #[python(identity)]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// Load the model early; a detector with nothing to warm ignores it.
    #[python(hook)]
    fn prewarm(&self) {}

    /// Open a stream detecting speech in the audio pushed into it.
    // `Err` is a construction failure. `async` as on [`Tts`](crate::Tts).
    #[python(
        hook,
        stream(
            name = "VADStream",
            kind = "vad",
            doc = "A live detection: audio in through ``push_frame``, ``VADEvent`` out.\n\nPlugins implement ``_main_task``, reading ``input`` and handing each event to ``send``."
        )
    )]
    async fn stream(&self) -> Result<(VadSink, VadSource), ProviderError>;
}
