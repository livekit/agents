//! `Vad`: `VAD.stream()` is abstract (`vad.py:94`), so it returns `Result`, not `Option<Result>`.

use crate::VadEvent;
use crate::errors::ProviderError;

use crate::AudioFrame;
use crate::utils::stream::{ProviderStream, StreamInput};

// `vad.py:71`.
#[crate::host(record(
    module = "vad",
    path = "livekit.agents",
    name = "VADCapabilities",
    dataclass,
    extract,
))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VadCapabilities {
    // Seconds between `INFERENCE_DONE` events; the driver guards against zero (`vad.py:127`).
    pub update_interval: f64,
}

/// May drop.
pub type VadSink = Box<dyn StreamInput<AudioFrame>>;
pub type VadSource = Box<dyn ProviderStream<VadEvent>>;

// A host `vad.VAD` is this trait through its generated proxy (`VadProxy`), calling the `_core_*`
// methods of its hand-written base class (`vad.py`).
/// A voice activity detector.
///
/// Plugins implement `stream()`, returning a `VADStream` whose `_main_task` turns the pushed
/// audio into `VADEvent`s.
#[crate::host(proxy(host = "livekit.agents.VAD", base = "vad.py", generic))]
pub trait Vad: Send + Sync {
    /// The name logs and metrics give this instance: its class path.
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> String;

    /// What this instance supports, fixed at construction.
    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> VadCapabilities;

    /// The model and provider labelling this instance's metrics.
    // `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// Load the model early; a detector with nothing to warm ignores it.
    #[host(notify, name = "_core_prewarm")]
    fn prewarm(&self) {}

    /// Open a stream detecting speech in the audio pushed into it.
    // `Err` is a construction failure. `async` as on [`Tts`](crate::Tts).
    #[host(await, name = "_core_stream")]
    fn stream(
        &self,
    ) -> crate::BoxFuture<'_, Result<crate::DuplexStream<VadSink, VadSource>, ProviderError>>;
}
