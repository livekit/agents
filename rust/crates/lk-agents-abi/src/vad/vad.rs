//! `Vad`: `VAD.stream()` is abstract (`vad.py:94`), so it returns `Result`, not `Option<Result>`.

use crate::VadEvent;
use crate::errors::ProviderError;

use crate::AudioFrame;
use crate::utils::stream::{StreamSink, StreamSource};

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
pub type VadSink = Box<dyn StreamSink<AudioFrame>>;
pub type VadSource = Box<dyn StreamSource<VadEvent>>;

// A host `vad.VAD` is this trait through its generated proxy (`VadProxy`), calling the `_core_*`
// methods of the base class `lk-agents-codegen` writes from the `#[python]` markers.
/// A voice activity detector.
///
/// Plugins implement `stream()`, returning a `VADStream` whose `_main_task` turns the pushed
/// audio into `VADEvent`s.
#[crate::host(proxy(host = "livekit.agents.VAD"))]
#[python(provider(module = "vad", path = "livekit.agents", events = "metrics_collected"))]
pub trait Vad: Send + Sync {
    /// The name logs and metrics give this instance: its class path.
    #[python(label)]
    #[host(read_once, name = "_core_label")]
    fn label(&self) -> String;

    /// What this instance supports, fixed at construction.
    #[python(init)]
    #[host(read_once, name = "_core_capabilities")]
    fn capabilities(&self) -> VadCapabilities;

    /// The model and provider labelling this instance's metrics.
    // `label`/`model`/`provider` as metrics and errors stamp them; defaults from the label.
    #[python(identity)]
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label(self.label())
    }

    /// Load the model early; a detector with nothing to warm ignores it.
    #[python(hook)]
    #[host(notify, name = "_core_prewarm")]
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
    #[host(await, name = "_core_stream")]
    fn stream(&self) -> crate::BoxFuture<'_, Result<(VadSink, VadSource), ProviderError>>;
}
