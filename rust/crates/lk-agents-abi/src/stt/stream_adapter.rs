use std::sync::Arc;

use crate::errors::ProviderError;
use crate::tokenize::language::LanguageCode;
use crate::utils::stream::{StreamInput, StreamOutput};
use crate::vad::types::Vad;
use crate::{ApiConnectOptions, AudioFrame, SpeechEvent, Stt};

/// `stt.StreamAdapter`'s engine as the host builds it; the public class stays Python.
#[crate::host(class(name = "SttStreamAdapter", native = Stt))]
pub trait SttStreamAdapter: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(sink(item = AudioFrame, class = "SttAdapterStream"))]
    type Audio: StreamInput<AudioFrame> + 'static;
    #[host(source(item = SpeechEvent, class = "SttAdapterStream"))]
    type Events: StreamOutput<SpeechEvent> + 'static;

    /// The wrapped STT and VAD; each stream binds its own recognition policy.
    #[host(init)]
    fn new(
        env: &Self::Env,
        #[host(label)] label: String,
        events: Arc<dyn crate::types::provider::ProviderEvents>,
        #[host(kw)] stt: Arc<dyn Stt>,
        vad: Arc<dyn Vad>,
    ) -> Self;

    /// Streaming, with the wrapped STT's keyterms and chat context support.
    #[host(getter)]
    fn capabilities(&self) -> crate::SttCapabilities;

    /// `_recognize_impl`: the wrapped STT's batch path.
    #[host(await, name = "_core_native_recognize")]
    async fn recognize(
        &self,
        cx: crate::telemetry::Cx,
        frames: Vec<AudioFrame>,
        #[host(default = None)] language: Option<LanguageCode>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<SpeechEvent, ProviderError>;

    /// Audio into the VAD, its utterances recognized.
    #[host(await, name = "_core_native_stream")]
    async fn stream(
        &self,
        cx: crate::telemetry::Cx,
        #[host(kw, default = None)] language: Option<LanguageCode>,
        #[host(default = None)] start_time_offset: Option<f64>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<crate::DuplexStream<Self::Audio, Self::Events>, ProviderError>;
}

/// `stt.StreamAdapter`: a recognize-only STT as a streaming one.
#[crate::host(facade(
    path = "livekit.agents.stt",
    extended,
    name = "StreamAdapter",
    base = "stt.STT",
    native = SttStreamAdapter,
    refuse(stream, recognize, _recognize_impl),
    node = "session.stt_node",
    base_args(capabilities),
    keep(stt, vad)
))]
pub trait SttStreamAdapterFacade {
    /// A recognize-only STT as a streaming one: the VAD cuts the audio into utterances, each
    /// recognized on its own.
    ///
    /// Args:
    ///     stt: The recognize-only STT.
    ///     vad: What cuts the audio into utterances.
    #[host(new)]
    fn new(#[host(kw)] stt: Arc<dyn Stt>, vad: Arc<dyn Vad>) -> Self;
}

/// `stt.StreamAdapter.stream()`: audio into the VAD, its utterances recognized, each retried on
/// its own.
#[crate::host(facade(
    path = "livekit.agents.stt",
    base = "stt.RecognizeStream",
    native = SttStreamAdapter,
    stream(
        op = stream,
        owner = "_stt",
        native_retries,
        method = "stream",
        base_args(conn_options),
        state(start_time_offset),
        default(conn_options = "..types.DEFAULT_API_CONNECT_OPTIONS"),
        annotation(language = "str"),
        not_given
    )
))]
pub trait StreamAdapterWrapper {}
