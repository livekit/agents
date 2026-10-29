//! `stt.StreamAdapter` (`stt/stream_adapter.py`): a recognize-only STT as a streaming one, the
//! VAD cutting the audio into utterances each recognized once it ends. Implemented by
//! `lk-agents-inference`.

use std::sync::Arc;

use crate::errors::ProviderError;
use crate::tokenize::language::LanguageCode;
use crate::utils::stream::{StreamSink, StreamSource};
use crate::vad::vad::Vad;
use crate::{ApiConnectOptions, AudioFrame, SpeechEvent, Stt};

/// `stt.StreamAdapter`'s engine as the host builds it; the public class stays Python.
#[crate::host(class(name = "SttStreamAdapter", native = Stt))]
pub trait SttStreamAdapter: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(sink(item = AudioFrame, class = "SttAdapterStream"))]
    type Audio: StreamSink<AudioFrame> + 'static;
    #[host(source(item = SpeechEvent, class = "SttAdapterStream"))]
    type Events: StreamSource<SpeechEvent> + 'static;

    /// The wrapped STT and VAD; each stream binds its own recognition policy.
    #[host(new)]
    fn new(env: &Self::Env, stt: Arc<dyn Stt>, vad: Arc<dyn Vad>) -> Self;

    /// `_recognize_impl`: the wrapped STT's batch path.
    #[host(await)]
    async fn recognize(
        &self,
        frames: Vec<AudioFrame>,
        #[host(default = None)] language: Option<LanguageCode>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<SpeechEvent, ProviderError>;

    /// Audio into the VAD, its utterances recognized.
    #[host(await)]
    async fn stream(
        &self,
        #[host(default = None)] language: Option<LanguageCode>,
        #[host(default = None)] start_time_offset: Option<f64>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<(Self::Audio, Self::Events), ProviderError>;
}
