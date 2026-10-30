//! `inference.STT`'s native client: the LiveKit Inference gateway's WebSocket, implemented by
//! `lk-agents-inference`.

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::errors::{ApiError, ProviderError};
use crate::inference::fallback::SttFallbackModel;
use crate::utils::stream::{StreamSink, StreamSource};
use crate::vad::types::Vad;
use crate::{ApiConnectOptions, AudioFrame, SpeechEvent, Stt};

/// The gateway STT behind `inference.STT`. Its options, keyterms and open streams are shared by
/// every view (`identified`); each view's connections carry its own job's identification. As the
/// session's STT it implements the keyterm and conversation hooks.
#[crate::host(class(name = "InferenceStt", native = Stt))]
pub trait InferenceStt: Stt + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(sink(item = AudioFrame, class = "InferenceSttStream"))]
    type Audio: InferenceSttAudio + 'static;
    #[host(source(item = SpeechEvent, class = "InferenceSttStream"))]
    type Events: StreamSource<SpeechEvent> + 'static;

    /// A client of `model` on the gateway at `base_url`. Starts nothing. `vad`'s end of speech
    /// finalizes the transcript (Speechmatics only).
    #[host(new)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &Self::Env,
        #[host(kw)] model: Option<String>,
        language: Option<String>,
        encoding: String,
        sample_rate: u32,
        base_url: String,
        api_key: String,
        api_secret: String,
        #[host(shape_error(
            kind = "ValueError",
            message = "expected a dict of extra kwargs, got {value}"
        ))]
        extra_kwargs: Map<String, Value>,
        #[host(default = None, shape_error(items, kind = "ValueError", message = "each fallback entry must be a dict"))]
        fallback: Option<Vec<SttFallbackModel>>,
        conn_options: ApiConnectOptions,
        #[host(default = None)] vad: Option<Arc<dyn Vad>>,
    ) -> Result<Self, ApiError>;

    /// The model's VAD as the host resolved it on `update_options(model=)`; `None` drops it.
    /// Later streams use it.
    #[host(call)]
    fn set_vad(&self, #[host(default = None)] vad: Option<Arc<dyn Vad>>);

    /// This client as a session of the job `headers` (`get_inference_headers()`) identifies holds
    /// it: the same options, keyterms and open streams, its connections carrying that job.
    /// `reporter` is the host `stt.STT` whose `error` events report this view's failures.
    #[host(call)]
    fn identified(
        &self,
        headers: Vec<(String, String)>,
        #[host(default = None)] reporter: Option<Arc<dyn Stt>>,
    ) -> Self;

    /// Read by later streams, and sent to the open ones as a mid-stream `session.update`.
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra: Option<Map<String, Value>>,
    );

    /// The session's keyterms: a warning once for a model without keyterm prompting, else every
    /// open stream applies them (now if the user is silent, at end of speech if not).
    #[host(call)]
    fn update_keyterms(&self, keyterms: Vec<String>) {
        Stt::update_keyterms(self, &keyterms);
    }

    /// An assistant turn's text, its tail, biases recognition as `agent_context` on a model that
    /// carries it.
    #[host(call)]
    fn push_conversation_item(&self, role: &str, text: &str) {
        Stt::push_conversation_item(self, role, text);
    }

    /// Opens a stream; its pump opens the socket. `start_time_offset` shifts its timestamps onto
    /// the caller's input timeline.
    #[host(call)]
    fn stream(
        &self,
        #[host(kw, default = None)] language: Option<String>,
        #[host(default = None)] start_time_offset: Option<f64>,
        #[host(default = None)] conn_options: Option<ApiConnectOptions>,
    ) -> Result<(Self::Audio, Self::Events), ProviderError>;
}

/// One live recognition's input: audio, and the options it is re-sent mid-stream.
#[crate::host(sink)]
pub trait InferenceSttAudio: StreamSink<AudioFrame> {
    /// A mid-stream `session.update`, for the providers that take one (AssemblyAI, Deepgram Flux).
    #[host(call)]
    fn update_options(
        &self,
        #[host(kw, default = None)] model: Option<String>,
        #[host(default = None)] language: Option<String>,
        #[host(default = None, shape_error(kind = "ValueError", message = "expected a dict of extra kwargs, got {value}"))]
        extra: Option<Map<String, Value>>,
    );
}
