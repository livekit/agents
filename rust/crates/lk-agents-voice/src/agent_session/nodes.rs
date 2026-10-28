//! An agent's nodes on a call. `llm_node`, `tts_node` and `stt_node`: the leg the driver performs
//! against when the agent overrides the node. Who it is (label, capabilities, rates) is the provider
//! below; every request opens the host node, which reaches the provider itself.
//! `transcription_node` and `realtime_audio_output_node`: [`OutputNodes`].

use super::api::NodeFactory;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::io::TextOutput;
use lk_agents_abi::{
    AudioFrame, ChatRequest, LanguageCode, Llm, LlmStream, SpeechEvent, StreamSource, Stt,
    SttCapabilities, SttSink, SttSource, Ts, Tts, TtsCapabilities, TtsSink, TtsSource,
};
use std::sync::Arc;

/// The nodes an agent overrides; a leg without one is the provider itself.
#[derive(Default, Clone)]
pub struct Nodes {
    pub llm: Option<Arc<dyn NodeFactory>>,
    pub tts: Option<Arc<dyn NodeFactory>>,
    pub stt: Option<Arc<dyn NodeFactory>>,
}

impl Nodes {
    pub fn llm(&self, below: Arc<dyn Llm>) -> Arc<dyn Llm> {
        match &self.llm {
            Some(host) => Arc::new(Node { below, host: Arc::clone(host) }),
            None => below,
        }
    }

    pub fn tts(&self, below: Arc<dyn Tts>) -> Arc<dyn Tts> {
        match &self.tts {
            Some(host) => Arc::new(Node { below, host: Arc::clone(host) }),
            None => below,
        }
    }

    pub fn stt(&self, below: Arc<dyn Stt>) -> Arc<dyn Stt> {
        match &self.stt {
            Some(host) => Arc::new(Node { below, host: Arc::clone(host) }),
            None => below,
        }
    }
}

/// The nodes an agent overrides on the way out, opened at each use from the agent a reply was
/// created under: `transcription_node` per spoken segment over the transcription output,
/// `realtime_audio_output_node` per realtime generation over the audio output.
#[derive(Default, Clone)]
pub struct OutputNodes {
    pub transcription: Option<Arc<dyn NodeFactory>>,
    pub realtime_audio: Option<Arc<dyn NodeFactory>>,
}

impl OutputNodes {
    /// The transcription output under the agent's `transcription_node`, opened for one segment.
    /// A node that fails to open drops the segment's transcript.
    pub fn text_output(&self, below: Option<Arc<dyn TextOutput>>) -> Option<Arc<dyn TextOutput>> {
        let below = below?;
        let Some(factory) = &self.transcription else { return Some(below) };
        match factory.text_node(Arc::clone(&below)) {
            Ok(node) => Some(Arc::new(crate::io::NodeTextOutput::new(node, below))),
            Err(e) => {
                log::error!(target: "livekit.agents", error = e.to_string().as_str();
                    "failed to open the transcription_node");
                None
            }
        }
    }
}

struct Node<P: ?Sized> {
    below: Arc<P>,
    host: Arc<dyn NodeFactory>,
}

/// The engine does not retry a node: the provider the node calls retries below it, and the node's
/// own code runs once per request, as in the reference.
fn final_error(error: ProviderError) -> ProviderError {
    error.into_final()
}

struct Final<T>(Box<dyn StreamSource<T>>);

impl<T: Send + 'static> StreamSource<T> for Final<T> {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move { self.0.next().await.map_err(final_error) })
    }

    fn cancel_safe(&self) -> bool {
        self.0.cancel_safe()
    }

    fn close(
        self: Box<Self>,
        deadline: Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        self.0.close(deadline)
    }

    fn started_time(&self) -> Option<f64> {
        self.0.started_time()
    }

    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<lk_agents_abi::metrics::base::AgentMetrics>> {
        self.0.take_call_metrics(in_flight)
    }

    fn call_ledger(&self) -> Option<lk_agents_abi::utils::stream::CallLedger> {
        self.0.call_ledger()
    }
}

fn duplex<I, T: Send + 'static>(
    opened: Result<
        (Box<dyn lk_agents_abi::StreamSink<I>>, Box<dyn StreamSource<T>>),
        ProviderError,
    >,
) -> Result<(Box<dyn lk_agents_abi::StreamSink<I>>, Box<dyn StreamSource<T>>), ProviderError> {
    let (sink, source) = opened.map_err(final_error)?;
    Ok((sink, Box::new(Final(source))))
}

impl Llm for Node<dyn Llm> {
    fn label(&self) -> String {
        self.below.label()
    }

    fn model(&self) -> String {
        self.below.model()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.below.id()
    }

    fn prewarm(&self) {
        self.below.prewarm();
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.below.failed(error, recoverable);
    }

    fn chat<'a, 'b, 'call>(
        &'a self,
        req: &'b ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'call, Result<LlmStream, ProviderError>>
    where
        'a: 'call,
        'b: 'call,
        Self: 'call,
    {
        Box::pin(async move {
            let stream = self.host.llm_node(req).await.map_err(final_error)?;
            Ok(Box::new(Final(stream)) as LlmStream)
        })
    }
}

impl Tts for Node<dyn Tts> {
    fn label(&self) -> String {
        self.below.label()
    }

    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, ..self.below.capabilities() }
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.below.id()
    }

    fn prewarm(&self) {
        self.below.prewarm();
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.below.failed(error, recoverable);
    }

    fn sample_rate(&self) -> u32 {
        self.below.sample_rate()
    }

    fn num_channels(&self) -> u16 {
        self.below.num_channels()
    }

    fn markup_provider(&self) -> String {
        self.below.markup_provider()
    }

    fn lowers_markup(&self) -> bool {
        self.below.lowers_markup()
    }

    /// Set on the TTS the node's default `tts_node` synthesises with.
    fn set_expressive(&self, on: bool) {
        self.below.set_expressive(on);
    }

    /// One text, through the node all the same.
    fn synthesize<'a, 'b, 'call>(
        &'a self,
        text: &'b str,
    ) -> lk_agents_abi::BoxFuture<'call, Result<TtsSource, ProviderError>>
    where
        'a: 'call,
        'b: 'call,
        Self: 'call,
    {
        Box::pin(async move {
            let (sink, source) = self.stream().await?;
            sink.push(text.to_owned()).map_err(final_error)?;
            sink.end_input();
            Ok(source)
        })
    }

    fn stream<'a, 'call>(
        &'a self,
    ) -> lk_agents_abi::BoxFuture<'call, Result<(TtsSink, TtsSource), ProviderError>>
    where
        'a: 'call,
        Self: 'call,
    {
        Box::pin(async move { duplex(self.host.tts_node().await) })
    }
}

impl Stt for Node<dyn Stt> {
    fn label(&self) -> String {
        self.below.label()
    }

    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..self.below.capabilities() }
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        self.below.id()
    }

    fn prewarm(&self) {
        self.below.prewarm();
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.below.failed(error, recoverable);
    }

    fn update_keyterms(&self, keyterms: &[String]) {
        self.below.update_keyterms(keyterms);
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        self.below.push_conversation_item(role, text);
    }

    /// `stt_node` is a stream; a batch recognition is the provider's.
    fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        self.below.recognize(buffer, language)
    }

    fn stream(
        &self,
        _language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>> {
        Box::pin(async move { duplex(self.host.stt_node().await).map(Some) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::errors::{ApiError, HostError};

    struct Below;

    impl Llm for Below {
        fn label(&self) -> String {
            "below".to_owned()
        }

        fn chat<'a, 'b, 'call>(
            &'a self,
            _req: &'b ChatRequest,
        ) -> lk_agents_abi::BoxFuture<'call, Result<LlmStream, ProviderError>>
        where
            'a: 'call,
            'b: 'call,
            Self: 'call,
        {
            unreachable!("the node makes the request")
        }
    }

    /// A node whose provider gave up after its own retries.
    struct GaveUp;

    impl NodeFactory for GaveUp {
        fn text_node(
            &self,
            _below: Arc<dyn TextOutput>,
        ) -> Result<Arc<dyn crate::io::TextNode>, HostError> {
            unreachable!()
        }

        fn audio_node(
            &self,
            _below: Arc<dyn lk_agents_abi::voice::io::AudioOutput>,
        ) -> Result<Arc<dyn crate::io::RealtimeAudioNode>, HostError> {
            unreachable!()
        }

        fn llm_node<'a>(
            &'a self,
            _req: &'a ChatRequest,
        ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
            Box::pin(async move { Err(ApiError::new("down").into()) })
        }

        fn tts_node(
            &self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
            Box::pin(async move { Err(ApiError::new("down").into()) })
        }

        fn stt_node(
            &self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(SttSink, SttSource), ProviderError>> {
            Box::pin(async move { Err(ApiError::new("down").into()) })
        }
    }

    // Catches: a node the engine's attempt loop retries (the node's code would run again on top
    // of the retries its provider already made), or a node that hides its provider from metrics.
    #[tokio::test]
    async fn a_node_speaks_as_its_provider_and_its_failure_is_final() {
        let nodes = Nodes { llm: Some(Arc::new(GaveUp)), ..Nodes::default() };
        let llm = nodes.llm(Arc::new(Below));
        assert_eq!(llm.label(), "below");
        let error = llm.chat(&ChatRequest::default()).await.err().expect("the node failed");
        assert!(!error.retryable(), "the engine would run the node again");
    }
}
