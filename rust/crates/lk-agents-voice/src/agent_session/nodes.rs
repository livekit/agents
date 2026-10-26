//! `Agent.llm_node`, `tts_node` and `stt_node` on a call: the leg the driver performs against when
//! the agent overrides the node. Who it is (label, capabilities, rates) is the provider below; every
//! request opens the host node, which reaches the provider itself.

use super::api::NodeFactory;
use async_trait::async_trait;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{
    AudioFrame, ChatRequest, LanguageCode, Llm, LlmStream, SpeechEvent, StreamSource, Stt,
    SttCapabilities, SttSink, SttSource, Tts, TtsCapabilities, TtsSink, TtsSource, Ts,
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

struct Node<P: ?Sized> {
    below: Arc<P>,
    host: Arc<dyn NodeFactory>,
}

/// The engine does not retry a node: the provider the node calls retries below it, and the node's
/// own code runs once per request, as in the reference.
fn final_error(error: ProviderError) -> ProviderError {
    match error {
        ProviderError::Host { error, api_kind, status_code, .. } => {
            ProviderError::Host { error, api_kind, retryable: false, status_code }
        }
        ProviderError::Local(error) => ProviderError::Local(lk_agents_abi::errors::ApiError {
            retryable: false,
            ..error
        }),
    }
}

struct Final<T>(Box<dyn StreamSource<T>>);

#[async_trait]
impl<T: Send> StreamSource<T> for Final<T> {
    async fn next(&mut self) -> Result<Option<T>, ProviderError> {
        self.0.next().await.map_err(final_error)
    }

    fn cancel_safe(&self) -> bool {
        self.0.cancel_safe()
    }

    async fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::Closed {
        self.0.close(deadline).await
    }

    fn started_time(&self) -> Option<f64> {
        self.0.started_time()
    }
}

fn duplex<I, T: Send + 'static>(
    opened: Result<(Box<dyn lk_agents_abi::StreamSink<I>>, Box<dyn StreamSource<T>>), ProviderError>,
) -> Result<(Box<dyn lk_agents_abi::StreamSink<I>>, Box<dyn StreamSource<T>>), ProviderError> {
    let (sink, source) = opened.map_err(final_error)?;
    Ok((sink, Box::new(Final(source))))
}

#[async_trait]
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

    async fn chat(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        let stream = self.host.llm_node(req).await.map_err(final_error)?;
        Ok(Box::new(Final(stream)))
    }
}

#[async_trait]
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
    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
        let (sink, source) = self.stream().await?;
        sink.push(text.to_owned()).map_err(final_error)?;
        sink.end_input();
        Ok(source)
    }

    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        duplex(self.host.tts_node().await)
    }
}

#[async_trait]
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

    fn update_keyterms(&self, keyterms: &[String]) {
        self.below.update_keyterms(keyterms);
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        self.below.push_conversation_item(role, text);
    }

    /// `stt_node` is a stream; a batch recognition is the provider's.
    async fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        self.below.recognize(buffer, language).await
    }

    async fn stream(
        &self,
        _language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> Option<Result<(SttSink, SttSource), ProviderError>> {
        Some(duplex(self.host.stt_node().await))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::errors::{ApiError, HostError};

    struct Below;

    #[async_trait]
    impl Llm for Below {
        fn label(&self) -> String {
            "below".to_owned()
        }

        async fn chat(&self, _req: &ChatRequest) -> Result<LlmStream, ProviderError> {
            unreachable!("the node makes the request")
        }
    }

    /// A node whose provider gave up after its own retries.
    struct GaveUp;

    #[async_trait]
    impl NodeFactory for GaveUp {
        fn text_node(
            &self,
            _below: crate::io::CoreTextOutput,
        ) -> Result<Arc<dyn crate::io::TextNode>, HostError> {
            unreachable!()
        }

        fn audio_node(
            &self,
            _below: crate::io::CoreAudioSink,
        ) -> Result<Arc<dyn crate::io::RealtimeAudioNode>, HostError> {
            unreachable!()
        }

        async fn llm_node(&self, _req: &ChatRequest) -> Result<LlmStream, ProviderError> {
            Err(ApiError::new("down").into())
        }

        async fn tts_node(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
            Err(ApiError::new("down").into())
        }

        async fn stt_node(&self) -> Result<(SttSink, SttSource), ProviderError> {
            Err(ApiError::new("down").into())
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
