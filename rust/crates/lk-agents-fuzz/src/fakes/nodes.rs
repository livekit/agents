//! `Agent.llm_node`, `tts_node` and `stt_node` overridden as the defaults are: each request
//! opens the provider below, as `Agent.default.llm_node` does. What the provider then does is the
//! case's, so the node path is exercised with every answer the plain one gets.
use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, HostError, ProviderError};
use lk_agents_abi::{
    ChatRequest, ChoiceDelta, Closed, Llm, LlmChunk, LlmStream, StreamSource, Stt, SttSink,
    SttSource, Ts, Tts, TtsSink, TtsSource,
};
use lk_agents_voice::agent_session::api::NodeFactory;
use lk_agents_voice::io::{CoreAudioSink, CoreTextOutput, RealtimeAudioNode, TextNode};

pub struct FakeNodes {
    llm: Option<Arc<dyn Llm>>,
    tts: Option<Arc<dyn Tts>>,
    stt: Option<Arc<dyn Stt>>,
    /// What the `llm_node` answers itself instead of calling the model.
    llm_text: Option<&'static str>,
}

impl FakeNodes {
    pub fn new(
        llm: Option<Arc<dyn Llm>>,
        tts: Option<Arc<dyn Tts>>,
        stt: Option<Arc<dyn Stt>>,
        llm_text: Option<&'static str>,
    ) -> Arc<Self> {
        Arc::new(FakeNodes { llm, tts, stt, llm_text })
    }
}

/// A node's own answer: one `str`, as the host glue hands it over.
struct NodeText(Option<&'static str>);

#[async_trait::async_trait]
impl StreamSource<LlmChunk> for NodeText {
    async fn next(&mut self) -> Result<Option<LlmChunk>, ProviderError> {
        Ok(self.0.take().map(|text| LlmChunk {
            id: "node".into(),
            delta: Some(ChoiceDelta { content: Some(text.into()), ..ChoiceDelta::default() }),
            node_text: true,
            ..LlmChunk::default()
        }))
    }

    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        Closed::Clean
    }
}

fn missing(leg: &str) -> ProviderError {
    ApiError::new(format!("no {leg} below the node")).into()
}

#[async_trait::async_trait]
impl NodeFactory for FakeNodes {
    fn text_node(&self, _below: CoreTextOutput) -> Result<Arc<dyn TextNode>, HostError> {
        Err(HostError::new("not a transcription node"))
    }

    fn audio_node(&self, _below: CoreAudioSink) -> Result<Arc<dyn RealtimeAudioNode>, HostError> {
        Err(HostError::new("not a realtime audio node"))
    }

    async fn llm_node(&self, req: &ChatRequest) -> Result<LlmStream, ProviderError> {
        if let Some(text) = self.llm_text {
            return Ok(Box::new(NodeText(Some(text))));
        }
        self.llm.as_ref().ok_or_else(|| missing("LLM"))?.chat(req).await
    }

    async fn tts_node(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        self.tts.as_ref().ok_or_else(|| missing("TTS"))?.stream().await
    }

    async fn stt_node(&self) -> Result<(SttSink, SttSource), ProviderError> {
        let stt = self.stt.as_ref().ok_or_else(|| missing("STT"))?;
        stt.stream(None, None).await.unwrap_or_else(|| Err(missing("streaming STT")))
    }
}
