//! `Agent.llm_node`, `tts_node` and `stt_node` overridden as the defaults are: each request
//! opens the provider below, as `Agent.default.llm_node` does. What the provider then does is the
//! case's, so the node path is exercised with every answer the plain one gets.
use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, HostError, ProviderError};
use lk_agents_abi::{ChatRequest, Llm, LlmStream, Stt, SttSink, SttSource, Tts, TtsSink, TtsSource};
use lk_agents_voice::agent_session::api::NodeFactory;
use lk_agents_voice::io::{CoreAudioSink, CoreTextOutput, RealtimeAudioNode, TextNode};

pub struct FakeNodes {
    llm: Option<Arc<dyn Llm>>,
    tts: Option<Arc<dyn Tts>>,
    stt: Option<Arc<dyn Stt>>,
}

impl FakeNodes {
    pub fn new(
        llm: Option<Arc<dyn Llm>>,
        tts: Option<Arc<dyn Tts>>,
        stt: Option<Arc<dyn Stt>>,
    ) -> Arc<Self> {
        Arc::new(FakeNodes { llm, tts, stt })
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
