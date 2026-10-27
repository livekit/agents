//! `Agent.llm_node`, `tts_node` and `stt_node` overridden as the defaults are: each request
//! opens the provider below, as `Agent.default.llm_node` does; `transcription_node` and
//! `realtime_audio_output_node` pass what they read down the leg below. What the provider then does is the
//! case's, so the node path is exercised with every answer the plain one gets.
use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, HostError, ProviderError};
use lk_agents_abi::{
    AudioFrame, ChatRequest, ChoiceDelta, Closed, Llm, LlmChunk, LlmStream, StreamSource, Stt,
    SttSink, SttSource, TimedString, Ts, Tts, TtsSink, TtsSource,
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

/// What the host's text node hands its drain: a delta, or the segment's end.
struct EchoText(tokio::sync::mpsc::UnboundedSender<Option<TimedString>>);

impl TextNode for EchoText {
    fn capture_text(&self, text: &str) {
        let _ = self.0.send(Some(TimedString::new(text)));
    }

    fn capture_timed(&self, text: &TimedString) {
        let _ = self.0.send(Some(text.clone()));
    }

    fn flush(&self) {
        let _ = self.0.send(None);
    }
}

/// Its drain, and whether it was cut: from then on no frame goes down, not even one already
/// queued.
struct EchoAudio(tokio::sync::mpsc::UnboundedSender<AudioFrame>, Arc<std::sync::atomic::AtomicBool>);

#[async_trait::async_trait]
impl RealtimeAudioNode for EchoAudio {
    async fn push(&self, frame: &AudioFrame) {
        let _ = self.0.send(frame.clone());
    }

    fn clear(&self) {
        self.1.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn end(&self) {}
}

fn missing(leg: &str) -> ProviderError {
    ApiError::new(format!("no {leg} below the node")).into()
}

#[async_trait::async_trait]
impl NodeFactory for FakeNodes {
    /// `transcription_node` as the default: each delta down the leg below, in order, and the
    /// segment's flush after its last one.
    fn text_node(&self, below: CoreTextOutput) -> Result<Arc<dyn TextNode>, HostError> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Option<TimedString>>();
        tokio::spawn(async move {
            while let Some(op) = rx.recv().await {
                let sent = match op {
                    Some(t) if t.start_time.is_none() && t.end_time.is_none() => {
                        below.capture_text(t.text).await
                    }
                    Some(t) => below.capture_timed(t.text, t.start_time, t.end_time).await,
                    None => {
                        below.flush();
                        Ok(())
                    }
                };
                if sent.is_err() {
                    break;
                }
            }
        });
        Ok(Arc::new(EchoText(tx)))
    }

    /// `realtime_audio_output_node` as the default: each frame to the leg below, in order, none
    /// once cut.
    fn audio_node(&self, below: CoreAudioSink) -> Result<Arc<dyn RealtimeAudioNode>, HostError> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AudioFrame>();
        let cut = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cleared = Arc::clone(&cut);
        tokio::spawn(async move {
            while let Some(frame) = rx.recv().await {
                if cleared.load(std::sync::atomic::Ordering::SeqCst)
                    || below.capture_frame(frame).await.is_err()
                {
                    break;
                }
            }
        });
        Ok(Arc::new(EchoAudio(tx, cut)))
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
