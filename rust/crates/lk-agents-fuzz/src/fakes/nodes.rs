//! `Agent.llm_node`, `tts_node` and `stt_node` overridden as the defaults are: each request
//! opens the provider below, as `Agent.default.llm_node` does; `transcription_node` and
//! `realtime_audio_output_node` pass what they read down the leg below. What the provider then does is the
//! case's, so the node path is exercised with every answer the plain one gets. Each opening is
//! recorded with the agent whose override it is: a request at its call, a transcript segment or a
//! realtime generation at its first input, which is where the host's node is invoked.
use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, HostError, ProviderError};
use lk_agents_abi::voice::io::{AudioOutput, AudioOutputExt as _, TextOutput};
use lk_agents_abi::{
    AudioFrame, ChatRequest, ChoiceDelta, Closed, Llm, LlmChunk, LlmStream, StreamSource, Stt,
    SttSink, SttSource, TimedString, Ts, Tts, TtsSink, TtsSource,
};
use lk_agents_voice::agent_session::api::NodeFactory;
use lk_agents_voice::io::{RealtimeAudioNode, TextNode};

use crate::record::NodeUse;

pub struct FakeNodes {
    registry: Arc<super::Registry>,
    /// The agent whose overrides these are.
    agent: String,
    llm: Option<Arc<dyn Llm>>,
    tts: Option<Arc<dyn Tts>>,
    stt: Option<Arc<dyn Stt>>,
    /// What the `llm_node` answers itself instead of calling the model.
    llm_text: Option<&'static str>,
}

impl FakeNodes {
    pub fn new(
        registry: Arc<super::Registry>,
        agent: &str,
        llm: Option<Arc<dyn Llm>>,
        tts: Option<Arc<dyn Tts>>,
        stt: Option<Arc<dyn Stt>>,
        llm_text: Option<&'static str>,
    ) -> Arc<Self> {
        Arc::new(FakeNodes { registry, agent: agent.to_owned(), llm, tts, stt, llm_text })
    }

    fn used(&self, node: NodeUse) {
        self.registry.node_used(&self.agent, node);
    }

    /// The node's invocation as the host makes it: recorded at its first input.
    fn invocation(&self, node: NodeUse) -> Invocation {
        Invocation {
            registry: self.registry.clone(),
            agent: self.agent.clone(),
            node,
            open: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

/// A node's own answer: one `str`, as the host glue hands it over.
struct NodeText(Option<&'static str>);

impl StreamSource<LlmChunk> for NodeText {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<LlmChunk>, ProviderError>> {
        Box::pin(async move {
            Ok(self.0.take().map(|text| LlmChunk {
                id: "node".into(),
                delta: Some(ChoiceDelta { content: Some(text.into()), ..ChoiceDelta::default() }),
                node_text: true,
                ..LlmChunk::default()
            }))
        })
    }

    fn close(self: Box<Self>, _deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

/// One invocation of a host node at a time: recorded as it opens, at its first input.
struct Invocation {
    registry: Arc<super::Registry>,
    agent: String,
    node: NodeUse,
    open: std::sync::atomic::AtomicBool,
}

impl Invocation {
    fn input(&self) {
        if !self.open.swap(true, std::sync::atomic::Ordering::SeqCst) {
            self.registry.node_used(&self.agent, self.node);
        }
    }

    fn end(&self) {
        self.open.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

/// What the host's text node hands its drain: a delta, or the segment's end.
struct EchoText(
    tokio::sync::mpsc::UnboundedSender<Option<TimedString>>,
    Invocation,
    tokio::sync::watch::Sender<Option<lk_agents_abi::voice::io::Result<()>>>,
    tokio::sync::watch::Sender<bool>,
    Arc<std::sync::Mutex<Option<Box<dyn Send>>>>,
);

impl TextNode for EchoText {
    fn capture_text(&self, text: &str) {
        self.1.input();
        let _ = self.0.send(Some(TimedString::new(text)));
    }

    fn capture_timed(&self, text: &TimedString) {
        self.1.input();
        let _ = self.0.send(Some(text.clone()));
    }

    fn retain_until_drained(&self, owner: Box<dyn Send>) {
        let mut retained = self.4.lock().unwrap();
        if self.2.borrow().is_none() {
            assert!(retained.is_none(), "a segment has one binding owner");
            *retained = Some(owner);
        } else {
            drop(retained);
            drop(owner);
        }
    }

    fn cancel(&self) {
        self.3.send_replace(true);
    }

    fn drain(&self) -> lk_agents_abi::BoxFuture<'_, lk_agents_abi::voice::io::Result<()>> {
        let mut completion = self.2.subscribe();
        Box::pin(async move {
            let result = completion.wait_for(Option::is_some).await.map_err(|_| {
                lk_agents_abi::voice::io::IoError::Rtc("text node completion was lost".into())
            })?;
            result.as_ref().expect("completed node").clone()
        })
    }

    fn flush(&self) {
        self.1.end();
        let _ = self.0.send(None);
    }
}

/// Its drain, and whether it was cut: from then on no frame goes down, not even one already
/// queued.
struct EchoAudio(
    tokio::sync::mpsc::UnboundedSender<AudioFrame>,
    Arc<std::sync::atomic::AtomicBool>,
    Invocation,
);

impl RealtimeAudioNode for EchoAudio {
    fn push<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, ()> {
        Box::pin(async move {
            self.2.input();
            let _ = self.0.send(frame.clone());
        })
    }

    fn clear(&self) {
        self.1.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn end(&self) {}
}

fn missing(leg: &str) -> ProviderError {
    ApiError::new(format!("no {leg} below the node")).into()
}

impl NodeFactory for FakeNodes {
    /// `transcription_node` as the default: each delta down the leg below, in order, and the
    /// segment's flush after its last one.
    fn text_node(&self, below: Arc<dyn TextOutput>) -> Result<Arc<dyn TextNode>, HostError> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Option<TimedString>>();
        let completion = tokio::sync::watch::channel(None).0;
        let done = completion.clone();
        let (cancel, mut cancelled) = tokio::sync::watch::channel(false);
        let retained = Arc::new(std::sync::Mutex::new(None::<Box<dyn Send>>));
        let released = retained.clone();
        self.registry.world.spawn(Box::pin(async move {
            let outcome = loop {
                let op = tokio::select! {
                    biased;
                    _ = async { let _ = cancelled.wait_for(|value| *value).await; } => {
                        below.cancel_flush();
                        break below.wait_for_flush().await;
                    }
                    op = rx.recv() => op,
                };
                let Some(op) = op else {
                    break Err(lk_agents_abi::voice::io::IoError::Rtc(
                        "text node input was cancelled".into(),
                    ));
                };
                let sent = match op {
                    Some(t) if t.start_time.is_none() && t.end_time.is_none() => {
                        below.capture_text(&t.text).await
                    }
                    Some(t) => below.capture_timed(&t).await,
                    None => {
                        below.flush();
                        break below.wait_for_flush().await;
                    }
                };
                if let Err(error) = sent {
                    break Err(error);
                }
            };
            let owner = {
                let mut owner = released.lock().unwrap();
                done.send_replace(Some(outcome));
                owner.take()
            };
            drop(owner);
        }));
        Ok(Arc::new(EchoText(
            tx,
            self.invocation(NodeUse::Transcription),
            completion,
            cancel,
            retained,
        )))
    }

    /// `realtime_audio_output_node` as the default: each frame to the leg below, in order, none
    /// once cut.
    fn audio_node(
        &self,
        below: Arc<dyn AudioOutput>,
    ) -> Result<Arc<dyn RealtimeAudioNode>, HostError> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AudioFrame>();
        let cut = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cleared = Arc::clone(&cut);
        self.registry.world.spawn(Box::pin(async move {
            while let Some(frame) = rx.recv().await {
                if cleared.load(std::sync::atomic::Ordering::SeqCst)
                    || below.capture_frame(&frame).await.is_err()
                {
                    break;
                }
            }
        }));
        Ok(Arc::new(EchoAudio(tx, cut, self.invocation(NodeUse::RealtimeAudio))))
    }

    fn llm_node<'a>(
        &'a self,
        req: &'a ChatRequest,
    ) -> lk_agents_abi::BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(async move {
            self.used(NodeUse::Llm);
            if let Some(text) = self.llm_text {
                return Ok(Box::new(NodeText(Some(text))) as LlmStream);
            }
            self.llm.as_ref().ok_or_else(|| missing("LLM"))?.chat(req).await
        })
    }

    fn tts_node(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(TtsSink, TtsSource), ProviderError>> {
        Box::pin(async move {
            self.used(NodeUse::Tts);
            self.tts.as_ref().ok_or_else(|| missing("TTS"))?.stream().await
        })
    }

    fn stt_node(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(SttSink, SttSource), ProviderError>> {
        Box::pin(async move {
            self.used(NodeUse::Stt);
            let stt = self.stt.as_ref().ok_or_else(|| missing("STT"))?;
            stt.stream(None, None).await?.ok_or_else(|| missing("streaming STT"))
        })
    }
}
