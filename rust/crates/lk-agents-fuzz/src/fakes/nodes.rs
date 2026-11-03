//! `Agent.llm_node`, `tts_node` and `stt_node` overridden as the defaults are: each request
//! opens the provider below, as `Agent.default.llm_node` does; `transcription_node`
//! passes what it reads down the leg below. What the provider then does is the
//! case's, so the node path is exercised with every answer the plain one gets. Each opening is
//! recorded with the agent whose override it is: a request at its call, a transcript segment
//! at its first input, which is where the host's node is invoked.
use std::sync::Arc;

use lk_agents_abi::SpawnerExt as _;
use lk_agents_abi::errors::{ApiError, HostError, ProviderError};
use lk_agents_abi::voice::io::TextOutput;
use lk_agents_abi::voice::nodes::NodeFactory;
use lk_agents_abi::voice::nodes::TextNode;
use lk_agents_abi::{
    ChatRequest, ChoiceDelta, Closed, Llm, LlmChunk, LlmStream, StreamOutput, Stt, SttSink,
    SttSource, TranscriptChunk, Ts, Tts, TtsSink, TtsSource,
};

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

impl StreamOutput<LlmChunk> for NodeText {
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

impl lk_agents_abi::utils::stream::ProviderStream<LlmChunk> for NodeText {}

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
    tokio::sync::mpsc::UnboundedSender<Option<TranscriptChunk>>,
    Invocation,
    tokio::sync::watch::Sender<Option<lk_agents_abi::voice::io::Result<()>>>,
    tokio::sync::watch::Sender<bool>,
    Arc<std::sync::Mutex<Option<Box<dyn Send>>>>,
);

impl TextNode for EchoText {
    fn capture_text(&self, text: &str) {
        self.1.input();
        let _ = self.0.send(Some(TranscriptChunk::new(text)));
    }

    fn capture_timed(&self, text: &TranscriptChunk) {
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

fn missing(leg: &str) -> ProviderError {
    ApiError::new(format!("no {leg} below the node")).into()
}

impl NodeFactory for FakeNodes {
    /// `transcription_node` as the default: each delta down the leg below, in order, and the
    /// segment's flush after its last one.
    fn text_node(&self, below: Arc<dyn TextOutput>) -> Result<Arc<dyn TextNode>, HostError> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Option<TranscriptChunk>>();
        let completion = tokio::sync::watch::channel(None).0;
        let done = completion.clone();
        let (cancel, mut cancelled) = tokio::sync::watch::channel(false);
        let retained = Arc::new(std::sync::Mutex::new(None::<Box<dyn Send>>));
        let released = retained.clone();
        // The world's own work: the run's executor ends it with the run.
        self.registry.world.spawn_detached(async move {
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
        });
        Ok(Arc::new(EchoText(
            tx,
            self.invocation(NodeUse::Transcription),
            completion,
            cancel,
            retained,
        )))
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
        expressive: bool,
        _trace: lk_agents_abi::telemetry::record::TraceContext,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>,
    > {
        Box::pin(async move {
            self.used(NodeUse::Tts);
            self.tts
                .as_ref()
                .ok_or_else(|| missing("TTS"))?
                .stream(expressive, Default::default())
                .await
        })
    }

    fn stt_node(
        &self,
        trace: lk_agents_abi::telemetry::record::TraceContext,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<SttSink, SttSource>, ProviderError>,
    > {
        Box::pin(async move {
            self.used(NodeUse::Stt);
            let stt = self.stt.as_ref().ok_or_else(|| missing("STT"))?;
            stt.stream(None, None, trace).await?.ok_or_else(|| missing("streaming STT"))
        })
    }
}
