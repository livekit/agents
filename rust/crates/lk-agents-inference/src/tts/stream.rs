//! One native synthesis operation as the host sees it: the gateway and fallback stream views.

use std::sync::Arc;

use lk_agents_abi::errors::{HostError, ProviderError, PullError};
use lk_agents_abi::tts::streams::{FallbackTtsStream, InferenceTtsStream};
use lk_agents_abi::{Closed, StreamSink, SynthesizedAudio, Ts, TtsSink, TtsSource};

use lk_agents_net::source::{NativeSource, Pulled};

pub struct TtsStream {
    sink: Option<Arc<dyn StreamSink<String>>>,
    source: NativeSource<SynthesizedAudio>,
}

impl TtsStream {
    pub fn new(sink: Option<TtsSink>, source: TtsSource) -> Self {
        Self { sink: sink.map(Arc::from), source: NativeSource::new(source) }
    }

    fn sink(&self) -> Result<&Arc<dyn StreamSink<String>>, ProviderError> {
        self.sink.as_ref().ok_or_else(|| ProviderError::Host {
            error: HostError {
                kind: "RuntimeError".into(),
                message: "this is a synthesize() stream; its text was given at the call".into(),
                host_ref: None,
            },
            api_kind: None,
            retryable: false,
            status_code: None,
        })
    }

    fn send_text(&self, text: String) -> Result<(), ProviderError> {
        self.sink()?.push(text)
    }

    fn flush_input(&self) -> Result<(), ProviderError> {
        self.sink()?.flush()
    }

    fn finish_input(&self) -> Result<(), ProviderError> {
        self.sink()?.end_input();
        Ok(())
    }

    async fn receive(&self) -> Result<Option<SynthesizedAudio>, PullError> {
        self.source.pull().await.map(|pulled| match pulled {
            Pulled::Item(audio) => Some(audio),
            Pulled::End | Pulled::Cancelled => None,
        })
    }

    async fn close(&self, end_input: bool) {
        if end_input && let Some(sink) = &self.sink {
            sink.end_input();
        }
        if self.source.close(Ts::from_nanos(u64::MAX)).await == Closed::TimedOut {
            log::error!(target: "livekit.agents", "a native stream did not release its resources in time");
        }
    }
}

impl InferenceTtsStream for TtsStream {
    fn push(&self, text: String) -> Result<(), ProviderError> {
        self.send_text(text)
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.flush_input()
    }
    fn end_input(&self) -> Result<(), ProviderError> {
        self.finish_input()
    }
    async fn recv(&self) -> Result<Option<SynthesizedAudio>, PullError> {
        self.receive().await
    }
    async fn aclose(&self) {
        self.close(true).await;
    }
}

impl FallbackTtsStream for TtsStream {
    fn push_text(&self, token: String) -> Result<(), ProviderError> {
        self.send_text(token)
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.flush_input()
    }
    fn end_input(&self) -> Result<(), ProviderError> {
        self.finish_input()
    }
    async fn recv(&self) -> Result<Option<SynthesizedAudio>, PullError> {
        self.receive().await
    }
    async fn aclose(&self) {
        self.close(false).await;
    }
}
