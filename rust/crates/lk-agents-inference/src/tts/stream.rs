//! One native synthesis operation as the host sees it, for the gateway and fallback paths alike.

use std::sync::Arc;

use lk_agents_abi::errors::{HostError, ProviderError, PullError};
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
}

impl lk_agents_abi::tts::streams::TtsStream for TtsStream {
    fn push_text(&self, text: String) -> Result<(), ProviderError> {
        self.sink()?.push(text)
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.sink()?.flush()
    }
    fn end_input(&self) -> Result<(), ProviderError> {
        self.sink()?.end_input();
        Ok(())
    }
    async fn recv(&self) -> Result<Option<SynthesizedAudio>, PullError> {
        self.source.pull().await.map(|pulled| match pulled {
            Pulled::Item(audio) => Some(audio),
            Pulled::End | Pulled::Cancelled => None,
        })
    }
    /// Cancels the synthesis; the input is not ended, so nothing pending is spoken.
    async fn aclose(&self) {
        if self.source.close(Ts::from_nanos(u64::MAX)).await == Closed::TimedOut {
            log::error!(target: "livekit.agents", "a native stream did not release its resources in time");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::tts::streams::TtsStream as _;
    use lk_agents_abi::{StreamSource, TtsSource};
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct Input(Arc<Mutex<Vec<&'static str>>>);
    impl StreamSink<String> for Input {
        fn push(&self, _: String) -> Result<(), ProviderError> {
            self.0.lock().unwrap().push("text");
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            self.0.lock().unwrap().push("flush");
            Ok(())
        }
        fn end_input(&self) {
            self.0.lock().unwrap().push("end_input");
        }
    }

    struct Silent;
    impl StreamSource<SynthesizedAudio> for Silent {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
            Box::pin(std::future::pending())
        }
        fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
            Box::pin(async { Closed::Clean })
        }
    }

    #[tokio::test]
    async fn aclose_cancels_without_ending_the_input() {
        let input = Input::default();
        let source: TtsSource = Box::new(Silent);
        let stream = TtsStream::new(Some(Box::new(input.clone())), source);
        stream.push_text("hello".into()).unwrap();
        stream.aclose().await;
        assert_eq!(*input.0.lock().unwrap(), ["text"], "aclose must not speak pending text");
    }
}
