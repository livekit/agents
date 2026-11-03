//! One native synthesis operation as the host sees it, for the gateway and fallback paths alike.

use lk_agents_abi::errors::{HostError, ProviderError, PullError};
use lk_agents_abi::utils::strings::PYTHON_WHITESPACE;
use lk_agents_abi::{Closed, StreamInput, SynthesizedAudio, TtsSink, TtsSource};
use lk_agents_net::source::{Pulled, SharedStream};
use std::sync::Arc;

pub struct TtsStream {
    sink: Option<Arc<dyn StreamInput<String>>>,
    source: SharedStream<SynthesizedAudio>,
}

impl TtsStream {
    pub fn new(
        sink: Option<TtsSink>,
        source: TtsSource,
        clock: Arc<dyn lk_agents_abi::Clock>,
    ) -> Self {
        Self { sink: sink.map(Arc::from), source: SharedStream::new(source, clock) }
    }

    fn sink(&self) -> Result<&Arc<dyn StreamInput<String>>, ProviderError> {
        if self.source.is_closed() {
            return Err(lk_agents_abi::errors::ApiError::new("stream input is closed")
                .with_retryable(false)
                .into());
        }
        self.sink.as_ref().ok_or_else(|| ProviderError::Host {
            error: HostError {
                kind: "RuntimeError".into(),
                message: "this is a synthesize() stream; its text was given at the call".into(),
                host_ref: None,
                traceback: None,
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
        if self.source.aclose().await == Closed::TimedOut {
            log::error!(target: "livekit.agents", "a native stream did not release its resources in time");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::tts::streams::TtsStream as _;
    use lk_agents_abi::{StreamOutput, Ts, TtsSource};
    use parking_lot::Mutex;

    #[derive(Clone, Default)]
    struct Input(Arc<Mutex<Vec<&'static str>>>);
    impl StreamInput<String> for Input {
        fn push(&self, _: String) -> Result<(), ProviderError> {
            self.0.lock().push("text");
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            self.0.lock().push("flush");
            Ok(())
        }
        fn end_input(&self) {
            self.0.lock().push("end_input");
        }
    }

    struct Silent;
    impl StreamOutput<SynthesizedAudio> for Silent {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
            Box::pin(std::future::pending())
        }
        fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
            Box::pin(async { Closed::Clean })
        }
    }

    impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for Silent {}

    #[tokio::test]
    async fn aclose_cancels_without_ending_the_input() {
        let input = Input::default();
        let source: TtsSource = Box::new(Silent);
        let stream = TtsStream::new(
            Some(Box::new(input.clone())),
            source,
            Arc::new(lk_agents_net::clock::RealClock::new()),
        );
        stream.push_text("hello".into()).unwrap();
        stream.aclose().await;
        assert!(stream.push_text("late".into()).is_err());
        assert_eq!(*input.0.lock(), ["text"], "aclose must not speak pending text");
    }
}

pub struct SynthesisChecks;

impl lk_agents_abi::tts::streams::SynthesisChecks for SynthesisChecks {
    fn validate_synthesis(
        text: &str,
        duration: f64,
        segments: Option<(usize, usize)>,
    ) -> Option<String> {
        if text.trim_matches(|c| PYTHON_WHITESPACE.contains(&c)).is_empty() {
            return None;
        }
        if duration <= 0.0 {
            return Some(format!("no audio frames were pushed for text: {text}"));
        }
        match segments {
            Some((expected, actual)) if expected != actual => {
                Some(format!("number of segments mismatch: expected {expected}, but got {actual}"))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod synthesis_checks_tests {
    use lk_agents_abi::tts::streams::SynthesisChecks as _;

    use super::SynthesisChecks;

    #[test]
    fn synthesis_without_audio_or_with_lost_segments_fails() {
        assert!(SynthesisChecks::validate_synthesis("hello", 0.0, None).is_some());
        assert!(SynthesisChecks::validate_synthesis("hello", 1.0, Some((2, 1))).is_some());
    }
}
