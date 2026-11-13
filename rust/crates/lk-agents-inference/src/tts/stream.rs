use lk_agents_abi::errors::{HostError, ProviderError, PullError};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::tts::stream_pacer::PushedAudio;
use lk_agents_abi::tts::streams::DrainError;
use lk_agents_abi::utils::stream::ProviderStream;
use lk_agents_abi::utils::strings::PYTHON_WHITESPACE;
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, Closed, StreamInput, StreamOutput, SynthesizedAudio, TextSink,
    Ts, Tts, TtsSource,
};
use lk_agents_net::source::{Pulled, SharedStream};
use lk_agents_net::strings::shortuuid;
use std::sync::Arc;

/// A synthesis as the host drives it: the text pushed into its request, its audio read out.
pub struct TtsStream {
    sink: Option<Arc<dyn StreamInput<String>>>,
    source: SharedStream<SynthesizedAudio>,
    /// Its TTS runs natively, not behind the host's proxy.
    native: bool,
}

impl TtsStream {
    pub fn new(
        sink: Option<TextSink>,
        source: TtsSource,
        clock: Arc<dyn lk_agents_abi::Clock>,
    ) -> Self {
        Self { sink: sink.map(Arc::from), source: SharedStream::new(source, clock), native: true }
    }

    /// Where text goes; `None` once the synthesis has ended, its audio or its failure being what
    /// reading it reports: input after a provider's failure never masks that failure.
    fn sink(&self) -> Result<Option<&Arc<dyn StreamInput<String>>>, ProviderError> {
        let sink = self.sink.as_ref().ok_or_else(|| {
            let message = "this is a synthesize() stream; its text was given at the call";
            ProviderError::from(HostError {
                class: "RuntimeError".into(),
                ..HostError::new(message)
            })
        })?;
        Ok((!self.source.is_closed()).then_some(sink))
    }
}

/// A synthesis's audio, its first attempt opened when the audio is first read.
enum Opening {
    Pending(BoxFuture<'static, Result<TtsSource, ProviderError>>),
    Open(TtsSource),
}

impl Opening {
    async fn source(&mut self) -> Result<&mut TtsSource, ProviderError> {
        if let Opening::Pending(opening) = self {
            *self = Opening::Open(opening.await?);
        }
        match self {
            Opening::Open(source) => Ok(source),
            Opening::Pending(_) => unreachable!("opened above"),
        }
    }
}

impl StreamOutput<SynthesizedAudio> for Opening {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move { self.source().await?.next().await })
    }

    /// An open still under way finishes, within the close's deadline, and what it opened is
    /// closed: never dropped halfway.
    fn close(self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        match *self {
            Opening::Pending(opening) => Box::pin(async move {
                match opening.await {
                    Ok(source) => source.close(deadline).await,
                    Err(_) => Closed::Clean,
                }
            }),
            Opening::Open(source) => source.close(deadline),
        }
    }
}

impl ProviderStream<SynthesizedAudio> for Opening {
    fn started_time(&self) -> Option<f64> {
        match self {
            Opening::Open(source) => source.started_time(),
            Opening::Pending(_) => None,
        }
    }
    fn take_call_metrics(&mut self, in_flight: bool) -> Option<Vec<AgentMetrics>> {
        match self {
            Opening::Open(source) => source.take_call_metrics(in_flight),
            Opening::Pending(_) => None,
        }
    }
}

impl lk_agents_abi::tts::streams::TtsStream for TtsStream {
    type Env = lk_agents_net::env::ProcessEnv;

    fn new(
        env: &Self::Env,
        cx: crate::telemetry::Cx,
        tts: Arc<dyn Tts>,
        text: Option<String>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Self {
        let options = conn_options.unwrap_or(lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS);
        // A host plugin has an identity of its own; a native provider none.
        let native = tts.instance_id().is_none();
        let clock = env.clock.clone();
        let (sink, opening): (_, BoxFuture<'static, Result<TtsSource, ProviderError>>) = match text
        {
            Some(text) => {
                let clock = clock.clone();
                let opening = async move {
                    super::request::synthesize(tts, &cx, &text, options, clock, None).await
                };
                (None, Box::pin(opening))
            }
            None => match super::request::start(tts, &cx, options, clock.clone(), None) {
                Ok((sink, opening)) => (Some(sink), Box::pin(async move { Ok(opening.await) })),
                Err(error) => (None, Box::pin(std::future::ready(Err(error)))),
            },
        };
        TtsStream { native, ..TtsStream::new(sink, Box::new(Opening::Pending(opening)), clock) }
    }

    fn native(&self) -> bool {
        self.native
    }

    fn push_text(&self, text: String) -> Result<(), ProviderError> {
        self.sink()?.map_or(Ok(()), |sink| sink.push(text))
    }
    fn flush(&self) -> Result<(), ProviderError> {
        self.sink()?.map_or(Ok(()), |sink| sink.flush())
    }
    fn end_input(&self) -> Result<(), ProviderError> {
        if let Some(sink) = self.sink()? {
            sink.end_input();
        }
        Ok(())
    }
    async fn recv(&self) -> Result<Option<SynthesizedAudio>, PullError> {
        let pull = self.source.try_pull().ok_or(PullError::Overlap)?;
        Ok(match pull.await? {
            Pulled::Item(audio) => Some(audio),
            Pulled::End | Pulled::Cancelled => None,
        })
    }
    async fn drain_into(
        &self,
        emitter: Arc<dyn PushedAudio>,
        stream: bool,
    ) -> Result<(), DrainError> {
        // a streamed synthesis keeps its provider's segments: a new id starts the next
        let mut segment: Option<String> = None;
        while let Some(audio) = self.recv().await? {
            let next = (!audio.segment_id.is_empty()).then_some(&audio.segment_id);
            if stream && (segment.is_none() || next.is_some_and(|id| segment.as_ref() != Some(id)))
            {
                if segment.is_some() {
                    emitter.end_segment()?;
                }
                let id = next.cloned().unwrap_or_else(|| shortuuid(""));
                emitter.start_segment(id.clone())?;
                segment = Some(id);
            }
            emitter.push_timed_transcript(audio.timed_transcript)?;
            emitter.push_frame(audio.frame)?;
            if audio.usage.is_some() || audio.characters_count.is_some() {
                emitter.report_usage(audio.usage.map(|usage| *usage), audio.characters_count)?;
            }
        }
        Ok(())
    }
    /// Cancels the synthesis; the input is not ended, so nothing pending is spoken.
    async fn aclose(&self) {
        if self.source.aclose().await == Closed::TimedOut {
            log::error!(target: lk_agents_abi::LOG_TARGET, "a native stream did not release its resources in time");
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
        // the synthesis is over: late input is taken and said by nothing, never an error that
        // would mask how the synthesis ended
        stream.push_text("late".into()).unwrap();
        stream.end_input().unwrap();
        assert_eq!(*input.0.lock(), ["text"], "aclose must not speak pending text");
    }

    /// Only a synthesis of text with words and no audio is a "no audio" failure.
    /// https://github.com/livekit/agents/pull/2420
    #[test]
    fn only_text_with_words_requires_audio() {
        use lk_agents_abi::tts::streams::SynthesisChecks as _;
        for text in ["", " ", "\n\t "] {
            assert_eq!(SynthesisChecks::validate_synthesis(text, 0.0, None), None, "{text:?}");
        }
        assert_eq!(
            SynthesisChecks::validate_synthesis("Hello there.", 0.0, None).as_deref(),
            Some("no audio frames were pushed for text: Hello there.")
        );
        assert_eq!(SynthesisChecks::validate_synthesis("Hello there.", 0.4, None), None);
    }
}

pub struct SynthesisChecks;

impl lk_agents_abi::tts::streams::SynthesisChecks for SynthesisChecks {
    fn validate_synthesis(
        text: &str,
        duration: f64,
        segments: Option<lk_agents_abi::tts::streams::SegmentCount>,
    ) -> Option<String> {
        if text.trim_matches(|c| PYTHON_WHITESPACE.contains(&c)).is_empty() {
            return None;
        }
        if duration <= 0.0 {
            return Some(format!("no audio frames were pushed for text: {text}"));
        }
        match segments {
            Some(lk_agents_abi::tts::streams::SegmentCount { expected, actual })
                if expected != actual =>
            {
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
        assert!(
            SynthesisChecks::validate_synthesis(
                "hello",
                1.0,
                Some(lk_agents_abi::tts::streams::SegmentCount { expected: 2, actual: 1 })
            )
            .is_some()
        );
    }
}
