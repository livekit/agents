use crate::telemetry;
use std::sync::Arc;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::types::provider::ProviderEvents;
use lk_agents_abi::{
    ApiConnectOptions, AudioFrame, Clock, Closed, ProviderId, SpeechEvent, SpeechEventType,
    StreamOutput, Stt, SttCapabilities, SttSink, SttSource, Ts, Vad, VadEventType, VadSource,
};
use lk_agents_net::env::ProcessEnv;

/// `E` is the hosting process a host-built adapter reads its clock from.
pub struct StreamAdapter {
    stt: Arc<dyn Stt>,
    vad: Arc<dyn Vad>,
    /// The host object it was built as: its name, and its `error` events; else it names and
    /// reports through the STT it wraps.
    host: Option<(String, Arc<dyn ProviderEvents>)>,
    /// What an utterance's `speech_end_time` is read off.
    clock: Arc<dyn Clock>,
}

impl Clone for StreamAdapter {
    fn clone(&self) -> Self {
        let adapter = StreamAdapter::new(
            Arc::clone(&self.stt),
            Arc::clone(&self.vad),
            Arc::clone(&self.clock),
        );
        StreamAdapter { host: self.host.clone(), ..adapter }
    }
}

impl StreamAdapter {
    fn configured(&self, conn_options: Option<ApiConnectOptions>) -> Self {
        let stt = conn_options
            .and_then(|conn| self.stt.with_conn_options(&conn))
            .unwrap_or_else(|| self.stt.clone());
        Self { stt, ..self.clone() }
    }

    pub fn new(stt: Arc<dyn Stt>, vad: Arc<dyn Vad>, clock: Arc<dyn Clock>) -> Self {
        StreamAdapter { stt, vad, host: None, clock }
    }
}

impl StreamAdapter {
    /// Itself when it streams; over `vad` when it does not and there is one (`Agent.stt_node`'s
    /// default); as it is otherwise, its stream refused when opened.
    pub fn adapted(
        stt: Arc<dyn Stt>,
        vad: Option<Arc<dyn Vad>>,
        clock: Arc<dyn Clock>,
    ) -> Arc<dyn Stt> {
        match vad {
            Some(vad) if !stt.capabilities().streaming => {
                Arc::new(StreamAdapter::new(stt, vad, clock))
            }
            _ => stt,
        }
    }
}

/// `_core.SttStreamAdapter`: the generated adapter calls this.
impl lk_agents_abi::stt::stream_adapter::SttStreamAdapter for StreamAdapter {
    type Audio = SttSink;
    type Events = SttSource;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        label: String,
        events: Arc<dyn ProviderEvents>,
        stt: Arc<dyn Stt>,
        vad: Arc<dyn Vad>,
    ) -> Self {
        StreamAdapter {
            host: Some((label, events)),
            ..StreamAdapter::new(stt, vad, env.clock.clone())
        }
    }

    fn capabilities(&self) -> SttCapabilities {
        Stt::capabilities(self)
    }

    async fn recognize(
        &self,
        cx: telemetry::Cx,
        frames: Vec<AudioFrame>,
        language: Option<LanguageCode>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<SpeechEvent, ProviderError> {
        let view = self.configured(conn_options);
        super::recognize(
            view.stt.clone(),
            view.stt.conn_options().unwrap_or_default(),
            view.clock.clone(),
            &cx,
            &frames,
            language.as_ref(),
        )
        .await
    }

    async fn stream(
        &self,
        cx: telemetry::Cx,
        language: Option<LanguageCode>,
        start_time_offset: Option<f64>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<lk_agents_abi::DuplexStream<SttSink, SttSource>, ProviderError> {
        Stt::stream(&self.configured(conn_options), &cx, language.as_ref(), start_time_offset)
            .await
            .map(|opened| opened.expect("the adapter's `stream` is always duplex"))
    }
}

impl Stt for StreamAdapter {
    fn label(&self) -> String {
        match &self.host {
            Some((label, _)) => label.clone(),
            None => format!("StreamAdapter[{}]", self.stt.label()),
        }
    }

    /// The adapter is transparent to the metrics.
    fn id(&self) -> ProviderId {
        self.stt.id()
    }

    /// Streaming, no interim results, no diarization (it needs a streaming STT).
    fn capabilities(&self) -> SttCapabilities {
        let inner = self.stt.capabilities();
        SttCapabilities {
            streaming: true,
            interim_results: false,
            diarization: false,
            keyterms: inner.keyterms,
            chat_context: inner.chat_context,
            ..SttCapabilities::default()
        }
    }

    fn prewarm(&self, cx: &telemetry::Cx) {
        self.stt.prewarm(cx);
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        match &self.host {
            Some((_, events)) => events.failed(error, recoverable),
            None => self.stt.failed(error, recoverable),
        }
    }

    fn update_keyterms(&self, keyterms: &[String]) {
        self.stt.update_keyterms(keyterms);
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        self.stt.push_conversation_item(role, text);
    }

    /// Each utterance's recognition is retried on its own.
    fn retries_itself(&self) -> bool {
        true
    }

    /// The options go to the wrapped STT's recognitions; the adapter's stream does not retry.
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    /// The wrapped STT's: what its recognitions run under.
    fn conn_options(&self) -> Option<ApiConnectOptions> {
        self.stt.conn_options()
    }

    fn recognize(
        &self,
        cx: &telemetry::Cx,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        let (cx, buffer, language) = (cx.clone(), buffer.to_vec(), language.cloned());
        Box::pin(async move {
            super::recognize(
                self.stt.clone(),
                self.stt.conn_options().unwrap_or_default(),
                self.clock.clone(),
                &cx,
                &buffer,
                language.as_ref(),
            )
            .await
        })
    }

    fn stream(
        &self,
        cx: &telemetry::Cx,
        language: Option<&LanguageCode>,
        _start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        let (cx, language) = (cx.clone(), language.cloned());
        Box::pin(async move {
            let opened = self.vad.stream(&cx).await.map(|stream| {
                let (sink, vad) = stream.split();
                let source = AdapterSource {
                    vad,
                    stt: Arc::clone(&self.stt),
                    cx,
                    language: language.clone(),
                    utterance: None,
                    speech_end: None,
                    clock: Arc::clone(&self.clock),
                };
                lk_agents_abi::DuplexStream::new(sink as SttSink, Box::new(source) as SttSource)
            });
            opened.map(Some)
        })
    }
}

/// The VAD's events as speech events; an ended utterance is recognized on the next read.
struct AdapterSource {
    vad: VadSource,
    stt: Arc<dyn Stt>,
    /// What each utterance's recognition runs under: the stream's caller.
    cx: telemetry::Cx,
    language: Option<LanguageCode>,
    /// The utterance the last end of speech closed, not yet recognized.
    utterance: Option<Vec<AudioFrame>>,
    /// When the last utterance ended, wall seconds: the VAD's report less its silence and its
    /// inference time, as the reference reads it.
    speech_end: Option<f64>,
    clock: Arc<dyn Clock>,
}

impl StreamOutput<SpeechEvent> for AdapterSource {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            loop {
                if let Some(frames) = self.utterance.take() {
                    let event = super::recognize(
                        self.stt.clone(),
                        self.stt.conn_options().unwrap_or_default(),
                        self.clock.clone(),
                        &self.cx,
                        &frames,
                        self.language.as_ref(),
                    )
                    .await
                    .map_err(ProviderError::into_final)?;
                    if let Some(mut transcript) = recognized(event) {
                        transcript.speech_end_time = transcript.speech_end_time.or(self.speech_end);
                        return Ok(Some(transcript));
                    }
                }
                let Some(event) = self.vad.next().await? else { return Ok(None) };
                match event.kind {
                    VadEventType::StartOfSpeech => {
                        return Ok(Some(SpeechEvent::new(SpeechEventType::StartOfSpeech)));
                    }
                    VadEventType::EndOfSpeech => {
                        let ended = lk_agents_abi::unix_secs(self.clock.wall())
                            - event.silence_duration
                            - event.inference_duration;
                        self.speech_end = Some(ended);
                        self.utterance = Some(event.frames);
                        let mut end = SpeechEvent::new(SpeechEventType::EndOfSpeech);
                        end.speech_end_time = Some(ended);
                        return Ok(Some(end));
                    }
                    VadEventType::InferenceDone => {}
                }
            }
        })
    }

    fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { self.vad.close(deadline).await })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for AdapterSource {}

/// The final transcript a recognition becomes (its best alternative), or `None` for an empty one.
fn recognized(event: SpeechEvent) -> Option<SpeechEvent> {
    let best = event.alternatives.into_iter().next()?;
    if best.text.is_empty() {
        return None;
    }
    let mut transcript = SpeechEvent::new(SpeechEventType::FinalTranscript);
    transcript.alternatives.push(best);
    transcript.speech_end_time = event.speech_end_time;
    Some(transcript)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{StreamInput, VadCapabilities, VadEvent};
    use std::collections::VecDeque;

    use parking_lot::Mutex;

    struct Ignore;

    impl StreamInput<AudioFrame> for Ignore {
        fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {}
    }

    struct Events(VecDeque<VadEventType>);

    impl StreamOutput<VadEvent> for Events {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<VadEvent>, ProviderError>> {
            Box::pin(async move {
                Ok(self.0.pop_front().map(|kind| VadEvent::new(kind, 0, 0.0, 0.0, 0.0)))
            })
        }
        fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
            Box::pin(async move { Closed::Clean })
        }
    }

    impl lk_agents_abi::utils::stream::ProviderStream<VadEvent> for Events {}

    struct ScriptedVad(Vec<VadEventType>);

    impl Vad for ScriptedVad {
        fn label(&self) -> String {
            "vad".into()
        }
        fn capabilities(&self) -> VadCapabilities {
            VadCapabilities { update_interval: 0.032 }
        }
        fn stream(
            &self,
            _cx: &telemetry::Cx,
        ) -> lk_agents_abi::BoxFuture<
            '_,
            Result<lk_agents_abi::DuplexStream<lk_agents_abi::VadSink, VadSource>, ProviderError>,
        > {
            let events = Events(self.0.iter().copied().collect());
            Box::pin(async move {
                Ok(lk_agents_abi::DuplexStream::new(
                    Box::new(Ignore) as lk_agents_abi::VadSink,
                    Box::new(events) as VadSource,
                ))
            })
        }
    }

    /// Recognizes each utterance as the next of its texts.
    struct BatchStt {
        replies: Mutex<VecDeque<Result<&'static str, lk_agents_abi::errors::ApiError>>>,
        loaded: std::sync::atomic::AtomicBool,
        requires_prewarm: bool,
    }

    impl BatchStt {
        fn new(
            replies: impl IntoIterator<Item = Result<&'static str, lk_agents_abi::errors::ApiError>>,
        ) -> Self {
            Self {
                replies: Mutex::new(replies.into_iter().collect()),
                loaded: std::sync::atomic::AtomicBool::new(false),
                requires_prewarm: false,
            }
        }
    }

    impl Stt for BatchStt {
        fn label(&self) -> String {
            "batch".into()
        }
        fn capabilities(&self) -> SttCapabilities {
            SttCapabilities { keyterms: true, ..SttCapabilities::default() }
        }
        fn prewarm(&self, _cx: &telemetry::Cx) {
            self.loaded.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        fn recognize(
            &self,
            _cx: &telemetry::Cx,
            _: &[AudioFrame],
            _: Option<&LanguageCode>,
        ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
            Box::pin(async move {
                if self.requires_prewarm && !self.loaded.load(std::sync::atomic::Ordering::SeqCst) {
                    return Err(lk_agents_abi::errors::ApiError::new(
                        lk_agents_abi::errors::ErrorKind::InvalidRequest,
                        "model is not loaded",
                    )
                    .into());
                }
                let text = self.replies.lock().pop_front().unwrap_or(Ok(""))?;
                let mut ev = SpeechEvent::new(SpeechEventType::FinalTranscript);
                ev.alternatives.push(lk_agents_abi::SpeechData::new("en", text));
                Ok(ev)
            })
        }
        fn stream(
            &self,
            _cx: &telemetry::Cx,
            _: Option<&LanguageCode>,
            _: Option<f64>,
        ) -> lk_agents_abi::BoxFuture<
            '_,
            Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
        > {
            Box::pin(async move { Ok(None) })
        }
    }

    // Catches: a transcript before its end of speech, an empty recognition reported, or the
    // VAD's inference ticks leaking out as events.
    #[tokio::test]
    async fn each_utterance_is_recognized_once_it_ends() {
        use VadEventType::{EndOfSpeech, InferenceDone, StartOfSpeech};
        let stt = Arc::new(BatchStt::new([Ok("hello"), Ok("")]));
        let vad = Arc::new(ScriptedVad(vec![
            StartOfSpeech,
            InferenceDone,
            EndOfSpeech,
            StartOfSpeech,
            EndOfSpeech,
        ]));
        let adapter =
            StreamAdapter::adapted(stt, Some(vad), Arc::new(lk_agents_net::clock::SimClock::new()));
        let caps = adapter.capabilities();
        assert!(caps.streaming && !caps.interim_results && caps.keyterms);
        let (_sink, mut source) =
            adapter.stream(&telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
        let mut seen = Vec::new();
        while let Some(ev) = source.next().await.unwrap() {
            seen.push((ev.kind, ev.best_alternative().map(|a| a.text.clone())));
        }
        use SpeechEventType as T;
        assert_eq!(
            seen,
            vec![
                (T::StartOfSpeech, None),
                (T::EndOfSpeech, None),
                (T::FinalTranscript, Some("hello".into())),
                (T::StartOfSpeech, None),
                (T::EndOfSpeech, None),
            ]
        );
    }

    /// An utterance recognized before a provider error is delivered, then the error ends the
    /// stream. https://github.com/livekit/agents/pull/2215
    #[tokio::test]
    async fn queued_utterances_deliver_their_transcript_and_then_the_provider_error() {
        use SpeechEventType as T;
        use lk_agents_abi::errors::ApiError;
        let stt = Arc::new(BatchStt::new([
            Ok("first utterance"),
            Err(ApiError::status("recognition refused", 401, None)),
        ]));
        let vad = Arc::new(ScriptedVad(vec![
            VadEventType::StartOfSpeech,
            VadEventType::EndOfSpeech,
            VadEventType::StartOfSpeech,
            VadEventType::EndOfSpeech,
        ]));
        let adapter =
            StreamAdapter::adapted(stt, Some(vad), Arc::new(lk_agents_net::clock::SimClock::new()));
        let (sink, mut source) =
            adapter.stream(&telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
        sink.end_input();
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            let mut events = Vec::new();
            loop {
                match source.next().await {
                    Ok(Some(event)) => events.push((
                        event.kind,
                        event.best_alternative().map(|alternative| alternative.text.clone()),
                    )),
                    Ok(None) => panic!("recognition failure became a clean end of stream"),
                    Err(error) => break (events, error),
                }
            }
        })
        .await;
        source.close(Ts::from_nanos(u64::MAX)).await;
        let (events, error) = result.expect("queued speech must drain without hanging");
        assert_eq!(
            events,
            [
                (T::StartOfSpeech, None),
                (T::EndOfSpeech, None),
                (T::FinalTranscript, Some("first utterance".into())),
                (T::StartOfSpeech, None),
                (T::EndOfSpeech, None),
            ]
        );
        assert_eq!(error.status_code(), Some(401));
        assert!(!error.retryable());
        assert_eq!(error.message(), "recognition refused");
    }

    /// Prewarming a fallback over a VAD stream adapter loads the model that recognizes.
    /// https://github.com/livekit/agents/pull/6583
    #[tokio::test]
    async fn prewarming_nested_fallback_and_stream_adapters_loads_the_recognizing_model() {
        use crate::fallback::{ChainPolicy, FallbackStt};
        let stt = Arc::new(BatchStt {
            requires_prewarm: true,
            ..BatchStt::new([Ok("the model is ready")])
        });
        let vad =
            Arc::new(ScriptedVad(vec![VadEventType::StartOfSpeech, VadEventType::EndOfSpeech]));
        let (clock, spawner, observer) = crate::testkit::fallback::parts();
        let stream_adapter = StreamAdapter::adapted(stt, Some(vad), clock.clone());
        let fallback = FallbackStt::new(
            vec![stream_adapter],
            ChainPolicy { max_retry_per_backend: 0, ..ChainPolicy::stt_defaults() },
            clock,
            spawner,
            observer,
        )
        .unwrap();
        fallback.prewarm(&telemetry::Cx::NONE);
        let (sink, mut source) =
            fallback.stream(&telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
        sink.end_input();
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            let mut words = Vec::new();
            while let Some(event) = source.next().await? {
                if let Some(alternative) = event.best_alternative() {
                    words.push(alternative.text.clone());
                }
            }
            Ok::<_, ProviderError>(words)
        })
        .await;
        source.close(Ts::from_nanos(u64::MAX)).await;
        fallback.aclose();
        assert_eq!(
            result.expect("the preloaded model must finish recognition").unwrap(),
            ["the model is ready"]
        );
    }
}
