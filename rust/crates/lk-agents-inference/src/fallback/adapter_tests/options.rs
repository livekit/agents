use super::*;

#[derive(Clone)]
struct PolicyProvider {
    conn: ApiConnectOptions,
    calls: mpsc::Sender<ApiConnectOptions>,
    streaming: bool,
}

impl PolicyProvider {
    fn record(&self) {
        self.calls.try_send(self.conn).unwrap();
    }
}

#[async_trait]
impl Tts for PolicyProvider {
    fn label(&self) -> String {
        "policy-tts".into()
    }
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: self.streaming, aligned_transcript: false }
    }
    fn sample_rate(&self) -> u32 {
        24_000
    }
    fn num_channels(&self) -> u16 {
        1
    }
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Tts>> {
        Some(Arc::new(Self { conn: *conn, ..self.clone() }))
    }
    async fn synthesize(&self, _text: &str) -> Result<TtsSource, ProviderError> {
        self.record();
        Ok(Box::new(Frames { left: 1, rate: 24_000, channels: 1, dies: false }))
    }
    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        self.record();
        Ok((
            Box::new(DiscardText),
            Box::new(Frames { left: 1, rate: 24_000, channels: 1, dies: false }),
        ))
    }
}

struct DiscardText;
impl StreamSink<String> for DiscardText {
    fn push(&self, _item: String) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

#[async_trait]
impl Stt for PolicyProvider {
    fn label(&self) -> String {
        "policy-stt".into()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..Default::default() }
    }
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
        Some(Arc::new(Self { conn: *conn, ..self.clone() }))
    }
    async fn recognize(
        &self,
        _buffer: AudioBuffer<'_>,
        _language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        self.record();
        Ok(SpeechEvent::new(SpeechEventType::FinalTranscript))
    }
    async fn stream(
        &self,
        _language: Option<&LanguageCode>,
        _offset: Option<f64>,
    ) -> Result<Option<(SttSink, SttSource)>, ProviderError> {
        self.record();
        Ok(Some((Box::new(DiscardAudio), Box::new(EmptySpeech))))
    }
}

struct DiscardAudio;
impl StreamSink<AudioFrame> for DiscardAudio {
    fn push(&self, _item: AudioFrame) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}
struct EmptySpeech;
#[async_trait]
impl StreamSource<SpeechEvent> for EmptySpeech {
    async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        Ok(None)
    }
    async fn close(self: Box<Self>, _deadline: Ts) -> Closed {
        Closed::Clean
    }
}

fn policy(read: f64) -> ApiConnectOptions {
    ApiConnectOptions {
        max_retry: 7,
        retry_interval: 0.0,
        pool_timeout: None,
        connect_timeout: Some(0.75),
        write_timeout: Some(0.5),
        read_timeout: Some(read),
        total_timeout: Some(9.0),
    }
}

#[tokio::test]
async fn fallback_tts_resolves_each_operation_for_streaming_and_sentence_backends() {
    for streaming in [true, false] {
        let (calls, mut seen) = mpsc::channel(16);
        let (clock, spawner, observer) = parts();
        let fallback: Arc<dyn Tts> = Arc::new(
            FallbackTts::new(
                vec![Arc::new(PolicyProvider { conn: policy(99.0), calls, streaming })],
                None,
                ChainPolicy::tts_defaults(),
                clock,
                spawner,
                observer,
            )
            .unwrap(),
        );
        // Binding a second call before opening the first must not replace its options.
        let first = fallback.with_conn_options(&policy(0.125)).unwrap_or_else(|| fallback.clone());
        let second = fallback.with_conn_options(&policy(0.25)).unwrap_or_else(|| fallback.clone());
        let mut source = first.synthesize("first").await.unwrap();
        drain_tts(&mut source).await;
        source.close(NO_DEADLINE).await;
        let (sink, mut source) = second.stream().await.unwrap();
        sink.push("second.".into()).unwrap();
        sink.end_input();
        drain_tts(&mut source).await;
        source.close(NO_DEADLINE).await;
        for read in [0.125, 0.25] {
            let actual = seen.try_recv().expect("the provider executed");
            assert_eq!(actual, ApiConnectOptions { max_retry: 0, ..policy(read) });
        }
    }
}

#[tokio::test]
async fn fallback_stt_resolves_each_operation_for_batch_and_stream() {
    let (calls, mut seen) = mpsc::channel(16);
    let (clock, spawner, observer) = parts();
    let fallback: Arc<dyn Stt> = Arc::new(
        FallbackStt::new(
            vec![Arc::new(PolicyProvider { conn: policy(99.0), calls, streaming: true })],
            ChainPolicy {
                attempt_timeout: None,
                retry_interval: None,
                ..ChainPolicy::stt_defaults()
            },
            clock,
            spawner,
            observer,
        )
        .unwrap(),
    );
    let first = fallback.with_conn_options(&policy(0.125)).unwrap_or_else(|| fallback.clone());
    let second = fallback.with_conn_options(&policy(0.25)).unwrap_or_else(|| fallback.clone());
    first.recognize(&[], None).await.unwrap();
    let (sink, mut source) = second.stream(None, None).await.unwrap().unwrap();
    sink.end_input();
    assert!(source.next().await.unwrap().is_none());
    source.close(NO_DEADLINE).await;
    for read in [0.125, 0.25] {
        let actual = seen.try_recv().expect("the provider executed");
        assert_eq!(actual, ApiConnectOptions { max_retry: 0, ..policy(read) });
    }
}

#[tokio::test]
async fn a_fallback_deadline_includes_the_wait_for_its_runtime_task() {
    let (calls, mut seen) = mpsc::channel(4);
    let clock = Arc::new(VirtualClock::new());
    let (_, spawner, observer) = parts();
    let fallback = FallbackTts::new(
        vec![Arc::new(PolicyProvider { conn: policy(99.0), calls, streaming: true })],
        None,
        ChainPolicy::tts_defaults(),
        clock.clone(),
        spawner,
        observer,
    )
    .unwrap();
    let call = fallback
        .with_conn_options(&ApiConnectOptions { total_timeout: Some(1.0), ..policy(0.125) })
        .unwrap();
    let mut source = call.synthesize("hello").await.unwrap();
    // No task has been polled yet: this is admitted work awaiting runtime execution.
    clock.advance(Duration::from_secs(2));
    let outcome = source.next().await;
    assert!(outcome.is_err(), "the deadline started at acceptance, before the task ran");
    assert!(seen.try_recv().is_err(), "expired work cannot open a provider");
    source.close(NO_DEADLINE).await;
    fallback.aclose();
}
