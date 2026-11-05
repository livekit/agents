use super::*;
use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use parking_lot::Mutex;

/// Each attempt's offset and the frames it heard; the first attempt fails when read.
#[derive(Default)]
struct Flaky {
    opened: Mutex<Vec<f64>>,
    heard: Arc<Mutex<Vec<usize>>>,
    /// Each reported failure's `recoverable`.
    reports: Mutex<Vec<bool>>,
    /// The first attempt is cancelled by the provider instead of failing.
    cancels: bool,
}

struct Count(Arc<Mutex<Vec<usize>>>, usize);

impl StreamInput<AudioFrame> for Count {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        self.0.lock()[self.1] += 1;
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

/// `Fail`s at once, or yields one final transcript then ends.
enum Attempt {
    Fail,
    Cancelled,
    Final,
    Done,
}

impl StreamOutput<SpeechEvent> for Attempt {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            match std::mem::replace(self, Attempt::Done) {
                Attempt::Fail => Err(ApiError::new("hiccup").into()),
                Attempt::Cancelled => Err(ProviderError::Host {
                    error: lk_agents_abi::errors::HostError {
                        kind: "CancelledError".into(),
                        message: String::new(),
                        host_ref: None,
                        traceback: None,
                    },
                    api_kind: None,
                    retryable: false,
                    status_code: None,
                }),
                Attempt::Final => Ok(Some(SpeechEvent::new(SpeechEventType::FinalTranscript))),
                Attempt::Done => Ok(None),
            }
        })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for Attempt {}

impl Stt for Flaky {
    fn label(&self) -> String {
        "flaky".into()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..SttCapabilities::default() }
    }
    fn failed(&self, _: ProviderError, recoverable: bool) {
        self.reports.lock().push(recoverable);
    }
    fn recognize(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { Err(ApiError::new("down").into()) })
    }
    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<&LanguageCode>,
        offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        Box::pin(async move {
            let mut opened = self.opened.lock();
            opened.push(offset.unwrap_or_default());
            self.heard.lock().push(0);
            let attempt = match (opened.len(), self.cancels) {
                (1, true) => Attempt::Cancelled,
                (1, false) => Attempt::Fail,
                _ => Attempt::Final,
            };
            let sink = Count(Arc::clone(&self.heard), opened.len() - 1);
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(sink) as SttSink,
                Box::new(attempt) as SttSource,
            )))
        })
    }
}

fn retrying(inner: Arc<Flaky>, max_retry: u32) -> RetryingStt {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_net::clock::SimClock::new());
    RetryingStt {
        inner,
        options: ApiConnectOptions { max_retry, ..ApiConnectOptions::default() },
        clock,
    }
}

// Catches: a provider that cancels its own stream at close reported as a failed recognition
// (a session `stt_error` carrying a CancelledError) or reopened as a retry.
#[tokio::test(start_paused = true)]
async fn a_stream_the_provider_cancels_ends_unreported() {
    let inner = Arc::new(Flaky { cancels: true, ..Flaky::default() });
    let stt = retrying(Arc::clone(&inner), 3);
    let (_sink, mut source) =
        stt.stream(&lk_agents_abi::telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
    assert!(source.next().await.unwrap().is_none());
    assert!(inner.reports.lock().is_empty());
    assert_eq!(inner.opened.lock().len(), 1);
}

// Catches: a retried stream that loses what was pushed while no attempt was open, reports the
// failure terminal, or reopens without moving its timestamps.
#[tokio::test(start_paused = true)]
async fn a_failed_attempt_is_reopened_and_hears_what_came_meanwhile() {
    let inner = Arc::new(Flaky::default());
    let stt = retrying(Arc::clone(&inner), 3);
    let (sink, mut source) =
        stt.stream(&lk_agents_abi::telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
    let hooked = Arc::new(Mutex::new(0));
    let calls = Arc::clone(&hooked);
    source.report_retries(Arc::new(move |_, _, _| *calls.lock() += 1));
    let next = tokio::spawn(async move { source.next().await.map(|ev| ev.map(|ev| ev.kind)) });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    sink.push(AudioFrame::new(vec![0u8; 4], 16_000, 1, 2)).unwrap();
    assert_eq!(next.await.unwrap().unwrap(), Some(SpeechEventType::FinalTranscript));
    assert_eq!(*inner.reports.lock(), vec![true]);
    assert_eq!(*hooked.lock(), 1);
    let opened = inner.opened.lock().clone();
    assert_eq!(opened.len(), 2);
    assert!((opened[1] - 0.1).abs() < 1e-3, "offset {}", opened[1]);
    assert_eq!(*inner.heard.lock(), vec![0, 1]);
}

// Catches: a batch recognition retried past its budget, or exhausted without the reference's
// message and the last failure as its cause.
#[tokio::test(start_paused = true)]
async fn a_recognition_gives_up_after_its_budget() {
    let inner = Arc::new(Flaky::default());
    let stt = retrying(Arc::clone(&inner), 1);
    let error = stt.recognize(&lk_agents_abi::telemetry::Cx::NONE, &[], None).await.unwrap_err();
    assert_eq!(error.message(), "failed to recognize speech after 2 attempts");
    assert!(matches!(error, ProviderError::Local(ApiError { cause: Some(_), .. })), "{error:?}");
    assert_eq!(*inner.reports.lock(), vec![true, false]);
}

/// Every attempt connects, stays silent for `uptime`, then drops; a recognition never answers.
struct Dropping {
    uptime: std::time::Duration,
    clock: Arc<dyn Clock>,
}

struct Silent(std::time::Duration, Arc<dyn Clock>);

impl StreamOutput<SpeechEvent> for Silent {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            self.1.wait_until(self.1.now().saturating_add(self.0)).await;
            Err(ApiError::connection(Some("socket dropped".into())).into())
        })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for Silent {}

struct Deaf;

impl StreamInput<AudioFrame> for Deaf {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl Stt for Dropping {
    fn label(&self) -> String {
        "dropping".into()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..SttCapabilities::default() }
    }
    fn recognize(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { std::future::pending().await })
    }
    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        Box::pin(async move {
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(Deaf) as SttSink,
                Box::new(Silent(self.uptime, Arc::clone(&self.clock))) as SttSource,
            )))
        })
    }
}

fn dropping(uptime_secs: u64, options: ApiConnectOptions) -> (RetryingStt, Arc<dyn Clock>) {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_net::clock::SimClock::new());
    let uptime = std::time::Duration::from_secs(uptime_secs);
    let inner = Arc::new(Dropping { uptime, clock: Arc::clone(&clock) });
    (RetryingStt { inner, options, clock: Arc::clone(&clock) }, clock)
}

// Catches: an idle provider socket closed every ~3 minutes (Cartesia) exhausting the budget
// when nobody speaks, while attempts failing before the connect limit still give up.
#[tokio::test(start_paused = true)]
async fn an_idle_socket_that_keeps_dropping_never_exhausts_the_budget() {
    let options = ApiConnectOptions::default();
    let (stt, clock) = dropping(180, options);
    let (_sink, mut source) =
        stt.stream(&lk_agents_abi::telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
    let drops = u64::from(options.max_retry) * 3;
    let later = clock.now().saturating_add(std::time::Duration::from_secs(185 * drops));
    assert!(before(&*clock, Some(later), source.next()).await.is_err(), "still reconnecting");

    let (stt, _) = dropping(1, options);
    let (_sink, mut source) =
        stt.stream(&lk_agents_abi::telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
    assert!(source.next().await.is_err(), "consecutive short attempts exhaust the budget");
}

// Catches: the core-held retry loop ignoring `total_timeout` (the host skips its own deadline
// for the attempts the core retries), or a reconnect restarting it.
#[tokio::test(start_paused = true)]
async fn total_timeout_ends_retried_recognitions_and_streams() {
    let options = ApiConnectOptions {
        max_retry: 100,
        total_timeout: Some(30.0),
        ..ApiConnectOptions::default()
    };
    let (stt, _) = dropping(20, options);
    let error = stt.recognize(&lk_agents_abi::telemetry::Cx::NONE, &[], None).await.unwrap_err();
    assert!(
        matches!(
            error,
            ProviderError::Local(ApiError {
                phase: Some(TimeoutPhase::Total),
                retryable: false,
                ..
            })
        ),
        "{error:?}"
    );

    let (stt, clock) = dropping(20, options);
    let (_sink, mut source) =
        stt.stream(&lk_agents_abi::telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
    let started = clock.now();
    let error = source.next().await.unwrap_err();
    assert!(
        matches!(
            error,
            ProviderError::Local(ApiError {
                phase: Some(TimeoutPhase::Total),
                retryable: false,
                ..
            })
        ),
        "{error:?}"
    );
    assert_eq!(clock.now().saturating_duration_since(started), std::time::Duration::from_secs(30));
}

struct StalledInput {
    accepted: std::sync::atomic::AtomicUsize,
    closed: Arc<std::sync::atomic::AtomicBool>,
}

struct RefusingSink(Arc<StalledInput>);
impl StreamInput<AudioFrame> for RefusingSink {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        if self.0.accepted.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= 1 {
            return Err(ApiError::new("recognition input refused").with_retryable(false).into());
        }
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}
struct StalledSource(Arc<StalledInput>);
impl StreamOutput<SpeechEvent> for StalledSource {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move { std::future::pending().await })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.0.closed.store(true, std::sync::atomic::Ordering::SeqCst);
            Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for StalledSource {}

struct StalledHost(Arc<StalledInput>);
impl Stt for StalledHost {
    fn label(&self) -> String {
        "stalled host".into()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..Default::default() }
    }
    fn recognize(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { unreachable!() })
    }
    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        Box::pin(async move {
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(RefusingSink(self.0.clone())) as SttSink,
                Box::new(StalledSource(self.0.clone())) as SttSource,
            )))
        })
    }
}

#[tokio::test(start_paused = true)]
async fn a_live_sink_refusal_terminates_a_stalled_host_attempt_instead_of_losing_its_input() {
    let closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let host = Arc::new(StalledInput {
        accepted: std::sync::atomic::AtomicUsize::new(0),
        closed: closed.clone(),
    });
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_net::clock::SimClock::new());
    let stt = RetryingStt {
        inner: Arc::new(StalledHost(host)),
        options: ApiConnectOptions { max_retry: 0, ..Default::default() },
        clock,
    };
    let (sink, mut source) =
        stt.stream(&lk_agents_abi::telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
    let waiting = tokio::spawn(async move { source.next().await });
    tokio::task::yield_now().await;
    for _ in 0..3 {
        sink.push(AudioFrame::new(vec![0; 640], 16_000, 1, 320)).unwrap();
    }
    sink.end_input();
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), waiting).await;
    let error =
        result.expect("a refused live sink must wake its pending source").unwrap().unwrap_err();
    assert_eq!(error.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Api));
    assert!(error.message().contains("refused"));
    assert!(closed.load(std::sync::atomic::Ordering::SeqCst));
}

struct ReplayRefuses {
    opened: std::sync::atomic::AtomicUsize,
    state: Arc<StalledInput>,
}

impl Stt for ReplayRefuses {
    fn label(&self) -> String {
        "refuses replay".into()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..Default::default() }
    }
    fn recognize(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { unreachable!() })
    }
    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        Box::pin(async move {
            let source: SttSource =
                if self.opened.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    Box::new(Attempt::Fail)
                } else {
                    self.state.accepted.store(1, std::sync::atomic::Ordering::SeqCst);
                    Box::new(StalledSource(self.state.clone()))
                };
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(RefusingSink(self.state.clone())) as SttSink,
                source,
            )))
        })
    }
}

#[tokio::test(start_paused = true)]
async fn a_refused_held_frame_terminates_the_reopened_attempt_instead_of_disappearing() {
    let closed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let state = Arc::new(StalledInput {
        accepted: std::sync::atomic::AtomicUsize::new(0),
        closed: closed.clone(),
    });
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_net::clock::SimClock::new());
    let stt = RetryingStt {
        inner: Arc::new(ReplayRefuses { opened: std::sync::atomic::AtomicUsize::new(0), state }),
        options: ApiConnectOptions { max_retry: 1, ..Default::default() },
        clock,
    };
    let (sink, mut source) =
        stt.stream(&lk_agents_abi::telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
    let waiting = tokio::spawn(async move { source.next().await });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    sink.push(AudioFrame::new(vec![0; 640], 16_000, 1, 320)).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), waiting).await;
    let error = result.expect("refused replay must wake its pending source").unwrap().unwrap_err();
    assert_eq!(error.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Api));
    assert!(error.message().contains("refused"));
    assert!(closed.load(std::sync::atomic::Ordering::SeqCst));
}

/// A host attempt that takes each entry only when the test releases it, and whose transcripts
/// the test scripts.
struct Slow {
    gate: Arc<tokio::sync::Semaphore>,
    heard: Arc<Mutex<Vec<&'static str>>>,
    events: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<SpeechEvent>>>,
}

struct SlowSink(Arc<tokio::sync::Semaphore>, Arc<Mutex<Vec<&'static str>>>);

impl StreamInput<AudioFrame> for SlowSink {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        unreachable!("the pump awaits each entry")
    }
    fn flush(&self) -> Result<(), ProviderError> {
        unreachable!("the pump awaits each entry")
    }
    fn end_input(&self) {}
    fn send(
        &self,
        entry: Entry<AudioFrame>,
    ) -> lk_agents_abi::BoxFuture<'static, Result<(), ProviderError>> {
        let (gate, heard) = (Arc::clone(&self.0), Arc::clone(&self.1));
        Box::pin(async move {
            gate.acquire().await.unwrap().forget();
            heard.lock().push(match entry {
                Entry::Item(_) => "audio",
                Entry::Flush => "flush",
                Entry::EndInput => "end",
            });
            Ok(())
        })
    }
}

struct Scripted(tokio::sync::mpsc::UnboundedReceiver<SpeechEvent>);

impl StreamOutput<SpeechEvent> for Scripted {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move { Ok(self.0.recv().await) })
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for Scripted {}

impl Stt for Slow {
    fn label(&self) -> String {
        "slow".into()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..Default::default() }
    }
    fn recognize(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { unreachable!() })
    }
    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        Box::pin(async move {
            let events = self.events.lock().take().expect("one attempt");
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(SlowSink(Arc::clone(&self.gate), Arc::clone(&self.heard))) as SttSink,
                Box::new(Scripted(events)) as SttSource,
            )))
        })
    }
}

fn transcript(start: f64, end: f64) -> SpeechEvent {
    let mut event = SpeechEvent::new(SpeechEventType::FinalTranscript);
    event.alternatives.push(lk_agents_abi::SpeechData {
        start_time: start,
        end_time: end,
        words: Some(vec![lk_agents_abi::TranscriptChunk {
            start_time: Some(start),
            end_time: Some(end),
            ..lk_agents_abi::TranscriptChunk::new("hi")
        }]),
        ..lk_agents_abi::SpeechData::new("en", "hi")
    });
    event
}

// Catches: a slow host attempt failing the stream on a burst, its input reordered against a
// flush, or the transcripts after a dropped stretch left behind call time.
#[tokio::test(start_paused = true)]
async fn a_slow_attempt_gets_a_gap_and_its_later_words_stay_on_call_time() {
    let (script, events) = tokio::sync::mpsc::unbounded_channel();
    let gate = Arc::new(tokio::sync::Semaphore::new(1));
    let heard = Arc::new(Mutex::new(Vec::new()));
    let slow = Slow {
        gate: Arc::clone(&gate),
        heard: Arc::clone(&heard),
        events: Mutex::new(Some(events)),
    };
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_net::clock::SimClock::new());
    let stt = RetryingStt { inner: Arc::new(slow), options: ApiConnectOptions::default(), clock };
    let (sink, mut source) =
        stt.stream(&lk_agents_abi::telemetry::Cx::NONE, None, None).await.unwrap().unwrap().split();
    let frame = || AudioFrame::new(vec![0; 3200], 16_000, 1, 1600);
    let reader = tokio::spawn(async move {
        let mut out = Vec::new();
        while let Ok(Some(event)) = source.next().await {
            out.push(event);
            if out.len() == 2 {
                break;
            }
        }
        out
    });
    // The first 100 ms reaches the attempt; the next 11 s arrive while it takes nothing.
    sink.push(frame()).unwrap();
    tokio::task::yield_now().await;
    for _ in 0..110 {
        sink.push(frame()).unwrap();
    }
    sink.flush().unwrap();
    sink.push(frame()).unwrap();
    gate.add_permits(1000);
    while heard.lock().len() < 103 {
        tokio::task::yield_now().await;
    }
    let heard = heard.lock().clone();
    assert_eq!(heard.len(), 103, "100 ms, 10 s kept, the flush and the frame after it");
    assert_eq!(&heard[100..], ["audio", "flush", "audio"]);
    script.send(transcript(0.02, 0.08)).unwrap();
    script.send(transcript(3.0, 3.5)).unwrap();
    let events = reader.await.unwrap();
    let times: Vec<_> = events
        .iter()
        .map(|e| {
            let alt = &e.alternatives[0];
            let word = &alt.words.as_ref().unwrap()[0];
            (alt.start_time, word.start_time.unwrap(), word.end_time.unwrap())
        })
        .collect();
    assert_eq!(times[0], (0.02, 0.02, 0.08), "before the gap");
    let (start, word_start, word_end) = times[1];
    assert!((start - 4.0).abs() < 1e-9, "{times:?}");
    assert!((word_start - 4.0).abs() < 1e-9 && (word_end - 4.5).abs() < 1e-9, "{times:?}");
}
