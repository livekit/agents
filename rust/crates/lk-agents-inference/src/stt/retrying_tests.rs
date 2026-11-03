use super::*;
use lk_agents_abi::errors::{ApiError, TimeoutPhase};

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
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { Err(ApiError::new("down").into()) })
    }
    fn stream(
        &self,
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
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap().split();
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
    let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap().split();
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
    let error = stt.recognize(&[], None).await.unwrap_err();
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
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { std::future::pending().await })
    }
    fn stream(
        &self,
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
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap().split();
    let drops = u64::from(options.max_retry) * 3;
    let later = clock.now().saturating_add(std::time::Duration::from_secs(185 * drops));
    assert!(before(&*clock, Some(later), source.next()).await.is_err(), "still reconnecting");

    let (stt, _) = dropping(1, options);
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap().split();
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
    let error = stt.recognize(&[], None).await.unwrap_err();
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
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap().split();
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
        if self.0.accepted.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= 256 {
            return Err(ApiError::new("recognition input queue capacity exceeded")
                .with_retryable(false)
                .into());
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
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { unreachable!() })
    }
    fn stream(
        &self,
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
    let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap().split();
    let waiting = tokio::spawn(async move { source.next().await });
    tokio::task::yield_now().await;
    for _ in 0..257 {
        sink.push(AudioFrame::new(vec![0; 640], 16_000, 1, 320)).unwrap();
    }
    sink.end_input();
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), waiting).await;
    let error =
        result.expect("a refused live sink must wake its pending source").unwrap().unwrap_err();
    assert_eq!(error.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Api));
    assert!(error.message().contains("capacity exceeded"));
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
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move { unreachable!() })
    }
    fn stream(
        &self,
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
                    self.state.accepted.store(256, std::sync::atomic::Ordering::SeqCst);
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
    let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap().split();
    let waiting = tokio::spawn(async move { source.next().await });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    sink.push(AudioFrame::new(vec![0; 640], 16_000, 1, 320)).unwrap();
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), waiting).await;
    let error = result.expect("refused replay must wake its pending source").unwrap().unwrap_err();
    assert_eq!(error.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Api));
    assert!(error.message().contains("capacity exceeded"));
    assert!(closed.load(std::sync::atomic::Ordering::SeqCst));
}

/// A sink calling back on each frame's sample rate, its identity in these tests.
struct Hook<F>(F);

impl<F: Fn(u32) -> Result<(), ProviderError> + Send + Sync> StreamInput<AudioFrame> for Hook<F> {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        (self.0)(frame.sample_rate)
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

fn frame(rate: u32) -> Pushed {
    Pushed::Audio(AudioFrame::new(vec![0; 640], rate, 1, 320))
}

// Catches: an attempt's sink called under the input's lock, so a sink that pushes again (a host
// sink re-entering through the GIL) deadlocks, in the replay or live.
#[test]
fn a_sink_that_pushes_again_from_its_push_does_not_deadlock() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let input = Arc::new(Input::default());
        let heard = Arc::new(Mutex::new(Vec::new()));
        let (weak, log) = (Arc::downgrade(&input), Arc::clone(&heard));
        input.send(frame(1)).unwrap();
        input.install(Box::new(Hook(move |rate| {
            log.lock().push(rate);
            match (rate % 2, weak.upgrade()) {
                (1, Some(input)) => input.send(frame(rate + 1)),
                _ => Ok(()),
            }
        })));
        input.send(frame(3)).unwrap();
        done.send(heard.lock().clone()).unwrap();
    });
    let heard = finished.recv_timeout(std::time::Duration::from_secs(5)).expect("deadlocked");
    assert_eq!(heard, [1, 2, 3, 4]);
}

// Catches: a frame refused by an attempt a later one already replaced held for the next failover,
// so it is replayed after the frames the later attempt heard since, or never.
#[test]
fn a_frame_refused_by_a_replaced_attempt_goes_to_the_one_serving() {
    let input = Arc::new(Input::default());
    let heard = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&heard);
    let next: Mutex<Option<SttSink>> = Mutex::new(Some(Box::new(Hook(move |rate| {
        log.lock().push(rate);
        Ok(())
    }))));
    let weak = Arc::downgrade(&input);
    input.install(Box::new(Hook(move |_| {
        // the attempt failed and the next one was installed before this refusal came back
        let input = weak.upgrade().unwrap();
        input.clear();
        if let Some(sink) = next.lock().take() {
            input.install(sink);
        }
        Err(ApiError::new("closed").into())
    })));
    input.send(frame(1)).unwrap();
    input.send(frame(2)).unwrap();
    assert_eq!(*heard.lock(), [1, 2]);
    assert!(input.state.lock().held.is_empty());
}
