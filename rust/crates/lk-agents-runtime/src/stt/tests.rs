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

impl StreamSink<AudioFrame> for Count {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        self.0.lock().unwrap()[self.1] += 1;
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

#[async_trait]
impl StreamSource<SpeechEvent> for Attempt {
    async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        match std::mem::replace(self, Attempt::Done) {
            Attempt::Fail => Err(ApiError::new("hiccup").into()),
            Attempt::Cancelled => Err(ProviderError::Host {
                error: lk_agents_abi::errors::HostError {
                    kind: "CancelledError".into(),
                    message: String::new(),
                    host_ref: None,
                },
                api_kind: None,
                retryable: false,
                status_code: None,
            }),
            Attempt::Final => Ok(Some(SpeechEvent::new(SpeechEventType::FinalTranscript))),
            Attempt::Done => Ok(None),
        }
    }
    async fn close(self: Box<Self>, _: Ts) -> Closed {
        Closed::Clean
    }
}

#[async_trait]
impl Stt for Flaky {
    fn label(&self) -> String {
        "flaky".into()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..SttCapabilities::default() }
    }
    fn failed(&self, _: ProviderError, recoverable: bool) {
        self.reports.lock().unwrap().push(recoverable);
    }
    async fn recognize(
        &self,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        Err(ApiError::new("down").into())
    }
    async fn stream(
        &self,
        _: Option<&LanguageCode>,
        offset: Option<f64>,
    ) -> Result<Option<(SttSink, SttSource)>, ProviderError> {
        let mut opened = self.opened.lock().unwrap();
        opened.push(offset.unwrap_or_default());
        self.heard.lock().unwrap().push(0);
        let attempt = match (opened.len(), self.cancels) {
            (1, true) => Attempt::Cancelled,
            (1, false) => Attempt::Fail,
            _ => Attempt::Final,
        };
        let sink = Count(Arc::clone(&self.heard), opened.len() - 1);
        Ok(Some((Box::new(sink), Box::new(attempt))))
    }
}

fn retrying(inner: Arc<Flaky>, max_retry: u32) -> RetryingStt {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_abi::utils::clock::SimClock::new());
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
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    assert!(source.next().await.unwrap().is_none());
    assert!(inner.reports.lock().unwrap().is_empty());
    assert_eq!(inner.opened.lock().unwrap().len(), 1);
}

// Catches: a retried stream that loses what was pushed while no attempt was open, reports the
// failure terminal, or reopens without moving its timestamps.
#[tokio::test(start_paused = true)]
async fn a_failed_attempt_is_reopened_and_hears_what_came_meanwhile() {
    let inner = Arc::new(Flaky::default());
    let stt = retrying(Arc::clone(&inner), 3);
    let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    let hooked = Arc::new(Mutex::new(0));
    let calls = Arc::clone(&hooked);
    source.report_retries(Arc::new(move |_, _, _| *calls.lock().unwrap() += 1));
    let next = tokio::spawn(async move { source.next().await.map(|ev| ev.map(|ev| ev.kind)) });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    sink.push(AudioFrame::new(vec![0u8; 4], 16_000, 1, 2)).unwrap();
    assert_eq!(next.await.unwrap().unwrap(), Some(SpeechEventType::FinalTranscript));
    assert_eq!(*inner.reports.lock().unwrap(), vec![true]);
    assert_eq!(*hooked.lock().unwrap(), 1);
    let opened = inner.opened.lock().unwrap().clone();
    assert_eq!(opened.len(), 2);
    assert!((opened[1] - 0.1).abs() < 1e-3, "offset {}", opened[1]);
    assert_eq!(*inner.heard.lock().unwrap(), vec![0, 1]);
}

// Catches: a batch recognition retried past its budget, or exhausted without the reference's
// message.
#[tokio::test(start_paused = true)]
async fn a_recognition_gives_up_after_its_budget() {
    // Exhausting it raises the host's chained connection error, so the interpreter must exist.
    #[cfg(feature = "python")]
    pyo3::Python::initialize();
    let inner = Arc::new(Flaky::default());
    let stt = retrying(Arc::clone(&inner), 1);
    let error = stt.recognize(&[], None).await.unwrap_err();
    assert_eq!(error.message(), "failed to recognize speech after 2 attempts");
    assert_eq!(*inner.reports.lock().unwrap(), vec![true, false]);
}

/// Every attempt connects, stays silent for `uptime`, then drops; a recognition never answers.
struct Dropping {
    uptime: std::time::Duration,
    clock: Arc<dyn Clock>,
}

struct Silent(std::time::Duration, Arc<dyn Clock>);

#[async_trait]
impl StreamSource<SpeechEvent> for Silent {
    async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        self.1.wait_until(self.1.now().saturating_add(self.0)).await;
        Err(ApiError::connection(Some("socket dropped".into())).into())
    }
    async fn close(self: Box<Self>, _: Ts) -> Closed {
        Closed::Clean
    }
}

struct Deaf;

impl StreamSink<AudioFrame> for Deaf {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

#[async_trait]
impl Stt for Dropping {
    fn label(&self) -> String {
        "dropping".into()
    }
    fn capabilities(&self) -> SttCapabilities {
        SttCapabilities { streaming: true, ..SttCapabilities::default() }
    }
    async fn recognize(
        &self,
        _: &[AudioFrame],
        _: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        std::future::pending().await
    }
    async fn stream(
        &self,
        _: Option<&LanguageCode>,
        _: Option<f64>,
    ) -> Result<Option<(SttSink, SttSource)>, ProviderError> {
        Ok(Some((Box::new(Deaf), Box::new(Silent(self.uptime, Arc::clone(&self.clock))))))
    }
}

fn dropping(uptime_secs: u64, options: ApiConnectOptions) -> (RetryingStt, Arc<dyn Clock>) {
    let clock: Arc<dyn Clock> = Arc::new(lk_agents_abi::utils::clock::SimClock::new());
    let uptime = std::time::Duration::from_secs(uptime_secs);
    let inner = Arc::new(Dropping { uptime, clock: Arc::clone(&clock) });
    (RetryingStt { inner, options, clock: Arc::clone(&clock) }, clock)
}

// Catches: an idle provider socket closed every ~3 minutes (Cartesia) exhausting the budget
// when nobody speaks, while attempts failing before the connect limit still give up.
#[tokio::test(start_paused = true)]
async fn an_idle_socket_that_keeps_dropping_never_exhausts_the_budget() {
    #[cfg(feature = "python")]
    pyo3::Python::initialize();
    let options = ApiConnectOptions::default();
    let (stt, clock) = dropping(180, options);
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    let drops = u64::from(options.max_retry) * 3;
    let later = clock.now().saturating_add(std::time::Duration::from_secs(185 * drops));
    assert!(before(&*clock, Some(later), source.next()).await.is_err(), "still reconnecting");

    let (stt, _) = dropping(1, options);
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
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
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
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
