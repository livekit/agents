//! [`RetryingStt`]: the retry loops of the reference's `STT.recognize` and
//! `RecognizeStream._main_task`, around a host plugin STT bound to its `conn_options`. The plugin
//! base runs one attempt; this reopens the stream on a retried failure, so its consumer sees one
//! stream that survives the attempts.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::events::{SpeechEvent, SpeechEventType};
use super::stt::{Stt, SttCapabilities, SttSink, SttSource};
use crate::errors::{ProviderError, chained_connection_error};
use crate::tokenize::language::LanguageCode;
use crate::types::identity::ProviderId;
use crate::types::retry::{RequestKind, RetryDecision, decide};
use crate::utils::clock::Ts;
use crate::utils::stream::{Closed, RetryReport, StreamSink, StreamSource};
use crate::{ApiConnectOptions, AudioFrame, Clock, HOST_TEARDOWN_TIMEOUT};

pub struct RetryingStt {
    inner: Arc<dyn Stt>,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
}

/// `_core.RetryingStt`: what a host STT bound to `conn_options` is to the core.
#[crate::host(object(name = "RetryingStt", cell = "arc", as_dyn(Stt = as_stt)))]
impl RetryingStt {
    /// `stt` is attempted under `max_retry`, `retry_interval` and `timeout`; every failed attempt
    /// is reported to it ([`Stt::failed`], the plugin's `error` event).
    #[host(new(kw_only = "max_retry"))]
    pub fn for_host(
        stt: Arc<dyn Stt>,
        max_retry: u32,
        retry_interval: f64,
        timeout: f64,
        #[host(ambient)] clock: Arc<dyn Clock>,
    ) -> Result<Self, crate::types::connect::ApiConnectOptionsError> {
        let options = ApiConnectOptions::new(max_retry, retry_interval, timeout)?;
        Ok(RetryingStt { inner: stt, options, clock })
    }

    /// The retried STT as the provider the core holds (not proxied).
    pub fn as_stt(self: &Arc<Self>) -> Arc<dyn Stt> {
        Arc::clone(self) as Arc<dyn Stt>
    }
}

/// One failed attempt: reported, then waited out when retried, else the error to raise.
struct Failure<'a> {
    kind: RequestKind,
    retries: u32,
    error: ProviderError,
    report: &'a dyn Stt,
    label: &'a str,
    options: &'a ApiConnectOptions,
}

impl Failure<'_> {
    /// `Ok(delay)` to retry after; `Err` is what the caller raises.
    fn decide(self, hook: Option<&RetryReport>) -> Result<f64, ProviderError> {
        let Failure { kind, retries, error, report, label, options } = self;
        let decision = match error.api_kind() {
            None => RetryDecision::Raise,
            Some(_) => {
                decide(kind, options, retries, error.retryable(), error.status_code(), false)
            }
        };
        match decision {
            RetryDecision::Retry(delay) => {
                report.failed(error.clone(), true);
                if let Some(hook) = hook {
                    hook(error.clone(), retries + 1, std::time::Duration::from_secs_f64(delay.max(0.0)));
                }
                let streamed = kind == RequestKind::SttStream;
                log::warn!(
                    target: "livekit.agents",
                    stt = label, attempt = retries + u32::from(!streamed), streamed = streamed;
                    "failed to recognize speech: {error}, retrying in {delay}s"
                );
                Ok(delay)
            }
            RetryDecision::Exhausted(message) => {
                report.failed(error.clone(), false);
                Err(chained_connection_error(message, &error))
            }
            RetryDecision::Raise | RetryDecision::Stop => {
                report.failed(error.clone(), false);
                Err(error)
            }
        }
    }
}

async fn sleep(clock: &dyn Clock, delay: f64) {
    let at = clock.now().saturating_add(std::time::Duration::from_secs_f64(delay.max(0.0)));
    clock.wait_until(at).await;
}

#[async_trait]
impl Stt for RetryingStt {
    fn label(&self) -> String {
        self.inner.label()
    }

    fn capabilities(&self) -> SttCapabilities {
        self.inner.capabilities()
    }

    fn id(&self) -> ProviderId {
        self.inner.id()
    }

    fn prewarm(&self) {
        self.inner.prewarm();
    }

    fn update_keyterms(&self, keyterms: &[String]) {
        self.inner.update_keyterms(keyterms);
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        self.inner.push_conversation_item(role, text);
    }

    fn failed(&self, error: ProviderError, recoverable: bool) {
        self.inner.failed(error, recoverable);
    }

    /// The host view rebinds, and comes back retried under the new options.
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
        self.inner.with_conn_options(conn)
    }

    async fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        let label = self.inner.label();
        let mut retries = 0;
        loop {
            let error = match self.inner.recognize(buffer, language).await {
                Ok(event) => return Ok(event),
                Err(error) => error,
            };
            let failure = Failure {
                kind: RequestKind::SttBatch,
                retries,
                error,
                report: &*self.inner,
                label: &label,
                options: &self.options,
            };
            let delay = failure.decide(None)?;
            sleep(&*self.clock, delay).await;
            retries += 1;
        }
    }

    async fn stream(
        &self,
        language: Option<&LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> Option<Result<(SttSink, SttSource), ProviderError>> {
        let offset = start_time_offset.unwrap_or(0.0);
        let input = Arc::new(Input::default());
        let (current, failed) = match self.inner.stream(language, Some(offset)).await? {
            Ok((sink, source)) => {
                input.install(sink);
                (Some(source), None)
            }
            Err(error) => (None, Some(error)),
        };
        let source = RetrySource {
            stt: Arc::clone(&self.inner),
            options: self.options.clone(),
            clock: Arc::clone(&self.clock),
            label: self.inner.label(),
            language: language.cloned(),
            input: Arc::clone(&input),
            current,
            failed,
            retries: 0,
            began: crate::unix_secs(self.clock.wall()),
            offset,
            hook: None,
        };
        Some(Ok((Box::new(InputSink(input)) as SttSink, Box::new(source) as SttSource)))
    }
}

/// How much audio a reconnecting recognition holds for the next connection: 2 s of 20 ms frames,
/// the oldest dropped first. One bound for every reconnect (this retry, a fallback chain's).
pub const HELD_FRAMES: usize = 100;

/// What the caller pushed, held while no attempt is open (the reference's `_input_ch`).
#[derive(Clone)]
enum Pushed {
    Audio(AudioFrame),
    Flush,
}

#[derive(Default)]
struct Input {
    state: Mutex<InputState>,
}

#[derive(Default)]
struct InputState {
    sink: Option<SttSink>,
    held: VecDeque<Pushed>,
    ended: bool,
}

impl Input {
    /// The next attempt's sink: what arrived meanwhile first, then the end if it came.
    fn install(&self, sink: SttSink) {
        let mut st = self.state.lock().expect("stt input");
        for item in st.held.drain(..) {
            let _ = match item {
                Pushed::Audio(frame) => sink.push(frame),
                Pushed::Flush => sink.flush(),
            };
        }
        if st.ended {
            sink.end_input();
        }
        st.sink = Some(sink);
    }

    fn clear(&self) {
        self.state.lock().expect("stt input").sink = None;
    }

    /// Held while no attempt takes it, a failed attempt's refusal included: the next one hears it.
    fn send(&self, item: Pushed) -> Result<(), ProviderError> {
        let mut st = self.state.lock().expect("stt input");
        if let Some(sink) = &st.sink {
            let sent = match item.clone() {
                Pushed::Audio(frame) => sink.push(frame),
                Pushed::Flush => sink.flush(),
            };
            if sent.is_ok() {
                return Ok(());
            }
            st.sink = None;
        }
        if st.held.len() == HELD_FRAMES {
            st.held.pop_front();
        }
        st.held.push_back(item);
        Ok(())
    }
}

struct InputSink(Arc<Input>);

impl StreamSink<AudioFrame> for InputSink {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.0.send(Pushed::Audio(item))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.0.send(Pushed::Flush)
    }

    fn end_input(&self) {
        let mut st = self.0.state.lock().expect("stt input");
        st.ended = true;
        if let Some(sink) = &st.sink {
            sink.end_input();
        }
    }
}

struct RetrySource {
    stt: Arc<dyn Stt>,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    label: String,
    language: Option<LanguageCode>,
    input: Arc<Input>,
    current: Option<SttSource>,
    /// The open of the current attempt failed.
    failed: Option<ProviderError>,
    retries: u32,
    /// When the current attempt began, wall seconds.
    began: f64,
    /// `start_time_offset`: grows by the time between attempts, so timestamps stay linear.
    offset: f64,
    hook: Option<RetryReport>,
}

impl RetrySource {
    async fn close_current(&mut self) {
        self.input.clear();
        if let Some(source) = self.current.take() {
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            source.close(deadline).await;
        }
    }
}

#[async_trait]
impl StreamSource<SpeechEvent> for RetrySource {
    async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        loop {
            let error = match (self.failed.take(), self.current.as_mut()) {
                (Some(error), _) => error,
                (None, Some(source)) => match source.next().await {
                    Ok(Some(event)) => {
                        if event.kind == SpeechEventType::FinalTranscript {
                            // a recognition succeeded: the budget restarts
                            self.retries = 0;
                        }
                        return Ok(Some(event));
                    }
                    Ok(None) => return Ok(None),
                    Err(error) => error,
                },
                (None, None) => return Ok(None),
            };
            self.close_current().await;
            // An attempt that outlived the connect timeout had connected: the budget restarts.
            if crate::unix_secs(self.clock.wall()) - self.began > self.options.timeout {
                self.retries = 0;
            }
            let failure = Failure {
                kind: RequestKind::SttStream,
                retries: self.retries,
                error,
                report: &*self.stt,
                label: &self.label,
                options: &self.options,
            };
            let delay = failure.decide(self.hook.as_ref())?;
            sleep(&*self.clock, delay).await;
            self.retries += 1;
            let now = crate::unix_secs(self.clock.wall());
            self.offset += now - self.began;
            self.began = now;
            match self.stt.stream(self.language.as_ref(), Some(self.offset)).await {
                Some(Ok((sink, source))) => {
                    self.input.install(sink);
                    self.current = Some(source);
                }
                Some(Err(error)) => self.failed = Some(error),
                None => return Ok(None),
            }
        }
    }

    async fn close(mut self: Box<Self>, deadline: Ts) -> Closed {
        self.input.clear();
        match self.current.take() {
            Some(source) => source.close(deadline).await,
            None => Closed::Clean,
        }
    }

    fn report_retries(&mut self, report: RetryReport) {
        self.hook = Some(report);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::ApiError;

    /// Each attempt's offset and the frames it heard; the first attempt fails when read.
    #[derive(Default)]
    struct Flaky {
        opened: Mutex<Vec<f64>>,
        heard: Arc<Mutex<Vec<usize>>>,
        /// Each reported failure's `recoverable`.
        reports: Mutex<Vec<bool>>,
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
        Final,
        Done,
    }

    #[async_trait]
    impl StreamSource<SpeechEvent> for Attempt {
        async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
            match std::mem::replace(self, Attempt::Done) {
                Attempt::Fail => Err(ApiError::new("hiccup").into()),
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
        ) -> Option<Result<(SttSink, SttSource), ProviderError>> {
            let mut opened = self.opened.lock().unwrap();
            opened.push(offset.unwrap_or_default());
            self.heard.lock().unwrap().push(0);
            let attempt = if opened.len() == 1 { Attempt::Fail } else { Attempt::Final };
            let sink = Count(Arc::clone(&self.heard), opened.len() - 1);
            Some(Ok((Box::new(sink), Box::new(attempt))))
        }
    }

    fn retrying(inner: Arc<Flaky>, max_retry: u32) -> RetryingStt {
        let clock: Arc<dyn Clock> = Arc::new(crate::utils::clock::SimClock::new());
        RetryingStt::for_host(inner, max_retry, 2.0, 10.0, clock).unwrap()
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
}
