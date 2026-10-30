//! [`RetryingStt`]: the retry loops of the reference's `STT.recognize` and
//! `RecognizeStream._main_task`, around a host plugin STT bound to its `conn_options`. The plugin
//! base runs one attempt; this reopens the stream on a retried failure, so its consumer sees one
//! stream that survives the attempts.

use std::collections::VecDeque;
use std::sync::Arc;

use lk_agents_net::lock::Mutex;
use tokio::sync::Notify;

use lk_agents_abi::errors::{ProviderError, chained_connection_error};
use lk_agents_abi::stt::HELD_FRAMES;
use lk_agents_abi::stt::events::{SpeechEvent, SpeechEventType};
use lk_agents_abi::stt::stt::{Stt, SttCapabilities, SttSink, SttSource};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::types::identity::ProviderId;
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::utils::clock::Ts;
use lk_agents_abi::utils::stream::{Closed, RetryReport, StreamSink, StreamSource};
use lk_agents_abi::{ApiConnectOptions, AudioFrame, Clock, HOST_TEARDOWN_TIMEOUT, before};
use lk_agents_core::attempts::{AttemptStep, Attempts, RequestPolicy};

pub struct RetryingStt {
    inner: Arc<dyn Stt>,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
}

impl RetryingStt {
    /// A host STT bound to `conn_options` (its proxy's `with_conn_options`), retried under them.
    pub fn bound(
        stt: Arc<dyn Stt>,
        conn_options: &ApiConnectOptions,
        clock: Arc<dyn Clock>,
    ) -> Arc<dyn Stt> {
        Arc::new(RetryingStt { inner: stt, options: conn_options.clone(), clock })
    }
}

/// Connect/write/read are observed by the underlying provider. This owner authorizes retries
/// and the one acceptance deadline across every reopened stream.
fn attempts(
    options: &ApiConnectOptions,
    now: Ts,
    report: &dyn Stt,
) -> Result<Attempts, ProviderError> {
    let mut policy = RequestPolicy::from_options(options);
    policy.connect_timeout = None;
    policy.write_timeout = None;
    policy.read_timeout = None;
    let mut steps = Vec::new();
    let attempts = Attempts::start(policy, now, &mut steps);
    for step in steps {
        if let AttemptStep::GiveUp(error) = step {
            let error = ProviderError::from(error);
            report.failed(error.clone(), false);
            return Err(error);
        }
    }
    Ok(attempts)
}

/// One failed attempt: reported, then waited out when retried, else the error to raise.
struct Failure<'a> {
    kind: RequestKind,
    attempts: &'a mut Attempts,
    now: Ts,
    error: ProviderError,
    report: &'a dyn Stt,
    label: &'a str,
    options: &'a ApiConnectOptions,
}

impl Failure<'_> {
    /// Report the core's retry decision; `Err` is what the caller raises.
    fn decide(self, hook: Option<&RetryReport>) -> Result<(), ProviderError> {
        let Failure { kind, attempts, now, error, report, label, options } = self;
        let Some(id) = attempts.current() else { return Err(error) };
        let retries = attempts.retries();
        let mut steps = Vec::new();
        let adapted = error.attempt_error();
        attempts.failed_error(id, adapted.clone(), now, &mut steps);
        attempts.closed(id, now, &mut steps);
        if steps.iter().any(|step| matches!(step, AttemptStep::Retry { .. })) {
            let delay = attempts.wake_at(None).expect("retry wake").saturating_duration_since(now);
            report.failed(error.clone(), true);
            if let Some(hook) = hook {
                hook(error.clone(), retries + 1, delay);
            }
            log::warn!(target: "livekit.agents", stt = label, streamed = kind == RequestKind::SttStream;
                "failed to recognize speech: {error}, retrying in {}s", delay.as_secs_f64());
            Ok(())
        } else {
            report.failed(error.clone(), false);
            if let RetryDecision::Exhausted(message) = lk_agents_core::attempts::terminal_failure(
                kind,
                &adapted,
                options.max_retry,
                attempts.retries(),
            ) {
                Err(chained_connection_error(message, &error))
            } else {
                Err(error)
            }
        }
    }
}

/// Observe supplied time before executing any provider work, including ready futures.
fn wake_attempts(attempts: &mut Attempts, now: Ts, report: &dyn Stt) -> Result<(), ProviderError> {
    let mut steps = Vec::new();
    attempts.wake(now, None, &mut steps);
    for step in steps {
        if let AttemptStep::GiveUp(error) = step {
            let error = ProviderError::from(error);
            report.failed(error.clone(), false);
            return Err(error);
        }
    }
    Ok(())
}

/// The runtime's deadline wait completed: the core produces the terminal failure.
fn total_expired(attempts: &mut Attempts, now: Ts, report: &dyn Stt) -> ProviderError {
    wake_attempts(attempts, now, report).expect_err("the operation deadline elapsed")
}

/// Execute the attempt core's next wake; its total deadline may win over the retry backoff.
async fn wait_for_retry(
    attempts: &mut Attempts,
    clock: &dyn Clock,
    report: &dyn Stt,
) -> Result<(), ProviderError> {
    loop {
        let at = attempts.wake_at(None).expect("retry has a wake");
        clock.wait_until(at).await;
        wake_attempts(attempts, clock.now(), report)?;
        if attempts.current().is_some() {
            return Ok(());
        }
    }
}

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

    /// Retried within `total_timeout`, which spans every attempt and wait.
    fn recognize(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        let buffer = buffer.to_vec();
        let language = language.cloned();
        Box::pin(async move {
            let mut attempts = attempts(&self.options, self.clock.now(), &*self.inner)?;
            let deadline = attempts.wake_at(None);
            match before(
                &*self.clock,
                deadline,
                self.recognize_attempts(&buffer, language.as_ref(), &mut attempts),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Err(total_expired(&mut attempts, self.clock.now(), &*self.inner)),
            }
        })
    }

    fn stream(
        &self,
        language: Option<&LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<(SttSink, SttSource)>, ProviderError>> {
        let language = language.cloned();
        Box::pin(async move {
            let accepted = self.clock.now();
            let mut attempts = attempts(&self.options, accepted, &*self.inner)?;
            let deadline = attempts.wake_at(None);
            wake_attempts(&mut attempts, self.clock.now(), &*self.inner)?;
            let offset = start_time_offset.unwrap_or(0.0);
            let input = Arc::new(Input::default());
            let opened =
                before(&*self.clock, deadline, self.inner.stream(language.as_ref(), Some(offset)))
                    .await;
            let (current, failed) = match opened {
                Err(_) => return Err(total_expired(&mut attempts, self.clock.now(), &*self.inner)),
                Ok(Ok(None)) => return Ok(None),
                Ok(Ok(Some((sink, source)))) => {
                    input.install(sink);
                    (Some(source), None)
                }
                Ok(Err(error)) => (None, Some(error)),
            };
            let source = RetrySource {
                stt: Arc::clone(&self.inner),
                options: self.options.clone(),
                clock: Arc::clone(&self.clock),
                label: self.inner.label(),
                language: language.clone(),
                input: Arc::clone(&input),
                current,
                failed,
                attempts,
                began: accepted,
                offset,
                deadline,
                hook: None,
            };
            Ok(Some((Box::new(InputSink(input)) as SttSink, Box::new(source) as SttSource)))
        })
    }
}

impl RetryingStt {
    async fn recognize_attempts(
        &self,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
        attempts: &mut Attempts,
    ) -> Result<SpeechEvent, ProviderError> {
        let label = self.inner.label();
        loop {
            wake_attempts(attempts, self.clock.now(), &*self.inner)?;
            let error = match self.inner.recognize(buffer, language).await {
                Ok(event) => return Ok(event),
                Err(error) => error,
            };
            let failure = Failure {
                kind: RequestKind::SttBatch,
                attempts,
                now: self.clock.now(),
                error,
                report: &*self.inner,
                label: &label,
                options: &self.options,
            };
            failure.decide(None)?;
            wait_for_retry(attempts, &*self.clock, &*self.inner).await?;
        }
    }
}

/// What the caller pushed, held while no attempt is open (the reference's `_input_ch`).
#[derive(Clone)]
enum Pushed {
    Audio(AudioFrame),
    Flush,
}

#[derive(Default)]
struct Input {
    state: Mutex<InputState>,
    failed: Notify,
}

#[derive(Default)]
struct InputState {
    sink: Option<SttSink>,
    held: VecDeque<Pushed>,
    ended: bool,
    failure: Option<ProviderError>,
}

impl Input {
    /// The next attempt's sink: what arrived meanwhile first, then the end if it came.
    fn install(&self, sink: SttSink) {
        let mut st = self.state.lock();
        while let Some(item) = st.held.pop_front() {
            let sent = match item.clone() {
                Pushed::Audio(frame) => sink.push(frame),
                Pushed::Flush => sink.flush(),
            };
            if let Err(error) = sent {
                st.held.push_front(item);
                st.failure = Some(error);
                self.failed.notify_one();
                return;
            }
        }
        if st.ended {
            sink.end_input();
        }
        st.sink = Some(sink);
    }

    fn clear(&self) {
        let mut st = self.state.lock();
        st.sink = None;
        st.failure = None;
    }

    async fn failure(&self) -> ProviderError {
        loop {
            let notified = self.failed.notified();
            if let Some(error) = self.state.lock().failure.take() {
                return error;
            }
            notified.await;
        }
    }

    /// Held while no attempt takes it, a failed attempt's refusal included: the next one hears it.
    fn send(&self, item: Pushed) -> Result<(), ProviderError> {
        let mut st = self.state.lock();
        if let Some(sink) = &st.sink {
            let sent = match item.clone() {
                Pushed::Audio(frame) => sink.push(frame),
                Pushed::Flush => sink.flush(),
            };
            match sent {
                Ok(()) => return Ok(()),
                Err(error) => st.failure = Some(error),
            }
            st.sink = None;
            self.failed.notify_one();
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
        let mut st = self.0.state.lock();
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
    attempts: Attempts,
    /// When the current attempt began, on the supplied monotonic clock.
    began: Ts,
    /// `start_time_offset`: grows by the time between attempts, so timestamps stay linear.
    offset: f64,
    /// `total_timeout` from the stream's opening: no reconnect or pull extends it.
    deadline: Option<Ts>,
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

    async fn next_attempted(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        loop {
            wake_attempts(&mut self.attempts, self.clock.now(), &*self.stt)?;
            let error = match (self.failed.take(), self.current.as_mut()) {
                (Some(error), _) => error,
                (None, Some(source)) => match tokio::select! {
                    biased;
                    error = self.input.failure() => Err(error),
                    event = source.next() => event,
                } {
                    Ok(Some(event)) => {
                        if event.kind == SpeechEventType::FinalTranscript {
                            // a recognition succeeded: the budget restarts
                            self.attempts.reset_retries();
                        }
                        return Ok(Some(event));
                    }
                    Ok(None) => return Ok(None),
                    // The provider stopped its own stream: the recognition ends, unreported and
                    // never retried.
                    Err(error) if error.cancelled() => return Ok(None),
                    Err(error) => error,
                },
                (None, None) => return Ok(None),
            };
            self.close_current().await;
            if self.options.attempt_connected(
                self.clock.now().saturating_duration_since(self.began).as_secs_f64(),
                &error.attempt_error(),
            ) {
                self.attempts.reset_retries();
            }
            let failure = Failure {
                kind: RequestKind::SttStream,
                attempts: &mut self.attempts,
                now: self.clock.now(),
                error,
                report: &*self.stt,
                label: &self.label,
                options: &self.options,
            };
            failure.decide(self.hook.as_ref())?;
            wait_for_retry(&mut self.attempts, &*self.clock, &*self.stt).await?;
            let now = self.clock.now();
            self.offset += now.saturating_duration_since(self.began).as_secs_f64();
            self.began = now;
            match self.stt.stream(self.language.as_ref(), Some(self.offset)).await {
                Ok(Some((sink, source))) => {
                    self.input.install(sink);
                    self.current = Some(source);
                }
                Err(error) => self.failed = Some(error),
                Ok(None) => return Ok(None),
            }
        }
    }
}

impl StreamSource<SpeechEvent> for RetrySource {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            let (clock, deadline) = (Arc::clone(&self.clock), self.deadline);
            match before(&*clock, deadline, self.next_attempted()).await {
                Ok(next) => next,
                Err(_) => {
                    self.close_current().await;
                    Err(total_expired(&mut self.attempts, self.clock.now(), &*self.stt))
                }
            }
        })
    }

    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.attempts.cancel(self.clock.now(), &mut Vec::new());
            self.input.clear();
            match self.current.take() {
                Some(source) => source.close(deadline).await,
                None => Closed::Clean,
            }
        })
    }

    fn report_retries(&mut self, report: RetryReport) {
        self.hook = Some(report);
    }
}

#[cfg(test)]
mod tests;
