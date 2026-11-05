//! [`RetryingStt`]: the retry loops of the reference's `STT.recognize` and
//! `RecognizeStream._main_task`, around a host plugin STT bound to its `conn_options`. The plugin
//! base runs one attempt; this reopens the stream on a retried failure, so its consumer sees one
//! stream that survives the attempts.

use crate::telemetry;
use std::ops::ControlFlow;
use std::sync::Arc;

use super::gaps::Gaps;
use super::input::{AudioInput, Flush};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::stt::events::{SpeechEvent, SpeechEventType};
use lk_agents_abi::stt::types::{Stt, SttCapabilities, SttSink, SttSource};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::types::identity::ProviderId;
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::utils::clock::Ts;
use lk_agents_abi::utils::stream::{Closed, Entry, RetryReport, StreamInput, StreamOutput};
use lk_agents_abi::voice::rtc::frame_duration;
use lk_agents_abi::{ApiConnectOptions, AudioFrame, Clock, HOST_TEARDOWN_TIMEOUT, before};
use lk_agents_core::attempts::{AttemptStep, Attempts, RequestPolicy};
use lk_agents_media::media_queue::{AudioReceiver, Item};

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
        Arc::new(RetryingStt { inner: stt, options: *conn_options, clock })
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
                // `raise APIConnectionError(message) from error`.
                Err(ProviderError::Local(ApiError {
                    cause: Some(Box::new(error)),
                    ..ApiError::connection(Some(message))
                }))
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
        cx: &telemetry::Cx,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        let (cx, buffer, language) = (cx.clone(), buffer.to_vec(), language.cloned());
        Box::pin(async move {
            let mut attempts = attempts(&self.options, self.clock.now(), &*self.inner)?;
            let deadline = attempts.wake_at(None);
            match before(
                &*self.clock,
                deadline,
                self.recognize_attempts(&cx, &buffer, language.as_ref(), &mut attempts),
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
        cx: &telemetry::Cx,
        language: Option<&LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError>,
    > {
        let language = language.cloned();
        let cx = cx.clone();
        Box::pin(async move {
            let accepted = self.clock.now();
            let mut attempts = attempts(&self.options, accepted, &*self.inner)?;
            let deadline = attempts.wake_at(None);
            wake_attempts(&mut attempts, self.clock.now(), &*self.inner)?;
            let offset = start_time_offset.unwrap_or(0.0);
            let (input, audio) = AudioInput::new();
            input.observe(cx.clone());
            let mut pump =
                Pump { audio, sink: None, sending: None, ended: false, gaps: Gaps::default() };
            let opened = before(
                &*self.clock,
                deadline,
                self.inner.stream(&cx, language.as_ref(), Some(offset)),
            )
            .await;
            let (current, failed) = match opened {
                Err(_) => return Err(total_expired(&mut attempts, self.clock.now(), &*self.inner)),
                Ok(Ok(None)) => return Ok(None),
                Ok(Ok(Some(stream))) => {
                    let (sink, source) = stream.split();
                    pump.install(sink, offset);
                    (Some(source), None)
                }
                Ok(Err(error)) => (None, Some(error)),
            };
            let source = RetrySource {
                stt: Arc::clone(&self.inner),
                options: self.options,
                clock: Arc::clone(&self.clock),
                label: self.inner.label(),
                language: language.clone(),
                cx,
                pump,
                current,
                failed,
                attempts,
                began: accepted,
                offset,
                deadline,
                hook: None,
            };
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(InputSink(input)) as SttSink,
                Box::new(source) as SttSource,
            )))
        })
    }
}

impl RetryingStt {
    async fn recognize_attempts(
        &self,
        cx: &telemetry::Cx,
        buffer: &[AudioFrame],
        language: Option<&LanguageCode>,
        attempts: &mut Attempts,
    ) -> Result<SpeechEvent, ProviderError> {
        let label = self.inner.label();
        loop {
            wake_attempts(attempts, self.clock.now(), &*self.inner)?;
            let error = match self.inner.recognize(cx, buffer, language).await {
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

/// What the caller pushed, handed to the attempt serving one entry at a time, each awaited: a slow
/// attempt, or none open, leaves the backlog in the queue, whose policy bounds it.
struct Pump {
    audio: AudioReceiver<Flush>,
    sink: Option<Arc<dyn StreamInput<AudioFrame>>>,
    sending: Option<lk_agents_abi::BoxFuture<'static, Result<(), ProviderError>>>,
    ended: bool,
    gaps: Gaps,
}

impl Pump {
    /// The next attempt's sink, its audio starting at `offset`; told at once if the input ended.
    fn install(&mut self, sink: SttSink, offset: f64) {
        let sink: Arc<dyn StreamInput<AudioFrame>> = Arc::from(sink);
        self.sending = self.ended.then(|| sink.send(Entry::EndInput));
        self.sink = Some(sink);
        self.gaps = Gaps::starting(offset);
    }

    fn clear(&mut self) {
        self.sink = None;
        self.sending = None;
    }

    /// Feeds the attempt until its sink fails. Cancel-safe: an entry in flight stays in flight.
    async fn run(&mut self) -> ProviderError {
        loop {
            if let Some(sending) = &mut self.sending {
                let sent = sending.await;
                self.sending = None;
                if let Err(error) = sent {
                    return error;
                }
            }
            let Some(sink) = self.sink.clone() else { return std::future::pending().await };
            let entry = match self.audio.recv().await {
                Some(Item::Media(frame)) => {
                    self.gaps.audio(frame_duration(&frame));
                    Entry::Item(frame)
                }
                Some(Item::Gap(gap)) => {
                    self.gaps.gap(gap);
                    continue;
                }
                Some(Item::Event(Flush)) => Entry::Flush,
                None if !self.ended => {
                    self.ended = true;
                    Entry::EndInput
                }
                None => return std::future::pending().await,
            };
            self.sending = Some(sink.send(entry));
        }
    }
}

struct InputSink(AudioInput);

impl StreamInput<AudioFrame> for InputSink {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        if self.0.push(item) { Ok(()) } else { Err(input_closed()) }
    }

    fn head(&self, item: AudioFrame) -> Result<(), ProviderError> {
        if self.0.head(item) { Ok(()) } else { Err(input_closed()) }
    }

    fn flush(&self) -> Result<(), ProviderError> {
        match self.0.flush() {
            Some(flushed) => flushed.map_err(|stuck| ApiError::new(stuck.to_string()).into()),
            None => Err(input_closed()),
        }
    }

    fn end_input(&self) {
        self.0.end();
    }
}

fn input_closed() -> ProviderError {
    ApiError::new("recognition input is closed").with_retryable(false).into()
}

struct RetrySource {
    stt: Arc<dyn Stt>,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    label: String,
    language: Option<LanguageCode>,
    /// The span every reconnect opens under.
    cx: telemetry::Cx,
    pump: Pump,
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
        self.pump.clear();
        if let Some(source) = self.current.take() {
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            source.close(deadline).await;
        }
    }

    async fn next_attempted(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        loop {
            wake_attempts(&mut self.attempts, self.clock.now(), &*self.stt)?;
            match self.read_current().await {
                Read::Deliver(event) => return Ok(event),
                Read::Failed(error) => {
                    if self.reopen_after(error).await?.is_break() {
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// The current attempt's next event, or the failure of its open or of its stream.
    async fn read_current(&mut self) -> Read {
        let source = match (self.failed.take(), self.current.as_mut()) {
            (Some(error), _) => return Read::Failed(error),
            (None, Some(source)) => source,
            (None, None) => return Read::Deliver(None),
        };
        let next = tokio::select! {
            biased;
            error = self.pump.run() => Err(error),
            event = source.next() => event,
        };
        match next {
            Ok(Some(event)) => {
                if event.kind == SpeechEventType::FinalTranscript {
                    // a recognition succeeded: the budget restarts
                    self.attempts.reset_retries();
                }
                Read::Deliver(Some(self.pump.gaps.shift(event)))
            }
            Ok(None) => Read::Deliver(None),
            // The provider stopped its own stream: the recognition ends, unreported and
            // never retried.
            Err(error) if error.cancelled() => Read::Deliver(None),
            Err(error) => Read::Failed(error),
        }
    }

    /// Closes the failed attempt and, when the policy allows, opens the next one after its wait.
    /// Breaks when the provider has no stream to give.
    async fn reopen_after(
        &mut self,
        error: ProviderError,
    ) -> Result<ControlFlow<()>, ProviderError> {
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
        match self.stt.stream(&self.cx, self.language.as_ref(), Some(self.offset)).await {
            Ok(Some(stream)) => {
                let (sink, source) = stream.split();
                self.pump.install(sink, self.offset);
                self.current = Some(source);
            }
            Err(error) => self.failed = Some(error),
            Ok(None) => return Ok(ControlFlow::Break(())),
        }
        Ok(ControlFlow::Continue(()))
    }
}

/// What reading the current attempt gave.
enum Read {
    /// An event for the caller, or the end of the recognition.
    Deliver(Option<SpeechEvent>),
    /// The attempt failed: a retry may follow.
    Failed(ProviderError),
}

impl StreamOutput<SpeechEvent> for RetrySource {
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
            self.pump.clear();
            match self.current.take() {
                Some(source) => source.close(deadline).await,
                None => Closed::Clean,
            }
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<SpeechEvent> for RetrySource {
    fn report_retries(&mut self, report: RetryReport) {
        self.hook = Some(report);
    }
}

#[cfg(test)]
#[path = "retrying_tests.rs"]
mod tests;
