use crate::telemetry;
use std::ops::ControlFlow;
use std::sync::Arc;

use super::gaps::Gaps;
use super::input::{AudioInput, Flush};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::stt::events::{SpeechEvent, SpeechEventType};
use lk_agents_abi::stt::types::{Stt, SttSink, SttSource};
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::utils::clock::Ts;
use lk_agents_abi::utils::stream::{Closed, Entry, RetryReport, StreamInput, StreamOutput};
use lk_agents_abi::voice::rtc::frame_duration;
use lk_agents_abi::{ApiConnectOptions, AudioFrame, Clock, HOST_TEARDOWN_TIMEOUT, before};
use lk_agents_core::request::{Policy, Request, Retry, Step};
use lk_agents_media::media_queue::{AudioReceiver, Item};

/// Who hears a recognition's failures: each attempt it retries is a recoverable `error` event of
/// its provider; its give-up is the provider's one non-recoverable event, unless a fallback chain
/// opened it, which retries or falls back past it and reports the member's failure itself.
#[derive(Clone, Copy)]
pub(crate) enum GiveUp {
    Reported,
    ByChain,
}

#[derive(Clone, Copy)]
struct Report<'a> {
    stt: &'a dyn Stt,
    give_up: GiveUp,
}

impl Report<'_> {
    fn retrying(&self, error: ProviderError) {
        self.stt.failed(error, true);
    }

    fn gave_up(&self, error: ProviderError) {
        if let GiveUp::Reported = self.give_up {
            self.stt.failed(error, false);
        }
    }
}

/// Connect/write/read are observed by the underlying provider. This owner authorizes retries
/// and the one acceptance deadline across every reopened stream.
fn attempts(
    options: &ApiConnectOptions,
    now: Ts,
    report: Report<'_>,
) -> Result<Request, ProviderError> {
    options.validate().map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
    let mut policy = Policy::from_options(options);
    policy.connect_timeout = None;
    policy.write_timeout = None;
    policy.read_timeout = None;
    let mut steps = Vec::new();
    let attempts = Request::start(policy, now, &mut steps);
    for step in steps {
        if let Step::GiveUp(error) = step {
            let error = ProviderError::from(error);
            report.gave_up(error.clone());
            return Err(error);
        }
    }
    Ok(attempts)
}

/// One failed attempt: reported, then waited out when retried, else the error to raise.
struct Failure<'a> {
    kind: RequestKind,
    attempts: &'a mut Request,
    now: Ts,
    error: ProviderError,
    report: Report<'a>,
    label: &'a str,
    options: &'a ApiConnectOptions,
}

impl Failure<'_> {
    /// Report the core's retry decision; `Err` is what the caller raises.
    fn decide(self, hook: Option<&RetryReport>) -> Result<(), ProviderError> {
        let Failure { kind, attempts, now, error, report, label, options } = self;
        let retries = attempts.retries();
        let adapted = error.attempt_error();
        if let Retry::At(backoff) = attempts.attempt_failed(adapted.clone(), now) {
            let delay = backoff.announce(label, &error).saturating_duration_since(now);
            report.retrying(error.clone());
            if let Some(hook) = hook {
                hook(error.clone(), retries + 1, delay);
            }
            Ok(())
        } else {
            report.gave_up(error.clone());
            if let RetryDecision::Exhausted(message) = lk_agents_core::request::terminal_failure(
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
fn wake_attempts(attempts: &mut Request, now: Ts, report: Report<'_>) -> Result<(), ProviderError> {
    let mut steps = Vec::new();
    attempts.wake(now, None, &mut steps);
    for step in steps {
        if let Step::GiveUp(error) = step {
            let error = ProviderError::from(error);
            report.gave_up(error.clone());
            return Err(error);
        }
    }
    Ok(())
}

/// The runtime's deadline wait completed: the core produces the terminal failure.
fn total_expired(attempts: &mut Request, now: Ts, report: Report<'_>) -> ProviderError {
    wake_attempts(attempts, now, report).expect_err("the operation deadline elapsed")
}

/// Execute the attempt core's next wake; its total deadline may win over the retry backoff.
async fn wait_for_retry(
    attempts: &mut Request,
    clock: &dyn Clock,
    report: Report<'_>,
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

/// The provider each attempt runs on, and the recognition's own retry policy
/// ([`crate::retry::owner`]).
fn attempt_view(
    stt: Arc<dyn Stt>,
    options: ApiConnectOptions,
) -> (Arc<dyn Stt>, ApiConnectOptions) {
    let (given, request) = crate::retry::owner(stt.retries_itself(), options);
    (stt.with_conn_options(&given).unwrap_or(stt), request)
}

/// Retried within `total_timeout`, which spans every attempt and wait.
pub async fn recognize(
    stt: Arc<dyn Stt>,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    cx: &telemetry::Cx,
    buffer: &[AudioFrame],
    language: Option<&LanguageCode>,
) -> Result<SpeechEvent, ProviderError> {
    recognize_reported(stt, options, clock, cx, buffer, language, GiveUp::Reported).await
}

pub(crate) async fn recognize_reported(
    stt: Arc<dyn Stt>,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    cx: &telemetry::Cx,
    buffer: &[AudioFrame],
    language: Option<&LanguageCode>,
    give_up: GiveUp,
) -> Result<SpeechEvent, ProviderError> {
    let (stt, options) = attempt_view(stt, options);
    let report = Report { stt: &*stt, give_up };
    let mut attempts = attempts(&options, clock.now(), report)?;
    let deadline = attempts.wake_at(None);
    match before(
        &*clock,
        deadline,
        recognize_attempts(report, &options, &*clock, cx, buffer, language, &mut attempts),
    )
    .await
    {
        Ok(result) => result,
        Err(_) => Err(total_expired(&mut attempts, clock.now(), report)),
    }
}

pub async fn stream(
    stt: Arc<dyn Stt>,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    cx: &telemetry::Cx,
    language: Option<&LanguageCode>,
    start_time_offset: Option<f64>,
) -> Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError> {
    stream_reported(stt, options, clock, cx, language, start_time_offset, GiveUp::Reported).await
}

pub(crate) async fn stream_reported(
    stt: Arc<dyn Stt>,
    options: ApiConnectOptions,
    clock: Arc<dyn Clock>,
    cx: &telemetry::Cx,
    language: Option<&LanguageCode>,
    start_time_offset: Option<f64>,
    give_up: GiveUp,
) -> Result<Option<lk_agents_abi::DuplexStream<SttSink, SttSource>>, ProviderError> {
    let (stt, options) = attempt_view(stt, options);
    let report = Report { stt: &*stt, give_up };
    let accepted = clock.now();
    let mut attempts = attempts(&options, accepted, report)?;
    let deadline = attempts.wake_at(None);
    wake_attempts(&mut attempts, clock.now(), report)?;
    let offset = start_time_offset.unwrap_or(0.0);
    let gate = lk_agents_net::source::InputGate::default();
    let (input, audio) = AudioInput::new(clock.clone(), gate.clone());
    input.observe(cx.clone());
    let mut pump = Pump { audio, sink: None, sending: None, ended: false, gaps: Gaps::default() };
    let opened = before(&*clock, deadline, stt.stream(cx, language, Some(offset))).await;
    let (current, failed) = match opened {
        Err(_) => return Err(total_expired(&mut attempts, clock.now(), report)),
        Ok(Ok(None)) => return Ok(None),
        Ok(Ok(Some(stream))) => {
            let (sink, source) = stream.split();
            pump.install(sink, offset);
            (Some(source), None)
        }
        Ok(Err(error)) => (None, Some(error)),
    };
    let source = RetrySource {
        stt: Arc::clone(&stt),
        options,
        clock,
        label: stt.label(),
        language: language.cloned(),
        cx: cx.clone(),
        pump,
        current,
        failed,
        sink_failed: None,
        attempts,
        began: accepted,
        offset,
        deadline,
        hook: None,
        give_up,
    };
    // once the recognition gave up, its failure is what the input answers
    Ok(Some(lk_agents_abi::DuplexStream::new(
        Box::new(InputSink(input)) as SttSink,
        Box::new(lk_agents_net::source::Gated::new(source, gate)) as SttSource,
    )))
}

async fn recognize_attempts(
    report: Report<'_>,
    options: &ApiConnectOptions,
    clock: &dyn Clock,
    cx: &telemetry::Cx,
    buffer: &[AudioFrame],
    language: Option<&LanguageCode>,
    attempts: &mut Request,
) -> Result<SpeechEvent, ProviderError> {
    let (stt, label) = (report.stt, report.stt.label());
    loop {
        wake_attempts(attempts, clock.now(), report)?;
        let error = match stt.recognize(cx, buffer, language).await {
            Ok(event) => return Ok(event),
            Err(error) => error,
        };
        let failure = Failure {
            kind: RequestKind::SttBatch,
            attempts,
            now: clock.now(),
            error,
            report,
            label: &label,
            options,
        };
        failure.decide(None)?;
        wait_for_retry(attempts, clock, report).await?;
    }
}

/// A context for one poll of what is already queued: nothing waits for its wake.
fn noop_context() -> std::task::Context<'static> {
    std::task::Context::from_waker(std::task::Waker::noop())
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
        if let Some(sink) = self.sink.take() {
            sink.end_input();
        }
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
        self.0.push(item)
    }

    fn head(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.0.head(item)
    }

    fn filler(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.0.filler(item)
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.0.flush()
    }

    fn end_input(&self) {
        self.0.end();
    }
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
    /// The current attempt's sink failed: what its stream already queued is read first.
    sink_failed: Option<ProviderError>,
    attempts: Request,
    /// When the current attempt began, on the supplied monotonic clock.
    began: Ts,
    /// `start_time_offset`: grows by the time between attempts, so timestamps stay linear.
    offset: f64,
    /// `total_timeout` from the stream's opening: no reconnect or pull extends it.
    deadline: Option<Ts>,
    hook: Option<RetryReport>,
    give_up: GiveUp,
}

impl RetrySource {
    async fn close_current(&mut self) {
        self.pump.clear();
        self.sink_failed = None;
        if let Some(source) = self.current.take() {
            let deadline = self.clock.now().saturating_add(HOST_TEARDOWN_TIMEOUT);
            source.close(deadline).await;
        }
    }

    async fn next_attempted(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        loop {
            let report = Report { stt: &*self.stt, give_up: self.give_up };
            wake_attempts(&mut self.attempts, self.clock.now(), report)?;
            match self.read_current().await {
                Ok(event) => return Ok(event),
                Err(error) => {
                    if self.reopen_after(error).await?.is_break() {
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// The current attempt's next event, or the failure of its open or of its stream.
    async fn read_current(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        let source = match (self.failed.take(), self.current.as_mut()) {
            (Some(error), _) => return Err(error),
            (None, Some(source)) => source,
            (None, None) => return Ok(None),
        };
        let next = loop {
            if let Some(error) = self.sink_failed.take() {
                // What the attempt already queued outlives its failed sink; then the failure.
                break match std::pin::pin!(source.next()).as_mut().poll(&mut noop_context()) {
                    std::task::Poll::Ready(Ok(Some(event))) => {
                        self.sink_failed = Some(error);
                        Ok(Some(event))
                    }
                    std::task::Poll::Ready(Err(failed)) => Err(failed),
                    std::task::Poll::Ready(Ok(None)) | std::task::Poll::Pending => Err(error),
                };
            }
            tokio::select! {
                biased;
                event = source.next() => break event,
                error = self.pump.run() => self.sink_failed = Some(error),
            }
        };
        match next {
            Ok(Some(event)) => {
                if event.kind == SpeechEventType::FinalTranscript {
                    // a recognition succeeded: the budget restarts
                    self.attempts.reset_retries();
                }
                Ok(Some(self.pump.gaps.shift(event)))
            }
            Ok(None) => Ok(None),
            // The provider stopped its own stream: the recognition ends, unreported and
            // never retried.
            Err(error) if error.cancelled() => Ok(None),
            Err(error) => Err(error),
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
            report: Report { stt: &*self.stt, give_up: self.give_up },
            label: &self.label,
            options: &self.options,
        };
        failure.decide(self.hook.as_ref())?;
        let report = Report { stt: &*self.stt, give_up: self.give_up };
        wait_for_retry(&mut self.attempts, &*self.clock, report).await?;
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

impl StreamOutput<SpeechEvent> for RetrySource {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            let (clock, deadline) = (Arc::clone(&self.clock), self.deadline);
            match before(&*clock, deadline, self.next_attempted()).await {
                Ok(next) => next,
                Err(_) => {
                    self.close_current().await;
                    let report = Report { stt: &*self.stt, give_up: self.give_up };
                    Err(total_expired(&mut self.attempts, self.clock.now(), report))
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
