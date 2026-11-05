//! The attempts of a host provider's request: [`AttemptMachine`] projects the host's
//! observations onto the shared supplied-time attempt machine, and [`ProviderAttempts`] executes
//! it, running the host's attempt (`_core_attempt`) under its deadlines and retry decisions.

use std::sync::Arc;

use lk_agents_abi::HostRef;
use lk_agents_abi::errors::{ApiError, ApiErrorKind, HostError, ProviderError};
use lk_agents_abi::invoke::CallContext;
use lk_agents_net::task::Cancellation;
use parking_lot::Mutex;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;

use crate::telemetry;
use lk_agents_abi::types::connect::{ApiConnectOptions, ApiConnectOptionsError};
use lk_agents_abi::types::provider::{AttemptHost, ProviderAttempts as Port, ResponseObservation};
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::{SpeechEvent, SpeechEventType, Ts};
use lk_agents_core::attempts::{AttemptStep, Attempts, RequestPolicy, terminal_failure};

/// The decisions of one request's attempts, on supplied time.
#[derive(Debug)]
pub struct AttemptMachine {
    kind: RequestKind,
    options: ApiConnectOptions,
    attempt_started: Ts,
    interrupted: Option<ApiError>,
    attempts: Attempts,
    permit: bool,
    expecting: Option<Ts>,
    input_ended: bool,
    pending_transcript: bool,
    error: Option<ApiError>,
}

impl AttemptMachine {
    pub fn new(
        kind: RequestKind,
        conn_options: ApiConnectOptions,
        now_ns: u64,
    ) -> Result<Self, ApiConnectOptionsError> {
        conn_options.validate()?;
        let mut policy = RequestPolicy::from_options(&conn_options);
        // A host callback does not expose its SDK's connect/write progress. Its total deadline
        // still covers those phases; read is observable only while a response is owed.
        policy.connect_timeout = None;
        policy.write_timeout = None;
        if !matches!(
            kind,
            RequestKind::Llm
                | RequestKind::SttStream
                | RequestKind::TtsBatch
                | RequestKind::TtsStream
        ) {
            policy.read_timeout = None;
        }
        let now = Ts::from_nanos(now_ns);
        let mut steps = Vec::new();
        let attempts = Attempts::start(policy, now, &mut steps);
        let mut result = Self {
            kind,
            options: conn_options,
            attempt_started: now,
            interrupted: None,
            attempts,
            permit: false,
            expecting: None,
            input_ended: false,
            pending_transcript: false,
            error: None,
        };
        result.apply(steps, now);
        result.reading(now);
        Ok(result)
    }

    pub fn external_retry_owner(&mut self) {
        self.attempts.disable_retries();
    }

    pub fn ready(&mut self, now_ns: u64) -> Result<bool, ApiError> {
        self.poll(Ts::from_nanos(now_ns))?;
        Ok(std::mem::take(&mut self.permit))
    }

    pub fn failure(
        &mut self,
        error: ProviderError,
        produced_output: bool,
        now_ns: u64,
    ) -> Result<RetryDecision, ApiError> {
        let now = Ts::from_nanos(now_ns);
        self.poll_machine(now);
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.interrupted.take().is_some() {
            // A read deadline the machine retries replays the request: not after delivered output.
            if produced_output && self.kind != RequestKind::SttStream {
                return Ok(RetryDecision::Raise);
            }
            return Ok(RetryDecision::Retry(self.retry_delay(now)));
        }
        let error = error.attempt_error();
        if self.kind == RequestKind::SttStream
            && self.options.attempt_connected(
                now.saturating_duration_since(self.attempt_started).as_secs_f64(),
                &error,
            )
        {
            self.attempts.reset_retries();
        }
        if matches!(self.kind, RequestKind::Llm | RequestKind::TtsBatch | RequestKind::TtsStream)
            && error.status_code == Some(499)
        {
            self.cancel(now_ns);
            return Ok(RetryDecision::Stop);
        }
        let Some(attempt) = self.attempts.current() else {
            return Ok(RetryDecision::Raise);
        };
        if produced_output && self.kind != RequestKind::SttStream {
            self.attempts.delivered();
        }
        let mut steps = Vec::new();
        self.attempts.failed_error(attempt, error.clone(), now, &mut steps);
        self.attempts.closed(attempt, now, &mut steps);
        let retried = steps.iter().any(|step| matches!(step, AttemptStep::Retry { .. }));
        self.apply(steps, now);
        if retried {
            return Ok(RetryDecision::Retry(self.retry_delay(now)));
        }
        // The callback still owns the original exception; it supplies the terminal cause.
        let error = self.error.take().expect("the attempt owner gave up");
        Ok(terminal_failure(self.kind, &error, self.options.max_retry, self.attempts.retries()))
    }

    pub fn wake_at(&self) -> Option<u64> {
        self.attempts.wake_at(self.expecting).map(Ts::as_nanos)
    }

    pub fn expect_response(&mut self, now_ns: u64) {
        self.expecting.get_or_insert(Ts::from_nanos(now_ns));
    }

    pub fn input_ended(&mut self, now_ns: u64) {
        self.input_ended = true;
        self.expect_response(now_ns);
    }

    pub fn observe(&mut self, event: SpeechEvent, now_ns: u64) {
        match event.kind {
            SpeechEventType::InterimTranscript | SpeechEventType::PreflightTranscript => {
                self.pending_transcript = true;
                self.response(false, now_ns);
            }
            SpeechEventType::EndOfSpeech if self.pending_transcript => self.expect_response(now_ns),
            SpeechEventType::FinalTranscript => self.response(true, now_ns),
            _ => {}
        }
    }

    pub fn observe_response(&mut self, observation: ResponseObservation, now_ns: u64) {
        match observation {
            ResponseObservation::Audio if !self.input_ended => self.complete_response(),
            ResponseObservation::Progress | ResponseObservation::Audio => {
                self.response(false, now_ns);
            }
            ResponseObservation::Complete => self.complete_response(),
        }
    }

    pub fn succeeded(&mut self, now_ns: u64) -> Result<(), ApiError> {
        let now = Ts::from_nanos(now_ns);
        self.poll(now)?;
        if self.kind == RequestKind::SttStream && (!self.input_ended || self.pending_transcript) {
            return Err(ApiError::connection(Some(
                "recognition ended before input and transcript completion".into(),
            ))
            .with_retryable(false));
        }
        if let Some(attempt) = self.attempts.current() {
            self.attempts.succeeded(attempt, now);
            self.attempts.closed(attempt, now, &mut Vec::new());
        }
        Ok(())
    }

    pub fn cleanup_timeout(&self) -> f64 {
        lk_agents_abi::HOST_TEARDOWN_TIMEOUT.as_secs_f64()
    }

    pub fn cancel(&mut self, now_ns: u64) {
        let mut steps = Vec::new();
        self.attempts.cancel(Ts::from_nanos(now_ns), &mut steps);
        self.permit = false;
    }
}

impl AttemptMachine {
    fn complete_response(&mut self) {
        self.expecting = None;
        self.input_ended = false;
    }

    fn response(&mut self, final_response: bool, now_ns: u64) {
        let now = Ts::from_nanos(now_ns);
        if let Some(attempt) = self.attempts.current() {
            self.attempts.progress(attempt, now);
        }
        if final_response {
            self.expecting = None;
            self.pending_transcript = false;
            self.attempts.reset_retries();
        }
    }

    fn reading(&mut self, now: Ts) {
        if let Some(attempt) = self.attempts.current() {
            self.attempts.connected(attempt, now);
            self.attempts.written(attempt, now);
        }
    }

    fn retry_delay(&self, now: Ts) -> f64 {
        self.attempts
            .wake_at(self.expecting)
            .map_or(0.0, |at| at.saturating_duration_since(now).as_secs_f64())
    }

    fn poll_machine(&mut self, now: Ts) {
        let mut steps = Vec::new();
        self.attempts.wake(now, self.expecting, &mut steps);
        self.apply(steps, now);
    }

    fn poll(&mut self, now: Ts) -> Result<(), ApiError> {
        self.poll_machine(now);
        match self.error.as_ref().or(self.interrupted.as_ref()) {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn apply(&mut self, steps: Vec<AttemptStep>, now: Ts) {
        let stopped = steps.iter().any(|step| matches!(step, AttemptStep::Stop(_)));
        for step in steps {
            match step {
                AttemptStep::Launch(_) => {
                    self.permit = true;
                    self.attempt_started = now;
                    self.reading(now);
                }
                // The provider consumed the whole input and never answered: a reconnect has
                // nothing left to send, so the transcript is lost rather than retried.
                AttemptStep::Retry { error, .. }
                    if stopped
                        && error.kind == ApiErrorKind::Timeout
                        && self.input_ended
                        && self.kind == RequestKind::SttStream =>
                {
                    self.permit = false;
                    self.attempts.cancel(now, &mut Vec::new());
                    self.error = Some(error.with_retryable(false));
                }
                AttemptStep::Retry { error, .. }
                    if stopped && error.kind == ApiErrorKind::Timeout =>
                {
                    self.interrupted = Some(error);
                }
                AttemptStep::GiveUp(error) => {
                    self.permit = false;
                    self.error = Some(error);
                }
                AttemptStep::Stop(attempt) => {
                    self.attempts.closed(attempt, now, &mut Vec::new());
                }
                _ => {}
            }
        }
    }
}

/// One host provider request, its attempts executed natively (`livekit.agents.ProviderAttempts`).
///
/// `run` waits until the machine permits an attempt, runs the host's attempt while racing it
/// against the next deadline and the host's observations, and turns a failure into the machine's
/// decision: retry after its delay, raise the attempt's own error, raise `APIConnectionError`
/// once retries are exhausted, or stop. Time is this request's monotonic clock.
pub struct ProviderAttempts {
    host: Arc<dyn AttemptHost>,
    context: CallContext,
    kind: RequestKind,
    epoch: Instant,
    state: Mutex<State>,
    /// Woken by every observation, so a deadline the host moved is re-read.
    changed: Notify,
}

struct State {
    machine: AttemptMachine,
    /// The session owns retries: `run` executes exactly one attempt.
    core: bool,
    /// The running attempt's settlement: `true` once its host task finished.
    inflight: Option<watch::Receiver<bool>>,
    /// Retries were exhausted.
    exhausted: bool,
}

impl Port for ProviderAttempts {
    fn new(
        kind: RequestKind,
        conn_options: ApiConnectOptions,
        host: Arc<dyn AttemptHost>,
        context: CallContext,
    ) -> Result<Self, ApiConnectOptionsError> {
        let machine = AttemptMachine::new(kind, conn_options, 0)?;
        Ok(Self {
            host,
            context,
            kind,
            epoch: Instant::now(),
            state: Mutex::new(State { machine, core: false, inflight: None, exhausted: false }),
            changed: Notify::new(),
        })
    }

    fn core(&self) -> bool {
        self.state.lock().core
    }

    fn set_core(&self, core: bool) {
        let mut state = self.state.lock();
        if core && !state.core {
            state.machine.external_retry_owner();
        }
        state.core = core;
    }

    async fn run(&self, cx: telemetry::Cx) -> Result<Option<HostRef>, ProviderError> {
        let result = self.drive(&cx).await;
        let now = self.now();
        self.state.lock().machine.cancel(now);
        result
    }

    fn expect_response(&self, r#final: bool) {
        self.observed(|machine, now| match r#final {
            true => machine.input_ended(now),
            false => machine.expect_response(now),
        });
    }

    fn input_ended(&self) {
        self.observed(AttemptMachine::input_ended);
    }

    fn progress(&self) {
        self.observed(|machine, now| machine.observe_response(ResponseObservation::Progress, now));
    }

    fn audio_received(&self) {
        self.observed(|machine, now| machine.observe_response(ResponseObservation::Audio, now));
    }

    fn complete_response(&self) {
        self.observed(|machine, now| machine.observe_response(ResponseObservation::Complete, now));
    }

    fn observe(&self, event: SpeechEvent) {
        self.observed(|machine, now| machine.observe(event, now));
    }

    fn exhausted(&self) -> bool {
        self.state.lock().exhausted
    }

    async fn settled(&self) {
        let inflight = self.state.lock().inflight.clone();
        if let Some(mut settled) = inflight {
            let cleanup = std::time::Duration::from_secs_f64(self.cleanup_timeout());
            if tokio::time::timeout(cleanup, settled.wait_for(|done| *done)).await.is_err() {
                stalled_cleanup();
            }
        }
    }

    fn cleanup_timeout(&self) -> f64 {
        lk_agents_abi::HOST_TEARDOWN_TIMEOUT.as_secs_f64()
    }
}

impl ProviderAttempts {
    fn now(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn at(&self, ns: u64) -> Instant {
        self.epoch + std::time::Duration::from_nanos(ns)
    }

    fn observed(&self, observe: impl FnOnce(&mut AttemptMachine, u64)) {
        let now = self.now();
        observe(&mut self.state.lock().machine, now);
        self.changed.notify_one();
    }

    /// The machine's permit for the next attempt, waiting for its backoff.
    async fn permitted(&self) -> Result<(), ProviderError> {
        loop {
            let now = self.now();
            let (ready, wake) = {
                let mut state = self.state.lock();
                (state.machine.ready(now)?, state.machine.wake_at())
            };
            if ready {
                return Ok(());
            }
            let Some(at) = wake else {
                return Err(ApiError::new("provider request has no pending attempt")
                    .with_retryable(false)
                    .into());
            };
            tokio::time::sleep_until(self.at(at)).await;
        }
    }

    async fn drive(&self, cx: &telemetry::Cx) -> Result<Option<HostRef>, ProviderError> {
        if self.state.lock().core {
            self.permitted().await?;
            let value = self.execute(cx, false).await?;
            let now = self.now();
            self.state.lock().machine.succeeded(now)?;
            return Ok(value);
        }
        let mut retrying = false;
        loop {
            self.permitted().await?;
            let error = match self.execute(cx, retrying).await {
                Ok(value) => {
                    let now = self.now();
                    self.state.lock().machine.succeeded(now)?;
                    return Ok(value);
                }
                Err(error) => error,
            };
            let produced = self.host.produced().map_err(host_error)?;
            let now = self.now();
            let decision = self.state.lock().machine.failure(error.clone(), produced, now)?;
            match decision {
                RetryDecision::Stop => return Ok(None),
                RetryDecision::Raise => return Err(error),
                RetryDecision::Exhausted(message) => {
                    let terminal = ApiError {
                        cause: Some(Box::new(error)),
                        ..ApiError::connection(Some(message))
                    };
                    self.state.lock().exhausted = true;
                    return Err(terminal.into());
                }
                RetryDecision::Retry(delay) => {
                    log::warn!(
                        target: "livekit.agents",
                        "{} attempt failed: {error}, retrying in {delay}s",
                        self.kind.as_str()
                    );
                    self.host.retrying(error);
                    retrying = true;
                }
            }
        }
    }

    /// One attempt, cancelled when a deadline passes: time and observations return to the
    /// machine before a result is accepted.
    async fn execute(
        &self,
        cx: &telemetry::Cx,
        retry: bool,
    ) -> Result<Option<HostRef>, ProviderError> {
        let cancel = Arc::new(Cancellation::default());
        // The attempt runs in its own task: a dropped `run` leaves it to the host, which cancels
        // it in the same turn (`_bindings.run_attempts`), and `settled` waits for its cleanup.
        let (done, settled) = watch::channel(false);
        self.state.lock().inflight = Some(settled);
        let (host, context, signal, cx) =
            (self.host.clone(), self.context.clone(), cancel.clone(), cx.clone());
        let mut attempt = tokio::spawn(async move {
            let result = host.attempt(&cx, &context, signal, retry).await;
            done.send_replace(true);
            result
        });
        let joined = |result: Result<_, tokio::task::JoinError>| {
            result.unwrap_or_else(|error| {
                Err(ApiError::new(format!("provider attempt task failed: {error}"))
                    .with_retryable(false)
                    .into())
            })
        };
        let interrupted = loop {
            let wake = self.state.lock().machine.wake_at().map(|at| self.at(at));
            let notified = self.changed.notified();
            tokio::select! {
                biased;
                result = &mut attempt => {
                    let now = self.now();
                    self.state.lock().machine.ready(now)?;
                    return joined(result);
                }
                () = notified => {}
                () = async {
                    match wake {
                        Some(at) => tokio::time::sleep_until(at).await,
                        None => std::future::pending().await,
                    }
                } => {}
            }
            let now = self.now();
            if let Err(error) = self.state.lock().machine.ready(now) {
                break error;
            }
        };
        // An attempt already finishing on its own (a provider's own timeout racing this one) is
        // not interrupted in its cleanup; its outcome is rejected all the same.
        if tokio::time::timeout(DEADLINE_GRACE, &mut attempt).await.is_ok() {
            return Err(interrupted.into());
        }
        cancel.cancel();
        let cleanup = std::time::Duration::from_secs_f64(self.cleanup_timeout());
        if tokio::time::timeout(cleanup, attempt).await.is_err() {
            stalled_cleanup();
        }
        Err(interrupted.into())
    }
}

/// How long an attempt past its deadline may take to finish by itself before it is cancelled.
const DEADLINE_GRACE: std::time::Duration = std::time::Duration::from_millis(50);

fn stalled_cleanup() {
    log::error!(
        target: "livekit.agents",
        "provider cleanup incomplete; its host task retains ownership until settlement"
    );
}

/// A host failure outside the API taxonomy: never retried.
fn host_error(error: HostError) -> ProviderError {
    ProviderError::Host { error, api_kind: None, retryable: false, status_code: None }
}

#[cfg(test)]
#[path = "provider_tests.rs"]
mod executor_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::errors::TimeoutPhase;

    fn at(seconds: u64) -> u64 {
        seconds * 1_000_000_000
    }

    // Replay eligibility belongs to the provider boundary, including delivered audio and 499.
    #[test]
    fn delivered_output_blocks_replay_except_for_recognition_and_499_stops_synthesis() {
        for kind in [
            RequestKind::Llm,
            RequestKind::SttBatch,
            RequestKind::SttStream,
            RequestKind::TtsBatch,
            RequestKind::TtsStream,
            RequestKind::Interruption,
            RequestKind::Avatar,
        ] {
            let mut operation = AttemptMachine::new(kind, ApiConnectOptions::default(), 0).unwrap();
            assert_eq!(operation.ready(0), Ok(true));
            let decision = operation.failure(ApiError::connection(None).into(), true, 0).unwrap();
            assert_eq!(
                decision,
                if kind == RequestKind::SttStream {
                    RetryDecision::Retry(0.1)
                } else {
                    RetryDecision::Raise
                }
            );
        }
        for kind in [RequestKind::Llm, RequestKind::TtsBatch, RequestKind::TtsStream] {
            let mut operation = AttemptMachine::new(kind, ApiConnectOptions::default(), 0).unwrap();
            operation.ready(0).unwrap();
            let cancelled = ApiError { status_code: Some(499), ..ApiError::connection(None) };
            assert_eq!(operation.failure(cancelled.into(), false, 0).unwrap(), RetryDecision::Stop);
        }
    }

    #[test]
    fn recognition_is_passive_until_a_dequeued_flush_and_final_transcript_clears_the_wait() {
        let options =
            ApiConnectOptions { max_retry: 0, read_timeout: Some(2.0), ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::SttStream, options, 0).unwrap();
        assert_eq!(operation.ready(0), Ok(true));
        assert_eq!(operation.ready(at(100)), Ok(false));
        operation.observe(SpeechEvent::new(SpeechEventType::StartOfSpeech), at(100));
        assert_eq!(operation.ready(at(200)), Ok(false));
        operation.expect_response(at(200));
        assert_eq!(operation.ready(at(201)), Ok(false));
        operation.response(true, at(201));
        assert_eq!(operation.ready(at(300)), Ok(false));
        operation.input_ended(at(300));
        let error = operation.ready(at(302)).unwrap_err();
        assert_eq!(error.phase, Some(TimeoutPhase::Read));
    }

    // Catches: a synthesis read deadline charging the application's input pauses, one that
    // response data or a completed response does not restart or disarm, or audio before the final
    // input keeping it armed (protocols with no per-text acknowledgement).
    #[test]
    fn a_synthesis_read_deadline_runs_only_while_a_response_is_owed() {
        let options =
            ApiConnectOptions { max_retry: 0, read_timeout: Some(2.0), ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::TtsStream, options, 0).unwrap();
        assert_eq!(operation.ready(0), Ok(true));
        assert_eq!(operation.ready(at(10)), Ok(false), "idle input owes no response");
        operation.expect_response(at(10));
        operation.observe_response(ResponseObservation::Progress, at(11));
        assert_eq!(operation.ready(at(12)), Ok(false), "progress restarted the deadline");
        operation.observe_response(ResponseObservation::Complete, at(12));
        assert_eq!(operation.ready(at(20)), Ok(false), "a completed response disarms it");
        operation.expect_response(at(20));
        operation.observe_response(ResponseObservation::Audio, at(21));
        assert_eq!(operation.ready(at(30)), Ok(false), "audio settles a non-final request");
        operation.input_ended(at(30));
        operation.observe_response(ResponseObservation::Audio, at(31));
        assert_eq!(operation.ready(at(32)), Ok(false), "after the final input, audio is progress");
        let error = operation.ready(at(33)).unwrap_err();
        assert_eq!((error.kind, error.phase), (ApiErrorKind::Timeout, Some(TimeoutPhase::Read)));
    }

    // Catches: a read deadline after delivered output replaying the request, which would repeat
    // text or audio the person already received. Recognition alone may reconnect.
    #[test]
    fn a_read_deadline_after_delivered_output_raises_except_for_recognition() {
        let options =
            ApiConnectOptions { max_retry: 3, read_timeout: Some(1.0), ..Default::default() };
        for kind in [RequestKind::Llm, RequestKind::TtsStream, RequestKind::SttStream] {
            for delivered in [false, true] {
                let mut operation = AttemptMachine::new(kind, options, 0).unwrap();
                assert_eq!(operation.ready(0), Ok(true));
                operation.expect_response(0);
                let error = operation.ready(at(1)).unwrap_err();
                assert_eq!(error.phase, Some(TimeoutPhase::Read));
                let expected = if delivered && kind != RequestKind::SttStream {
                    RetryDecision::Raise
                } else {
                    RetryDecision::Retry(0.1)
                };
                assert_eq!(
                    operation.failure(error.into(), delivered, at(1)).unwrap(),
                    expected,
                    "{kind:?} delivered={delivered}"
                );
            }
        }
    }

    #[test]
    fn sdk_timeouts_retry_without_creating_a_second_transport_interruption() {
        let options = ApiConnectOptions { max_retry: 1, ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::Llm, options, 0).unwrap();
        let timeout = ApiError::timed_out(TimeoutPhase::Read);
        assert_eq!(operation.ready(0), Ok(true));
        assert_eq!(
            operation.failure(timeout.clone().into(), false, at(1)).unwrap(),
            RetryDecision::Retry(0.1)
        );
        assert_eq!(operation.ready(at(2)), Ok(true), "the SDK failure was already observed");
        assert_eq!(operation.failure(timeout.into(), false, at(2)).unwrap(), RetryDecision::Raise);
    }

    #[test]
    fn final_transcripts_reset_retries_without_restarting_the_total_deadline() {
        let options =
            ApiConnectOptions { max_retry: 1, total_timeout: Some(5.0), ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::SttStream, options, 0).unwrap();
        assert_eq!(operation.ready(0), Ok(true));
        assert_eq!(
            operation.failure(ApiError::connection(None).into(), false, at(1)).unwrap(),
            RetryDecision::Retry(0.1)
        );
        assert_eq!(operation.ready(at(2)), Ok(true));
        operation.response(true, at(2));
        assert_eq!(
            operation.failure(ApiError::connection(None).into(), false, at(3)).unwrap(),
            RetryDecision::Retry(0.1)
        );
        assert_eq!(operation.ready(at(4)), Ok(true));
        operation.response(true, at(4));
        let error = operation.ready(at(5)).unwrap_err();
        assert_eq!(error.phase, Some(TimeoutPhase::Total));
    }

    #[test]
    fn a_read_timeout_spends_one_retry_and_keeps_the_original_total_deadline() {
        let options = ApiConnectOptions {
            max_retry: 1,
            read_timeout: Some(1.0),
            total_timeout: Some(5.0),
            ..Default::default()
        };
        let mut operation = AttemptMachine::new(RequestKind::SttStream, options, 0).unwrap();
        assert_eq!(operation.ready(0), Ok(true));
        operation.expect_response(0);
        assert_eq!(operation.ready(at(1)).unwrap_err().phase, Some(TimeoutPhase::Read));
        assert_eq!(
            operation.failure(ApiError::connection(None).into(), false, at(1)).unwrap(),
            RetryDecision::Retry(0.1)
        );
        assert_eq!(operation.ready(at(1)), Ok(false));
        assert_eq!(operation.ready(at(1) + 100_000_000), Ok(true));
        assert_eq!(
            operation
                .failure(ApiError::connection(None).into(), false, at(1) + 200_000_000)
                .unwrap(),
            RetryDecision::Exhausted("failed to recognize speech after 2 attempts".into())
        );
    }

    // Catches: a recognition whose provider consumed the end of input and never answered being
    // reconnected: the new attempt has nothing to send, so the timeout must end it, unretryable.
    #[test]
    fn a_response_timeout_after_the_input_ended_is_not_retried() {
        let options =
            ApiConnectOptions { max_retry: 3, read_timeout: Some(1.0), ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::SttStream, options, 0).unwrap();
        assert_eq!(operation.ready(0), Ok(true));
        operation.input_ended(0);
        let error = operation.ready(at(1)).unwrap_err();
        assert_eq!((error.phase, error.retryable), (Some(TimeoutPhase::Read), false));
        assert_eq!(operation.failure(error.clone().into(), false, at(1)), Err(error));
        assert_eq!(operation.wake_at(), None, "no attempt follows");
    }

    // Catches: a handshake that times out after the connect limit counted as a connected attempt,
    // restarting the retry budget so a peer that never completes it is retried forever.
    #[test]
    fn a_timed_out_attempt_never_restarts_the_retry_budget() {
        let options =
            ApiConnectOptions { max_retry: 1, connect_timeout: Some(0.5), ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::SttStream, options, 0).unwrap();
        let timeout = ApiError::timed_out(TimeoutPhase::Read);
        assert_eq!(operation.ready(0), Ok(true));
        assert_eq!(
            operation.failure(timeout.clone().into(), false, at(1)).unwrap(),
            RetryDecision::Retry(0.1)
        );
        assert_eq!(operation.ready(at(2)), Ok(true));
        assert_eq!(operation.failure(timeout.into(), false, at(3)).unwrap(), RetryDecision::Raise);
    }

    #[test]
    fn an_external_retry_owner_keeps_stream_deadlines_and_completion_validation() {
        let options =
            ApiConnectOptions { max_retry: 3, read_timeout: Some(0.0), ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::SttStream, options, 0).unwrap();
        operation.external_retry_owner();
        assert_eq!(operation.ready(0), Ok(true));
        assert!(operation.succeeded(0).is_err());
        operation.input_ended(0);
        assert_eq!(operation.ready(0).unwrap_err().phase, Some(TimeoutPhase::Read));
        assert!(operation.failure(ApiError::connection(None).into(), false, 0).is_err());
    }

    #[test]
    fn noise_start_and_end_do_not_require_a_transcript_or_arm_a_read_timer() {
        let options =
            ApiConnectOptions { max_retry: 0, read_timeout: Some(1.0), ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::SttStream, options, 0).unwrap();
        operation.ready(0).unwrap();
        operation.observe(SpeechEvent::new(SpeechEventType::StartOfSpeech), 0);
        operation.observe(SpeechEvent::new(SpeechEventType::EndOfSpeech), 0);
        assert_eq!(operation.ready(at(100)), Ok(false));
        operation.input_ended(at(100));
        assert_eq!(operation.succeeded(at(100)), Ok(()));
    }

    #[test]
    fn recognition_completion_refuses_premature_eof_and_pending_transcripts() {
        let options = ApiConnectOptions { max_retry: 0, ..Default::default() };
        let mut operation = AttemptMachine::new(RequestKind::SttStream, options, 0).unwrap();
        operation.ready(0).unwrap();
        assert!(operation.succeeded(0).is_err());
        operation.input_ended(0);
        operation.observe(SpeechEvent::new(SpeechEventType::InterimTranscript), 0);
        assert!(operation.succeeded(0).is_err());
        operation.response(true, 0);
        assert_eq!(operation.succeeded(0), Ok(()));
    }
}
