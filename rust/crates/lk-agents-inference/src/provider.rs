//! Host provider observations projected onto the shared supplied-time attempt machine.

use crate::attempts::{AttemptStep, Attempts, RequestPolicy};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::types::connect::{ApiConnectOptions, ApiConnectOptionsError};
use lk_agents_abi::types::provider::ProviderAttempts as Port;
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::{SpeechEvent, SpeechEventType, Ts};

#[derive(Debug)]
pub struct ProviderAttempts {
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

impl Port for ProviderAttempts {
    fn new(
        kind: RequestKind,
        conn_options: ApiConnectOptions,
        now_ns: u64,
    ) -> Result<Self, ApiConnectOptionsError> {
        conn_options.validate()?;
        let mut policy = RequestPolicy::from_options(&conn_options);
        // A host callback does not expose its SDK's connect/write progress. Its total deadline
        // still covers those phases; read is observable only while a recognition awaits a result.
        policy.connect_timeout = None;
        policy.write_timeout = None;
        if kind != RequestKind::SttStream {
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

    fn external_retry_owner(&mut self) {
        self.attempts.disable_retries();
    }

    fn ready(&mut self, now_ns: u64) -> Result<bool, ApiError> {
        self.poll(Ts::from_nanos(now_ns))?;
        Ok(std::mem::take(&mut self.permit))
    }

    fn failure(
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
            return Ok(RetryDecision::Retry(self.retry_delay(now)));
        }
        if self.kind == RequestKind::SttStream
            && self.options.attempt_connected(
                now.saturating_duration_since(self.attempt_started).as_secs_f64(),
            )
        {
            self.attempts.reset_retries();
        }
        let error = error.attempt_error();
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
        Ok(self.kind.terminal_failure(&error, self.options.max_retry, self.attempts.retries()))
    }

    fn wake_at(&self) -> Option<u64> {
        self.attempts.wake_at(self.expecting).map(Ts::as_nanos)
    }

    fn capacity_exceeded(&mut self, input: bool, now_ns: u64) -> Result<(), ApiError> {
        let queue = if input { "input" } else { "event" };
        let error = ApiError::new(format!("recognition {queue} queue capacity exceeded"))
            .with_retryable(false);
        self.error = Some(error.clone());
        self.cancel(now_ns);
        Err(error)
    }

    fn expect_response(&mut self, now_ns: u64) {
        self.expecting.get_or_insert(Ts::from_nanos(now_ns));
    }

    fn input_ended(&mut self, now_ns: u64) {
        self.input_ended = true;
        self.expect_response(now_ns);
    }

    fn observe(&mut self, event: SpeechEvent, now_ns: u64) {
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

    fn succeeded(&mut self, now_ns: u64) -> Result<(), ApiError> {
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

    fn cleanup_timeout(&self) -> f64 {
        crate::attempts::ATTEMPT_CLEANUP_TIMEOUT.as_secs_f64()
    }

    fn cancel(&mut self, now_ns: u64) {
        let mut steps = Vec::new();
        self.attempts.cancel(Ts::from_nanos(now_ns), &mut steps);
        self.permit = false;
    }
}

impl ProviderAttempts {
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
                AttemptStep::Retry { error, .. }
                    if stopped && error.kind == lk_agents_abi::errors::ApiErrorKind::Timeout =>
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

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::errors::TimeoutPhase;

    fn at(seconds: u64) -> u64 {
        seconds * 1_000_000_000
    }

    #[test]
    fn recognition_is_passive_until_a_dequeued_flush_and_final_transcript_clears_the_wait() {
        let options =
            ApiConnectOptions { max_retry: 0, read_timeout: Some(2.0), ..Default::default() };
        let mut operation = ProviderAttempts::new(RequestKind::SttStream, options, 0).unwrap();
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

    #[test]
    fn sdk_timeouts_retry_without_creating_a_second_transport_interruption() {
        let options = ApiConnectOptions { max_retry: 1, ..Default::default() };
        let mut operation = ProviderAttempts::new(RequestKind::Llm, options, 0).unwrap();
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
        let mut operation = ProviderAttempts::new(RequestKind::SttStream, options, 0).unwrap();
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
        let mut operation = ProviderAttempts::new(RequestKind::SttStream, options, 0).unwrap();
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

    #[test]
    fn an_external_retry_owner_keeps_stream_deadlines_and_completion_validation() {
        let options =
            ApiConnectOptions { max_retry: 3, read_timeout: Some(0.0), ..Default::default() };
        let mut operation = ProviderAttempts::new(RequestKind::SttStream, options, 0).unwrap();
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
        let mut operation = ProviderAttempts::new(RequestKind::SttStream, options, 0).unwrap();
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
        let mut operation = ProviderAttempts::new(RequestKind::SttStream, options, 0).unwrap();
        operation.ready(0).unwrap();
        assert!(operation.succeeded(0).is_err());
        operation.input_ended(0);
        operation.observe(SpeechEvent::new(SpeechEventType::InterimTranscript), 0);
        assert!(operation.succeeded(0).is_err());
        operation.response(true, 0);
        assert_eq!(operation.succeeded(0), Ok(()));
    }
}
