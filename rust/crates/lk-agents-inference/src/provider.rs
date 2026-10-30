//! Host provider observations projected onto the shared supplied-time attempt machine.

use lk_agents_abi::errors::{ApiError, ApiErrorKind, ProviderError};
use lk_agents_abi::types::connect::{ApiConnectOptions, ApiConnectOptionsError};
use lk_agents_abi::types::provider::{ProviderAttempts as Port, ResponseObservation};
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::{SpeechEvent, SpeechEventType, Ts};
use lk_agents_core::attempts::{AttemptStep, Attempts, RequestPolicy, terminal_failure};

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

    fn observe_response(&mut self, observation: ResponseObservation, now_ns: u64) {
        match observation {
            ResponseObservation::Audio if !self.input_ended => self.complete_response(),
            ResponseObservation::Progress | ResponseObservation::Audio => {
                self.response(false, now_ns);
            }
            ResponseObservation::Complete => self.complete_response(),
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
        lk_agents_abi::HOST_TEARDOWN_TIMEOUT.as_secs_f64()
    }

    fn cancel(&mut self, now_ns: u64) {
        let mut steps = Vec::new();
        self.attempts.cancel(Ts::from_nanos(now_ns), &mut steps);
        self.permit = false;
    }
}

impl ProviderAttempts {
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
            let mut operation =
                ProviderAttempts::new(kind, ApiConnectOptions::default(), 0).unwrap();
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
            let mut operation =
                ProviderAttempts::new(kind, ApiConnectOptions::default(), 0).unwrap();
            operation.ready(0).unwrap();
            let cancelled = ApiError { status_code: Some(499), ..ApiError::connection(None) };
            assert_eq!(operation.failure(cancelled.into(), false, 0).unwrap(), RetryDecision::Stop);
        }
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

    // Catches: a synthesis read deadline charging the application's input pauses, one that
    // response data or a completed response does not restart or disarm, or audio before the final
    // input keeping it armed (protocols with no per-text acknowledgement).
    #[test]
    fn a_synthesis_read_deadline_runs_only_while_a_response_is_owed() {
        let options =
            ApiConnectOptions { max_retry: 0, read_timeout: Some(2.0), ..Default::default() };
        let mut operation = ProviderAttempts::new(RequestKind::TtsStream, options, 0).unwrap();
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
                let mut operation = ProviderAttempts::new(kind, options, 0).unwrap();
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

    // Catches: a recognition whose provider consumed the end of input and never answered being
    // reconnected: the new attempt has nothing to send, so the timeout must end it, unretryable.
    #[test]
    fn a_response_timeout_after_the_input_ended_is_not_retried() {
        let options =
            ApiConnectOptions { max_retry: 3, read_timeout: Some(1.0), ..Default::default() };
        let mut operation = ProviderAttempts::new(RequestKind::SttStream, options, 0).unwrap();
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
        let mut operation = ProviderAttempts::new(RequestKind::SttStream, options, 0).unwrap();
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
