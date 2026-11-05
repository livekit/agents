//! The provider attempt executor's contract (docs/models.md "Provider attempts"): one host
//! attempt at a time, retried, raised or stopped as the attempt policy decides, cancelled at a
//! deadline and settled before `settled` returns.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ApiErrorKind, HostError, ProviderError};
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::types::connect::ApiConnectOptions;
use lk_agents_abi::types::provider::{AttemptHost, ProviderAttempts as Port};
use lk_agents_abi::types::retry::RequestKind;
use lk_agents_abi::{BoxFuture, HostRef};
use parking_lot::Mutex;

use super::ProviderAttempts;

/// What one scripted attempt does.
enum Step {
    Succeed,
    Fail(ApiError),
    /// Never answers until cancelled, then takes this long to clean up.
    Hang(Duration),
    /// Asks for a response, then never sends one.
    Silent,
}

#[derive(Default)]
struct Log {
    attempts: Vec<bool>,
    retried: Vec<String>,
    cleaned: usize,
}

struct Host {
    /// What the host's own cancellation of its attempt task (`run_attempts`) delivers.
    cancelled: tokio::sync::Notify,
    steps: Mutex<VecDeque<Step>>,
    produced: bool,
    log: Mutex<Log>,
    attempts: Mutex<Option<Arc<ProviderAttempts>>>,
}

impl Host {
    fn new(steps: impl IntoIterator<Item = Step>, produced: bool) -> Arc<Self> {
        Arc::new(Self {
            cancelled: tokio::sync::Notify::new(),
            steps: Mutex::new(steps.into_iter().collect()),
            produced,
            log: Mutex::default(),
            attempts: Mutex::new(None),
        })
    }
}

impl AttemptHost for Host {
    fn attempt<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _context: &'a CallContext,
        cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
        retry: bool,
    ) -> BoxFuture<'a, Result<Option<HostRef>, ProviderError>> {
        self.log.lock().attempts.push(retry);
        let step = self.steps.lock().pop_front().expect("an attempt the test did not script");
        let attempts = self.attempts.lock().clone();
        Box::pin(async move {
            match step {
                Step::Succeed => Ok(None),
                Step::Fail(error) => Err(error.into()),
                Step::Hang(cleanup) => {
                    self.cancelled.notified().await;
                    tokio::time::sleep(cleanup).await;
                    self.log.lock().cleaned += 1;
                    Err(ApiError::new("cancelled").with_retryable(false).into())
                }
                Step::Silent => {
                    attempts.expect("attempts installed").expect_response(false);
                    cancel.cancelled().await;
                    self.log.lock().cleaned += 1;
                    Err(ApiError::new("cancelled").with_retryable(false).into())
                }
            }
        })
    }

    fn produced(&self) -> Result<bool, HostError> {
        Ok(self.produced)
    }

    fn retrying(&self, error: ProviderError) {
        self.log.lock().retried.push(error.to_string());
    }
}

fn options(max_retry: u32) -> ApiConnectOptions {
    ApiConnectOptions {
        max_retry,
        retry_interval: 0.5,
        read_timeout: Some(1.0),
        ..ApiConnectOptions::default()
    }
}

fn attempts(kind: RequestKind, max_retry: u32, host: &Arc<Host>) -> Arc<ProviderAttempts> {
    let attempts = Arc::new(
        ProviderAttempts::new(kind, options(max_retry), host.clone(), CallContext::default())
            .unwrap(),
    );
    *host.attempts.lock() = Some(attempts.clone());
    attempts
}

fn transient() -> ApiError {
    ApiError::new("transient")
}

// Catches: a retryable failure ending the request, a retry that does not rewind, or the
// recoverable error event missing.
#[tokio::test(start_paused = true)]
async fn a_retryable_failure_runs_the_attempt_again_after_reporting_it() {
    let host = Host::new([Step::Fail(transient()), Step::Succeed], false);
    let result = attempts(RequestKind::Llm, 2, &host).run(lk_agents_abi::telemetry::Cx::NONE).await;
    assert!(matches!(result, Ok(None)));
    let log = host.log.lock();
    assert_eq!(log.attempts, [false, true], "the second attempt is told it is a retry");
    assert_eq!(log.retried, ["transient"]);
}

// Catches: a non-retryable failure being retried, or raised as anything but its own error.
#[tokio::test(start_paused = true)]
async fn a_non_retryable_failure_raises_the_attempts_own_error() {
    let fatal = ApiError::new("fatal").with_retryable(false);
    let host = Host::new([Step::Fail(fatal)], false);
    let error = attempts(RequestKind::Llm, 2, &host)
        .run(lk_agents_abi::telemetry::Cx::NONE)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "fatal");
    let log = host.log.lock();
    assert_eq!(log.attempts.len(), 1);
    assert!(log.retried.is_empty(), "the ending error is the host's to report");
}

// Catches: exhausted retries raising the last error itself, or losing it as the cause.
#[tokio::test(start_paused = true)]
async fn exhausted_retries_raise_a_connection_error_from_the_last_failure() {
    let host = Host::new([Step::Fail(transient()), Step::Fail(transient())], false);
    let attempts = attempts(RequestKind::Llm, 1, &host);
    let error = attempts.run(lk_agents_abi::telemetry::Cx::NONE).await.unwrap_err();
    assert!(attempts.exhausted());
    let ProviderError::Local(error) = error else { panic!("a framework error: {error:?}") };
    assert_eq!(error.kind, ApiErrorKind::Connection);
    assert_eq!(error.cause.as_deref().map(ToString::to_string).as_deref(), Some("transient"));
    assert_eq!(host.log.lock().retried, ["transient"], "only the retried attempt is reported");
}

// Catches: a request that already delivered output being replayed.
#[tokio::test(start_paused = true)]
async fn delivered_output_is_never_replayed() {
    let host = Host::new([Step::Fail(transient())], true);
    let error = attempts(RequestKind::Llm, 3, &host)
        .run(lk_agents_abi::telemetry::Cx::NONE)
        .await
        .unwrap_err();
    assert_eq!(error.to_string(), "transient");
    assert_eq!(host.log.lock().attempts.len(), 1);
}

// Catches: an owed response that never comes hanging the request instead of timing out the
// attempt, cancelling it and retrying.
#[tokio::test(start_paused = true)]
async fn a_read_deadline_cancels_the_attempt_and_retries_it() {
    let host = Host::new([Step::Silent, Step::Succeed], false);
    let result = attempts(RequestKind::Llm, 1, &host).run(lk_agents_abi::telemetry::Cx::NONE).await;
    assert!(matches!(result, Ok(None)), "{result:?}");
    let log = host.log.lock();
    assert_eq!(log.attempts, [false, true]);
    assert_eq!(log.cleaned, 1, "the timed-out attempt settled before the retry");
    assert_eq!(log.retried.len(), 1);
}

// Catches: a session-owned request retrying on its own.
#[tokio::test(start_paused = true)]
async fn inside_a_session_exactly_one_attempt_runs() {
    let host = Host::new([Step::Fail(transient())], false);
    let attempts = attempts(RequestKind::Llm, 3, &host);
    attempts.set_core(true);
    let error = attempts.run(lk_agents_abi::telemetry::Cx::NONE).await.unwrap_err();
    assert_eq!(error.to_string(), "transient");
    assert_eq!(host.log.lock().attempts.len(), 1);
}

// Catches: a cancelled request reporting closed before its attempt cleaned up.
#[tokio::test(start_paused = true)]
async fn settled_waits_for_the_cancelled_attempts_cleanup() {
    let host = Host::new([Step::Hang(Duration::from_millis(200))], false);
    let attempts = attempts(RequestKind::Llm, 0, &host);
    let running = tokio::spawn({
        let attempts = attempts.clone();
        async move { attempts.run(lk_agents_abi::telemetry::Cx::NONE).await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    running.abort();
    let _ = running.await;
    // the host cancels its own attempt, as `_bindings.run_attempts` does
    host.cancelled.notify_one();
    assert_eq!(host.log.lock().cleaned, 0);
    attempts.settled().await;
    assert_eq!(host.log.lock().cleaned, 1);
}
