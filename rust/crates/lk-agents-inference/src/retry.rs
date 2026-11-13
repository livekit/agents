use futures_util::FutureExt;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{
    ApiConnectOptions, BoxFuture, Clock, Closed, ProviderStream, StreamOutput, Ts,
};
use lk_agents_core::request::{Policy, Request, Retry};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Each failed attempt: the delay before its retry, `None` once the request gave up, and the
/// retries consumed.
pub type Observer = Arc<dyn Fn(&ProviderError, Option<Duration>, u32) + Send + Sync>;
pub type Open<T> = Arc<
    dyn Fn() -> BoxFuture<'static, Result<Box<dyn ProviderStream<T>>, ProviderError>> + Send + Sync,
>;

/// The one retry owner of a provider request. A provider that retries itself (a fallback chain,
/// a stream adapter retrying each chunk) is handed the whole policy and the request runs once;
/// any other is handed one attempt's view (`max_retry = 0`) and the request retries it. Returns
/// the provider's options, then the request's.
pub fn owner(
    retries_itself: bool,
    options: ApiConnectOptions,
) -> (ApiConnectOptions, ApiConnectOptions) {
    let once = ApiConnectOptions { max_retry: 0, ..options };
    if retries_itself { (options, once) } else { (once, options) }
}

/// No provider wrapper: the request returns this single stream across all its attempts. Each
/// retry is announced as `subject`'s.
pub async fn stream<T: Send + 'static>(
    subject: String,
    options: &ApiConnectOptions,
    clock: Arc<dyn Clock>,
    open: Open<T>,
    delivered: fn(&T) -> bool,
    observer: Observer,
    stop: CancellationToken,
) -> Box<dyn ProviderStream<T>> {
    let mut policy = Policy::from_options(options);
    policy.connect_timeout = None;
    policy.write_timeout = None;
    policy.read_timeout = None;
    let request = Request::start(policy, clock.now(), &mut Vec::new());
    let deadline = options
        .total_timeout
        .map(|seconds| clock.now().saturating_add(std::time::Duration::from_secs_f64(seconds)));
    let initial = lk_agents_abi::before(clock.as_ref(), deadline, open()).await;
    let (source, failed) = match initial {
        Ok(Ok(source)) => (Some(source), None),
        Ok(Err(error)) => (None, Some(error)),
        Err(_) => (
            None,
            Some(
                lk_agents_abi::errors::ApiError::timed_out(lk_agents_abi::errors::Timeout::Total)
                    .with_retryable(false)
                    .into(),
            ),
        ),
    };
    let hook = Arc::new(parking_lot::Mutex::new(None));
    let ledger = source.as_ref().and_then(|source| source.call_ledger());
    Box::new(Output {
        state: Some(State {
            subject,
            request,
            clock,
            open,
            delivered,
            observer,
            source,
            failed,
            ended: false,
            deadline,
            stop: stop.clone(),
            hook: hook.clone(),
        }),
        pending: None,
        stop,
        hook,
        ledger,
    })
}

struct State<T> {
    subject: String,
    request: Request,
    clock: Arc<dyn Clock>,
    open: Open<T>,
    delivered: fn(&T) -> bool,
    observer: Observer,
    source: Option<Box<dyn ProviderStream<T>>>,
    failed: Option<ProviderError>,
    ended: bool,
    deadline: Option<Ts>,
    stop: CancellationToken,
    hook: Arc<parking_lot::Mutex<Option<lk_agents_abi::utils::stream::RetryReport>>>,
}

impl<T: Send + 'static> State<T> {
    async fn next(&mut self) -> Result<Option<T>, ProviderError> {
        if self.ended {
            return Ok(None);
        }
        let (clock, stop, deadline) = (self.clock.clone(), self.stop.clone(), self.deadline);
        let next = tokio::select! {
            biased;
            () = stop.cancelled() => return Ok(None),
            next = lk_agents_abi::before(clock.as_ref(), deadline, self.next_attempted()) => next,
        };
        match next {
            Ok(next) => next,
            Err(_) => {
                self.ended = true;
                let error: ProviderError = lk_agents_abi::errors::ApiError::timed_out(
                    lk_agents_abi::errors::Timeout::Total,
                )
                .with_retryable(false)
                .into();
                (self.observer)(&error, None, self.request.retries());
                Err(error)
            }
        }
    }

    async fn next_attempted(&mut self) -> Result<Option<T>, ProviderError> {
        while !self.ended {
            let result = match (self.failed.take(), &mut self.source) {
                (Some(error), _) => Err(error),
                (None, Some(source)) => source.next().await,
                (None, None) => {
                    self.failed = self.open_source().await.err();
                    continue;
                }
            };
            let error = match result {
                Ok(Some(item)) => {
                    if (self.delivered)(&item) {
                        self.request.delivered();
                    }
                    return Ok(Some(item));
                }
                Ok(None) => {
                    self.ended = true;
                    if let Some(attempt) = self.request.current() {
                        let now = self.clock.now();
                        self.request.succeeded(attempt, now);
                        self.request.closed(attempt, now, &mut Vec::new());
                    }
                    if let Some(source) = self.source.take() {
                        source
                            .close(
                                self.clock
                                    .now()
                                    .saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT),
                            )
                            .await;
                    }
                    return Ok(None);
                }
                Err(error) => error,
            };
            if let Some(source) = self.source.take() {
                source
                    .close(self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT))
                    .await;
            }
            let now = self.clock.now();
            match self.request.attempt_failed(error.attempt_error(), now) {
                Retry::At(backoff) => {
                    let at = backoff.announce(&self.subject, &error);
                    let delay = at.saturating_duration_since(now);
                    (self.observer)(&error, Some(delay), self.request.retries());
                    let hook = self.hook.lock().clone();
                    if let Some(hook) = hook {
                        hook(error.clone(), self.request.retries(), delay);
                    }
                    self.clock.wait_until(at).await;
                    self.request.wake(self.clock.now(), None, &mut Vec::new());
                }
                Retry::GiveUp(_) => {
                    (self.observer)(&error, None, self.request.retries());
                    self.ended = true;
                    return Err(error);
                }
            }
        }
        Ok(None)
    }

    async fn open_source(&mut self) -> Result<(), ProviderError> {
        let mut source = (self.open)().await?;
        let hook = self.hook.lock().clone();
        if let Some(hook) = hook {
            source.report_retries(hook);
        }
        self.source = Some(source);
        Ok(())
    }
}

type Pull<T> =
    BoxFuture<'static, (State<T>, std::thread::Result<Result<Option<T>, ProviderError>>)>;

struct Output<T> {
    state: Option<State<T>>,
    // An interrupted pull retains its provider future rather than opening another attempt.
    pending: Option<Pull<T>>,
    stop: CancellationToken,
    hook: Arc<parking_lot::Mutex<Option<lk_agents_abi::utils::stream::RetryReport>>>,
    ledger: Option<lk_agents_abi::utils::stream::CallLedger>,
}

impl<T: Send + 'static> StreamOutput<T> for Output<T> {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<T>, ProviderError>> {
        Box::pin(async move {
            if self.pending.is_none() {
                let Some(mut state) = self.state.take() else { return Ok(None) };
                // The pull never panics itself: a panic inside ends the stream but keeps its
                // state, so the pull is never resumed and close still closes the source.
                self.pending = Some(Box::pin(async move {
                    let result = AssertUnwindSafe(state.next()).catch_unwind().await;
                    state.ended |= result.is_err();
                    (state, result)
                }));
            }
            let (state, result) = self.pending.as_mut().expect("pending pull").await;
            self.pending = None;
            self.state = Some(state);
            result.unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(mut self: Box<Self>, deadline: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.stop.cancel();
            if let Some(pending) = self.pending.take() {
                let (state, _) = pending.await;
                self.state = Some(state);
            }
            match self.state.take().and_then(|mut state| state.source.take()) {
                Some(source) => source.close(deadline).await,
                None => Closed::Clean,
            }
        })
    }
}

impl<T: Send + 'static> ProviderStream<T> for Output<T> {
    fn started_time(&self) -> Option<f64> {
        self.state.as_ref()?.source.as_ref()?.started_time()
    }
    fn take_call_metrics(
        &mut self,
        in_flight: bool,
    ) -> Option<Vec<lk_agents_abi::metrics::base::AgentMetrics>> {
        if let Some(ledger) = &self.ledger {
            return Some(ledger(in_flight));
        }
        self.state.as_mut()?.source.as_mut()?.take_call_metrics(in_flight)
    }
    fn call_ledger(&self) -> Option<lk_agents_abi::utils::stream::CallLedger> {
        self.ledger.clone().or_else(|| self.state.as_ref()?.source.as_ref()?.call_ledger())
    }
    fn report_retries(&mut self, report: lk_agents_abi::utils::stream::RetryReport) {
        *self.hook.lock() = Some(report.clone());
        if let Some(source) = self.state.as_mut().and_then(|state| state.source.as_mut()) {
            source.report_retries(report);
        }
    }
}

impl<T> Drop for Output<T> {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

#[cfg(test)]
#[path = "retry_tests.rs"]
mod tests;

/// The host's websocket helpers: a provider's close code as a retry decision.
pub struct WebSocketCloses;

impl lk_agents_abi::errors::WebSocketFunctions for WebSocketCloses {
    fn websocket_close_kind(code: Option<u16>) -> lk_agents_abi::errors::CloseKind {
        lk_agents_abi::errors::CloseKind::of(code)
    }
}
