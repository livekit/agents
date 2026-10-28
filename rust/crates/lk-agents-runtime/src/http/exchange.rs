//! A demand-driven HTTP response over the shared request-attempt lifecycle. Dropping an
//! attempt releases its reqwest exchange directly; no detached attempt task owns I/O.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::{ApiConnectOptions, Clock, Ts};
use lk_agents_inference::attempts::{AttemptFailure, AttemptStep, Attempts, RequestPolicy};
use lk_agents_net::transport::HttpRequest;
use tokio::sync::mpsc;

use super::{GatewayHttp, Reporter, Sent, Stamped, error_chain};

type Sending = Pin<Box<dyn Future<Output = Result<reqwest::Response, AttemptFailure>> + Send>>;
type Body = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

pub(crate) enum HttpEvent {
    /// A retry discards response data that has not been delivered to the consumer.
    Started,
    Headers(reqwest::header::HeaderMap),
    Bytes(Bytes),
    End,
}

pub(crate) struct Exchange {
    core: Attempts,
    http: Arc<GatewayHttp>,
    request: HttpRequest,
    method: reqwest::Method,
    reporter: Reporter<(u32, Sent)>,
    observations: mpsc::Receiver<Stamped<(u32, Sent)>>,
    sending: Option<Sending>,
    body: Option<Body>,
    error: Option<ApiError>,
    started: bool,
    ended: bool,
    now: Ts,
}

impl Exchange {
    pub(crate) fn new(
        http: Arc<GatewayHttp>,
        method: reqwest::Method,
        request: HttpRequest,
        options: &ApiConnectOptions,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let now = clock.now();
        let (tx, observations) = mpsc::channel(2);
        let mut steps = Vec::new();
        let core = Attempts::start(RequestPolicy::from_options(options), now, &mut steps);
        let mut exchange = Self {
            core,
            http,
            request,
            method,
            reporter: Reporter { tx, clock },
            observations,
            sending: None,
            body: None,
            error: None,
            started: false,
            ended: false,
            now,
        };
        exchange.apply(steps);
        exchange
    }

    pub(crate) fn delivered(&mut self) {
        self.core.delivered();
    }

    pub(crate) async fn next(&mut self) -> Result<HttpEvent, ApiError> {
        let expecting = self.reporter.clock.now();
        loop {
            // Observations already queued precede a wall-clock wake. This matters when a
            // consumer briefly stops pulling: transport progress keeps its original time.
            while let Ok(observed) = self.observations.try_recv() {
                self.observed(observed, expecting);
            }
            self.wake(self.reporter.clock.now(), Some(expecting));
            if let Some(error) = self.error.take() {
                return Err(error);
            }
            if self.started {
                self.started = false;
                return Ok(HttpEvent::Started);
            }
            if self.ended {
                return Ok(HttpEvent::End);
            }
            let wake = self.core.wake_at(Some(expecting));
            let clock = self.reporter.clock.clone();
            let sending = &mut self.sending;
            let body = &mut self.body;
            let is_sending = sending.is_some();
            let is_reading = body.is_some();
            tokio::select! {
                biased;
                Some(observed) = self.observations.recv() => self.observed(observed, expecting),
                () = async { clock.wait_until(wake.unwrap_or(Ts::from_nanos(u64::MAX))).await }, if wake.is_some() => {},
                result = async { sending.as_mut().expect("sending guarded").await }, if is_sending => {
                    self.sending = None;
                    while let Ok(observed) = self.observations.try_recv() {
                        self.observed(observed, expecting);
                    }
                    let now = self.reporter.clock.now();
                    self.wake(now, Some(expecting));
                    if self.core.current().is_none() { continue; }
                    match result {
                        Ok(response) => {
                            self.core.progress(self.core.current().expect("active attempt"), now);
                            let headers = response.headers().clone();
                            self.body = Some(Box::pin(response.bytes_stream()));
                            return Ok(HttpEvent::Headers(headers));
                        }
                        Err(failure) => self.failed(failure, now),
                    }
                },
                chunk = async { body.as_mut().expect("body guarded").next().await }, if is_reading => {
                    let now = self.reporter.clock.now();
                    self.wake(now, Some(expecting));
                    let Some(attempt) = self.core.current() else { continue; };
                    match chunk {
                        Some(Ok(bytes)) => {
                            self.core.progress(attempt, now);
                            return Ok(HttpEvent::Bytes(bytes));
                        }
                        Some(Err(error)) => self.failed(AttemptFailure::Connection(error_chain(&error)), now),
                        None => {
                            self.core.succeeded(attempt, now);
                            let mut steps = Vec::new();
                            self.core.closed(attempt, now, &mut steps);
                            self.apply(steps);
                            self.body = None;
                            self.ended = true;
                            return Ok(HttpEvent::End);
                        }
                    }
                },
            }
        }
    }

    /// Consumer backpressure suspends remote read inactivity, while the operation's total
    /// deadline keeps running under the same request core.
    pub(crate) async fn wait_for_consumer<T>(
        &mut self,
        delivery: impl Future<Output = T>,
    ) -> Result<T, ApiError> {
        tokio::pin!(delivery);
        loop {
            self.wake(self.reporter.clock.now(), None);
            if let Some(error) = self.error.take() {
                return Err(error);
            }
            let wake = self.core.wake_at(None);
            let clock = self.reporter.clock.clone();
            tokio::select! {
                biased;
                () = async { clock.wait_until(wake.unwrap_or(Ts::from_nanos(u64::MAX))).await }, if wake.is_some() => {},
                result = &mut delivery => return Ok(result),
            }
        }
    }

    fn observed(&mut self, observed: Stamped<(u32, Sent)>, expecting: Ts) {
        self.wake(observed.at, Some(expecting));
        let (attempt, sent) = observed.event;
        match sent {
            Sent::Connected => self.core.connected(attempt, self.now),
            Sent::RequestWritten => self.core.written(attempt, self.now),
        }
    }

    fn wake(&mut self, at: Ts, expecting: Option<Ts>) {
        self.now = at.max(self.now);
        let mut steps = Vec::new();
        self.core.wake(self.now, expecting, &mut steps);
        self.apply(steps);
    }

    fn failed(&mut self, failure: AttemptFailure, now: Ts) {
        let Some(attempt) = self.core.current() else { return };
        let mut steps = Vec::new();
        self.core.failed(attempt, failure, now, &mut steps);
        self.core.closed(attempt, now, &mut steps);
        self.body = None;
        self.apply(steps);
    }

    fn apply(&mut self, steps: Vec<AttemptStep>) {
        for step in steps {
            match step {
                AttemptStep::Launch(attempt) => {
                    let http = self.http.clone();
                    let request = self.request.clone();
                    let reporter = self.reporter.clone();
                    let method = self.method.clone();
                    self.sending = Some(Box::pin(async move {
                        http.send_method(method, request, &reporter, |sent| (attempt, sent)).await
                    }));
                    self.started = true;
                }
                AttemptStep::Stop(attempt) => {
                    self.sending = None;
                    self.body = None;
                    let mut released = Vec::new();
                    self.core.closed(attempt, self.now, &mut released);
                    self.apply(released);
                }
                AttemptStep::GiveUp(error) => self.error = Some(error),
                AttemptStep::Retry { .. } => {}
                AttemptStep::CleanupIncomplete(_) | AttemptStep::Unresolved => {
                    unreachable!("directly owned HTTP exchanges release synchronously")
                }
            }
        }
    }
}
