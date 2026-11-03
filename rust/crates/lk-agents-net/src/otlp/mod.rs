//! The OTLP exporter: every span, report record and measurement reaches LiveKit Cloud and the
//! application's endpoint from here (docs/internals/telemetry.md "Destinations").
//!
//! Producers hand records to [`Exporter`] through `RecordSink`, which never blocks: each
//! destination copies the record into a bounded queue, or drops and counts it. An owned task
//! exports full batches as they fill, everything queued each second and metrics every 30 s.
//!
//! - [`config`]: identities, the Cloud endpoint and recording decision, the application endpoint.
//! - `destination`: the job registry, queues, holding and batch selection.
//! - `encode`: content policy, identity stamps and the OTLP request bodies.
//! - `metrics`: per-destination aggregation, delta or cumulative.
//! - `json`: the OTLP/JSON mapping of the request messages.
//! - [`proto`]: the OTLP messages, generated from opentelemetry-proto (`proto_tests.rs`).

pub mod config;
mod destination;
mod encode;
mod json;
mod metrics;
pub mod proto;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use futures_util::StreamExt as _;
use lk_agents_abi::telemetry::record::{JobKey, Record, RecordSink};
use lk_agents_abi::telemetry::report::UploadGate as _;
use lk_agents_abi::{Clock, OwnedTask, Spawner, SpawnerExt as _, Ts, before, timeout};
use lk_agents_core::telemetry::report::records::UploadGate;
use parking_lot::Mutex;
use tokio::sync::{Notify, OnceCell};

pub use config::{
    AppConfig, CloudEndpoint, CloudRecording, JobIdentity, ProcessIdentity, Protocol, SdkName,
    Simulation, Temporality,
};
pub use destination::Limits;
use destination::{Batch, Payload, State, Take};
use lk_agents_core::telemetry::policy::ContentPolicy;

const SPAN_INTERVAL: Duration = Duration::from_secs(1);
const METRIC_INTERVAL: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const RESPONSE_BYTES: usize = 64 * 1024;

/// What the exporter could not do, since the process started.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Diagnostics {
    /// Records refused by a full queue or a metric cardinality limit, or abandoned at a deadline.
    pub dropped: u64,
    /// Records in requests the endpoint refused or that failed in transport.
    pub failed: u64,
    /// Configuration that could not apply: a second Cloud endpoint, a job opened twice, a
    /// decision for a job that is not open.
    pub conflicts: u64,
    /// Job closes and shutdowns that reached their deadline before every record was sent.
    pub incomplete_cleanup: u64,
    /// Jobs whose project refused recording.
    pub recording_disabled: u64,
}

#[derive(Default)]
struct Counters {
    dropped: AtomicU64,
    failed: AtomicU64,
    conflicts: AtomicU64,
    incomplete_cleanup: AtomicU64,
    recording_disabled: AtomicU64,
}

fn add(counter: &AtomicU64, n: usize) {
    if n > 0 {
        counter.fetch_add(n as u64, Ordering::Relaxed);
    }
}

struct Shared {
    state: Mutex<State>,
    clock: Arc<dyn Clock>,
    counters: Counters,
    /// A queue filled a batch.
    wake: Notify,
    /// Exports in progress, and their settling.
    in_flight: AtomicUsize,
    idle: Notify,
    client: OnceCell<reqwest::Client>,
}

/// The process's telemetry exporter.
pub struct Exporter {
    shared: Arc<Shared>,
    task: Mutex<Option<OwnedTask<()>>>,
}

impl Exporter {
    /// Starts the periodic export on `spawner`, timed by `clock`.
    pub fn start(
        spawner: &dyn Spawner,
        clock: Arc<dyn Clock>,
        process: ProcessIdentity,
        limits: Limits,
    ) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::new(process, limits)),
            clock,
            counters: Counters::default(),
            wake: Notify::new(),
            in_flight: AtomicUsize::new(0),
            idle: Notify::new(),
            client: OnceCell::new(),
        });
        let task = spawner.spawn(run(shared.clone()));
        Exporter { shared, task: Mutex::new(Some(task)) }
    }

    /// Enables Cloud export. The first endpoint stays; a different one is a conflict.
    pub fn configure_cloud(&self, endpoint: CloudEndpoint) {
        let mut state = self.shared.state.lock();
        match &state.cloud {
            None => state.cloud = Some(endpoint),
            Some(current) if *current == endpoint => {}
            Some(_) => add(&self.shared.counters.conflicts, 1),
        }
    }

    /// Replaces the application endpoint for subsequent records; `None` disables it. What the
    /// previous configuration queued is still sent there.
    pub fn configure_app(&self, config: Option<AppConfig>) {
        let now = self.shared.wall_nanos();
        self.shared.state.lock().configure_app(config, now);
    }

    /// A job started: its identity stamps its records, and Cloud holds its spans until
    /// [`Exporter::register_job`].
    pub fn open_job(&self, key: JobKey, identity: JobIdentity) {
        let now = self.shared.wall_nanos();
        if !self.shared.state.lock().open_job(key, identity, now) {
            add(&self.shared.counters.conflicts, 1);
        }
    }

    /// The job's Cloud recording decision: held spans are released or dropped with it.
    pub fn register_job(&self, key: &JobKey, recording: CloudRecording) {
        let now = self.shared.wall_nanos();
        if !self.shared.state.lock().register_job(key, recording, now) {
            add(&self.shared.counters.conflicts, 1);
        }
    }

    /// Sends what the job has queued, its last metrics included, then forgets it. Bounded by
    /// `limit`: what is not sent by then is dropped and counted.
    pub async fn close_job(&self, key: &JobKey, limit: Duration) {
        let shared = &*self.shared;
        let deadline = shared.clock.now().saturating_add(limit);
        let now = shared.wall_nanos();
        let batches =
            shared.state.lock().take(Take { job: Some(key), metrics: true, all: true }, now);
        let complete = shared.send(batches, Some(deadline)).await && shared.settled(deadline).await;
        add(&shared.counters.dropped, shared.state.lock().remove_job(key));
        if !complete {
            add(&shared.counters.incomplete_cleanup, 1);
        }
    }

    /// Sends everything queued, metrics included. False when `limit` passed first.
    pub async fn flush(&self, limit: Duration) -> bool {
        let shared = &*self.shared;
        let deadline = shared.clock.now().saturating_add(limit);
        let now = shared.wall_nanos();
        let batches = shared.state.lock().take(Take { job: None, metrics: true, all: true }, now);
        shared.send(batches, Some(deadline)).await && shared.settled(deadline).await
    }

    /// The process-exit flush: refuses new records, sends everything queued within `limit` and
    /// stops the periodic export. Later calls return at once.
    pub async fn shutdown(&self, limit: Duration) {
        {
            let mut state = self.shared.state.lock();
            if state.closed {
                return;
            }
            state.closed = true;
        }
        let complete = self.flush(limit).await;
        drop(self.task.lock().take());
        if !complete {
            add(&self.shared.counters.incomplete_cleanup, 1);
        }
    }

    pub fn diagnostics(&self) -> Diagnostics {
        let c = &self.shared.counters;
        let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        Diagnostics {
            dropped: read(&c.dropped),
            failed: read(&c.failed),
            conflicts: read(&c.conflicts),
            incomplete_cleanup: read(&c.incomplete_cleanup),
            recording_disabled: read(&c.recording_disabled),
        }
    }
}

impl RecordSink for Exporter {
    fn record(&self, record: Record) {
        let bytes = encode::charge(&record);
        let admitted = self.shared.state.lock().record(record, bytes);
        add(&self.shared.counters.dropped, admitted.dropped as usize);
        if admitted.batch_ready {
            self.shared.wake.notify_one();
        }
    }
}

/// Full batches when one fills, everything queued every second, metrics every 30 s.
async fn run(shared: Arc<Shared>) {
    let clock = shared.clock.clone();
    let mut next_metrics = clock.now().saturating_add(METRIC_INTERVAL);
    loop {
        let at = clock.now().saturating_add(SPAN_INTERVAL).min(next_metrics);
        let woken = tokio::select! {
            biased;
            _ = shared.wake.notified() => true,
            _ = clock.wait_until(at) => false,
        };
        let now = clock.now();
        let metrics = now >= next_metrics;
        if metrics {
            next_metrics = now.saturating_add(METRIC_INTERVAL);
        }
        let wall = shared.wall_nanos();
        let batches = shared.state.lock().take(Take { job: None, metrics, all: !woken }, wall);
        shared.send(batches, None).await;
    }
}

enum Outcome {
    Sent,
    Failed(String),
    /// The project disabled recording (Cloud only).
    Disabled,
}

/// Counts an export in progress until dropped, cancellation included.
struct InFlight<'a>(&'a Shared);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if self.0.in_flight.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle.notify_waiters();
        }
    }
}

impl Shared {
    fn wall_nanos(&self) -> u64 {
        let since = self.clock.wall().duration_since(SystemTime::UNIX_EPOCH);
        since.map_or(0, |d| d.as_nanos().min(u64::MAX as u128) as u64)
    }

    /// Sends each batch in turn. False when `deadline` cut it short; the rest is dropped.
    async fn send(&self, batches: Vec<Batch>, deadline: Option<Ts>) -> bool {
        if batches.is_empty() {
            return true;
        }
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        let _in_flight = InFlight(self);
        let mut complete = true;
        for batch in batches {
            let records = batch.payload().records();
            let limit = match deadline {
                None => REQUEST_TIMEOUT,
                Some(at) => at.saturating_duration_since(self.clock.now()).min(REQUEST_TIMEOUT),
            };
            if limit.is_zero() {
                add(&self.counters.dropped, records);
                complete = false;
                continue;
            }
            let job = match &batch {
                Batch::Cloud { job, .. } => Some(job.clone()),
                Batch::App { .. } => None,
            };
            match timeout(&*self.clock, limit, self.export(batch)).await {
                Ok(Outcome::Sent) => {}
                Ok(Outcome::Disabled) => {
                    add(&self.counters.recording_disabled, 1);
                    if let Some(job) = job {
                        self.state.lock().disable_job(&job);
                    }
                }
                Ok(Outcome::Failed(reason)) => {
                    add(&self.counters.failed, records);
                    log::warn!(target: "livekit.agents", "telemetry export failed: {reason}");
                }
                Err(_) => {
                    add(&self.counters.failed, records);
                    complete &= deadline.is_none_or(|at| self.clock.now() < at);
                    log::warn!(target: "livekit.agents", "telemetry export timed out");
                }
            }
        }
        complete
    }

    /// Waits until no export is in progress. False when `deadline` passed first.
    async fn settled(&self, deadline: Ts) -> bool {
        loop {
            let idle = self.idle.notified();
            tokio::pin!(idle);
            idle.as_mut().enable();
            if self.in_flight.load(Ordering::Acquire) == 0 {
                return true;
            }
            if before(&*self.clock, Some(deadline), idle).await.is_err() {
                return false;
            }
        }
    }

    async fn export(&self, batch: Batch) -> Outcome {
        let (process, cloud) = {
            let state = self.state.lock();
            (state.process.clone(), state.cloud.clone())
        };
        let request = match batch {
            Batch::Cloud { identity, recording, payload, .. } => {
                let Some(cloud) = cloud else { return Outcome::Sent };
                let base = cloud.base_url.trim_end_matches('/');
                let resource = encode::cloud_resource(&process, &identity);
                let mut stamp = encode::job_stamp(&process, &identity, true);
                stamp.extend(encode::session_flags(&recording));
                let policy = ContentPolicy::cloud(recording.redaction);
                let (signal, body) = match payload {
                    Payload::Spans(spans) => {
                        let spans: Vec<_> = spans
                            .into_iter()
                            .map(|(mut span, _)| {
                                policy.span(&mut span);
                                encode::stamp(&mut span.attributes, &stamp);
                                span
                            })
                            .collect();
                        let request =
                            encode::Message::Traces(encode::traces_request(resource, &spans));
                        ("traces", encode::body(&request, Protocol::Protobuf, true))
                    }
                    Payload::Logs(logs) => {
                        let logs: Vec<_> = logs
                            .into_iter()
                            .map(|mut log| {
                                policy.log(&mut log);
                                encode::stamp(&mut log.attributes, &stamp);
                                log
                            })
                            .collect();
                        let request = encode::Message::Logs(encode::logs_request(resource, &logs));
                        ("logs", encode::body(&request, Protocol::Protobuf, true))
                    }
                    Payload::Metrics(metrics) => {
                        let request =
                            encode::Message::Metrics(encode::metrics_request(resource, metrics));
                        ("metrics", encode::body(&request, Protocol::Protobuf, true))
                    }
                };
                let bearer =
                    match crate::token::observability_bearer(&cloud.api_key, &cloud.api_secret) {
                        Ok(token) => token,
                        Err(error) => return Outcome::Failed(error.to_string()),
                    };
                let url = format!("{base}/observability/{signal}/otlp/v0");
                (url, body, vec![("authorization".to_owned(), format!("Bearer {bearer}"))], true)
            }
            Batch::App { config, payload } => {
                let resource = encode::app_resource(&process, &config.resource);
                let (url, body) = match payload {
                    Payload::Spans(spans) => {
                        let policy = ContentPolicy::application(config.allow_pii);
                        let spans: Vec<_> = spans
                            .into_iter()
                            .map(|(mut span, identity)| {
                                policy.span(&mut span);
                                if let Some(identity) = identity {
                                    let mut stamp = encode::job_stamp(&process, &identity, true);
                                    policy.attributes(&mut stamp);
                                    encode::stamp(&mut span.attributes, &stamp);
                                }
                                encode::stamp(&mut span.attributes, &config.metadata);
                                span
                            })
                            .collect();
                        let request =
                            encode::Message::Traces(encode::traces_request(resource, &spans));
                        (
                            &config.traces_endpoint,
                            encode::body(&request, config.protocol, config.gzip),
                        )
                    }
                    Payload::Metrics(metrics) => {
                        let request =
                            encode::Message::Metrics(encode::metrics_request(resource, metrics));
                        (
                            &config.metrics_endpoint,
                            encode::body(&request, config.protocol, config.gzip),
                        )
                    }
                    Payload::Logs(_) => return Outcome::Sent,
                };
                let Some(url) = url.clone() else { return Outcome::Sent };
                (url, body, config.headers.clone(), false)
            }
        };
        let (url, body, headers, cloud) = request;
        match body {
            Ok(body) => self.post(&url, headers, body, cloud).await,
            Err(error) => Outcome::Failed(error),
        }
    }

    async fn post(
        &self,
        url: &str,
        headers: Vec<(String, String)>,
        body: encode::Body,
        cloud: bool,
    ) -> Outcome {
        let client = self
            .client
            .get_or_try_init(|| async {
                crate::tls::config().await?;
                crate::tls::http_client_builder()?
                    .retry(reqwest::retry::never())
                    .connect_timeout(REQUEST_TIMEOUT)
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|error| error.to_string())
            })
            .await;
        let client = match client {
            Ok(client) => client,
            Err(error) => return Outcome::Failed(error),
        };
        let mut request = client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, body.content_type)
            .body(body.bytes);
        if body.gzip {
            request = request.header(reqwest::header::CONTENT_ENCODING, "gzip");
        }
        for (key, value) in headers {
            request = request.header(key, value);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => return Outcome::Failed(error.to_string()),
        };
        let status = response.status();
        if status.is_success() {
            return Outcome::Sent;
        }
        let mut body = Vec::new();
        let mut chunks = response.bytes_stream();
        while let Some(Ok(chunk)) = chunks.next().await {
            if body.len() + chunk.len() > RESPONSE_BYTES {
                break;
            }
            body.extend_from_slice(&chunk);
        }
        if cloud && UploadGate::is_disabled_response(status.as_u16(), body) {
            return Outcome::Disabled;
        }
        Outcome::Failed(format!("{url} answered HTTP {}", status.as_u16()))
    }
}

#[cfg(test)]
#[path = "exporter_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "proto_tests.rs"]
mod proto_tests;
