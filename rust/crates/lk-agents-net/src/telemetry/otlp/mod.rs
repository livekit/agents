//! The OTLP exporter: every span, report record and measurement reaches LiveKit Cloud and the
//! application's endpoint from here (docs/internals/telemetry.md "Destinations").
//!
//! Producers hand records to [`Exporter`] through `RecordSink`, which never blocks: each
//! destination copies the record into a bounded queue, or drops and counts it. An owned task
//! exports full batches as they fill, everything queued each second and metrics every 30 s.
//!
//! - [`config`]: identities, the Cloud endpoint and recording decision, the application endpoint.
//! - `cloud`: the process's telemetry HTTP client and the Cloud endpoint's requests.
//! - `destination`: the job registry, queues, holding and batch selection.
//! - `encode`: content policy, identity stamps and the OTLP request bodies.
//! - `metrics`: per-destination aggregation, delta or cumulative.
//! - [`proto`]: the OTLP messages, generated from opentelemetry-proto (`proto_tests.rs`).

pub(crate) mod cloud;
pub mod config;
mod destination;
mod encode;
mod metrics;
pub mod proto;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use lk_agents_abi::telemetry::record::{JobKey, Record, RecordSink, SpanRecord};
use lk_agents_abi::{Clock, OwnedTask, Spawner, SpawnerExt as _, Ts, before, timeout};
use parking_lot::Mutex;
use tokio::sync::Notify;

use cloud::Answer;
pub use config::{
    AppConfig, CloudEndpoint, CloudRecording, JobIdentity, ProcessIdentity, SdkName, Simulation,
    Temporality,
};
pub use destination::Limits;
use destination::{App, AppSpan, Batch, Cloud, State, Take};
use lk_agents_core::telemetry::policy::ContentPolicy;

const SPAN_INTERVAL: Duration = Duration::from_secs(1);
const METRIC_INTERVAL: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

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
        let complete = self.flush(Some(key), limit).await;
        add(&self.shared.counters.dropped, self.shared.state.lock().remove_job(key));
        if !complete {
            add(&self.shared.counters.incomplete_cleanup, 1);
        }
    }

    /// Whether the job's project refused recording.
    pub fn job_disabled(&self, key: &JobKey) -> bool {
        self.shared.state.lock().job_disabled(key)
    }

    /// Stops the job's Cloud export: another of its uploads found recording disabled.
    pub fn disable_job(&self, key: &JobKey) {
        add(&self.shared.counters.recording_disabled, 1);
        self.shared.state.lock().disable_job(key);
    }

    /// Sends everything queued, metrics included: `job`'s Cloud records only, or everything.
    /// False when `limit` passed first.
    pub async fn flush(&self, job: Option<&JobKey>, limit: Duration) -> bool {
        let shared = &*self.shared;
        let deadline = shared.clock.now().saturating_add(limit);
        let now = shared.wall_nanos();
        let batches = shared.state.lock().take(Take { job, metrics: true, all: true }, now);
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
        let complete = self.flush(None, limit).await;
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
            let records = batch.records();
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
        match batch {
            Batch::Cloud { identity, recording, payload, .. } => {
                let Some(cloud) = cloud else { return Outcome::Sent };
                let (signal, body) = cloud_request(&process, &identity, &recording, payload);
                let body = match body {
                    Ok(body) => body,
                    Err(error) => return Outcome::Failed(error),
                };
                let path = format!("{signal}/otlp/v0");
                match cloud.post(&path, |request| body.attach(request)).await {
                    Ok(answer) => Outcome::of(answer, &path),
                    Err(error) => Outcome::Failed(error.to_string()),
                }
            }
            Batch::App { config, payload } => export_app(&process, &config, payload).await,
        }
    }
}

/// One job's Cloud request, under Cloud's content policy and the job's identity stamp, and
/// the signal it is posted as.
fn cloud_request(
    process: &ProcessIdentity,
    identity: &JobIdentity,
    recording: &CloudRecording,
    payload: Cloud,
) -> (&'static str, Result<encode::Body, String>) {
    let resource = encode::cloud_resource(process, identity);
    let stamp = encode::cloud_stamp(process, identity, recording);
    let policy = ContentPolicy::cloud(recording.redaction);
    match payload {
        Cloud::Spans(mut spans) => {
            for span in &mut spans {
                policy.span(span);
                encode::stamp(&mut span.attributes, &stamp);
            }
            ("traces", encode::body(&encode::traces_request(resource, &spans), true))
        }
        Cloud::Logs(mut logs) => {
            for log in &mut logs {
                policy.log(log);
                encode::stamp(&mut log.attributes, &stamp);
            }
            ("logs", encode::body(&encode::logs_request(resource, &logs), true))
        }
        Cloud::Metrics(metrics) => {
            ("metrics", encode::body(&encode::metrics_request(resource, metrics), true))
        }
    }
}

async fn export_app(process: &ProcessIdentity, config: &AppConfig, payload: App) -> Outcome {
    let resource = encode::app_resource(process, &config.resource);
    let (url, body) = match payload {
        App::Spans(spans) => {
            let spans = stamp_app_spans(process, config, spans);
            let body = encode::body(&encode::traces_request(resource, &spans), config.gzip);
            (&config.traces_endpoint, body)
        }
        App::Metrics(metrics) => {
            let request = encode::metrics_request(resource, metrics);
            (&config.metrics_endpoint, encode::body(&request, config.gzip))
        }
    };
    let Some(url) = url else { return Outcome::Sent };
    let body = match body {
        Ok(body) => body,
        Err(error) => return Outcome::Failed(error),
    };
    let client = match cloud::client().await {
        Ok(client) => client,
        Err(error) => return Outcome::Failed(error),
    };
    let mut request = body.attach(client.post(url));
    for (key, value) in &config.headers {
        request = request.header(key, value);
    }
    match request.send().await {
        Ok(response) => Outcome::of(cloud::answer(response, false).await, url),
        Err(error) => Outcome::Failed(error.to_string()),
    }
}

/// The application's content policy, then its job stamp and metadata, on each span.
fn stamp_app_spans(
    process: &ProcessIdentity,
    config: &AppConfig,
    spans: Vec<AppSpan>,
) -> Vec<SpanRecord> {
    let policy = ContentPolicy::application(config.allow_pii);
    let stamped = spans.into_iter().map(|mut queued| {
        policy.span(&mut queued.span);
        if let Some(job) = &queued.job {
            let mut stamp = encode::app_stamp(process, job);
            policy.attributes(&mut stamp);
            encode::stamp(&mut queued.span.attributes, &stamp);
        }
        encode::stamp(&mut queued.span.attributes, &config.metadata);
        queued.span
    });
    stamped.collect()
}

impl Outcome {
    fn of(answer: Answer, url: &str) -> Outcome {
        match answer {
            Answer::Sent => Outcome::Sent,
            Answer::Disabled => Outcome::Disabled,
            Answer::Refused { status, .. } => {
                Outcome::Failed(format!("{url} answered HTTP {status}"))
            }
        }
    }
}

#[cfg(test)]
#[path = "exporter_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "proto_tests.rs"]
mod proto_tests;
