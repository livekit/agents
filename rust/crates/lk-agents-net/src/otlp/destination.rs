//! The destinations' state behind the exporter's lock: the job registry, Cloud's per-job queues
//! and held spans, the application queue, and the batches taken from them for export.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use super::proto::metrics::v1::Metric;
use lk_agents_abi::telemetry::policy::TelemetryExportAction as Action;
use lk_agents_abi::telemetry::record::{JobKey, LogRecord, Record, SpanRecord};
use lk_agents_core::telemetry::policy::cloud_export_action;

use super::config::{AppConfig, CloudEndpoint, CloudRecording, JobIdentity, ProcessIdentity};
use super::metrics::Aggregator;

/// Records one request carries.
pub(crate) const BATCH_RECORDS: usize = 512;

/// Per-destination queue bounds. Sized for a session's end-of-session report burst.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub records: usize,
    pub bytes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { records: 8192, bytes: 16 * 1024 * 1024 }
    }
}

/// A queue's retained records and bytes against its destination's bound.
#[derive(Default)]
struct Usage {
    records: usize,
    bytes: usize,
}

impl Usage {
    fn admit(&mut self, limits: &Limits, bytes: usize) -> bool {
        if self.records >= limits.records || self.bytes.saturating_add(bytes) > limits.bytes {
            return false;
        }
        self.records += 1;
        self.bytes += bytes;
        true
    }

    fn release(&mut self, bytes: usize) {
        self.records = self.records.saturating_sub(1);
        self.bytes = self.bytes.saturating_sub(bytes);
    }
}

enum Decision {
    Undecided,
    Recording(CloudRecording),
    /// The project disabled recording; nothing more is sent for this job.
    Disabled,
}

struct Job {
    identity: Arc<JobIdentity>,
    decision: Decision,
    held: Vec<(SpanRecord, usize)>,
    spans: VecDeque<(SpanRecord, usize)>,
    logs: VecDeque<(LogRecord, usize)>,
    metrics: Aggregator,
}

pub(crate) struct AppDestination {
    pub config: Arc<AppConfig>,
    spans: VecDeque<(SpanRecord, Option<Arc<JobIdentity>>, usize)>,
    usage: Usage,
    metrics: Aggregator,
}

/// One request's worth of records, taken from the queues; encoded outside the lock.
pub(crate) enum Batch {
    Cloud { job: JobKey, identity: Arc<JobIdentity>, recording: CloudRecording, payload: Payload },
    App { config: Arc<AppConfig>, payload: Payload },
}

pub(crate) enum Payload {
    Spans(Vec<(SpanRecord, Option<Arc<JobIdentity>>)>),
    Logs(Vec<LogRecord>),
    Metrics(Vec<Metric>),
}

impl Payload {
    /// What a failed or abandoned batch counts.
    pub fn records(&self) -> usize {
        match self {
            Payload::Spans(s) => s.len(),
            Payload::Logs(l) => l.len(),
            Payload::Metrics(m) => m.len(),
        }
    }
}

impl Batch {
    pub fn payload(&self) -> &Payload {
        match self {
            Batch::Cloud { payload, .. } | Batch::App { payload, .. } => payload,
        }
    }
}

/// Which records a take selects.
#[derive(Clone, Copy)]
pub(crate) struct Take<'a> {
    /// Only this job's Cloud records, and no application records.
    pub job: Option<&'a JobKey>,
    pub metrics: bool,
    /// Everything queued, not only full batches.
    pub all: bool,
}

/// What `record` decided, for the caller's diagnostics and wake-up.
#[derive(Default)]
pub(crate) struct Admitted {
    pub dropped: u64,
    /// A queue reached a full batch.
    pub batch_ready: bool,
}

pub(crate) struct State {
    pub process: ProcessIdentity,
    pub limits: Limits,
    pub cloud: Option<CloudEndpoint>,
    jobs: BTreeMap<JobKey, Job>,
    cloud_usage: Usage,
    pub app: Option<AppDestination>,
    /// Replaced application destinations, exported once more and then dropped.
    retired: Vec<AppDestination>,
    pub closed: bool,
}

impl State {
    pub fn new(process: ProcessIdentity, limits: Limits) -> Self {
        State {
            process,
            limits,
            cloud: None,
            jobs: BTreeMap::new(),
            cloud_usage: Usage::default(),
            app: None,
            retired: Vec::new(),
            closed: false,
        }
    }

    /// False when the job was already open.
    pub fn open_job(&mut self, key: JobKey, identity: JobIdentity, now: u64) -> bool {
        if self.jobs.contains_key(&key) {
            return false;
        }
        let metrics = Aggregator::new(super::config::Temporality::Delta, now);
        self.jobs.insert(
            key,
            Job {
                identity: Arc::new(identity),
                decision: Decision::Undecided,
                held: Vec::new(),
                spans: VecDeque::new(),
                logs: VecDeque::new(),
                metrics,
            },
        );
        true
    }

    /// Releases the held spans when the job records traces, else drops them. False when the job
    /// is not open or already decided.
    pub fn register_job(&mut self, key: &JobKey, recording: CloudRecording, now: u64) -> bool {
        let Some(job) = self.jobs.get_mut(key) else { return false };
        if !matches!(job.decision, Decision::Undecided) {
            return false;
        }
        let held = std::mem::take(&mut job.held);
        if self.cloud.is_some() && recording.traces {
            job.spans.extend(held);
        } else {
            held.iter().for_each(|(_, bytes)| self.cloud_usage.release(*bytes));
        }
        if self.cloud.is_none() || !recording.metrics {
            job.metrics = Aggregator::new(super::config::Temporality::Delta, now);
        }
        job.decision = Decision::Recording(recording);
        true
    }

    /// Stops a job's Cloud export after the project refused it.
    pub fn disable_job(&mut self, key: &JobKey) {
        if let Some(job) = self.jobs.get_mut(key) {
            job.decision = Decision::Disabled;
            Self::clear(job, &mut self.cloud_usage);
        }
    }

    /// Removes a job; returns the records it still had queued.
    pub fn remove_job(&mut self, key: &JobKey) -> usize {
        let Some(mut job) = self.jobs.remove(key) else { return 0 };
        let queued = job.spans.len() + job.logs.len();
        Self::clear(&mut job, &mut self.cloud_usage);
        queued
    }

    fn clear(job: &mut Job, usage: &mut Usage) {
        job.held.drain(..).for_each(|(_, bytes)| usage.release(bytes));
        job.spans.drain(..).for_each(|(_, bytes)| usage.release(bytes));
        job.logs.drain(..).for_each(|(_, bytes)| usage.release(bytes));
        job.metrics = Aggregator::new(super::config::Temporality::Delta, 0);
    }

    /// Replaces the application destination for subsequent records; the old one is exported once
    /// more by the next take.
    pub fn configure_app(&mut self, config: Option<AppConfig>, now: u64) {
        let next = config.map(|config| AppDestination {
            metrics: Aggregator::new(config.temporality, now),
            config: Arc::new(config),
            spans: VecDeque::new(),
            usage: Usage::default(),
        });
        if let Some(old) = std::mem::replace(&mut self.app, next) {
            self.retired.push(old);
        }
    }

    pub fn record(&mut self, record: Record, bytes: usize) -> Admitted {
        let mut admitted = Admitted::default();
        if self.closed {
            return admitted;
        }
        let job = match &record {
            Record::Span(s) => s.job.as_ref(),
            Record::Log(l) => l.job.as_ref(),
            Record::Measurement(m) => m.job.as_ref(),
        };
        let cloud = self.cloud_action(job, &record);
        let identity = job.and_then(|k| self.jobs.get(k)).map(|j| j.identity.clone());
        if let Some(app) = &mut self.app {
            let limits = self.limits;
            match &record {
                Record::Span(span) if app.config.traces_endpoint.is_some() => {
                    if app.usage.admit(&limits, bytes) {
                        app.spans.push_back((span.clone(), identity, bytes));
                        admitted.batch_ready |= app.spans.len() >= BATCH_RECORDS;
                    } else {
                        admitted.dropped += 1;
                    }
                }
                Record::Measurement(m) if app.config.metrics_endpoint.is_some() => {
                    admitted.dropped += u64::from(!app.metrics.record(m));
                }
                _ => {}
            }
        }
        let (Some(action), Some(key)) = (cloud, job) else { return admitted };
        let job = self.jobs.get_mut(key).expect("an admitted record's job is open");
        match (action, record) {
            (Action::Export, Record::Span(span)) => {
                if self.cloud_usage.admit(&self.limits, bytes) {
                    job.spans.push_back((span, bytes));
                    admitted.batch_ready |= job.spans.len() >= BATCH_RECORDS;
                } else {
                    admitted.dropped += 1;
                }
            }
            (Action::Export, Record::Log(log)) => {
                if self.cloud_usage.admit(&self.limits, bytes) {
                    job.logs.push_back((log, bytes));
                    admitted.batch_ready |= job.logs.len() >= BATCH_RECORDS;
                } else {
                    admitted.dropped += 1;
                }
            }
            (Action::Export | Action::Hold, Record::Measurement(m)) => {
                admitted.dropped += u64::from(!job.metrics.record(&m));
            }
            (Action::Hold, Record::Span(span)) => {
                if self.cloud_usage.admit(&self.limits, bytes) {
                    job.held.push((span, bytes));
                } else {
                    admitted.dropped += 1;
                }
            }
            (Action::Capacity, _) => admitted.dropped += 1,
            _ => {}
        }
        admitted
    }

    /// The Cloud admission of a record: `None` when Cloud never sees it.
    fn cloud_action(&self, key: Option<&JobKey>, record: &Record) -> Option<Action> {
        self.cloud.as_ref()?;
        let job = self.jobs.get(key?)?;
        let (registered, enabled) = match (&job.decision, record) {
            (Decision::Undecided, _) => (false, false),
            (Decision::Recording(r), Record::Span(_)) => (true, r.traces),
            (Decision::Recording(r), Record::Log(_)) => (true, r.logs),
            (Decision::Recording(r), Record::Measurement(_)) => (true, r.metrics),
            (Decision::Disabled, _) => return Some(Action::Disabled),
        };
        // Logs are never held: report records are produced after the decision.
        let open = !registered && !matches!(record, Record::Log(_));
        let held = if matches!(record, Record::Span(_)) { job.held.len() } else { 0 };
        Some(cloud_export_action(registered, enabled, open, held, false))
    }

    /// Batches to export now: full batches, or everything with `take.all`; metrics with
    /// `take.metrics`.
    pub fn take(&mut self, take: Take<'_>, now: u64) -> Vec<Batch> {
        let mut batches = Vec::new();
        let due = |n: usize| take.all && n > 0 || n >= BATCH_RECORDS;
        if self.cloud.is_some() {
            for (key, job) in &mut self.jobs {
                if take.job.is_some_and(|k| k != key) {
                    continue;
                }
                let Decision::Recording(recording) = &job.decision else { continue };
                let batch = |payload| Batch::Cloud {
                    job: key.clone(),
                    identity: job.identity.clone(),
                    recording: recording.clone(),
                    payload,
                };
                while due(job.spans.len()) {
                    let spans = drain(&mut job.spans, &mut self.cloud_usage, |(s, b)| (s, b));
                    batches.push(batch(Payload::Spans(
                        spans.into_iter().map(|s| (s, None)).collect(),
                    )));
                }
                while due(job.logs.len()) {
                    let logs = drain(&mut job.logs, &mut self.cloud_usage, |(l, b)| (l, b));
                    batches.push(batch(Payload::Logs(logs)));
                }
                if take.metrics && !job.metrics.is_empty() {
                    let mut stamp = super::encode::job_stamp(&self.process, &job.identity, true);
                    stamp.extend(super::encode::session_flags(recording));
                    batches.push(batch(Payload::Metrics(job.metrics.collect(now, &stamp))));
                }
            }
        }
        if take.job.is_some() {
            return batches;
        }
        let retired = std::mem::take(&mut self.retired);
        if let Some(app) = &mut self.app {
            app.take(take.all, take.metrics, now, &mut batches);
        }
        for mut app in retired {
            app.take(true, true, now, &mut batches);
        }
        batches
    }
}

impl AppDestination {
    fn take(&mut self, all: bool, metrics: bool, now: u64, batches: &mut Vec<Batch>) {
        while all && !self.spans.is_empty() || self.spans.len() >= BATCH_RECORDS {
            let spans = drain(&mut self.spans, &mut self.usage, |(s, i, b)| ((s, i), b));
            batches
                .push(Batch::App { config: self.config.clone(), payload: Payload::Spans(spans) });
        }
        if metrics && !self.metrics.is_empty() {
            let metrics = self.metrics.collect(now, &Vec::new());
            batches.push(Batch::App {
                config: self.config.clone(),
                payload: Payload::Metrics(metrics),
            });
        }
    }
}

/// Up to one batch from the front of `queue`, releasing its retained bytes.
fn drain<T, R>(
    queue: &mut VecDeque<T>,
    usage: &mut Usage,
    split: impl Fn(T) -> (R, usize),
) -> Vec<R> {
    let n = queue.len().min(BATCH_RECORDS);
    queue
        .drain(..n)
        .map(|item| {
            let (record, bytes) = split(item);
            usage.release(bytes);
            record
        })
        .collect()
}
