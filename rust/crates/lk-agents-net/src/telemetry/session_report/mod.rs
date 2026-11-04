//! The LiveKit Cloud session report at job end (docs/internals/telemetry.md "Session report"):
//! the report's records go to the exporter with the job's final flush, then the recording upload.
//! Not user telemetry: Cloud only.
//!
//! - [`events`]: a session's retained event snapshots.
//! - [`recording`]: the multipart recording upload.
//! - [`tagger`]: the session's tags, stamped by the process clock.

pub mod events;
pub mod recording;
pub mod tagger;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use lk_agents_abi::Clock;
use lk_agents_abi::telemetry::record::{JobKey, Record, RecordSink as _};
use lk_agents_abi::telemetry::report::{RecordingOptions, SessionReport};
use lk_agents_core::telemetry::report;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::telemetry::otlp::{CloudEndpoint, Exporter};
pub use events::EventRetention;
pub use recording::Uploaded;

/// A session report upload that did not complete.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct UploadError(pub String);

/// Where a job's session report goes.
pub struct ReportOutputs<'a> {
    pub exporter: Option<&'a Exporter>,
    pub cloud: Option<&'a CloudEndpoint>,
    pub clock: &'a dyn Clock,
}

/// The job's session report outputs, in order and within `limit`: its records to the exporter,
/// the job's queued telemetry flushed, then the recording upload, unless the project refused
/// recording.
pub async fn export(
    outputs: ReportOutputs<'_>,
    key: &JobKey,
    report: &SessionReport,
    options: &RecordingOptions,
    simulated: bool,
    limit: Duration,
) -> Result<(), UploadError> {
    let ReportOutputs { exporter, cloud, clock } = outputs;
    let deadline = clock.now().saturating_add(limit);
    if let Some(exporter) = exporter {
        for record in report::records(report, options) {
            exporter.record(Record::Log(record));
        }
        if !exporter.flush_job(key, deadline.saturating_duration_since(clock.now())).await {
            log::warn!(target: "livekit.agents", "the session's telemetry was not all sent in time");
        }
        if exporter.job_disabled(key) {
            return Ok(());
        }
    }
    let (Some(cloud), Some(upload)) = (cloud, report::upload(report, options, simulated)) else {
        return Ok(());
    };
    let limit = deadline.saturating_duration_since(clock.now());
    if recording::upload(cloud, &upload, clock, limit).await? == Uploaded::Disabled {
        log::warn!(target: "livekit.agents", "session recording is disabled by the project owner");
        if let Some(exporter) = exporter {
            exporter.disable_job(key);
        }
    }
    Ok(())
}

/// Recordings in flight; further uploads wait, within their own deadline.
const UPLOADS: usize = 2;
/// The retained session events of every session in the process.
const PROCESS_EVENT_ITEMS: usize = 65_536;
const PROCESS_EVENT_BYTES: usize = 64 * 1024 * 1024;

/// The session report's process admission: recordings in flight and the retained session events
/// of every session. One per process; clones share it.
#[derive(Clone)]
pub struct ReportAdmission {
    uploads: Arc<Semaphore>,
    events: Arc<EventQuota>,
}

impl Default for ReportAdmission {
    fn default() -> Self {
        Self { uploads: Arc::new(Semaphore::new(UPLOADS)), events: Arc::new(EventQuota::default()) }
    }
}

impl ReportAdmission {
    /// The process's admission.
    pub fn process() -> Self {
        static ADMISSION: lk_agents_media::pool::PerProcess<ReportAdmission> =
            lk_agents_media::pool::PerProcess::new(ReportAdmission::default);
        ADMISSION.get()
    }

    /// Waits for a recording upload slot.
    async fn upload(&self) -> OwnedSemaphorePermit {
        self.uploads.clone().acquire_owned().await.expect("the upload semaphore is never closed")
    }
}

/// Retained event items and bytes. Quotas publish no payload; in-flight reservations may
/// conservatively refuse optional data.
#[derive(Default)]
struct EventQuota {
    items: AtomicUsize,
    bytes: AtomicUsize,
}

impl EventQuota {
    fn admit(&self, bytes: usize) -> bool {
        if self
            .items
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |items| {
                (items < PROCESS_EVENT_ITEMS).then_some(items + 1)
            })
            .is_err()
        {
            return false;
        }
        if self
            .bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |retained| {
                retained.checked_add(bytes).filter(|total| *total <= PROCESS_EVENT_BYTES)
            })
            .is_err()
        {
            self.items.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    fn release(&self, items: usize, bytes: usize) {
        self.bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.items.fetch_sub(items, Ordering::Relaxed);
    }
}

#[cfg(test)]
#[path = "../session_report_tests.rs"]
mod tests;
