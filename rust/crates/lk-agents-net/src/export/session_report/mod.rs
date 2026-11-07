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
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::time::Duration;

use lk_agents_abi::Clock;
use lk_agents_abi::telemetry::record::{JobKey, Record, RecordSink as _};
use lk_agents_abi::telemetry::report::{RecordingOptions, SessionReport};
use lk_agents_core::telemetry::report;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::export::otlp::cloud::PostError;
use crate::export::otlp::{CloudEndpoint, Exporter};
pub use events::EventRetention;

/// Whether the recording upload carries the job's profile: on once Cloud ingest accepts the
/// `profile` part (an unknown part fails the whole upload).
const UPLOAD_PROFILES: bool = false;

/// The job's profile, encoded on the blocking pool (an hour of samples takes tens of ms).
async fn encode(profile: crate::export::profile::Profile) -> Option<Vec<u8>> {
    let encoded = crate::blocking_io::blocking_io().run((), move || profile.encode()).await;
    encoded.ok()
}

/// Writes the job's profile beside its audio recording, as `profile.pb.gz`.
async fn write_profile(audio: &str, profile: Vec<u8>) {
    let Some(dir) = std::path::Path::new(audio).parent().filter(|dir| !dir.as_os_str().is_empty())
    else {
        return;
    };
    let file = dir.join("profile.pb.gz");
    let io = crate::blocking_io::blocking_io();
    if let Err(error) = io.try_run((), move || std::fs::write(file, profile)).await {
        log::debug!(target: lk_agents_abi::LOG_TARGET, "the job's profile was not written: {error}");
    }
}

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
        if !exporter.flush(Some(key), deadline.saturating_duration_since(clock.now())).await {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "the session's telemetry was not all sent in time");
        }
        if exporter.job_disabled(key) {
            return Ok(());
        }
    }
    let agent = exporter.map(Exporter::cloud_agent).unwrap_or_default();
    let upload = cloud.and_then(|_| report::upload(report, options, simulated, &agent));
    let send_profile = UPLOAD_PROFILES && upload.is_some();
    let audio = report.audio_recording_path.clone();
    let profile = match crate::export::profile::take(key) {
        Some(profile) if audio.is_some() || send_profile => encode(profile).await,
        _ => None,
    };
    if let (Some(profile), Some(audio)) = (&profile, &audio) {
        write_profile(audio, profile.clone()).await;
    }
    let (Some(cloud), Some(upload)) = (cloud, upload) else {
        return Ok(());
    };
    let profile = profile.filter(|profile| {
        let fits = profile.len() <= crate::export::profile::UPLOAD_BYTES;
        if send_profile && !fits {
            log::debug!(target: lk_agents_abi::LOG_TARGET, "the job's profile is too large to upload");
        }
        send_profile && fits
    });
    let limit = deadline.saturating_duration_since(clock.now());
    let uploaded = recording::upload(cloud, &upload, profile.as_deref(), clock);
    let uploaded = lk_agents_abi::timeout(clock, limit, uploaded)
        .await
        .map_err(|_| UploadError("session recording upload deadline elapsed".into()))?;
    match uploaded {
        Ok(()) => Ok(()),
        Err(PostError::Disabled) => {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "session recording is disabled by the project owner");
            if let Some(exporter) = exporter {
                exporter.disable_job(key);
            }
            Ok(())
        }
        Err(PostError::Setup(error)) => Err(UploadError(error)),
        Err(error) => Err(UploadError(format!("recording upload failed: {error}"))),
    }
}

/// The session report's files.
pub struct ReportFiles;

impl lk_agents_abi::telemetry::report::SessionReportFunctions for ReportFiles {
    fn session_report_json(report: SessionReport) -> Result<String, lk_agents_abi::ToDictError> {
        report::console_json(&report)
    }
}

/// Recordings in flight; further uploads wait, within their own deadline.
const UPLOADS: usize = 2;

/// The session report's process admission: recordings in flight and the retained session events
/// of every session. One per process; clones share it.
#[derive(Clone)]
pub struct ReportAdmission {
    uploads: Arc<Semaphore>,
    events: Arc<RetainedEvents>,
}

impl Default for ReportAdmission {
    fn default() -> Self {
        Self { uploads: Arc::new(Semaphore::new(UPLOADS)), events: Arc::default() }
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
    /// `None` once the semaphore closed (never, while the process lives).
    async fn upload(&self) -> Option<OwnedSemaphorePermit> {
        self.uploads.clone().acquire_owned().await.ok()
    }
}

/// The retained session events of every session in the process, and whether their count was
/// already reported.
#[derive(Default)]
struct RetainedEvents {
    items: AtomicUsize,
    warned: AtomicBool,
}

#[cfg(test)]
#[path = "../session_report_tests.rs"]
mod tests;
