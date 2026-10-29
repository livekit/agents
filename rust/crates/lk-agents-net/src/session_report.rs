//! The LiveKit Cloud session report: its transcript, prepared incrementally, and the recording
//! upload at session end. Not user telemetry (decision 87): it has its own small process
//! admission, and cancelling an upload drops its transport and file.
//!
//! - [`admission`]: the process's preparation, upload and event retention capacity.
//! - [`events`]: a session's retained event snapshots.
//! - [`transcript`]: the bounded transcript and console report.

mod admission;
pub mod events;
pub mod projection;
pub mod records;
pub mod transcript;

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures_util::{StreamExt, stream};
use livekit_protocol::MetricsRecordingHeader;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::telemetry::report::UploadGate as UploadGateApi;
use lk_agents_abi::telemetry::report::projection::SessionReportProjection as _;
use lk_agents_abi::telemetry::upload::{SessionReportUpload, UploadError};
use prost::Message;
use reqwest::multipart::{Form, Part};
use tokio::io::AsyncReadExt;

use crate::attempts::{AttemptStep, Attempts, RequestPolicy};
use crate::env::ProcessEnv;
pub use admission::ReportAdmission;
pub use events::EventRetention;
pub use transcript::TranscriptBuilder;

const CHUNK_BYTES: usize = 64 * 1024;
const TRANSCRIPT_BYTES: usize = 2 * 1024 * 1024;
const RESPONSE_BYTES: usize = 64 * 1024;
const UPLOAD_TIMEOUT: Duration =
    Duration::from_secs(lk_agents_abi::telemetry::policy::UPLOAD_TIMEOUT_SECONDS);
/// Four attempts; the outer [`UPLOAD_TIMEOUT`] bounds them all.
const UPLOAD_ATTEMPTS: RequestPolicy = RequestPolicy {
    connect_timeout: None,
    write_timeout: None,
    read_timeout: None,
    total_timeout: None,
    max_retry: 3,
    retry_interval: Duration::from_secs(2),
};

/// The immutable input of one recording upload; no host objects or file contents are retained.
#[derive(Debug, Clone)]
struct RecordingUpload {
    base_url: String,
    room_id: String,
    job_id: String,
    simulated: bool,
    redaction_enabled: bool,
    start_time_millis: i64,
    transcript: Option<String>,
    audio_path: Option<String>,
    disabled: bool,
}

/// Credentials for the session recording endpoint and host OTLP exporters, under the process's
/// upload admission (decision 88).
pub struct SessionReportUploader {
    api_key: String,
    api_secret: String,
    admission: ReportAdmission,
    disabled: AtomicBool,
}

impl SessionReportUpload for SessionReportUploader {
    type Env = ProcessEnv;

    fn new(env: &ProcessEnv, api_key: String, api_secret: String) -> Result<Self, UploadError> {
        Self::with_admission(api_key, api_secret, (env.report_admission)().map_err(UploadError)?)
    }

    fn bearer(&self) -> Result<String, UploadError> {
        crate::token::observability_bearer(&self.api_key, &self.api_secret).map_err(upload_error)
    }

    fn content_policy(
        &self,
        transcript_requested: bool,
        audio_requested: bool,
        capture_content: bool,
        redaction_enabled: bool,
    ) -> (bool, bool) {
        let projection = projection::ReportProjection::new(
            capture_content,
            &lk_agents_abi::telemetry::report::RecordingOptions {
                transcript: transcript_requested,
                audio: audio_requested,
                ..Default::default()
            },
            redaction_enabled,
        );
        (projection.transcript(), projection.audio())
    }

    async fn upload(
        &self,
        base_url: String,
        room_id: String,
        job_id: String,
        simulated: bool,
        redaction_enabled: bool,
        start_time_millis: i64,
        transcript: Option<String>,
        audio_path: Option<String>,
        disabled: bool,
    ) -> Result<bool, UploadError> {
        let recording = RecordingUpload {
            base_url,
            room_id,
            job_id,
            simulated,
            redaction_enabled,
            start_time_millis,
            transcript,
            audio_path,
            disabled,
        };
        self.send(recording).await
    }
}

fn upload_error(err: impl std::fmt::Display) -> UploadError {
    UploadError(err.to_string())
}

impl SessionReportUploader {
    fn with_admission(
        api_key: String,
        api_secret: String,
        admission: ReportAdmission,
    ) -> Result<Self, UploadError> {
        // Validate credentials before retaining a report.
        crate::token::observability_bearer(&api_key, &api_secret).map_err(upload_error)?;
        Ok(Self { api_key, api_secret, admission, disabled: AtomicBool::new(false) })
    }

    async fn send(&self, mut recording: RecordingUpload) -> Result<bool, UploadError> {
        if recording.disabled || self.disabled.load(Ordering::Acquire) {
            return Ok(false);
        }
        // Raw audio cannot be redacted safely; project redaction denies both content parts.
        if recording.redaction_enabled {
            recording.transcript = None;
            recording.audio_path = None;
        }
        if recording.transcript.is_none() && recording.audio_path.is_none() {
            return Ok(true);
        }
        let transcript_bytes = recording.transcript.as_ref().map_or(0, String::len);
        if transcript_bytes > TRANSCRIPT_BYTES {
            return Err(UploadError("session transcript exceeds the 2 MiB upload limit".into()));
        }
        let _permit = self
            .admission
            .upload()
            .ok_or_else(|| UploadError("session recording upload capacity exhausted".into()))?;
        let url =
            format!("{}/observability/recordings/v0", recording.base_url.trim_end_matches('/'));
        let url = reqwest::Url::parse(&url).map_err(upload_error)?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(UploadError("observability endpoint must use HTTP or HTTPS".into()));
        }
        crate::transport::install_tls_provider();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(upload_error)?;
        tokio::time::timeout(UPLOAD_TIMEOUT, self.attempts(&client, recording, url))
            .await
            .map_err(|_| UploadError("session recording upload deadline elapsed".into()))?
    }

    async fn attempts(
        &self,
        client: &reqwest::Client,
        recording: RecordingUpload,
        url: reqwest::Url,
    ) -> Result<bool, UploadError> {
        let epoch = tokio::time::Instant::now();
        let now = || Ts::from_nanos(epoch.elapsed().as_nanos() as u64);
        let mut steps = Vec::new();
        let mut attempts = Attempts::start(UPLOAD_ATTEMPTS, now(), &mut steps);
        let mut failure = String::new();
        loop {
            let mut launched = None;
            for step in steps.drain(..) {
                match step {
                    AttemptStep::Launch(attempt) => launched = Some(attempt),
                    AttemptStep::GiveUp(_) => return Err(UploadError(failure)),
                    _ => {}
                }
            }
            let Some(attempt) = launched else {
                let at = attempts.wake_at(None).expect("an undecided upload is waiting to retry");
                tokio::time::sleep_until(epoch + at.saturating_duration_since(Ts::ZERO)).await;
                attempts.wake(now(), None, &mut steps);
                continue;
            };
            let result = client
                .post(url.clone())
                .bearer_auth(self.bearer()?)
                .multipart(multipart(&recording).await?)
                .send()
                .await;
            let (error, retry_after) = match result {
                Ok(response) if response.status().is_success() => return Ok(true),
                Ok(response) => {
                    let status = response.status().as_u16();
                    let body = response_body(response).await?;
                    if <records::UploadGate as UploadGateApi>::is_disabled_response(
                        status,
                        body.clone(),
                    ) {
                        self.disabled.store(true, Ordering::Release);
                        return Ok(false);
                    }
                    failure = format!("recording upload failed with HTTP {status}");
                    let retry_after = retry_delay(&body);
                    let error = ApiError::from_http(&failure, status.into(), None);
                    let retryable = error.retryable || retry_after.is_some();
                    (error.with_retryable(retryable), retry_after)
                }
                Err(err) => {
                    failure = format!("recording upload transport failed: {err}");
                    let retryable = err.is_connect() || err.is_timeout() || err.is_body();
                    (ApiError::connection(Some(failure.clone())).with_retryable(retryable), None)
                }
            };
            let retry_after = Duration::from_millis(retry_after.unwrap_or(0));
            attempts.failed_retry_after(attempt, error, retry_after, now(), &mut steps);
            attempts.closed(attempt, now(), &mut steps);
        }
    }
}

async fn multipart(recording: &RecordingUpload) -> Result<Form, UploadError> {
    let millis = recording.start_time_millis;
    let header = MetricsRecordingHeader {
        room_id: recording.room_id.clone(),
        job_id: recording.job_id.clone(),
        simulated: recording.simulated,
        redaction_enabled: recording.redaction_enabled,
        start_time: Some(pbjson_types::Timestamp {
            seconds: millis.div_euclid(1_000),
            nanos: (millis.rem_euclid(1_000) * 1_000_000) as i32,
        }),
        ..Default::default()
    };
    let header = Part::bytes(header.encode_to_vec())
        .file_name("header.binpb")
        .mime_str("application/protobuf")
        .map_err(upload_error)?;
    let mut form = Form::new().part("header", header);
    if let Some(transcript) = &recording.transcript {
        let part = Part::text(transcript.clone())
            .file_name("chat_history.json")
            .mime_str("application/json")
            .map_err(upload_error)?;
        form = form.part("chat_history", part);
    }
    if let Some(path) = &recording.audio_path {
        let file = tokio::fs::File::open(path).await.map_err(upload_error)?;
        let len = file.metadata().await.map_err(upload_error)?.len();
        if len > 0 {
            let chunks = stream::try_unfold(file, |mut file| async move {
                let mut bytes = vec![0; CHUNK_BYTES];
                let read = file.read(&mut bytes).await?;
                if read == 0 {
                    return Ok::<_, std::io::Error>(None);
                }
                bytes.truncate(read);
                Ok(Some((bytes::Bytes::from(bytes), file)))
            });
            let part = Part::stream_with_length(reqwest::Body::wrap_stream(chunks), len)
                .file_name("recording.ogg")
                .mime_str("audio/ogg")
                .map_err(upload_error)?;
            form = form.part("audio", part);
        }
    }
    Ok(form)
}

async fn response_body(response: reqwest::Response) -> Result<Vec<u8>, UploadError> {
    let mut body = Vec::new();
    let mut chunks = response.bytes_stream();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(upload_error)?;
        if body.len().saturating_add(chunk.len()) > RESPONSE_BYTES {
            return Err(UploadError("recording upload response exceeds the 64 KiB limit".into()));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn retry_delay(body: &[u8]) -> Option<u64> {
    let status = tonic_types::pb::Status::decode(body).ok()?;
    for detail in status.details {
        if detail.type_url == "type.googleapis.com/google.rpc.RetryInfo" {
            let retry = tonic_types::pb::RetryInfo::decode(detail.value.as_slice()).ok()?;
            let delay = retry.retry_delay?;
            if delay.seconds < 0 || !(0..1_000_000_000).contains(&delay.nanos) {
                return None;
            }
            return Some(
                (delay.seconds as u64)
                    .saturating_mul(1_000)
                    .saturating_add(delay.nanos as u64 / 1_000_000),
            );
        }
    }
    None
}

#[cfg(test)]
mod tests;
