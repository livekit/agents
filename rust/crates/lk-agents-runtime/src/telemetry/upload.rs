//! Bounded recording uploads: small transcript bodies and streamed recording files, using the
//! process telemetry admission budget. Cancelling an upload drops its async transport and file.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures_util::{StreamExt, stream};
use livekit_protocol::MetricsRecordingHeader;
use lk_agents_abi::telemetry::report::UploadGate;
use lk_agents_abi::telemetry::upload::{RecordingUpload, SessionReportUpload, UploadError};
use prost::Message;
use reqwest::multipart::{Form, Part};
use tokio::io::AsyncReadExt;

use super::upload_policy::{UploadOutcome, UploadPolicy};

const CHUNK_BYTES: usize = 64 * 1024;
const TRANSCRIPT_BYTES: usize = 2 * 1024 * 1024;
const RESPONSE_BYTES: usize = 64 * 1024;
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(900);

pub struct SessionReportUploader {
    api_key: String,
    api_secret: String,
    disabled: AtomicBool,
    client: reqwest::Client,
}

#[async_trait::async_trait]
impl SessionReportUpload for SessionReportUploader {
    fn new(api_key: String, api_secret: String) -> Result<Self, UploadError> {
        // Validate credentials before creating a client or retaining a report.
        crate::token::observability_bearer(&api_key, &api_secret).map_err(upload_error)?;
        lk_agents_inference::transport::install_tls_provider();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(upload_error)?;
        Ok(Self { api_key, api_secret, disabled: AtomicBool::new(false), client })
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
        (
            !redaction_enabled && transcript_requested && capture_content,
            !redaction_enabled && audio_requested,
        )
    }

    async fn transcript(&self, items: Vec<lk_agents_abi::ChatItem>) -> Result<String, UploadError> {
        let permit = super::work::try_admit_upload(TRANSCRIPT_BYTES)
            .ok_or_else(|| UploadError("session report preparation capacity exhausted".into()))?;
        let handle = crate::process::handle().map_err(upload_error)?;
        let join = handle.spawn_blocking(move || {
            let _permit = permit;
            serialize_transcript(&items)
        });
        let (reply, answer) = tokio::sync::oneshot::channel();
        // The supervisor observes the blocking job even if its host caller is cancelled.
        handle.spawn(async move {
            let result = join.await.map_err(upload_error).and_then(|result| result);
            let _ = reply.send(result);
        });
        answer.await.map_err(upload_error)?
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
        let retained_bytes = transcript_bytes
            .saturating_mul(2)
            .saturating_add(recording.base_url.len())
            .saturating_add(recording.room_id.len())
            .saturating_add(recording.job_id.len())
            .saturating_add(recording.audio_path.as_ref().map_or(0, String::len))
            .saturating_add(CHUNK_BYTES + RESPONSE_BYTES);
        let _permit = super::work::try_admit_upload(retained_bytes)
            .ok_or_else(|| UploadError("session recording upload capacity exhausted".into()))?;
        let url =
            format!("{}/observability/recordings/v0", recording.base_url.trim_end_matches('/'));
        let url = reqwest::Url::parse(&url).map_err(upload_error)?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(UploadError("observability endpoint must use HTTP or HTTPS".into()));
        }
        tokio::time::timeout(UPLOAD_TIMEOUT, self.attempts(recording, url))
            .await
            .map_err(|_| UploadError("session recording upload deadline elapsed".into()))?
    }

    async fn attempts(
        &self,
        recording: RecordingUpload,
        url: reqwest::Url,
    ) -> Result<bool, UploadError> {
        let epoch = tokio::time::Instant::now();
        let mut policy = UploadPolicy::new(false);
        loop {
            let now = epoch.elapsed().as_millis() as u64;
            if policy.next_attempt(now).is_none() {
                return Err(UploadError("recording upload attempt was not due".into()));
            }
            let result = self
                .client
                .post(url.clone())
                .bearer_auth(self.bearer()?)
                .multipart(multipart(&recording).await?)
                .send()
                .await;
            let (outcome, failure) = match result {
                Ok(response) if response.status().is_success() => (UploadOutcome::Success, None),
                Ok(response) => {
                    let status = response.status();
                    let body = response_body(response).await?;
                    if UploadGate::is_disabled_response(status.as_u16(), &body) {
                        (UploadOutcome::Disabled, None)
                    } else {
                        let delay = retry_delay(&body);
                        let retryable = delay.is_some()
                            || status.as_u16() == 408
                            || status.as_u16() == 429
                            || status.is_server_error();
                        let outcome = if retryable {
                            UploadOutcome::Retryable(delay)
                        } else {
                            UploadOutcome::Permanent
                        };
                        (
                            outcome,
                            Some(format!("recording upload failed with HTTP {}", status.as_u16())),
                        )
                    }
                }
                Err(err) => {
                    let retryable = err.is_connect() || err.is_timeout() || err.is_body();
                    let outcome = if retryable {
                        UploadOutcome::Retryable(None)
                    } else {
                        UploadOutcome::Permanent
                    };
                    (outcome, Some(format!("recording upload transport failed: {err}")))
                }
            };
            let now = epoch.elapsed().as_millis() as u64;
            policy.observed(now, outcome, rand::random_range(0.0..=1.0));
            match outcome {
                UploadOutcome::Success => return Ok(true),
                UploadOutcome::Disabled => {
                    self.disabled.store(true, Ordering::Release);
                    return Ok(false);
                }
                _ if policy.finished() => return Err(UploadError(failure.unwrap_or_default())),
                _ => {
                    let due = policy.retry_at().expect("retry retains a supplied deadline");
                    tokio::time::sleep_until(epoch + Duration::from_millis(due)).await;
                }
            }
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

fn serialize_transcript(items: &[lk_agents_abi::ChatItem]) -> Result<String, UploadError> {
    #[derive(serde::Serialize)]
    struct Transcript<'a> {
        items: TranscriptItems<'a>,
    }
    struct TranscriptItems<'a>(&'a [lk_agents_abi::ChatItem]);
    impl serde::Serialize for TranscriptItems<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::{Error, SerializeSeq};
            let options =
                lk_agents_abi::ToDictOptions { exclude_timestamp: false, ..Default::default() };
            let mut sequence = serializer.serialize_seq(None)?;
            for item in self.0 {
                if let Some(item) = lk_agents_abi::item_to_dict(item, &options, &Default::default())
                    .map_err(S::Error::custom)?
                {
                    sequence.serialize_element(&item)?;
                }
            }
            sequence.end()
        }
    }
    struct BoundedWriter(Vec<u8>);
    impl std::io::Write for BoundedWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > TRANSCRIPT_BYTES.saturating_sub(self.0.len()) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::FileTooLarge,
                    "session transcript exceeds the 2 MiB upload limit",
                ));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut output = BoundedWriter(Vec::with_capacity(TRANSCRIPT_BYTES));
    serde_json::to_writer(&mut output, &Transcript { items: TranscriptItems(items) })
        .map_err(upload_error)?;
    String::from_utf8(output.0).map_err(upload_error)
}

#[cfg(test)]
mod tests;
