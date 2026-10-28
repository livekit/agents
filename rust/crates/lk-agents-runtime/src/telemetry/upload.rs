//! Bounded recording uploads: small transcript bodies and streamed recording files, using the
//! process telemetry admission budget. Cancelling an upload drops its async transport and file.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{StreamExt, stream};
use livekit_protocol::MetricsRecordingHeader;
use lk_agents_abi::telemetry::report::UploadGate;
use lk_agents_abi::telemetry::report::projection::SessionReportProjection as _;
use lk_agents_abi::telemetry::upload::{
    PreparedTranscript, RecordingUpload, SessionReportUpload, UploadError,
};
use prost::Message;
use reqwest::multipart::{Form, Part};
use tokio::io::AsyncReadExt;

use super::upload_policy::{UploadOutcome, UploadPolicy};

const CHUNK_BYTES: usize = 64 * 1024;
const TRANSCRIPT_BYTES: usize = 2 * 1024 * 1024;
const RESPONSE_BYTES: usize = 64 * 1024;
const UPLOAD_TIMEOUT: Duration =
    Duration::from_secs(lk_agents_abi::telemetry::policy::UPLOAD_TIMEOUT_SECONDS);

pub struct SessionReportUploader {
    api_key: String,
    api_secret: String,
    disabled: AtomicBool,
    client: reqwest::Client,
}

impl SessionReportUpload for SessionReportUploader {
    fn new(api_key: String, api_secret: String) -> Result<Self, UploadError> {
        let create = move || {
            // Validate credentials before creating a client or retaining a report.
            lk_agents_net::token::observability_bearer(&api_key, &api_secret)
                .map_err(upload_error)?;
            lk_agents_net::transport::install_tls_provider();
            let client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(upload_error)?;
            Ok(Self { api_key, api_secret, disabled: AtomicBool::new(false), client })
        };
        #[cfg(feature = "python")]
        {
            pyo3::Python::attach(|py| py.detach(create))
        }
        #[cfg(not(feature = "python"))]
        {
            create()
        }
    }

    fn bearer(&self) -> Result<String, UploadError> {
        lk_agents_net::token::observability_bearer(&self.api_key, &self.api_secret)
            .map_err(upload_error)
    }

    fn content_policy(
        &self,
        transcript_requested: bool,
        audio_requested: bool,
        capture_content: bool,
        redaction_enabled: bool,
    ) -> (bool, bool) {
        let projection = lk_agents_abi::telemetry::report::projection::ReportProjection::new(
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

const TRANSCRIPT_ITEMS: usize = 10_000;
const TRANSCRIPT_ITEM_BYTES: usize = 64 * 1024;

/// Retains only bounded JSON, rather than a second copy of every native chat item.
pub struct TranscriptBuilder {
    state: Mutex<TranscriptState>,
    _permit: Arc<super::work::WorkPermit>,
}

struct TranscriptState {
    bytes: Vec<u8>,
    items: usize,
    input_items: usize,
    events: Vec<serde_json::Value>,
    event_bytes: usize,
    failure: Option<String>,
    finished: bool,
}

impl PreparedTranscript for TranscriptBuilder {
    fn new() -> Result<Self, UploadError> {
        let permit = super::work::try_admit_upload(TRANSCRIPT_BYTES)
            .ok_or_else(|| UploadError("session report preparation capacity exhausted".into()))?;
        let mut bytes = Vec::with_capacity(TRANSCRIPT_BYTES);
        bytes.extend_from_slice(b"{\"items\":[");
        Ok(Self {
            state: Mutex::new(TranscriptState {
                bytes,
                items: 0,
                input_items: 0,
                events: Vec::new(),
                event_bytes: 0,
                failure: None,
                finished: false,
            }),
            _permit: Arc::new(permit),
        })
    }

    fn input_batch_size(&self) -> usize {
        1
    }

    fn content_policy(&self) -> (bool, bool) {
        let options = transcript_options();
        (!options.exclude_image, !options.exclude_audio)
    }

    fn check_input(&self, items: usize, item_bytes: usize) -> Result<(), UploadError> {
        let failure = if items > TRANSCRIPT_ITEMS {
            Some("session transcript exceeds the 10000 item limit")
        } else if item_bytes > TRANSCRIPT_ITEM_BYTES {
            Some("session transcript item exceeds the 64 KiB input limit")
        } else {
            None
        };
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if state.finished {
            return Err(UploadError("session transcript is already finished".into()));
        }
        if let Some(failure) = failure {
            state.failure = Some(failure.into());
            state.bytes.clear();
        }
        if let Some(failure) = &state.failure { Err(UploadError(failure.clone())) } else { Ok(()) }
    }

    fn append(&self, item: lk_agents_abi::ChatItem) -> Result<(), UploadError> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(failure) = &state.failure {
            return Err(UploadError(failure.clone()));
        }
        if state.finished {
            return Err(UploadError("session transcript is already finished".into()));
        }
        let result = append_item(&mut state, &item);
        if let Err(error) = &result {
            state.failure = Some(error.0.clone());
            state.bytes.clear();
        }
        result
    }

    fn finish(&self) -> Result<String, UploadError> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(failure) = &state.failure {
            return Err(UploadError(failure.clone()));
        }
        if !state.finished {
            state.bytes.extend_from_slice(b"]}");
            state.finished = true;
        }
        String::from_utf8(state.bytes.clone()).map_err(upload_error)
    }

    fn append_event(&self, event: serde_json::Value) -> Result<bool, UploadError> {
        use lk_agents_abi::telemetry::policy;
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(failure) = &state.failure {
            return Err(UploadError(failure.clone()));
        }
        if state.finished {
            return Err(UploadError("session transcript is already finished".into()));
        }
        let mut encoded =
            BoundedWriter { bytes: Vec::new(), limit: policy::REPORT_EVENT_RECORD_BYTES };
        if serde_json::to_writer(&mut encoded, &event).is_err() {
            return Ok(false);
        }
        if state.events.len() >= policy::REPORT_EVENT_ITEMS
            || encoded.bytes.len() > policy::REPORT_EVENT_BYTES.saturating_sub(state.event_bytes)
        {
            return Ok(false);
        }
        if state.events.is_empty() && !self._permit.try_add_bytes(policy::REPORT_EVENT_BYTES) {
            return Ok(false);
        }
        state.event_bytes += encoded.bytes.len();
        state.events.push(event);
        Ok(true)
    }

    async fn finish_report(
        &self,
        mut report: lk_agents_abi::telemetry::report::projection::SessionReportSnapshot,
        path: String,
    ) -> Result<(), UploadError> {
        let output_permit =
            super::work::try_admit_upload(lk_agents_abi::telemetry::policy::REPORT_RETAINED_BYTES)
                .ok_or_else(|| {
                    UploadError("console report preparation capacity exhausted".into())
                })?;
        let workers = crate::process::workers().map_err(upload_error)?;
        let permit = workers
            .try_admit(crate::media_workers::MediaWorkerClass::Formatting)
            .map_err(upload_error)?;
        let (bytes, events) = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(failure) = &state.failure {
                return Err(UploadError(failure.clone()));
            }
            if !state.finished {
                state.bytes.extend_from_slice(b"]}");
                state.finished = true;
            }
            (std::mem::take(&mut state.bytes), std::mem::take(&mut state.events))
        };
        let input_permit = self._permit.clone();
        workers
            .spawn_admitted(permit, move || {
                let _retained = (input_permit, output_permit);
                report.events = events;
                // Keep the existing bounded transcript bytes; a second parsed history would retain
                // one allocation per JSON node during serialization.
                let history = serde_json::value::RawValue::from_string(
                    String::from_utf8(bytes).map_err(upload_error)?,
                )
                .map_err(upload_error)?;
                report.chat_history = None;
                #[derive(serde::Serialize)]
                struct ConsoleReport<'a> {
                    #[serde(flatten)]
                    report: lk_agents_abi::telemetry::report::projection::SessionReportSnapshot,
                    chat_history: &'a serde_json::value::RawValue,
                }
                let mut encoded = BoundedWriter {
                    bytes: Vec::new(),
                    limit: lk_agents_abi::telemetry::policy::REPORT_RETAINED_BYTES,
                };
                serde_json::to_writer(
                    &mut encoded,
                    &ConsoleReport { report, chat_history: &history },
                )
                .map_err(upload_error)?;
                let path = std::path::Path::new(&path);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(upload_error)?;
                }
                std::fs::write(path, &encoded.bytes).map_err(upload_error)
            })
            .await
            .map_err(upload_error)?
    }
}

fn transcript_options() -> lk_agents_abi::ToDictOptions {
    lk_agents_abi::ToDictOptions { exclude_timestamp: false, ..Default::default() }
}

fn append_item(
    state: &mut TranscriptState,
    item: &lk_agents_abi::ChatItem,
) -> Result<(), UploadError> {
    if state.input_items >= TRANSCRIPT_ITEMS {
        return Err(UploadError("session transcript exceeds the 10000 item limit".into()));
    }
    state.input_items += 1;
    let options = transcript_options();
    let Some(projected) =
        lk_agents_abi::item_to_dict(item, &options, &Default::default()).map_err(upload_error)?
    else {
        return Ok(());
    };
    let mut item = BoundedWriter { bytes: Vec::new(), limit: TRANSCRIPT_ITEM_BYTES };
    serde_json::to_writer(&mut item, &projected).map_err(upload_error)?;
    let separator = usize::from(state.items > 0);
    if item.bytes.len().saturating_add(separator + 2)
        > TRANSCRIPT_BYTES.saturating_sub(state.bytes.len())
    {
        return Err(UploadError("session transcript exceeds the 2 MiB upload limit".into()));
    }
    if separator > 0 {
        state.bytes.push(b',');
    }
    state.bytes.extend_from_slice(&item.bytes);
    state.items += 1;
    Ok(())
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl std::io::Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::FileTooLarge,
                format!("session recording exceeds its {} KiB serialized limit", self.limit / 1024),
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
