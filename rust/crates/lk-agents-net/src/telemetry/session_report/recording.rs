//! The recording upload: one multipart request per attempt to `/observability/recordings/v0`, as
//! Cloud ingest reads it (docs/internals/telemetry-reference.md "Cloud ingest and redaction"): the
//! `header` part first, each part with its own `Content-Length`, text parts uncompressed, the
//! audio streamed from its file.

use std::time::Duration;

use futures_util::{StreamExt as _, stream};
use livekit_protocol::MetricsRecordingHeader;
use lk_agents_abi::Clock;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::telemetry::report::RecordingHeader;
use lk_agents_core::attempts::{AttemptStep, Attempts, RequestPolicy};
use lk_agents_core::telemetry::report::RecordingUpload;
use prost::Message;

use super::ReportAdmission;
use super::UploadError;
use crate::telemetry::otlp::CloudEndpoint;
use crate::telemetry::otlp::cloud::{Answer, Unsent};

const CHUNK_BYTES: usize = 64 * 1024;
/// Four attempts, bounded together by the caller's limit; a `RetryInfo` delay overrides the
/// jittered backoff.
const ATTEMPTS: RequestPolicy = RequestPolicy {
    connect_timeout: None,
    write_timeout: None,
    read_timeout: None,
    total_timeout: None,
    max_retry: 3,
    retry_interval: Duration::from_secs(2),
};

/// What Cloud made of an upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uploaded {
    Sent,
    /// The project disabled recording: the job's other exports stop too.
    Disabled,
}

/// Uploads `recording` to `cloud` within `limit`, retrying refusals that carry a `RetryInfo` and
/// transport failures. A missing audio file uploads the rest.
pub async fn upload(
    cloud: &CloudEndpoint,
    recording: &RecordingUpload,
    clock: &dyn Clock,
    limit: Duration,
) -> Result<Uploaded, UploadError> {
    let send = async {
        let _permit = ReportAdmission::process().upload().await;
        attempts(cloud, recording, clock).await
    };
    lk_agents_abi::timeout(clock, limit, send)
        .await
        .map_err(|_| UploadError("session recording upload deadline elapsed".into()))?
}

/// One request per attempt: this is the retry loop.
async fn attempts(
    cloud: &CloudEndpoint,
    recording: &RecordingUpload,
    clock: &dyn Clock,
) -> Result<Uploaded, UploadError> {
    let mut steps = Vec::new();
    let mut attempts = Attempts::start(ATTEMPTS, clock.now(), &mut steps);
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
            let Some(at) = attempts.wake_at(None) else { return Err(UploadError(failure)) };
            clock.wait_until(at).await;
            attempts.wake(clock.now(), None, &mut steps);
            continue;
        };
        let body = Multipart::new(recording).await;
        let result = cloud
            .post("recordings/v0", |request| {
                request
                    .header(reqwest::header::CONTENT_TYPE, body.content_type())
                    .header(reqwest::header::CONTENT_LENGTH, body.len)
                    .body(body.into_body())
            })
            .await;
        let (error, retry_after) = match result {
            Ok(Answer::Sent) => return Ok(Uploaded::Sent),
            Ok(Answer::Disabled) => return Ok(Uploaded::Disabled),
            Ok(Answer::Refused { status, body }) => {
                failure = format!("recording upload failed with HTTP {status}");
                let retry_after = retry_delay(&body);
                let error = ApiError::from_http(&failure, status.into(), None);
                let retryable = error.retryable || retry_after.is_some();
                (error.with_retryable(retryable), retry_after)
            }
            Err(Unsent::Setup(error)) => return Err(UploadError(error)),
            Err(Unsent::Transport(err)) => {
                failure = format!("recording upload transport failed: {err}");
                let retryable = err.is_connect() || err.is_timeout() || err.is_body();
                (ApiError::connection(Some(failure.clone())).with_retryable(retryable), None)
            }
        };
        let retry_after = Duration::from_millis(retry_after.unwrap_or(0));
        attempts.failed_retry_after(attempt, error, retry_after, clock.now(), &mut steps);
        attempts.closed(attempt, clock.now(), &mut steps);
    }
}

/// The multipart body: every part framed with its own headers, the audio file streamed between
/// its framing.
pub(super) struct Multipart {
    boundary: String,
    head: Vec<u8>,
    audio: Option<(std::fs::File, u64)>,
    tail: Vec<u8>,
    pub(super) len: u64,
}

impl Multipart {
    pub(super) async fn new(recording: &RecordingUpload) -> Self {
        let boundary = format!("lk-recording-{:032x}", rand::random::<u128>());
        let mut head = Vec::new();
        let header = encode_header(&recording.header);
        part(&mut head, &boundary, "header", "header.binpb", "application/protobuf", &header);
        if let Some(history) = &recording.chat_history {
            let (name, file) = ("chat_history", "chat_history.json");
            part(&mut head, &boundary, name, file, "application/json", history.as_bytes());
        }
        let audio = match &recording.audio_path {
            Some(path) => open(path).await.filter(|(_, len)| *len > 0),
            None => None,
        };
        let mut tail = Vec::new();
        if let Some((_, len)) = &audio {
            part_head(&mut head, &boundary, "audio", "recording.ogg", "audio/ogg", *len);
            tail.extend_from_slice(b"\r\n");
        }
        tail.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let audio_len = audio.as_ref().map_or(0, |(_, len)| *len);
        let len = head.len() as u64 + audio_len + tail.len() as u64;
        Multipart { boundary, head, audio, tail, len }
    }

    pub(super) fn content_type(&self) -> String {
        format!("multipart/form-data; boundary={}", self.boundary)
    }

    fn into_body(self) -> reqwest::Body {
        let Multipart { head, audio, tail, .. } = self;
        let Some((file, _)) = audio else {
            return reqwest::Body::from([head, tail].concat());
        };
        let io = crate::blocking_io::blocking_io();
        let chunks = stream::try_unfold(file, move |mut file| {
            let io = io.clone();
            async move {
                let (file, bytes) = io
                    .run((), move || {
                        let mut bytes = vec![0; CHUNK_BYTES];
                        let read = std::io::Read::read(&mut file, &mut bytes)?;
                        bytes.truncate(read);
                        Ok::<_, std::io::Error>((file, bytes))
                    })
                    .await
                    .map_err(std::io::Error::other)??;
                Ok::<_, std::io::Error>(
                    (!bytes.is_empty()).then(|| (bytes::Bytes::from(bytes), file)),
                )
            }
        });
        let body = stream::once(async move { Ok(bytes::Bytes::from(head)) })
            .chain(chunks)
            .chain(stream::once(async move { Ok(bytes::Bytes::from(tail)) }));
        reqwest::Body::wrap_stream(body)
    }
}

fn part_head(out: &mut Vec<u8>, boundary: &str, name: &str, file: &str, kind: &str, len: u64) {
    out.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{file}\"\r\nContent-Type: {kind}\r\nContent-Length: {len}\r\n\r\n"
        )
        .as_bytes(),
    );
}

fn part(out: &mut Vec<u8>, boundary: &str, name: &str, file: &str, kind: &str, bytes: &[u8]) {
    part_head(out, boundary, name, file, kind, bytes.len() as u64);
    out.extend_from_slice(bytes);
    out.extend_from_slice(b"\r\n");
}

/// The audio file and its length; a file that cannot be read uploads without its part.
async fn open(path: &str) -> Option<(std::fs::File, u64)> {
    let owned = path.to_owned();
    let opened = crate::blocking_io::blocking_io()
        .run((), move || {
            let file = std::fs::File::open(owned)?;
            let len = file.metadata()?.len();
            Ok::<_, std::io::Error>((file, len))
        })
        .await;
    match opened {
        Ok(Ok(opened)) => Some(opened),
        Ok(Err(error)) => {
            log::warn!(
                target: "livekit.agents",
                "failed to read the audio recording, uploading without it (path={path}): {error}"
            );
            None
        }
        Err(error) => {
            log::warn!(target: "livekit.agents", "failed to read the audio recording: {error}");
            None
        }
    }
}

pub(super) fn encode_header(h: &RecordingHeader) -> Vec<u8> {
    let millis = h.start_time_millis;
    MetricsRecordingHeader {
        room_id: h.room_id.clone(),
        job_id: h.job_id.clone(),
        duration: h.duration_millis,
        simulated: h.simulated,
        redaction_enabled: h.redaction_enabled,
        session_summary: h.session_summary.clone(),
        start_time: Some(pbjson_types::Timestamp {
            seconds: millis.div_euclid(1_000),
            nanos: (millis.rem_euclid(1_000) * 1_000_000) as i32,
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

/// A `google.rpc.Status` body's `RetryInfo` delay, in milliseconds.
pub(super) fn retry_delay(body: &[u8]) -> Option<u64> {
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
