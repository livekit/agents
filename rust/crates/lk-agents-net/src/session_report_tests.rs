use std::sync::Arc;

use livekit_protocol::MetricsRecordingHeader;
use lk_agents_abi::telemetry::report::{RecordingHeader, RecordingOptions, SessionReport};
use lk_agents_core::telemetry::report::RecordingUpload;
use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::recording::{Multipart, encode_header, retry_delay};
use super::*;
use crate::otlp::{CloudRecording, JobIdentity, Limits, ProcessIdentity, SdkName};

struct Request {
    path: String,
    headers: String,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }
}

/// A local endpoint answering each request in turn with `replies`, then 200.
async fn peer(
    replies: Vec<(u16, Vec<u8>)>,
) -> (String, tokio::sync::mpsc::UnboundedReceiver<Request>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut replies = replies.into_iter();
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut data = Vec::new();
            let end = loop {
                let mut bytes = [0; 8192];
                let n = socket.read(&mut bytes).await.unwrap();
                assert_ne!(n, 0);
                data.extend_from_slice(&bytes[..n]);
                if let Some(end) = data.windows(4).position(|s| s == b"\r\n\r\n") {
                    break end + 4;
                }
            };
            let headers = String::from_utf8(data[..end].to_vec()).unwrap();
            let path = headers.split(' ').nth(1).unwrap().to_owned();
            let mut request = Request { path, headers, body: Vec::new() };
            let len: usize =
                request.header("content-length").expect("a known length").parse().unwrap();
            while data.len() - end < len {
                let mut bytes = [0; 8192];
                let n = socket.read(&mut bytes).await.unwrap();
                assert_ne!(n, 0);
                data.extend_from_slice(&bytes[..n]);
            }
            request.body = data[end..].to_vec();
            let _ = tx.send(request);
            let (status, response) = replies.next().unwrap_or((200, Vec::new()));
            let head = format!(
                "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(&response).await.unwrap();
        }
    });
    (format!("http://{address}"), rx)
}

fn cloud(url: &str) -> CloudEndpoint {
    CloudEndpoint { base_url: url.to_owned(), api_key: "key".into(), api_secret: "secret".into() }
}

fn clock() -> Arc<dyn Clock> {
    Arc::new(crate::clock::RealClock::new())
}

fn recording(audio_path: Option<String>) -> RecordingUpload {
    RecordingUpload {
        header: RecordingHeader {
            room_id: "RM_1".into(),
            job_id: "AJ_1".into(),
            start_time_millis: 1_700_000_000_123,
            duration_millis: 9_500,
            simulated: true,
            redaction_enabled: true,
            session_summary: "{\"job_id\":\"AJ_1\"}".into(),
        },
        chat_history: Some("{\"items\":[{\"type\":\"message\"}]}".into()),
        audio_path,
    }
}

/// The multipart parts in order: each one's headers and bytes, read by its own `Content-Length`.
fn parts(request: &Request) -> Vec<(String, Vec<u8>)> {
    let content_type = request.header("content-type").unwrap();
    let boundary = content_type.split("boundary=").nth(1).unwrap();
    let mut rest = &request.body[..];
    let mut out = Vec::new();
    loop {
        let delimiter = format!("--{boundary}");
        assert!(rest.starts_with(delimiter.as_bytes()));
        rest = &rest[delimiter.len()..];
        if rest == b"--\r\n" {
            return out;
        }
        rest = &rest[2..];
        let end = rest.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let headers = String::from_utf8(rest[..end].to_vec()).unwrap();
        let len: usize = headers
            .lines()
            .find_map(|l| l.strip_prefix("Content-Length: "))
            .expect("every part has its own Content-Length")
            .parse()
            .unwrap();
        let body = rest[end + 4..end + 4 + len].to_vec();
        rest = &rest[end + 4 + len..];
        assert!(rest.starts_with(b"\r\n"));
        rest = &rest[2..];
        out.push((headers, body));
    }
}

// Catches: a part out of order, a part without its own length, a compressed text part, a header
// field lost, or an upload without the observability token.
#[tokio::test]
async fn the_upload_follows_the_ingest_contract() {
    let audio: Vec<u8> = (0..200_000).map(|n| (n % 251) as u8).collect();
    let path = std::env::temp_dir().join(format!("lk-recording-{}.ogg", rand::random::<u64>()));
    std::fs::write(&path, &audio).unwrap();
    let (url, mut rx) = peer(vec![]).await;
    let upload = recording(Some(path.to_str().unwrap().to_owned()));
    let sent = recording::upload(&cloud(&url), &upload, &*clock(), Duration::from_secs(10)).await;
    std::fs::remove_file(path).unwrap();
    assert_eq!(sent.unwrap(), Uploaded::Sent);
    let request = rx.recv().await.unwrap();
    assert_eq!(request.path, "/observability/recordings/v0");
    assert_eq!(request.header("content-encoding"), None);
    let token = request.header("authorization").unwrap().strip_prefix("Bearer ").unwrap();
    let claims = livekit_token::TokenVerifier::with_api_key("key", "secret").verify(token).unwrap();
    assert!(claims.observability.write);

    let parts = parts(&request);
    let names: Vec<&str> = parts
        .iter()
        .map(|(h, _)| h.split("name=\"").nth(1).unwrap().split('"').next().unwrap())
        .collect();
    assert_eq!(names, ["header", "chat_history", "audio"]);
    assert!(parts[0].0.contains("filename=\"header.binpb\"\r\nContent-Type: application/protobuf"));
    assert!(
        parts[1].0.contains("filename=\"chat_history.json\"\r\nContent-Type: application/json")
    );
    assert!(parts[2].0.contains("filename=\"recording.ogg\"\r\nContent-Type: audio/ogg"));
    assert_eq!(parts[1].1, upload.chat_history.as_ref().unwrap().as_bytes());
    assert_eq!(parts[2].1, audio);
    let header = MetricsRecordingHeader::decode(parts[0].1.as_slice()).unwrap();
    assert_eq!((header.room_id.as_str(), header.job_id.as_str()), ("RM_1", "AJ_1"));
    assert_eq!(header.duration, 9_500);
    assert!(header.simulated && header.redaction_enabled);
    assert_eq!(header.session_summary, "{\"job_id\":\"AJ_1\"}");
    let start = header.start_time.unwrap();
    assert_eq!((start.seconds, start.nanos), (1_700_000_000, 123_000_000));
    assert_eq!(parts[0].1, encode_header(&upload.header));
}

// Catches: a missing audio file failing the upload rather than sending the rest.
#[tokio::test]
async fn a_missing_audio_file_uploads_the_rest() {
    let (url, mut rx) = peer(vec![]).await;
    let upload = recording(Some("/nonexistent/recording.ogg".into()));
    let body = Multipart::new(&upload).await;
    assert!(!body.content_type().is_empty());
    let sent = recording::upload(&cloud(&url), &upload, &*clock(), Duration::from_secs(10)).await;
    assert_eq!(sent.unwrap(), Uploaded::Sent);
    let names: Vec<String> = parts(&rx.recv().await.unwrap())
        .into_iter()
        .map(|(h, _)| h.split("name=\"").nth(1).unwrap().split('"').next().unwrap().to_owned())
        .collect();
    assert_eq!(names, ["header", "chat_history"]);
}

// Catches: a `RetryInfo` refusal given up on, or the recording-disabled 401 retried.
#[tokio::test]
async fn retries_follow_retry_info_and_disabled_recording_stops() {
    let retry = tonic_types::pb::RetryInfo { retry_delay: Some(Default::default()) };
    let mut status = tonic_types::pb::Status { code: 8, ..Default::default() };
    status.details.push(Default::default());
    status.details[0].type_url = "type.googleapis.com/google.rpc.RetryInfo".into();
    status.details[0].value = retry.encode_to_vec();
    let body = status.encode_to_vec();
    assert_eq!(retry_delay(&body), Some(0));
    let (url, mut rx) = peer(vec![(429, body)]).await;
    let sent = send(&url).await;
    assert_eq!(sent.unwrap(), Uploaded::Sent);
    assert!(rx.recv().await.is_some() && rx.recv().await.is_some());

    let disabled = b"project data recording is disabled by owner".to_vec();
    let (url, mut rx) = peer(vec![(401, disabled), (500, Vec::new())]).await;
    let sent = send(&url).await;
    assert_eq!(sent.unwrap(), Uploaded::Disabled);
    rx.recv().await.unwrap();
    assert!(rx.try_recv().is_err());

    let (url, _rx) = peer(vec![(400, Vec::new())]).await;
    let sent = send(&url).await;
    assert!(sent.is_err());
}

async fn send(url: &str) -> Result<Uploaded, UploadError> {
    recording::upload(&cloud(url), &recording(None), &*clock(), Duration::from_secs(10)).await
}

fn job_exporter(url: &str) -> Exporter {
    let exporter = Exporter::start(
        &crate::task::TokioSpawner::current(),
        clock(),
        ProcessIdentity {
            sdk_name: SdkName::Python,
            sdk_version: "2.0.0".into(),
            cloud_agent_id: None,
            deployment_id: None,
        },
        Limits::default(),
    );
    exporter.configure_cloud(cloud(url));
    let key = JobKey("AJ_1".into());
    let identity = JobIdentity {
        room_id: "RM_1".into(),
        job_id: "AJ_1".into(),
        room_name: "room".into(),
        agent_name: None,
    };
    exporter.open_job(key.clone(), identity);
    let recording =
        CloudRecording { traces: true, logs: true, metrics: true, ..Default::default() };
    exporter.register_job(&key, recording);
    exporter
}

fn report() -> SessionReport {
    SessionReport {
        job_id: "AJ_1".into(),
        room_id: "RM_1".into(),
        timestamp: 1_700_000_000.0,
        sdk_version: "2.0.0".into(),
        ..Default::default()
    }
}

// Catches: the upload racing ahead of the job's records, or an upload after the project refused
// recording.
#[tokio::test]
async fn the_records_are_flushed_before_the_upload() {
    let (key, options, clock) = (JobKey("AJ_1".into()), RecordingOptions::default(), clock());
    let run = |url: String| {
        let (key, clock) = (key.clone(), clock.clone());
        async move {
            let (exporter, endpoint) = (job_exporter(&url), cloud(&url));
            let outputs =
                ReportOutputs { exporter: Some(&exporter), cloud: Some(&endpoint), clock: &*clock };
            export(outputs, &key, &report(), &options, false, Duration::from_secs(10))
                .await
                .unwrap();
            exporter
        }
    };
    let (url, mut rx) = peer(vec![]).await;
    run(url).await;
    let mut paths = Vec::new();
    while let Ok(request) = rx.try_recv() {
        paths.push(request.path);
    }
    assert_eq!(paths, ["/observability/logs/otlp/v0", "/observability/recordings/v0"]);

    let (url, mut rx) = peer(vec![(401, b"data recording is disabled".to_vec())]).await;
    let exporter = run(url).await;
    assert_eq!(rx.recv().await.unwrap().path, "/observability/logs/otlp/v0");
    assert!(exporter.job_disabled(&key));
    assert!(rx.try_recv().is_err());
    assert_eq!(exporter.diagnostics().recording_disabled, 1);
}
