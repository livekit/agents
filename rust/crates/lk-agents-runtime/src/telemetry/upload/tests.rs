use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

static UPLOAD_TESTS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Request {
    headers: String,
    body: Vec<u8>,
}

async fn peer(replies: Vec<(u16, Vec<u8>)>) -> (String, tokio::task::JoinHandle<Vec<Request>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task =
        tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, response) in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut data = Vec::new();
                let header_end = loop {
                    let mut bytes = [0; 8192];
                    let n = socket.read(&mut bytes).await.unwrap();
                    assert_ne!(n, 0);
                    data.extend_from_slice(&bytes[..n]);
                    if let Some(end) = data.windows(4).position(|s| s == b"\r\n\r\n") {
                        break end + 4;
                    }
                };
                let headers = String::from_utf8(data[..header_end].to_vec()).unwrap();
                let len: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        key.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().unwrap())
                    })
                    .expect("all multipart parts have known lengths");
                while data.len() - header_end < len {
                    let mut bytes = [0; 8192];
                    let n = socket.read(&mut bytes).await.unwrap();
                    assert_ne!(n, 0);
                    data.extend_from_slice(&bytes[..n]);
                }
                requests.push(Request { headers, body: data[header_end..].to_vec() });
                socket.write_all(format!(
                "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response.len(),
            ).as_bytes()).await.unwrap();
                socket.write_all(&response).await.unwrap();
            }
            requests
        });
    (format!("http://{address}"), task)
}

fn recording(base_url: String) -> RecordingUpload {
    RecordingUpload {
        base_url,
        room_id: "RM_fixture".into(),
        job_id: "AJ_fixture".into(),
        simulated: true,
        redaction_enabled: false,
        start_time_millis: 1_700_000_000_123,
        transcript: Some("{\"messages\":[\"private transcript\"]}".into()),
        audio_path: None,
        disabled: false,
    }
}

#[tokio::test]
async fn session_upload_streams_file_bytes_and_uses_the_runtime_token() {
    let _serial = UPLOAD_TESTS.lock().await;
    let audio: Vec<u8> = (0..200_000).map(|n| (n % 251) as u8).collect();
    let path = std::env::temp_dir().join(format!("lk-telemetry-{}.ogg", rand::random::<u64>()));
    tokio::fs::write(&path, &audio).await.unwrap();
    let (url, peer) = peer(vec![(200, vec![])]).await;
    let uploader = SessionReportUploader::new("key".into(), "secret".into()).unwrap();
    let mut report = recording(url);
    report.audio_path = Some(path.to_str().unwrap().to_owned());
    let result = uploader.send(report).await;
    tokio::fs::remove_file(path).await.unwrap();
    assert!(result.unwrap());
    let requests = peer.await.unwrap();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert!(request.headers.starts_with("POST /observability/recordings/v0 HTTP/1.1"));
    let token = request
        .headers
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("authorization")
                .then(|| value.trim().strip_prefix("Bearer ").unwrap())
        })
        .unwrap();
    let claims = livekit_token::TokenVerifier::with_api_key("key", "secret").verify(token).unwrap();
    assert!(claims.observability.write);
    assert!(!claims.inference.perform);
    assert!(request.body.windows(audio.len()).any(|part| part == audio));
    assert!(
        request.body.windows(b"private transcript".len()).any(|part| part == b"private transcript")
    );
    let marker = b"Content-Type: application/protobuf\r\n\r\n";
    let start = request
        .body
        .windows(marker.len())
        .position(|s| s == marker)
        .or_else(|| request.body.windows(marker.len()).position(|s| s.eq_ignore_ascii_case(marker)))
        .unwrap()
        + marker.len();
    let end = request.body[start..].windows(2).position(|s| s == b"\r\n").unwrap() + start;
    let header = MetricsRecordingHeader::decode(&request.body[start..end]).unwrap();
    assert_eq!(
        (header.room_id.as_str(), header.job_id.as_str(), header.simulated),
        ("RM_fixture", "AJ_fixture", true)
    );
    assert_eq!(header.start_time.unwrap().nanos, 123_000_000);
}

#[tokio::test]
async fn disabled_and_redacted_recordings_never_open_the_endpoint() {
    let _serial = UPLOAD_TESTS.lock().await;
    let uploader = SessionReportUploader::new("key".into(), "secret".into()).unwrap();
    let mut report = recording("http://127.0.0.1:1".into());
    report.disabled = true;
    assert!(!uploader.send(report.clone()).await.unwrap());
    report.disabled = false;
    report.redaction_enabled = true;
    report.audio_path = Some("missing-sensitive-recording.ogg".into());
    assert!(uploader.send(report).await.unwrap());
}

#[tokio::test]
async fn a_recording_disabled_response_latches_without_retries() {
    let _serial = UPLOAD_TESTS.lock().await;
    let (url, peer) = peer(vec![(401, b"data recording is disabled for project".to_vec())]).await;
    let uploader = SessionReportUploader::new("key".into(), "secret".into()).unwrap();
    assert!(!uploader.send(recording(url)).await.unwrap());
    assert_eq!(peer.await.unwrap().len(), 1);
    assert!(!uploader.send(recording("http://127.0.0.1:1".into())).await.unwrap());
}

#[tokio::test]
async fn retry_info_is_decoded_from_the_upstream_google_rpc_record() {
    let _serial = UPLOAD_TESTS.lock().await;
    let retry = tonic_types::pb::RetryInfo { retry_delay: Some(Default::default()) };
    let mut response = tonic_types::pb::Status { code: 8, ..Default::default() };
    response.details.push(Default::default());
    response.details[0].type_url = "type.googleapis.com/google.rpc.RetryInfo".into();
    response.details[0].value = retry.encode_to_vec();
    let body = response.encode_to_vec();
    assert_eq!(retry_delay(&body), Some(0));
    let (url, peer) = peer(vec![(429, body), (200, vec![])]).await;
    let uploader = SessionReportUploader::new("key".into(), "secret".into()).unwrap();
    assert!(uploader.send(recording(url)).await.unwrap());
    assert_eq!(peer.await.unwrap().len(), 2);
}

#[tokio::test]
async fn ten_thousand_item_session_reports_serialize_without_stalling_the_loop() {
    use lk_agents_abi::{ChatContent, ChatItem, ChatMessage, Role};
    let _serial = UPLOAD_TESTS.lock().await;
    let items: Vec<_> = (0..10_000)
        .map(|index| {
            ChatItem::Message(ChatMessage::new(
                index.to_string(),
                Role::User,
                vec![ChatContent::Text("hello".into())],
                std::time::UNIX_EPOCH,
            ))
        })
        .collect();
    let uploader = SessionReportUploader::new("key".into(), "secret".into()).unwrap();
    let work = uploader.transcript(items);
    tokio::pin!(work);
    let cadence = Duration::from_millis(20);
    let mut heartbeat = tokio::time::interval(Duration::from_millis(1));
    heartbeat.tick().await;
    let mut previous = tokio::time::Instant::now();
    let mut worst = Duration::ZERO;
    let mut ticks = 0;
    let json = loop {
        tokio::select! {
            result = &mut work => {
                worst = worst.max(tokio::time::Instant::now().duration_since(previous));
                break result.unwrap();
            },
            _ = heartbeat.tick() => {
                let now = tokio::time::Instant::now();
                worst = worst.max(now.duration_since(previous));
                previous = now;
                ticks += 1;
            }
        }
    };
    eprintln!("session_report_off_loop items=10000 heartbeat_ticks={ticks} worst_gap={worst:?}");
    assert!(ticks > 0, "the session loop must progress while the large report is prepared");
    assert!(worst < cadence, "heartbeat gap {worst:?} exceeded one audio frame");
    let decoded: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded["items"].as_array().unwrap().len(), 10_000);
    assert!(json.len() < TRANSCRIPT_BYTES);
}

#[test]
fn transcript_serialization_preserves_the_existing_projection_and_rejects_oversize_content() {
    use lk_agents_abi::{ChatContent, ChatContext, ChatItem, ChatMessage, Role, ToDictOptions};
    let message = |text: String| {
        ChatItem::Message(ChatMessage::new(
            "message",
            Role::User,
            vec![ChatContent::Text(text)],
            std::time::UNIX_EPOCH,
        ))
    };
    let items = vec![message("a transcript".into())];
    let expected = ChatContext::new(items.clone())
        .to_dict(
            &ToDictOptions { exclude_timestamp: false, ..Default::default() },
            &Default::default(),
        )
        .unwrap();
    let actual: serde_json::Value =
        serde_json::from_str(&serialize_transcript(&items).unwrap()).unwrap();
    assert_eq!(actual, expected);
    let error = serialize_transcript(&[message("x".repeat(TRANSCRIPT_BYTES))]).unwrap_err();
    assert!(error.0.contains("2 MiB"));
}

#[tokio::test]
async fn an_ordinary_authorization_failure_is_permanent_but_does_not_disable_recording() {
    let _serial = UPLOAD_TESTS.lock().await;
    let (url, peer) =
        peer(vec![(401, b"missing observability grant".to_vec()), (200, vec![])]).await;
    let uploader = SessionReportUploader::new("key".into(), "secret".into()).unwrap();
    let failure = uploader.send(recording(url.clone())).await.unwrap_err();
    assert!(failure.0.contains("HTTP 401"));
    assert!(uploader.send(recording(url)).await.unwrap());
    assert_eq!(peer.await.unwrap().len(), 2);
}

#[tokio::test]
async fn a_stalled_recording_endpoint_hits_the_total_deadline_and_releases_admission() {
    let _serial = UPLOAD_TESTS.lock().await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (started, received) = tokio::sync::oneshot::channel();
    let peer = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut byte = [0; 1];
        socket.read_exact(&mut byte).await.unwrap();
        let _ = started.send(());
        std::future::pending::<()>().await;
        drop(socket);
    });
    let uploader = SessionReportUploader::new("key".into(), "secret".into()).unwrap();
    let mut upload =
        tokio::spawn(async move { uploader.send(recording(format!("http://{address}"))).await });
    let mut early_result = None;
    let started = tokio::select! {
        ready = received => ready.is_ok(),
        result = &mut upload => {
            early_result = Some(result);
            false
        }
    };
    if started {
        tokio::time::pause();
        tokio::time::advance(UPLOAD_TIMEOUT + Duration::from_millis(1)).await;
    }
    if !started && early_result.is_none() {
        upload.abort();
    }
    let result = match early_result {
        Some(result) => result,
        None => upload.await,
    };
    if started {
        tokio::time::resume();
    }
    peer.abort();
    let _ = peer.await;
    assert!(started, "upload failed before reaching the peer: {result:?}");
    let failure = result.unwrap().unwrap_err();
    assert!(failure.0.contains("deadline elapsed"), "{failure}");
    let first = super::super::work::try_admit_upload(1).expect("first upload slot released");
    let second = super::super::work::try_admit_upload(1).expect("second upload slot remains free");
    assert!(super::super::work::try_admit_upload(1).is_none());
    drop((first, second));
}
