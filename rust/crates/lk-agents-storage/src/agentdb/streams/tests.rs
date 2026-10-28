use super::*;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

/// The documented HTTP stream boundary, with deliberately opaque upstream tokens.
struct Peer {
    streams: Streams,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Peer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Peer {
    async fn start() -> Self {
        Self::paginated(false).await
    }

    async fn paginated(paginated: bool) -> Self {
        super::super::install_tls_provider();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/streams", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = Vec::new();
                let (method, path, length, header_len) = loop {
                    let mut chunk = [0; 4096];
                    let n = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(n, 0);
                    bytes.extend_from_slice(&chunk[..n]);
                    let mut headers = [httparse::EMPTY_HEADER; 32];
                    let mut req = httparse::Request::new(&mut headers);
                    if let httparse::Status::Complete(len) = req.parse(&bytes).unwrap() {
                        let length = req
                            .headers
                            .iter()
                            .find(|h| h.name.eq_ignore_ascii_case("content-length"))
                            .map(|h| {
                                std::str::from_utf8(h.value).unwrap().parse::<usize>().unwrap()
                            })
                            .unwrap_or(0);
                        break (
                            req.method.unwrap().to_owned(),
                            req.path.unwrap().to_owned(),
                            length,
                            len,
                        );
                    }
                };
                while bytes.len() < header_len + length {
                    let mut chunk = [0; 4096];
                    let n = socket.read(&mut chunk).await.unwrap();
                    assert_ne!(n, 0);
                    bytes.extend_from_slice(&chunk[..n]);
                }
                let url = url::Url::parse(&format!("http://peer{path}")).unwrap();
                let offset = url
                    .query_pairs()
                    .find(|(k, _)| k == "offset")
                    .map(|(_, v)| v.into_owned())
                    .unwrap_or_default();
                let (status, headers, body) = if path.contains("AST_missing_next") {
                    (200, "", "[]")
                } else if method == "POST" {
                    (
                        200,
                        "Stream-Offset: anchor-first\r\nStream-Next-Offset: anchor-after-first\r\n",
                        "",
                    )
                } else {
                    match offset.as_str() {
                        "-1" if paginated => {
                            (200, "Stream-Next-Offset: anchor-after-first\r\n", "[\"café\"]")
                        }
                        "-1" | "anchor-first" => (
                            200,
                            "Stream-Next-Offset: anchor-end\r\nStream-Up-To-Date: true\r\n",
                            "[\"café\",2,3]",
                        ),
                        "anchor-after-first" => (
                            200,
                            "Stream-Next-Offset: anchor-end\r\nStream-Up-To-Date: true\r\n",
                            "[2,3]",
                        ),
                        "anchor-end" => (
                            200,
                            "Stream-Next-Offset: anchor-end\r\nStream-Up-To-Date: true\r\n",
                            "[]",
                        ),
                        _ => (404, "", "expired stream"),
                    }
                };
                let live = url.query_pairs().any(|(k, v)| k == "live" && v == "sse");
                let sse_body = format!(
                    "event: data\r\ndata: {body}\r\n\r\nevent: control\r\ndata: {{\"streamNextOffset\":\"anchor-end\",\"upToDate\":true,\"streamClosed\":true}}\r\n\r\n"
                );
                let body = if live { sse_body.as_str() } else { body };
                let response = format!(
                    "HTTP/1.1 {status} Response\r\nConnection: close\r\nContent-Length: {}\r\n{headers}\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        let streams =
            Streams::new(reqwest::Client::new(), base, TokenMinter::new("key", "secret").unwrap());
        Self { streams, task }
    }
}

#[tokio::test]
async fn append_cursor_starts_after_the_appended_record() {
    let peer = Peer::start().await;
    let cursor = peer.streams.append_now("AST_test", "\"café\"".as_bytes()).await.unwrap();
    let mut reader = peer.streams.read("AST_test", Some(&cursor), false);
    assert_eq!(reader.next().await.unwrap().unwrap(), b"2");
}

#[tokio::test]
async fn catchup_resume_in_the_middle_of_a_response_is_exact() {
    let peer = Peer::start().await;
    let mut reader = peer.streams.read("AST_test", None, false);
    assert_eq!(reader.next().await.unwrap().unwrap(), "\"café\"".as_bytes());
    let cursor = reader.next_offset().to_owned();
    let mut resumed = peer.streams.read("AST_test", Some(&cursor), true);
    assert_eq!(resumed.next().await.unwrap().unwrap(), b"2");
    assert_eq!(resumed.next().await.unwrap().unwrap(), b"3");
    assert!(resumed.next().await.unwrap().is_none());
}

#[tokio::test]
async fn live_resume_before_control_does_not_repeat_split_utf8_records() {
    let peer = Peer::start().await;
    let mut reader = peer.streams.read("AST_test", None, true);
    let body = "event: data\r\ndata: [\"café\",2,3]\r\n\r\n";
    let chunks: Vec<Result<bytes::Bytes, reqwest::Error>> =
        body.as_bytes().iter().map(|b| Ok(bytes::Bytes::copy_from_slice(&[*b]))).collect();
    reader.sse = Some(Sse {
        body: Box::pin(futures_util::stream::iter(chunks)),
        decoder: SseDecoder::new(),
        events: VecDeque::new(),
    });
    assert_eq!(reader.next().await.unwrap().unwrap(), "\"café\"".as_bytes());
    let cursor = reader.next_offset().to_owned();
    drop(reader); // the upstream control event never arrived
    let mut resumed = peer.streams.read("AST_test", Some(&cursor), true);
    assert_eq!(resumed.next().await.unwrap().unwrap(), b"2");
    assert_eq!(resumed.next().await.unwrap().unwrap(), b"3");
    assert!(resumed.next().await.unwrap().is_none());
}

#[tokio::test]
async fn replay_count_survives_changed_page_boundaries() {
    let peer = Peer::start().await;
    let mut reader = peer.streams.read("AST_test", None, false);
    reader.next().await.unwrap();
    assert_eq!(reader.next().await.unwrap().unwrap(), b"2");
    let cursor = reader.next_offset().to_owned();
    let regrouped = Peer::paginated(true).await;
    let mut resumed = regrouped.streams.read("AST_test", Some(&cursor), false);
    assert_eq!(resumed.next().await.unwrap().unwrap(), b"3");
    assert!(resumed.next().await.unwrap().is_none());
}

#[tokio::test]
async fn invalid_and_expired_positions_fail_explicitly() {
    let peer = Peer::start().await;
    for cursor in [
        "as2.unknown".to_owned(),
        "as1.%".to_owned(),
        "x".repeat(4097),
        Position { anchor: "-1".into(), consumed: MAX_RECORDS + 1 }.encode(),
        Position { anchor: "-1".into(), consumed: 4 }.encode(),
    ] {
        let mut reader = peer.streams.read("AST_test", Some(&cursor), false);
        assert!(matches!(reader.next().await, Err(StorageError::Invalid(_))));
        assert!(reader.next().await.unwrap().is_none(), "invalid reader remains settled");
    }
    let expired = Position::at("expired-official-anchor".into()).unwrap().encode();
    let mut reader = peer.streams.read("AST_test", Some(&expired), false);
    assert!(matches!(reader.next().await, Err(StorageError::NotFound(_))));
}

#[test]
fn replay_and_response_budgets_are_finite() {
    super::super::install_tls_provider();
    let streams = Streams::new(
        reqwest::Client::new(),
        "http://unused",
        TokenMinter::new("key", "secret").unwrap(),
    );
    let mut reader = streams.read("AST_test", None, false);
    assert!(reader.records(&vec![b' '; MAX_BYTES + 1]).is_err());
    reader.replay = 1;
    for _ in 0..128 {
        reader.replay_page().unwrap();
    }
    assert!(reader.replay_page().is_err());
}

#[tokio::test]
async fn live_rotation_without_control_replays_only_unconsumed_records() {
    let peer = Peer::start().await;
    let mut reader = peer.streams.read("AST_test", None, true);
    let chunks =
        vec![Ok(bytes::Bytes::copy_from_slice("event: data\ndata: [\"café\",2,3]\n\n".as_bytes()))];
    reader.sse = Some(Sse {
        body: Box::pin(futures_util::stream::iter(chunks)),
        decoder: SseDecoder::new(),
        events: VecDeque::new(),
    });
    for expected in ["\"café\"".as_bytes(), b"2", b"3"] {
        assert_eq!(reader.next().await.unwrap().unwrap(), expected);
    }
    // The peer replays all three retained records from the same official anchor. They must
    // be counted and skipped even though the first response ended before its control frame.
    assert!(reader.next().await.unwrap().is_none());
}

#[tokio::test]
async fn a_missing_official_next_offset_is_never_an_empty_cursor() {
    let peer = Peer::start().await;
    assert!(matches!(
        peer.streams.append_now("AST_missing_next", b"1").await,
        Err(StorageError::Unavailable(_))
    ));
    let mut reader = peer.streams.read("AST_missing_next", None, false);
    assert!(matches!(reader.next().await, Err(StorageError::Unavailable(_))));
}
