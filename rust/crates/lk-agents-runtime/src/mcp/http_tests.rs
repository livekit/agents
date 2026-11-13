use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use lk_agents_abi::{OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use super::*;

const READ_INACTIVITY: Duration = Duration::from_millis(50);

struct Peer {
    url: Arc<str>,
    requests: Arc<AtomicUsize>,
    received: Arc<Mutex<Vec<u8>>>,
    stop: Option<oneshot::Sender<()>>,
    closed: Option<oneshot::Receiver<()>>,
    task: OwnedTask,
}

impl Peer {
    async fn new(response: Option<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap()).into();
        let requests = Arc::new(AtomicUsize::new(0));
        let count = requests.clone();
        let received = Arc::new(Mutex::new(Vec::new()));
        let wire = received.clone();
        let (stop, mut stopped) = oneshot::channel();
        let (closed_tx, closed) = oneshot::channel();
        let mut closed_tx = Some(closed_tx);
        let task = TokioSpawner::current().spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stopped => break,
                    accepted = listener.accept() => {
                        let (mut socket, _) = accepted.unwrap();
                        let mut request = Vec::new();
                        loop {
                            let mut bytes = [0; 1024];
                            let n = socket.read(&mut bytes).await.unwrap();
                            if n == 0 { return; }
                            request.extend_from_slice(&bytes[..n]);
                            if request.windows(4).any(|bytes| bytes == b"\r\n\r\n") { break; }
                        }
                        *wire.lock() = request;
                        count.fetch_add(1, Ordering::SeqCst);
                        if let Some(response) = &response {
                            socket.write_all(response.as_bytes()).await.unwrap();
                            loop {
                                let mut bytes = [0; 1024];
                                tokio::select! {
                                    _ = &mut stopped => return,
                                    read = socket.read(&mut bytes) => {
                                        if read.unwrap_or(0) == 0 {
                                            let _ = closed_tx.take().unwrap().send(());
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                        // The server received the POST but loses the response.
                    }
                }
            }
        });
        Self { url, requests, received, stop: Some(stop), closed: Some(closed), task }
    }

    async fn close(mut self) {
        self.stop.take();
        (&mut self.task).await.unwrap();
    }
}

fn client() -> CleanupClient {
    CleanupClient {
        http: Arc::new(GatewayHttp::new().unwrap()),
        headers: HeaderMap::new(),
        options: ApiConnectOptions {
            read_timeout: Some(READ_INACTIVITY.as_secs_f64()),
            ..Default::default()
        },
        clock: Arc::new(lk_agents_net::clock::RealClock::new()),
        receipt: Arc::new(TransportReceipt::default()),
        cancel: tokio_util::sync::CancellationToken::new(),
    }
}

#[tokio::test]
async fn a_passive_mcp_stream_keeps_its_cursor_without_timing_out_between_messages() {
    let frame = "retry: 25\nid: resume-1\n\n";
    let peer = Peer::new(Some(format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{frame}\r\n", frame.len()
    ))).await;
    let client = client();
    let mut stream =
        client.get_stream(peer.url.clone(), None, None, None, HashMap::new()).await.unwrap();
    let primer = stream.next().await.unwrap().unwrap();
    assert_eq!((primer.id.as_deref(), primer.retry), (Some("resume-1"), Some(25)));
    assert!(tokio::time::timeout(READ_INACTIVITY * 2, stream.next()).await.is_err());
    drop(stream);
    client.receipt.update(|state| state.close());
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, client.receipt.closed())
        .await
        .unwrap();
    peer.close().await;
}

#[tokio::test]
async fn a_failed_mcp_tool_post_is_not_replayed_by_the_transport() {
    let peer = Peer::new(None).await;
    let message = tool_call();
    let result = client().post_message(peer.url.clone(), message, None, None, HashMap::new()).await;
    assert!(result.is_err());
    assert_eq!(peer.requests.load(Ordering::SeqCst), 1);
    peer.close().await;
}

#[tokio::test]
async fn a_decoding_failure_releases_http_even_while_the_consumer_keeps_its_stream() {
    let mut peer = Peer::new(Some(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\ne\r\ndata: overflow\r\n".into()
    )).await;
    let client = client();
    let mut stream = client
        .get_stream_with_max_sse_event_size(peer.url.clone(), None, None, None, HashMap::new(), 8)
        .await
        .unwrap();
    assert!(stream.next().await.unwrap().is_err());
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, peer.closed.take().unwrap())
        .await
        .expect("framing failure must release the exchange before the consumer drops it")
        .unwrap();
    assert!(stream.next().await.is_none());
    peer.close().await;
}

fn tool_call() -> ClientJsonRpcMessage {
    serde_json::from_value(serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "change", "arguments": {}}
    }))
    .unwrap()
}

#[tokio::test]
async fn request_authorization_overrides_the_default_and_sdk_bearer() {
    let peer = Peer::new(Some(
        "HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
    ))
    .await;
    let mut client = client();
    client
        .headers
        .insert(reqwest::header::AUTHORIZATION, HeaderValue::from_static("Bearer default"));
    let custom = HashMap::from([(
        reqwest::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer custom"),
    )]);
    client
        .post_message(peer.url.clone(), tool_call(), None, Some("sdk".into()), custom)
        .await
        .unwrap();
    let wire = String::from_utf8(peer.received.lock().clone()).unwrap();
    assert!(
        wire.lines().any(|line| line.eq_ignore_ascii_case("authorization: Bearer custom")),
        "{wire}"
    );
    peer.close().await;
}

// Catches: the SDK awaiting its handshake POST without a cancellation, so a server that accepts
// and never answers held a cancelled connection until `read_timeout`: setup took init_timeout
// plus the teardown deadline, delaying a call's first reply.
#[tokio::test]
async fn a_cancelled_connection_abandons_a_post_the_server_never_answers() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url: Arc<str> = format!("http://{}/mcp", listener.local_addr().unwrap()).into();
    let silent = tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.unwrap();
        std::future::pending::<()>().await;
    });
    let client = CleanupClient { options: ApiConnectOptions::default(), ..client() };
    let cancel = client.cancel.clone();
    let post = client.post_message(url, tool_call(), None, None, HashMap::new());
    tokio::pin!(post);
    assert!(tokio::time::timeout(READ_INACTIVITY, &mut post).await.is_err(), "no answer yet");
    cancel.cancel();
    let result = tokio::time::timeout(Duration::from_secs(1), post).await.expect("abandoned");
    assert!(matches!(result, Err(Error::TransportChannelClosed)), "{result:?}");
    silent.abort();
}
