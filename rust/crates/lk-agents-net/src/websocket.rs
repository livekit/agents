use crate::transport::{Headers, TransportError, WsMessage, WsReceiver, WsSender, WsTransport};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;

pub use tokio_tungstenite::tungstenite::{self, Message};

/// A socket for protocols that need binary frames and explicit ping/pong handling.
pub type RawWebSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Connect with an optional protocol message limit; `None` preserves the transport default.
/// The caller owns its connection deadline and the returned socket's lifecycle.
pub async fn connect_with_limit(
    url: &str,
    headers: Headers,
    max_message_size: Option<usize>,
) -> Result<RawWebSocket, TransportError> {
    connect_through(url, headers, max_message_size, None).await
}

/// [`connect_with_limit`] through `proxy` (`http://[user:password@]host[:port]`), tunnelled with
/// `CONNECT` as the host's HTTP session does with the job's proxy; `None` connects directly.
pub async fn connect_through(
    url: &str,
    headers: Headers,
    max_message_size: Option<usize>,
    proxy: Option<&str>,
) -> Result<RawWebSocket, TransportError> {
    use tungstenite::client::IntoClientRequest;

    let mut request =
        url.into_client_request().map_err(|error| TransportError::Connect(error.to_string()))?;
    for (name, value) in headers {
        let name = tungstenite::http::HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| TransportError::Connect(error.to_string()))?;
        let value = tungstenite::http::HeaderValue::from_str(&value)
            .map_err(|error| TransportError::Connect(error.to_string()))?;
        request.headers_mut().insert(name, value);
    }
    let config = max_message_size.map(|limit| {
        tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(Some(limit))
            .max_frame_size(Some(limit))
    });
    let uri = request.uri();
    let connector = if uri.scheme_str() == Some("wss") {
        Some(tokio_tungstenite::Connector::Rustls(
            crate::tls::config().await.map_err(|e| TransportError::Connect(e.to_string()))?,
        ))
    } else {
        None
    };
    let host = uri.host().ok_or_else(|| TransportError::Connect(format!("no host in {url}")))?;
    let host = host.trim_start_matches('[').trim_end_matches(']').to_owned();
    let port = uri.port_u16().unwrap_or(if uri.scheme_str() == Some("wss") { 443 } else { 80 });
    let stream = match proxy {
        Some(proxy) => tunnel(proxy, &host, port).await?,
        None => crate::dns::connect((host, port))
            .await
            .map_err(|error| TransportError::Connect(error.to_string()))?,
    };
    let (socket, _response) =
        tokio_tungstenite::client_async_tls_with_config(request, stream, config, connector)
            .await
            .map_err(classify_tungstenite)?;
    Ok(socket)
}

/// The longest proxy response head read before the proxy is refused.
const MAX_PROXY_HEAD: usize = 8 * 1024;

/// A TCP stream to `host:port` through the HTTP proxy at `proxy`, opened with `CONNECT`.
async fn tunnel(
    proxy: &str,
    host: &str,
    port: u16,
) -> Result<tokio::net::TcpStream, TransportError> {
    use base64::Engine as _;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let connect = |message: String| TransportError::Connect(message);
    let proxy = url::Url::parse(proxy).map_err(|e| connect(format!("invalid proxy URL: {e}")))?;
    if proxy.scheme() != "http" {
        return Err(connect(format!("unsupported proxy scheme {}", proxy.scheme())));
    }
    let proxy_host = proxy.host_str().ok_or_else(|| connect("no host in the proxy URL".into()))?;
    let proxy_host = proxy_host.trim_start_matches('[').trim_end_matches(']').to_owned();
    let proxy_port = proxy.port_or_known_default().unwrap_or(80);
    let mut stream = crate::dns::connect((proxy_host, proxy_port))
        .await
        .map_err(|error| connect(format!("proxy: {error}")))?;
    let authority =
        if host.contains(':') { format!("[{host}]:{port}") } else { format!("{host}:{port}") };
    let mut request = format!("CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n");
    if !proxy.username().is_empty() {
        let credentials = format!(
            "{}:{}",
            percent_decoded(proxy.username()),
            percent_decoded(proxy.password().unwrap_or(""))
        );
        let encoded = base64::engine::general_purpose::STANDARD.encode(credentials);
        request.push_str(&format!("Proxy-Authorization: Basic {encoded}\r\n"));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes()).await.map_err(|e| connect(format!("proxy: {e}")))?;
    // One byte at a time: the tunnel's first bytes after the head belong to the TLS handshake.
    let mut head = Vec::new();
    while !head.ends_with(b"\r\n\r\n") {
        if head.len() >= MAX_PROXY_HEAD {
            return Err(connect("the proxy's response head is too long".into()));
        }
        let mut byte = [0u8; 1];
        match stream.read(&mut byte).await {
            Ok(0) => return Err(connect("the proxy closed the connection".into())),
            Ok(_) => head.push(byte[0]),
            Err(error) => return Err(connect(format!("proxy: {error}"))),
        }
    }
    let head = String::from_utf8_lossy(&head);
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<i64>().ok())
        .ok_or_else(|| connect("the proxy sent no status".into()))?;
    if !(200..300).contains(&status) {
        return Err(TransportError::Status {
            status,
            message: format!("proxy refused the tunnel: {}", response_reason(status)),
            headers: Headers::new(),
            body: None,
        });
    }
    Ok(stream)
}

/// A URL's percent-encoded user or password as the proxy expects it in its credentials.
fn percent_decoded(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let escaped = (bytes[i] == b'%')
            .then(|| value.get(i + 1..i + 3))
            .flatten()
            .and_then(|hex| u8::from_str_radix(hex, 16).ok());
        match escaped {
            Some(byte) => {
                out.push(byte);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn response_reason(status: i64) -> String {
    lk_agents_abi::errors::http_reason(status).unwrap_or("HTTP error").to_owned()
}

pub struct TungsteniteTransport;

/// [`TungsteniteTransport`] through the job's HTTP proxy, when it has one.
pub struct ProxiedTransport {
    pub proxy: Option<String>,
}

#[async_trait]
impl WsTransport for ProxiedTransport {
    /// Unbounded: the caller bounds it on its session clock.
    async fn connect(&self, url: &str, headers: Headers) -> Result<Halves, TransportError> {
        split(connect_through(url, headers, None, self.proxy.as_deref()).await?)
    }
}

/// A connected socket's two halves.
type Halves = (Arc<dyn WsSender>, Box<dyn WsReceiver>);

fn split(stream: RawWebSocket) -> Result<Halves, TransportError> {
    let (sink, source) = stream.split();
    Ok((
        Arc::new(TungsteniteSender {
            sink: tokio::sync::Mutex::new(sink),
            closed: std::sync::atomic::AtomicBool::new(false),
        }),
        Box::new(TungsteniteReceiver { source, close_code: None, close_reason: None }),
    ))
}

#[async_trait]
impl WsTransport for TungsteniteTransport {
    /// Unbounded: the clients wrap this in `lk_agents_abi::timeout` on their session clock.
    async fn connect(
        &self,
        url: &str,
        headers: Headers,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        split(connect_with_limit(url, headers, None).await?)
    }
}

/// A rejected handshake's HTTP status (401 not retryable, 503 retryable).
pub fn classify_tungstenite(err: tokio_tungstenite::tungstenite::Error) -> TransportError {
    use tokio_tungstenite::tungstenite::Error;
    match err {
        Error::Http(response) => {
            let status = i64::from(response.status().as_u16());
            let headers: Headers = response
                .headers()
                .iter()
                .filter_map(|(k, v)| v.to_str().ok().map(|v| (k.as_str().to_owned(), v.to_owned())))
                .collect();
            let body = response.body().as_ref().map(|b| String::from_utf8_lossy(b).into_owned());
            TransportError::Status {
                status,
                message: body.clone().unwrap_or_else(|| response_reason(status)),
                headers,
                body,
            }
        }
        Error::ConnectionClosed | Error::AlreadyClosed => TransportError::Closed,
        Error::Io(e) => TransportError::Connect(e.to_string()),
        other => TransportError::Io(other.to_string()),
    }
}

struct TungsteniteSender {
    /// A `tokio` mutex: held across the `send` await, and `&self` is required.
    sink: tokio::sync::Mutex<futures_util::stream::SplitSink<RawWebSocket, Message>>,
    closed: std::sync::atomic::AtomicBool,
}

#[async_trait]
impl WsSender for TungsteniteSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(TransportError::Closed);
        }
        self.sink
            .lock()
            .await
            .send(tokio_tungstenite::tungstenite::Message::text(text))
            .await
            .map_err(classify_tungstenite)
    }

    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        if self.closed.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(TransportError::Closed);
        }
        self.sink
            .lock()
            .await
            .send(tokio_tungstenite::tungstenite::Message::binary(data))
            .await
            .map_err(classify_tungstenite)
    }

    fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn close(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        // The close frame is the peer's courtesy; a broken connection has nothing left to close.
        if let Err(error) = self.sink.lock().await.close().await {
            match classify_tungstenite(error) {
                TransportError::Closed => {}
                error => {
                    log::debug!(target: lk_agents_abi::LOG_TARGET, "websocket close failed: {error}")
                }
            }
        }
    }
}

struct TungsteniteReceiver {
    source: futures_util::stream::SplitStream<RawWebSocket>,
    close_code: Option<u16>,
    close_reason: Option<String>,
}

#[async_trait]
impl WsReceiver for TungsteniteReceiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        use tokio_tungstenite::tungstenite::Message;
        Ok(match self.source.next().await {
            Some(Ok(Message::Text(text))) => WsMessage::Text(text.to_string()),
            Some(Ok(Message::Binary(data))) => WsMessage::Binary(data.to_vec()),
            Some(Ok(Message::Close(frame))) => {
                if let Some(frame) = frame {
                    self.close_code = Some(u16::from(frame.code));
                    self.close_reason =
                        Some(frame.reason.to_string()).filter(|reason| !reason.is_empty());
                }
                WsMessage::Closed
            }
            None => WsMessage::Closed,
            // Ping/pong are answered by tungstenite.
            Some(Ok(_)) => WsMessage::Other,
            Some(Err(err)) => return Err(classify_tungstenite(err)),
        })
    }

    fn close_code(&self) -> Option<u16> {
        self.close_code
    }

    fn close_reason(&self) -> Option<String> {
        self.close_reason.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;

    /// A proxy that answers `CONNECT` with `status`, then serves the websocket itself on the
    /// tunnel; returns the request head it received.
    async fn proxy(status: &str) -> (String, impl Future<Output = String>) {
        use tokio::io::AsyncReadExt;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://user%40corp:p%3Ass@{}", listener.local_addr().unwrap());
        let status = status.to_owned();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut head = Vec::new();
            while !head.ends_with(b"\r\n\r\n") {
                let mut byte = [0u8; 1];
                socket.read_exact(&mut byte).await.unwrap();
                head.push(byte[0]);
            }
            socket.write_all(format!("HTTP/1.1 {status}\r\n\r\n").as_bytes()).await.unwrap();
            if status.starts_with("200") {
                let mut ws = tokio_tungstenite::accept_async(socket).await.unwrap();
                ws.send(Message::text("through")).await.unwrap();
            }
            String::from_utf8(head).unwrap()
        });
        (url, async move { server.await.unwrap() })
    }

    // Catches: a job's HTTP proxy ignored by the Rust transport, so a socket the host's session
    // would tunnel connects directly (and fails behind a corporate egress).
    #[tokio::test]
    async fn a_proxy_tunnels_the_socket_with_its_credentials() {
        let (proxy_url, server) = proxy("200 Connection established").await;
        let mut socket =
            connect_through("ws://example.test:9000/v1", Headers::new(), None, Some(&proxy_url))
                .await
                .unwrap();
        let first = socket.next().await.unwrap().unwrap();
        assert_eq!(first.into_text().unwrap().as_str(), "through");
        let head = server.await;
        assert!(head.starts_with("CONNECT example.test:9000 HTTP/1.1\r\n"), "{head}");
        // "user@corp:p:ss", percent-decoded from the URL, as Basic credentials
        assert!(head.contains("Proxy-Authorization: Basic dXNlckBjb3JwOnA6c3M=\r\n"), "{head}");
    }

    // Catches: a proxy's refusal read as a transport failure with no status, so a 407 is retried.
    #[tokio::test]
    async fn a_refused_tunnel_carries_the_proxy_status() {
        let (proxy_url, server) = proxy("407 Proxy Authentication Required").await;
        let error =
            connect_through("wss://example.test/v1", Headers::new(), None, Some(&proxy_url))
                .await
                .unwrap_err();
        server.await;
        assert!(matches!(error, TransportError::Status { status: 407, .. }), "{error:?}");
    }

    #[tokio::test]
    async fn connection_message_limit_refuses_an_oversized_prefix_without_waiting_for_payload() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let (release, hold) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(socket).await.unwrap();
            // A binary message advertising 129 bytes, with all payload withheld.
            socket.get_mut().write_all(&[0x82, 126, 0, 129]).await.unwrap();
            let _ = hold.await;
        });
        let mut socket = connect_with_limit(&url, Headers::new(), Some(128)).await.unwrap();
        let result = tokio::time::timeout(Duration::from_secs(1), socket.next()).await;
        let _ = release.send(());
        server.await.unwrap();
        assert!(matches!(
            result.expect("the size limit must be checked before waiting for payload"),
            Some(Err(tungstenite::Error::Capacity(_)))
        ));
    }
}
