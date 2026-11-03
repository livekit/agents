//! The WebTransport worker connection (livekit/protocol#1728), client side of livekit-server's
//! `pkg/agent/endpoint/conformance/worker.go`. One QUIC session at `https://<host>:<port>/agent`;
//! the worker opens the control stream first, messages framed with a 4-byte big-endian length.
//! Every server-opened stream is an [`EndpointStream`]. No QUIC keep-alive: the 2.5 s status
//! update keeps it busy and quinn's 30 s idle timeout detects a dead peer.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use web_transport_quinn::proto::ConnectRequest;
use web_transport_quinn::{ClientBuilder, RecvStream, SendStream, Session};

use super::{Connected, WorkerSink, WorkerStream, WorkerTransport};
use super::{TransportError, TransportEvent};
use livekit_protocol::{ServerMessage, WorkerMessage};
use prost::Message as _;

use super::MAX_CONTROL_MESSAGE;
pub use livekit_protocol::agent_http::{AgentEndpointKind, HttpStreamResetCode, StreamPreamble};

/// `endpoint.SessionCloseOK`.
const SESSION_CLOSE_OK: u32 = 0;
/// `wire.MaxPreambleSize`: the one allocation a peer can size before any HTTP bytes are parsed.
const MAX_PREAMBLE: usize = 64 << 10;

/// One exchange stream the node opened on the session (livekit_agent.proto, "agent HTTP endpoints
/// data plane"): `[len:u32be][StreamPreamble]`, then opaque bytes. [`Self::read_preamble`] consumes
/// the framing; what follows is one HTTP/1.1 exchange to FIN or RESET_STREAM, or, after a `101`,
/// a byte pipe both ways. The core carries bytes; meaning is the host's.
pub struct EndpointStream {
    send: SendStream,
    recv: RecvStream,
}

impl std::fmt::Debug for EndpointStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EndpointStream").finish_non_exhaustive()
    }
}

/// The QUIC error code a reset carries: the `HttpStreamResetCode` number itself (`wire.StreamCode`).
fn stream_code(code: HttpStreamResetCode) -> u32 {
    code as u32
}

/// Control admission never waits for the data-plane owner or flow control.
#[derive(Clone)]
pub struct EndpointControl(tokio::sync::watch::Sender<EndpointControlState>);

#[derive(Clone, Copy, Default)]
struct EndpointControlState {
    finish: bool,
    reset: Option<HttpStreamResetCode>,
    stop: Option<HttpStreamResetCode>,
}

impl EndpointControl {
    pub fn finish(&self) {
        self.0.send_modify(|state| state.finish = true);
    }

    pub fn reset(&self, code: HttpStreamResetCode) {
        self.0.send_modify(|state| {
            state.reset.get_or_insert(code);
            state.stop.get_or_insert(code);
        });
    }

    pub fn stop(&self, code: HttpStreamResetCode) {
        self.0.send_modify(|state| {
            state.stop.get_or_insert(code);
        });
    }
}

/// The receiving half of an exchange: a node's QUIC stream, or a connection the public listener
/// accepted, starting with the request head it rewrote.
enum RecvHalf {
    Quic(RecvStream),
    Tcp { pending: Vec<u8>, half: tokio::net::tcp::OwnedReadHalf },
}

impl RecvHalf {
    async fn read(&mut self, buf: &mut [u8]) -> Result<Option<usize>, TransportError> {
        match self {
            RecvHalf::Quic(stream) => stream.read(buf).await.map_err(transport_err),
            RecvHalf::Tcp { pending, .. } if !pending.is_empty() => {
                let n = pending.len().min(buf.len());
                buf[..n].copy_from_slice(&pending[..n]);
                pending.drain(..n);
                Ok(Some(n))
            }
            RecvHalf::Tcp { half, .. } => {
                use tokio::io::AsyncReadExt as _;
                match half.read(buf).await.map_err(transport_err)? {
                    0 => Ok(None),
                    n => Ok(Some(n)),
                }
            }
        }
    }

    fn stop(&mut self, code: HttpStreamResetCode) {
        if let RecvHalf::Quic(stream) = self {
            let _ = stream.stop(stream_code(code));
        }
    }
}

/// The sending half; a TCP half is dropped (FIN) once the response ends either way.
enum SendHalf {
    Quic(SendStream),
    Tcp(Option<tokio::net::tcp::OwnedWriteHalf>),
}

impl SendHalf {
    async fn write_all(&mut self, bytes: &[u8]) -> Result<(), TransportError> {
        match self {
            SendHalf::Quic(stream) => stream.write_all(bytes).await.map_err(transport_err),
            SendHalf::Tcp(Some(half)) => {
                use tokio::io::AsyncWriteExt as _;
                half.write_all(bytes).await.map_err(transport_err)
            }
            SendHalf::Tcp(None) => Err(TransportError::Closed("endpoint response ended".into())),
        }
    }

    fn reset(&mut self, code: HttpStreamResetCode) {
        match self {
            SendHalf::Quic(stream) => {
                let _ = stream.reset(stream_code(code));
            }
            SendHalf::Tcp(half) => drop(half.take()),
        }
    }

    fn finish(&mut self) -> Result<(), TransportError> {
        match self {
            SendHalf::Quic(stream) => stream.finish().map_err(transport_err),
            SendHalf::Tcp(half) => {
                drop(half.take());
                Ok(())
            }
        }
    }
}

/// An exchange on a connection the public listener accepted: `head` (the rewritten request head
/// and any bytes read past it) is read first.
pub fn tcp_exchange(
    socket: tokio::net::TcpStream,
    head: Vec<u8>,
) -> (EndpointWriter, EndpointReader) {
    let (recv, send) = socket.into_split();
    halves(SendHalf::Tcp(Some(send)), RecvHalf::Tcp { pending: head, half: recv })
}

fn halves(send: SendHalf, recv: RecvHalf) -> (EndpointWriter, EndpointReader) {
    let control = EndpointControl(tokio::sync::watch::channel(EndpointControlState::default()).0);
    (
        EndpointWriter { stream: send, control: control.clone(), finished: false },
        EndpointReader { stream: recv, control },
    )
}

/// EOF means the peer half-closed its request.
pub struct EndpointReader {
    stream: RecvHalf,
    control: EndpointControl,
}

impl EndpointReader {
    pub async fn read(&mut self, buf: &mut [u8]) -> Result<Option<usize>, TransportError> {
        if self.control.0.borrow().stop.is_some() {
            self.apply_control();
            return Err(TransportError::Closed("endpoint request stopped".into()));
        }
        let mut control = self.control.0.subscribe();
        let result = {
            let read = Box::pin(self.stream.read(buf));
            let stopped = Box::pin(control.wait_for(|state| state.stop.is_some()));
            match futures_util::future::select(read, stopped).await {
                futures_util::future::Either::Left((result, stopped)) => {
                    drop(stopped);
                    Some(result)
                }
                futures_util::future::Either::Right((_, read)) => {
                    drop(read);
                    None
                }
            }
        };
        match result {
            Some(result) => {
                self.apply_control();
                result
            }
            None => {
                self.apply_control();
                Err(TransportError::Closed("endpoint request stopped".into()))
            }
        }
    }

    pub fn apply_control(&mut self) {
        if let Some(code) = self.control.0.borrow().stop {
            self.stream.stop(code);
        }
    }
}

pub struct EndpointWriter {
    stream: SendHalf,
    control: EndpointControl,
    finished: bool,
}

impl EndpointWriter {
    pub fn control(&self) -> EndpointControl {
        self.control.clone()
    }

    pub async fn write_all(&mut self, bytes: &[u8]) -> Result<(), TransportError> {
        let res = {
            let state = self.control.0.borrow();
            state.reset.is_some() || state.finish
        };
        if res {
            self.apply_control()?;
            return Err(TransportError::Closed("endpoint response stopped".into()));
        }
        let mut control = self.control.0.subscribe();
        let result = {
            let write = Box::pin(self.stream.write_all(bytes));
            let stopped = Box::pin(control.wait_for(|state| state.reset.is_some() || state.finish));
            match futures_util::future::select(write, stopped).await {
                futures_util::future::Either::Left((result, stopped)) => {
                    drop(stopped);
                    Some(result)
                }
                futures_util::future::Either::Right((_, write)) => {
                    drop(write);
                    None
                }
            }
        };
        match result {
            Some(result) => {
                self.apply_control()?;
                result
            }
            None => {
                self.apply_control()?;
                Err(TransportError::Closed("endpoint response stopped".into()))
            }
        }
    }

    pub fn apply_control(&mut self) -> Result<(), TransportError> {
        let control = *self.control.0.borrow();
        if let Some(code) = control.reset {
            self.stream.reset(code);
        } else if control.finish && !self.finished {
            self.stream.finish()?;
            self.finished = true;
        }
        Ok(())
    }
}

impl EndpointStream {
    /// Consumes the length-prefixed preamble. A length past the cap or a message that does not
    /// decode is a protocol error; the caller resets with `HsrProtocol`.
    pub async fn read_preamble(&mut self) -> Result<StreamPreamble, TransportError> {
        let mut header = [0u8; 4];
        self.read_exact(&mut header).await?;
        let n = u32::from_be_bytes(header) as usize;
        if n > MAX_PREAMBLE {
            return Err(TransportError::Protocol(format!("preamble too large: {n} bytes")));
        }
        let mut body = vec![0u8; n];
        self.read_exact(&mut body).await?;
        <StreamPreamble as prost::Message>::decode(body.as_slice())
            .map_err(|e| TransportError::Protocol(format!("preamble: {e}")))
    }

    async fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), TransportError> {
        let mut filled = 0;
        while filled < buf.len() {
            match self.recv.read(&mut buf[filled..]).await.map_err(transport_err)? {
                Some(n) => filled += n,
                None => {
                    return Err(TransportError::Protocol(
                        "stream ended inside the preamble".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn split(self) -> (EndpointWriter, EndpointReader) {
        halves(SendHalf::Quic(self.send), RecvHalf::Quic(self.recv))
    }
    /// Both directions, with `code`.
    pub fn reset(mut self, code: HttpStreamResetCode) {
        let _ = self.send.reset(stream_code(code));
        let _ = self.recv.stop(stream_code(code));
    }
    /// `HsrRefused`: the application never saw the request, so the front retries elsewhere.
    pub fn refuse(self) {
        self.reset(HttpStreamResetCode::HsrRefused);
    }
}

pub struct WtTransport {
    /// Skip certificate validation (a dev `livekit-server`'s self-signed identity).
    pub insecure: bool,
}

fn transport_err(e: impl std::fmt::Display) -> TransportError {
    TransportError::Io(e.to_string())
}

fn loopback_url(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        Some(url::Host::Domain(host)) => host == "localhost",
        None => false,
    }
}

#[async_trait]
impl WorkerTransport for WtTransport {
    fn kind(&self) -> super::TransportKind {
        super::TransportKind::WebTransport
    }

    async fn connect(
        &self,
        url: &str,
        worker_token: Option<&str>,
        headers: Vec<(String, String)>,
    ) -> Result<Connected, TransportError> {
        let url = super::agent_url(url, worker_token, true);
        let mut url = url.parse::<url::Url>().map_err(transport_err)?;
        if self.insecure && !loopback_url(&url) {
            return Err(TransportError::Io(
                "insecure worker transport is only allowed for loopback URLs".into(),
            ));
        }
        if self.insecure && url.host_str() == Some("localhost") {
            // Do not let DNS or a hosts-file entry turn the local-only bypass
            // into an unverified connection to a remote address.
            url.set_host(Some("127.0.0.1")).map_err(transport_err)?;
        }
        let mut request = ConnectRequest::new(url);
        for (name, value) in headers {
            let name: web_transport_quinn::http::HeaderName =
                name.parse().map_err(transport_err)?;
            request = request.with_header(name, value.parse().map_err(transport_err)?);
        }
        // `insecure` is reserved for local servers with a self-signed certificate.
        let client = if self.insecure {
            ClientBuilder::new()
                .dangerous()
                .with_no_certificate_verification()
                .map_err(transport_err)?
        } else {
            let mut tls = (*lk_agents_net::tls::config().await.map_err(transport_err)?).clone();
            tls.alpn_protocols = vec![web_transport_quinn::ALPN.as_bytes().to_vec()];
            let crypto =
                web_transport_quinn::quinn::crypto::rustls::QuicClientConfig::try_from(tls)
                    .map_err(transport_err)?;
            let config = web_transport_quinn::quinn::ClientConfig::new(Arc::new(crypto));
            let endpoint = web_transport_quinn::quinn::Endpoint::client(
                "[::]:0".parse().expect("valid unspecified address"),
            )
            .map_err(transport_err)?;
            web_transport_quinn::Client::new(endpoint, config)
        };
        let session = client.connect(request).await.map_err(transport_err)?;
        let (send, recv) = session.open_bi().await.map_err(transport_err)?;
        Ok(Connected {
            sink: Arc::new(WtSink { send: Mutex::new(send), session: session.clone() }),
            stream: Box::new(WtStream { recv, buf: Vec::new(), session }),
        })
    }
}

/// `WriteControlMessage`.
fn frame_control(message: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + message.len());
    out.extend_from_slice(&(message.len() as u32).to_be_bytes());
    out.extend_from_slice(message);
    out
}

/// `ReadControlMessage`: a complete message and its byte length, nothing yet, or an illegal length.
fn parse_control(buf: &[u8]) -> Result<Option<(Vec<u8>, usize)>, TransportError> {
    if buf.len() < 4 {
        return Ok(None);
    }
    let n = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if n > MAX_CONTROL_MESSAGE {
        return Err(TransportError::Io(format!("control message too large: {n} bytes")));
    }
    if buf.len() < 4 + n {
        return Ok(None);
    }
    Ok(Some((buf[4..4 + n].to_vec(), 4 + n)))
}

struct WtSink {
    send: Mutex<SendStream>,
    session: Session,
}

impl Drop for WtSink {
    fn drop(&mut self) {
        self.session.close(SESSION_CLOSE_OK, b"worker transport dropped");
    }
}

#[async_trait]
impl WorkerSink for WtSink {
    async fn send(&self, message: WorkerMessage) -> Result<(), TransportError> {
        let bytes = message.encode_to_vec();
        if bytes.len() > MAX_CONTROL_MESSAGE {
            return Err(TransportError::Io(format!(
                "control message too large: {} bytes",
                bytes.len()
            )));
        }
        self.send.lock().await.write_all(&frame_control(&bytes)).await.map_err(transport_err)
    }

    async fn close(&self) {
        let _ = self.send.lock().await.finish();
        self.session.close(SESSION_CLOSE_OK, b"");
    }
}

struct WtStream {
    session: Session,
    recv: RecvStream,
    buf: Vec<u8>,
}

#[async_trait]
impl WorkerStream for WtStream {
    async fn recv(&mut self) -> Result<TransportEvent, TransportError> {
        loop {
            if let Some((msg, used)) = parse_control(&self.buf)? {
                self.buf.drain(..used);
                return ServerMessage::decode(msg.as_slice())
                    .map(Box::new)
                    .map(TransportEvent::ServerMessage)
                    .map_err(|e| TransportError::Protocol(e.to_string()));
            }
            let mut chunk = [0u8; 16 * 1024];
            let read = tokio::select! {
                accepted = self.session.accept_bi() => {
                    let (send, recv) = accepted.map_err(transport_err)?;
                    return Ok(TransportEvent::EndpointRequest(EndpointStream { send, recv }));
                }
                read = self.recv.read(&mut chunk) => read,
            };
            match read {
                Ok(Some(n)) => self.buf.extend_from_slice(&chunk[..n]),
                // Control stream finished or session gone: `Closed` is what the loop reconnects on.
                Ok(None) if !self.buf.is_empty() => {
                    return Err(TransportError::Protocol("truncated control message".into()));
                }
                Ok(None) => {
                    return Err(TransportError::Closed("webtransport control stream ended".into()));
                }
                Err(e) => {
                    return Err(TransportError::Closed(format!("webtransport: {e}")));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use web_transport_quinn::ServerBuilder;

    #[tokio::test]
    async fn insecure_worker_connections_are_refused_before_dialing_a_remote_peer() {
        for host in ["project.livekit.cloud", "localhost.example.com", "192.0.2.1", "[2001:db8::1]"]
        {
            let result = WtTransport { insecure: true }
                .connect(&format!("https://{host}:443"), None, vec![])
                .await;
            assert!(
                matches!(result, Err(TransportError::Io(message)) if message.contains("loopback"))
            );
        }
        for host in ["localhost", "127.0.0.1", "127.0.1.2", "[::1]"] {
            assert!(loopback_url(&format!("https://{host}").parse().unwrap()));
        }
    }

    // Catches: a length header that is not four big-endian bytes, or a partial message returned.
    #[test]
    fn control_framing_is_four_big_endian_bytes_and_whole_messages() {
        let f = frame_control(b"abc");
        assert_eq!(f, [0, 0, 0, 3, b'a', b'b', b'c']);
        assert_eq!(parse_control(&f[..5]).unwrap(), None, "a partial message is not a message");
        let two = [frame_control(b"abc"), frame_control(b"")].concat();
        let (m, used) = parse_control(&two).unwrap().unwrap();
        assert_eq!((m.as_slice(), used), (&b"abc"[..], 7));
        let (m, used) = parse_control(&two[used..]).unwrap().unwrap();
        assert_eq!((m.len(), used), (0, 4));
        let huge = [(MAX_CONTROL_MESSAGE as u32 + 1).to_be_bytes().to_vec(), vec![0; 8]].concat();
        assert!(parse_control(&huge).is_err(), "a length past the cap must fail, not allocate");
    }

    fn loopback_server() -> (web_transport_quinn::Server, u16) {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let chain = vec![cert.cert.der().clone()];
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(cert.signing_key.serialize_der().into());
        let port = std::net::UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let server = ServerBuilder::new()
            .with_addr(([127, 0, 0, 1], port).into())
            .with_certificate(chain, key)
            .unwrap();
        (server, port)
    }

    #[tokio::test]
    async fn endpoint_stop_during_stalled_io() {
        use lk_agents_abi::{Clock, timeout};
        use lk_agents_net::clock::RealClock;

        let clock = RealClock::new();
        let (mut server, port) = loopback_server();
        let peer = async {
            let session = server.accept().await.unwrap().ok().await.unwrap();
            let _control = session.accept_bi().await.unwrap();
            let (mut send, mut recv) = session.open_bi().await.unwrap();
            send.write_all(b"x").await.unwrap();
            let stopped = send.stopped().await.unwrap();
            let response = recv.read_to_end(32 << 20).await;
            (stopped, response)
        };
        let worker = async {
            let transport = WtTransport { insecure: true };
            let Connected { sink, mut stream } = transport
                .connect(&format!("https://127.0.0.1:{port}"), None, vec![])
                .await
                .unwrap();
            sink.send(WorkerMessage::default()).await.unwrap();
            let TransportEvent::EndpointRequest(request) = stream.recv().await.unwrap() else {
                panic!("expected the peer's endpoint stream");
            };
            let (mut writer, mut reader) = request.split();
            let control = writer.control();
            let mut first = [0_u8];
            assert_eq!(reader.read(&mut first).await.unwrap(), Some(1));
            let mut input = [0_u8];
            let output = vec![1_u8; 16 << 20];
            let mut pending = Box::pin(futures_util::future::join(
                reader.read(&mut input),
                writer.write_all(&output),
            ));
            assert!(futures_util::poll!(pending.as_mut()).is_pending());
            let before = clock.now();
            control.reset(HttpStreamResetCode::HsrProtocol);
            assert!(clock.now().saturating_duration_since(before) < Duration::from_millis(2));
            let (read, write) = timeout(&clock, Duration::from_secs(2), pending).await.unwrap();
            assert!(read.is_err(), "stalled read did not settle after reset");
            assert!(write.is_err(), "stalled write did not settle after reset");
            sink
        };
        let ((stopped, response), sink) =
            timeout(&clock, Duration::from_secs(5), futures_util::future::join(peer, worker))
                .await
                .unwrap();
        assert_eq!(stopped, Some(HttpStreamResetCode::HsrProtocol as u32));
        assert!(response.unwrap_err().to_string().contains('4'));
        sink.close().await;
    }

    // Catches: the preamble framing read as HTTP bytes (or HTTP bytes eaten by the preamble read),
    // `session_id` dropped, a finished response not reaching the peer as EOF, or a bad preamble
    // answered with anything but HSR_PROTOCOL.
    #[tokio::test]
    async fn an_exchange_stream_yields_its_preamble_then_opaque_bytes_and_a_bad_one_is_reset() {
        use lk_agents_abi::worker::http::ByteStream as _;
        use prost::Message as _;
        let (mut server, port) = loopback_server();
        let server_task = tokio::spawn(async move {
            let session = server.accept().await.unwrap().ok().await.unwrap();
            let _control = session.accept_bi().await.unwrap();
            // A well-formed exchange: preamble, then a request, then FIN.
            let preamble = StreamPreamble {
                kind: AgentEndpointKind::AekText as i32,
                request_id: "AER_1".into(),
                authorized: true,
                route: "/a2a".into(),
                ..Default::default()
            };
            let (mut send, mut recv) = session.open_bi().await.unwrap();
            let body = preamble.encode_to_vec();
            send.write_all(&(body.len() as u32).to_be_bytes()).await.unwrap();
            send.write_all(&body).await.unwrap();
            send.write_all(b"\0opaque request\xff").await.unwrap();
            send.finish().unwrap();
            let response = recv.read_to_end(1 << 16).await.unwrap();
            // A stream whose length prefix exceeds the cap.
            let (mut bad_send, mut bad_recv) = session.open_bi().await.unwrap();
            bad_send.write_all(&(1u32 << 30).to_be_bytes()).await.unwrap();
            let reset = bad_recv
                .read_to_end(16)
                .await
                .map(|_| String::new())
                .unwrap_or_else(|e| e.to_string());
            (response, reset)
        });

        let transport = WtTransport { insecure: true };
        let Connected { sink, mut stream } =
            transport.connect(&format!("https://127.0.0.1:{port}"), None, vec![]).await.unwrap();
        sink.send(WorkerMessage::default()).await.unwrap();
        let mut served = 0;
        while served < 2 {
            let TransportEvent::EndpointRequest(mut request) =
                tokio::time::timeout(Duration::from_secs(5), stream.recv()).await.unwrap().unwrap()
            else {
                continue;
            };
            served += 1;
            match request.read_preamble().await {
                Ok(preamble) => {
                    assert_eq!(preamble.kind, AgentEndpointKind::AekText as i32);
                    let (writer, reader) = request.split();
                    let stream = crate::http::HttpStream::new(reader, writer);
                    let mut bytes = Vec::new();
                    loop {
                        let chunk = stream.read(256).await.unwrap();
                        if chunk.is_empty() {
                            break;
                        }
                        bytes.extend_from_slice(&chunk);
                    }
                    assert_eq!(bytes, b"\0opaque request\xff");
                    stream.write(b"\xffopaque response\0".to_vec()).await.unwrap();
                    stream.finish().unwrap();
                }
                Err(_) => request.reset(HttpStreamResetCode::HsrProtocol),
            }
        }
        let (response, reset) =
            tokio::time::timeout(Duration::from_secs(5), server_task).await.unwrap().unwrap();
        assert_eq!(response, b"\xffopaque response\0");
        assert!(
            reset.contains('4'),
            "the bad preamble was not reset with HSR_PROTOCOL (4): {reset}"
        );
        sink.close().await;
    }

    // Catches: control stream not opened first, mis-framed, or a server-opened stream left dangling.
    #[tokio::test]
    async fn a_worker_dials_speaks_on_the_control_stream_and_refuses_a_server_stream() {
        let (mut server, port) = loopback_server();

        let server_task = tokio::spawn(async move {
            let request = server.accept().await.expect("a session request");
            assert_eq!(request.connect().url.path(), "/agent");
            let auth = request
                .connect()
                .headers
                .get("authorization")
                .map(|v| v.to_str().unwrap().to_owned());
            assert_eq!(auth.as_deref(), Some("Bearer t0k"));
            let session = request.ok().await.unwrap();
            // The control stream is the worker's first stream; the server echoes what it reads.
            let (mut send, mut recv) = session.accept_bi().await.unwrap();
            let mut buf = vec![0u8; 64];
            let n = recv.read(&mut buf).await.unwrap().unwrap();
            let (msg, _) = parse_control(&buf[..n]).unwrap().expect("one whole frame");
            send.write_all(&frame_control(&msg)).await.unwrap();
            // One server-opened stream, which this worker has no handler for.
            let (mut es, mut er) = session.open_bi().await.unwrap();
            es.write_all(b"GET / HTTP/1.1\r\n\r\n").await.unwrap();
            match er.read(&mut buf).await {
                Err(e) => e.to_string(),
                Ok(other) => format!("{other:?}"),
            }
        });

        let transport = WtTransport { insecure: true };
        let url = format!("https://127.0.0.1:{port}");
        let Connected { sink, mut stream } = transport
            .connect(&url, None, vec![("Authorization".into(), "Bearer t0k".into())])
            .await
            .expect("dial the loopback server");
        sink.send(WorkerMessage::default()).await.unwrap();
        let mut control_seen = false;
        let mut endpoint_seen = false;
        while !control_seen || !endpoint_seen {
            match tokio::time::timeout(Duration::from_secs(5), stream.recv())
                .await
                .unwrap()
                .unwrap()
            {
                TransportEvent::ServerMessage(message) => {
                    assert!(message.message.is_none());
                    control_seen = true;
                }
                TransportEvent::EndpointRequest(request) => {
                    request.refuse();
                    endpoint_seen = true;
                }
            }
        }
        let refused =
            tokio::time::timeout(Duration::from_secs(5), server_task).await.unwrap().unwrap();
        assert!(
            refused.contains("(1)") || refused.contains("code 1") || refused.ends_with('1'),
            "the server did not see the stream refused with HSR_REFUSED (1): {refused}"
        );
        sink.close().await;
    }
}
