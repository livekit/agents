#![allow(clippy::disallowed_types, clippy::disallowed_methods, reason = "migrating to signal")]
pub mod exchange;

/// The HTTP vocabulary of [`GatewayHttp`]'s callers.
pub use reqwest::{Method, StatusCode, header::HeaderMap};

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::task::{Context, Poll};

use crate::transport::{Headers, HttpRequest};
use futures_util::StreamExt;
use lk_agents_abi::{Clock, Ts};
use lk_agents_core::request::AttemptFailure;
use tokio::sync::{Notify, mpsc};

/// `bytes` as one gzip member (RFC 1952), for a `Content-Encoding: gzip` body or a `.gz` file.
/// The fastest deflate level: every body is small and the network dominates. Compressing in
/// memory cannot fail, so neither can this.
pub fn gzip(bytes: &[u8]) -> Vec<u8> {
    let mut crc = flate2::Crc::new();
    crc.update(bytes);
    // ID1 ID2, deflate, no flags, no modification time, no extra flags, unknown OS.
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
    out.extend(miniz_oxide::deflate::compress_to_vec(bytes, 1));
    out.extend(crc.sum().to_le_bytes());
    // ISIZE: the input length modulo 2^32.
    out.extend(&bytes.len().to_le_bytes()[..4]);
    out
}

/// The host's `APIError` helpers: an HTTP status's message as the native client words it.
pub struct HttpErrors;

impl lk_agents_abi::errors::ApiErrorFunctions for HttpErrors {
    fn http_error_message(message: &str, status: i64, reason: Option<String>) -> String {
        lk_agents_abi::errors::http_error_message(message, status, reason.as_deref())
    }

    fn retryable_for_status(status_code: i64, retryable: Option<bool>) -> bool {
        lk_agents_abi::errors::retryable_for_status(status_code, retryable)
    }
}

/// A transport observation and the instant it was made.
#[derive(Debug)]
pub struct Stamped<E> {
    pub at: Ts,
    pub event: E,
}

/// Where an attempt's observations go: its request's bounded queue, so an attempt reads the
/// response no faster than the request consumes it.
pub struct Reporter<E> {
    pub tx: mpsc::Sender<Stamped<E>>,
    pub clock: Arc<dyn Clock>,
}

impl<E> Clone for Reporter<E> {
    fn clone(&self) -> Self {
        Reporter { tx: self.tx.clone(), clock: self.clock.clone() }
    }
}

impl<E> Reporter<E> {
    pub async fn report(&self, event: E) {
        // A closed queue is a request that stopped listening; nothing is owed to it.
        let _ = self.tx.send(Stamped { at: self.clock.now(), event }).await;
    }
}

/// How far the transport took the request before its response.
#[derive(Debug, Clone, Copy)]
pub enum Sent {
    Connected,
    RequestWritten,
}

/// Quota telemetry headers the gateway stamps on LLM completions responses (including 429
/// rejections). Absent means "not enforced / unknown", never zero.
pub const QUOTA_HEADER_FIELDS: [(&str, &str); 6] = [
    ("X-LiveKit-Inference-RPM-Limit", "rpm_limit"),
    ("X-LiveKit-Inference-RPM-Used", "rpm_used"),
    ("X-LiveKit-Inference-TPM-Limit", "tpm_limit"),
    ("X-LiveKit-Inference-TPM-Used", "tpm_used"),
    ("X-LiveKit-Inference-Credits-Limit", "credits_limit"),
    ("X-LiveKit-Inference-Credits-Used", "credits_used"),
];

/// `lookup` should be case-insensitive, like the reference's header maps.
pub fn extract_quota_usage<'a>(
    lookup: impl Fn(&str) -> Option<&'a str>,
) -> Vec<(&'static str, String)> {
    QUOTA_HEADER_FIELDS
        .iter()
        .filter_map(|(header, field)| {
            // An empty header value is dropped, not reported as "".
            match lookup(header) {
                Some(v) if !v.is_empty() => Some((*field, v.to_owned())),
                _ => None,
            }
        })
        .collect()
}

/// The response's text when the gateway rejects the request is read up to this many bytes.
const ERROR_BODY_LIMIT: usize = 64 * 1024;

/// One client for the process: keep-alive connections are shared by every request. No reqwest
/// timeout is set: the request core times each limit.
pub struct GatewayHttp {
    client: reqwest::Client,
}

impl GatewayHttp {
    /// The same connection pool and TLS configuration for typed SDK clients.
    pub fn client(&self) -> reqwest::Client {
        self.client.clone()
    }

    pub fn new() -> Result<Self, crate::tls::TlsError> {
        let builder = crate::tls::http_client_builder()?
            .retry(reqwest::retry::never())
            .pool_max_idle_per_host(50)
            .pool_idle_timeout(std::time::Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() > 10 {
                    attempt.error("too many HTTP redirects")
                } else if attempt
                    .previous()
                    .first()
                    .is_some_and(|first| first.origin() != attempt.url().origin())
                {
                    // A 307/308 could forward private caller history to a card-selected peer.
                    attempt.error("HTTP redirects must preserve the request origin")
                } else {
                    attempt.follow()
                }
            }));
        Ok(GatewayHttp { client: crate::tls::build(builder)? })
    }

    /// A GET whose response is read to its end, so its connection stays pooled (`prewarm()`).
    pub async fn warm(&self, url: String, headers: Headers) {
        let mut builder = self.client.get(url);
        for (name, value) in headers {
            builder = builder.header(name, value);
        }
        let result = async {
            let response = builder
                .send()
                .await
                .map_err(|error| AttemptFailure::Connection(error_chain(&error)))?;
            if !response.status().is_success() {
                return Err(rejected(response).await);
            }
            read_body(response, ERROR_BODY_LIMIT).await.map(|_| ())
        }
        .await;
        if let Err(error) = result {
            log::warn!(target: lk_agents_abi::LOG_TARGET, "HTTP prewarm failed: {error:?}");
        }
    }

    /// Sends `request`, reporting through `sent` how far it got; the response once its status is
    /// a success.
    pub async fn send<E>(
        &self,
        request: HttpRequest,
        reporter: &Reporter<E>,
        sent: impl Fn(Sent) -> E,
    ) -> Result<reqwest::Response, AttemptFailure> {
        self.send_method(reqwest::Method::POST, request, reporter, sent).await
    }

    /// GET discovery and POST RPC share connection/write observations and the request clock.
    pub async fn send_method<E>(
        &self,
        method: reqwest::Method,
        request: HttpRequest,
        reporter: &Reporter<E>,
        sent: impl Fn(Sent) -> E,
    ) -> Result<reqwest::Response, AttemptFailure> {
        let response = self.send_raw(method, request, reporter, sent).await?;
        if response.status().is_success() { Ok(response) } else { Err(rejected(response).await) }
    }

    /// Protocols such as MCP interpret non-success responses themselves, including authentication
    /// challenges, session expiry and JSON-RPC errors returned with an HTTP error status.
    pub async fn send_raw<E>(
        &self,
        method: reqwest::Method,
        request: HttpRequest,
        reporter: &Reporter<E>,
        sent: impl Fn(Sent) -> E,
    ) -> Result<reqwest::Response, AttemptFailure> {
        // Hyper otherwise suppresses an unknown-length GET body without polling it. Explicit
        // empty chunked framing preserves its real connection/write observations without a payload.
        let empty_get = method == reqwest::Method::GET && request.body.is_empty();
        let progress = Arc::new(BodyProgress::default());
        let body = ReportingBody { bytes: Some(request.body.into()), progress: progress.clone() };
        let mut builder =
            self.client.request(method, request.url).body(reqwest::Body::wrap_stream(body));
        if empty_get {
            builder = builder.header(reqwest::header::TRANSFER_ENCODING, "chunked");
        }
        for (name, value) in request.headers {
            builder = builder.header(name, value);
        }
        let mut send = std::pin::pin!(builder.send());
        let mut reported = 0;
        let response = loop {
            tokio::select! {
                biased;
                response = &mut send => break response,
                () = progress.changed.notified() => progress.report(&mut reported, reporter, &sent).await,
            }
        };
        progress.report(&mut reported, reporter, &sent).await;
        response.map_err(|err| AttemptFailure::Connection(error_chain(&err)))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("the answer exceeds {limit} bytes")]
pub(crate) struct BodyLimitExceeded {
    limit: usize,
}

/// A bounded response body, shared by streamed exchanges and direct HTTP reads.
pub(crate) struct BodyReader {
    bytes: Vec<u8>,
    limit: usize,
}

impl BodyReader {
    pub(crate) fn new(limit: usize) -> Self {
        Self { bytes: Vec::new(), limit }
    }

    pub(crate) fn append(&mut self, bytes: &[u8]) -> Result<(), BodyLimitExceeded> {
        if bytes.len() > self.limit - self.bytes.len() {
            return Err(BodyLimitExceeded { limit: self.limit });
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.bytes.clear();
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// A JSON answer's body, read to its end; one larger than `limit` bytes is malformed.
pub async fn read_body(
    response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, AttemptFailure> {
    let mut body = BodyReader::new(limit);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| AttemptFailure::Connection(error_chain(&err)))?;
        body.append(&chunk).map_err(|error| AttemptFailure::Malformed(error.to_string()))?;
    }
    Ok(body.finish())
}

async fn rejected(response: reqwest::Response) -> AttemptFailure {
    let status = i64::from(response.status().as_u16());
    let header =
        |name: &str| response.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let request_id = header("X-LiveKit-Request-Id");
    let retryable = match header("X-Should-Retry").as_deref() {
        Some("false") => Some(false),
        Some("true") => Some(true),
        _ => None,
    };
    if status == 429 {
        let quota =
            extract_quota_usage(|name| response.headers().get(name).and_then(|v| v.to_str().ok()));
        let quota: Vec<String> = quota.iter().map(|(k, v)| format!("{k}={v}")).collect();
        log::warn!(
            target: lk_agents_abi::LOG_TARGET,
            request_id = request_id.as_deref().unwrap_or(""), quota = quota.join(" ").as_str();
            "request rate limited by inference gateway"
        );
    }
    let message = match read_body(response, ERROR_BODY_LIMIT).await {
        Ok(body) => error_message(&String::from_utf8_lossy(&body)).unwrap_or_default(),
        Err(error) => format!("cannot read HTTP error response: {error:?}"),
    };
    AttemptFailure::Rejected { status, message, request_id, retryable }
}

/// The gateway's flat error, OpenAI's nested message, or the trimmed text itself
/// (a plain `404 page not found` must survive).
pub(crate) fn error_message(body: &str) -> Option<String> {
    let text = || Some(body.trim().chars().take(512).collect::<String>()).filter(|t| !t.is_empty());
    match serde_json::from_str::<ErrorBody>(body) {
        Ok(ErrorBody::OpenAi { error }) => Some(error.message),
        Ok(ErrorBody::Gateway { error, hint }) => Some(match hint.filter(|h| !h.is_empty()) {
            Some(hint) => format!("{error} {hint}"),
            None => error,
        }),
        Ok(ErrorBody::Message { message }) => Some(message),
        Err(_) => text(),
    }
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum ErrorBody {
    OpenAi {
        error: ErrorMessage,
    },
    Gateway {
        error: String,
        #[serde(default)]
        hint: Option<String>,
    },
    Message {
        message: String,
    },
}

#[derive(serde::Deserialize)]
struct ErrorMessage {
    message: String,
}

pub fn error_chain(err: &dyn std::error::Error) -> String {
    let mut message = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

/// How far the transport has taken the request body: hyper polls it only on a ready
/// connection, and polls it to its end once every byte is handed to the connection.
#[derive(Default)]
struct BodyProgress {
    stage: AtomicU8,
    changed: Notify,
}

const CONNECTED: u8 = 1;
const WRITTEN: u8 = 2;

impl BodyProgress {
    fn reach(&self, stage: u8) {
        self.stage.fetch_max(stage, Ordering::SeqCst);
        self.changed.notify_one();
    }

    async fn report<E>(
        &self,
        reported: &mut u8,
        reporter: &Reporter<E>,
        sent: &impl Fn(Sent) -> E,
    ) {
        let stage = self.stage.load(Ordering::SeqCst);
        if *reported < CONNECTED && stage >= CONNECTED {
            reporter.report(sent(Sent::Connected)).await;
        }
        if *reported < WRITTEN && stage >= WRITTEN {
            reporter.report(sent(Sent::RequestWritten)).await;
        }
        *reported = (*reported).max(stage);
    }
}

struct ReportingBody {
    bytes: Option<bytes::Bytes>,
    progress: Arc<BodyProgress>,
}

impl futures_util::Stream for ReportingBody {
    type Item = Result<bytes::Bytes, std::io::Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match self.bytes.take() {
            Some(bytes) => {
                self.progress.reach(CONNECTED);
                Poll::Ready(Some(Ok(bytes)))
            }
            None => {
                self.progress.reach(WRITTEN);
                Poll::Ready(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Catches: a gzip member a standard decoder rejects (header, CRC or length trailer).
    #[test]
    fn gzip_is_a_standard_member_any_decoder_reads() {
        use std::io::Read as _;
        for input in [&b""[..], b"hello", &[7u8; 100_000][..]] {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(&gzip(input)[..]).read_to_end(&mut out).unwrap();
            assert_eq!(out, input);
        }
    }

    #[test]
    fn body_limit_counts_all_chunks_and_restarts_without_old_bytes() {
        let mut body = BodyReader::new(4);
        body.append(b"ab").unwrap();
        body.append(b"cd").unwrap();
        assert!(body.append(b"e").is_err());
        assert_eq!(body.as_bytes(), b"abcd");
        body.clear();
        body.append(b"efgh").unwrap();
        assert_eq!(body.finish(), b"efgh");
    }

    #[tokio::test]
    async fn rejected_responses_keep_the_status_and_explain_incomplete_or_oversized_bodies() {
        for oversized in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let peer = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    request.push(socket.read_u8().await.unwrap());
                }
                socket.write_all(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 65537\r\nX-LiveKit-Request-Id: test-id\r\nConnection: close\r\n\r\n").await.unwrap();
                if oversized {
                    socket.write_all(&vec![b'x'; ERROR_BODY_LIMIT + 1]).await.unwrap();
                } else {
                    socket.write_all(b"partial").await.unwrap();
                }
            });
            let response = GatewayHttp::new()
                .unwrap()
                .client()
                .get(format!("http://{addr}"))
                .send()
                .await
                .unwrap();
            let AttemptFailure::Rejected { status, message, request_id, .. } =
                rejected(response).await
            else {
                panic!("must preserve the rejected status");
            };
            assert_eq!(status, 503);
            assert_eq!(request_id.as_deref(), Some("test-id"));
            assert!(message.contains("cannot read HTTP error response"));
            assert!(message.contains(if oversized { "exceeds" } else { "Connection" }));
            peer.await.unwrap();
        }
    }
}
