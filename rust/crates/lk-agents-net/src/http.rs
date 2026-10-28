//! HTTPS exchanges: each attempt sends its request and reports what the transport observed
//! (connection, request handed over, response bytes, decoded answer, end or failure) and decides
//! nothing. The request's core owns every deadline and retry.

pub mod exchange;

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::task::{Context, Poll};

use crate::attempts::AttemptFailure;
use crate::transport::HttpRequest;
use futures_util::StreamExt;
use lk_agents_abi::{Clock, Ts};
use tokio::sync::{Notify, mpsc};

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
/// timeout is set: the request core times each phase.
pub struct GatewayHttp {
    client: reqwest::Client,
}

impl GatewayHttp {
    /// The process's client, built on first use.
    pub fn shared() -> Result<Arc<GatewayHttp>, String> {
        static SHARED: std::sync::OnceLock<Result<Arc<GatewayHttp>, String>> =
            std::sync::OnceLock::new();
        SHARED.get_or_init(|| GatewayHttp::new().map(Arc::new)).clone()
    }

    pub fn new() -> Result<Self, String> {
        crate::transport::install_tls_provider();
        let client = reqwest::Client::builder()
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
            }))
            .build()
            .map_err(|e| error_chain(&e))?;
        Ok(GatewayHttp { client })
    }

    /// A GET whose response is read to its end, so its connection stays pooled (`prewarm()`).
    pub async fn warm(&self, url: String, headers: Vec<(String, String)>) {
        let mut builder = self.client.get(url);
        for (name, value) in headers {
            builder = builder.header(name, value);
        }
        if let Ok(response) = builder.send().await {
            let _ = response.bytes().await;
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

/// A JSON answer's body, read to its end; one larger than `limit` bytes is malformed.
pub async fn read_body(
    response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, AttemptFailure> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| AttemptFailure::Connection(error_chain(&err)))?;
        if body.len() + chunk.len() > limit {
            return Err(AttemptFailure::Malformed(format!("the answer exceeds {limit} bytes")));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

async fn rejected(response: reqwest::Response) -> AttemptFailure {
    let status = i64::from(response.status().as_u16());
    let header =
        |name: &str| response.headers().get(name).and_then(|v| v.to_str().ok()).map(str::to_owned);
    let request_id = header("x-request-id");
    if status == 429 {
        let quota =
            extract_quota_usage(|name| response.headers().get(name).and_then(|v| v.to_str().ok()));
        let quota: Vec<String> = quota.iter().map(|(k, v)| format!("{k}={v}")).collect();
        log::warn!(
            target: "livekit.agents",
            request_id = request_id.as_deref().unwrap_or(""), quota = quota.join(" ").as_str();
            "request rate limited by inference gateway"
        );
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(Ok(chunk)) = stream.next().await {
        body.extend_from_slice(&chunk[..chunk.len().min(ERROR_BODY_LIMIT - body.len())]);
        if body.len() >= ERROR_BODY_LIMIT {
            break;
        }
    }
    let body = String::from_utf8_lossy(&body);
    let message = error_message(&body).unwrap_or_default();
    AttemptFailure::Rejected { status, message, request_id }
}

/// OpenAI's `{"error": {"message": ..}}`, a top-level `message`, or the trimmed text itself
/// (a plain `404 page not found` must survive).
fn error_message(body: &str) -> Option<String> {
    let text = || Some(body.trim().chars().take(512).collect::<String>()).filter(|t| !t.is_empty());
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(body) else { return text() };
    let message =
        parsed.get("error").and_then(|e| e.get("message")).or_else(|| parsed.get("message"));
    message.and_then(|m| m.as_str()).map(str::to_owned).or_else(text)
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
