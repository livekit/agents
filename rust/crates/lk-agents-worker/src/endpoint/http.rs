//! One HTTP/1.1 exchange: request bytes in, ASGI `http` events to the app, response bytes out.

use lk_agents_abi::SpawnerExt;
use lk_agents_net::task::TokioSpawner;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{Mutex, watch};

use super::admission::EndpointReservation;
use super::request::{Head, find, scope_base};
use super::{
    AsgiApp, Conn, EndpointLimits, OnCancel, READ_SIZE, Received, RequestInfo, Reset, Sent,
    StreamIo, chunk, kind_of, response_head, sanitize, valid_header,
};
use crate::LOG;

enum Body {
    Length(u64),
    Chunked(ChunkState),
    /// HTTP/1.0 without a length: none.
    Done,
}

#[derive(Clone, Copy)]
enum ChunkState {
    Size,
    Data(u64),
    DataEnd,
    Trailers,
}

struct RequestSide {
    buf: Vec<u8>,
    body: Body,
    eof: bool,
    done: bool,
}

#[derive(Default)]
struct ResponseSide {
    started: bool,
    ended: bool,
    chunked: bool,
    /// Declared `content-length` left to send.
    remaining: Option<u64>,
}

/// One HTTP exchange: request bytes in, ASGI `http` events to the app, response bytes out.
pub struct HttpExchange {
    _payload: EndpointReservation,
    limits: EndpointLimits,
    io: Arc<dyn StreamIo>,
    is_head: bool,
    http10: bool,
    req: Mutex<RequestSide>,
    resp: Mutex<ResponseSide>,
    head_sent: watch::Sender<bool>,
    done: watch::Sender<bool>,
    peer_gone: AtomicBool,
}

impl HttpExchange {
    fn new(
        io: Arc<dyn StreamIo>,
        head: &Head,
        payload: EndpointReservation,
        limits: EndpointLimits,
    ) -> Result<Self, String> {
        let te = head.tokens(b"transfer-encoding");
        let body = if !te.is_empty() {
            if te.last().map(Vec::as_slice) != Some(b"chunked") {
                return Err("Transfer-Encoding must end with chunked".into());
            }
            Body::Chunked(ChunkState::Size)
        } else {
            let lengths: Vec<&[u8]> = head.values(b"content-length").collect();
            match lengths.first() {
                None => Body::Done,
                Some(first) => {
                    if lengths.iter().any(|l| l != first) {
                        return Err("conflicting Content-Length headers".into());
                    }
                    let s = std::str::from_utf8(first).map_err(|e| e.to_string())?.trim();
                    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
                        return Err("bad Content-Length".into());
                    }
                    Body::Length(s.parse().map_err(|_| "bad Content-Length".to_owned())?)
                }
            }
        };
        let body = match body {
            Body::Length(0) => Body::Done,
            b => b,
        };
        Ok(HttpExchange {
            _payload: payload,
            limits,
            io,
            is_head: head.method == "HEAD",
            http10: head.http10,
            req: Mutex::new(RequestSide { buf: head.rest.clone(), body, eof: false, done: false }),
            resp: Mutex::new(ResponseSide::default()),
            head_sent: watch::channel(false).0,
            done: watch::channel(false).0,
            peer_gone: AtomicBool::new(false),
        })
    }

    pub async fn receive(&self) -> Received {
        let mut r = self.req.lock().await;
        if r.done {
            drop(r);
            // The node half-closes once the request is written; that is the end of the body, not
            // a disconnect. A client that goes away shows as a failed write, so an app listening
            // for a disconnect waits here until the exchange ends.
            let mut done = self.done.subscribe();
            let _ = done.wait_for(|d| *d).await;
            return Received::Disconnect;
        }
        match self.next_body(&mut r).await {
            Ok(Some(body)) => Received::Request { body, more_body: true },
            Ok(None) => {
                r.done = true;
                Received::Request { body: Vec::new(), more_body: false }
            }
            Err(()) => {
                r.done = true;
                self.done.send_replace(true);
                self.io.reset(Reset::HsrProtocol).await;
                Received::Disconnect
            }
        }
    }

    /// The next piece of the body, `None` at its end, `Err` when the stream ended early or broke.
    async fn next_body(&self, r: &mut RequestSide) -> Result<Option<Vec<u8>>, ()> {
        loop {
            match &mut r.body {
                Body::Done => return Ok(None),
                Body::Length(left) => {
                    if !r.buf.is_empty() {
                        let n = (*left).min(r.buf.len() as u64) as usize;
                        let data: Vec<u8> = r.buf.drain(..n).collect();
                        *left -= n as u64;
                        if *left == 0 {
                            r.body = Body::Done;
                        }
                        return Ok(Some(data));
                    }
                }
                Body::Chunked(state) => match *state {
                    ChunkState::Size => {
                        let end = find(&r.buf, b"\r\n");
                        if end.map_or(r.buf.len() >= self.limits.chunk_line_bytes, |i| {
                            i + 2 > self.limits.chunk_line_bytes
                        }) {
                            return Err(());
                        }
                        if let Some(i) = end {
                            let line: Vec<u8> = r.buf.drain(..i + 2).collect();
                            let size = line[..i].split(|b| *b == b';').next().unwrap_or_default();
                            let size = std::str::from_utf8(size).map_err(|_| ())?.trim();
                            let size = u64::from_str_radix(size, 16).map_err(|_| ())?;
                            *state = if size == 0 {
                                ChunkState::Trailers
                            } else {
                                ChunkState::Data(size)
                            };
                            continue;
                        }
                    }
                    ChunkState::Data(left) => {
                        if !r.buf.is_empty() {
                            let n = left.min(r.buf.len() as u64) as usize;
                            let data: Vec<u8> = r.buf.drain(..n).collect();
                            *state = if left == n as u64 {
                                ChunkState::DataEnd
                            } else {
                                ChunkState::Data(left - n as u64)
                            };
                            return Ok(Some(data));
                        }
                    }
                    ChunkState::DataEnd => {
                        if r.buf.len() >= 2 {
                            if &r.buf[..2] != b"\r\n" {
                                return Err(());
                            }
                            r.buf.drain(..2);
                            *state = ChunkState::Size;
                            continue;
                        }
                    }
                    ChunkState::Trailers => {
                        if r.buf.starts_with(b"\r\n") {
                            r.buf.drain(..2);
                            r.body = Body::Done;
                            continue;
                        }
                        let end = find(&r.buf, b"\r\n\r\n");
                        if end.map_or(r.buf.len() >= self.limits.trailers_bytes, |i| {
                            i + 4 > self.limits.trailers_bytes
                        }) {
                            return Err(());
                        }
                        if let Some(i) = end {
                            r.buf.drain(..i + 4);
                            r.body = Body::Done;
                            continue;
                        }
                    }
                },
            }
            if r.eof {
                return Err(());
            }
            let chunk = self.io.read(READ_SIZE).await.map_err(|_| ())?;
            if chunk.is_empty() {
                r.eof = true;
            }
            r.buf.extend_from_slice(&chunk);
        }
    }

    async fn write(&self, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        self.io.write(data).await.inspect_err(|_| {
            // the peer reset the stream or stopped reading: the client is gone
            self.peer_gone.store(true, Ordering::Release);
        })
    }

    pub async fn send(&self, msg: Sent) -> Result<(), String> {
        if *self.done.borrow() {
            return Err("endpoint exchange is closed".into());
        }
        match msg {
            Sent::ResponseStart { status, mut headers } => {
                let mut resp = self.resp.lock().await;
                if resp.started {
                    return Err("response already started".into());
                }
                if let Some((k, v)) = headers.iter().find(|(k, v)| !valid_header(k, v)) {
                    return Err(format!(
                        "illegal header: {:?}: {:?}",
                        String::from_utf8_lossy(k),
                        String::from_utf8_lossy(v)
                    ));
                }
                let has = |name: &[u8]| headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
                let length = headers
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(b"content-length"))
                    .and_then(|(_, v)| std::str::from_utf8(v).ok()?.trim().parse::<u64>().ok());
                let bodiless = matches!(status, 204 | 304) || self.is_head;
                // h11 frames an HTTP/1.1 response without a length as chunked, the one framing
                // that can carry a completion trailer, so a dying stream reads as truncation
                let unframed = !has(b"content-length") && !has(b"transfer-encoding");
                if unframed && !matches!(status, 204 | 304) {
                    if self.http10 {
                        headers.push((b"connection".to_vec(), b"close".to_vec()));
                    } else {
                        headers.push((b"transfer-encoding".to_vec(), b"chunked".to_vec()));
                    }
                }
                resp.chunked = unframed && !self.http10 && !bodiless;
                resp.remaining = if bodiless { None } else { length };
                resp.started = true;
                self.head_sent.send_replace(true);
                self.write(&response_head(status, &headers)).await
            }
            Sent::ResponseBody { body, more_body } => {
                let mut resp = self.resp.lock().await;
                if !resp.started {
                    return Err("response body before its head".into());
                }
                if resp.ended {
                    return Ok(());
                }
                if !body.is_empty() && !self.is_head {
                    if let Some(left) = resp.remaining {
                        if body.len() as u64 > left {
                            return Err("Too much data for declared Content-Length".into());
                        }
                        resp.remaining = Some(left - body.len() as u64);
                    }
                    let data = if resp.chunked { chunk(&body) } else { body };
                    self.write(&data).await?;
                }
                if !more_body {
                    if resp.remaining.is_some_and(|l| l > 0) {
                        return Err("Too little data for declared Content-Length".into());
                    }
                    if resp.chunked {
                        self.write(b"0\r\n\r\n").await?;
                    }
                    resp.ended = true;
                    self.io.finish().await;
                    // nothing more of the request is wanted
                    self.io.stop(Reset::HsrAbort).await;
                    self.done.send_replace(true);
                }
                Ok(())
            }
            // the x-lk- namespace is ours; applications do not send trailers here
            Sent::ResponseTrailers => Ok(()),
            other => Err(format!("unexpected ASGI message {:?} on an http scope", kind_of(&other))),
        }
    }

    /// Report an outcome other than success in the only sound place for it.
    async fn fail(&self, completion: &str, reason: &str, code: Reset) {
        if *self.done.borrow() {
            return;
        }
        let mut resp = self.resp.lock().await;
        if resp.ended {
            return;
        }
        if !resp.started {
            self.io.reset(code).await;
            return;
        }
        if resp.chunked {
            // in band, where it cannot race the body bytes
            let trailers = format!(
                "0\r\nx-lk-completion: {completion}\r\nx-lk-error: {}\r\n\r\n",
                sanitize(reason)
            );
            if self.write(trailers.as_bytes()).await.is_ok() {
                self.io.finish().await;
            }
            resp.ended = true;
            return;
        }
        // a declared length cannot carry trailers: the short body is the signal
        self.io.reset(Reset::HsrAbort).await;
    }
}

pub(super) async fn serve_http(
    io: Arc<dyn StreamIo>,
    info: RequestInfo,
    head: Head,
    app: &dyn AsgiApp,
    payload: EndpointReservation,
    limits: EndpointLimits,
) {
    let ex = match HttpExchange::new(io.clone(), &head, payload, limits) {
        Ok(ex) => Arc::new(ex),
        Err(e) => {
            log::warn!(target: LOG, requestId = info.request_id.as_str(), error = e.as_str();
                "endpoint request is not valid HTTP/1.1");
            io.reset(Reset::HsrProtocol).await;
            return;
        }
    };
    let mut scope = scope_base(&info, "http", head.headers.clone(), &head.target);
    scope.method = head.method.clone();
    scope.scheme = if info.scheme.is_empty() { "http".into() } else { info.scheme.clone() };

    // the worker is stopping or reconnecting: the stream goes with it
    let guard = {
        let ex = ex.clone();
        OnCancel(Some(Box::new(move || {
            // Detached: the exchange's owner is being dropped, and the reset must still go out.
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                TokioSpawner::new(rt).spawn_detached(async move {
                    ex.fail("canceled", "worker stopped", Reset::HsrAbort).await
                });
            }
        })))
    };
    let mut call = app.call(scope, Conn::Http(ex.clone()));
    let result = match (info.timeout_ms > 0).then(|| Duration::from_millis(info.timeout_ms)) {
        None => Some(call.await),
        // the deadline bounds the wait for a response head only; bodies are unbounded
        Some(timeout) => {
            let mut head_sent = ex.head_sent.subscribe();
            let first = tokio::select! {
                r = &mut call => Some(Some(r)),
                _ = head_sent.wait_for(|s| *s) => Some(None),
                () = tokio::time::sleep(timeout) => None,
            };
            match first {
                Some(Some(r)) => Some(r),
                Some(None) => Some(call.await),
                None => {
                    drop(call);
                    None
                }
            }
        }
    };
    guard.disarm();
    let rid = info.request_id.as_str();
    let route = info.route.as_str();
    match result {
        None => {
            log::warn!(target: LOG, requestId = rid, route = route;
                "endpoint request timed out before a response head");
            ex.fail("timeout", "no response head before the deadline", Reset::HsrTimeout).await;
        }
        Some(Err(_)) if ex.peer_gone.load(Ordering::Acquire) => {
            log::debug!(target: LOG, requestId = rid, route = route; "endpoint client went away");
        }
        Some(Err(e)) => {
            log::error!(target: LOG, requestId = rid, route = route;
                "endpoint request failed\n{}{}", e.traceback, e.summary);
            ex.fail("internal", &e.summary, Reset::HsrInternal).await;
        }
        Some(Ok(())) => {
            let (started, ended) = {
                let r = ex.resp.lock().await;
                (r.started, r.ended)
            };
            if !started {
                log::error!(target: LOG, requestId = rid, route = route;
                    "ASGI app returned without a response");
                ex.fail("internal", "no response", Reset::HsrInternal).await;
            } else if !ended {
                ex.fail("internal", "response body left open", Reset::HsrInternal).await;
            }
        }
    }
    ex.done.send_replace(true);
}
