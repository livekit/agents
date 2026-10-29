//! A WebSocket after its upgrade request: frames in, ASGI `websocket` events to the app,
//! frames out.

use lk_agents_abi::SpawnerExt;
use lk_agents_net::task::TokioSpawner;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lk_agents_net::websocket::tungstenite::{
    self, Message,
    protocol::{CloseFrame, Role, WebSocket, WebSocketConfig, frame::coding::CloseCode},
};
use tokio::sync::{Mutex, mpsc, watch};

use super::admission::{
    EndpointAdmission, EndpointReservation, MessageAdmission, MessageReservation, Registry,
};
use super::request::{Head, scope_base};
use super::{
    AsgiApp, CLOSE_FLUSH_TIMEOUT, Conn, Headers, OnCancel, READ_SIZE, Received, RequestInfo, Reset,
    Sent, StreamIo, kind_of, response_head,
};
use crate::LOG;

/// tungstenite over memory: frames are fed in and taken out; the stream I/O is ours.
#[derive(Default)]
struct Pipe {
    inbound: VecDeque<u8>,
    outbound: Vec<u8>,
}

impl Read for Pipe {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.inbound.is_empty() {
            return Err(std::io::ErrorKind::WouldBlock.into());
        }
        let n = buf.len().min(self.inbound.len());
        for (dst, src) in buf.iter_mut().zip(self.inbound.drain(..n)) {
            *dst = src;
        }
        Ok(n)
    }
}

impl Write for Pipe {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.outbound.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// After the upgrade request the stream is a byte pipe: frames in, ASGI `websocket` events to the
/// app, frames out. The `101` is written here.
pub struct WsExchange {
    _payload: EndpointReservation,
    _parser: EndpointReservation,
    io: Arc<dyn StreamIo>,
    key: Vec<u8>,
    subprotocols: Vec<String>,
    ws: std::sync::Mutex<WebSocket<Pipe>>,

    /// Holds the order of writes: bytes are taken from the pipe and written under it.
    write_lock: Mutex<()>,
    events: mpsc::Sender<InboxEvent>,
    inbox: Mutex<mpsc::Receiver<InboxEvent>>,
    admission: EndpointAdmission,
    inbox_admission: MessageAdmission,
    write_admission: MessageAdmission,
    pub(super) disconnect: watch::Sender<Option<u16>>,
    accept_ready: watch::Sender<bool>,
    accepted: AtomicBool,
    closed: AtomicBool,
}

struct InboxEvent {
    event: Received,
    _payload: Option<MessageReservation>,
}

impl WsExchange {
    fn ws(&self) -> std::sync::MutexGuard<'_, WebSocket<Pipe>> {
        self.ws.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub async fn receive(&self) -> Received {
        let mut inbox = self.inbox.lock().await;
        if let Ok(event) = inbox.try_recv() {
            return event.event;
        }
        let mut disconnect = self.disconnect.subscribe();
        if let Some(code) = *disconnect.borrow() {
            return Received::WsDisconnect { code };
        }
        let next = {
            let event = Box::pin(inbox.recv());
            let closed = Box::pin(disconnect.changed());
            match futures_util::future::select(event, closed).await {
                futures_util::future::Either::Left((event, closed)) => {
                    drop(closed);
                    event
                }
                futures_util::future::Either::Right((_, event)) => {
                    drop(event);
                    None
                }
            }
        };
        next.map(|event| event.event)
            .unwrap_or(Received::WsDisconnect { code: (*disconnect.borrow()).unwrap_or(1006) })
    }

    async fn write_raw(&self, data: &[u8]) -> Result<(), String> {
        let _w = self.write_lock.lock().await;
        if data.is_empty() { Ok(()) } else { self.io.write(data).await }
    }

    /// Whatever tungstenite queued (a frame, a pong, a close reply), in order.
    async fn flush(&self) -> Result<(), String> {
        let _w = self.write_lock.lock().await;
        let out = {
            let mut ws = self.ws();
            let _ = ws.flush();
            std::mem::take(&mut ws.get_mut().outbound)
        };
        if out.is_empty() { Ok(()) } else { self.io.write(&out).await }
    }

    async fn send_frame(&self, msg: Message) -> Result<(), String> {
        if msg.len() > self.admission.limits.websocket_message_bytes {
            return Err("websocket response exceeds message limit".into());
        }
        let _payload = self
            .write_admission
            .try_message(&self.admission, msg.len().max(READ_SIZE))
            .ok_or("websocket response capacity exhausted")?;
        {
            let _w = self.write_lock.lock().await;
            let out = {
                let mut ws = self.ws();
                ws.send(msg).map_err(|e| e.to_string())?;
                std::mem::take(&mut ws.get_mut().outbound)
            };
            if !out.is_empty() {
                self.io.write(&out).await?;
            }
        }
        Ok(())
    }

    fn enqueue(&self, message: Message) -> Result<(), &'static str> {
        let bytes = message.len();
        let payload = self
            .inbox_admission
            .try_message(&self.admission, bytes)
            .ok_or("websocket inbox byte limit exceeded")?;
        let event = match message {
            Message::Text(text) => Received::WsText(text.to_string()),
            Message::Binary(bytes) => Received::WsBytes(bytes.to_vec()),
            _ => unreachable!("only data messages enter the inbox"),
        };
        self.events
            .try_send(InboxEvent { event, _payload: Some(payload) })
            .map_err(|_| "websocket inbox item limit exceeded")
    }

    /// Frames from the peer, once accepted. `false` when the peer closed or broke the protocol.
    async fn process(&self) -> bool {
        if !self.accepted.load(Ordering::Acquire) {
            return true;
        }
        let mut parsed = 0;
        loop {
            parsed += 1;
            if parsed == 16 {
                tokio::task::yield_now().await;
                parsed = 0;
            }
            let next = self.ws().read();
            match next {
                Ok(message @ (Message::Text(_) | Message::Binary(_))) => {
                    if let Err(reason) = self.enqueue(message) {
                        self.close(1009, reason).await;
                        return false;
                    }
                }
                // the reply is queued; `flush` below sends it
                Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)) => {}
                Ok(Message::Close(frame)) => {
                    let _ = self.flush().await;
                    if !self.closed.swap(true, Ordering::AcqRel) {
                        let code = frame.map_or(1005, |f| u16::from(f.code));
                        self.disconnect.send_replace(Some(code));
                    }
                    self.io.finish().await;
                    return false;
                }
                Err(tungstenite::Error::Io(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    let _ = self.flush().await;
                    return true;
                }
                Err(tungstenite::Error::Capacity(_)) => {
                    self.close(1009, "websocket frame or message limit exceeded").await;
                    return false;
                }
                Err(_) => {
                    self.close(1002, "invalid websocket frame").await;
                    return false;
                }
            }
        }
    }

    async fn read_loop(self: Arc<Self>) {
        let mut accepted = self.accept_ready.subscribe();
        let _ = accepted.wait_for(|ready| *ready).await;
        while !self.closed.load(Ordering::Acquire) {
            let Ok(chunk) = self.io.read(READ_SIZE).await else { break };
            if chunk.is_empty() {
                break;
            }
            self.ws().get_mut().inbound.extend(chunk);
            if !self.process().await {
                break;
            }
        }
        // the peer is gone (FIN, reset or a protocol error): tell the app once
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.disconnect.send_replace(Some(1006));
            self.io.finish().await;
        }
    }

    pub async fn send(&self, msg: Sent) -> Result<(), String> {
        match msg {
            Sent::WsAccept { subprotocol, headers } => {
                if self.accepted.swap(true, Ordering::AcqRel) {
                    return Err("websocket already accepted".into());
                }
                let accept = tungstenite::handshake::derive_accept_key(&self.key);
                let mut out = vec![
                    (b"Upgrade".to_vec(), b"WebSocket".to_vec()),
                    (b"Connection".to_vec(), b"Upgrade".to_vec()),
                    (b"Sec-WebSocket-Accept".to_vec(), accept.into_bytes()),
                ];
                if let Some(p) = subprotocol {
                    if !self.subprotocols.contains(&p) {
                        return Err(format!("unexpected subprotocol {p}"));
                    }
                    out.push((b"Sec-WebSocket-Protocol".to_vec(), p.into_bytes()));
                }
                out.extend(headers);
                self.write_raw(&response_head(101, &out)).await?;
                // frames the peer sent ahead of the 101
                if !self.process().await && !self.closed.load(Ordering::Acquire) {
                    self.close(1002, "").await;
                }
                self.accept_ready.send_replace(true);
                Ok(())
            }
            Sent::WsText(_) | Sent::WsBytes(_) => {
                if !self.accepted.load(Ordering::Acquire) {
                    return Err("websocket.send before accept".into());
                }
                if self.closed.load(Ordering::Acquire) {
                    return Ok(());
                }
                self.send_frame(match msg {
                    Sent::WsText(t) => Message::text(t),
                    Sent::WsBytes(b) => Message::binary(b),
                    _ => unreachable!(),
                })
                .await
            }
            Sent::WsClose { code, reason } => {
                if self.accepted.load(Ordering::Acquire) {
                    self.close(code, &reason).await;
                } else {
                    self.reject(403, Vec::new()).await;
                }
                Ok(())
            }
            // a rejection with a body, as the ASGI extension spells it
            Sent::WsHttpResponseStart { status, headers } => {
                self.reject(status, headers).await;
                Ok(())
            }
            Sent::WsHttpResponseBody { body, more_body } => {
                if !body.is_empty() {
                    self.write_raw(&body).await?;
                }
                if !more_body {
                    self.io.finish().await;
                }
                Ok(())
            }
            other => {
                Err(format!("unexpected ASGI message {:?} on a websocket scope", kind_of(&other)))
            }
        }
    }

    /// A declined upgrade is an ordinary HTTP response, then FIN.
    async fn reject(&self, status: u16, mut headers: Headers) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        headers.push((b"content-length".to_vec(), b"0".to_vec()));
        if self.write_raw(&response_head(status, &headers)).await.is_ok() {
            self.io.finish().await;
        }
    }

    /// A close frame to the client, then FIN. Idempotent.
    pub async fn close(&self, code: u16, reason: &str) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let frame = CloseFrame { code: CloseCode::from(code), reason: reason.to_owned().into() };
        let _ = self.ws().close(Some(frame));
        if tokio::time::timeout(CLOSE_FLUSH_TIMEOUT, self.flush()).await.is_err() {
            self.io.reset(Reset::HsrAbort).await;
        } else {
            self.io.finish().await;
        }
        self.disconnect.send_replace(Some(code));
    }
}

pub(super) async fn serve_ws(
    io: Arc<dyn StreamIo>,
    info: RequestInfo,
    mut head: Head,
    app: &dyn AsgiApp,
    registry: &Registry,
    mut payload: EndpointReservation,
) {
    // wsproto's checks on the upgrade request
    let version = head.values(b"sec-websocket-version").next();
    let key = head.values(b"sec-websocket-key").next().map(<[u8]>::to_vec);
    let host = head.values(b"host").next().map(<[u8]>::to_vec);
    let (Some(key), Some(host), Some(b"13")) = (key, host, version) else {
        log::warn!(target: LOG, requestId = info.request_id.as_str();
            "endpoint websocket upgrade is incomplete");
        io.reset(Reset::HsrProtocol).await;
        return;
    };
    let skipped: [&[u8]; 3] = [b"host", b"sec-websocket-extensions", b"sec-websocket-protocol"];
    let mut headers: Headers =
        head.headers.iter().filter(|(k, _)| !skipped.contains(&k.as_slice())).cloned().collect();
    headers.extend([
        (b"host".to_vec(), host),
        (b"upgrade".to_vec(), b"websocket".to_vec()),
        (b"connection".to_vec(), b"upgrade".to_vec()),
    ]);
    let subprotocols: Vec<String> = head
        .values(b"sec-websocket-protocol")
        .flat_map(|v| v.split(|b| *b == b','))
        .map(|t| String::from_utf8_lossy(t.trim_ascii()).into_owned())
        .filter(|t| !t.is_empty())
        .collect();
    let mut scope = scope_base(&info, "websocket", headers, &head.target);
    scope.scheme = if info.scheme == "https" { "wss" } else { "ws" }.into();
    scope.subprotocols = subprotocols.clone();

    let limits = registry.admission.limits;
    let Some(parser) = registry.admission.try_payload(limits.websocket_message_bytes + READ_SIZE)
    else {
        io.reset(Reset::HsrRefused).await;
        return;
    };
    let config = WebSocketConfig::default()
        .max_message_size(Some(limits.websocket_message_bytes))
        .max_frame_size(Some(limits.websocket_frame_bytes));
    let mut pipe = Pipe::default();
    pipe.inbound.extend(std::mem::take(&mut head.rest));
    drop(head);
    payload.release_payload();
    let (events, inbox) = mpsc::channel(limits.websocket_inbox);
    let _ = events.try_send(InboxEvent { event: Received::WsConnect, _payload: None });
    let ex = Arc::new(WsExchange {
        _payload: payload,
        _parser: parser,
        io,
        key,
        subprotocols,
        ws: std::sync::Mutex::new(WebSocket::from_raw_socket(pipe, Role::Server, Some(config))),
        write_lock: Mutex::new(()),
        events,
        inbox: Mutex::new(inbox),
        admission: registry.admission.clone(),
        inbox_admission: MessageAdmission::new(limits.websocket_inbox_bytes),
        write_admission: MessageAdmission::new(limits.websocket_outbox_bytes),
        disconnect: watch::channel(None).0,
        accept_ready: watch::channel(false).0,
        accepted: AtomicBool::new(false),
        closed: AtomicBool::new(false),
    });
    let id = registry.next.fetch_add(1, Ordering::Relaxed);
    registry.lock().insert(id, Arc::downgrade(&ex));
    // Cancelled when this exchange is, with or without the guard.
    let reader = TokioSpawner::current().spawn(ex.clone().read_loop());

    let guard = {
        let ex = ex.clone();
        OnCancel(Some(Box::new(move || {
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move { ex.close(1001, "worker stopped").await });
            }
        })))
    };
    let result = app.call(scope, Conn::Ws(ex.clone())).await;
    guard.disarm();
    let undecided = || !ex.accepted.load(Ordering::Acquire) && !ex.closed.load(Ordering::Acquire);
    match result {
        // the app returned without deciding: ASGI servers answer 403
        Ok(()) if undecided() => ex.reject(403, Vec::new()).await,
        Ok(()) => {}
        Err(e) => {
            log::error!(target: LOG, requestId = info.request_id.as_str(), route = info.route.as_str();
                "websocket endpoint failed\n{}{}", e.traceback, e.summary);
            if undecided() {
                ex.reject(500, Vec::new()).await;
            } else {
                ex.close(1011, &e.summary).await;
            }
        }
    }
    registry.lock().remove(&id);
    reader.cancel();
    let _ = reader.await;
    ex.close(1000, "").await;
}
