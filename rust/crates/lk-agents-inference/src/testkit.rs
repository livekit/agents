//! Fakes for the [`crate::transport`] seam. Not `#[cfg(test)]`: the driver and the binding use
//! them too. Deterministic; the only clock is the caller's `VirtualClock`.

use crate::transport::{
    HttpRequest, HttpStream, HttpTransport, LineStream, TransportError, WsMessage, WsReceiver,
    WsSender, WsTransport,
};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// A recorded `connect` call: the URL and its headers.
pub type ConnectRecord = (String, Vec<(String, String)>);

/// An [`ImageResolver`](lk_agents_abi::llm::provider_format::ImageResolver) that refuses every image.
pub struct NoImages;

impl lk_agents_abi::llm::provider_format::ImageResolver for NoImages {
    fn resolve(
        &self,
        _image: &lk_agents_abi::ImageContent,
    ) -> Result<lk_agents_abi::llm::provider_format::SerializedImage, String> {
        Err("no image resolver is attached to this inference client".to_owned())
    }
}

/// A scripted [`HttpTransport`]: one canned SSE stream, replayed in order.
pub struct FakeHttp {
    events: Mutex<std::collections::VecDeque<Result<String, TransportError>>>,
    seen: Mutex<Vec<HttpRequest>>,
    /// `post_stream` never resolves; the client's connect bound ends it.
    hang_connect: bool,
    /// After the script, `next_event` never resolves; the client's per-event bound ends it.
    hang_at_end: bool,
}

impl FakeHttp {
    pub fn new(events: Vec<Result<String, TransportError>>) -> Self {
        FakeHttp {
            events: Mutex::new(events.into()),
            seen: Mutex::new(Vec::new()),
            hang_connect: false,
            hang_at_end: false,
        }
    }

    pub fn hanging() -> Self {
        FakeHttp { hang_connect: true, ..FakeHttp::new(Vec::new()) }
    }

    pub fn then_hang(mut self) -> Self {
        self.hang_at_end = true;
        self
    }

    pub fn requests(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    pub fn last_request(&self) -> Option<HttpRequest> {
        self.seen.lock().unwrap().last().cloned()
    }
}

#[async_trait]
impl HttpTransport for FakeHttp {
    async fn post_stream(&self, req: HttpRequest) -> Result<HttpStream, TransportError> {
        self.seen.lock().unwrap().push(req);
        if self.hang_connect {
            std::future::pending::<()>().await;
        }
        let mut events = self.events.lock().unwrap();
        // An error in first position is a connect failure; anywhere else a mid-stream failure.
        if let Some(Err(_)) = events.front() {
            let Some(Err(err)) = events.pop_front() else { unreachable!() };
            return Err(err);
        }
        let events = std::mem::take(&mut *events);
        Ok(HttpStream {
            status: 200,
            headers: vec![],
            body: Box::new(ScriptedLines { events, hang_at_end: self.hang_at_end }),
        })
    }
}

struct ScriptedLines {
    events: std::collections::VecDeque<Result<String, TransportError>>,
    hang_at_end: bool,
}

#[async_trait]
impl LineStream for ScriptedLines {
    async fn next_event(&mut self) -> Result<Option<String>, TransportError> {
        match self.events.pop_front() {
            Some(Ok(payload)) => Ok(Some(payload)),
            Some(Err(err)) => Err(err),
            None => {
                if self.hang_at_end {
                    std::future::pending::<()>().await;
                }
                Ok(None)
            }
        }
    }
}

/// A scripted WebSocket: records sends, replays canned incoming messages unsynchronised with them.
type Incoming = Arc<Mutex<std::collections::VecDeque<Result<WsMessage, TransportError>>>>;

pub struct FakeWs {
    incoming: Incoming,
    sent: Arc<Mutex<Vec<String>>>,
    connects: Mutex<Vec<ConnectRecord>>,
    /// Fail the handshake instead of connecting.
    connect_error: Option<TransportError>,
    /// Park after the script instead of reporting a close.
    keep_open: bool,
    /// Never complete the handshake; the client's connect bound ends it.
    hang_connect: bool,
}

impl FakeWs {
    pub fn new(incoming: Vec<Result<WsMessage, TransportError>>) -> Self {
        FakeWs {
            incoming: Arc::new(Mutex::new(incoming.into_iter().collect())),
            sent: Arc::new(Mutex::new(Vec::new())),
            connects: Mutex::new(Vec::new()),
            connect_error: None,
            keep_open: false,
            hang_connect: false,
        }
    }

    /// A gateway that accepts the TCP connection and never finishes the WebSocket handshake.
    pub fn hanging() -> Self {
        FakeWs { hang_connect: true, ..FakeWs::new(Vec::new()) }
    }

    /// Stay connected once the script runs out instead of reporting a close (for send-side tests).
    pub fn keep_open(mut self) -> Self {
        self.keep_open = true;
        self
    }

    pub fn failing(err: TransportError) -> Self {
        FakeWs { connect_error: Some(err), ..FakeWs::new(vec![]) }
    }

    /// Queue another incoming message after the stream is running. Only meaningful with
    /// [`keep_open`](Self::keep_open).
    pub fn feed(&self, msg: Result<WsMessage, TransportError>) {
        self.incoming.lock().unwrap().push_back(msg);
    }

    pub fn sent(&self) -> Vec<String> {
        self.sent.lock().unwrap().clone()
    }

    /// Panics on a non-JSON frame.
    pub fn sent_json(&self) -> Vec<serde_json::Value> {
        self.sent().iter().map(|s| serde_json::from_str(s).expect("a JSON frame")).collect()
    }

    pub fn last_connect(&self) -> Option<ConnectRecord> {
        self.connects.lock().unwrap().last().cloned()
    }
}

#[async_trait]
impl WsTransport for FakeWs {
    async fn connect(
        &self,
        url: &str,
        headers: Vec<(String, String)>,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        self.connects.lock().unwrap().push((url.to_owned(), headers));
        if let Some(err) = self.connect_error.clone() {
            return Err(err);
        }
        if self.hang_connect {
            std::future::pending::<()>().await;
        }
        let sender = Arc::new(FakeWsSender { sent: self.sent.clone(), closed: Mutex::new(false) });
        let receiver =
            Box::new(FakeWsReceiver { incoming: self.incoming.clone(), keep_open: self.keep_open });
        Ok((sender, receiver))
    }
}

struct FakeWsSender {
    sent: Arc<Mutex<Vec<String>>>,
    closed: Mutex<bool>,
}

#[async_trait]
impl WsSender for FakeWsSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        if *self.closed.lock().unwrap() {
            return Err(TransportError::Closed);
        }
        self.sent.lock().unwrap().push(text);
        Ok(())
    }

    /// Recorded as its length, so a test can count the audio sent.
    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        if *self.closed.lock().unwrap() {
            return Err(TransportError::Closed);
        }
        self.sent.lock().unwrap().push(format!("<{} bytes>", data.len()));
        Ok(())
    }

    fn is_closed(&self) -> bool {
        *self.closed.lock().unwrap()
    }

    async fn close(&self) {
        *self.closed.lock().unwrap() = true;
    }
}

struct FakeWsReceiver {
    incoming: Incoming,
    keep_open: bool,
}

#[async_trait]
impl WsReceiver for FakeWsReceiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        loop {
            if let Some(msg) = self.incoming.lock().unwrap().pop_front() {
                return msg;
            }
            if !self.keep_open {
                // The default: hang up once the script is done.
                return Ok(WsMessage::Closed);
            }
                // Yield rather than park, so `feed` can still deliver.
            tokio::task::yield_now().await;
        }
    }
}

pub fn text(json: serde_json::Value) -> Result<WsMessage, TransportError> {
    Ok(WsMessage::Text(json.to_string()))
}

/// Let the wait under test arm its deadline, then move the virtual clock past it by `by`.
pub async fn arm_then_advance(clock: &lk_agents_abi::VirtualClock, by: std::time::Duration) {
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    clock.advance(by);
}
