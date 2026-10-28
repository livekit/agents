//! Fakes for the [`crate::transport`] seam, for tests only (`cfg(test)` or the `testkit` feature).
//! Deterministic; the only clock is the caller's `VirtualClock`.

use crate::transport::{
    TransportError, WsMessage, WsReceiver, WsSender, WsTransport,
};
use async_trait::async_trait;
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
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

#[allow(clippy::disallowed_types, reason = "S0: runtime migration")]
/// A scripted WebSocket: records sends, replays canned incoming messages unsynchronised with them.
type Incoming = Arc<Mutex<std::collections::VecDeque<Result<WsMessage, TransportError>>>>;

pub struct FakeWs {
    incoming: Incoming,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    sent: Arc<Mutex<Vec<String>>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    connects: Mutex<Vec<ConnectRecord>>,
    /// Fail the handshake instead of connecting.
    connect_error: Option<TransportError>,
    /// Park after the script instead of reporting a close.
    keep_open: bool,
    /// Never complete the handshake; the client's connect bound ends it.
    hang_connect: bool,
    /// Never complete a send: a peer that stopped reading, its TCP window full.
    stall_sends: bool,
}

impl FakeWs {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    pub fn new(incoming: Vec<Result<WsMessage, TransportError>>) -> Self {
        FakeWs {
            incoming: Arc::new(Mutex::new(incoming.into_iter().collect())),
            sent: Arc::new(Mutex::new(Vec::new())),
            connects: Mutex::new(Vec::new()),
            connect_error: None,
            keep_open: false,
            hang_connect: false,
            stall_sends: false,
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

    /// Connected and kept open, but no send ever completes.
    pub fn stalling_sends() -> Self {
        FakeWs { stall_sends: true, ..FakeWs::new(Vec::new()).keep_open() }
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

    pub fn connect_count(&self) -> usize {
        self.connects.lock().unwrap().len()
    }

    pub fn last_connect(&self) -> Option<ConnectRecord> {
        self.connects.lock().unwrap().last().cloned()
    }
}

#[async_trait]
impl WsTransport for FakeWs {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
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
        let sender = Arc::new(FakeWsSender {
            sent: self.sent.clone(),
            closed: Mutex::new(false),
            stall: self.stall_sends,
        });
        let receiver =
            Box::new(FakeWsReceiver { incoming: self.incoming.clone(), keep_open: self.keep_open });
        Ok((sender, receiver))
    }
}

struct FakeWsSender {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    sent: Arc<Mutex<Vec<String>>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    closed: Mutex<bool>,
    stall: bool,
}

#[async_trait]
impl WsSender for FakeWsSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        if *self.closed.lock().unwrap() {
            return Err(TransportError::Closed);
        }
        if self.stall {
            std::future::pending::<()>().await;
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
