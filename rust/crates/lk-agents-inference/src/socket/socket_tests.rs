use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use lk_agents_abi::errors::ErrorKind;
use lk_agents_abi::utils::socket::{KeepaliveFrame, SocketSend, SocketTarget};
use lk_agents_abi::{BoxFuture, VirtualClock};
use parking_lot::Mutex;
use tokio::sync::mpsc;

use super::*;

// The port the DST fakes: a scripted provider socket per connect.

enum Peer {
    Message(WsMessage),
    Close(Option<u16>, Option<&'static str>),
}

struct FakeSocket {
    url: String,
    peer: mpsc::UnboundedSender<Peer>,
    sent: Mutex<Vec<String>>,
    closed: AtomicBool,
}

impl FakeSocket {
    fn text(&self, text: &str) {
        let _ = self.peer.send(Peer::Message(WsMessage::Text(text.to_owned())));
    }

    fn close(&self, code: Option<u16>, reason: Option<&'static str>) {
        let _ = self.peer.send(Peer::Close(code, reason));
    }

    fn sent(&self) -> Vec<String> {
        self.sent.lock().clone()
    }
}

#[derive(Clone, Default)]
struct Provider {
    sockets: Arc<Mutex<Vec<Arc<FakeSocket>>>>,
}

impl Provider {
    fn transports(&self) -> Transports {
        let provider = self.clone();
        Arc::new(move |_proxy| Arc::new(provider.clone()) as Arc<dyn WsTransport>)
    }

    fn socket(&self, index: usize) -> Arc<FakeSocket> {
        Arc::clone(&self.sockets.lock()[index])
    }

    fn count(&self) -> usize {
        self.sockets.lock().len()
    }
}

#[async_trait]
impl WsTransport for Provider {
    async fn connect(
        &self,
        url: &str,
        _headers: Headers,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        let (peer, inbound) = mpsc::unbounded_channel();
        let socket = Arc::new(FakeSocket {
            url: url.to_owned(),
            peer,
            sent: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
        });
        self.sockets.lock().push(Arc::clone(&socket));
        let receiver = FakeReceiver { inbound, code: None, reason: None };
        Ok((Arc::new(FakeSender(socket)), Box::new(receiver)))
    }
}

struct FakeSender(Arc<FakeSocket>);

#[async_trait]
impl WsSender for FakeSender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        if self.0.closed.load(Ordering::SeqCst) {
            return Err(TransportError::Closed);
        }
        self.0.sent.lock().push(text);
        Ok(())
    }

    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        self.send_text(format!("<{} bytes>", data.len())).await
    }

    fn is_closed(&self) -> bool {
        self.0.closed.load(Ordering::SeqCst)
    }

    async fn close(&self) {
        self.0.closed.store(true, Ordering::SeqCst);
    }
}

struct FakeReceiver {
    inbound: mpsc::UnboundedReceiver<Peer>,
    code: Option<u16>,
    reason: Option<String>,
}

#[async_trait]
impl WsReceiver for FakeReceiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        match self.inbound.recv().await {
            Some(Peer::Message(message)) => Ok(message),
            Some(Peer::Close(code, reason)) => {
                self.code = code;
                self.reason = reason.map(str::to_owned);
                Ok(WsMessage::Closed)
            }
            None => std::future::pending().await,
        }
    }

    fn close_code(&self) -> Option<u16> {
        self.code
    }

    fn close_reason(&self) -> Option<String> {
        self.reason.clone()
    }
}

// A plugin's protocol: scripted input, a record of what it read.

struct Protocol {
    input: tokio::sync::Mutex<mpsc::UnboundedReceiver<SocketSend>>,
    connection: Mutex<SocketConnection>,
    read: Mutex<Vec<String>>,
    retried: Mutex<Vec<ErrorKind>>,
    connects: AtomicUsize,
}

impl Protocol {
    fn new(connection: SocketConnection) -> (Arc<Self>, mpsc::UnboundedSender<SocketSend>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let protocol = Protocol {
            input: tokio::sync::Mutex::new(rx),
            connection: Mutex::new(connection),
            read: Mutex::new(Vec::new()),
            retried: Mutex::new(Vec::new()),
            connects: AtomicUsize::new(0),
        };
        (Arc::new(protocol), tx)
    }
}

impl SocketHost for Protocol {
    fn connect(
        &self,
        _cx: &Cx,
        _conn_options: ApiConnectOptions,
    ) -> BoxFuture<'_, Result<SocketConnection, ProviderError>> {
        let n = self.connects.fetch_add(1, Ordering::SeqCst);
        let mut connection = self.connection.lock().clone();
        connection.target.params.insert("n".into(), n.to_string());
        Box::pin(async move { Ok(connection) })
    }

    fn next(&self, _cx: &Cx) -> BoxFuture<'_, Result<SocketSend, ProviderError>> {
        Box::pin(async move {
            match self.input.lock().await.recv().await {
                Some(item) => Ok(item),
                None => std::future::pending().await,
            }
        })
    }

    fn on_message(&self, _cx: &Cx, frame: WsFrame) -> BoxFuture<'_, Result<bool, ProviderError>> {
        Box::pin(async move {
            let WsFrame::Text(text) = frame else { panic!("text frames only") };
            if text == "malformed" {
                let error =
                    ApiError::new(ErrorKind::Protocol, "the provider sent a malformed frame");
                return Err(error.into());
            }
            self.read.lock().push(text.clone());
            // a multiplexed frame leads with its context: `<context>:done`
            Ok(text == "done" || text.ends_with(":done"))
        })
    }

    fn retrying(&self, error: ProviderError) {
        self.retried.lock().push(error.attempt_error().error_kind);
    }
}

fn connection(goodbye: bool) -> SocketConnection {
    SocketConnection {
        target: SocketTarget {
            url: "wss://provider.test/listen".into(),
            headers: BTreeMap::new(),
            params: BTreeMap::new(),
            protocols: Vec::new(),
            request_id_header: None,
        },
        keepalive: Some(KeepaliveFrame { interval: 5.0, frame: WsFrame::Text("keepalive".into()) }),
        goodbye: goodbye.then(|| WsFrame::Text("goodbye".into())),
        max_session_duration: None,
        proxy: None,
    }
}

fn item(text: &str) -> SocketSend {
    SocketSend { frames: vec![WsFrame::Text(text.into())], boundary: false, end: false }
}

fn end() -> SocketSend {
    SocketSend { frames: vec![WsFrame::Text("finalize".into())], boundary: true, end: true }
}

fn options(max_retry: u32) -> ApiConnectOptions {
    ApiConnectOptions { max_retry, retry_interval: 0.0, ..ApiConnectOptions::default() }
}

fn env(clock: &Arc<VirtualClock>) -> ProcessEnv {
    ProcessEnv { clock: Arc::clone(clock) as Arc<dyn Clock>, ..ProcessEnv::current_runtime() }
}

/// A running session and where its outcome lands; every wait on it is bounded.
struct Run {
    session: Arc<Session>,
    outcome: Arc<Mutex<Option<Result<(), ProviderError>>>>,
}

impl Run {
    fn spawn(session: Session) -> Run {
        let session = Arc::new(session);
        let outcome = Arc::new(Mutex::new(None));
        let (running, slot) = (Arc::clone(&session), Arc::clone(&outcome));
        tokio::spawn(async move {
            let ended = running.run(&Cx::NONE).await;
            *slot.lock() = Some(ended);
        });
        Run { session, outcome }
    }

    fn is_finished(&self) -> bool {
        self.outcome.lock().is_some()
    }

    /// The outcome, failing the test if the run never ends.
    async fn ended(&self) -> Result<(), ProviderError> {
        until("the run ended", || self.is_finished()).await;
        self.outcome.lock().take().expect("the run ended")
    }
}

fn start(
    protocol: &Arc<Protocol>,
    provider: &Provider,
    clock: &Arc<VirtualClock>,
    retries: u32,
) -> Run {
    let host: Arc<dyn SocketHost> = Arc::clone(protocol) as Arc<dyn SocketHost>;
    Run::spawn(Session::direct(
        host,
        options(retries),
        "fake".into(),
        env(clock),
        provider.transports(),
    ))
}

async fn until(what: &str, mut ready: impl FnMut() -> bool) {
    for _ in 0..2000 {
        if ready() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("never happened: {what}");
}

// Catches: a stream ended at the input's end without the goodbye, or before the provider's
// last answer (finals lost).
#[tokio::test]
async fn the_input_end_sends_the_goodbye_and_drains_until_the_provider_finishes() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection(true));
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("audio")).unwrap();
    input.send(end()).unwrap();
    until("the goodbye", || provider.count() == 1 && provider.socket(0).sent().len() == 3).await;
    assert_eq!(provider.socket(0).sent(), ["audio", "finalize", "goodbye"]);
    assert!(!run.is_finished(), "the provider's final answer is still owed");
    provider.socket(0).text("final transcript");
    provider.socket(0).text("done");
    run.ended().await.unwrap();
    assert_eq!(*protocol.read.lock(), ["final transcript", "done"]);
    assert!(provider.socket(0).closed.load(Ordering::SeqCst), "closed after the drain");
}

// Catches: a close nobody expected read as the stream's end (the loop that ends silently), and
// a retried failure not reported once.
#[tokio::test]
async fn an_unexpected_close_is_retried_on_a_new_connection_and_reported() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection(true));
    let run = start(&protocol, &provider, &clock, 1);
    until("connected", || provider.count() == 1).await;
    provider.socket(0).close(Some(1000), None);
    until("reconnected", || provider.count() == 2).await;
    assert_eq!(*protocol.retried.lock(), [ErrorKind::Connection]);
    assert!(
        provider.socket(1).url.contains("n=1"),
        "connect ran again: {}",
        provider.socket(1).url
    );
    input.send(end()).unwrap();
    until("the goodbye", || provider.socket(1).sent().len() == 2).await;
    provider.socket(1).close(Some(1000), None);
    run.ended().await.unwrap();
}

// Catches: an auth refusal retried until the budget ran out, or its reason dropped.
#[tokio::test]
async fn a_final_close_code_fails_at_once_with_its_kind_and_reason() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, _input) = Protocol::new(connection(true));
    let run = start(&protocol, &provider, &clock, 3);
    until("connected", || provider.count() == 1).await;
    provider.socket(0).close(Some(1008), Some("invalid api key"));
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!((error.error_kind, error.retryable), (ErrorKind::Auth, false));
    assert_eq!(error.said.and_then(|said| said.detail).as_deref(), Some("invalid api key"));
    assert_eq!(provider.count(), 1);
    assert!(protocol.retried.lock().is_empty());
}

// Catches: an `on_message` failure logged and dropped while the stream went on.
#[tokio::test]
async fn a_frame_the_protocol_refuses_fails_the_stream() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, _input) = Protocol::new(connection(true));
    let run = start(&protocol, &provider, &clock, 3);
    until("connected", || provider.count() == 1).await;
    provider.socket(0).text("malformed");
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!((error.error_kind, error.retryable), (ErrorKind::Protocol, false));
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
}

// Catches: a hand-written reconnect that spends a retry, drops what the old connection still
// held, or loses the input queued meanwhile.
#[tokio::test]
async fn reconnect_drains_the_old_connection_and_keeps_the_input() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection(true));
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("one")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["one"]).await;
    run.session.reconnect("options changed");
    until("the goodbye", || provider.socket(0).sent() == ["one", "goodbye"]).await;
    input.send(item("two")).unwrap();
    provider.socket(0).text("last of one");
    provider.socket(0).close(Some(1000), None);
    until("the new connection", || provider.count() == 2).await;
    until("held input sent", || provider.socket(1).sent() == ["two"]).await;
    assert!(provider.socket(0).sent().iter().all(|frame| frame != "two"));
    assert_eq!(*protocol.read.lock(), ["last of one"]);
    assert!(protocol.retried.lock().is_empty(), "a rotation is not a failure");
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a keepalive on a fixed beat (or none), and a drain with no bound.
#[tokio::test]
async fn keepalive_fills_idle_time_and_a_silent_drain_times_out() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection(true));
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || provider.count() == 1).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(5)).await;
    until("keepalive", || provider.socket(0).sent() == ["keepalive"]).await;
    input.send(end()).unwrap();
    until("the goodbye", || provider.socket(0).sent().len() == 3).await;
    crate::testkit::arm_then_advance(&clock, DRAIN_LIMIT).await;
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!(error.timeout, Some(lk_agents_abi::errors::Timeout::Read));
    assert_eq!(provider.socket(0).sent(), ["keepalive", "finalize", "goodbye"]);
}

// Catches: `aclose()` waiting on a stream parked on its input, or leaving the socket open.
#[tokio::test]
async fn aclose_ends_a_stream_waiting_for_input_and_closes_the_socket() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, _input) = Protocol::new(connection(true));
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || provider.count() == 1).await;
    run.session.aclose().await;
    run.ended().await.unwrap();
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
}

// A multiplexer: frames `<context>:<payload>`.

struct Contexts {
    connection: SocketConnection,
}

impl MuxHost for Contexts {
    fn connect(
        &self,
        _cx: &Cx,
        _conn_options: ApiConnectOptions,
    ) -> BoxFuture<'_, Result<SocketConnection, ProviderError>> {
        Box::pin(async move { Ok(self.connection.clone()) })
    }

    fn route(
        &self,
        _cx: &Cx,
        frame: WsFrame,
    ) -> BoxFuture<'_, Result<Option<String>, ProviderError>> {
        Box::pin(async move {
            let WsFrame::Text(text) = frame else { return Ok(None) };
            Ok(text.split_once(':').map(|(context, _)| context.to_owned()))
        })
    }

    fn open_context(
        &self,
        _cx: &Cx,
        context_id: String,
    ) -> BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>> {
        Box::pin(async move { Ok(vec![WsFrame::Text(format!("open {context_id}"))]) })
    }

    fn close_context(
        &self,
        _cx: &Cx,
        context_id: String,
    ) -> BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>> {
        Box::pin(async move { Ok(vec![WsFrame::Text(format!("cancel {context_id}"))]) })
    }
}

// Catches: each multiplexed stream opening its own socket, a frame reaching the wrong stream,
// or a stream left early without the provider's cancel.
#[tokio::test]
async fn multiplexed_streams_share_one_socket_and_get_only_their_frames() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = Arc::new(Mux::new(
        Arc::new(Contexts { connection: connection(false) }),
        None,
        env(&clock),
        provider.transports(),
    ));
    let mut streams = Vec::new();
    for context in ["a", "b"] {
        let (protocol, input) = Protocol::new(connection(false));
        let run = Run::spawn(Session::shared(
            Arc::clone(&protocol) as Arc<dyn SocketHost>,
            options(0),
            "fake".into(),
            env(&clock),
            Arc::clone(&mux),
            context.into(),
        ));
        streams.push((protocol, input, run));
        until("opened", || {
            provider.count() == 1 && provider.socket(0).sent().len() == streams.len()
        })
        .await;
    }
    assert_eq!(provider.count(), 1, "one socket for both streams");
    let socket = provider.socket(0);
    socket.text("b:hello b");
    socket.text("a:done");
    let (protocol_a, _, run_a) = streams.remove(0);
    run_a.ended().await.unwrap();
    assert_eq!(*protocol_a.read.lock(), ["a:done"]);
    let (protocol_b, _, run_b) = streams.remove(0);
    until("b read", || protocol_b.read.lock().len() == 1).await;
    assert_eq!(*protocol_b.read.lock(), ["b:hello b"]);
    run_b.session.aclose().await;
    run_b.ended().await.unwrap();
    assert_eq!(socket.sent(), ["open a", "open b", "cancel b"], "a finished, b was cut short");
    assert!(!socket.closed.load(Ordering::SeqCst), "the socket stays for the next stream");
    mux.aclose().await;
    assert!(socket.closed.load(Ordering::SeqCst));
}
