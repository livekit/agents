use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use lk_agents_abi::errors::{ErrorKind, HostError};
use lk_agents_abi::utils::websocket::{
    WebSocketContextsCancel, WebSocketInput, WebSocketKeepaliveFrame, WebSocketMuxRoute,
    WebSocketRead, WebSocketSignal, WebSocketTarget,
};
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
    /// While set, a write reaches the peer but does not return to the writer.
    held_writes: Arc<AtomicBool>,
}

impl FakeSocket {
    fn text(&self, text: &str) {
        let _ = self.peer.send(Peer::Message(WsMessage::Text(text.to_owned())));
    }

    fn close(&self, code: Option<u16>, reason: Option<&'static str>) {
        let _ = self.peer.send(Peer::Close(code, reason));
    }

    /// The answer to a ping.
    fn pong(&self) {
        let _ = self.peer.send(Peer::Message(WsMessage::Other));
    }

    fn sent(&self) -> Vec<String> {
        self.sent.lock().clone()
    }
}

#[derive(Clone, Default)]
struct Provider {
    sockets: Arc<Mutex<Vec<Arc<FakeSocket>>>>,
    /// The handshake yields once its socket exists, before it is handed over.
    slow_handshake: bool,
    /// The handshake's response headers.
    headers: Vec<(&'static str, &'static str)>,
    /// The handshake is refused with this status and reason.
    refused: Option<(i64, &'static str)>,
    held_writes: Arc<AtomicBool>,
}

impl Provider {
    fn transports(&self) -> Transports {
        let provider = self.clone();
        Arc::new(move |_proxy, _max_message_size| {
            Arc::new(provider.clone()) as Arc<dyn WsTransport>
        })
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
        if let Some((status, reason)) = self.refused {
            let message = reason.to_owned();
            return Err(TransportError::Status {
                status,
                message,
                headers: Headers::default(),
                body: None,
            });
        }
        let (peer, inbound) = mpsc::unbounded_channel();
        let socket = Arc::new(FakeSocket {
            url: url.to_owned(),
            peer,
            sent: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
            held_writes: Arc::clone(&self.held_writes),
        });
        self.sockets.lock().push(Arc::clone(&socket));
        if self.slow_handshake {
            tokio::task::yield_now().await;
        }
        let headers = self.headers.clone();
        let receiver = FakeReceiver { inbound, code: None, reason: None, headers };
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
        while self.0.held_writes.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
        Ok(())
    }

    async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
        self.send_text(format!("<{} bytes>", data.len())).await
    }

    async fn ping(&self) -> Result<(), TransportError> {
        self.send_text("<ping>".into()).await
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
    headers: Vec<(&'static str, &'static str)>,
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

    fn response_header(&self, name: &str) -> Option<String> {
        self.headers.iter().find(|(key, _)| *key == name).map(|(_, value)| (*value).to_owned())
    }
}

// A plugin's protocol: scripted input, a record of what it read.

struct Protocol {
    input: tokio::sync::Mutex<mpsc::UnboundedReceiver<Option<WebSocketInput>>>,
    controls: tokio::sync::Mutex<mpsc::UnboundedReceiver<Vec<WsFrame>>>,
    controls_tx: mpsc::UnboundedSender<Vec<WsFrame>>,
    connection: Mutex<WebSocketConnection>,
    read: Mutex<Vec<String>>,
    retried: Mutex<Vec<ErrorKind>>,
    connects: AtomicUsize,
    /// What the stream answers: it owes the provider's reply at the input's end, nothing it sent
    /// is pending, it asked the provider to end its session, it can be replaced now.
    owes_reply: AtomicBool,
    nothing_pending: AtomicBool,
    expects_close: AtomicBool,
    can_reconnect: AtomicBool,
    /// Seconds without input after which its input-idle frames go.
    input_idle: Mutex<Option<f64>>,
    /// The provider's ids the connections opened with, and the sessions the provider finished.
    opened: Mutex<Vec<Option<String>>>,
    finished_sessions: AtomicUsize,
}

impl Protocol {
    fn new(
        connection: WebSocketConnection,
    ) -> (Arc<Self>, mpsc::UnboundedSender<Option<WebSocketInput>>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let (controls_tx, controls) = mpsc::unbounded_channel();
        let protocol = Protocol {
            input: tokio::sync::Mutex::new(rx),
            controls: tokio::sync::Mutex::new(controls),
            controls_tx,
            connection: Mutex::new(connection),
            read: Mutex::new(Vec::new()),
            retried: Mutex::new(Vec::new()),
            connects: AtomicUsize::new(0),
            owes_reply: AtomicBool::new(true),
            nothing_pending: AtomicBool::new(false),
            expects_close: AtomicBool::new(false),
            can_reconnect: AtomicBool::new(true),
            input_idle: Mutex::new(None),
            opened: Mutex::new(Vec::new()),
            finished_sessions: AtomicUsize::new(0),
        };
        (Arc::new(protocol), tx)
    }

    /// A control message the stream sends now.
    fn send_now(&self, text: &str) {
        self.controls_tx.send(frames(&[text])).unwrap();
    }
}

impl WebSocketHost for Protocol {
    fn connect(
        &self,
        _cx: &Cx,
        _conn_options: ApiConnectOptions,
    ) -> BoxFuture<'_, Result<WebSocketConnection, ProviderError>> {
        let n = self.connects.fetch_add(1, Ordering::SeqCst);
        let mut connection = self.connection.lock().clone();
        connection.target.params.insert("n".into(), n.to_string());
        Box::pin(async move { Ok(connection) })
    }

    fn next_input(&self, _cx: &Cx) -> BoxFuture<'_, Result<Option<WebSocketInput>, ProviderError>> {
        Box::pin(async move {
            match self.input.lock().await.recv().await {
                Some(item) => Ok(item),
                None => std::future::pending().await,
            }
        })
    }

    fn next_control(&self, _cx: &Cx) -> BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>> {
        Box::pin(async move {
            match self.controls.lock().await.recv().await {
                Some(frames) => Ok(frames),
                None => std::future::pending().await,
            }
        })
    }

    fn on_message(
        &self,
        _cx: &Cx,
        frame: WsFrame,
        _message: Option<HostRef>,
    ) -> BoxFuture<'_, Result<WebSocketRead, ProviderError>> {
        Box::pin(async move {
            let WsFrame::Text(text) = frame else { panic!("text frames only") };
            if text == "malformed" {
                let error =
                    ApiError::new(ErrorKind::Protocol, "the provider sent a malformed frame");
                return Err(error.into());
            }
            self.read.lock().push(text.clone());
            // a multiplexed frame leads with its context: `<context>:finished`
            let payload = text.rsplit(':').next().unwrap_or_default();
            let signal = match payload {
                "finished" => Some(WebSocketSignal::SessionFinished),
                "ready" => Some(WebSocketSignal::Ready),
                _ => None,
            };
            let request_id = payload.strip_prefix("request ").map(str::to_owned);
            Ok(WebSocketRead { signal, request_id, response: false })
        })
    }

    fn input_idle_after(&self) -> Result<Option<f64>, HostError> {
        Ok(*self.input_idle.lock())
    }

    fn input_idle(&self, _cx: &Cx) -> BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>> {
        Box::pin(async move { Ok(frames(&["idle"])) })
    }

    fn response_idle_after(&self) -> Result<Option<f64>, HostError> {
        Ok(None)
    }

    fn response_idle(&self, _cx: &Cx) -> BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>> {
        Box::pin(async move { Ok(Vec::new()) })
    }

    /// Its one open context, `mine`, which the provider ends with `mine:cancelled`.
    fn close_context(
        &self,
        _cx: &Cx,
    ) -> BoxFuture<'_, Result<WebSocketContextsCancel, ProviderError>> {
        Box::pin(async move {
            Ok(WebSocketContextsCancel {
                contexts: vec!["mine".into()],
                frames: frames(&["cancel mine"]),
            })
        })
    }

    fn opened(&self, request_id: Option<String>) {
        self.opened.lock().push(request_id);
    }

    fn session_finished(&self) {
        self.finished_sessions.fetch_add(1, Ordering::SeqCst);
    }

    fn retrying(&self, error: ProviderError) {
        self.retried.lock().push(error.attempt_error().error_kind);
    }

    fn owes_reply(&self) -> Result<bool, HostError> {
        Ok(self.owes_reply.load(Ordering::SeqCst))
    }

    fn completion(
        &self,
    ) -> Result<lk_agents_abi::utils::websocket::WebSocketCompletion, HostError> {
        Ok(Default::default())
    }

    fn nothing_pending(&self) -> Result<bool, HostError> {
        Ok(self.nothing_pending.load(Ordering::SeqCst))
    }

    fn expects_close(&self) -> Result<bool, HostError> {
        Ok(self.expects_close.load(Ordering::SeqCst))
    }

    fn can_reconnect(&self) -> Result<bool, HostError> {
        Ok(self.can_reconnect.load(Ordering::SeqCst))
    }
}

fn connection() -> WebSocketConnection {
    WebSocketConnection {
        target: WebSocketTarget {
            url: "wss://provider.test/listen".into(),
            headers: BTreeMap::new(),
            params: BTreeMap::new(),
            protocols: Vec::new(),
            request_id_header: None,
            max_message_size: None,
        },
        keepalive: Some(WebSocketKeepaliveFrame {
            interval: 5.0,
            frame: WsFrame::Text("keepalive".into()),
            only_while_unused: false,
        }),
        max_session_duration: None,
        ping_interval: None,
        setup: Vec::new(),
        awaits_ready: false,
        goodbye: Vec::new(),
    }
}

/// A connection with no keepalive.
fn quiet() -> WebSocketConnection {
    WebSocketConnection { keepalive: None, ..connection() }
}

/// A connection that opens with a setup and waits for the provider to say it is ready.
fn setup_then_ready() -> WebSocketConnection {
    WebSocketConnection {
        setup: vec![WsFrame::Text("setup".into())],
        awaits_ready: true,
        ..quiet()
    }
}

fn frames(texts: &[&str]) -> Vec<WsFrame> {
    texts.iter().map(|text| WsFrame::Text((*text).into())).collect()
}

/// One input item.
fn item(text: &str) -> Option<WebSocketInput> {
    Some(WebSocketInput { frames: frames(&[text]), pace: 0.0 })
}

/// The input ended.
fn end() -> Option<WebSocketInput> {
    None
}

fn options(max_retry: u32) -> ApiConnectOptions {
    ApiConnectOptions { max_retry, retry_interval: 0.0, ..ApiConnectOptions::default() }
}

/// A multiplexer that routes each frame, up to `max_contexts` streams a socket.
fn routed(max_contexts: Option<u32>) -> WebSocketMuxOptions {
    WebSocketMuxOptions {
        max_contexts,
        max_sockets: None,
        idle_timeout: None,
        keep_warm: false,
        routed: true,
        reuses_sockets: true,
        reads_cancel_end: false,
    }
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
    let host: Arc<dyn WebSocketHost> = Arc::clone(protocol) as Arc<dyn WebSocketHost>;
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

// Catches: a stream ended at the input's end before the provider's last answer (finals lost).
#[tokio::test]
async fn the_input_end_waits_for_the_providers_reply_until_it_finishes() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("audio")).unwrap();
    input.send(item("close stream")).unwrap();
    input.send(end()).unwrap();
    until("the end item", || provider.count() == 1 && provider.socket(0).sent().len() == 2).await;
    crate::testkit::arm_then_advance(&clock, Duration::ZERO).await;
    assert!(!run.is_finished(), "the provider's final answer is still owed");
    provider.socket(0).text("final transcript");
    provider.socket(0).text("finished");
    run.ended().await.unwrap();
    assert_eq!(*protocol.read.lock(), ["final transcript", "finished"]);
    assert_eq!(provider.socket(0).sent(), ["audio", "close stream"]);
    assert!(provider.socket(0).closed.load(Ordering::SeqCst), "closed after the reply");
}

// Catches: a stream that owes nothing at its end (every TTS segment answered) waiting out the
// read limit for an answer that never comes, then failing with a read timeout.
#[tokio::test]
async fn an_end_that_owes_no_reply_closes_at_once() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    protocol.owes_reply.store(false, Ordering::SeqCst);
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("text")).unwrap();
    input.send(end()).unwrap();
    run.ended().await.unwrap();
    assert_eq!(provider.socket(0).sent(), ["text"]);
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
}

// Catches: an end on a connection that carried no input (only the one before a replacement did)
// waiting for a reply the provider has nothing to give.
#[tokio::test]
async fn an_end_with_no_input_on_its_connection_closes_at_once() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("audio")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["audio"]).await;
    run.session.reconnect("options changed");
    until("the new connection", || provider.count() == 2).await;
    input.send(end()).unwrap();
    run.ended().await.unwrap();
    assert!(provider.socket(1).sent().is_empty());
}

// Catches: a close nobody expected read as the stream's end (the loop that ends silently), and
// a retried failure not reported once.
#[tokio::test]
async fn an_unexpected_close_is_retried_on_a_new_connection_and_reported() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
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
    input.send(item("audio")).unwrap();
    input.send(end()).unwrap();
    until("sent", || provider.socket(1).sent() == ["audio"]).await;
    // the provider answered the audio, then closes: the stream's end
    protocol.nothing_pending.store(true, Ordering::SeqCst);
    provider.socket(1).close(Some(1000), None);
    run.ended().await.unwrap();
}

// Catches: an auth refusal retried until the budget ran out, or its reason dropped.
#[tokio::test]
async fn a_final_close_code_fails_at_once_with_its_kind_and_reason() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, _input) = Protocol::new(connection());
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
    let (protocol, _input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 3);
    until("connected", || provider.count() == 1).await;
    provider.socket(0).text("malformed");
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!((error.error_kind, error.retryable), (ErrorKind::Protocol, false));
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
}

// Catches: a hand-written reconnect that spends a retry, or loses the input queued meanwhile.
#[tokio::test]
async fn reconnect_replaces_the_connection_and_keeps_the_input() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("one")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["one"]).await;
    run.session.reconnect("options changed");
    until("the new connection", || provider.count() == 2).await;
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
    input.send(item("two")).unwrap();
    until("the input on the new connection", || provider.socket(1).sent() == ["two"]).await;
    assert!(protocol.retried.lock().is_empty(), "a replacement is not a failure");
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a reconnect asked for while the connection opens lost, so the socket keeps the options
// it was opened with.
#[tokio::test]
async fn a_reconnect_while_the_connection_opens_replaces_it() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider { slow_handshake: true, ..Provider::default() };
    let (protocol, _input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    until("opening", || provider.count() == 1).await;
    run.session.reconnect("options changed");
    until("the new connection", || provider.count() == 2).await;
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a socket gone half-open (its peer no longer answers pings) never noticed, so the
// stream waits on it forever; or a peer that answers each ping failed anyway.
#[tokio::test]
async fn a_socket_that_stops_answering_pings_fails_the_attempt() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let pinged = WebSocketConnection { ping_interval: Some(30.0), ..quiet() };
    let (protocol, _input) = Protocol::new(pinged);
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || provider.count() == 1).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(30)).await;
    until("a ping", || provider.socket(0).sent() == ["<ping>"]).await;
    provider.socket(0).pong();
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(30)).await;
    until("the next ping", || provider.socket(0).sent() == ["<ping>", "<ping>"]).await;
    assert!(!run.is_finished(), "an answered ping keeps the connection");
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(15)).await;
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!(error.error_kind, ErrorKind::Connection);
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
}

// Catches: a keepalive on a fixed beat (or none), and a reply wait with no bound.
#[tokio::test]
async fn keepalive_fills_idle_time_and_a_silent_reply_wait_times_out() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || provider.count() == 1).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(5)).await;
    until("keepalive", || provider.socket(0).sent() == ["keepalive"]).await;
    input.send(item("audio")).unwrap();
    input.send(end()).unwrap();
    until("the audio", || provider.socket(0).sent().len() == 2).await;
    crate::testkit::arm_then_advance(&clock, DEFAULT_WAIT).await;
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!(error.timeout, Some(lk_agents_abi::errors::Timeout::Read));
    assert_eq!(provider.socket(0).sent(), ["keepalive", "audio"]);
}

// Catches: a provider that answered everything it heard and then stays silent after the input's
// end failing the stream with a read timeout.
#[tokio::test]
async fn a_silent_reply_wait_with_nothing_pending_ends_cleanly_at_the_read_limit() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    protocol.nothing_pending.store(true, Ordering::SeqCst);
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("audio")).unwrap();
    input.send(end()).unwrap();
    until("the audio", || provider.count() == 1 && provider.socket(0).sent().len() == 1).await;
    crate::testkit::arm_then_advance(&clock, DEFAULT_WAIT).await;
    run.ended().await.unwrap();
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
}

// Catches: `aclose()` waiting on a stream parked on its input, or leaving the socket open.
#[tokio::test]
async fn aclose_ends_a_stream_waiting_for_input_and_closes_the_socket() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, _input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || provider.count() == 1).await;
    run.session.aclose().await;
    run.ended().await.unwrap();
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
}

// A multiplexer: frames `<context>:<payload>`.

struct Contexts {
    connection: WebSocketConnection,
}

impl WebSocketMuxHost for Contexts {
    fn connect(
        &self,
        _cx: &Cx,
        _conn_options: ApiConnectOptions,
    ) -> BoxFuture<'_, Result<WebSocketConnection, ProviderError>> {
        Box::pin(async move { Ok(self.connection.clone()) })
    }

    fn route(
        &self,
        _cx: &Cx,
        frame: WsFrame,
    ) -> BoxFuture<'_, Result<WebSocketMuxRoute, ProviderError>> {
        Box::pin(async move {
            let WsFrame::Text(text) = frame else { return Ok(WebSocketMuxRoute::default()) };
            if text == "socket ready" {
                return Ok(WebSocketMuxRoute { ready: true, ..WebSocketMuxRoute::default() });
            }
            let (context, payload) = match text.split_once(':') {
                Some((context, payload)) => (Some(context.to_owned()), payload),
                None => (None, ""),
            };
            // `<context>:cancelled`: the provider ended a cancelled context
            let ended_context = context.clone().filter(|_| payload == "cancelled");
            Ok(WebSocketMuxRoute { context, message: None, ready: false, ended_context })
        })
    }

    fn open_context(
        &self,
        _cx: &Cx,
        context_id: String,
    ) -> BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>> {
        Box::pin(async move { Ok(vec![WsFrame::Text(format!("open {context_id}"))]) })
    }

    fn goodbye(&self, _cx: &Cx) -> BoxFuture<'_, Result<Vec<WsFrame>, ProviderError>> {
        Box::pin(async move { Ok(frames(&["goodbye"])) })
    }
}

// Catches: each multiplexed stream opening its own socket, a frame reaching the wrong stream,
// or a stream left early without the provider's cancel.
#[tokio::test]
async fn multiplexed_streams_share_one_socket_and_get_only_their_frames() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = Arc::new(Mux::new(
        Arc::new(Contexts { connection: quiet() }),
        routed(None),
        env(&clock),
        provider.transports(),
    ));
    let mut streams = Vec::new();
    for context in ["a", "b"] {
        let (protocol, input) = Protocol::new(quiet());
        let run = Run::spawn(Session::shared(
            Arc::clone(&protocol) as Arc<dyn WebSocketHost>,
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
    socket.text("a:hello a");
    let (protocol_a, input_a, run_a) = streams.remove(0);
    until("a read", || protocol_a.read.lock().len() == 1).await;
    protocol_a.owes_reply.store(false, Ordering::SeqCst);
    input_a.send(end()).unwrap();
    run_a.ended().await.unwrap();
    assert_eq!(*protocol_a.read.lock(), ["a:hello a"]);
    let (protocol_b, _, run_b) = streams.remove(0);
    until("b read", || protocol_b.read.lock().len() == 1).await;
    assert_eq!(*protocol_b.read.lock(), ["b:hello b"]);
    run_b.session.aclose().await;
    run_b.ended().await.unwrap();
    assert_eq!(socket.sent(), ["open a", "open b", "cancel mine"], "a finished, b was cut short");
    assert!(!socket.closed.load(Ordering::SeqCst), "the socket stays for the next stream");
    mux.aclose().await;
    assert!(socket.closed.load(Ordering::SeqCst));
}

// Catches: a shared socket gone half-open (its peer no longer answers pings) never noticed, so
// every stream on it waits forever.
#[tokio::test]
async fn a_shared_socket_that_stops_answering_pings_fails_every_stream_on_it() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let pinged = WebSocketConnection { ping_interval: Some(30.0), ..quiet() };
    let mux = Arc::new(Mux::new(
        Arc::new(Contexts { connection: pinged }),
        routed(None),
        env(&clock),
        provider.transports(),
    ));
    let mut runs = Vec::new();
    for context in ["a", "b"] {
        let (protocol, input) = Protocol::new(quiet());
        let run = Run::spawn(Session::shared(
            Arc::clone(&protocol) as Arc<dyn WebSocketHost>,
            options(0),
            "fake".into(),
            env(&clock),
            Arc::clone(&mux),
            context.into(),
        ));
        runs.push((input, run));
        until("opened", || provider.count() == 1 && provider.socket(0).sent().len() == runs.len())
            .await;
    }
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(30)).await;
    until("a ping", || provider.socket(0).sent().contains(&"<ping>".to_owned())).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(15)).await;
    for (_input, run) in &runs {
        let error = run.ended().await.unwrap_err().attempt_error();
        assert_eq!(error.error_kind, ErrorKind::Connection);
    }
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
    mux.aclose().await;
}

// Catches: a multiplexing provider's prewarm opening nothing, so the first stream of a call
// pays the socket's connect; and a stream then opening a second socket beside the warm one.
#[tokio::test]
async fn prewarm_opens_the_shared_socket_the_first_stream_then_takes() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let mux = Arc::new(Mux::new(
        Arc::new(Contexts { connection: quiet() }),
        routed(None),
        env(&clock),
        provider.transports(),
    ));
    mux.prewarm();
    until("prewarmed", || provider.count() == 1).await;
    mux.prewarm();
    let (protocol, _input) = Protocol::new(quiet());
    let run = Run::spawn(Session::shared(
        Arc::clone(&protocol) as Arc<dyn WebSocketHost>,
        options(0),
        "fake".into(),
        env(&clock),
        Arc::clone(&mux),
        "a".into(),
    ));
    until("opened", || provider.socket(0).sent() == ["open a"]).await;
    assert_eq!(provider.count(), 1, "the stream took the warm socket");
    run.session.aclose().await;
    mux.aclose().await;
}

// Catches: a provider that connects but never says it is ready (no setup answer, no handshake)
// leaving the session waiting forever, and input or commands sent before it is ready.
#[tokio::test]
async fn a_provider_that_never_becomes_ready_fails_the_attempt_with_a_connect_timeout() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(setup_then_ready());
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("audio")).unwrap();
    until("the setup", || provider.count() == 1 && provider.socket(0).sent() == ["setup"]).await;
    let limit = options(0).connect_timeout.expect("a connect limit");
    crate::testkit::arm_then_advance(&clock, Duration::from_secs_f64(limit)).await;
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!(error.error_kind, ErrorKind::Timeout);
    assert_eq!(provider.socket(0).sent(), ["setup"], "nothing goes out before ready");
}

// Catches: input and commands sent ahead of the provider's readiness, or the setup not first.
#[tokio::test]
async fn input_waits_for_the_provider_to_be_ready_after_the_setup() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(setup_then_ready());
    let run = start(&protocol, &provider, &clock, 0);
    until("the setup", || provider.count() == 1 && provider.socket(0).sent() == ["setup"]).await;
    protocol.send_now("hello");
    input.send(item("audio")).unwrap();
    crate::testkit::arm_then_advance(&clock, Duration::ZERO).await;
    assert_eq!(provider.socket(0).sent(), ["setup"], "nothing goes out before ready");
    provider.socket(0).text("ready");
    until("the input", || provider.socket(0).sent() == ["setup", "hello", "audio"]).await;
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a provider that finishes its session after each request (its own "finished" message)
// failing the stream as a lost connection or ending it while input still comes, or the next input
// sent on the finished session.
#[tokio::test]
async fn a_finished_session_continues_the_input_on_a_new_connection() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("one")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["one"]).await;
    provider.socket(0).text("finished");
    input.send(item("two")).unwrap();
    until("the next input", || provider.count() == 2 && provider.socket(1).sent() == ["two"]).await;
    assert_eq!(provider.socket(0).sent(), ["one"]);
    assert!(provider.socket(0).closed.load(Ordering::SeqCst));
    assert_eq!(protocol.finished_sessions.load(Ordering::SeqCst), 1);
    assert!(protocol.retried.lock().is_empty(), "a finished session is not a failure");
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a provider that takes one request per session (the stream asked it to end the
// session; its last answer, then its close) read as a lost connection, or its last answer cut off.
#[tokio::test]
async fn a_close_the_stream_expected_finishes_the_session_and_continues_on_a_new_connection() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("end of request")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["end of request"])
        .await;
    protocol.expects_close.store(true, Ordering::SeqCst);
    provider.socket(0).text("last answer");
    provider.socket(0).close(Some(1000), None);
    until("the session finished", || protocol.finished_sessions.load(Ordering::SeqCst) == 1).await;
    protocol.expects_close.store(false, Ordering::SeqCst);
    input.send(item("two")).unwrap();
    until("the next input", || provider.count() == 2 && provider.socket(1).sent() == ["two"]).await;
    assert_eq!(*protocol.read.lock(), ["last answer"]);
    assert!(protocol.retried.lock().is_empty());
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a session the stream read as finished (a segment done with no message of the
// provider's) kept for the next item, which then reaches a provider session already over.
#[tokio::test]
async fn an_ended_session_sends_the_next_item_on_a_new_connection() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("one")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["one"]).await;
    run.session.end_session();
    input.send(item("two")).unwrap();
    until("the next item", || provider.count() == 2 && provider.socket(1).sent() == ["two"]).await;
    assert_eq!(provider.socket(0).sent(), ["one"]);
    assert_eq!(protocol.finished_sessions.load(Ordering::SeqCst), 1);
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a connection opened as soon as the provider finished a session, idling (and timing
// out at the provider) until the next input, or opened only to send an end that owes nothing.
#[tokio::test]
async fn after_a_finished_session_the_next_connection_opens_only_for_the_next_input() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(quiet());
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("one")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["one"]).await;
    provider.socket(0).text("finished");
    until("the session finished", || provider.socket(0).closed.load(Ordering::SeqCst)).await;
    crate::testkit::arm_then_advance(&clock, Duration::ZERO).await;
    assert_eq!(provider.count(), 1, "no connection before the next input");
    input.send(item("two")).unwrap();
    until("the next input", || provider.count() == 2 && provider.socket(1).sent() == ["two"]).await;
    provider.socket(1).text("finished");
    until("closed", || provider.socket(1).closed.load(Ordering::SeqCst)).await;
    input.send(end()).unwrap();
    run.ended().await.unwrap();
    assert_eq!(provider.count(), 2, "an end opens no connection of its own");
}

// Catches: a provider's inactivity rule (finalize after a silence) written as its own timer task,
// or its frames sent again and again while the input stays silent.
#[tokio::test]
async fn the_input_idle_frames_go_once_per_silence() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let (protocol, input) = Protocol::new(quiet());
    *protocol.input_idle.lock() = Some(2.0);
    let run = start(&protocol, &provider, &clock, 0);
    input.send(item("audio")).unwrap();
    until("sent", || provider.count() == 1 && provider.socket(0).sent() == ["audio"]).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(2)).await;
    until("the idle frames", || provider.socket(0).sent() == ["audio", "idle"]).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(2)).await;
    crate::testkit::arm_then_advance(&clock, Duration::ZERO).await;
    assert_eq!(provider.socket(0).sent(), ["audio", "idle"], "once per silence");
    input.send(item("more")).unwrap();
    until("more", || provider.socket(0).sent().len() == 3).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_secs(2)).await;
    until("idle again", || provider.socket(0).sent() == ["audio", "idle", "more", "idle"]).await;
    run.session.aclose().await;
    run.ended().await.unwrap();
}

// Catches: a provider's rate limit (one frame per 60 ms per key) broken by two streams sending at
// once, or a frame held back longer than the interval.
#[tokio::test]
async fn shared_pacing_spaces_the_frames_of_every_stream_using_it() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider::default();
    let pacing = Arc::new(Pacing(Mutex::new(Pacer::new(Some(Duration::from_millis(60)), false))));
    let mut runs = Vec::new();
    for _ in 0..2 {
        let (protocol, input) = Protocol::new(quiet());
        let host: Arc<dyn WebSocketHost> = Arc::clone(&protocol) as Arc<dyn WebSocketHost>;
        let session =
            Session::direct(host, options(0), "fake".into(), env(&clock), provider.transports())
                .paced(Arc::clone(&pacing));
        runs.push((input, Run::spawn(session)));
        until("connected", || provider.count() == runs.len()).await;
    }
    let sent = || (0..2).map(|i| provider.socket(i).sent().len()).sum::<usize>();
    for (input, _) in &runs {
        input.send(item("audio")).unwrap();
    }
    until("the first frame", || sent() == 1).await;
    crate::testkit::arm_then_advance(&clock, Duration::from_millis(59)).await;
    assert_eq!(sent(), 1, "the second waits out the interval");
    crate::testkit::arm_then_advance(&clock, Duration::from_millis(1)).await;
    until("the second frame", || sent() == 2).await;
    for (_, run) in runs {
        run.session.aclose().await;
        run.ended().await.unwrap();
    }
}

// Catches: the provider's request id (from the handshake, or named by a message) lost, so a
// failure cannot be traced on the provider's side.
#[tokio::test]
async fn failures_carry_the_providers_request_id() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider { headers: vec![("x-request-id", "req-1")], ..Provider::default() };
    let mut traced = connection();
    traced.target.request_id_header = Some("x-request-id".into());
    let (protocol, _input) = Protocol::new(traced);
    let run = start(&protocol, &provider, &clock, 0);
    until("connected", || protocol.opened.lock().len() == 1).await;
    assert_eq!(*protocol.opened.lock(), [Some("req-1".to_owned())]);
    provider.socket(0).text("request req-2");
    until("read", || protocol.read.lock().len() == 1).await;
    provider.socket(0).close(Some(1011), None);
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!((error.error_kind, error.request_id.as_deref()), (ErrorKind::Server, Some("req-2")));
}

// Catches: a rejected handshake's reason phrase or body (which can echo a key) put in the error's
// message, where every log and trace shows it, instead of its detail.
#[tokio::test]
async fn a_rejected_handshake_keeps_the_providers_words_out_of_the_message() {
    let clock = Arc::new(VirtualClock::new());
    let provider = Provider { refused: Some((401, "secret-reason")), ..Provider::default() };
    let (protocol, _input) = Protocol::new(connection());
    let run = start(&protocol, &provider, &clock, 0);
    let error = run.ended().await.unwrap_err().attempt_error();
    assert_eq!((error.status_code, error.error_kind), (Some(401), ErrorKind::Auth));
    assert!(!error.message.contains("secret-reason"), "{}", error.message);
    assert_eq!(error.said.and_then(|said| said.detail).as_deref(), Some("secret-reason"));
}

#[path = "executor_tests.rs"]
mod executor_tests;

#[path = "mux_tests.rs"]
mod mux_tests;
