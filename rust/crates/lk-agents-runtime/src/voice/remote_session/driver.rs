//! The session host as a running thing: loops, in-flight table and drain
//! (`voice/remote_session.py`'s `SessionHost`). Tasks go through the injected [`Spawner`]; every
//! wait is bounded on the session clock.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::voice::session::Timebase;
use lk_agents_abi::Spawner;
use lk_agents_abi::errors::HostError;
use lk_agents_net::task::TaskSet;
use tokio::sync::mpsc;

use super::{
    ConsoleSink, NonFiniteStamp, SessionHost, SessionPort, SessionTransport, WireEvent, proto,
    request_name, stamp_event,
};
use proto::agent_session_message::Message;

/// The log target every line here lands on (the host maps it to its logger name).
const LOG: &str = "livekit.agents";

/// `SessionHost` (`remote_session.py:321`), running. Events queue from construction and one writer
/// drains them in order; each request is a task named by its type.
pub struct SessionServer {
    shared: Arc<HostShared>,
}

struct HostShared {
    time: Timebase,
    spawner: Arc<dyn Spawner>,
    events: Mutex<Option<mpsc::UnboundedSender<proto::AgentSessionMessage>>>,
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<proto::AgentSessionMessage>>>,
    state: Mutex<HostState>,
    handlers: Arc<TaskSet>,
    recv: Arc<TaskSet>,
    writer: Arc<TaskSet>,
}

#[derive(Default)]
struct HostState {
    started: bool,
    host: Option<Arc<SessionHost>>,
    transport: Option<Arc<dyn SessionTransport>>,
}

/// `voice/remote_session.py`'s `SessionHost`, running in Rust. Events queue from construction.
impl SessionServer {
    pub fn new(time: Timebase, spawner: Arc<dyn Spawner>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        SessionServer {
            shared: Arc::new(HostShared {
                time,
                spawner,
                events: Mutex::new(Some(tx)),
                events_rx: Mutex::new(Some(rx)),
                state: Mutex::new(HostState::default()),
                handlers: Arc::default(),
                recv: Arc::default(),
                writer: Arc::default(),
            }),
        }
    }
    pub fn started(&self) -> bool {
        self.shared.state.lock().expect("host state").started
    }

    /// `register_session`; until set, requests get no response.
    pub fn set_port(&self, port: Arc<dyn SessionPort>) {
        self.shared.state.lock().expect("host state").host =
            Some(Arc::new(SessionHost::new(port, self.shared.time.clone())));
    }

    /// The request types in flight, one per running handler.
    pub fn in_flight(&self) -> Vec<String> {
        self.shared.handlers.names()
    }

    /// Queue one event for the writer. After the queue is closed (`aclose`), dropped.
    pub fn send_message(&self, msg: proto::AgentSessionMessage) {
        if let Some(tx) = self.shared.events.lock().expect("event queue").as_ref() {
            let _ = tx.send(msg);
        }
    }

    /// `_send_event(event, created_at)`: stamp and queue an encoded event.
    pub fn send_event(
        &self,
        event: proto::AgentSessionEvent,
        created_at: f64,
    ) -> Result<(), NonFiniteStamp> {
        self.send_message(stamp_event(event, created_at)?);
        Ok(())
    }

    /// Encode one session event from the facts the host reports and queue it, stamped `created_at`
    /// (now when `None`). `false` when the event has no wire form (a tool status outside the closed
    /// set), which the reference drops.
    pub fn send_wire(&self, event: WireEvent, created_at: Option<f64>) -> Result<bool, String> {
        let Some(ev) = super::encode_wire(event)? else { return Ok(false) };
        let at = created_at.unwrap_or_else(|| self.shared.time.wall_secs());
        let ev = proto::AgentSessionEvent { created_at: None, event: Some(ev) };
        self.send_event(ev, at).map_err(|e| e.to_string())?;
        Ok(true)
    }

    /// `start()`: transport, then writer and receive loop. Idempotent; a failed start still counts
    /// as started, as the reference. `console` is where the console messages go.
    pub async fn start(
        &self,
        transport: Arc<dyn SessionTransport>,
        console: Option<Arc<dyn ConsoleSink>>,
    ) -> Result<(), HostError> {
        {
            let mut st = self.shared.state.lock().expect("host state");
            if st.started {
                return Ok(());
            }
            st.started = true;
            st.transport = Some(Arc::clone(&transport));
        }
        transport.start().await?;
        let sh = &self.shared;
        // Closed while the transport started: nothing is left to run.
        if !self.started() {
            return Ok(());
        }
        if let Some(rx) = sh.events_rx.lock().expect("event queue").take() {
            sh.writer.spawn(
                &*sh.spawner,
                "SessionHost._write_loop",
                write_loop(rx, Arc::clone(&transport)),
            );
        }
        sh.recv.spawn(
            &*sh.spawner,
            "SessionHost._recv_loop",
            recv_loop(Arc::clone(sh), transport, console),
        );
        Ok(())
    }

    /// `SessionHost.aclose()`: the receive loop cancelled, in-flight handlers drained, the writer
    /// flushed, then the transport closed, all within one shared `drain` budget. Returns the
    /// request types cancelled at the deadline, sorted and distinct.
    pub async fn aclose(&self, drain: Duration) -> Vec<String> {
        if !self.started() {
            return Vec::new();
        }
        let sh = &self.shared;
        let clock = sh.time.clock();
        let clock = &*clock;
        // First, so no new event is queued during the rest.
        sh.state.lock().expect("host state").started = false;
        let deadline = clock.now().saturating_add(drain);
        // Before the drain, so no new request handler starts.
        sh.recv.cancel_and_wait(clock).await;
        // Not a cancel: that would drop responses of requests that already did their work.
        let mut cancelled = sh.handlers.drain(clock, deadline).await;
        // After the drain: handlers' events on their way out are still sent.
        sh.events.lock().expect("event queue").take();
        sh.events_rx.lock().expect("event queue").take();
        // Let the writer flush what is queued, then exit.
        sh.writer.drain(clock, deadline).await;
        // Last: everything above still needs to write.
        let transport = sh.state.lock().expect("host state").transport.take();
        if let Some(t) = transport {
            t.close().await;
        }
        cancelled.sort();
        cancelled.dedup();
        cancelled
    }
}

/// How long the host's close waits for in-flight requests (`_SHUTDOWN_DRAIN_TIMEOUT`), the
/// writer's flush included: well under the worker's shutdown allowance.
pub const SHUTDOWN_DRAIN: Duration = Duration::from_secs(3);

/// A session's remote session protocol, as the session runs it: the host on its transport (the
/// room's byte streams, or the console's socket), fed the session's events as they are emitted,
/// started once the session is up and closed with it.
pub struct RemoteSessionLink {
    host: Arc<SessionServer>,
    spawner: Arc<dyn Spawner>,
    /// Set by the first close: a start after it (the session closed while opening) is a no-op.
    closing: std::sync::atomic::AtomicBool,
    /// Whether the close has run to its end; every close waits for it.
    closed: tokio::sync::watch::Sender<bool>,
    transport: Arc<dyn SessionTransport>,
    console: Option<Arc<dyn ConsoleSink>>,
    /// The close, which a cancelled caller leaves to finish.
    tasks: Arc<TaskSet>,
}

impl RemoteSessionLink {
    pub fn new(
        port: Arc<dyn SessionPort>,
        transport: Arc<dyn SessionTransport>,
        console: Option<Arc<dyn ConsoleSink>>,
        time: Timebase,
        spawner: Arc<dyn Spawner>,
    ) -> Self {
        let host = Arc::new(SessionServer::new(time, Arc::clone(&spawner)));
        host.set_port(port);
        RemoteSessionLink {
            host,
            spawner,
            closing: Default::default(),
            closed: tokio::sync::watch::Sender::new(false),
            transport,
            console,
            tasks: Arc::default(),
        }
    }

    /// One of the session's events, sent if the client is sent it.
    pub fn forward(&self, wire: &lk_agents_abi::voice::events::HostWire) {
        if let Some((event, at)) = super::session_event(wire) {
            self.send(event, at);
        }
    }

    /// `session_usage_updated`: the session's usage moved.
    pub fn forward_usage(&self, usage: &lk_agents_abi::usage::Usage) {
        use lk_agents_abi::worker::remote_session::SessionUsageUpdatedWire;
        self.send(
            WireEvent::SessionUsageUpdated(SessionUsageUpdatedWire { usage: usage.clone() }),
            None,
        );
    }

    /// `error`, as the host words it.
    pub fn forward_error(&self, message: String) {
        use lk_agents_abi::worker::remote_session::ErrorWire;
        self.send(WireEvent::Error(ErrorWire { message }), None);
    }

    /// `function_tools_executed`, emitted by the host's hook rather than as an event.
    pub fn forward_tools_executed(&self, ev: &lk_agents_abi::voice::events::ToolsExecutedEvent) {
        self.send(super::tools_executed(ev), None);
    }

    fn send(&self, event: WireEvent, at: Option<f64>) {
        if let Err(e) = self.host.send_wire(event, at) {
            log::error!(target: LOG, error = e.as_str();
                "failed to forward a session event to the remote client");
        }
    }

    pub async fn start(&self) -> Result<(), HostError> {
        if self.closing.load(Ordering::Acquire) {
            return Ok(());
        }
        self.host.start(Arc::clone(&self.transport), self.console.clone()).await
    }

    /// Once the session's loop has returned. The close runs on a task of its own, so a caller
    /// cancelled midway leaves it to finish; every call waits for its end.
    pub async fn aclose(&self) {
        if !self.closing.swap(true, Ordering::AcqRel) {
            let (host, closed) = (Arc::clone(&self.host), self.closed.clone());
            self.tasks.spawn(self.spawner.as_ref(), "remote session close", async move {
                let cancelled = host.aclose(SHUTDOWN_DRAIN).await;
                if !cancelled.is_empty() {
                    log::warn!(target: LOG, request_types = cancelled.join(", ").as_str();
                        "cancelled in-flight session requests while closing");
                }
                closed.send_replace(true);
            });
        }
        let _ = self.closed.subscribe().wait_for(|closed| *closed).await;
    }
}

async fn write_loop(
    mut rx: mpsc::UnboundedReceiver<proto::AgentSessionMessage>,
    transport: Arc<dyn SessionTransport>,
) {
    // nobody awaits an event, so a failed one is logged rather than raised
    while let Some(msg) = rx.recv().await {
        if let Err(e) = transport.send_message(&msg).await {
            log::warn!(target: LOG, error = e.to_string().as_str(); "failed to send session event");
        }
    }
}

async fn recv_loop(
    sh: Arc<HostShared>,
    transport: Arc<dyn SessionTransport>,
    console: Option<Arc<dyn ConsoleSink>>,
) {
    loop {
        let msg = match transport.recv().await {
            Ok(Some(msg)) => msg,
            Ok(None) => return,
            Err(e) => {
                log::warn!(target: LOG, error = e.to_string().as_str(); "error processing session message");
                return;
            }
        };
        // `_dispatch_transport_message` (`remote_session.py:454`): requests, then the console legs.
        match (msg.message, console.as_deref()) {
            (Some(Message::Request(req)), _) => {
                let host = sh.state.lock().expect("host state").host.clone();
                if let Some(host) = host {
                    let name = req.request.as_ref().map(request_name).unwrap_or("unknown");
                    sh.handlers.spawn(
                        &*sh.spawner,
                        name,
                        handle_request(host, Arc::clone(&transport), req),
                    );
                }
            }
            (Some(Message::AudioInput(frame)), Some(console)) => console.push_input_frame(&frame),
            (Some(Message::AudioPlaybackFinished(_)), Some(console)) => console.playout_finished(),
            _ => {}
        }
    }
}

/// `_handle_request_safe`: a failed handler still owes the opaque `"internal error"` response.
async fn handle_request(
    host: Arc<SessionHost>,
    transport: Arc<dyn SessionTransport>,
    req: proto::SessionRequest,
) {
    let outcome = match host.handle_request(&req).await {
        Ok(Some(resp)) => transport.send_message(&resp).await,
        Ok(None) => {
            log::warn!(target: LOG, request_id = req.request_id.as_str(); "unknown session request");
            Ok(())
        }
        Err(e) => Err(e),
    };
    if let Err(e) = outcome {
        log::warn!(target: LOG, request_id = req.request_id.as_str(), error = e.to_string().as_str();
            "error handling session request");
        let _ = transport.send_message(&super::internal_error_response(&req.request_id)).await;
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::Notify;

    use super::super::client::{ReadyError, RemoteSession, RequestError};
    use super::super::{
        AgentInfo, ClientSink, IoUpdate, RtcStats, RunOutcome, SessionState, SimulationVerdict,
        VerdictOutcome,
    };
    use super::*;
    use lk_agents_abi::secs;
    use lk_agents_abi::{ChatItem, Clock};
    use std::sync::atomic::AtomicBool;
    use tokio::sync::Mutex as AsyncMutex;

    /// Two linked transports: what one sends, the other receives.
    struct Pipe {
        tx: mpsc::UnboundedSender<proto::AgentSessionMessage>,
        rx: AsyncMutex<mpsc::UnboundedReceiver<proto::AgentSessionMessage>>,
        fail: AtomicBool,
        closed: AtomicBool,
        sent: Mutex<Vec<proto::AgentSessionMessage>>,
    }

    impl SessionTransport for Pipe {
        fn start(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { Ok(()) })
        }
        fn send_message<'a>(
            &'a self,
            msg: &'a proto::AgentSessionMessage,
        ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
            Box::pin(async move {
                if self.fail.load(Ordering::SeqCst) {
                    return Err(HostError::new("transport is down"));
                }
                self.sent.lock().unwrap().push(msg.clone());
                let _ = self.tx.send(msg.clone());
                Ok(())
            })
        }
        fn recv(
            &self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<proto::AgentSessionMessage>, HostError>>
        {
            Box::pin(async move { Ok(self.rx.lock().await.recv().await) })
        }
        fn close(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
            Box::pin(async move {
                self.closed.store(true, Ordering::SeqCst);
            })
        }
    }

    fn pair() -> (Arc<Pipe>, Arc<Pipe>) {
        let (a_tx, a_rx) = mpsc::unbounded_channel();
        let (b_tx, b_rx) = mpsc::unbounded_channel();
        let pipe = |tx, rx| {
            Arc::new(Pipe {
                tx,
                rx: AsyncMutex::new(rx),
                fail: AtomicBool::new(false),
                closed: AtomicBool::new(false),
                sent: Mutex::new(Vec::new()),
            })
        };
        (pipe(b_tx, a_rx), pipe(a_tx, b_rx))
    }

    /// A port whose `run_input` waits for a release, and whose history read fails.
    struct Port {
        release: Arc<Notify>,
    }

    impl SessionPort for Port {
        fn chat_history(&self) -> lk_agents_abi::BoxFuture<'_, Result<Vec<ChatItem>, HostError>> {
            Box::pin(async move { Err(HostError::new("no history here")) })
        }
        fn agent_info(&self) -> lk_agents_abi::BoxFuture<'_, Result<AgentInfo, HostError>> {
            Box::pin(async move { Ok(("agent-1".into(), None, vec![], vec![])) })
        }
        fn run_input<'a>(
            &'a self,
            _text: &'a str,
        ) -> lk_agents_abi::BoxFuture<'a, Result<RunOutcome, HostError>> {
            Box::pin(async move {
                self.release.notified().await;
                Ok((Some(vec![]), None))
            })
        }
        fn session_state(&self) -> lk_agents_abi::BoxFuture<'_, Result<SessionState, HostError>> {
            Box::pin(async move {
                Ok((
                    "not-a-state".into(),
                    "away".into(),
                    "agent-1".into(),
                    Default::default(),
                    None,
                ))
            })
        }
        fn rtc_stats(&self) -> lk_agents_abi::BoxFuture<'_, Result<Option<RtcStats>, HostError>> {
            Box::pin(async move { Ok(None) })
        }
        fn session_usage(
            &self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<lk_agents_abi::usage::Usage, HostError>> {
            Box::pin(async move { Ok(Default::default()) })
        }
        fn framework_info(&self) -> (String, String) {
            ("rust".into(), "0".into())
        }
        fn update_io(
            &self,
            _update: IoUpdate,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
            Box::pin(async move { Ok(()) })
        }
        fn finalize_simulation(
            &self,
            _p: SimulationVerdict,
        ) -> lk_agents_abi::BoxFuture<'_, Result<VerdictOutcome, HostError>> {
            Box::pin(async move { Ok((None, None)) })
        }
    }

    struct Events(Mutex<Vec<String>>);
    impl ClientSink for Events {
        fn event(&self, name: &str, _event: &proto::AgentSessionEvent) {
            self.0.lock().unwrap().push(name.to_owned());
        }
    }

    fn rt() -> (Arc<dyn Clock>, Arc<dyn Spawner>) {
        (
            Arc::new(lk_agents_net::clock::RealClock::default()),
            Arc::new(lk_agents_net::task::TokioSpawner::current()),
        )
    }

    async fn setup(release: Arc<Notify>) -> (SessionServer, RemoteSession, Arc<Pipe>, Arc<Events>) {
        let (ht, ct) = pair();
        let (clock, spawner) = rt();
        let host = SessionServer::new(Timebase::of_clock(Arc::clone(&clock)), Arc::clone(&spawner));
        host.set_port(Arc::new(Port { release }));
        host.start(ht.clone(), None).await.unwrap();
        let client = RemoteSession::new(clock, spawner);
        let events = Arc::new(Events(Mutex::new(Vec::new())));
        client.start(ct, events.clone()).await.unwrap();
        (host, client, ht, events)
    }

    async fn nap(d: Duration) {
        let clock = lk_agents_net::clock::RealClock::default();
        clock.wait_until(clock.now().saturating_add(d)).await;
    }

    #[tokio::test]
    async fn replies_and_default_events_use_the_supplied_session_wall_clock() {
        let clock = Arc::new(lk_agents_abi::VirtualClock::new());
        let time = Timebase::from_virtual(clock);
        time.advance(Duration::from_secs(17));
        let host =
            SessionHost::new(Arc::new(Port { release: Arc::new(Notify::new()) }), time.clone());
        for request in [
            proto::session_request::Request::GetSessionState(Default::default()),
            proto::session_request::Request::GetSessionUsage(Default::default()),
        ] {
            let request =
                proto::SessionRequest { request_id: "clock".into(), request: Some(request) };
            let before = time.wall_secs();
            let response = host.handle_request(&request).await.unwrap().unwrap();
            let after = time.wall_secs();
            let Some(Message::Response(response)) = response.message else {
                panic!("response expected")
            };
            let stamp = match response.response.unwrap() {
                proto::session_response::Response::GetSessionState(state) => state.created_at,
                proto::session_response::Response::GetSessionUsage(usage) => usage.created_at,
                _ => panic!("unexpected response"),
            }
            .unwrap();
            let stamped = stamp.seconds as f64 + f64::from(stamp.nanos) / 1e9;
            assert!((before..=after).contains(&stamped), "{stamped} outside {before}..={after}");
        }
        let (transport, _peer) = pair();
        let runtime = SessionServer::new(
            time.clone(),
            Arc::new(lk_agents_net::task::TokioSpawner::current()),
        );
        let before = time.wall_secs();
        runtime
            .send_wire(
                WireEvent::Error(lk_agents_abi::worker::remote_session::ErrorWire {
                    message: "clock".into(),
                }),
                None,
            )
            .unwrap();
        runtime.start(transport.clone(), None).await.unwrap();
        runtime.aclose(Duration::from_secs(1)).await;
        let sent = transport.sent.lock().unwrap();
        let Some(Message::Event(event)) = &sent[0].message else { panic!("event expected") };
        let stamp = event.created_at.as_ref().unwrap();
        let after = time.wall_secs();
        let stamped = stamp.seconds as f64 + f64::from(stamp.nanos) / 1e9;
        assert!((before..=after).contains(&stamped), "{stamped} outside {before}..={after}");
    }

    #[tokio::test]
    async fn empty_run_input_never_enters_the_session_port() {
        use std::future::Future as _;
        let (clock, _) = rt();
        let host = SessionHost::new(
            Arc::new(Port { release: Arc::new(Notify::new()) }),
            Timebase::of_clock(clock),
        );
        let request = proto::SessionRequest {
            request_id: "empty".into(),
            request: Some(proto::session_request::Request::RunInput(Default::default())),
        };
        let mut reply = Box::pin(host.handle_request(&request));
        let ready =
            std::future::poll_fn(|cx| std::task::Poll::Ready(reply.as_mut().poll(cx))).await;
        let std::task::Poll::Ready(Ok(Some(reply))) = ready else {
            panic!("empty input entered the blocking session port")
        };
        let Some(Message::Response(reply)) = reply.message else { panic!("response expected") };
        assert_eq!(reply.error.as_deref(), Some("empty run_input text"));
    }

    // Catches: a failing port read left unanswered, or the exception text leaking to the peer.
    #[tokio::test]
    async fn a_request_whose_port_fails_is_answered_with_an_opaque_internal_error() {
        let (host, client, _, _) = setup(Arc::new(Notify::new())).await;
        let history = proto::session_request::Request::GetChatHistory(Default::default());
        let err = client.request(history, secs(2.0)).await.unwrap_err();
        assert_eq!(
            err,
            RequestError::Failed("session request get_chat_history failed: internal error".into())
        );
        client.aclose().await;
        host.aclose(secs(1.0)).await;
    }

    // Catches: an unknown state literal not defaulting to AS_IDLE, or a known one defaulted.
    #[tokio::test]
    async fn session_state_defaults_an_unknown_literal_and_keeps_a_known_one() {
        let (host, client, _, _) = setup(Arc::new(Notify::new())).await;
        let req = proto::session_request::Request::GetSessionState(Default::default());
        let resp = client.request(req, secs(2.0)).await.unwrap();
        let Some(proto::session_response::Response::GetSessionState(st)) = resp.response else {
            panic!("{resp:?}")
        };
        assert_eq!(st.agent_state, proto::AgentState::AsIdle as i32);
        assert_eq!(st.user_state, proto::UserState::UsAway as i32);
        assert!(
            st.created_at.unwrap().seconds > 0,
            "an unset start is stamped with the answer time"
        );
        client.aclose().await;
        host.aclose(secs(1.0)).await;
    }

    // Catches: aclose cancelling in-flight handlers instead of draining them, or a straggler not
    // reported by its type.
    #[tokio::test]
    async fn aclose_drains_a_handler_that_finishes_and_names_one_that_does_not() {
        let release = Arc::new(Notify::new());
        let (host, client, _, _) = setup(Arc::clone(&release)).await;
        let client = Arc::new(client);
        let run = || {
            proto::session_request::Request::RunInput(proto::session_request::RunInput {
                text: "hi".into(),
            })
        };
        let c = Arc::clone(&client);
        let r = run();
        let answered = tokio::spawn(async move { c.request(r, secs(3.0)).await });
        while host.in_flight().is_empty() {
            nap(secs(0.005)).await;
        }
        let rel = Arc::clone(&release);
        tokio::spawn(async move {
            nap(secs(0.05)).await;
            rel.notify_waiters();
        });
        assert!(host.aclose(secs(1.0)).await.is_empty());
        assert!(answered.await.unwrap().is_ok(), "the response of a finished run was dropped");
        client.aclose().await;

        let (host, client, _, _) = setup(Arc::new(Notify::new())).await;
        let client = Arc::new(client);
        let c = Arc::clone(&client);
        let r = run();
        tokio::spawn(async move { c.request(r, secs(3.0)).await });
        while host.in_flight().is_empty() {
            nap(secs(0.005)).await;
        }
        assert_eq!(host.aclose(secs(0.05)).await, ["run_input"]);
        assert!(host.in_flight().is_empty());
        client.aclose().await;
    }

    // Catches: events out of order, dropped before start, lost at close; a closed host accepting events.
    #[tokio::test]
    async fn events_queue_before_start_leave_in_order_and_flush_at_close() {
        let (ht, _peer) = pair();
        let (clock, spawner) = rt();
        let host = SessionServer::new(Timebase::of_clock(clock), spawner);
        let ev = |i: usize| proto::AgentSessionEvent {
            created_at: None,
            event: Some(proto::agent_session_event::Event::Error(
                proto::agent_session_event::Error { message: i.to_string() },
            )),
        };
        host.send_event(ev(0), 1.0).unwrap();
        host.start(ht.clone(), None).await.unwrap();
        for i in 1..10 {
            host.send_event(ev(i), 1.0).unwrap();
        }
        assert!(host.send_event(ev(99), f64::NAN).is_err(), "a non-finite stamp is refused");
        host.aclose(secs(1.0)).await;
        host.send_event(ev(100), 1.0).unwrap();
        let sent: Vec<String> = ht
            .sent
            .lock()
            .unwrap()
            .iter()
            .map(|m| match &m.message {
                Some(proto::agent_session_message::Message::Event(e)) => match &e.event {
                    Some(proto::agent_session_event::Event::Error(x)) => x.message.clone(),
                    _ => panic!(),
                },
                _ => panic!(),
            })
            .collect();
        assert_eq!(sent, (0..10).map(|i| i.to_string()).collect::<Vec<_>>());
        assert!(ht.closed.load(Ordering::SeqCst), "the transport is closed last");
    }

    // Catches: a send failure reported as a bare timeout, or a late transport not polled through.
    #[tokio::test]
    async fn wait_for_ready_raises_the_latched_send_error_and_polls_through_a_late_transport() {
        let (ht, ct) = pair();
        let (clock, spawner) = rt();
        let client = RemoteSession::new(clock.clone(), spawner.clone());
        ct.fail.store(true, Ordering::SeqCst);
        client.start(ct.clone(), Arc::new(Events(Mutex::new(vec![])))).await.unwrap();
        let err = client.wait_for_ready(secs(0.2), secs(0.05)).await.unwrap_err();
        assert_eq!(err, ReadyError::Transport(HostError::new("transport is down")));

        let host = SessionServer::new(Timebase::of_clock(clock), spawner);
        host.set_port(Arc::new(Port { release: Arc::new(Notify::new()) }));
        host.start(ht, None).await.unwrap();
        let late = Arc::clone(&ct);
        tokio::spawn(async move {
            nap(secs(0.1)).await;
            late.fail.store(false, Ordering::SeqCst);
        });
        client.wait_for_ready(secs(2.0), secs(0.05)).await.unwrap();
        client.aclose().await;
        host.aclose(secs(1.0)).await;
    }

    // Catches: a timed-out request leaving its slot, two requests sharing an id, or a close
    // leaving a pending caller hanging.
    #[tokio::test]
    async fn the_client_times_out_mints_distinct_ids_and_cancels_at_close() {
        let (_ht, ct) = pair();
        let (clock, spawner) = rt();
        let client = Arc::new(RemoteSession::new(clock, spawner));
        let events = Arc::new(Events(Mutex::new(vec![])));
        client.start(ct.clone(), events).await.unwrap();
        let ping = || proto::session_request::Request::Ping(Default::default());
        assert_eq!(client.request(ping(), secs(0.05)).await.unwrap_err(), RequestError::Timeout);
        assert!(
            client.shared.pending.lock().unwrap().is_empty(),
            "a timed-out request kept its slot"
        );

        let c = Arc::clone(&client);
        let waiting = tokio::spawn(async move { c.request(ping(), secs(5.0)).await });
        while ct.sent.lock().unwrap().len() < 2 {
            nap(secs(0.005)).await;
        }
        client.aclose().await;
        assert_eq!(waiting.await.unwrap().unwrap_err(), RequestError::Cancelled);
        let ids: Vec<String> = ct
            .sent
            .lock()
            .unwrap()
            .iter()
            .map(|m| match &m.message {
                Some(proto::agent_session_message::Message::Request(r)) => r.request_id.clone(),
                _ => panic!(),
            })
            .collect();
        assert_ne!(ids[0], ids[1]);
        assert!(ids.iter().all(|i| i.starts_with("req_") && i.len() == 15), "{ids:?}");
    }
}

impl lk_agents_abi::worker::remote_session::SessionHosting for RemoteSessionLink {
    fn forward(&self, event: &lk_agents_abi::voice::events::HostWire) {
        RemoteSessionLink::forward(self, event);
    }

    fn forward_usage(&self, usage: &lk_agents_abi::usage::Usage) {
        RemoteSessionLink::forward_usage(self, usage);
    }

    fn forward_error(&self, message: String) {
        RemoteSessionLink::forward_error(self, message);
    }

    fn forward_tools_executed(&self, event: &lk_agents_abi::voice::events::ToolsExecutedEvent) {
        RemoteSessionLink::forward_tools_executed(self, event);
    }

    fn start(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(RemoteSessionLink::start(self))
    }

    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(RemoteSessionLink::aclose(self))
    }
}
