//! The session host as a running thing: loops, in-flight table and drain
//! (`voice/remote_session.py`'s `SessionHost`). Tasks go through the injected [`Spawner`]; every
//! wait is bounded on the session clock.

use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::voice::session::Timebase;
use lk_agents_abi::Spawner;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::remote_session::RemoteSessionRequestKind;
use lk_agents_net::task::TaskSet;
use tokio::sync::mpsc;

use super::{
    ConsoleSink, NonFiniteStamp, SessionHost, SessionPort, SessionTransport, WireEvent, proto,
    request_kind, stamp_event,
};
use proto::agent_session_message::Message;

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
        self.shared.state.lock().started
    }

    /// `register_session`; until set, requests get no response.
    pub fn set_port(&self, port: Arc<dyn SessionPort>) {
        self.shared.state.lock().host =
            Some(Arc::new(SessionHost::new(port, self.shared.time.clone())));
    }

    /// The request types in flight, one per running handler.
    pub fn in_flight(&self) -> Vec<String> {
        self.shared.handlers.names()
    }

    /// Queue one event for the writer. After the queue is closed (`aclose`), dropped.
    pub fn send_message(&self, msg: proto::AgentSessionMessage) {
        if let Some(tx) = self.shared.events.lock().as_ref() {
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
    /// (now when `None`). An event with no wire form (a tool status outside the closed set) is
    /// dropped, as the reference does; one that cannot be encoded is logged.
    pub fn send_wire(&self, event: WireEvent, created_at: Option<f64>) {
        let unsent = |error: &dyn std::fmt::Display| {
            log::error!(target: lk_agents_abi::LOG_TARGET, error = error.to_string().as_str();
                "failed to forward a session event to the remote client");
        };
        let ev = match super::encode_wire(event) {
            Ok(Some(ev)) => ev,
            Ok(None) => return,
            Err(error) => return unsent(&error),
        };
        let at = created_at.unwrap_or_else(|| self.shared.time.wall_secs());
        let ev = proto::AgentSessionEvent { created_at: None, event: Some(ev) };
        if let Err(error) = self.send_event(ev, at) {
            unsent(&error);
        }
    }

    /// `start()`: transport, then writer and receive loop. Idempotent; a failed start still counts
    /// as started, as the reference. `console` is where the console messages go.
    pub async fn start(
        &self,
        transport: Arc<dyn SessionTransport>,
        console: Option<Arc<dyn ConsoleSink>>,
    ) -> Result<(), HostError> {
        {
            let mut st = self.shared.state.lock();
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
        if let Some(rx) = sh.events_rx.lock().take() {
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
        sh.state.lock().started = false;
        let deadline = clock.now().saturating_add(drain);
        // Before the drain, so no new request handler starts.
        sh.recv.cancel_and_wait(clock).await;
        // Not a cancel: that would drop responses of requests that already did their work.
        let mut cancelled = sh.handlers.drain(clock, deadline).await;
        // After the drain: handlers' events on their way out are still sent.
        sh.events.lock().take();
        sh.events_rx.lock().take();
        // Let the writer flush what is queued, then exit.
        sh.writer.drain(clock, deadline).await;
        // Last: everything above still needs to write.
        let transport = sh.state.lock().transport.take();
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
        self.host.send_wire(event, at);
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
                    log::warn!(target: lk_agents_abi::LOG_TARGET, request_types = cancelled.join(", ").as_str();
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
            log::warn!(target: lk_agents_abi::LOG_TARGET, error = e.to_string().as_str(); "failed to send session event");
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
                log::warn!(target: lk_agents_abi::LOG_TARGET, error = e.to_string().as_str(); "error processing session message");
                return;
            }
        };
        // `_dispatch_transport_message` (`remote_session.py:454`): requests, then the console legs.
        match (msg.message, console.as_deref()) {
            (Some(Message::Request(req)), _) => {
                let host = sh.state.lock().host.clone();
                if let Some(host) = host {
                    let kind = req.request.as_ref().map(request_kind);
                    let name = kind.map_or("unknown", RemoteSessionRequestKind::as_str);
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
            log::warn!(target: lk_agents_abi::LOG_TARGET, request_id = req.request_id.as_str(); "unknown session request");
            Ok(())
        }
        Err(e) => Err(e),
    };
    if let Err(e) = outcome {
        log::warn!(target: lk_agents_abi::LOG_TARGET, request_id = req.request_id.as_str(), error = e.to_string().as_str();
            "error handling session request");
        if let Err(error) =
            transport.send_message(&super::internal_error_response(&req.request_id)).await
        {
            log::warn!(target: lk_agents_abi::LOG_TARGET, error:% = error, request_id = req.request_id.as_str(); "remote session error response could not be sent");
        }
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

#[cfg(test)]
#[path = "driver_tests.rs"]
mod tests;
