//! The session host and the remote client as running things: loops, in-flight table, drain,
//! correlation and timeouts (`voice/remote_session.py`'s `SessionHost` and `RemoteSession`).
//! Tasks go through the injected [`Spawner`]; every wait is bounded on the injected [`Clock`].

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::{Clock, Spawner, SpawnerExt, Ts};
use tokio::sync::{Notify, mpsc, oneshot};
use lk_agents_abi::AbortHandle;

use super::{
    ClientSink, ConsoleSink, NonFiniteStamp, SessionHost, SessionPort, SessionTransport, WireEvent, proto,
    request_name, stamp_event,
};
use proto::agent_session_message::Message;

/// The Python logger every line here lands on (`pyo3-log` maps the target to the logger name).
const LOG: &str = "livekit.agents";

/// Named tasks and a way to wait for them to be gone (`utils.aio.TaskSet`). Entries are inserted
/// before spawning and removed by a guard the task owns.
#[derive(Default)]
struct TaskTable {
    inner: Mutex<TaskEntries>,
    changed: Notify,
}

#[derive(Default)]
struct TaskEntries {
    next: u64,
    tasks: BTreeMap<u64, (String, Option<AbortHandle>)>,
}

struct Leave {
    table: Arc<TaskTable>,
    id: u64,
}

impl Drop for Leave {
    fn drop(&mut self) {
        self.table.inner.lock().expect("task table").tasks.remove(&self.id);
        self.table.changed.notify_waiters();
    }
}

impl TaskTable {
    fn spawn(
        self: &Arc<Self>,
        spawner: &dyn Spawner,
        name: &str,
        fut: impl Future<Output = ()> + Send + 'static,
    ) {
        let id = {
            let mut g = self.inner.lock().expect("task table");
            g.next += 1;
            let id = g.next;
            g.tasks.insert(id, (name.to_owned(), None));
            id
        };
        let leave = Leave { table: Arc::clone(self), id };
        let handle = spawner.spawn_unboxed(async move {
            let _leave = leave;
            fut.await;
        });
        if let Some(entry) = self.inner.lock().expect("task table").tasks.get_mut(&id) {
            entry.1 = Some(handle);
        }
    }

    fn names(&self) -> Vec<String> {
        self.inner.lock().expect("task table").tasks.values().map(|(n, _)| n.clone()).collect()
    }

    fn is_empty(&self) -> bool {
        self.inner.lock().expect("task table").tasks.is_empty()
    }

    /// `true` when the table emptied.
    async fn wait_empty(&self, clock: &dyn Clock, deadline: Option<Ts>) -> bool {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_empty() {
                return true;
            }
            match deadline {
                None => notified.await,
                Some(at) => {
                    tokio::select! {
                        _ = &mut notified => {}
                        _ = clock.wait_until(at) => return self.is_empty(),
                    }
                }
            }
        }
    }

    /// `cancel_and_wait(*tasks)`; returns the names aborted.
    async fn cancel_and_wait(&self, clock: &dyn Clock) -> Vec<String> {
        let names = {
            let g = self.inner.lock().expect("task table");
            for (_, handle) in g.tasks.values() {
                if let Some(h) = handle {
                    h.abort();
                }
            }
            g.tasks.values().map(|(n, _)| n.clone()).collect()
        };
        self.wait_empty(clock, None).await;
        names
    }

    /// `_drain(tasks, deadline)`: wait until `deadline`, then cancel the stragglers and name them.
    async fn drain(&self, clock: &dyn Clock, deadline: Ts) -> Vec<String> {
        if self.wait_empty(clock, Some(deadline)).await {
            return Vec::new();
        }
        self.cancel_and_wait(clock).await
    }
}

/// `SessionHost` (`remote_session.py:321`), running. Events queue from construction and one writer
/// drains them in order; each request is a task named by its type.
pub struct HostRuntime {
    shared: Arc<HostShared>,
}

struct HostShared {
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    events: Mutex<Option<mpsc::UnboundedSender<proto::AgentSessionMessage>>>,
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<proto::AgentSessionMessage>>>,
    state: Mutex<HostState>,
    handlers: Arc<TaskTable>,
    recv: Arc<TaskTable>,
    writer: Arc<TaskTable>,
}

#[derive(Default)]
struct HostState {
    started: bool,
    host: Option<Arc<SessionHost>>,
    transport: Option<Arc<dyn SessionTransport>>,
}

/// `voice/remote_session.py`'s `SessionHost`, running in Rust. Events queue from construction.
#[lk_agents_abi::host(object(name = "RemoteSessionHost", cell = "arc"))]
impl HostRuntime {
    #[host(new)]
    pub fn new(#[host(ambient)] clock: Arc<dyn Clock>, #[host(ambient)] spawner: Arc<dyn Spawner>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        HostRuntime {
            shared: Arc::new(HostShared {
                clock,
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

    #[host(getter)]
    pub fn started(&self) -> bool {
        self.shared.state.lock().expect("host state").started
    }

    /// `register_session`; until set, requests get no response.
    #[host(method)]
    pub fn set_port(&self, port: Arc<dyn SessionPort>) {
        self.shared.state.lock().expect("host state").host = Some(Arc::new(SessionHost::new(port)));
    }

    /// The request types in flight, one per running handler.
    #[host(method)]
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
    #[host(method(name = "send_event"))]
    pub fn send_wire(&self, event: WireEvent, created_at: Option<f64>) -> Result<bool, String> {
        let Some(ev) = super::encode_wire(event)? else { return Ok(false) };
        let at = created_at.unwrap_or_else(super::epoch_secs_now);
        let ev = proto::AgentSessionEvent { created_at: None, event: Some(ev) };
        self.send_event(ev, at).map_err(|e| e.to_string())?;
        Ok(true)
    }

    /// Queue an already-encoded `AgentSessionEvent`, stamped `created_at` (now when `None`).
    #[host(method(name = "send_encoded_event"))]
    pub fn send_encoded(&self, event: &[u8], created_at: Option<f64>) -> Result<(), String> {
        use prost::Message as _;
        let event = proto::AgentSessionEvent::decode(event).map_err(|e| format!("not an AgentSessionEvent: {e}"))?;
        self.send_event(event, created_at.unwrap_or_else(super::epoch_secs_now)).map_err(|e| e.to_string())
    }

    /// `start()`: transport, then writer and receive loop. Idempotent; a failed start still counts
    /// as started, as the reference. `console` is where the console messages go.
    #[host(method)]
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
    #[host(method)]
    pub async fn aclose(&self, drain: Duration) -> Vec<String> {
        if !self.started() {
            return Vec::new();
        }
        let sh = &self.shared;
        let clock = &*sh.clock;
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

async fn recv_loop(sh: Arc<HostShared>, transport: Arc<dyn SessionTransport>, console: Option<Arc<dyn ConsoleSink>>) {
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
                    sh.handlers.spawn(&*sh.spawner, name, handle_request(host, Arc::clone(&transport), req));
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
        let _ =
            transport.send_message(&SessionHost::internal_error_response(&req.request_id)).await;
    }
}

/// Why a client request did not produce a response, as the exception the host raises.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestError {
    /// The transport refused the request; the transport's own error.
    #[error(transparent)]
    #[host(delegate)]
    Transport(HostError),
    /// No response inside the timeout (`asyncio.TimeoutError`).
    #[error("")]
    #[host(kind = "TimeoutError")]
    Timeout,
    /// The response carried a non-empty `error`: `session request {type} failed: {error}`.
    #[error("{0}")]
    Failed(String),
    /// The client closed while the request was pending (`future.cancel()`).
    #[error("")]
    #[host(kind = "asyncio.CancelledError")]
    Cancelled,
    /// A request before `start()`.
    #[error("remote session is not started")]
    NotStarted,
    /// Bytes that are no `SessionRequest`, or one that sets no request.
    #[error("{0}")]
    #[host(kind = "ValueError")]
    Malformed(String),
}

/// Why `wait_for_ready` gave up.
#[lk_agents_abi::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReadyError {
    #[error("wait_for_ready timed out")]
    #[host(kind = "TimeoutError")]
    Timeout,
    /// The latched send failure, preferred to a bare timeout.
    #[error(transparent)]
    #[host(delegate)]
    Transport(HostError),
    #[error("")]
    #[host(kind = "asyncio.CancelledError")]
    Cancelled,
}

/// Host seconds as a wait: negative is zero, and infinity (or past `Duration`'s range) is "never".
fn secs(s: f64) -> Duration {
    Duration::try_from_secs_f64(s.max(0.0)).unwrap_or(if s.is_nan() { Duration::ZERO } else { Duration::MAX / 4 })
}

/// `RemoteSession` (`remote_session.py:826`), running.
pub struct ClientRuntime {
    shared: Arc<ClientShared>,
}

struct ClientShared {
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    state: Mutex<ClientState>,
    /// `_pending_requests` (`remote_session.py:981`), by request id.
    pending: Mutex<HashMap<String, oneshot::Sender<proto::SessionResponse>>>,
    recv: Arc<TaskTable>,
    ids: IdMint,
}

#[derive(Default)]
struct ClientState {
    started: bool,
    transport: Option<Arc<dyn SessionTransport>>,
}

/// Request ids as `utils.shortuuid("req_")` spells them. Random, since clients in one room share a
/// response stream.
struct IdMint {
    keys: std::collections::hash_map::RandomState,
    counter: AtomicU64,
}

impl IdMint {
    const ALPHABET: &'static [u8] = b"23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    fn new() -> Self {
        IdMint { keys: Default::default(), counter: AtomicU64::new(0) }
    }

    fn mint(&self) -> String {
        let mut n = self.keys.hash_one(self.counter.fetch_add(1, Ordering::Relaxed));
        let mut out = String::from("req_");
        for _ in 0..11 {
            out.push(Self::ALPHABET[(n % 57) as usize] as char);
            n /= 57;
        }
        out
    }
}

struct Unregister<'a> {
    shared: &'a ClientShared,
    id: String,
}

impl Drop for Unregister<'_> {
    fn drop(&mut self) {
        self.shared.pending.lock().expect("pending requests").remove(&self.id);
    }
}

/// `voice/remote_session.py`'s `RemoteSession`, running in Rust: ids, correlation, timeouts and the
/// ready poll.
#[lk_agents_abi::host(object(name = "RemoteSessionClient", cell = "arc"))]
impl ClientRuntime {
    #[host(new)]
    pub fn new(#[host(ambient)] clock: Arc<dyn Clock>, #[host(ambient)] spawner: Arc<dyn Spawner>) -> Self {
        ClientRuntime {
            shared: Arc::new(ClientShared {
                clock,
                spawner,
                state: Mutex::new(ClientState::default()),
                pending: Mutex::default(),
                recv: Arc::default(),
                ids: IdMint::new(),
            }),
        }
    }

    #[host(getter)]
    pub fn started(&self) -> bool {
        self.shared.state.lock().expect("client state").started
    }

    /// `start()`: start the transport and the receive loop; arriving events go to `sink`. Idempotent.
    #[host(method)]
    pub async fn start(&self, transport: Arc<dyn SessionTransport>, sink: Arc<dyn ClientSink>) -> Result<(), HostError> {
        {
            let mut st = self.shared.state.lock().expect("client state");
            if st.started {
                return Ok(());
            }
            st.started = true;
            st.transport = Some(Arc::clone(&transport));
        }
        transport.start().await?;
        let sh = &self.shared;
        sh.recv.spawn(
            &*sh.spawner,
            "RemoteSession._recv_loop",
            client_recv_loop(Arc::clone(sh), transport, sink),
        );
        Ok(())
    }

    /// `aclose()`: cancel every pending request, stop reading, close the transport.
    #[host(method)]
    pub async fn aclose(&self) {
        let transport = {
            let mut st = self.shared.state.lock().expect("client state");
            if !st.started {
                return;
            }
            st.started = false;
            st.transport.take()
        };
        // dropping the senders is the cancel: each waiting caller sees `Cancelled`
        self.shared.pending.lock().expect("pending requests").clear();
        self.shared.recv.cancel_and_wait(&*self.shared.clock).await;
        if let Some(t) = transport {
            t.close().await;
        }
    }

    /// Send one encoded `SessionRequest` (its `request_id` is minted here and any given one is
    /// ignored) and return the encoded `SessionResponse`, waiting up to `timeout` seconds.
    #[host(method(name = "request"))]
    pub async fn request_encoded(&self, request: Vec<u8>, timeout: f64) -> Result<Vec<u8>, RequestError> {
        use prost::Message as _;
        let request = proto::SessionRequest::decode(request.as_slice())
            .map_err(|e| RequestError::Malformed(format!("not a SessionRequest: {e}")))?
            .request
            .ok_or_else(|| RequestError::Malformed("the SessionRequest sets no request".to_owned()))?;
        Ok(self.request(request, secs(timeout)).await?.encode_to_vec())
    }

    /// `wait_for_ready` in host seconds.
    #[host(method(name = "wait_for_ready"))]
    pub async fn wait_for_ready_secs(&self, timeout: f64, retry_interval: f64) -> Result<(), ReadyError> {
        self.wait_for_ready(secs(timeout), secs(retry_interval)).await
    }

    /// `_send_request`: register an id, send, wait up to `timeout`, refuse a non-empty `error`.
    pub async fn request(
        &self,
        request: proto::session_request::Request,
        timeout: Duration,
    ) -> Result<proto::SessionResponse, RequestError> {
        let sh = &*self.shared;
        let transport = sh
            .state
            .lock()
            .expect("client state")
            .transport
            .clone()
            .ok_or(RequestError::NotStarted)?;
        let kind = request_name(&request);
        let id = sh.ids.mint();
        let (tx, rx) = oneshot::channel();
        if sh.pending.lock().expect("pending requests").insert(id.clone(), tx).is_some() {
            // unreachable with a working id mint; the displaced caller sees `Cancelled`
            log::error!(target: LOG, request_id = id.as_str();
                "duplicate remote session request id; cancelling the displaced caller");
        }
        let _unregister = Unregister { shared: sh, id: id.clone() };
        let msg = proto::AgentSessionMessage {
            message: Some(Message::Request(proto::SessionRequest { request_id: id.clone(), request: Some(request) })),
        };
        transport.send_message(&msg).await.map_err(RequestError::Transport)?;
        let resp = match lk_agents_abi::timeout(&*sh.clock, timeout, rx).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(_)) => return Err(RequestError::Cancelled),
            Err(_) => {
                log::warn!(target: LOG, request_id = id.as_str(), request_type = kind, timeout = timeout.as_secs_f64();
                    "remote session request timed out");
                return Err(RequestError::Timeout);
            }
        };
        // `if resp.error: raise` (`remote_session.py:1057`): `Some("")` is success.
        match resp.error.as_deref() {
            Some(e) if !e.is_empty() => Err(RequestError::Failed(format!("session request {kind} failed: {e}"))),
            _ => Ok(resp),
        }
    }

    /// `wait_for_ready` (`remote_session.py:1070`): ping, each bounded by `min(retry, remaining)`,
    /// until answered. At the deadline, the latched send error is preferred to a bare timeout.
    pub async fn wait_for_ready(
        &self,
        timeout: Duration,
        retry_interval: Duration,
    ) -> Result<(), ReadyError> {
        let clock = &*self.shared.clock;
        let deadline = clock.now().saturating_add(timeout);
        let mut latched: Option<HostError> = None;
        let mut now = clock.now();
        loop {
            let remaining = deadline.saturating_duration_since(now);
            if remaining.is_zero() {
                return Err(latched.map_or(ReadyError::Timeout, ReadyError::Transport));
            }
            let ping = proto::session_request::Request::Ping(proto::session_request::Ping {});
            let error = match self.request(ping, retry_interval.min(remaining)).await {
                Ok(_) => return Ok(()),
                // No sleep: the elapsed timeout was the pacing. A bare timeout at the deadline even
                // with a latched send error, as in the reference.
                Err(RequestError::Timeout) => {
                    now = clock.now();
                    if now >= deadline {
                        return Err(ReadyError::Timeout);
                    }
                    continue;
                }
                Err(RequestError::Cancelled) => return Err(ReadyError::Cancelled),
                Err(RequestError::Transport(e)) => e,
                Err(RequestError::Failed(m)) => HostError::new(m),
                Err(e @ (RequestError::NotStarted | RequestError::Malformed(_))) => HostError::new(e.to_string()),
            };
            // The ping could not be sent: latch the error and sleep (nothing else paces the loop).
            now = clock.now();
            if now >= deadline {
                return Err(ReadyError::Transport(error));
            }
            latched = Some(error);
            let nap = retry_interval.min(deadline.saturating_duration_since(now));
            clock.wait_until(clock.now().saturating_add(nap)).await;
            now = clock.now();
        }
    }
}

async fn client_recv_loop(sh: Arc<ClientShared>, transport: Arc<dyn SessionTransport>, sink: Arc<dyn ClientSink>) {
    loop {
        let msg = match transport.recv().await {
            Ok(Some(msg)) => msg,
            Ok(None) => return,
            Err(e) => {
                log::warn!(target: LOG, error = e.to_string().as_str(); "error processing session message");
                return;
            }
        };
        match msg.message {
            Some(Message::Response(resp)) => {
                // a response to a request that already timed out finds nothing and is dropped
                let slot = sh.pending.lock().expect("pending requests").remove(&resp.request_id);
                if let Some(tx) = slot {
                    let _ = tx.send(resp);
                }
            }
            // Emitted under `msg.event.WhichOneof("event")`; an event with no arm set is dropped.
            Some(Message::Event(e)) => {
                if let Some(inner) = &e.event {
                    sink.event(super::event_name(inner), &e);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        AgentInfo, ChatItem, IoUpdate, RtcStats, RunOutcome, SessionState, SimulationVerdict,
        VerdictOutcome,
    };
    use lk_agents_abi::metrics::usage::ModelUsage;
    use super::*;
    use async_trait::async_trait;
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

    #[async_trait]
    impl SessionTransport for Pipe {
        async fn start(&self) -> Result<(), HostError> {
            Ok(())
        }
        async fn send_message(&self, msg: &proto::AgentSessionMessage) -> Result<(), HostError> {
            if self.fail.load(Ordering::SeqCst) {
                return Err(HostError::new("transport is down"));
            }
            self.sent.lock().unwrap().push(msg.clone());
            let _ = self.tx.send(msg.clone());
            Ok(())
        }
        fn send_message_nowait(&self, _msg: &proto::AgentSessionMessage) {}
        async fn recv(&self) -> Result<Option<proto::AgentSessionMessage>, HostError> {
            Ok(self.rx.lock().await.recv().await)
        }
        async fn close(&self) {
            self.closed.store(true, Ordering::SeqCst);
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

    #[async_trait]
    impl SessionPort for Port {
        async fn chat_history(&self) -> Result<Vec<ChatItem>, HostError> {
            Err(HostError::new("no history here"))
        }
        async fn agent_info(&self) -> Result<AgentInfo, HostError> {
            Ok(("agent-1".into(), None, vec![], vec![]))
        }
        async fn run_input(&self, _text: &str) -> Result<RunOutcome, HostError> {
            self.release.notified().await;
            Ok((Some(vec![]), None))
        }
        async fn session_state(&self) -> Result<SessionState, HostError> {
            Ok(("not-a-state".into(), "away".into(), "agent-1".into(), Default::default(), None))
        }
        async fn rtc_stats(&self) -> Result<Option<RtcStats>, HostError> {
            Ok(None)
        }
        async fn session_usage(&self) -> Result<Vec<ModelUsage>, HostError> {
            Ok(Vec::new())
        }
        fn framework_info(&self) -> (String, String) {
            ("rust".into(), "0".into())
        }
        async fn update_io(&self, _update: IoUpdate) -> Result<(), HostError> {
            Ok(())
        }
        async fn finalize_simulation(&self, _p: SimulationVerdict) -> Result<VerdictOutcome, HostError> {
            Ok((None, None))
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
            Arc::new(lk_agents_abi::RealClock::default()),
            Arc::new(lk_agents_abi::TokioSpawner::current()),
        )
    }

    async fn setup(release: Arc<Notify>) -> (HostRuntime, ClientRuntime, Arc<Pipe>, Arc<Events>) {
        let (ht, ct) = pair();
        let (clock, spawner) = rt();
        let host = HostRuntime::new(Arc::clone(&clock), Arc::clone(&spawner));
        host.set_port(Arc::new(Port { release }));
        host.start(ht.clone(), None).await.unwrap();
        let client = ClientRuntime::new(clock, spawner);
        let events = Arc::new(Events(Mutex::new(Vec::new())));
        client.start(ct, events.clone()).await.unwrap();
        (host, client, ht, events)
    }

    fn secs(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    async fn nap(d: Duration) {
        let clock = lk_agents_abi::RealClock::default();
        clock.wait_until(clock.now().saturating_add(d)).await;
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
        let host = HostRuntime::new(clock, spawner);
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
        let client = ClientRuntime::new(clock.clone(), spawner.clone());
        ct.fail.store(true, Ordering::SeqCst);
        client.start(ct.clone(), Arc::new(Events(Mutex::new(vec![])))).await.unwrap();
        let err = client.wait_for_ready(secs(0.2), secs(0.05)).await.unwrap_err();
        assert_eq!(err, ReadyError::Transport(HostError::new("transport is down")));

        let host = HostRuntime::new(clock, spawner);
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
        let client = Arc::new(ClientRuntime::new(clock, spawner));
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
