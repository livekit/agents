//! The far end of the remote session protocol (`voice/remote_session.py`'s `RemoteSession`): ids,
//! correlation, timeouts and the ready poll. No agent runs it; the host's tests drive the
//! session host with it.

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::{Clock, Spawner, TaskSet};
use tokio::sync::oneshot;

use super::{ClientSink, SessionTransport, proto, request_name};
use proto::agent_session_message::Message;

const LOG: &str = "livekit.agents";

/// Why a client request did not produce a response, as the exception the host raises.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestError {
    /// The transport refused the request; the transport's own error.
    #[error(transparent)]
    Transport(HostError),
    /// No response inside the timeout (`asyncio.TimeoutError`).
    #[error("")]
    Timeout,
    /// The response carried a non-empty `error`: `session request {type} failed: {error}`.
    #[error("{0}")]
    Failed(String),
    /// The client closed while the request was pending (`future.cancel()`).
    #[error("")]
    Cancelled,
    /// A request before `start()`.
    #[error("remote session is not started")]
    NotStarted,
}

/// Why `wait_for_ready` gave up.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReadyError {
    #[error("wait_for_ready timed out")]
    Timeout,
    /// The latched send failure, preferred to a bare timeout.
    #[error(transparent)]
    Transport(HostError),
    #[error("")]
    Cancelled,
}

/// `RemoteSession` (`remote_session.py:826`), running.
pub struct ClientRuntime {
    pub(super) shared: Arc<ClientShared>,
}

pub(super) struct ClientShared {
    clock: Arc<dyn Clock>,
    spawner: Arc<dyn Spawner>,
    state: Mutex<ClientState>,
    /// `_pending_requests` (`remote_session.py:981`), by request id.
    pub(super) pending: Mutex<HashMap<String, oneshot::Sender<proto::SessionResponse>>>,
    recv: Arc<TaskSet>,
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
impl ClientRuntime {
    pub fn new(clock: Arc<dyn Clock>, spawner: Arc<dyn Spawner>) -> Self {
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

    /// `start()`: start the transport and the receive loop; arriving events go to `sink`. Idempotent.
    pub async fn start(
        &self,
        transport: Arc<dyn SessionTransport>,
        sink: Arc<dyn ClientSink>,
    ) -> Result<(), HostError> {
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
            message: Some(Message::Request(proto::SessionRequest {
                request_id: id.clone(),
                request: Some(request),
            })),
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
            Some(e) if !e.is_empty() => {
                Err(RequestError::Failed(format!("session request {kind} failed: {e}")))
            }
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
                Err(e @ RequestError::NotStarted) => HostError::new(e.to_string()),
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

async fn client_recv_loop(
    sh: Arc<ClientShared>,
    transport: Arc<dyn SessionTransport>,
    sink: Arc<dyn ClientSink>,
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
