//! One WebSocket connection: the answer to the upgrade, frames both ways and the close handshake.
//! A reader task answers pings and queues messages in a bounded inbox, so a connection that only
//! sends still notices its peer leaving. Closing is one shared operation, whoever asks for it.

use std::sync::Arc;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::http::{ExchangeError, ExchangeEvent, ExchangeMessage};
use lk_agents_abi::{BoxFuture, OwnedTask, SpawnerExt};
use lk_agents_net::task::TokioSpawner;
use tokio::io::{BufReader, Join};
use tokio::sync::{Mutex, mpsc, watch};
use tokio::time::Instant;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::{CloseFrame, Role, WebSocketConfig};
use tokio_tungstenite::tungstenite::{Error as WsError, Message, error::CapacityError};

use super::exchange::Response;
use super::{HttpServerLimits, ReadHalf, ServerState, StreamEnd, WriteHalf};
use crate::transport::HttpStreamResetCode as Code;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const GOING_AWAY: u16 = 1001;
/// A close frame's payload is at most 125 bytes, two of them the code.
const MAX_CLOSE_REASON: usize = 123;

/// At most `max` bytes of `text`, cut at a character boundary.
fn truncate(text: &str, max: usize) -> &str {
    let mut end = text.len().min(max);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

type Io = Join<BufReader<ReadHalf>, WriteHalf>;

fn code(code: CloseCode) -> u16 {
    code.into()
}

/// How the reader stopped.
enum Ending {
    /// The peer closed, or its stream ended.
    Peer,
    /// The peer broke the protocol or a limit: close with this code.
    Failed(u16, &'static str),
    /// The peer answered no ping.
    Unresponsive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The application has not answered the upgrade.
    Handshake,
    Open,
    Closing,
    Closed(u16),
}

enum State {
    Handshake {
        read: BufReader<ReadHalf>,
        response: Response,
    },
    /// The application answers the upgrade with an HTTP response.
    Denying(Response),
    Open(SplitSink<WebSocketStream<Io>, Message>),
    Ended,
}

/// A message waiting for the application.
struct Inbound {
    event: ExchangeEvent,
    _queued: Queued,
}

/// A message's bytes in its connection's inbox, released when the application takes it.
struct Queued {
    total: Arc<AtomicUsize>,
    size: usize,
}

impl Drop for Queued {
    fn drop(&mut self) {
        self.total.fetch_sub(self.size, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct Tasks {
    reader: Option<OwnedTask>,
    close: Option<OwnedTask>,
}

struct Shared {
    end: Arc<dyn StreamEnd>,
    /// Bytes waiting in the inbox, bounded by `websocket_inbox_bytes`.
    queued: Arc<AtomicUsize>,
    limits: HttpServerLimits,
    accept: String,
    state: Mutex<State>,
    inbox: Mutex<mpsc::Receiver<Inbound>>,
    /// Handed to the reader on accept.
    inbox_tx: parking_lot::Mutex<Option<mpsc::Sender<Inbound>>>,
    phase: watch::Sender<Phase>,
    reader_done: watch::Sender<bool>,
    tasks: parking_lot::Mutex<Tasks>,
}

#[derive(Clone)]
pub(super) struct Connection(Arc<Shared>);

impl Connection {
    pub(super) fn new(
        read: BufReader<ReadHalf>,
        response: Response,
        end: Arc<dyn StreamEnd>,
        key: &str,
        limits: HttpServerLimits,
    ) -> Self {
        let (inbox_tx, inbox) = mpsc::channel(limits.websocket_inbox.max(1));
        Self(Arc::new(Shared {
            end,
            queued: Arc::default(),
            limits,
            accept: tokio_tungstenite::tungstenite::handshake::derive_accept_key(key.as_bytes()),
            state: Mutex::new(State::Handshake { read, response }),
            inbox: Mutex::new(inbox),
            inbox_tx: parking_lot::Mutex::new(Some(inbox_tx)),
            phase: watch::Sender::new(Phase::Handshake),
            reader_done: watch::Sender::new(false),
            tasks: Default::default(),
        }))
    }

    /// Run the application, then close what it left open and join the reader. An upgrade the
    /// application does not answer within `answer_by` is reset and the application cancelled.
    pub(super) async fn run(
        &self,
        handled: BoxFuture<'_, Result<(), HostError>>,
        answer_by: std::time::Duration,
        mut server: watch::Receiver<ServerState>,
    ) {
        let mut handled = handled;
        let stopped = async {
            let _ = server.wait_for(|state| *state == ServerState::Stopping).await;
        };
        let unanswered = async {
            let mut phase = self.0.phase.subscribe();
            let answered = phase.wait_for(|phase| *phase != Phase::Handshake);
            if tokio::time::timeout(answer_by, answered).await.is_ok() {
                std::future::pending::<()>().await;
            }
        };
        let result = tokio::select! {
            biased;
            () = stopped => {
                drop(handled);
                self.close(GOING_AWAY, "worker stopped").await;
                return self.shutdown();
            }
            () = unanswered => {
                drop(handled);
                self.0.end.reset(Code::HsrTimeout);
                *self.0.state.lock().await = State::Ended;
                self.mark_closed(code(CloseCode::Abnormal));
                return self.shutdown();
            }
            result = &mut handled => result,
        };
        match result {
            Ok(()) => self.close(code(CloseCode::Normal), "").await,
            Err(_) => self.close(code(CloseCode::Error), "application failed").await,
        }
        self.shutdown();
    }

    /// Refuse the upgrade before the application saw it.
    pub(super) fn refuse(&self) {
        self.0.end.reset(Code::HsrRefused);
        self.mark_closed(code(CloseCode::Abnormal));
    }

    /// Cancel the reader and close tasks, which hold the connection.
    pub(super) fn shutdown(&self) {
        let tasks = std::mem::take(&mut *self.0.tasks.lock());
        drop(tasks);
    }

    pub(super) async fn receive(&self) -> Result<ExchangeEvent, ExchangeError> {
        let _ = self.0.phase.subscribe().wait_for(|phase| *phase != Phase::Handshake).await;
        let mut inbox = self.0.inbox.lock().await;
        if *self.0.phase.borrow() == Phase::Open
            && let Some(inbound) = inbox.recv().await
            && *self.0.phase.borrow() == Phase::Open
        {
            return Ok(inbound.event);
        }
        drop(inbox);
        Ok(ExchangeEvent::Closed { code: self.wait_closed().await })
    }

    pub(super) async fn send(&self, message: ExchangeMessage) -> Result<(), ExchangeError> {
        match message {
            ExchangeMessage::Accept { subprotocol, headers } => {
                self.accept(subprotocol.as_deref(), &headers).await
            }
            ExchangeMessage::Text { text } => {
                let size = text.len();
                self.send_message(Message::text(text), size).await
            }
            ExchangeMessage::Binary { data } => {
                let size = data.len();
                self.send_message(Message::binary(data), size).await
            }
            ExchangeMessage::Close { code, reason } => {
                if !CloseCode::from(code).is_allowed() {
                    return Err(ExchangeError::Protocol(format!(
                        "{code} is not a close code an endpoint sends"
                    )));
                }
                self.close(code, &reason).await;
                Ok(())
            }
            ExchangeMessage::Start { status, headers } => {
                let mut state = self.0.state.lock().await;
                let State::Handshake { response, .. } = &mut *state else {
                    return Err(ExchangeError::Protocol("the upgrade was already answered".into()));
                };
                response.start(status, &headers).await?;
                let State::Handshake { response, .. } =
                    std::mem::replace(&mut *state, State::Ended)
                else {
                    unreachable!("checked above")
                };
                *state = State::Denying(response);
                self.0.phase.send_replace(Phase::Closing);
                Ok(())
            }
            ExchangeMessage::Body { data, more } => {
                let mut state = self.0.state.lock().await;
                let State::Denying(response) = &mut *state else {
                    return Err(ExchangeError::Protocol(
                        "a response body on an accepted WebSocket".into(),
                    ));
                };
                if response.body(&data, more).await? {
                    *state = State::Ended;
                    self.0.end.finish();
                    self.mark_closed(code(CloseCode::Abnormal));
                }
                Ok(())
            }
        }
    }

    async fn accept(
        &self,
        subprotocol: Option<&str>,
        headers: &[(Vec<u8>, Vec<u8>)],
    ) -> Result<(), ExchangeError> {
        let mut state = self.0.state.lock().await;
        if !matches!(*state, State::Handshake { .. }) {
            return Err(ExchangeError::Protocol("the upgrade was already answered".into()));
        }
        let State::Handshake { read, mut response } = std::mem::replace(&mut *state, State::Ended)
        else {
            unreachable!("checked above")
        };
        if let Err(error) = response.switch(&self.0.accept, subprotocol, headers).await {
            self.0.end.reset(Code::HsrAbort);
            self.mark_closed(code(CloseCode::Abnormal));
            return Err(error);
        }
        let limits = &self.0.limits;
        let config = WebSocketConfig::default()
            .max_frame_size(Some(limits.websocket_frame_bytes))
            .max_message_size(Some(limits.websocket_message_bytes));
        let io = tokio::io::join(read, response.into_write());
        let (sink, stream) =
            WebSocketStream::from_raw_socket(io, Role::Server, Some(config)).await.split();
        *state = State::Open(sink);
        let inbox = self.0.inbox_tx.lock().take().expect("accepted once");
        let reader = TokioSpawner::current().spawn(self.clone().read_frames(stream, inbox));
        self.0.tasks.lock().reader = Some(reader);
        self.0.phase.send_replace(Phase::Open);
        Ok(())
    }

    async fn send_message(&self, message: Message, size: usize) -> Result<(), ExchangeError> {
        if size > self.0.limits.websocket_message_bytes {
            return Err(ExchangeError::Protocol("the WebSocket message exceeds its limit".into()));
        }
        match *self.0.phase.borrow() {
            Phase::Handshake => {
                return Err(ExchangeError::Protocol("a WebSocket message before accept".into()));
            }
            // A message sent while the connection closes is dropped, as the peer is leaving.
            Phase::Closing | Phase::Closed(_) => return Ok(()),
            Phase::Open => {}
        }
        let mut state = self.0.state.lock().await;
        let State::Open(sink) = &mut *state else {
            return Ok(());
        };
        if let Err(error) = sink.send(message).await {
            // The peer stopped reading or went away: the connection cannot carry a close.
            drop(state);
            self.0.end.reset(Code::HsrAbort);
            self.mark_closed(code(CloseCode::Abnormal));
            return Err(ExchangeError::Closed(error.to_string()));
        }
        Ok(())
    }

    /// Close the connection and wait until it closed. The first caller's code wins; the close
    /// is owned by the connection, so a caller giving up does not leave it half done.
    pub(super) async fn close(&self, code: u16, reason: &str) {
        self.begin_close(code, reason);
        self.wait_closed().await;
    }

    fn begin_close(&self, code: u16, reason: &str) {
        let mut tasks = self.0.tasks.lock();
        let closed = matches!(*self.0.phase.borrow(), Phase::Closed(_));
        if tasks.close.is_none() && !closed {
            let reason = reason.to_owned();
            let this = self.clone();
            tasks.close = Some(
                TokioSpawner::current().spawn(async move { this.finish_close(code, reason).await }),
            );
        }
    }

    async fn finish_close(self, close_code: u16, reason: String) {
        self.0.phase.send_if_modified(|phase| {
            let open = !matches!(phase, Phase::Closed(_));
            if open {
                *phase = Phase::Closing;
            }
            open
        });
        let limit = self.0.limits.close_flush_timeout;
        let Ok(mut state) = tokio::time::timeout(limit, self.0.state.lock()).await else {
            self.0.end.reset(Code::HsrAbort);
            return self.mark_closed(close_code);
        };
        match std::mem::replace(&mut *state, State::Ended) {
            State::Open(mut sink) => {
                let frame = CloseFrame {
                    code: CloseCode::from(close_code),
                    reason: truncate(&reason, MAX_CLOSE_REASON).into(),
                };
                let sent =
                    tokio::time::timeout(limit, sink.send(Message::Close(Some(frame)))).await;
                if matches!(sent, Ok(Ok(()))) {
                    self.0.end.finish();
                    // Wait for the peer's close reply, which ends the reader.
                    let mut done = self.0.reader_done.subscribe();
                    let replied = tokio::time::timeout(limit, done.wait_for(|done| *done)).await;
                    if replied.map(drop).is_err() {
                        self.0.end.stop(Code::HsrAbort);
                    }
                } else {
                    self.0.end.reset(Code::HsrAbort);
                }
            }
            State::Handshake { mut response, .. } => {
                // Refusing the upgrade: a plain 403.
                let headers = [(b"content-length".to_vec(), b"0".to_vec())];
                let refused = async {
                    response.start(403, &headers).await?;
                    response.body(b"", false).await
                };
                match tokio::time::timeout(limit, refused).await {
                    Ok(Ok(_)) => self.0.end.finish(),
                    _ => self.0.end.reset(Code::HsrAbort),
                }
            }
            // The application closed in the middle of its own denial response.
            State::Denying(_) => self.0.end.reset(Code::HsrAbort),
            State::Ended => {}
        }
        self.mark_closed(close_code);
    }

    fn mark_closed(&self, close_code: u16) {
        self.0.phase.send_if_modified(|phase| {
            let open = !matches!(phase, Phase::Closed(_));
            if open {
                *phase = Phase::Closed(close_code);
            }
            open
        });
        if let Some(reader) = self.0.tasks.lock().reader.as_ref() {
            reader.cancel();
        }
    }

    async fn wait_closed(&self) -> u16 {
        let mut phase = self.0.phase.subscribe();
        match phase.wait_for(|phase| matches!(phase, Phase::Closed(_))).await.as_deref() {
            Ok(Phase::Closed(code)) => *code,
            _ => code(CloseCode::Abnormal),
        }
    }

    /// Read frames until the connection ends: queue messages, answer pings (tungstenite replies
    /// while reading) and the peer's close, and ping a quiet peer to learn it is still there.
    async fn read_frames(
        self,
        mut stream: SplitStream<WebSocketStream<Io>>,
        inbox: mpsc::Sender<Inbound>,
    ) {
        let (interval, patience) =
            (self.0.limits.websocket_ping_interval, self.0.limits.websocket_ping_timeout);
        let mut peer_code = code(CloseCode::Abnormal);
        let mut heard = Instant::now();
        let mut pinged: Option<Instant> = None;
        let ending = loop {
            let wake = pinged.map_or(heard + interval, |at| at + patience);
            let next = tokio::select! {
                biased;
                next = stream.next() => next,
                () = tokio::time::sleep_until(wake) => {
                    if *self.0.phase.borrow() != Phase::Open {
                        // Closing: the close handshake bounds the wait.
                        (heard, pinged) = (Instant::now(), None);
                        continue;
                    }
                    if pinged.is_some() || !self.ping().await {
                        break Ending::Unresponsive;
                    }
                    pinged = Some(Instant::now());
                    continue;
                }
            };
            // Any frame, a pong included, shows the peer is there.
            (heard, pinged) = (Instant::now(), None);
            let event = match next {
                Some(Ok(Message::Text(text))) => ExchangeEvent::Text { text: text.to_string() },
                Some(Ok(Message::Binary(data))) => ExchangeEvent::Binary { data: data.to_vec() },
                Some(Ok(Message::Close(frame))) => {
                    peer_code = frame.map_or(code(CloseCode::Status), |frame| frame.code.into());
                    // Keep reading: the next read flushes the close reply, then ends.
                    continue;
                }
                Some(Ok(_)) => continue,
                Some(Err(WsError::Capacity(CapacityError::MessageTooLong { .. }))) => {
                    let reason = "WebSocket message limit exceeded";
                    break Ending::Failed(code(CloseCode::Size), reason);
                }
                Some(Err(WsError::Utf8(_))) => {
                    let reason = "invalid UTF-8 in a text message";
                    break Ending::Failed(code(CloseCode::Invalid), reason);
                }
                Some(Err(WsError::Protocol(_) | WsError::Capacity(_))) => {
                    break Ending::Failed(code(CloseCode::Protocol), "WebSocket protocol error");
                }
                Some(Err(_)) | None => break Ending::Peer,
            };
            if !self.deliver(&inbox, event) {
                let reason = "WebSocket inbox capacity exhausted";
                break Ending::Failed(code(CloseCode::Size), reason);
            }
        };
        drop(inbox);
        self.0.reader_done.send_replace(true);
        match ending {
            Ending::Failed(close_code, reason) => self.begin_close(close_code, reason),
            Ending::Unresponsive => {
                // A peer that answers nothing cannot complete a close handshake either.
                self.0.end.reset(Code::HsrAbort);
                *self.0.state.lock().await = State::Ended;
                self.mark_closed(code(CloseCode::Abnormal));
            }
            Ending::Peer if *self.0.phase.borrow() == Phase::Open => {
                // The peer ended the connection: its close was answered, or it went away.
                *self.0.state.lock().await = State::Ended;
                self.0.end.finish();
                self.mark_closed(peer_code);
            }
            Ending::Peer => {}
        }
    }

    /// Whether a ping went out.
    async fn ping(&self) -> bool {
        let mut state = self.0.state.lock().await;
        let State::Open(sink) = &mut *state else {
            return false;
        };
        sink.send(Message::Ping(Default::default())).await.is_ok()
    }

    fn deliver(&self, inbox: &mpsc::Sender<Inbound>, event: ExchangeEvent) -> bool {
        let size = match &event {
            ExchangeEvent::Text { text } => text.len(),
            ExchangeEvent::Binary { data } => data.len(),
            _ => 0,
        };
        let total = self.0.queued.clone();
        let queued = Queued { size, total: total.clone() };
        if total.fetch_add(size, Ordering::Relaxed) + size > self.0.limits.websocket_inbox_bytes {
            return false;
        }
        // A full inbox drops the message, and its `Queued` releases the bytes.
        inbox.try_send(Inbound { event, _queued: queued }).is_ok()
    }
}
