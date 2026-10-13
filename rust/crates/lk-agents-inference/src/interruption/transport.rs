//! The driver: one WebSocket, a reconnect ladder, and the effects [`InterruptionCore`] asked for.
//! `_forward_data` and `recv_task` (`inference/interruption.py:793`) are `select!` arms in
//! `run_once`; only the send half is a task. The `Mutex<InterruptionCore>` is never held across an
//! `.await`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, UNIX_EPOCH};

use lk_agents_abi::{Clock, HostRef};
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::types::retry::{self, RequestKind, RetryDecision};
use lk_agents_abi::metrics::base::InterruptionMetrics;
use tokio::sync::{Notify, mpsc};
use tokio::task::JoinHandle;

use crate::interruption::core::{
    InputEvent, InputOutcome, InterruptionCore, RecvOutcome, SpanAttributes, StreamError,
};
use crate::interruption::event::{InterruptionDetectionError, OverlappingSpeechEvent, interruption_metrics};
use crate::interruption::options::InterruptionOptions;
use crate::interruption::proto::{
    ClientMessage, ENCODING_S16LE, ServerMessage, SessionCreateSettings, frame_request,
};
use crate::interruption::retry::{
    ConnectFailure, closed_unexpectedly, connect_error, session_create_failed,
};
use crate::headers::HeaderProvider;
use crate::token::{DEFAULT_TTL_SECS, create_access_token};
use crate::interruption::{LABEL, MODEL, PROVIDER};
use crate::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};

/// One connection's two halves.
struct Socket {
    sender: Arc<dyn WsSender>,
    receiver: Box<dyn WsReceiver>,
}

/// Everything the reference reads off the ambient job context, supplied by the driver.
#[derive(Clone)]
pub struct ConnectionConfig {
    pub conn_options: ApiConnectOptions,
    /// The socket: the native one, or the caller's own (`AdaptiveInterruptionDetector(
    /// http_session=..)`, whose `ws_connect` the reference dials through).
    pub transport: Arc<dyn WsTransport>,
    /// `get_inference_headers()`, called per connection as `_connect_ws` does (§1.9), so each
    /// connection has its own session id. `Authorization` is added here and must not be in it.
    pub headers: HeaderProvider,
    pub token_ttl_secs: u64,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        ConnectionConfig {
            conn_options: ApiConnectOptions::default(),
            transport: Arc::new(crate::native::TungsteniteTransport),
            headers: Arc::new(Vec::new),
            token_ttl_secs: DEFAULT_TTL_SECS,
        }
    }
}

/// The three things `AdaptiveInterruptionDetector` emits for a [`stream`](AdaptiveInterruptionDetector::stream).
#[derive(Debug, Clone)]
pub enum DetectorEvent {
    OverlappingSpeech(Box<OverlappingSpeechEvent>),
    Error(Box<InterruptionDetectionError>),
    MetricsCollected(Box<InterruptionMetrics>),
}

/// `AdaptiveInterruptionDetector` (`inference/interruption.py:264`).
pub struct AdaptiveInterruptionDetector {
    opts: Arc<Mutex<InterruptionOptions>>,
    config: ConnectionConfig,
    clock: Arc<dyn Clock>,
    events: mpsc::UnboundedSender<DetectorEvent>,
    /// `weakref.WeakSet[InterruptionWebSocketStream]`, so `update_options` reaches live streams
    /// without keeping dead ones alive.
    streams: Mutex<Vec<Weak<Notify>>>,
}

impl AdaptiveInterruptionDetector {
    pub fn new(
        opts: InterruptionOptions,
        config: ConnectionConfig,
        clock: Arc<dyn Clock>,
    ) -> (Self, mpsc::UnboundedReceiver<DetectorEvent>) {
        let (events, rx) = mpsc::unbounded_channel();
        let detector = AdaptiveInterruptionDetector {
            opts: Arc::new(Mutex::new(opts)),
            config,
            clock,
            events,
            streams: Mutex::new(Vec::new()),
        };
        (detector, rx)
    }

    /// `update_options` (`inference/interruption.py:391`). The options are shared with live
    /// streams, which are only nudged to reconnect; the next `session.create` carries them.
    pub fn update_options(&self, threshold: Option<f64>, min_interruption_duration: Option<f64>) {
        self.opts.lock().unwrap().update(threshold, min_interruption_duration);
        let mut streams = self.streams.lock().unwrap();
        streams.retain(|s| {
            match s.upgrade() {
                Some(reconnect) => {
                    reconnect.notify_one();
                    true
                }
                None => false,
            }
        });
    }

    /// `stream(conn_options=...)`, whose verdicts, metrics and errors also reach the detector's
    /// channel: the session's stream, since the session's host drains that channel.
    pub fn stream(&self) -> InterruptionStream {
        self.open(true)
    }

    /// A stream whose caller emits its own events on the detector (the host's
    /// `InterruptionWebSocketStream`): nothing reaches the detector's channel, so it never
    /// competes with the session's stream there.
    pub fn private_stream(&self) -> InterruptionStream {
        self.open(false)
    }

    /// A [`private_stream`](Self::private_stream) under the caller's own retry ladder (the host's
    /// `_main_task`): the first failure ends it, reported once. `transport` replaces the
    /// detector's socket for this stream (the host's `http_session`).
    pub fn private_stream_once(&self, transport: Option<Arc<dyn WsTransport>>) -> InterruptionStream {
        let mut config = self.config.clone();
        config.conn_options.max_retry = 0;
        if let Some(transport) = transport {
            config.transport = transport;
        }
        self.open_with(false, config)
    }

    fn open(&self, fan_out: bool) -> InterruptionStream {
        self.open_with(fan_out, self.config.clone())
    }

    fn open_with(&self, fan_out: bool, config: ConnectionConfig) -> InterruptionStream {
        let reconnect = Arc::new(Notify::new());
        self.streams.lock().unwrap().push(Arc::downgrade(&reconnect));

        let (input_tx, input_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let closed = Arc::new(AtomicBool::new(false));

        let ctx = StreamContext {
            opts: self.opts.clone(),
            config,
            clock: self.clock.clone(),
            detector_events: fan_out.then(|| self.events.clone()),
            stream_events: event_tx,
            reconnect: reconnect.clone(),
            closed: closed.clone(),
        };
        let task = tokio::spawn(main_task(ctx, input_rx));
        InterruptionStream { input_tx: Some(input_tx), event_rx, reconnect, closed, task }
    }
}

/// What one stream reports to its caller: a verdict, the failure (recoverable or not) that
/// explains why none came, so the session can hand barge-in back to the VAD (I-24), or the verdict
/// to write on the span the caller pushed (a span is a host object, §1.11).
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Overlap(Box<OverlappingSpeechEvent>),
    Error(Box<InterruptionDetectionError>),
    SpanUpdate { span: HostRef, attrs: SpanAttributes },
}

#[derive(Clone)]
struct StreamContext {
    opts: Arc<Mutex<InterruptionOptions>>,
    config: ConnectionConfig,
    clock: Arc<dyn Clock>,
    /// `None` for a [`private_stream`](AdaptiveInterruptionDetector::private_stream).
    detector_events: Option<mpsc::UnboundedSender<DetectorEvent>>,
    stream_events: mpsc::UnboundedSender<StreamEvent>,
    reconnect: Arc<Notify>,
    closed: Arc<AtomicBool>,
}

impl StreamContext {
    fn wall(&self) -> f64 {
        self.clock.wall().duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs_f64()
    }

    /// `self.send(ev)` plus `_metrics_monitor_task`'s fold, inline.
    fn emit_overlap(&self, ev: OverlappingSpeechEvent) {
        let metrics = interruption_metrics(&ev, self.wall(), MODEL, PROVIDER);
        let _ = self.stream_events.send(StreamEvent::Overlap(Box::new(ev.clone())));
        if let Some(detector) = &self.detector_events {
            let _ = detector.send(DetectorEvent::OverlappingSpeech(Box::new(ev)));
            let _ = detector.send(DetectorEvent::MetricsCollected(Box::new(metrics)));
        }
    }

    /// The `"error"` channel, and the same failure on the stream's own channel.
    fn emit_error(&self, error: ApiError, recoverable: bool) {
        let err = Box::new(InterruptionDetectionError {
            timestamp: self.wall(),
            label: LABEL.to_owned(),
            error,
            recoverable,
        });
        let _ = self.stream_events.send(StreamEvent::Error(err.clone()));
        if let Some(detector) = &self.detector_events {
            let _ = detector.send(DetectorEvent::Error(err));
        }
    }
}

/// `InterruptionWebSocketStream`, from the caller's side.
pub struct InterruptionStream {
    /// `Option` so `end_input()` can drop the sender (`self._input_ch.close()`).
    input_tx: Option<mpsc::UnboundedSender<InputEvent>>,
    event_rx: mpsc::UnboundedReceiver<StreamEvent>,
    reconnect: Arc<Notify>,
    closed: Arc<AtomicBool>,
    task: JoinHandle<Result<(), ApiError>>,
}

/// `RuntimeError("... is closed")` / `RuntimeError("... input ended")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StreamClosed {
    #[error("livekit.agents.inference.interruption.InterruptionWebSocketStream is closed")]
    Closed,
    #[error("livekit.agents.inference.interruption.InterruptionWebSocketStream input ended")]
    InputEnded,
}

impl InterruptionStream {
    /// `push_frame`. Event channel closed = the stream failed; input channel closed = the caller
    /// already ended input.
    pub fn push(&self, ev: InputEvent) -> Result<(), StreamClosed> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StreamClosed::Closed);
        }
        let tx = self.input_tx.as_ref().ok_or(StreamClosed::InputEnded)?;
        tx.send(ev).map_err(|_| StreamClosed::InputEnded)
    }

    /// `flush()`.
    pub fn flush(&self) -> Result<(), StreamClosed> {
        self.push(InputEvent::Flush)
    }

    /// A clone of the input sender, for pushing from another thread than the one reading
    /// verdicts. A holder must drop it when it ends input, or the stream's task parks forever.
    pub fn sender(&self) -> Option<mpsc::UnboundedSender<InputEvent>> {
        self.input_tx.clone()
    }

    /// `end_input()`: flush, then close the input channel.
    pub fn end_input(&mut self) -> Result<(), StreamClosed> {
        self.flush()?;
        self.input_tx = None;
        Ok(())
    }

    /// The next verdict, or `None` once finished. Failures and span writes are skipped (as
    /// `async for ev in stream`); [`recv_event`](Self::recv_event) yields all three.
    pub async fn recv(&mut self) -> Option<OverlappingSpeechEvent> {
        loop {
            if let StreamEvent::Overlap(ev) = self.event_rx.recv().await? {
                return Some(*ev);
            }
        }
    }

    pub async fn recv_event(&mut self) -> Option<StreamEvent> {
        self.event_rx.recv().await
    }

    /// The options are shared, so this only forces a reconnect.
    pub fn request_reconnect(&self) {
        self.reconnect.notify_one();
    }

    /// `__anext__`'s tail (`inference/interruption.py:527`): once the events are exhausted, the
    /// task's exception belongs to the caller.
    pub async fn finish(mut self) -> Result<(), ApiError> {
        self.input_tx = None;
        join_task(self.task).await
    }

    /// `aclose()` -- cancel immediately, without waiting for the gateway to close back.
    pub async fn aclose(mut self) -> Result<(), ApiError> {
        self.input_tx = None;
        self.task.abort();
        join_task(self.task).await
    }
}

/// Shared by [`InterruptionStream::finish`] and [`InterruptionStream::aclose`]: a cancelled task
/// is a *successful* close (that is what `aclose` asks for), and a panicked one is not.
async fn join_task(task: JoinHandle<Result<(), ApiError>>) -> Result<(), ApiError> {
    match task.await {
        Ok(res) => res,
        Err(e) if e.is_cancelled() => Ok(()),
        Err(e) => Err(ApiError::new(format!("interruption stream task panicked: {e}"))),
    }
}

/// `_main_task` (`inference/interruption.py:462`): the retry ladder around [`run`].
async fn main_task(
    ctx: StreamContext,
    input_rx: mpsc::UnboundedReceiver<InputEvent>,
) -> Result<(), ApiError> {
    let result = main_task_inner(&ctx, input_rx).await;
    // `self._task.add_done_callback(lambda _: self._event_ch.close())` -- the consumer has to stop
    // waiting whether the task succeeded or failed.
    ctx.closed.store(true, Ordering::Release);
    result
}

async fn main_task_inner(
    ctx: &StreamContext,
    mut input_rx: mpsc::UnboundedReceiver<InputEvent>,
) -> Result<(), ApiError> {
    let mut num_retries = 0u32;
    loop {
        let err = match run(ctx, &mut input_rx).await {
            Ok(()) => return Ok(()),
            Err(e) => e,
        };
        // `_main_task`'s ladder (`inference/interruption.py:462`): every arm emits an error event
        // first; `recoverable` differs (#5727). `max_retry = 0` re-raises the real error.
        match retry::decide(RequestKind::Interruption, &ctx.config.conn_options, num_retries, err.error.retryable, None, false) {
            RetryDecision::Stop | RetryDecision::Raise => {
                ctx.emit_error(err.error.clone(), false);
                return Err(err.error);
            }
            RetryDecision::Exhausted(message) => {
                ctx.emit_error(err.error, false);
                return Err(ApiError::connection(Some(message)));
            }
            RetryDecision::Retry(after) => {
                ctx.emit_error(err.error, true);
                // On the session's clock.
                let resume = ctx.clock.now().saturating_add(Duration::from_secs_f64(after));
                ctx.clock.wait_until(resume).await;
                num_retries += 1;
            }
        }
    }
}

/// `_run`'s outer `while True` (`inference/interruption.py:944`): connect, pump, and go round again
/// if `update_options` asked for a reconnect.
async fn run(
    ctx: &StreamContext,
    input_rx: &mut mpsc::UnboundedReceiver<InputEvent>,
) -> Result<(), StreamError> {
    // The core outlives a reconnect, as in the reference: a mid-turn `update_options` keeps the
    // buffered agent audio.
    let core = {
        let opts = ctx.opts.lock().unwrap().clone();
        Arc::new(Mutex::new(InterruptionCore::new(opts)))
    };
    loop {
        let socket = connect_ws(ctx).await?;
        if !run_once(ctx, input_rx, &core, socket).await? {
            return Ok(());
        }
    }
}

/// One connection's lifetime. `Ok(true)` means "reconnect", `Ok(false)` means "the input ended".
async fn run_once(
    ctx: &StreamContext,
    input_rx: &mut mpsc::UnboundedReceiver<InputEvent>,
    core: &Arc<Mutex<InterruptionCore>>,
    socket: Socket,
) -> Result<bool, StreamError> {
    let Socket { sender: sink, receiver: mut source } = socket;
    let (data_tx, data_rx) = mpsc::unbounded_channel();
    // `closing_ws`: per connection, reset on reconnect; tells an orderly shutdown from a hang-up.
    let closing = Arc::new(AtomicBool::new(false));
    let mut send_task = tokio::spawn(send_task(
        ctx.clone(),
        core.clone(),
        Arc::clone(&sink),
        data_rx,
        closing.clone(),
    ));
    let mut input_open = true;
    let mut data_tx = Some(data_tx);

    let outcome = loop {
        tokio::select! {
            // Biased: a verdict closes the overlap and must not be raced by audio that would open
            // another request.
            biased;

            msg = source.recv() => {
                let code = source.close_code();
                match handle_incoming(ctx, core, msg, code, &closing)? {
                    Incoming::Continue => {}
                    // The gateway closed after we asked it to: this is the orderly end.
                    Incoming::Closed => break Ok(false),
                }
            }
            joined = &mut send_task => {
                break match joined {
                    Ok(Ok(())) => Ok(false),
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(StreamError {
                        error: ApiError::new(format!("interruption send task panicked: {e}")),
                        body: None,
                    }),
                };
            }
            _ = ctx.reconnect.notified() => break Ok(true),
            ev = input_rx.recv(), if input_open => match ev {
                Some(ev) => {
                    let wall = ctx.wall();
                    let outcome = {
                        let mut core = core.lock().unwrap();
                        core.on_input(ev, wall)
                    };
                    match outcome {
                        Ok(Some(InputOutcome::Infer(samples))) => {
                            if let Some(tx) = &data_tx {
                                let _ = tx.send(samples);
                            }
                        }
                        Ok(Some(InputOutcome::Overlap(ev))) => ctx.emit_overlap(*ev),
                        Ok(None) => {}
                        // `AudioArrayBuffer.push_frame` raises `ValueError`: non-recoverable, a
                        // caller bug a reconnect cannot fix.
                        Err(e) => break Err(StreamError {
                            error: ApiError::new(e.to_string()).with_retryable(false),
                            body: None,
                        }),
                    }
                }
                // `output_ch.close()` at the end of `_forward_data`: the send task drains what is
                // queued, writes `session.close`, and the gateway closes back.
                None => {
                    input_open = false;
                    data_tx = None;
                }
            },
        }
    };

    send_task.abort();
    sink.close().await;
    outcome
}

enum Incoming {
    Continue,
    Closed,
}

fn handle_incoming(
    ctx: &StreamContext,
    core: &Arc<Mutex<InterruptionCore>>,
    msg: Result<WsMessage, TransportError>,
    close_code: Option<u16>,
    closing: &AtomicBool,
) -> Result<Incoming, StreamError> {
    let msg = match msg {
        Ok(WsMessage::Text(text)) => text,
        // `ws_msg.type in (CLOSED, CLOSE, CLOSING)`: expected if we asked for it; with the input
        // still open, the gateway hung up (`ws.close_code or -1`).
        Ok(WsMessage::Closed) => {
            return if closing.load(Ordering::Acquire) {
                Ok(Incoming::Closed)
            } else {
                Err(closed_unexpectedly(close_code, "", ""))
            };
        }
        Ok(WsMessage::Other) => return Ok(Incoming::Continue),
        Err(e) => {
            return Err(StreamError {
                error: ApiError::connection(Some(format!(
                    "LiveKit Adaptive Interruption connection error: {e}"
                ))),
                body: None,
            });
        }
    };

    // An unknown `type` is a `ValidationError` in the reference and takes the connection down.
    let parsed: ServerMessage = serde_json::from_str(&msg).map_err(|e| StreamError {
        error: ApiError::new(format!("failed to parse LiveKit Adaptive Interruption message: {e}"))
            .with_retryable(false),
        body: Some(msg.to_string()),
    })?;

    let now = ctx.clock.now();
    let wall = ctx.wall();
    let outcome = core.lock().unwrap().on_server_message(parsed, now, wall)?;
    match outcome {
        RecvOutcome::Detected { span_update, event } => {
            if let Some((span, attrs)) = span_update {
                let _ = ctx.stream_events.send(StreamEvent::SpanUpdate { span, attrs });
            }
            ctx.emit_overlap(*event);
        }
        RecvOutcome::Ignored => {}
    }
    Ok(Incoming::Continue)
}

/// `send_task` (`inference/interruption.py:796`).
async fn send_task(
    ctx: StreamContext,
    core: Arc<Mutex<InterruptionCore>>,
    sink: Arc<dyn WsSender>,
    mut data_rx: mpsc::UnboundedReceiver<Vec<i16>>,
    closing: Arc<AtomicBool>,
) -> Result<(), StreamError> {
    while let Some(samples) = data_rx.recv().await {
        let now = ctx.clock.now();
        core.lock().unwrap().check_inflight_timeout(now)?;

    // Stamped just before the write, so the round trip excludes queueing time.
        let created_at = ctx.clock.now();
        let payload = frame_request(created_at.as_nanos(), &samples);
        sink.send_binary(payload).await.map_err(send_failed)?;
        core.lock().unwrap().record_request(created_at, samples);
    }

    closing.store(true, Ordering::Release);
    let close = serde_json::to_string(&ClientMessage::SessionClose)
        .expect("ClientMessage::SessionClose is a two-field struct with no failure mode");
    sink.send_text(close).await.map_err(send_failed)?;
    Ok(())
}

fn send_failed(e: TransportError) -> StreamError {
    StreamError {
        error: ApiError::connection(Some(format!(
            "failed to send to LiveKit Adaptive Interruption: {e}"
        ))),
        body: None,
    }
}

/// `_connect_ws` (`inference/interruption.py:977`).
async fn connect_ws(ctx: &StreamContext) -> Result<Socket, StreamError> {
    let (url, settings, api_key, api_secret) = {
        let opts = ctx.opts.lock().unwrap();
        (
            // `_connect_ws`'s URL (`inference/interruption.py:986`).
            format!("{}/bargein", crate::transport::to_ws_url(&opts.base_url)),
            SessionCreateSettings {
                sample_rate: opts.sample_rate,
                num_channels: 1,
                threshold: opts.threshold,
                min_frames: opts.min_frames,
                encoding: ENCODING_S16LE,
            },
            opts.api_key.clone(),
            opts.api_secret.clone(),
        )
    };

    // The token's `iat`/`exp` come from the session clock's `wall()`.
    let now_unix = ctx.clock.wall().duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs();
    let token = create_access_token(&api_key, &api_secret, ctx.config.token_ttl_secs, now_unix)
        .map_err(|e| StreamError {
            error: ApiError::new(format!("failed to mint an inference access token: {e}"))
                .with_retryable(false),
            body: None,
        })?;

    // An unusable URL is the caller's mistake, which no retry fixes.
    {
        use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
        if let Err(e) = url.as_str().into_client_request() {
            return Err(StreamError {
                error: ApiError::new(format!("invalid inference base_url {url:?}: {e}"))
                    .with_retryable(false),
                body: None,
            });
        }
    }
    let mut headers: Vec<(String, String)> = (ctx.config.headers)()
        .into_iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("authorization"))
        .collect();
    headers.push(("Authorization".to_owned(), format!("Bearer {token}")));

    let timeout = Duration::from_secs_f64(ctx.config.conn_options.timeout.max(0.0));
    let connect = ctx.config.transport.connect(&url, headers);
    let (sender, receiver) = match lk_agents_abi::timeout(&*ctx.clock, timeout, connect).await {
        Err(_) | Ok(Err(TransportError::Timeout)) => {
            return Err(StreamError { error: connect_error(ConnectFailure::Timeout), body: None });
        }
        Ok(Err(TransportError::Status { status, .. })) => {
            return Err(StreamError {
                error: connect_error(ConnectFailure::Http { status }),
                body: None,
            });
        }
        Ok(Err(_other)) => {
            return Err(StreamError {
                error: connect_error(ConnectFailure::Transport),
                body: None,
            });
        }
        Ok(Ok(halves)) => halves,
    };

    let create = ClientMessage::SessionCreate { settings };
    let json = serde_json::to_string(&create)
        .expect("SessionCreateSettings is plain data with no non-finite defaults");
    if sender.send_text(json).await.is_err() {
        // `await ws.close()` before raising, or the half-open socket lingers through the retries.
        sender.close().await;
        return Err(StreamError { error: session_create_failed(), body: None });
    }
    Ok(Socket { sender, receiver })
}
