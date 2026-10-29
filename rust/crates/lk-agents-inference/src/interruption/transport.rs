//! The driver: one WebSocket, a reconnect ladder, and the effects [`InterruptionCore`] asked for.
//! `_forward_data` and `recv_task` (`inference/interruption.py:793`) are `select!` arms in
//! `run_once`; only the send half is a task. The `Mutex<InterruptionCore>` is never held across an
//! `.await`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, UNIX_EPOCH};

use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::metrics::base::InterruptionMetrics;
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::{Clock, HostRef};
use lk_agents_net::attempts::RequestPolicy;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, mpsc, watch};
use tokio::task::JoinHandle;

use crate::headers::HeaderProvider;
use crate::interruption::core::{
    InputEvent, InterruptionCore, InterruptionEffect, InterruptionEvent, RequestIdentity,
    SpanAttributes, StreamError,
};
use crate::interruption::event::{
    InterruptionDetectionError, OverlappingSpeechEvent, interruption_metrics,
};
use crate::interruption::options::InterruptionOptions;
use crate::interruption::proto::{
    ClientMessage, ENCODING_S16LE, ServerMessage, SessionCreateSettings, frame_request,
};
use crate::interruption::retry::{
    ConnectFailure, closed_unexpectedly, connect_error, session_create_failed,
};
use crate::interruption::{LABEL, MODEL, PROVIDER};
use lk_agents_net::token::{INFERENCE_TOKEN_TTL, inference_bearer_with_ttl};
use lk_agents_net::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};

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
            transport: Arc::new(lk_agents_net::websocket::TungsteniteTransport),
            headers: Arc::new(Vec::new),
            token_ttl_secs: INFERENCE_TOKEN_TTL.as_secs(),
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
        streams.retain(|s| match s.upgrade() {
            Some(reconnect) => {
                reconnect.notify_one();
                true
            }
            None => false,
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

    fn open(&self, fan_out: bool) -> InterruptionStream {
        let reconnect = Arc::new(Notify::new());
        self.streams.lock().unwrap().push(Arc::downgrade(&reconnect));

        let (input_tx, input_rx) = mpsc::channel(INPUT_CAPACITY);
        let (input_failure, _) = watch::channel(None);
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let closed = Arc::new(AtomicBool::new(false));

        let ctx = StreamContext {
            opts: self.opts.clone(),
            config: self.config.clone(),
            clock: self.clock.clone(),
            detector_events: fan_out.then(|| self.events.clone()),
            stream_events: event_tx,
            reconnect: reconnect.clone(),
            closed: closed.clone(),
            input_failure: input_failure.clone(),
        };
        let input_tx = InterruptionInput {
            sender: input_tx,
            bytes: Arc::new(Semaphore::new(INPUT_BYTES)),
            failure: input_failure,
        };
        let task = tokio::spawn(main_task(ctx, input_rx));
        InterruptionStream { input_tx: Some(input_tx), event_rx, reconnect, closed, task }
    }
}

/// What one stream reports to its caller: a verdict, the failure (recoverable or not) that
/// explains why none came, so the session can hand interruption back to the VAD (I-24), or the verdict
/// to write on the span the caller pushed (a span is a host object, §1.11).
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Overlap(Box<OverlappingSpeechEvent>),
    CorrelatedOverlap { request: ReqId, event: Box<OverlappingSpeechEvent> },
    Error(Box<InterruptionDetectionError>),
    CorrelatedError { request: ReqId, error: Box<InterruptionDetectionError> },
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
    input_failure: watch::Sender<Option<ApiError>>,
}

impl StreamContext {
    async fn input_failed(&self) -> StreamError {
        let mut failure = self.input_failure.subscribe();
        loop {
            if let Some(error) = failure.borrow_and_update().clone() {
                return StreamError { request: None, error, body: None };
            }
            let _ = failure.changed().await;
        }
    }

    fn wall(&self) -> f64 {
        self.clock.wall().duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs_f64()
    }

    /// `self.send(ev)` plus `_metrics_monitor_task`'s fold, inline.
    fn emit_overlap(&self, request: Option<ReqId>, ev: OverlappingSpeechEvent) {
        let metrics = interruption_metrics(&ev, self.wall(), MODEL, PROVIDER);
        let event = match request {
            Some(request) => {
                StreamEvent::CorrelatedOverlap { request, event: Box::new(ev.clone()) }
            }
            None => StreamEvent::Overlap(Box::new(ev.clone())),
        };
        let _ = self.stream_events.send(event);
        if let Some(detector) = &self.detector_events {
            let _ = detector.send(DetectorEvent::OverlappingSpeech(Box::new(ev)));
            let _ = detector.send(DetectorEvent::MetricsCollected(Box::new(metrics)));
        }
    }

    fn emit_effect(&self, effect: InterruptionEffect) -> Option<(RequestIdentity, Vec<i16>)> {
        match effect {
            InterruptionEffect::InterruptionInference { identity, samples } => {
                Some((identity, samples))
            }
            InterruptionEffect::Overlap { request, event } => {
                self.emit_overlap(request, *event);
                None
            }
            InterruptionEffect::SpanUpdate { span, attrs } => {
                let _ = self.stream_events.send(StreamEvent::SpanUpdate { span, attrs });
                None
            }
        }
    }

    /// The `"error"` channel, and the same failure on the stream's own channel.
    fn emit_error(&self, request: Option<ReqId>, error: ApiError, recoverable: bool) {
        let err = Box::new(InterruptionDetectionError {
            timestamp: self.wall(),
            label: LABEL.to_owned(),
            error,
            recoverable,
        });
        let event = match request {
            Some(request) => StreamEvent::CorrelatedError { request, error: err.clone() },
            None => StreamEvent::Error(err.clone()),
        };
        let _ = self.stream_events.send(event);
        if let Some(detector) = &self.detector_events {
            let _ = detector.send(DetectorEvent::Error(err));
        }
    }
}

// Five seconds of 20 ms input plus control events; retained audio is also bounded by bytes.
const INPUT_CAPACITY: usize = 256;
const INPUT_BYTES: usize = 2 * 1024 * 1024;

struct QueuedInput {
    event: InputEvent,
    _bytes: OwnedSemaphorePermit,
}

/// The stream's synchronous input half. Saturation is terminal: silently losing a speech
/// boundary or an audio frame would leave the detector deciding against a different sequence.
#[derive(Clone)]
pub struct InterruptionInput {
    sender: mpsc::Sender<QueuedInput>,
    bytes: Arc<Semaphore>,
    failure: watch::Sender<Option<ApiError>>,
}

impl InterruptionInput {
    pub fn send(&self, event: InputEvent) -> Result<(), StreamClosed> {
        if self.failure.borrow().is_some() {
            return Err(StreamClosed::CapacityExceeded);
        }
        if self.sender.is_closed() {
            return Err(StreamClosed::InputEnded);
        }
        let bytes = match &event {
            InputEvent::Audio(frame) => frame.data.len(),
            _ => 0,
        };
        let reservation = u32::try_from(bytes)
            .ok()
            .filter(|&bytes| bytes as usize <= INPUT_BYTES)
            .and_then(|bytes| self.bytes.clone().try_acquire_many_owned(bytes).ok());
        let Some(reservation) = reservation else {
            return Err(self.overloaded());
        };
        self.sender.try_send(QueuedInput { event, _bytes: reservation }).map_err(
            |error| match error {
                mpsc::error::TrySendError::Full(_) => self.overloaded(),
                mpsc::error::TrySendError::Closed(_) => StreamClosed::InputEnded,
            },
        )
    }

    fn overloaded(&self) -> StreamClosed {
        self.failure.send_replace(Some(
            ApiError::new("interruption input queue capacity exceeded").with_retryable(false),
        ));
        StreamClosed::CapacityExceeded
    }
}

async fn while_input_open<F: std::future::Future>(
    ctx: &StreamContext,
    future: F,
) -> Result<F::Output, StreamError> {
    use futures_util::future::{Either, select};
    match select(Box::pin(ctx.input_failed()), Box::pin(future)).await {
        Either::Left((error, _)) => Err(error),
        Either::Right((value, _)) => Ok(value),
    }
}

/// `InterruptionWebSocketStream`, from the caller's side.
pub struct InterruptionStream {
    /// `Option` so `end_input()` can drop the sender (`self._input_ch.close()`).
    input_tx: Option<InterruptionInput>,
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
    #[error("interruption input queue capacity exceeded")]
    CapacityExceeded,
}

impl From<StreamClosed> for lk_agents_abi::errors::ProviderError {
    fn from(error: StreamClosed) -> Self {
        ApiError::new(error.to_string()).with_retryable(false).into()
    }
}

impl InterruptionStream {
    /// `push_frame`. Event channel closed = the stream failed; input channel closed = the caller
    /// already ended input.
    pub fn push(&self, ev: InputEvent) -> Result<(), StreamClosed> {
        if self.closed.load(Ordering::Acquire) {
            return Err(StreamClosed::Closed);
        }
        let tx = self.input_tx.as_ref().ok_or(StreamClosed::InputEnded)?;
        tx.send(ev)
    }

    /// `flush()`.
    pub fn flush(&self) -> Result<(), StreamClosed> {
        self.push(InputEvent::Flush)
    }

    /// A clone of the input sender, for pushing from another thread than the one reading
    /// verdicts. A holder must drop it when it ends input, or the stream's task parks forever.
    pub fn sender(&self) -> Option<InterruptionInput> {
        self.input_tx.clone()
    }

    /// Close input after all accepted events. The core's flush is a no-op, so EOF does not
    /// consume another queue entry (including when the last accepted event filled the queue).
    pub fn end_input(&mut self) -> Result<(), StreamClosed> {
        self.input_tx = None;
        Ok(())
    }

    /// The next verdict, or `None` once finished. Failures and span writes are skipped (as
    /// `async for ev in stream`); [`recv_event`](Self::recv_event) yields all three.
    pub async fn recv(&mut self) -> Option<OverlappingSpeechEvent> {
        loop {
            match self.event_rx.recv().await? {
                StreamEvent::Overlap(event) | StreamEvent::CorrelatedOverlap { event, .. } => {
                    return Some(*event);
                }
                _ => {}
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
        join_task(self.task).await?
    }

    /// `aclose()` -- cancel immediately, without waiting for the gateway to close back.
    pub async fn aclose(mut self) -> Result<(), ApiError> {
        self.input_tx = None;
        self.task.abort();
        join_task(self.task).await.map(|_| ())
    }
}

/// Joining settles the task separately from the provider's operation result. `finish`
/// retrieves that result; `aclose` only owns cancellation and settlement.
async fn join_task(
    task: JoinHandle<Result<(), ApiError>>,
) -> Result<Result<(), ApiError>, ApiError> {
    match task.await {
        Ok(result) => Ok(result),
        Err(error) if error.is_cancelled() => Ok(Ok(())),
        Err(error) => Err(ApiError::new(format!("interruption stream task panicked: {error}"))),
    }
}

/// `_main_task` (`inference/interruption.py:462`): the retry ladder around [`run`].
async fn main_task(
    ctx: StreamContext,
    input_rx: mpsc::Receiver<QueuedInput>,
) -> Result<(), ApiError> {
    let result = main_task_inner(&ctx, input_rx).await;
    // `self._task.add_done_callback(lambda _: self._event_ch.close())` -- the consumer has to stop
    // waiting whether the task succeeded or failed.
    ctx.closed.store(true, Ordering::Release);
    result
}

async fn main_task_inner(
    ctx: &StreamContext,
    mut input_rx: mpsc::Receiver<QueuedInput>,
) -> Result<(), ApiError> {
    let policy = RequestPolicy::from_options(&ctx.config.conn_options);
    let mut num_retries = 0u32;
    loop {
        let err = match run(ctx, &mut input_rx).await {
            Ok(()) => return Ok(()),
            Err(e) => e,
        };
        if let Some(delay) = policy.retry_delay(num_retries, err.error.retryable) {
            ctx.emit_error(err.request, err.error, true);
            let resume = ctx.clock.now().saturating_add(delay);
            if let Err(error) = while_input_open(ctx, ctx.clock.wait_until(resume)).await {
                ctx.emit_error(error.request, error.error.clone(), false);
                return Err(error.error);
            }
            num_retries += 1;
        } else {
            ctx.emit_error(err.request, err.error.clone(), false);
            return match lk_agents_net::attempts::terminal_failure(
                RequestKind::Interruption,
                &err.error,
                ctx.config.conn_options.max_retry,
                num_retries,
            ) {
                RetryDecision::Exhausted(message) => Err(ApiError::connection(Some(message))),
                _ => Err(err.error),
            };
        }
    }
}

/// `_run`'s outer `while True` (`inference/interruption.py:944`): connect, pump, and go round again
/// if `update_options` asked for a reconnect.
async fn run(
    ctx: &StreamContext,
    input_rx: &mut mpsc::Receiver<QueuedInput>,
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
    input_rx: &mut mpsc::Receiver<QueuedInput>,
    core: &Arc<Mutex<InterruptionCore>>,
    socket: Socket,
) -> Result<bool, StreamError> {
    let Socket { sender: sink, receiver: mut source } = socket;
    let opened = core.lock().unwrap().handle(
        InterruptionEvent::ConnectionOpened,
        ctx.clock.now(),
        ctx.wall(),
    );
    if let Err(error) = opened {
        sink.close().await;
        return Err(error);
    }
    let connection = core.lock().unwrap().connection();
    let (data_tx, data_rx) = mpsc::unbounded_channel();
    // `closing_ws`: per connection, reset on reconnect; tells an orderly shutdown from a hang-up.
    let closing = Arc::new(AtomicBool::new(false));
    let mut send_task =
        tokio::spawn(send_task(core.clone(), Arc::clone(&sink), data_rx, closing.clone()));
    let mut send_finished = false;
    let mut input_open = true;
    let mut data_tx = Some(data_tx);

    let outcome = loop {
        let deadline = core.lock().unwrap().next_deadline();
        let inference_deadline = async {
            match deadline {
                Some(deadline) => ctx.clock.wait_until(deadline).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::select! {
            // Biased: a verdict closes the overlap and must not be raced by audio that would open
            // another request.
            biased;

            error = ctx.input_failed() => break Err(error),
            _ = inference_deadline => {
                let outcome = core.lock().unwrap().handle(
                    InterruptionEvent::Deadline, ctx.clock.now(), ctx.wall(),
                );
                if let Err(error) = outcome {
                    break Err(error);
                }
            }
            msg = source.recv() => {
                let code = source.close_code();
                match handle_incoming(ctx, core, connection, msg, code, &closing) {
                    Err(error) => break Err(error),
                    Ok(Incoming::Continue) => {}
                    // The gateway closed after we asked it to: this is the orderly end.
                    Ok(Incoming::Closed) => break Ok(false),
                }
            }
            joined = &mut send_task => {
                send_finished = true;
                break match joined {
                    Ok(Ok(())) => Ok(false),
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(StreamError {
                        request: None,
                        error: ApiError::new(format!("interruption send task panicked: {e}")),
                        body: None,
                    }),
                };
            }
            _ = ctx.reconnect.notified() => break Ok(true),
            ev = input_rx.recv(), if input_open => match ev {
                Some(queued) => {
                    let ev = queued.event;
                    let wall = ctx.wall();
                    let outcome = {
                        let mut core = core.lock().unwrap();
                        core.handle(InterruptionEvent::Input(ev), ctx.clock.now(), wall)
                    };
                    match outcome {
                        Ok(effects) => {
                            for effect in effects {
                                if let Some(request) = ctx.emit_effect(effect)
                                    && let Some(tx) = &data_tx {
                                    let _ = tx.send(request);
                                }
                            }
                        }
                        Err(error) => break Err(error),
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
    if !send_finished {
        let _ = send_task.await;
    }
    sink.close().await;
    outcome.map_err(|mut error| {
        if error.request.is_none() {
            error.request = core.lock().unwrap().overlap_request();
        }
        error
    })
}

enum Incoming {
    Continue,
    Closed,
}

fn handle_incoming(
    ctx: &StreamContext,
    core: &Arc<Mutex<InterruptionCore>>,
    connection: u64,
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
                request: None,
                error: ApiError::connection(Some(format!(
                    "LiveKit Adaptive Interruption connection error: {e}"
                ))),
                body: None,
            });
        }
    };

    // An unknown `type` is a `ValidationError` in the reference and takes the connection down.
    let parsed: ServerMessage = serde_json::from_str(&msg).map_err(|e| StreamError {
        request: None,
        error: ApiError::new(format!("failed to parse LiveKit Adaptive Interruption message: {e}"))
            .with_retryable(false),
        body: Some(msg.to_string()),
    })?;

    let now = ctx.clock.now();
    let wall = ctx.wall();
    let effects = core.lock().unwrap().handle(
        InterruptionEvent::ServerMessage { connection, message: parsed },
        now,
        wall,
    )?;
    for effect in effects {
        ctx.emit_effect(effect);
    }
    Ok(Incoming::Continue)
}

/// `send_task` (`inference/interruption.py:796`).
async fn send_task(
    core: Arc<Mutex<InterruptionCore>>,
    sink: Arc<dyn WsSender>,
    mut data_rx: mpsc::UnboundedReceiver<(RequestIdentity, Vec<i16>)>,
    closing: Arc<AtomicBool>,
) -> Result<(), StreamError> {
    while let Some((identity, samples)) = data_rx.recv().await {
        if !core.lock().unwrap().request_eligible(identity) {
            continue;
        }
        let payload = frame_request(identity.request_id, &samples);
        sink.send_binary(payload).await.map_err(send_failed)?;
    }

    closing.store(true, Ordering::Release);
    let close = serde_json::to_string(&ClientMessage::SessionClose)
        .expect("ClientMessage::SessionClose is a two-field struct with no failure mode");
    sink.send_text(close).await.map_err(send_failed)?;
    Ok(())
}

fn send_failed(e: TransportError) -> StreamError {
    StreamError {
        request: None,
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
            format!("{}/bargein", lk_agents_net::transport::to_ws_url(&opts.base_url)),
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

    let token = inference_bearer_with_ttl(
        &api_key,
        &api_secret,
        Duration::from_secs(ctx.config.token_ttl_secs),
    )
    .map_err(|e| StreamError {
        request: None,
        error: ApiError::new(format!("failed to mint an inference access token: {e}"))
            .with_retryable(false),
        body: None,
    })?;

    // An unusable URL is the caller's mistake, which no retry fixes.
    {
        use lk_agents_net::websocket::tungstenite::client::IntoClientRequest as _;
        if let Err(e) = url.as_str().into_client_request() {
            return Err(StreamError {
                request: None,
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

    let deadline = ctx.config.conn_options.connect_deadline(ctx.clock.now());
    let connect = ctx.config.transport.connect(&url, headers);
    let (sender, receiver) =
        match while_input_open(ctx, lk_agents_abi::before(&*ctx.clock, deadline, connect)).await? {
            Err(_) | Ok(Err(TransportError::Timeout)) => {
                return Err(StreamError {
                    request: None,
                    error: connect_error(ConnectFailure::Timeout),
                    body: None,
                });
            }
            Ok(Err(TransportError::Status { status, .. })) => {
                return Err(StreamError {
                    request: None,
                    error: connect_error(ConnectFailure::Http { status }),
                    body: None,
                });
            }
            Ok(Err(_other)) => {
                return Err(StreamError {
                    request: None,
                    error: connect_error(ConnectFailure::Transport),
                    body: None,
                });
            }
            Ok(Ok(halves)) => halves,
        };

    let create = ClientMessage::SessionCreate { settings };
    let json = serde_json::to_string(&create)
        .expect("SessionCreateSettings is plain data with no non-finite defaults");
    match while_input_open(ctx, sender.send_text(json)).await {
        Ok(Ok(())) => {}
        outcome => {
            sender.close().await;
            return Err(match outcome {
                Err(error) => error,
                _ => StreamError { request: None, error: session_create_failed(), body: None },
            });
        }
    }
    Ok(Socket { sender, receiver })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interruption::{
        AUDIO_PREFIX_DURATION, DETECTION_INTERVAL, MAX_AUDIO_DURATION, MIN_INTERRUPTION_DURATION,
        OverlapStart, REMOTE_INFERENCE_TIMEOUT,
    };
    use crate::testkit::FakeWs;

    fn detector(transport: Arc<FakeWs>) -> AdaptiveInterruptionDetector {
        let opts = InterruptionOptions::new(
            Some(0.7),
            MIN_INTERRUPTION_DURATION,
            MAX_AUDIO_DURATION,
            AUDIO_PREFIX_DURATION,
            DETECTION_INTERVAL,
            REMOTE_INFERENCE_TIMEOUT,
            "ws://localhost".into(),
            "key".into(),
            "secret".into(),
        )
        .unwrap();
        AdaptiveInterruptionDetector::new(
            opts,
            ConnectionConfig {
                transport,
                conn_options: ApiConnectOptions { max_retry: 0, ..Default::default() },
                ..Default::default()
            },
            Arc::new(lk_agents_abi::RealClock::new()),
        )
        .0
    }

    #[tokio::test]
    async fn stalled_input_refuses_overload_and_releases_the_detector_to_fallback() {
        let audio_bytes = 16_000 * 3 * 2;
        let cases = [
            (InputEvent::Flush, INPUT_CAPACITY + 1),
            (
                InputEvent::Audio(lk_agents_abi::AudioFrame {
                    data: vec![0; audio_bytes].into(),
                    sample_rate: 16_000,
                    num_channels: 1,
                    samples_per_channel: 48_000,
                }),
                INPUT_BYTES / audio_bytes + 1,
            ),
        ];
        for (input, count) in cases {
            let detector = detector(Arc::new(FakeWs::hanging()));
            let mut stream = detector.private_stream();
            let refused = (0..count).any(|_| stream.push(input.clone()).is_err());
            if !refused {
                stream.aclose().await.unwrap();
                panic!("a stalled interruption stream accepted unbounded input");
            }
            let event = lk_agents_abi::timeout(
                &lk_agents_abi::RealClock::new(),
                Duration::from_secs(1),
                stream.recv_event(),
            )
            .await
            .expect("queue failure must reach the caller without waiting for connect timeout");
            assert!(matches!(event, Some(StreamEvent::Error(error))
                if !error.recoverable && error.error.message.contains("capacity")));
            assert!(stream.finish().await.is_err());
        }
    }

    #[tokio::test]
    async fn end_input_sends_accepted_audio_before_session_close() {
        let peer = Arc::new(FakeWs::new(Vec::new()).keep_open());
        let detector = detector(peer.clone());
        let mut stream = detector.private_stream();
        stream.push(InputEvent::AgentSpeechStarted).unwrap();
        stream
            .push(InputEvent::OverlapSpeechStarted(OverlapStart {
                speech_duration: 0.0,
                started_at: 1.0,
                user_speaking_span: None,
            }))
            .unwrap();
        stream
            .push(InputEvent::Audio(lk_agents_abi::AudioFrame {
                data: vec![0; 3200].into(),
                sample_rate: 16_000,
                num_channels: 1,
                samples_per_channel: 1600,
            }))
            .unwrap();
        for _ in 3..INPUT_CAPACITY {
            stream.flush().unwrap();
        }
        stream.end_input().unwrap();
        lk_agents_abi::timeout(
            &lk_agents_abi::RealClock::new(),
            Duration::from_secs(1),
            stream.finish(),
        )
        .await
        .expect("accepted input must drain before close")
        .unwrap();
        let sent = peer.sent();
        assert_eq!(sent.len(), 3, "create, accepted inference, then close: {sent:?}");
        assert!(sent[1].starts_with('<'), "accepted audio was discarded: {sent:?}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&sent[2]).unwrap()["type"],
            serde_json::json!("session.close")
        );
    }
}
