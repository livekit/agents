//! The driver: one WebSocket, a reconnect ladder, and the effects [`InterruptionMachine`] asked for.
//! `_forward_data` and `recv_task` (`inference/interruption.py:793`) are `select!` arms in
//! `run_once`; only the send half is a task. The `Mutex<InterruptionMachine>` is never held across an
//! `.await`.

use crate::effects::InterruptionEffect;
use crate::headers::HeaderProvider;
use crate::interruption::LABEL;
use crate::interruption::event::{InterruptionDetectionError, OverlappingSpeechEvent};
use crate::interruption::machine::{
    InputEvent, InterruptionEvent, InterruptionMachine, RequestIdentity, StreamError,
};
use crate::interruption::options::InterruptionOptions;
use crate::interruption::proto::{
    ClientMessage, ENCODING_S16LE, ServerMessage, SessionCreateSettings, frame_request,
};
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::Clock;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::types::retry::{RequestKind, RetryDecision};
use lk_agents_abi::{OwnedTask, SpawnerExt, TaskError};
use lk_agents_core::attempts::RequestPolicy;
use lk_agents_media::media_queue::{self, AudioSender, Item, Policy};
use lk_agents_net::audio_drops::AudioDrops;
use lk_agents_net::task::TokioSpawner;
use lk_agents_net::telemetry;
use lk_agents_net::token::{INFERENCE_TOKEN_TTL, inference_bearer_with_ttl};
use lk_agents_net::transport::{
    TransportError, WsMessage, WsReceiver, WsSender, WsTransport, close_status,
};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, UNIX_EPOCH};
use tokio::sync::{Notify, mpsc, watch};

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

/// `AdaptiveInterruptionDetector` (`inference/interruption.py:264`).
pub struct AdaptiveInterruptionDetector {
    opts: Arc<Mutex<InterruptionOptions>>,
    config: ConnectionConfig,
    clock: Arc<dyn Clock>,
    /// `weakref.WeakSet[InterruptionWebSocketStream]`, so `update_options` reaches live streams
    /// without keeping dead ones alive.
    streams: Mutex<Vec<Weak<Notify>>>,
}

impl AdaptiveInterruptionDetector {
    pub fn new(opts: InterruptionOptions, config: ConnectionConfig, clock: Arc<dyn Clock>) -> Self {
        AdaptiveInterruptionDetector {
            opts: Arc::new(Mutex::new(opts)),
            config,
            clock,
            streams: Mutex::new(Vec::new()),
        }
    }

    /// `update_options` (`inference/interruption.py:391`). The options are shared with live
    /// streams, which are only nudged to reconnect; the next `session.create` carries them.
    pub fn update_options(&self, threshold: Option<f64>, min_interruption_duration: Option<f64>) {
        self.opts.lock().update(threshold, min_interruption_duration);
        let mut streams = self.streams.lock();
        streams.retain(|s| match s.upgrade() {
            Some(reconnect) => {
                reconnect.notify_one();
                true
            }
            None => false,
        });
    }

    /// `stream(conn_options=...)`: its verdicts and errors reach only this stream's reader.
    pub fn stream(&self) -> InterruptionStream {
        let reconnect = Arc::new(Notify::new());
        self.streams.lock().push(Arc::downgrade(&reconnect));

        let (queue, input_rx) = media_queue::audio(INPUT_POLICY);
        let (input_failure, _) = watch::channel(None);
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let closed = Arc::new(AtomicBool::new(false));

        let ctx = StreamContext {
            opts: self.opts.clone(),
            config: self.config.clone(),
            clock: self.clock.clone(),
            stream_events: event_tx,
            reconnect: reconnect.clone(),
            closed: closed.clone(),
            input_failure: input_failure.clone(),
        };
        let input_tx = InterruptionInput(Arc::new(InputShared {
            queue: Mutex::new(Some(queue)),
            drops: AudioDrops::new(telemetry::AudioConsumer::Interruption, INPUT_POLICY),
            failure: input_failure,
        }));
        let task = TokioSpawner::current().spawn(main_task(ctx, input_rx));
        InterruptionStream { input_tx: Some(input_tx), event_rx, reconnect, closed, task }
    }
}

/// What one stream reports to its caller: a verdict, the failure (recoverable or not) that
/// explains why none came, so the session can hand interruption back to the VAD (I-24).
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Overlap(Box<OverlappingSpeechEvent>),
    CorrelatedOverlap { request: ReqId, event: Box<OverlappingSpeechEvent> },
    Error(Box<InterruptionDetectionError>),
    CorrelatedError { request: ReqId, error: Box<InterruptionDetectionError> },
}

#[derive(Clone)]
struct StreamContext {
    opts: Arc<Mutex<InterruptionOptions>>,
    config: ConnectionConfig,
    clock: Arc<dyn Clock>,
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

    fn emit_overlap(&self, request: Option<ReqId>, ev: OverlappingSpeechEvent) {
        let event = Box::new(ev);
        let _ = self.stream_events.send(match request {
            Some(request) => StreamEvent::CorrelatedOverlap { request, event },
            None => StreamEvent::Overlap(event),
        });
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
            Some(request) => StreamEvent::CorrelatedError { request, error: err },
            None => StreamEvent::Error(err),
        };
        let _ = self.stream_events.send(event);
    }
}

/// Only current audio: past 500 ms of lag the window restarts on the newest.
const INPUT_POLICY: Policy = Policy::Now(Duration::from_millis(500));

/// The input queue: audio, and the sentinels in order with it.
type InputRx = media_queue::AudioReceiver<InputEvent>;

/// The stream's synchronous input half. Audio the queue drops reaches the detector as a
/// [`InputEvent::Gap`], which restarts its window; a sentinel is never dropped, so a detector
/// that stops draining them fails the stream instead.
#[derive(Clone)]
pub struct InterruptionInput(Arc<InputShared>);

struct InputShared {
    /// `None` once the input ended.
    queue: Mutex<Option<AudioSender<InputEvent>>>,
    drops: AudioDrops,
    failure: watch::Sender<Option<ApiError>>,
}

impl InterruptionInput {
    pub fn send(&self, event: InputEvent) -> Result<(), StreamClosed> {
        let shared = &*self.0;
        if shared.failure.borrow().is_some() {
            return Err(StreamClosed::Stuck);
        }
        let queue = shared.queue.lock();
        let queue = queue.as_ref().filter(|queue| !queue.is_closed());
        let queue = queue.ok_or(StreamClosed::InputEnded)?;
        match event {
            InputEvent::Audio(frame) => shared.drops.record(queue.push(frame)),
            event => queue.event(event).map_err(|stuck| {
                shared.failure.send_replace(Some(ApiError::new(stuck.to_string())));
                StreamClosed::Stuck
            })?,
        }
        Ok(())
    }

    /// Measures the audio the queue drops under `cx`'s job.
    pub fn observe(&self, cx: telemetry::Cx) {
        self.0.drops.observe(cx);
    }

    /// Ends the input for every clone: the detector drains what was queued, then closes.
    fn end(&self) {
        drop(self.0.queue.lock().take());
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
    task: OwnedTask<Result<(), ApiError>>,
}

/// `RuntimeError("... is closed")` / `RuntimeError("... input ended")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StreamClosed {
    #[error("livekit.agents.inference.interruption.InterruptionWebSocketStream is closed")]
    Closed,
    #[error("livekit.agents.inference.interruption.InterruptionWebSocketStream input ended")]
    InputEnded,
    #[error("interruption input stopped draining")]
    Stuck,
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
    /// verdicts.
    pub fn sender(&self) -> Option<InterruptionInput> {
        self.input_tx.clone()
    }

    /// Close input after all accepted events, for this stream and every [`sender`](Self::sender)
    /// clone. The core's flush is a no-op, so EOF does not take another queue entry.
    pub fn end_input(&mut self) -> Result<(), StreamClosed> {
        if let Some(input) = self.input_tx.take() {
            input.end();
        }
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
        let _ = self.end_input();
        join_task(self.task).await?
    }

    /// `aclose()` -- cancel immediately, without waiting for the gateway to close back.
    pub async fn aclose(mut self) -> Result<(), ApiError> {
        let _ = self.end_input();
        self.task.cancel();
        join_task(self.task).await.map(|_| ())
    }
}

/// Joining settles the task separately from the provider's operation result. `finish`
/// retrieves that result; `aclose` only owns cancellation and settlement.
async fn join_task(
    task: OwnedTask<Result<(), ApiError>>,
) -> Result<Result<(), ApiError>, ApiError> {
    match task.await {
        Ok(result) => Ok(result),
        Err(TaskError::Cancelled) => Ok(Ok(())),
        Err(error) => Err(ApiError::new(format!("interruption stream task panicked: {error}"))),
    }
}

/// `_main_task` (`inference/interruption.py:462`): the retry ladder around [`run`].
async fn main_task(ctx: StreamContext, input_rx: InputRx) -> Result<(), ApiError> {
    let result = main_task_inner(&ctx, input_rx).await;
    // `self._task.add_done_callback(lambda _: self._event_ch.close())` -- the consumer has to stop
    // waiting whether the task succeeded or failed.
    ctx.closed.store(true, Ordering::Release);
    result
}

async fn main_task_inner(ctx: &StreamContext, mut input_rx: InputRx) -> Result<(), ApiError> {
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
            return match lk_agents_core::attempts::terminal_failure(
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
async fn run(ctx: &StreamContext, input_rx: &mut InputRx) -> Result<(), StreamError> {
    // The core outlives a reconnect, as in the reference: a mid-turn `update_options` keeps the
    // buffered agent audio.
    let core = {
        let opts = ctx.opts.lock().clone();
        Arc::new(Mutex::new(InterruptionMachine::new(opts)))
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
    input_rx: &mut InputRx,
    core: &Arc<Mutex<InterruptionMachine>>,
    socket: Socket,
) -> Result<bool, StreamError> {
    let Socket { sender: sink, receiver: mut source } = socket;
    let opened =
        core.lock().handle(InterruptionEvent::ConnectionOpened, ctx.clock.now(), ctx.wall());
    if let Err(error) = opened {
        sink.close().await;
        return Err(error);
    }
    let connection = core.lock().connection();
    let (data_tx, data_rx) = mpsc::unbounded_channel();
    // `closing_ws`: per connection, reset on reconnect; tells an orderly shutdown from a hang-up.
    let closing = Arc::new(AtomicBool::new(false));
    let mut send_task = TokioSpawner::current().spawn(send_task(
        core.clone(),
        Arc::clone(&sink),
        data_rx,
        closing.clone(),
    ));
    let mut send_finished = false;
    let mut input_open = true;
    let mut data_tx = Some(data_tx);

    let outcome = loop {
        let deadline = core.lock().next_deadline();
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
                let outcome = core.lock().handle(
                    InterruptionEvent::Deadline, ctx.clock.now(), ctx.wall(),
                );
                if let Err(error) = outcome {
                    break Err(error);
                }
            }
            msg = source.recv() => {
                let status = close_status(source.close_code());
                match handle_incoming(ctx, core, connection, msg, status, &closing) {
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
                Some(item) => {
                    let ev = match item {
                        Item::Media(frame) => InputEvent::Audio(frame),
                        Item::Gap(_) => InputEvent::Gap,
                        Item::Event(ev) => ev,
                    };
                    let wall = ctx.wall();
                    let outcome = {
                        let mut core = core.lock();
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

    send_task.cancel();
    if !send_finished {
        let _ = send_task.await;
    }
    sink.close().await;
    outcome.map_err(|mut error| {
        if error.request.is_none() {
            error.request = core.lock().overlap_request();
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
    core: &Arc<Mutex<InterruptionMachine>>,
    connection: u64,
    msg: Result<WsMessage, TransportError>,
    close_status: i64,
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
                Err(closed_unexpectedly(close_status, "", ""))
            };
        }
        Ok(WsMessage::Binary(_) | WsMessage::Other) => return Ok(Incoming::Continue),
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
    let effects = core.lock().handle(
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
    core: Arc<Mutex<InterruptionMachine>>,
    sink: Arc<dyn WsSender>,
    mut data_rx: mpsc::UnboundedReceiver<(RequestIdentity, Vec<i16>)>,
    closing: Arc<AtomicBool>,
) -> Result<(), StreamError> {
    while let Some((identity, samples)) = data_rx.recv().await {
        if !core.lock().request_eligible(identity) {
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
        let opts = ctx.opts.lock();
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
            Arc::new(lk_agents_net::clock::RealClock::new()),
        )
    }

    // Catches: a detector that cannot connect failing on the audio piling up meanwhile, or one
    // whose sentinels stop draining still accepting them silently.
    #[tokio::test]
    async fn a_stalled_stream_keeps_current_audio_and_fails_only_once_its_sentinels_pile_up() {
        let detector = detector(Arc::new(FakeWs::hanging()));
        let mut stream = detector.stream();
        let audio = InputEvent::Audio(lk_agents_abi::AudioFrame {
            data: vec![0; 3200].into(),
            sample_rate: 16_000,
            num_channels: 1,
            samples_per_channel: 1600,
        });
        for _ in 0..100 {
            stream.push(audio.clone()).expect("a stall drops old audio, it never refuses it");
        }
        for _ in 0..media_queue::MAX_EVENTS {
            stream.flush().unwrap();
        }
        assert_eq!(stream.flush(), Err(StreamClosed::Stuck));
        assert_eq!(stream.push(audio), Err(StreamClosed::Stuck));
        let event = lk_agents_abi::timeout(
            &lk_agents_net::clock::RealClock::new(),
            Duration::from_secs(1),
            stream.recv_event(),
        )
        .await
        .expect("a stuck input must reach the caller without waiting for connect timeout");
        assert!(matches!(event, Some(StreamEvent::Error(error))
            if !error.recoverable && error.error.message.contains("stopped draining")));
        assert!(stream.finish().await.is_err());
    }

    #[tokio::test]
    async fn end_input_sends_accepted_audio_before_session_close() {
        let peer = Arc::new(FakeWs::new(Vec::new()).keep_open());
        let detector = detector(peer.clone());
        let mut stream = detector.stream();
        stream.push(InputEvent::AgentSpeechStarted).unwrap();
        stream
            .push(InputEvent::OverlapSpeechStarted(OverlapStart {
                speech_duration: 0.0,
                started_at: 1.0,
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
        for _ in 2..media_queue::MAX_EVENTS {
            stream.flush().unwrap();
        }
        stream.end_input().unwrap();
        lk_agents_abi::timeout(
            &lk_agents_net::clock::RealClock::new(),
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

/// How the WebSocket handshake failed, at the granularity `_connect_ws` distinguishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectFailure {
    /// `asyncio.TimeoutError` from the `wait_for` around `ws_connect`.
    Timeout,
    /// `aiohttp.ClientResponseError` -- the server answered the upgrade with a status.
    Http { status: i64 },
    /// `aiohttp.ClientConnectorError` and everything else.
    Transport,
}

/// `_connect_ws`'s except block (`inference/interruption.py:1008`). A 429 and a handshake timeout
/// are not retryable; a plain connector error is.
pub fn connect_error(failure: ConnectFailure) -> ApiError {
    match failure {
        ConnectFailure::Http { status: 429 } => {
            ApiError::status("LiveKit Adaptive Interruption quota exceeded", 429, None, Some(false))
        }
        ConnectFailure::Timeout => ApiError::connection(Some(
            "failed to connect to LiveKit Adaptive Interruption: timeout".to_owned(),
        ))
        .with_retryable(false),
        ConnectFailure::Http { .. } | ConnectFailure::Transport => ApiError::connection(Some(
            "failed to connect to LiveKit Adaptive Interruption".to_owned(),
        )),
    }
}

/// `recv_task`'s unexpected-close branch (`inference/interruption.py:836`), `status` the
/// [`close_status`](lk_agents_net::transport::close_status).
pub fn closed_unexpectedly(status: i64, data: &str, extra: &str) -> StreamError {
    StreamError {
        request: None,
        error: ApiError::status(
            format!("LiveKit Adaptive Interruption connection closed unexpectedly: {data}"),
            status,
            None,
            None,
        ),
        // `f"{ws_msg.data=} {ws_msg.extra=}"` uses `repr`; `{:?}` on a `&str` matches it.
        body: Some(format!("ws_msg.data={data:?} ws_msg.extra={extra:?}")),
    }
}

/// `_connect_ws`'s post-handshake failure (`inference/interruption.py:1030`): the socket opened but
/// `session.create` could not be written. Retryable, unlike the handshake failures above.
pub fn session_create_failed() -> ApiError {
    ApiError::connection(Some(
        "failed to send session.create message to LiveKit Adaptive Interruption".to_owned(),
    ))
}

#[cfg(test)]
mod connection_error_tests {
    use super::*;

    // Catches: a handshake timeout or a 429 marked retryable.
    #[test]
    fn a_handshake_timeout_is_not_retryable_but_a_connector_error_is() {
        assert!(!connect_error(ConnectFailure::Timeout).retryable);
        assert!(!connect_error(ConnectFailure::Http { status: 429 }).retryable);
        assert!(connect_error(ConnectFailure::Transport).retryable);
        // A non-429 HTTP answer takes the generic connection path, message and all.
        let other = connect_error(ConnectFailure::Http { status: 503 });
        assert_eq!(other.message, "failed to connect to LiveKit Adaptive Interruption");
        assert_eq!(other.status_code, None);
    }
}
