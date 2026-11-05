//! The stream's self-running executor: owns the gateway socket, reconnect attempts, audio
//! re-framing and the event translation for one live stream.

use super::*;
use crate::effects::SttStreamEffect;
use crate::telemetry;
use gaps::Gaps;
use lk_agents_abi::metrics::base::ConnectionAcquired;
use lk_agents_media::Frame;
use lk_agents_media::media_queue::{self, Item};
use std::ops::ControlFlow;

pub(super) struct SttStreamExecutor {
    pub(super) ws: Arc<dyn WsTransport>,
    pub(super) identification: Option<Headers>,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) create_frame: serde_json::Value,
    pub(super) opts: SttOptions,
    pub(super) vad: Option<Arc<dyn crate::abi::Vad>>,
    /// Seconds added to every timestamp: the caller's, plus the time spent on failed attempts, so
    /// transcripts stay on one linear timeline across reconnects.
    pub(super) offset: f64,
    pub(super) audio_rx: media_queue::AudioReceiver<Flush>,
    pub(super) update_rx: mpsc::UnboundedReceiver<serde_json::Value>,
    pub(super) deferred: Arc<DeferredExtra>,
    pub(super) update_tx: mpsc::UnboundedSender<serde_json::Value>,
    pub(super) event_tx: mpsc::Sender<Result<SpeechEvent, ApiError>>,
    pub(super) failure: tokio::sync::watch::Sender<Option<ApiError>>,
    pub(super) stop: tokio::sync::watch::Receiver<bool>,
    pub(super) accepted: lk_agents_abi::Ts,
}

impl SttStreamExecutor {
    async fn failed(&self, error: ApiError) {
        self.failure.send_replace(Some(error.clone()));
        let _ = self.event_tx.send(Err(error)).await;
    }

    /// `RecognizeStream._main_task`: each failed attempt reconnects, under `conn_options`, with the
    /// audio pushed meanwhile still queued. An attempt that outlived the connect limit had
    /// connected, so the budget restarts.
    pub(super) async fn run(mut self) {
        let mut policy = RequestPolicy::from_options(&self.opts.conn_options);
        // The socket adapter observes phase progress within each attempt. Reopening it has one
        // retry owner and retains the operation's original acceptance deadline.
        policy.connect_timeout = None;
        policy.write_timeout = None;
        policy.read_timeout = None;
        let mut steps = Vec::new();
        let mut attempts = Attempts::start(policy, self.accepted, &mut steps);
        let mut last_start = self.clock.now();
        loop {
            if *self.stop.borrow() {
                attempts.cancel(self.clock.now(), &mut Vec::new());
                return;
            }
            let started = self.clock.now();
            attempts.wake(started, None, &mut steps);
            if let Some(error) = steps.iter().find_map(|step| match step {
                AttemptStep::GiveUp(error) => Some(error.clone()),
                _ => None,
            }) {
                self.failed(error).await;
                return;
            }
            steps.clear();
            let Some(id) = attempts.current() else { return };
            self.offset += started.saturating_duration_since(last_start).as_secs_f64();
            last_start = started;
            let Err(error) = self.attempt().await else {
                attempts.succeeded(id, self.clock.now());
                attempts.closed(id, self.clock.now(), &mut Vec::new());
                return;
            };
            let now = self.clock.now();
            if self
                .opts
                .conn_options
                .attempt_connected(now.saturating_duration_since(started).as_secs_f64(), &error)
            {
                attempts.reset_retries();
            }
            steps.clear();
            attempts.failed_error(id, error.clone(), now, &mut steps);
            attempts.closed(id, now, &mut steps);
            if !steps.iter().any(|step| matches!(step, AttemptStep::Retry { .. })) {
                let error = match lk_agents_core::attempts::terminal_failure(
                    lk_agents_abi::types::retry::RequestKind::SttStream,
                    &error,
                    self.opts.conn_options.max_retry,
                    attempts.retries(),
                ) {
                    lk_agents_abi::types::retry::RetryDecision::Exhausted(message) => {
                        ApiError::connection(Some(format!("{message}: {error}")))
                    }
                    _ => error,
                };
                self.failed(error).await;
                return;
            }
            let at = attempts.wake_at(None).expect("authorized retry wake");
            use futures_util::future::{Either, select};
            if matches!(
                select(Box::pin(self.clock.wait_until(at)), Box::pin(self.stop.changed())).await,
                Either::Right(_)
            ) {
                attempts.cancel(self.clock.now(), &mut Vec::new());
                return;
            }
        }
    }

    /// One connection. `Ok` when the stream ended, the caller gone or the input finished.
    async fn attempt(&mut self) -> Result<(), ApiError> {
        let (headers, request_id) = self.connection_headers()?;
        let url = format!(
            "{}/stt?model={}",
            to_ws_url(&self.opts.base_url),
            self.opts.model.as_deref().unwrap_or("")
        );
        // `_speech_duration`: written by send, read by recv.
        let machine = Arc::new(Mutex::new(SttStream::new(
            0,
            lk_agents_abi::resources::ResourceId(0),
            request_id,
            self.opts.language.as_deref(),
            FINALIZATION_TIMEOUT,
            self.opts
                .conn_options
                .total_timeout
                .map(|seconds| self.accepted.saturating_add(Duration::from_secs_f64(seconds))),
        )));
        let opening = machine.lock().handle(SttStreamEvent::Start, self.clock.now());
        if !opening.iter().any(|effect| matches!(effect, SttStreamEffect::OpenSttStream { .. })) {
            return finished(opening).unwrap_or(Ok(()));
        }
        let acquiring = self.clock.now();
        let (sender, mut receiver) = open_ws(
            self.ws.as_ref(),
            &url,
            headers.into_vec(),
            &*self.clock,
            &self.opts.conn_options,
            WHAT,
            &self.create_frame,
        )
        .await?;
        let acquire_time = self.clock.now().saturating_duration_since(acquiring).as_secs_f64();
        let connection = ConnectionAcquired { acquire_time, reused: false };

        // Built before the loops so the send half can push into it while the finalize half drains it.
        let (vad_sink, mut vad_source) = self.open_vad().await;

        // Wakes the settlement timer whenever the input or a message moves its deadline.
        let rearm = Arc::new(tokio::sync::Notify::new());
        let gaps = Arc::new(Mutex::new(Gaps::starting(self.offset)));
        let sample_rate = self.opts.sample_rate;
        let uplink = AudioUplink {
            sender: sender.clone(),
            gaps: gaps.clone(),
            vad_sink,
            vad_push_warned: false,
            sample_rate,
            machine: machine.clone(),
            clock: self.clock.clone(),
            rearm: rearm.clone(),
            bstream: AudioByteStream::new(
                sample_rate,
                GATEWAY_CHANNELS,
                Some(sample_rate / FRAMES_PER_SECOND),
                false,
            ),
            // `RecognizeStream.push_frame` resamples to the rate the gateway was told.
            resampler: AudioResampler::new(sample_rate, Quality::High, OnChange::Drain),
            audio: String::new(),
        };
        let send = uplink.run(&mut self.audio_rx, &mut self.update_rx);
        let recv = RecvCtx {
            event_tx: self.event_tx.clone(),
            deferred: self.deferred.clone(),
            update_tx: self.update_tx.clone(),
            clock: self.clock.clone(),
            ack_deadline: self.opts.conn_options.read_deadline(self.clock.now()),
            offset: self.offset,
            gaps,
            rearm: rearm.clone(),
            connection: Some(connection),
            awaiting_ack: true,
        }
        .run(receiver.as_mut(), machine.clone());
        let finalize = Self::vad_finalize_loop(sender.clone(), &mut vad_source);

        let settle = async {
            loop {
                let deadline = machine.lock().wake_at();
                let wait = async {
                    match deadline {
                        Some(deadline) => self.clock.wait_until(deadline).await,
                        None => std::future::pending().await,
                    }
                };
                tokio::select! {
                    biased;
                    _ = rearm.notified() => continue,
                    _ = wait => {}
                }
                let effects = machine.lock().handle(SttStreamEvent::Wake, self.clock.now());
                if let Some(result) = finished(effects) {
                    return result;
                }
            }
        };
        let outcome = tokio::select! {
            _ = self.stop.wait_for(|stop| *stop) => Ok(()),
            r = settle => r,
            r = send => r,
            r = recv => r,
            r = finalize => r,
        };
        sender.close().await;
        if let Some(source) = vad_source {
            let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
            if source.close(deadline).await == lk_agents_abi::Closed::TimedOut {
                log::error!("STT VAD cleanup incomplete");
            }
        }
        outcome.map_err(|error| {
            if machine.lock().replay_safe() { error } else { error.with_retryable(false) }
        })
    }

    /// The connection's gateway headers under a fresh session id, and the request id. The
    /// reference reads the session id back out of the header map (a job's `inference_headers`
    /// cannot override it) and uses it as the request id.
    fn connection_headers(&self) -> Result<(Headers, String), ApiError> {
        let identification = self.identification.as_ref().map(|h| {
            let mut h = h.clone();
            h.set(HEADER_SESSION_ID, crate::ids::shortuuid(crate::ids::SESSION_ID_PREFIX));
            h
        });
        let headers = gateway_headers(
            identification.as_ref(),
            None,
            None,
            &self.opts.api_key,
            &self.opts.api_secret,
        )?;
        let request_id = headers
            .get(HEADER_SESSION_ID)
            .map(str::to_owned)
            .unwrap_or_else(|| crate::ids::shortuuid(crate::ids::STT_REQUEST_ID_PREFIX));
        Ok((headers, request_id))
    }

    /// The client-side VAD's stream halves, when a VAD is set and its stream opens.
    async fn open_vad(&self) -> (Option<crate::abi::VadSink>, Option<crate::abi::VadSource>) {
        let Some(vad) = self.vad.as_ref() else { return (None, None) };
        match vad.stream(&telemetry::Cx::NONE).await {
            Ok(stream) => {
                let (sink, source) = stream.split();
                (Some(sink), Some(source))
            }
            Err(_) => (None, None),
        }
    }

    /// `vad_task`: every end of speech from the client-side VAD sends `session.finalize`.
    async fn vad_finalize_loop(
        sender: Arc<dyn WsSender>,
        source: &mut Option<crate::abi::VadSource>,
    ) -> Result<(), ApiError> {
        let Some(source) = source.as_mut() else {
            // No VAD: this arm must never resolve, or `select!` tears the socket down at once.
            std::future::pending::<()>().await;
            return Ok(());
        };
        while let Ok(Some(event)) = source.next().await {
            if event.kind != lk_agents_abi::VadEventType::EndOfSpeech {
                continue;
            }
            if sender.is_closed() {
                return Ok(());
            }
            if sender.send_text(wire::session_finalize().to_string()).await.is_err() {
                return Ok(());
            }
        }
        std::future::pending::<()>().await;
        Ok(())
    }
}

/// The gateway takes audio in 50 ms frames: `samples_per_channel=self._opts.sample_rate // 20`.
const FRAMES_PER_SECOND: u32 = 20;

/// The gateway takes mono audio.
const GATEWAY_CHANNELS: u16 = 1;

/// The bytes of one 16-bit PCM sample.
const PCM_SAMPLE_BYTES: usize = std::mem::size_of::<i16>();

/// The machine's verdict, when `effects` end the stream.
fn finished(effects: impl IntoIterator<Item = SttStreamEffect>) -> Option<Result<(), ApiError>> {
    effects.into_iter().find_map(|effect| match effect {
        SttStreamEffect::Finished(result) => Some(result),
        _ => None,
    })
}

/// The send half of one connection: the input re-framed, resampled and written to the gateway,
/// with `session.update`s going ahead of the audio.
struct AudioUplink {
    sender: Arc<dyn WsSender>,
    /// Where this attempt's input dropped audio.
    gaps: Arc<Mutex<Gaps>>,
    vad_sink: Option<crate::abi::VadSink>,
    vad_push_warned: bool,
    sample_rate: u32,
    machine: Arc<Mutex<SttStream>>,
    clock: Arc<dyn Clock>,
    rearm: Arc<tokio::sync::Notify>,
    bstream: AudioByteStream,
    resampler: AudioResampler,
    /// The base64 buffer each frame is encoded into.
    audio: String,
}

impl AudioUplink {
    async fn run(
        mut self,
        audio_rx: &mut media_queue::AudioReceiver<Flush>,
        update_rx: &mut mpsc::UnboundedReceiver<serde_json::Value>,
    ) -> Result<(), ApiError> {
        loop {
            let item = tokio::select! {
                // Biased so a pending `session.update` is not starved by continuous audio.
                biased;
                Some(frame) = update_rx.recv() => {
                    self.send_update(frame).await;
                    continue;
                }
                item = audio_rx.recv() => item,
            };
            let frames = match item {
                Some(Item::Media(frame)) => self.chunk_media(frame).await?,
                Some(Item::Gap(gap)) => {
                    self.gaps.lock().gap(gap);
                    continue;
                }
                Some(Item::Event(Flush)) => self.drain_tail()?,
                // `end_input()` or a dropped sink: the queue's end ends the send half.
                None => {
                    let frames = self.drain_tail()?;
                    self.send_frames(frames).await?;
                    break;
                }
            };
            self.send_frames(frames).await?;
        }
        self.finish_input().await
    }

    /// "failed to send session.update, ws may be closing" -- swallowed, as in the reference.
    async fn send_update(&mut self, frame: serde_json::Value) {
        if !self.sender.is_closed() {
            let _ = self.sender.send_text(frame.to_string()).await;
        }
    }

    /// One input frame, to the VAD and cut into 50 ms chunks at the gateway's rate.
    async fn chunk_media(&mut self, frame: AudioFrame) -> Result<Vec<Frame>, ApiError> {
        self.gaps.lock().audio(lk_agents_abi::voice::rtc::frame_duration(&frame));
        if let Some(sink) = self.vad_sink.as_mut()
            && let Err(error) = sink.push(frame.clone())
            && !self.vad_push_warned
        {
            self.vad_push_warned = true;
            log::warn!(target: "livekit.agents", "STT VAD refused audio: {error}; suppressing repeats");
        }
        let samples = usize::try_from(frame.sample_rate / FRAMES_PER_SECOND).unwrap_or(0).max(1);
        let stride = usize::from(frame.num_channels) * PCM_SAMPLE_BYTES;
        if stride == 0 || frame.sample_rate == 0 || frame.data.len() % stride != 0 {
            return Err(ApiError::new("Invalid STT audio frame").with_retryable(false));
        }
        let mut frames = Vec::new();
        for start in (0..frame.data.len()).step_by(samples * stride) {
            let end = frame.data.len().min(start + samples * stride);
            let bytes = frame.data.slice(start..end);
            let chunk = AudioFrame::new(
                bytes.clone(),
                frame.sample_rate,
                frame.num_channels,
                (bytes.len() / stride) as u32,
            );
            let resampled = self
                .resampler
                .push_frame(&chunk)
                .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
            frames.extend(resampled.iter().flat_map(|frame| self.bstream.push(&frame.data)));
            tokio::task::yield_now().await;
        }
        Ok(frames)
    }

    /// What the resampler and the framer still hold.
    fn drain_tail(&mut self) -> Result<Vec<Frame>, ApiError> {
        let tail = self
            .resampler
            .flush()
            .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
        let mut frames: Vec<_> =
            tail.iter().flat_map(|frame| self.bstream.push(&frame.data)).collect();
        frames.extend(self.bstream.flush());
        Ok(frames)
    }

    async fn send_frames(&mut self, frames: Vec<Frame>) -> Result<(), ApiError> {
        for frame in frames {
            let duration = f64::from(frame.samples_per_channel) / f64::from(self.sample_rate);
            self.machine.lock().handle(SttStreamEvent::AudioWritten(duration), self.clock.now());
            self.audio.clear();
            base64::engine::general_purpose::STANDARD.encode_string(&frame.data, &mut self.audio);
            let sent = self.sender.send_text(wire::input_audio(&self.audio).to_string()).await;
            if let Err(err) = sent {
                return Err(match err {
                    TransportError::Closed => {
                        ApiError::connection(Some(format!("{WHAT} connection closed unexpectedly")))
                    }
                    other => ApiError::connection(Some(other.to_string())),
                });
            }
        }
        Ok(())
    }

    /// The input ended: the VAD hears it, the machine may ask the gateway to finalize, and the
    /// settlement timer re-arms. Completion is an acknowledgment or a quiet period: the receive
    /// half and the settlement timer observe it.
    async fn finish_input(mut self) -> Result<(), ApiError> {
        if let Some(sink) = self.vad_sink.as_mut() {
            sink.end_input();
        }
        let effects = self.machine.lock().handle(SttStreamEvent::InputEnded, self.clock.now());
        if effects.iter().any(|effect| matches!(effect, SttStreamEffect::SttFinalize { .. })) {
            self.sender
                .send_text(wire::session_finalize().to_string())
                .await
                .map_err(|error| connect_error(WHAT, error))?;
        }
        self.rearm.notify_one();
        std::future::pending().await
    }
}

/// The receive half of one connection: gateway messages through the machine, transcripts out.
struct RecvCtx {
    event_tx: mpsc::Sender<Result<SpeechEvent, ApiError>>,
    deferred: Arc<DeferredExtra>,
    update_tx: mpsc::UnboundedSender<serde_json::Value>,
    clock: Arc<dyn Clock>,
    /// `read_timeout` from the session-create frame: the acknowledgement is a response.
    ack_deadline: Option<lk_agents_abi::Ts>,
    offset: f64,
    /// Where this attempt's input dropped audio: its transcripts move past each gap.
    gaps: Arc<Mutex<Gaps>>,
    rearm: Arc<tokio::sync::Notify>,
    /// The connection, until the first usage report it delivers carries it.
    connection: Option<ConnectionAcquired>,
    /// Only the first read (the `session.create` ack) is a response the session waits for;
    /// later silence is the user's.
    awaiting_ack: bool,
}

impl RecvCtx {
    async fn run(
        mut self,
        receiver: &mut dyn lk_agents_net::transport::WsReceiver,
        machine: Arc<Mutex<SttStream>>,
    ) -> Result<(), ApiError> {
        loop {
            let effects = match self.read(receiver).await? {
                WsMessage::Closed => {
                    let effects =
                        machine.lock().handle(SttStreamEvent::ServerEof, self.clock.now());
                    return finished(effects).unwrap_or(Ok(()));
                }
                // "unexpected LiveKit Inference STT message type" -- logged and skipped.
                WsMessage::Binary(_) | WsMessage::Other => continue,
                WsMessage::Text(text) => self.handle_text(&machine, &text)?,
            };
            if let Some(result) = self.apply(effects).await {
                return result;
            }
        }
    }

    /// The next message; the acknowledgement must arrive within its read deadline.
    async fn read(
        &self,
        receiver: &mut dyn lk_agents_net::transport::WsReceiver,
    ) -> Result<WsMessage, ApiError> {
        let recv = receiver.recv();
        let received = if self.awaiting_ack {
            within(&*self.clock, self.ack_deadline, recv).await
        } else {
            recv.await
        };
        match received {
            Ok(msg) => Ok(msg),
            Err(TransportError::Timeout) if self.awaiting_ack => {
                Err(ApiError::timed_out(lk_agents_abi::errors::TimeoutPhase::Read))
            }
            Err(err) => Err(connect_error(WHAT, err)),
        }
    }

    /// A gateway message through the machine; the settlement timer re-arms after the lock.
    fn handle_text(
        &mut self,
        machine: &Mutex<SttStream>,
        text: &str,
    ) -> Result<Vec<SttStreamEffect>, ApiError> {
        let incoming = wire::decode(text)
            .ok_or_else(|| ApiError::new("Malformed STT gateway message").with_retryable(false))?;
        if matches!(incoming, wire::Incoming::SessionCreated) {
            self.awaiting_ack = false;
        }
        let effects = {
            let mut machine = machine.lock();
            let effects =
                machine.handle(SttStreamEvent::Message(Box::new(incoming)), self.clock.now());
            self.deferred
                .speaking
                .store(machine.is_speaking(), std::sync::atomic::Ordering::SeqCst);
            effects
        };
        self.rearm.notify_one();
        Ok(effects)
    }

    /// Carries out the machine's effects; `Some` ends the stream.
    async fn apply(&mut self, effects: Vec<SttStreamEffect>) -> Option<Result<(), ApiError>> {
        for effect in effects {
            match effect {
                SttStreamEffect::Deliver(event) => {
                    if self.deliver(event).await.is_break() {
                        return Some(Ok(()));
                    }
                }
                SttStreamEffect::EndOfSpeech => self.send_deferred_update(),
                SttStreamEffect::Finished(result) => return Some(result),
                _ => {}
            }
        }
        None
    }

    /// Sends a transcript on the caller's timeline; the first usage report carries the
    /// connection. Breaks once the caller is gone.
    async fn deliver(&mut self, mut event: SpeechEvent) -> ControlFlow<()> {
        if let Some(usage) = &mut event.recognition_usage {
            usage.connection = self.connection.take();
        }
        let event = self.gaps.lock().shift(shifted(event, self.offset));
        if self.event_tx.send(Ok(event)).await.is_err() {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }

    /// An end of speech releases the options update deferred until it.
    fn send_deferred_update(&self) {
        let extra = self.deferred.pending.lock().take();
        if let Some(extra) = extra
            && let Some(frame) = wire::session_update(None, None, Some(&extra))
        {
            let _ = self.update_tx.send(frame);
        }
    }
}

/// The event on the caller's timeline: every alternative's and word's times moved by `offset`.
pub(super) fn shifted(mut event: SpeechEvent, offset: f64) -> SpeechEvent {
    for alt in &mut event.alternatives {
        alt.start_time += offset;
        alt.end_time += offset;
        for word in alt.words.iter_mut().flatten() {
            word.start_time = Some(word.start_time.unwrap_or(0.0) + offset);
            word.end_time = Some(word.end_time.unwrap_or(0.0) + offset);
            word.start_time_offset = Some(offset);
        }
    }
    event
}
