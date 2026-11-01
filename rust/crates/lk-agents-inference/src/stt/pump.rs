//! The stream's pump: one task that owns the gateway socket, its reconnect attempts, the audio
//! re-framing and the event translation for one live stream.

use super::*;

pub(super) struct Pump {
    pub(super) ws: Arc<dyn WsTransport>,
    pub(super) identification: Option<Headers>,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) create_frame: serde_json::Value,
    pub(super) opts: SttOptions,
    pub(super) vad: Option<Arc<dyn crate::abi::Vad>>,
    /// Seconds added to every timestamp: the caller's, plus the time spent on failed attempts, so
    /// transcripts stay on one linear timeline across reconnects.
    pub(super) offset: f64,
    pub(super) audio_rx: mpsc::Receiver<SinkCmd>,
    pub(super) update_rx: mpsc::Receiver<serde_json::Value>,
    pub(super) deferred: Arc<DeferredExtra>,
    pub(super) update_tx: mpsc::Sender<serde_json::Value>,
    pub(super) event_tx: mpsc::Sender<Result<SpeechEvent, ApiError>>,
    pub(super) failure: tokio::sync::watch::Sender<Option<ApiError>>,
    pub(super) stop: tokio::sync::watch::Receiver<bool>,
    pub(super) ended: tokio::sync::watch::Receiver<bool>,
    pub(super) accepted: lk_agents_abi::Ts,
}

impl Pump {
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
        let headers = gateway_headers(
            self.identification
                .as_ref()
                .map(|h| {
                    let mut h = h.clone();
                    h.set(HEADER_SESSION_ID, crate::ids::shortuuid("inference_"));
                    h
                })
                .as_ref(),
            None,
            None,
            &self.opts.api_key,
            &self.opts.api_secret,
        )?;
        // The reference reads the session id back out of the header map (a job's
        // `inference_headers` cannot override it) and uses it as the request id.
        let request_id = headers
            .get(HEADER_SESSION_ID)
            .map(str::to_owned)
            .unwrap_or_else(|| crate::ids::shortuuid("stt_request_"));
        let url = format!(
            "{}/stt?model={}",
            to_ws_url(&self.opts.base_url),
            self.opts.model.as_deref().unwrap_or("")
        );
        // `_speech_duration`: written by send, read by recv.
        let machine = Arc::new(Mutex::new(SttProtocol::new(
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
        let opening = machine.lock().handle(SttEvent::Start, self.clock.now());
        if !opening.iter().any(|effect| matches!(effect, SttEffect::SttStreamOpen { .. })) {
            return opening
                .into_iter()
                .find_map(|effect| match effect {
                    SttEffect::Finished(result) => Some(result),
                    _ => None,
                })
                .unwrap_or(Ok(()));
        }
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

        // Built before the loops so the send half can push into it while the finalize half drains it.
        let vad_stream = match self.vad.as_ref() {
            Some(vad) => vad.stream().await.ok(),
            None => None,
        };
        let (vad_sink, mut vad_source) = match vad_stream {
            Some(stream) => {
                let (sink, source) = stream.split();
                (Some(sink), Some(source))
            }
            None => (None, None),
        };

        // Wakes the settlement timer whenever the input or a message moves its deadline.
        let rearm = Arc::new(tokio::sync::Notify::new());
        let send = Self::send_loop(
            sender.clone(),
            &mut self.audio_rx,
            &mut self.update_rx,
            &mut self.ended,
            vad_sink,
            self.opts.sample_rate,
            machine.clone(),
            self.clock.clone(),
            rearm.clone(),
        );
        let recv = Self::recv_loop(
            receiver.as_mut(),
            machine.clone(),
            RecvCtx {
                event_tx: self.event_tx.clone(),
                deferred: self.deferred.clone(),
                update_tx: self.update_tx.clone(),
                clock: self.clock.clone(),
                ack_deadline: self.opts.conn_options.read_deadline(self.clock.now()),
                offset: self.offset,
                rearm: rearm.clone(),
            },
        );
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
                let effects = machine.lock().handle(SttEvent::Wake, self.clock.now());
                if let Some(result) = effects.into_iter().find_map(|effect| match effect {
                    SttEffect::Finished(result) => Some(result),
                    _ => None,
                }) {
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

    #[allow(
        clippy::too_many_arguments,
        reason = "Keep the operation’s existing explicit arguments"
    )]
    async fn send_loop(
        sender: Arc<dyn WsSender>,
        audio_rx: &mut mpsc::Receiver<SinkCmd>,
        update_rx: &mut mpsc::Receiver<serde_json::Value>,
        ended: &mut tokio::sync::watch::Receiver<bool>,
        mut vad_sink: Option<crate::abi::VadSink>,
        sample_rate: u32,
        machine: Arc<Mutex<SttProtocol>>,
        clock: Arc<dyn Clock>,
        rearm: Arc<tokio::sync::Notify>,
    ) -> Result<(), ApiError> {
        // 50 ms frames: `samples_per_channel=self._opts.sample_rate // 20`.
        let mut bstream = AudioByteStream::new(sample_rate, 1, Some(sample_rate / 20), false);
        // `RecognizeStream.push_frame` resamples to the rate the gateway was told.
        let mut resampler = FrameResampler::new(sample_rate, Quality::High, OnChange::Drain);

        if *ended.borrow() {
            audio_rx.close();
        }
        let mut ending = false;
        let mut vad_push_warned = false;
        let mut audio = String::new();
        loop {
            let item = tokio::select! {
                // Biased so a pending `session.update` is not starved by continuous audio.
                biased;
                changed = ended.changed(), if !audio_rx.is_closed() => {
                    if changed.is_err() || *ended.borrow() { audio_rx.close(); }
                    continue;
                }
                Some(frame) = update_rx.recv() => {
                    // "failed to send session.update, ws may be closing" -- swallowed, as in the
                    // reference.
                    if !sender.is_closed() {
                        let _ = sender.send_text(frame.to_string()).await;
                    }
                    continue;
                }
                item = audio_rx.recv() => item,
            };

            let frames = match item {
                Some(SinkCmd::Frame(frame)) => {
                    if let Some(sink) = vad_sink.as_mut()
                        && let Err(error) = sink.push(frame.clone())
                        && !vad_push_warned
                    {
                        vad_push_warned = true;
                        log::warn!(
                            target: "livekit.agents",
                            "STT VAD refused audio: {error}; suppressing repeats"
                        );
                    }
                    let samples = usize::try_from(frame.sample_rate / 20).unwrap_or(0).max(1);
                    let stride = usize::from(frame.num_channels) * 2;
                    if stride == 0 || frame.sample_rate == 0 || frame.data.len() % stride != 0 {
                        return Err(ApiError::new("Invalid STT audio frame").with_retryable(false));
                    }
                    let mut frames = Vec::new();
                    for bytes in frame.data.chunks(samples * stride) {
                        let chunk = AudioFrame::new(
                            bytes,
                            frame.sample_rate,
                            frame.num_channels,
                            (bytes.len() / stride) as u32,
                        );
                        let resampled = resampler.push(&chunk).map_err(|error| {
                            ApiError::new(error.to_string()).with_retryable(false)
                        })?;
                        frames.extend(resampled.iter().flat_map(|frame| bstream.push(&frame.data)));
                        tokio::task::yield_now().await;
                    }
                    frames
                }
                Some(SinkCmd::Flush) => {
                    let tail = resampler
                        .flush()
                        .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
                    let mut frames: Vec<_> =
                        tail.iter().flat_map(|f| bstream.push(&f.data)).collect();
                    frames.extend(bstream.flush());
                    frames
                }
                // `end_input()` or a dropped sink: both end the send half.
                None => {
                    ending = true;
                    let tail = resampler
                        .flush()
                        .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
                    let mut frames: Vec<_> =
                        tail.iter().flat_map(|frame| bstream.push(&frame.data)).collect();
                    frames.extend(bstream.flush());
                    frames
                }
            };

            for frame in frames {
                machine.lock().handle(
                    SttEvent::AudioWritten(
                        f64::from(frame.samples_per_channel) / f64::from(sample_rate),
                    ),
                    clock.now(),
                );
                audio.clear();
                base64::engine::general_purpose::STANDARD.encode_string(&frame.data, &mut audio);
                if let Err(err) = sender.send_text(wire::input_audio(&audio).to_string()).await {
                    return Err(match err {
                        TransportError::Closed => ApiError::connection(Some(format!(
                            "{WHAT} connection closed unexpectedly"
                        ))),
                        other => ApiError::connection(Some(other.to_string())),
                    });
                }
            }
            if ending {
                break;
            }
        }

        if let Some(sink) = vad_sink.as_mut() {
            sink.end_input();
        }
        let effects = machine.lock().handle(SttEvent::InputEof, clock.now());
        if effects.iter().any(|effect| matches!(effect, SttEffect::SttFinalize { .. })) {
            sender
                .send_text(wire::session_finalize().to_string())
                .await
                .map_err(|error| connect_error(WHAT, error))?;
        }
        rearm.notify_one();
        // Completion is an acknowledgment or a quiet period: the receive half and the settlement
        // timer observe it.
        std::future::pending().await
    }

    async fn recv_loop(
        receiver: &mut dyn lk_agents_net::transport::WsReceiver,
        machine: Arc<Mutex<SttProtocol>>,
        ctx: RecvCtx,
    ) -> Result<(), ApiError> {
        let RecvCtx { event_tx, deferred, update_tx, clock, ack_deadline, offset, rearm } = ctx;
        // Only the first read (the `session.create` ack) is a response the session waits for;
        // later silence is the user's.
        let mut awaiting_first = true;
        loop {
            let recv = receiver.recv();
            let received =
                if awaiting_first { within(&*clock, ack_deadline, recv).await } else { recv.await };
            let msg = match received {
                Ok(msg) => msg,
                Err(TransportError::Timeout) if awaiting_first => {
                    return Err(ApiError::timed_out(lk_agents_abi::errors::TimeoutPhase::Read));
                }
                Err(err) => return Err(connect_error(WHAT, err)),
            };
            match msg {
                WsMessage::Closed => {
                    let effects = machine.lock().handle(SttEvent::ServerEof, clock.now());
                    return effects
                        .into_iter()
                        .find_map(|effect| match effect {
                            SttEffect::Finished(result) => Some(result),
                            _ => None,
                        })
                        .unwrap_or(Ok(()));
                }
                // "unexpected LiveKit Inference STT message type" -- logged and skipped.
                WsMessage::Binary(_) | WsMessage::Other => continue,
                WsMessage::Text(text) => {
                    let incoming = wire::decode(&text).ok_or_else(|| {
                        ApiError::new("Malformed STT gateway message").with_retryable(false)
                    })?;
                    if matches!(incoming, wire::Incoming::SessionCreated) {
                        awaiting_first = false;
                    }
                    let effects = {
                        let mut machine = machine.lock();
                        let effects =
                            machine.handle(SttEvent::Message(Box::new(incoming)), clock.now());
                        deferred
                            .speaking
                            .store(machine.is_speaking(), std::sync::atomic::Ordering::SeqCst);
                        effects
                    };
                    rearm.notify_one();
                    for effect in effects {
                        match effect {
                            SttEffect::Deliver(event) => {
                                if event_tx.send(Ok(shifted(event, offset))).await.is_err() {
                                    return Ok(());
                                }
                            }
                            SttEffect::EndOfSpeech => {
                                let extra = deferred.pending.lock().take();
                                if let Some(extra) = extra
                                    && let Some(frame) =
                                        wire::session_update(None, None, Some(&extra))
                                {
                                    update_tx.try_send(frame).map_err(|_| {
                                        ApiError::new("STT update queue capacity exceeded")
                                            .with_retryable(false)
                                    })?;
                                }
                            }
                            SttEffect::Finished(result) => return result,
                            _ => {}
                        }
                    }
                }
            }
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

struct RecvCtx {
    event_tx: mpsc::Sender<Result<SpeechEvent, ApiError>>,
    deferred: Arc<DeferredExtra>,
    update_tx: mpsc::Sender<serde_json::Value>,
    clock: Arc<dyn Clock>,
    /// `read_timeout` from the session-create frame: the acknowledgement is a response.
    ack_deadline: Option<lk_agents_abi::Ts>,
    offset: f64,
    rearm: Arc<tokio::sync::Notify>,
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
