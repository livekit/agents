//! `TurnDetectorStream` (`inference/eot.py`): the session's decisions carried out against the
//! mounted transport, with the input resampled to the detector's rate, the transport swapped for
//! the local model on a cloud failure, and each `predict()` answered by its own waiter.

use crate::telemetry;
use std::collections::HashMap;
use std::sync::Arc;

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tokenize::language::{ThresholdOptions, TurnDetectorModel};
use lk_agents_abi::turn::inference::EotStream as AbiEotStream;
use lk_agents_abi::{AudioFrame, EotInferenceMetrics, EotPrediction, ProviderId};
use lk_agents_media::pool::{JobClass, MediaPoolError};
use lk_agents_media::{Quality, Resampler};
use parking_lot::Mutex;
use tokio::sync::{oneshot, watch};

use crate::eot::{
    AudioRingBuffer, CLIENT_BUFFER_SAMPLES, CancelReason, EotAction, EotRequestId, EotSession,
    EotWarning, MountedTransport, SinkEvent, TransportPrediction,
};

/// Mounts the transport for a model: the gateway for `turn-detector-v1`, the mini model otherwise.
pub type TransportFactory = Arc<dyn Fn(TurnDetectorModel) -> Arc<MountedTransport> + Send + Sync>;

/// `self._detector.emit("metrics_collected", ...)`.
pub type MetricsCallback = Arc<dyn Fn(EotInferenceMetrics) + Send + Sync>;

const LOG: &str = "livekit.agents";

use lk_agents_net::clock::unix_now as now;

fn pcm_frame(pcm: Vec<i16>, rate: u32, channels: u16) -> AudioFrame {
    let per_channel = (pcm.len() / usize::from(channels.max(1))) as u32;
    AudioFrame::new(lk_agents_media::pcm_bytes(&pcm), rate, channels, per_channel)
}

/// The `_WARNINGS` sentences, with the warning's detail where the reference interpolates it.
fn sentence(warning: &EotWarning) -> String {
    let detail = warning.detail().unwrap_or_default();
    match warning {
        EotWarning::CloudFailedFallingBack { .. } => {
            format!("cloud turn detector failed ({detail}); falling back to local mini model")
        }
        EotWarning::CloudFailedDegraded { .. } => format!(
            "cloud turn detector failed ({detail}); local fallback is disabled, so turn detection \
             is off for the rest of this stream and turns commit on the endpointing delay"
        ),
        EotWarning::LocalFailed { .. } => format!(
            "local audio turn detector failed ({detail}); defaulting to 1.0 and retrying on next turn"
        ),
        EotWarning::LocalPredictionFailed { .. } => {
            format!("local audio turn detector prediction failed ({detail})")
        }
        EotWarning::TransportLatency { .. } => format!("audio turn detector is behind ({detail})"),
        EotWarning::UnexpectedServerMessage => {
            "unexpected message from the turn detector (None)".to_owned()
        }
    }
}

struct Inner {
    session: EotSession,
    shell: Option<Arc<MountedTransport>>,
    /// While the local model loads, retain only its trailing prediction window.
    replacement_audio: AudioRingBuffer,
    replacement_request: Option<(EotRequestId, Vec<i16>)>,
    retired: Option<Arc<MountedTransport>>,
    waiters: HashMap<String, oneshot::Sender<EotPrediction>>,
    resampler: Option<Resampler>,
    input_format: Option<(u32, u16)>,
    degraded: bool,
}

/// One stream of the user's audio, answering `predict()` at each pause.
pub struct EotStream {
    inner: Mutex<Inner>,
    rate: u32,
    make: TransportFactory,
    /// Where the local model is constructed.
    media_pool: lk_agents_media::pool::MediaPool,
    on_metrics: Option<MetricsCallback>,
    /// A swap or prediction wakes a driver whose local construction was refused.
    swaps: watch::Sender<u64>,
    settled: watch::Sender<bool>,
    id: ProviderId,
    /// The stream's run, which holds the stream until it ends: it outlives a close while model
    /// construction is still settling.
    runner: std::sync::OnceLock<lk_agents_abi::OwnedTask>,
}

impl EotStream {
    /// Mounts the transport for `session`'s model and starts the run loop on `spawner`.
    pub fn open(
        session: EotSession,
        rate: u32,
        make: TransportFactory,
        media_pool: lk_agents_media::pool::MediaPool,
        on_metrics: Option<MetricsCallback>,
        spawner: &dyn lk_agents_abi::Spawner,
    ) -> DetectorStream {
        let shell = (session.model() == TurnDetectorModel::V1).then(|| make(session.model()));
        let id = ProviderId {
            label: "livekit.agents.inference.TurnDetector".into(),
            model: session.model().as_str().into(),
            provider: crate::eot::PROVIDER.into(),
        };
        let stream = Arc::new(EotStream {
            inner: Mutex::new(Inner {
                session,
                shell,
                replacement_audio: AudioRingBuffer::new(CLIENT_BUFFER_SAMPLES),
                replacement_request: None,
                retired: None,
                waiters: HashMap::new(),
                resampler: None,
                input_format: None,
                degraded: false,
            }),
            rate,
            make,
            media_pool,
            on_metrics,
            swaps: watch::channel(0).0,
            settled: watch::channel(false).0,
            id,
            runner: std::sync::OnceLock::new(),
        });
        let owner = Arc::clone(&stream);
        let settled = stream.settled.clone();
        let run = async move {
            owner.run().await;
            settled.send_replace(true);
        };
        let _ = stream.runner.set(lk_agents_abi::SpawnerExt::spawn(spawner, run));
        DetectorStream(stream)
    }

    fn lock(&self) -> parking_lot::MutexGuard<'_, Inner> {
        self.inner.lock()
    }

    fn push(inner: &mut Inner, frame: AudioFrame, head: bool) {
        if !inner.degraded && !inner.session.is_closed() {
            if let Some(shell) = &inner.shell {
                if head {
                    shell.transport.push_head(&frame);
                } else {
                    shell.transport.push_frame(&frame);
                }
            } else {
                inner.replacement_audio.push(&lk_agents_media::pcm_samples(&frame.data));
            }
        }
    }

    fn apply(&self, inner: &mut Inner, actions: Vec<EotAction>) -> Vec<EotWarning> {
        let mut warnings = Vec::new();
        for action in actions {
            match action {
                EotAction::Resolve(r) => {
                    if inner.replacement_request.as_ref().is_some_and(|(id, _)| id == &r.id) {
                        inner.replacement_request = None;
                    }
                    if let Some(tx) = inner.waiters.remove(r.id.as_str()) {
                        let _ = tx.send(EotPrediction {
                            end_of_turn_probability: r.end_of_turn_probability,
                            last_speaking_time: r.last_speaking_time,
                            detection_delay: r.detection_delay,
                            inference_duration: r.inference_duration,
                            backchannel_probability: r.backchannel_probability,
                        });
                    }
                }
                EotAction::StartInference(s) => {
                    if let Some(shell) = &inner.shell {
                        shell.transport.run_inference(&s.id);
                    } else {
                        inner.replacement_request = Some((s.id, inner.replacement_audio.read()));
                    }
                }
                EotAction::Flush(_) => {
                    inner.replacement_audio.clear();
                    if let Some(shell) = &inner.shell {
                        shell.transport.flush();
                    }
                }
                EotAction::SwapToLocal(_) => {
                    // Keep the old shell alive in the run loop until after this lock is released.
                    // Its replacement is constructed there, never on the host's calling thread.
                    inner.retired = inner.shell.take();
                    self.swaps.send_modify(|n| *n += 1);
                }
                EotAction::Degrade(_) => inner.degraded = true,
                EotAction::Warn(w) => warnings.push(w),
            }
        }
        warnings
    }

    fn warn(warnings: Vec<EotWarning>) {
        for warning in warnings {
            log::warn!(target: LOG, "{}", sentence(&warning));
        }
    }

    fn on_event(&self, shell: &Arc<MountedTransport>, event: SinkEvent) {
        match event {
            SinkEvent::Prediction(p) => {
                let mut inner = self.lock();
                if !inner.shell.as_ref().is_some_and(|current| Arc::ptr_eq(current, shell)) {
                    return;
                }
                let prediction = TransportPrediction {
                    probability: p.probability,
                    detection_delay: p.detection_delay,
                    inference_duration: p.inference_duration,
                    backchannel_probability: p.backchannel_probability,
                };
                let actions = inner.session.resolve(now(), &p.id, prediction);
                let warnings = self.apply(&mut inner, actions);
                drop(inner);
                Self::warn(warnings);
            }
            SinkEvent::Metrics(m) => {
                if let Some(cb) = &self.on_metrics {
                    cb(*m);
                }
            }
            SinkEvent::Warn(w) => log::warn!(target: LOG, "{}", sentence(&w)),
            SinkEvent::End => {}
        }
    }

    /// `_run`: the mounted transport for its lifetime, remounted after a swap.
    async fn run(self: Arc<Self>) {
        let mut swaps = self.swaps.subscribe();
        loop {
            swaps.borrow_and_update();
            let (shell, retired, closed, model) = {
                let mut inner = self.lock();
                (
                    inner.shell.clone(),
                    inner.retired.take(),
                    inner.degraded || inner.session.is_closed(),
                    inner.session.model(),
                )
            };
            if let Some(retired) = retired {
                retired.detach();
            }
            if closed {
                return;
            }
            let Some(shell) = shell else {
                let make = self.make.clone();
                let (send, receive) = oneshot::channel();
                let admitted = self.media_pool.try_spawn(JobClass::EotModel, move || {
                    let _ = send.send(make(model));
                });
                let replacement = match admitted {
                    Ok(worker) => match worker.await {
                        Ok(()) => receive.await.map_err(|error| error.to_string()),
                        Err(error) => Err(error.to_string()),
                    },
                    Err(error @ MediaPoolError::Exhausted(_)) => {
                        let warnings = {
                            let mut inner = self.lock();
                            if inner.session.is_closed() {
                                return;
                            }
                            let actions =
                                inner.session.construction_refused(now(), &error.to_string());
                            let warnings = self.apply(&mut inner, actions);
                            swaps.borrow_and_update();
                            warnings
                        };
                        Self::warn(warnings);
                        // Another prediction retries; returning capacity alone queues no work.
                        if swaps.changed().await.is_err() {
                            return;
                        }
                        continue;
                    }
                    Err(error) => Err(error.to_string()),
                };
                match replacement {
                    Ok(shell) => {
                        let mut inner = self.lock();
                        if inner.session.is_closed() {
                            drop(inner);
                            shell.detach();
                            return;
                        }
                        if let Some((id, audio)) = inner.replacement_request.take() {
                            shell.transport.push_frame(&pcm_frame(audio, self.rate, 1));
                            shell.transport.run_inference(&id);
                            shell.transport.flush();
                        }
                        let audio = inner.replacement_audio.read();
                        inner.replacement_audio.clear();
                        if !audio.is_empty() {
                            shell.transport.push_frame(&pcm_frame(audio, self.rate, 1));
                        }
                        inner.shell = Some(shell);
                    }
                    Err(error) => {
                        let mut inner = self.lock();
                        let actions = inner.session.construction_failed(now(), &error);
                        let warnings = self.apply(&mut inner, actions);
                        drop(inner);
                        Self::warn(warnings);
                        return;
                    }
                }
                continue;
            };
            let events = async {
                while let Some(ev) = shell.next_event().await {
                    self.on_event(&shell, ev);
                }
                std::future::pending::<()>().await
            };
            let outcome = tokio::select! {
                r = shell.run() => Some(r),
                _ = swaps.changed() => None,
                () = events => None,
            };
            match outcome {
                // swapped: remount
                None => {
                    shell.detach();
                    continue;
                }
                Some(Ok(())) => {
                    if swaps.has_changed().unwrap_or(false) {
                        continue;
                    }
                    return;
                }
                Some(Err(ApiError { message, .. })) => {
                    let mut inner = self.lock();
                    if !inner.shell.as_ref().is_some_and(|current| Arc::ptr_eq(current, &shell)) {
                        drop(inner);
                        shell.detach();
                        continue;
                    }
                    let actions = inner.session.transport_failed(now(), &message);
                    let warnings = self.apply(&mut inner, actions);
                    let replaced = inner.shell.is_none();
                    drop(inner);
                    Self::warn(warnings);
                    shell.detach();
                    if !replaced {
                        return;
                    }
                }
            }
        }
    }
}

/// `_core.EotStream`: a stream the detector opened, shared with its run loop.
pub struct DetectorStream(Arc<EotStream>);

impl lk_agents_abi::turn::inference::EotStream for DetectorStream {
    fn model(&self) -> TurnDetectorModel {
        self.0.lock().session.model()
    }

    fn is_fallback(&self) -> bool {
        self.0.lock().session.is_fallback()
    }

    fn is_degraded(&self) -> bool {
        self.0.lock().session.is_degraded()
    }

    fn predict(&self) -> impl Future<Output = Result<EotPrediction, ProviderError>> {
        lk_agents_abi::EotDetector::predict(self, &telemetry::Cx::NONE, None)
    }

    fn end_input(&self) {
        let mut inner = self.0.lock();
        let actions = inner.session.end_input(now());
        let warnings = self.0.apply(&mut inner, actions);
        if let Some(shell) = &inner.shell {
            shell.transport.end_input();
        }
        self.0.swaps.send_modify(|n| *n += 1);
        drop(inner);
        EotStream::warn(warnings);
    }

    fn aclose(&self) -> impl Future<Output = ()> {
        lk_agents_abi::EotDetector::aclose(self, &telemetry::Cx::NONE)
    }
}

impl DetectorStream {
    fn hear(&self, frame: AudioFrame, head: bool) {
        let mut inner = self.0.lock();
        let format = (frame.sample_rate, frame.num_channels);
        match inner.input_format {
            None => {
                inner.input_format = Some(format);
                if frame.sample_rate != self.0.rate {
                    inner.resampler = lk_agents_media::Resampler::new(
                        frame.sample_rate,
                        self.0.rate,
                        u32::from(frame.num_channels),
                        Quality::Quick,
                    )
                    .ok();
                }
            }
            Some(f) if f != format => {
                drop(inner);
                log::error!(target: LOG, "a frame with a different audio format was already pushed");
                return;
            }
            Some(_) => {}
        }
        let out = match inner.resampler.as_mut() {
            Some(r) => match r.push(&lk_agents_media::pcm_samples(&frame.data)) {
                Ok(s) if !s.is_empty() => {
                    Some(pcm_frame(s.to_vec(), self.0.rate, frame.num_channels))
                }
                _ => None,
            },
            None => Some(frame),
        };
        if let Some(out) = out {
            EotStream::push(&mut inner, out, head);
        }
    }
}

impl lk_agents_abi::EotDetector for DetectorStream {
    fn id(&self) -> ProviderId {
        ProviderId { model: self.model().as_str().into(), ..self.0.id.clone() }
    }

    fn push_audio(&self, frame: AudioFrame) {
        self.hear(frame, false);
    }

    fn push_head(&self, frame: AudioFrame) {
        self.hear(frame, true);
    }

    /// The cloud transport is mounted at open, before the session that observes it.
    fn observe(&self, cx: &telemetry::Cx) {
        if let Some(shell) = &self.0.lock().shell {
            shell.transport.observe(cx.clone());
        }
    }

    fn cancel_inference(&self, timed_out: bool) {
        let reason = if timed_out { CancelReason::TimedOut } else { CancelReason::Superseded };
        let mut inner = self.0.lock();
        let actions = inner.session.cancel(now(), None, reason);
        let warnings = self.0.apply(&mut inner, actions);
        drop(inner);
        EotStream::warn(warnings);
    }

    fn flush(&self, _reason: Option<String>) {
        let mut inner = self.0.lock();
        if let Some(mut r) = inner.resampler.take() {
            let tail = r.flush().map(<[i16]>::to_vec).unwrap_or_default();
            if !tail.is_empty() {
                let ch = inner.input_format.map_or(1, |(_, c)| c);
                EotStream::push(&mut inner, pcm_frame(tail, self.0.rate, ch), false);
            }
        }
        inner.input_format = None;
        let actions = inner.session.flush(now());
        let warnings = self.0.apply(&mut inner, actions);
        drop(inner);
        EotStream::warn(warnings);
    }

    fn thresholds(&self) -> Option<ThresholdOptions> {
        Some(self.0.lock().session.thresholds())
    }

    /// Ask for a prediction over the audio so far; supersedes any in flight. The request is
    /// registered when called, so a later cancel or flush settles it.
    fn predict(
        &self,
        _cx: &telemetry::Cx,
        _timeout: Option<std::time::Duration>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<EotPrediction, ProviderError>> {
        let rx = {
            let mut inner = self.0.lock();
            let prefix = match inner.shell.as_ref().and_then(|s| s.transport.session_id()) {
                Some(sid) => format!("{sid}_eot_"),
                None => "turn_request_".to_owned(),
            };
            let request_id = crate::ids::shortuuid(&prefix);
            let (tx, rx) = oneshot::channel();
            inner.waiters.insert(request_id.clone(), tx);
            let actions = inner.session.start(now(), EotRequestId::from_string(request_id));
            let warnings = self.0.apply(&mut inner, actions);
            if inner.shell.is_none() && !inner.session.is_closed() {
                self.0.swaps.send_modify(|n| *n += 1);
            }
            drop(inner);
            EotStream::warn(warnings);
            rx
        };
        Box::pin(async {
            rx.await.map_err(|_| ApiError::new("turn detector stream closed").into())
        })
    }

    /// Closes when called; the future waits for the run loop to settle.
    fn aclose(&self, _cx: &telemetry::Cx) -> lk_agents_abi::BoxFuture<'_, ()> {
        self.end_input();
        let shell = self.0.lock().shell.clone();
        if let Some(shell) = shell {
            shell.detach();
        }
        lk_agents_abi::EotDetector::cancel_inference(self, false);
        let mut settled = self.0.settled.subscribe();
        Box::pin(async move {
            let _ = settled.wait_for(|done| *done).await;
        })
    }
}
