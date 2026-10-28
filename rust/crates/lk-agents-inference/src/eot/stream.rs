//! `TurnDetectorStream` (`inference/eot.py`): the session's decisions carried out against the
//! mounted transport, with the input resampled to the detector's rate, the transport swapped for
//! the local model on a cloud failure, and each `predict()` answered by its own waiter.

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tokenize::language::TurnDetectorModel;
use lk_agents_abi::turn::eot::MetricsEvent;
use lk_agents_abi::utils::resample::{Quality, Resampler};
use lk_agents_abi::{AudioFrame, EotPrediction, ProviderId};
use tokio::sync::{oneshot, watch};

use crate::eot::{
    AudioRingBuffer, CLIENT_BUFFER_SAMPLES, CancelReason, EotAction, EotRequestId, EotSession,
    EotWarning, SinkEvent, TransportPrediction, TransportShell,
};

/// Mounts the transport for a model: the gateway for `turn-detector-v1`, the mini model otherwise.
pub type TransportFactory = Arc<dyn Fn(TurnDetectorModel) -> Arc<TransportShell> + Send + Sync>;

/// `self._detector.emit("metrics_collected", ...)`.
pub type MetricsCallback = Arc<dyn Fn(MetricsEvent) + Send + Sync>;

const LOG: &str = "livekit.agents";

use lk_agents_abi::unix_now as now;

fn samples(frame: &AudioFrame) -> Vec<i16> {
    frame.data.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect()
}

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
    shell: Option<Arc<TransportShell>>,
    /// While the local model loads, retain only its trailing prediction window.
    replacement_audio: AudioRingBuffer,
    replacement_request: Option<(EotRequestId, Vec<i16>)>,
    retired: Option<Arc<TransportShell>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    waiters: HashMap<String, oneshot::Sender<EotPrediction>>,
    resampler: Option<Box<dyn Resampler>>,
    input_format: Option<(u32, u16)>,
    degraded: bool,
}

/// One stream of the user's audio, answering `predict()` at each pause.
pub struct EotStream {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    inner: Mutex<Inner>,
    rate: u32,
    make: TransportFactory,
    on_metrics: Option<MetricsCallback>,
    /// Bumped by each swap, so the run loop remounts.
    swaps: watch::Sender<u64>,
    settled: watch::Sender<bool>,
    id: ProviderId,
}

impl EotStream {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    /// Mounts the transport for `session`'s model and starts the run loop on `spawner`.
    pub fn open(
        session: EotSession,
        rate: u32,
        make: TransportFactory,
        on_metrics: Option<MetricsCallback>,
        spawner: &dyn lk_agents_abi::Spawner,
    ) -> Arc<Self> {
        let shell = (session.model() == TurnDetectorModel::V1).then(|| make(session.model()));
        let id = ProviderId {
            label: "livekit.agents.inference.eot.TurnDetector".into(),
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
            on_metrics,
            swaps: watch::channel(0).0,
            settled: watch::channel(false).0,
            id,
        });
        let owner = Arc::clone(&stream);
        let settled = stream.settled.clone();
        let run = async move {
            owner.run().await;
            settled.send_replace(true);
        };
        // The owner must outlive close if model construction is still settling.
        lk_agents_abi::SpawnerExt::spawn_unboxed(spawner, run);
        stream
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn model(&self) -> &'static str {
        self.lock().session.model().as_str()
    }

    pub fn is_fallback(&self) -> bool {
        self.lock().session.is_fallback()
    }

    pub fn is_degraded(&self) -> bool {
        self.lock().session.is_degraded()
    }

    /// Ask for a prediction over the audio so far; supersedes any in flight.
    pub fn predict(&self) -> oneshot::Receiver<EotPrediction> {
        let mut inner = self.lock();
        let prefix = match inner.shell.as_ref().and_then(|s| s.transport.session_id()) {
            Some(sid) => format!("{sid}_eot_"),
            None => "turn_request_".to_owned(),
        };
        let request_id = crate::ids::shortuuid(&prefix);
        let (tx, rx) = oneshot::channel();
        inner.waiters.insert(request_id.clone(), tx);
        let actions = inner.session.start(now(), EotRequestId::from_string(request_id));
        let warnings = self.apply(&mut inner, actions);
        drop(inner);
        Self::warn(warnings);
        rx
    }

    pub fn cancel_inference(&self, timed_out: bool) {
        let reason = if timed_out { CancelReason::TimedOut } else { CancelReason::Superseded };
        let mut inner = self.lock();
        let actions = inner.session.cancel(now(), None, reason);
        let warnings = self.apply(&mut inner, actions);
        drop(inner);
        Self::warn(warnings);
    }

    pub fn flush(&self) {
        let mut inner = self.lock();
        if let Some(mut r) = inner.resampler.take() {
            let tail = r.flush().map(<[i16]>::to_vec).unwrap_or_default();
            if !tail.is_empty() {
                let ch = inner.input_format.map_or(1, |(_, c)| c);
                Self::push(&mut inner, pcm_frame(tail, self.rate, ch));
            }
        }
        inner.input_format = None;
        let actions = inner.session.flush(now());
        let warnings = self.apply(&mut inner, actions);
        drop(inner);
        Self::warn(warnings);
    }

    pub fn push_audio(&self, frame: AudioFrame) {
        let mut inner = self.lock();
        let format = (frame.sample_rate, frame.num_channels);
        match inner.input_format {
            None => {
                inner.input_format = Some(format);
                if frame.sample_rate != self.rate {
                    inner.resampler = lk_agents_media::new_resampler(
                        frame.sample_rate,
                        self.rate,
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
            Some(r) => match r.push(&samples(&frame)) {
                Ok(s) if !s.is_empty() => {
                    Some(pcm_frame(s.to_vec(), self.rate, frame.num_channels))
                }
                _ => None,
            },
            None => Some(frame),
        };
        if let Some(out) = out {
            Self::push(&mut inner, out);
        }
    }

    fn push(inner: &mut Inner, frame: AudioFrame) {
        if !inner.degraded && !inner.session.is_closed() {
            if let Some(shell) = &inner.shell {
                shell.transport.push_frame(&frame);
            } else {
                inner.replacement_audio.push(&samples(&frame));
            }
        }
    }

    pub fn end_input(&self) {
        let mut inner = self.lock();
        let actions = inner.session.end_input(now());
        let warnings = self.apply(&mut inner, actions);
        if let Some(shell) = &inner.shell {
            shell.transport.end_input();
        }
        self.swaps.send_modify(|n| *n += 1);
        drop(inner);
        Self::warn(warnings);
    }

    pub fn aclose(&self) {
        self.end_input();
        let shell = self.lock().shell.clone();
        if let Some(shell) = shell {
            shell.detach();
        }
        self.cancel_inference(false);
    }

    pub async fn wait_closed(&self) {
        let mut settled = self.settled.subscribe();
        let _ = settled.wait_for(|done| *done).await;
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

    fn on_event(&self, shell: &Arc<TransportShell>, event: SinkEvent) {
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

    #[allow(clippy::disallowed_macros, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
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
                // Admission precedes spawn_blocking: simultaneous cloud failures cannot fill
                // Tokio's blocking queue with model constructors waiting for the GIL.
                static CONSTRUCTION: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
                let permit = tokio::select! {
                    permit = CONSTRUCTION.acquire() => permit.expect("construction semaphore"),
                    _ = swaps.changed() => continue,
                };
                let make = self.make.clone();
                let replacement = tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    make(model)
                })
                .await;
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
                        let actions = inner.session.transport_failed(now(), &error.to_string());
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

#[async_trait::async_trait]
impl lk_agents_abi::EotDetector for EotStream {
    fn id(&self) -> ProviderId {
        ProviderId { model: self.model().into(), ..self.id.clone() }
    }

    fn push_audio(&self, frame: AudioFrame) {
        EotStream::push_audio(self, frame);
    }

    fn cancel(&self, timed_out: bool) {
        self.cancel_inference(timed_out);
    }

    fn flush(&self, _reason: &'static str) {
        EotStream::flush(self);
    }

    fn thresholds(&self) -> Option<lk_agents_abi::tokenize::language::ThresholdOptions> {
        Some(self.lock().session.thresholds())
    }

    async fn predict(
        &self,
        _timeout: Option<std::time::Duration>,
    ) -> Result<EotPrediction, ProviderError> {
        self.predict().await.map_err(|_| ApiError::new("turn detector stream closed").into())
    }

    async fn aclose(&self) {
        EotStream::aclose(self);
        self.wait_closed().await;
    }
}
