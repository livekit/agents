//! `TurnDetectorStream` (`inference/eot.py`): the session's decisions carried out against the
//! mounted transport, with the input resampled to the detector's rate, the transport swapped for
//! the local model on a cloud failure, and each `predict()` answered by its own waiter.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tokenize::language::TurnDetectorModel;
use lk_agents_abi::turn::eot::MetricsEvent;
use lk_agents_abi::utils::resample::{Quality, Resampler};
use lk_agents_abi::{AudioFrame, EotPrediction, ProviderId};
use tokio::sync::{oneshot, watch};

use crate::eot::{
    CancelReason, EotAction, EotRequestId, EotSession, EotWarning, SinkEvent, TransportPrediction,
    TransportShell,
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
    shell: Arc<TransportShell>,
    waiters: HashMap<String, oneshot::Sender<EotPrediction>>,
    resampler: Option<Box<dyn Resampler>>,
    input_format: Option<(u32, u16)>,
    degraded: bool,
}

/// One stream of the user's audio, answering `predict()` at each pause.
pub struct EotStream {
    inner: Mutex<Inner>,
    rate: u32,
    make: TransportFactory,
    on_metrics: Option<MetricsCallback>,
    /// Bumped by each swap, so the run loop remounts.
    swaps: watch::Sender<u64>,
    task: Mutex<Option<lk_agents_abi::AbortHandle>>,
    id: ProviderId,
}

impl EotStream {
    /// Mounts the transport for `session`'s model and starts the run loop on `spawner`.
    pub fn open(
        session: EotSession,
        rate: u32,
        make: TransportFactory,
        on_metrics: Option<MetricsCallback>,
        spawner: &dyn lk_agents_abi::Spawner,
    ) -> Arc<Self> {
        let shell = make(session.model());
        let id = ProviderId {
            label: "livekit.agents.inference.eot.TurnDetector".into(),
            model: session.model().as_str().into(),
            provider: crate::eot::PROVIDER.into(),
        };
        let stream = Arc::new(EotStream {
            inner: Mutex::new(Inner {
                session,
                shell,
                waiters: HashMap::new(),
                resampler: None,
                input_format: None,
                degraded: false,
            }),
            rate,
            make,
            on_metrics,
            swaps: watch::channel(0).0,
            task: Mutex::new(None),
            id,
        });
        let run = Arc::clone(&stream).run();
        let abort = lk_agents_abi::SpawnerExt::spawn_unboxed(spawner, run);
        *stream.task.lock().unwrap_or_else(|p| p.into_inner()) = Some(abort);
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
        let prefix = match inner.shell.transport.session_id() {
            Some(sid) => format!("{sid}_eot_"),
            None => "turn_request_".to_owned(),
        };
        let request_id = crate::ids::shortuuid(&prefix);
        let (tx, rx) = oneshot::channel();
        inner.waiters.insert(request_id.clone(), tx);
        let actions = inner.session.start(now(), EotRequestId::from_string(request_id));
        self.apply(&mut inner, actions);
        rx
    }

    pub fn cancel_inference(&self, timed_out: bool) {
        let reason = if timed_out { CancelReason::TimedOut } else { CancelReason::Superseded };
        let mut inner = self.lock();
        let actions = inner.session.cancel(now(), None, reason);
        self.apply(&mut inner, actions);
    }

    pub fn flush(&self) {
        let mut inner = self.lock();
        if let Some(mut r) = inner.resampler.take() {
            let tail = r.flush().map(<[i16]>::to_vec).unwrap_or_default();
            if !tail.is_empty() {
                let ch = inner.input_format.map_or(1, |(_, c)| c);
                Self::push(&inner, pcm_frame(tail, self.rate, ch));
            }
        }
        inner.input_format = None;
        let actions = inner.session.flush(now());
        self.apply(&mut inner, actions);
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
            Self::push(&inner, out);
        }
    }

    fn push(inner: &Inner, frame: AudioFrame) {
        if !inner.degraded {
            inner.shell.transport.push_frame(&frame);
        }
    }

    pub fn end_input(&self) {
        let mut inner = self.lock();
        let actions = inner.session.end_input(now());
        self.apply(&mut inner, actions);
        inner.shell.transport.end_input();
    }

    pub fn aclose(&self) {
        self.lock().shell.detach();
        self.end_input();
        if let Some(task) = self.task.lock().unwrap_or_else(|p| p.into_inner()).take() {
            task.abort();
        }
        self.cancel_inference(false);
    }

    fn apply(&self, inner: &mut Inner, actions: Vec<EotAction>) {
        for action in actions {
            match action {
                EotAction::Resolve(r) => {
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
                EotAction::StartInference(s) => inner.shell.transport.run_inference(&s.id),
                EotAction::Flush(_) => inner.shell.transport.flush(),
                EotAction::SwapToLocal(_) => {
                    inner.shell.detach();
                    inner.shell = (self.make)(inner.session.model());
                    self.swaps.send_modify(|n| *n += 1);
                }
                EotAction::Degrade(_) => inner.degraded = true,
                EotAction::Warn(w) => log::warn!(target: LOG, "{}", sentence(&w)),
            }
        }
    }

    fn on_event(&self, event: SinkEvent) {
        match event {
            SinkEvent::Prediction(p) => {
                let mut inner = self.lock();
                let prediction = TransportPrediction {
                    probability: p.probability,
                    detection_delay: p.detection_delay,
                    inference_duration: p.inference_duration,
                    backchannel_probability: p.backchannel_probability,
                };
                let actions = inner.session.resolve(now(), &p.id, prediction);
                self.apply(&mut inner, actions);
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
            let shell = {
                let inner = self.lock();
                if inner.degraded {
                    return;
                }
                Arc::clone(&inner.shell)
            };
            swaps.mark_unchanged();
            let events = async {
                while let Some(ev) = shell.next_event().await {
                    self.on_event(ev);
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
                None => continue,
                Some(Ok(())) => {
                    if swaps.has_changed().unwrap_or(false) {
                        continue;
                    }
                    return;
                }
                Some(Err(ApiError { message, .. })) => {
                    let mut inner = self.lock();
                    let actions = inner.session.transport_failed(now(), &message);
                    self.apply(&mut inner, actions);
                    if Arc::ptr_eq(&inner.shell, &shell) {
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

    async fn predict(
        &self,
        _timeout: Option<std::time::Duration>,
    ) -> Result<EotPrediction, ProviderError> {
        self.predict().await.map_err(|_| ApiError::new("turn detector stream closed").into())
    }

    async fn aclose(&self) {
        EotStream::aclose(self);
    }
}
