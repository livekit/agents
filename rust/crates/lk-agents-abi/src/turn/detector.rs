//! The end-of-turn detector seam: batch (`_TurnDetector.predict_end_of_turn`) and streaming
//! (`_StreamingTurnDetector.stream()`) behind one trait. Thresholds come from [`crate::tokenize::language`].

use std::sync::Arc;
use std::time::Duration;

use crate::errors::ProviderError;
use crate::{AudioFrame, ChatContext, EotPrediction, Spawner, SpawnerExt, StreamSink};

/// Strictly increasing per session; the driver drops an answer whose id is not the outstanding one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReqId(pub u64);

/// `req` is `None` when the leg cannot correlate; `recoverable` is `false` on the attempt that
/// gives up.
#[derive(Debug, Clone, PartialEq)]
pub enum LegEvent<V> {
    Verdict { req: Option<ReqId>, verdict: V },
    Error { req: Option<ReqId>, recoverable: bool, err: ProviderError },
    Closed,
}

/// The driver runs one pump per leg onto the session's input queue, so no leg is polled inside the
/// loop's `select!` and none has to be cancel-safe.
#[async_trait::async_trait]
pub trait DetectorLeg: Send + Sync {
    /// Not `Copy`: the EOT request carries the chat context.
    type Signal: Send;
    type Verdict: Send;

    /// `None` for a leg that hears no audio.
    fn audio(&self) -> Option<Arc<dyn StreamSink<AudioFrame>>>;

    /// Non-blocking (called from the session loop). `None` for a signal that opens no request.
    fn signal(&self, req: Option<ReqId>, signal: Self::Signal);

    /// Pumped by one task per leg; `Closed` ends that task.
    async fn next(&self) -> LegEvent<Self::Verdict>;
}

/// A host `turn_detection` object is this trait through its bare `#[host(proxy)]` (`PyEotDetector`),
/// over `voice.turn._CoreTurnDetector`, which speaks both reference protocols (batch
/// `predict_end_of_turn`, streaming `stream()`) behind the `_core_*` surface.
#[crate::host(proxy(host = "livekit.agents.voice.turn._CoreTurnDetector"))]
#[async_trait::async_trait]
pub trait EotDetector: Send + Sync + 'static {
    /// The detector's `model`/`provider`, as `EOUMetrics.metadata` names them.
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label("turn_detector")
    }

    /// One frame of the user's audio, for a streaming detector. Fire-and-forget, gated as the STT
    /// leg is; a batch detector ignores it.
    fn push_audio(&self, frame: crate::AudioFrame);

    /// `cancel_inference(timed_out=..)` (`audio_recognition.py:1592`); the detector reports a
    /// timeout and a superseding speech differently.
    fn cancel(&self, timed_out: bool);

    /// `flush(reason)`: a segment boundary, not a close.
    fn flush(&self, reason: &'static str);
    /// Ask for a prediction: `stream.predict()` for a streaming detector, or
    /// `predict_end_of_turn(chat_ctx)` for a batch one.
    ///
    /// The countdown already bounds `timeout` itself, so ignoring it cannot hang a turn. An `Err`
    /// is not a turn failure: the countdown proceeds as if no detector were configured.
    async fn predict(
        &self,
        chat_ctx: &ChatContext,
        timeout: Option<Duration>,
    ) -> Result<EotPrediction, ProviderError>;

    /// Close what the detector holds open (a streaming detector's stream) at session close: an
    /// open stream keeps the session alive (`audio_recognition.py:855`). Nothing by default.
    async fn aclose(&self) {}
}

/// The conversation a host turn detector scores, as `_core.TurnContext`: its
/// `ChatContext.to_dict()` form (timestamps kept; images and audio are not rendered), which the
/// host rebuilds with `ChatContext.from_dict`.
#[crate::host(object(name = "TurnContext"))]
impl ChatContext {
    #[host(method)]
    pub fn to_host_dict(&self) -> serde_json::Value {
        let opts = crate::llm::to_dict::ToDictOptions { exclude_timestamp: false, ..Default::default() };
        self.to_dict(&opts, &Default::default()).unwrap_or(serde_json::Value::Null)
    }
}

#[derive(Debug, Clone)]
pub struct EotRequest {
    pub chat_ctx: ChatContext,
    pub timeout: Option<Duration>,
}

/// A failed prediction is `Error { recoverable: true }`; a superseded request's late answer keeps
/// its own id and the driver drops it. Never `Closed` on its own.
pub struct EotLeg<D: EotDetector + ?Sized> {
    detector: Arc<D>,
    spawner: Arc<dyn Spawner>,
    tx: tokio::sync::mpsc::UnboundedSender<LegEvent<EotPrediction>>,
    rx: tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<LegEvent<EotPrediction>>>,
    in_flight: std::sync::Mutex<Vec<crate::AbortHandle>>,
}

impl<D: EotDetector + ?Sized> EotLeg<D> {
    pub fn new(detector: Arc<D>, spawner: Arc<dyn Spawner>) -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        EotLeg { detector, spawner, tx, rx: tokio::sync::Mutex::new(rx), in_flight: std::sync::Mutex::new(Vec::new()) }
    }

    pub fn close(&self) {
        for task in self.in_flight.lock().unwrap_or_else(|p| p.into_inner()).drain(..) {
            task.abort();
        }
    }
}

impl<D: EotDetector + ?Sized> std::ops::Deref for EotLeg<D> {
    type Target = D;
    fn deref(&self) -> &D {
        &self.detector
    }
}

#[async_trait::async_trait]
impl<D: EotDetector + ?Sized> DetectorLeg for EotLeg<D> {
    type Signal = EotRequest;
    type Verdict = EotPrediction;

    fn audio(&self) -> Option<Arc<dyn StreamSink<AudioFrame>>> {
        Some(Arc::new(EotAudioSink(Arc::clone(&self.detector))))
    }

    fn signal(&self, req: Option<ReqId>, signal: Self::Signal) {
        let detector = Arc::clone(&self.detector);
        let tx = self.tx.clone();
        let EotRequest { chat_ctx, timeout } = signal;
        // Spawned: the driver is mid-transition and cannot await here. Failures travel too.
        let task = self.spawner.spawn_unboxed(async move {
            let ev = match detector.predict(&chat_ctx, timeout).await {
                Ok(verdict) => LegEvent::Verdict { req, verdict },
                Err(err) => LegEvent::Error { req, recoverable: true, err },
            };
            let _ = tx.send(ev);
        });
        let mut in_flight = self.in_flight.lock().unwrap_or_else(|p| p.into_inner());
        in_flight.retain(|t| !t.is_finished());
        in_flight.push(task);
    }

    async fn next(&self) -> LegEvent<EotPrediction> {
        match self.rx.lock().await.recv().await {
            Some(ev) => ev,
            None => LegEvent::Closed,
        }
    }
}

/// `flush` is not the detector's `flush`, which is a turn boundary the machine decides.
struct EotAudioSink<D: EotDetector + ?Sized>(Arc<D>);

impl<D: EotDetector + ?Sized> StreamSink<AudioFrame> for EotAudioSink<D> {
    fn push(&self, frame: AudioFrame) -> Result<(), ProviderError> {
        self.0.push_audio(frame);
        Ok(())
    }

    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        // The detector's stream outlives a handoff.
    }
}
