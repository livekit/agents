//! The end-of-turn detector seam: an audio detector (`_StreamingTurnDetector.stream()`) that hears
//! the user and predicts on request. Thresholds come from [`crate::tokenize::language`].

use std::time::Duration;

use crate::EotPrediction;
use crate::errors::ProviderError;

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

/// A host `turn_detection` object is this trait through `_core_view.TurnDetector`, which speaks
/// the detector's `stream()` behind the `_core_*` surface.
#[crate::host(proxy(host = "livekit.agents._bindings.TurnDetector"))]
pub trait EotDetector: Send + Sync + 'static {
    /// The detector's `model`/`provider`, as `EOUMetrics.metadata` names them.
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label("turn_detector")
    }

    /// One frame of the user's audio. Fire-and-forget, gated as the STT leg is.
    #[host(notify(lossy), name = "_core_push_audio")]
    fn push_audio(&self, frame: crate::AudioFrame);

    /// Pre-connect audio: kept in full where `push_audio` would drop to stay current.
    fn push_head(&self, frame: crate::AudioFrame) {
        self.push_audio(frame);
    }

    /// Measures the stream's input under `cx`'s job: the host opens it before the session.
    fn observe(&self, _cx: &crate::telemetry::Cx) {}

    /// `cancel_inference(timed_out=..)` (`audio_recognition.py:1592`); the detector reports a
    /// timeout and a superseding speech differently.
    #[host(notify, name = "_core_cancel")]
    fn cancel_inference(&self, timed_out: bool);

    /// `flush(reason)`: a segment boundary, not a close.
    #[host(notify, name = "_core_flush")]
    fn flush(&self, reason: Option<String>);
    /// Ask for a prediction (`stream.predict()`).
    ///
    /// The countdown already bounds `timeout` itself, so ignoring it cannot hang a turn. An `Err`
    /// is not a turn failure: the countdown proceeds as if no detector were configured.
    #[host(await, name = "_core_predict")]
    fn predict(
        &self,
        cx: &crate::telemetry::Cx,
        timeout: Option<Duration>,
    ) -> crate::BoxFuture<'_, Result<EotPrediction, ProviderError>>;

    /// What a prediction is judged against (`unlikely_threshold(language)`,
    /// `backchannel_threshold(language)`, `supports_language(language)`), as it is now: the user's
    /// overrides over the model's defaults, the gateway's once its session is created. `None`: the
    /// detector keeps no table, and the mini model's stock one judges it. Native detectors only:
    /// a host detector answers per language itself, so its proxy keeps this default.
    fn thresholds(&self) -> Option<crate::tokenize::language::ThresholdOptions> {
        None
    }

    /// Close what the detector holds open (a streaming detector's stream) at session close: an
    /// open stream keeps the session alive (`audio_recognition.py:855`). Nothing by default.
    #[host(await, name = "_core_aclose")]
    fn aclose(&self, _cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, ()> {
        Box::pin(async move {})
    }
}
