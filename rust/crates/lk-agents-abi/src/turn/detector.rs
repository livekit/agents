//! The end-of-turn detector seam: an audio detector (`inference.TurnDetector.stream()`) that hears
//! the user and predicts on request. Thresholds come from [`crate::tokenize::language`].

use std::time::Duration;

use crate::EotPrediction;
use crate::errors::ProviderError;

/// Why the session flushes the detector's segment; the host logs these words.
#[crate::host(record(module = "turn_events", extract, emit))]
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushReason {
    /// `audio_recognition._bounce_eou_task`: the user turn was accepted.
    #[host(rename = "turn committed")]
    TurnCommitted,
    /// `AudioRecognition._clear_user_turn`: the turn was dropped and its state cleared.
    ClearUserTurn,
}

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

/// The session's end-of-turn detector: an `inference.TurnDetector`'s stream, which hears the user
/// and predicts on request. Only a native detector crosses: there is no host protocol to proxy.
#[crate::host(native(
    refused = "turn_detection must be a turn detection mode or an `inference.TurnDetector`"
))]
pub trait EotDetector: Send + Sync + 'static {
    /// The detector's `model`/`provider`, as `EOUMetrics.metadata` names them.
    fn id(&self) -> crate::ProviderId {
        crate::types::identity::ProviderId::of_label("turn_detector")
    }

    /// One frame of the user's audio. Fire-and-forget, gated as the STT leg is.
    fn push_audio(&self, frame: crate::AudioFrame);

    /// Pre-connect audio: kept in full where `push_audio` would drop to stay current.
    fn push_head(&self, frame: crate::AudioFrame) {
        self.push_audio(frame);
    }

    /// Measures the stream's input under `cx`'s job: the host opens it before the session.
    fn observe(&self, _cx: &crate::telemetry::Cx) {}

    /// `cancel_inference(timed_out=..)` (`audio_recognition.py:1592`); the detector reports a
    /// timeout and a superseding speech differently.
    fn cancel_inference(&self, timed_out: bool);

    /// `flush(reason)`: a segment boundary, not a close.
    fn flush(&self, reason: Option<FlushReason>);
    /// Ask for a prediction (`stream.predict()`).
    ///
    /// The countdown already bounds `timeout` itself, so ignoring it cannot hang a turn. An `Err`
    /// is not a turn failure: the countdown proceeds as if no detector were configured.
    fn predict(
        &self,
        cx: &crate::telemetry::Cx,
        timeout: Option<Duration>,
    ) -> crate::BoxFuture<'_, Result<EotPrediction, ProviderError>>;

    /// What a prediction is judged against (`unlikely_threshold(language)`,
    /// `backchannel_threshold(language)`, `supports_language(language)`), as it is now: the user's
    /// overrides over the model's defaults, the gateway's once its session is created.
    fn thresholds(&self) -> crate::tokenize::language::ThresholdOptions;

    /// Close what the detector holds open (a streaming detector's stream) at session close: an
    /// open stream keeps the session alive (`audio_recognition.py:855`). Nothing by default.
    fn aclose(&self, _cx: &crate::telemetry::Cx) -> crate::BoxFuture<'_, ()> {
        Box::pin(async move {})
    }
}
