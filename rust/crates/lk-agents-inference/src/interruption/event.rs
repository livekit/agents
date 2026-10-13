use lk_agents_abi::errors::ApiError;
use lk_agents_abi::metrics::base::{InterruptionMetrics, Metadata};

use crate::interruption::options::MIN_INTERRUPTION_DURATION;

/// One model frame, in seconds.
const FRAME_DURATION: f64 = 0.025;

/// The 2nd-largest per-frame probability (the reference's body, not its docstring); 0.0 for fewer than two frames.
#[lk_agents_abi::host(export(name = "interruption_estimate_probability"))]
pub fn estimate_probability(probabilities: &[f32]) -> f64 {
    let n_th = (MIN_INTERRUPTION_DURATION / FRAME_DURATION).ceil() as usize;
    if probabilities.len() < n_th {
        return 0.0;
    }
    let mut scratch = probabilities.to_vec();
    // `total_cmp`: NaN sorts high as in numpy's partition, instead of panicking.
    let idx = scratch.len() - n_th;
    let (_, nth, _) = scratch.select_nth_unstable_by(idx, f32::total_cmp);
    *nth as f64
}

/// `InterruptionCacheEntry` (`inference/interruption.py:88`): one in-flight or answered request.
/// `total_duration.is_some()` means "answered".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheEntry {
    /// `perf_counter_ns()` when the request went out: the cache key and the id the server echoes,
    /// so a raw `u64`.
    pub created_at: u64,
    pub speech_input: Option<Vec<i16>>,
    pub total_duration: Option<f64>,
    pub prediction_duration: Option<f64>,
    pub detection_delay: Option<f64>,
    pub probabilities: Option<Vec<f32>>,
    pub is_interruption: Option<bool>,
}

impl CacheEntry {
    pub fn new(created_at: u64) -> Self {
        CacheEntry { created_at, ..Default::default() }
    }

    /// Round-trip time in seconds; 0.0 when unanswered.
    pub fn total_duration(&self) -> f64 {
        self.total_duration.unwrap_or(0.0)
    }

    pub fn prediction_duration(&self) -> f64 {
        self.prediction_duration.unwrap_or(0.0)
    }

    pub fn detection_delay(&self) -> f64 {
        self.detection_delay.unwrap_or(0.0)
    }

    pub fn probability(&self) -> f64 {
        match &self.probabilities {
            Some(p) => estimate_probability(p),
            None => 0.0,
        }
    }
}

pub use lk_agents_abi::voice::events::OverlappingSpeechEvent;

/// `OverlappingSpeechEvent.from_cache_entry` (`inference/interruption.py:174`). `wall` is
/// `time.time()`; `ended_at` falls back on truthiness, as in the reference.
pub fn overlapping_speech_from_cache_entry(
    entry: &CacheEntry,
    is_interruption: bool,
    started_at: Option<f64>,
    ended_at: Option<f64>,
    agent_ended: bool,
    wall: f64,
) -> OverlappingSpeechEvent {
    OverlappingSpeechEvent {
        created_at: wall,
        detected_at: ended_at.filter(|v| *v != 0.0).unwrap_or(wall),
        is_interruption,
        agent_ended,
        overlap_started_at: started_at,
        speech_input: entry.speech_input.clone(),
        probabilities: entry.probabilities.clone(),
        total_duration: entry.total_duration(),
        detection_delay: entry.detection_delay(),
        prediction_duration: entry.prediction_duration(),
        probability: entry.probability(),
        num_requests: 0,
    }
}

/// `_metrics_monitor_task`'s body (`inference/interruption.py:659`), as a fold. An agent-ended
/// overlap is not a backchannel (#6564).
pub fn interruption_metrics(
    ev: &OverlappingSpeechEvent,
    wall: f64,
    model: &str,
    provider: &str,
) -> InterruptionMetrics {
    InterruptionMetrics {
        timestamp: wall,
        total_duration: ev.total_duration,
        prediction_duration: ev.prediction_duration,
        detection_delay: ev.detection_delay,
        num_interruptions: u32::from(ev.is_interruption),
        num_backchannels: u32::from(!ev.is_interruption && !ev.agent_ended),
        num_requests: ev.num_requests,
        metadata: Some(Metadata {
            model_name: Some(model.to_owned()),
            model_provider: Some(provider.to_owned()),
        }),
    }
}

/// `InterruptionDetectionError` (`inference/interruption.py:59`). `recoverable` is `true` while
/// the retry loop will reconnect, `false` when it gives up.
#[derive(Debug, Clone, PartialEq)]
pub struct InterruptionDetectionError {
    pub timestamp: f64,
    pub label: String,
    pub error: ApiError,
    pub recoverable: bool,
}

impl InterruptionDetectionError {
    pub const TYPE: &'static str = "interruption_detection_error";
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: the max or min instead of the 2nd largest, the docstring's sliding-window min-max
    // instead of the n-th largest, an off-by-one in the length guard (`select_nth_unstable` panics
    // on one frame), or `partial_cmp().unwrap()` panicking on NaN.
    #[test]
    fn probability_is_the_second_highest_frame_score() {
        for (scores, want) in [
            (&[0.1, 0.9, 0.4, 0.8][..], 0.8_f32 as f64),
            (&[0.9, 0.1], 0.1_f32 as f64),
            (&[0.9, 0.0, 0.9], 0.9_f32 as f64),
            (&[], 0.0),
            (&[0.99], 0.0),
            (&[0.99, 0.99], 0.99_f32 as f64),
            (&[0.3, f32::NAN, 0.2], 0.3_f32 as f64),
        ] {
            assert_eq!(estimate_probability(scores), want, "{scores:?}");
        }
    }

    // Catches: #6564's fix being lost (agent-ended overlaps counted as backchannels).
    #[test]
    fn an_agent_ended_overlap_counts_as_neither_interruption_nor_backchannel() {
        let base = overlapping_speech_from_cache_entry(
            &CacheEntry::default(),
            false,
            Some(1.0),
            Some(2.0),
            true,
            3.0,
        );
        let m = interruption_metrics(&base, 3.0, "m", "p");
        assert_eq!((m.num_interruptions, m.num_backchannels), (0, 0));

        let user_ended = OverlappingSpeechEvent { agent_ended: false, ..base.clone() };
        let m = interruption_metrics(&user_ended, 3.0, "m", "p");
        assert_eq!((m.num_interruptions, m.num_backchannels), (0, 1));

        let interrupted = OverlappingSpeechEvent { is_interruption: true, ..base };
        let m = interruption_metrics(&interrupted, 3.0, "m", "p");
        assert_eq!((m.num_interruptions, m.num_backchannels), (1, 0));
    }

    // Catches: `detected_at` read from the clock instead of the supplied `ended_at`.
    #[test]
    fn detected_at_prefers_the_supplied_end_timestamp() {
        let ev = overlapping_speech_from_cache_entry(
            &CacheEntry::default(),
            false,
            Some(1.0),
            Some(2.0),
            false,
            9.0,
        );
        assert_eq!((ev.detected_at, ev.created_at), (2.0, 9.0));
        let ev = overlapping_speech_from_cache_entry(
            &CacheEntry::default(),
            false,
            Some(1.0),
            None,
            false,
            9.0,
        );
        assert_eq!(ev.detected_at, 9.0);
    }

    // Catches: a missing entry's `None`s propagating as anything but 0.0.
    #[test]
    fn an_empty_entry_yields_zeroed_durations_not_missing_ones() {
        let ev = overlapping_speech_from_cache_entry(
            &CacheEntry::default(),
            false,
            None,
            None,
            false,
            0.0,
        );
        assert_eq!(
            (ev.total_duration, ev.prediction_duration, ev.detection_delay, ev.probability),
            (0.0, 0.0, 0.0, 0.0)
        );
    }
}
