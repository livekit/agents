use lk_agents_abi::Ts;
use std::time::Duration;

/// `VoiceActivityVideoSampler`: sample video faster while the user is talking. The last-pass
/// instant moves only on a pass, so the rate doesn't depend on the camera's frame rate.
#[derive(Clone)]
pub struct VideoSampler {
    state: SamplerState,
}

#[derive(Clone)]
struct SamplerState {
    speaking_fps: f64,
    silent_fps: f64,
    /// On the caller's clock. `None` until the first frame passes.
    last_sampled: Option<Ts>,
}

impl VideoSampler {
    /// The reference's defaults (1.0 and 0.3) live at the call site.
    pub fn new(speaking_fps: f64, silent_fps: f64) -> Self {
        VideoSampler { state: SamplerState { speaking_fps, silent_fps, last_sampled: None } }
    }
    /// Should this frame reach the model? `now` is the session clock. Unlike the reference, a
    /// non-positive rate is off (not every frame), and a clock stepping backwards re-anchors.
    pub fn should_sample_at(&mut self, speaking: bool, now: Ts) -> bool {
        let s = &mut self.state;
        let target_fps = if speaking { s.speaking_fps } else { s.silent_fps };
        if target_fps <= 0.0 || target_fps.is_nan() {
            return false;
        }
        // A rate too small to express as a `Duration` is an interval longer than any session.
        let min_interval = Duration::try_from_secs_f64(1.0 / target_fps).unwrap_or(Duration::MAX);
        match s.last_sampled {
            // The first frame always passes.
            None => {
                s.last_sampled = Some(now);
                true
            }
            Some(last) => {
                let stepped_back = now < last;
                let due = now.saturating_duration_since(last) >= min_interval;
                if stepped_back || due {
                    s.last_sampled = Some(now);
                }
                stepped_back || due
            }
        }
    }
}
