//! Where a stream's input queue dropped audio, so its transcripts stay on call time.

use lk_agents_abi::SpeechEvent;
use std::time::Duration;

/// The audio one attempt was handed and the gaps in it, on the stream's timeline: a timestamp at
/// or past a gap moves by that gap's length.
#[derive(Debug, Default)]
pub(crate) struct Gaps {
    /// The end of the audio handed so far.
    at: f64,
    /// Each gap's position and length, in seconds.
    gaps: Vec<(f64, f64)>,
}

impl Gaps {
    /// An attempt whose audio starts at `start` on the stream's timeline.
    pub(crate) fn starting(start: f64) -> Self {
        Gaps { at: start, gaps: Vec::new() }
    }

    pub(crate) fn audio(&mut self, duration: Duration) {
        self.at += duration.as_secs_f64();
    }

    pub(crate) fn gap(&mut self, duration: Duration) {
        self.gaps.push((self.at, duration.as_secs_f64()));
    }

    pub(crate) fn shift(&self, mut event: SpeechEvent) -> SpeechEvent {
        if self.gaps.is_empty() {
            return event;
        }
        let shift =
            |t: f64| t + self.gaps.iter().filter(|(at, _)| t >= *at).map(|(_, d)| d).sum::<f64>();
        for alt in &mut event.alternatives {
            alt.start_time = shift(alt.start_time);
            alt.end_time = shift(alt.end_time);
            for word in alt.words.iter_mut().flatten() {
                word.start_time = word.start_time.map(shift);
                word.end_time = word.end_time.map(shift);
            }
        }
        event
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::{SpeechData, SpeechEventType};

    // Catches: a gap shifting the transcripts of audio heard before it, or not the ones after.
    #[test]
    fn only_timestamps_past_a_gap_move_by_it() {
        let mut gaps = Gaps::starting(1.0);
        gaps.audio(Duration::from_secs(2));
        gaps.gap(Duration::from_secs(5));
        gaps.audio(Duration::from_secs(1));
        gaps.gap(Duration::from_secs(1));
        let mut event = SpeechEvent::new(SpeechEventType::FinalTranscript);
        for (start, end) in [(2.5, 3.5), (3.0, 3.5), (4.0, 4.5)] {
            event.alternatives.push(SpeechData {
                start_time: start,
                end_time: end,
                ..SpeechData::new("en", "hi")
            });
        }
        let times: Vec<_> =
            gaps.shift(event).alternatives.iter().map(|a| (a.start_time, a.end_time)).collect();
        assert_eq!(times, [(2.5, 8.5), (8.0, 8.5), (10.0, 10.5)]);
    }
}
