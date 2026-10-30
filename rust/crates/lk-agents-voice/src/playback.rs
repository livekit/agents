//! The segment bookkeeping of one `io.AudioOutput` (`_playback_segments_count`,
//! `_playback_finished_count`, `_last_playback_ev`): which keyed segment is capturing, which owe a
//! report, how much of the open one was heard, and when a `wait_for_playout` is satisfied.

use lk_agents_abi::voice::io::{PlaybackFinished, PlaybackSegment};
use std::collections::VecDeque;
use std::time::Duration;

#[derive(Debug, Default)]
pub struct PlaybackSegments {
    /// Between the first captured frame of a segment and its flush.
    capturing: bool,
    selected: Option<PlaybackSegment>,
    /// Captured segments owing their finish, oldest first: only the front may report.
    pending: VecDeque<PlaybackSegment>,
    /// The highest key begun: a begun key is never reused.
    highest_segment: u64,
    /// The tracker's own keys (attachment 0) for audio captured with none begun, apart from the
    /// begun ones so they never take a key the session will hand out.
    unkeyed_segments: u64,
    /// Counts for waits latched at entry; keyed reports additionally enforce FIFO completion.
    segments: u64,
    finished: u64,
    last: PlaybackFinished,
    /// Bumped by [`reset`](Self::reset); a wait latches it so a reset releases it.
    epoch: u64,
    progressed: Duration,
    /// Audio captured into the open segment that no clear has dropped yet.
    uncleared: Duration,
}

/// What a `wait_for_playout` latched at entry: every segment captured so far, in this epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayoutWait {
    target: u64,
    epoch: u64,
}

impl PlaybackSegments {
    /// Select the supplied correlation for the next captured segment. Already admitted or
    /// still-capturing work cannot be replaced by a second begin.
    pub fn begin(&mut self, segment: PlaybackSegment) -> bool {
        if self.selected == Some(segment) {
            return true;
        }
        if self.capturing || segment.segment <= self.highest_segment {
            return false;
        }
        self.highest_segment = segment.segment;
        self.selected = Some(segment);
        true
    }

    pub fn is_pending(&self, segment: PlaybackSegment) -> bool {
        self.pending.contains(&segment)
    }

    /// Only the oldest segment owing a report may report.
    pub fn accepts(&self, segment: PlaybackSegment) -> bool {
        self.pending.front() == Some(&segment)
    }

    /// The key to latch before starting an external operation; never look it up on completion.
    pub fn current(&self) -> Option<PlaybackSegment> {
        self.selected
    }

    /// `duration` of audio captured: the first frame after a flush opens a segment.
    pub fn captured(&mut self, duration: Duration) -> PlaybackSegment {
        if !self.capturing {
            self.capturing = true;
            self.segments += 1;
            let segment = self.selected.unwrap_or_else(|| {
                self.unkeyed_segments += 1;
                PlaybackSegment { attachment: 0, segment: self.unkeyed_segments }
            });
            self.selected = Some(segment);
            self.pending.push_back(segment);
            // A new segment: what was heard of the previous one is not this one's position.
            self.progressed = Duration::ZERO;
            self.uncleared = Duration::ZERO;
        }
        self.uncleared += duration;
        self.selected.expect("capture opens a segment")
    }

    /// The audio captured into the current segment since it opened or since the last take: what
    /// a sink's clear still drops. Zero when a clear already dropped everything captured.
    pub fn take_uncleared(&mut self) -> Duration {
        std::mem::take(&mut self.uncleared)
    }

    /// The segment is closed; its playout is still owed.
    pub fn flush(&mut self) {
        self.capturing = false;
        self.selected = None;
    }

    /// `_reset_playback_count`, plus: a parked wait is released with `interrupted` at what the
    /// open segment played (never the previous segment's finish), rather than left waiting.
    pub fn reset(&mut self) {
        self.last = PlaybackFinished {
            playback_position: self.position(),
            interrupted: true,
            synchronized_transcript: None,
        };
        self.finished = self.segments;
        self.flush();
        self.pending.clear();
        self.epoch += 1;
        self.progressed = Duration::ZERO;
    }

    /// A reset below is told above, not repeated: each node's count is its own. The segment this
    /// node had open is over, though, once the sink reported it cut: the next frame opens a
    /// segment of its own.
    pub fn reset_below(&mut self) {
        if self.finished == self.segments {
            self.flush();
        }
    }

    /// The playhead per segment: what the sink reported played of the open segment, zero when
    /// none is open. Never the previous segment's finish (a cut speech would truncate at it).
    pub fn position(&self) -> Duration {
        if self.finished < self.segments { self.progressed } else { Duration::ZERO }
    }

    pub fn progressed(&mut self, segment: PlaybackSegment, duration: Duration) -> bool {
        if !self.accepts(segment) {
            return false;
        }
        self.progressed += duration;
        true
    }

    /// Only the admitted oldest segment can finish; duplicate, reset and replaced attachment
    /// reports are refused before anything changes.
    pub fn finished(&mut self, segment: PlaybackSegment, ev: PlaybackFinished) -> bool {
        if !self.accepts(segment) {
            return false;
        }
        self.pending.pop_front();
        self.finished += 1;
        self.last = ev;
        self.progressed = Duration::ZERO;
        true
    }

    /// Latched at entry, as the reference does, or a wait never returns while the agent talks.
    pub fn wait(&self) -> PlayoutWait {
        PlayoutWait { target: self.segments, epoch: self.epoch }
    }

    /// The last event once `wait` is satisfied: its segments finished, or a reset released it.
    pub fn settled(&self, wait: PlayoutWait) -> Option<&PlaybackFinished> {
        (self.epoch != wait.epoch || self.finished >= wait.target).then_some(&self.last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEARD: Duration = Duration::from_millis(20);

    fn finished() -> PlaybackFinished {
        PlaybackFinished { playback_position: HEARD, ..Default::default() }
    }

    #[test]
    fn a_delayed_old_finish_cannot_complete_a_new_segment_on_the_same_output() {
        let mut s = PlaybackSegments::default();
        let old = PlaybackSegment { attachment: 4, segment: 1 };
        let new = PlaybackSegment { attachment: 5, segment: 2 };
        assert!(s.begin(old));
        s.captured(HEARD);
        s.flush();
        assert!(s.finished(old, finished()));
        assert!(!s.begin(old), "a completed key cannot be reused");
        assert!(s.begin(new));
        s.captured(HEARD);
        s.flush();
        assert!(!s.finished(old, finished()));
        assert!(!s.accepts(old));
        assert!(!s.progressed(old, HEARD));
        assert!(s.finished(new, finished()));
    }

    #[test]
    fn reset_rejects_reports_from_the_previous_attachment_without_advancing_new_work() {
        let mut s = PlaybackSegments::default();
        let old = PlaybackSegment { attachment: 4, segment: 1 };
        let new = PlaybackSegment { attachment: 5, segment: 2 };
        s.begin(old);
        s.captured(Duration::ZERO);
        s.reset();
        s.begin(new);
        s.captured(Duration::ZERO);
        assert!(!s.finished(old, PlaybackFinished::default()));
        assert!(s.finished(new, PlaybackFinished::default()));
    }

    /// A reset mid-capture leaves the rest of the audio a segment the session never keyed; its
    /// number must not take the key the session hands the next speech (seed 477 refused that
    /// speech's output after its words were forwarded).
    #[test]
    fn audio_captured_after_a_reset_does_not_take_the_next_speechs_key() {
        let mut s = PlaybackSegments::default();
        assert!(s.begin(PlaybackSegment { attachment: 4, segment: 1 }));
        s.captured(Duration::ZERO);
        s.reset();
        s.captured(Duration::ZERO);
        s.flush();
        assert!(s.begin(PlaybackSegment { attachment: 4, segment: 2 }));
    }

    #[test]
    fn reset_releases_a_wait_at_the_open_segments_heard_position() {
        let mut s = PlaybackSegments::default();
        let segment = s.captured(Duration::ZERO);
        s.progressed(segment, HEARD);
        let wait = s.wait();
        assert!(s.settled(wait).is_none());
        s.reset();
        let released = s.settled(wait).expect("reset must release a parked wait");
        assert!(released.interrupted);
        assert_eq!(released.playback_position, HEARD);
    }
}
