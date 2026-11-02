//! Queue-backed audio admission, cut accounting and playback report correlation.

use lk_agents_abi::AudioFrame;
use lk_agents_abi::utils::video::AudioReceiverFrame;
use lk_agents_abi::voice::avatar::AudioReceiverObservation;
use lk_agents_abi::voice::avatar_queue::{QueueClosed, QueueEffects, QueueFinished};
use lk_agents_abi::voice::io::{AvatarPlaybackSegment, PlaybackSegment};
use std::collections::VecDeque;

struct Segment {
    key: PlaybackSegment,
    yielded: bool,
    started: bool,
    discarded: bool,
}
struct Item {
    key: PlaybackSegment,
    epoch: u64,
    frame: AudioReceiverFrame,
}

pub struct Queue {
    wait_start: bool,
    segments: VecDeque<Segment>,
    items: VecDeque<Item>,
    capture: Option<(PlaybackSegment, u64)>,
    clear_requested: bool,
    closed: bool,
}

impl Queue {
    pub fn new(wait_start: bool) -> Self {
        Self {
            wait_start,
            segments: VecDeque::new(),
            items: VecDeque::new(),
            capture: None,
            clear_requested: true,
            closed: false,
        }
    }
    pub fn check_open(&self) -> Result<(), QueueClosed> {
        if self.closed { Err(QueueClosed) } else { Ok(()) }
    }
    pub fn capture(
        &mut self,
        frame: AudioFrame,
        key: PlaybackSegment,
        epoch: u64,
    ) -> Result<QueueEffects, QueueClosed> {
        self.check_open()?;
        let mut effects = QueueEffects::default();
        self.clear_requested = false;
        let (key, epoch) = *self.capture.get_or_insert_with(|| {
            self.segments.push_back(Segment {
                key,
                yielded: false,
                started: false,
                discarded: false,
            });
            if !self.wait_start {
                effects.started.push(key);
            }
            (key, epoch)
        });
        self.items.push_back(Item { key, epoch, frame: AudioReceiverFrame::Audio(frame) });
        Ok(effects)
    }
    pub fn flush(&mut self) -> Result<(), QueueClosed> {
        if let Some((key, epoch)) = self.capture.take() {
            self.check_open()?;
            self.items.push_back(Item { key, epoch, frame: AudioReceiverFrame::SegmentEnd });
        }
        Ok(())
    }
    pub fn clear(&mut self) -> QueueEffects {
        if self.closed {
            return QueueEffects::default();
        }
        self.capture = None;
        self.items.clear();
        for segment in &mut self.segments {
            if !segment.yielded {
                segment.discarded = true;
            }
        }
        let mut effects =
            QueueEffects { flushed: true, clear: !self.clear_requested, ..Default::default() };
        self.clear_requested = true;
        self.finish_discarded(&mut effects);
        effects
    }
    fn finish_discarded(&mut self, effects: &mut QueueEffects) {
        // Unread segments cannot receive a consumer acknowledgement. Settle them only after
        // older delivered segments have received theirs, preserving tracker order.
        while self.segments.front().is_some_and(|s| s.discarded) {
            let segment = self.segments.pop_front().expect("checked front");
            effects.finished.push(QueueFinished {
                segment: segment.key,
                playback_position: 0.0,
                interrupted: true,
            });
        }
    }
    pub fn finished(
        &mut self,
        position: f64,
        interrupted: bool,
        key: Option<AvatarPlaybackSegment>,
    ) -> QueueEffects {
        let mut effects = QueueEffects::default();
        let Some(oldest) = self.segments.front() else {
            return effects;
        };
        if !oldest.yielded || key.is_some_and(|key| key.playback != oldest.key) {
            return effects;
        }
        let segment = self.segments.pop_front().expect("checked front");
        effects.finished.push(QueueFinished {
            segment: segment.key,
            playback_position: position,
            interrupted,
        });
        self.finish_discarded(&mut effects);
        effects
    }
    pub fn started(&mut self, key: Option<AvatarPlaybackSegment>) -> QueueEffects {
        let mut effects = QueueEffects::default();
        if !self.wait_start {
            return effects;
        }
        if let Some(key) = key {
            effects.started.push(key.playback);
            return effects;
        }
        if let Some(queued) = self.segments.iter_mut().find(|s| s.yielded && !s.started) {
            queued.started = true;
            effects.started.push(queued.key);
        }
        effects
    }
    /// None means wait; a closed queue returns its terminal observation on every read.
    pub fn next_audio(&mut self) -> Option<AudioReceiverObservation> {
        if let Some(item) = self.items.pop_front() {
            let starts = if matches!(item.frame, AudioReceiverFrame::Audio(_)) {
                if let Some(segment) = self.segments.iter_mut().find(|s| s.key == item.key) {
                    !std::mem::replace(&mut segment.yielded, true)
                } else {
                    false
                }
            } else {
                false
            };
            return Some((
                item.epoch,
                starts,
                Some(AvatarPlaybackSegment { playback: item.key, stream_id: None }),
                Some(item.frame),
            ));
        }
        self.closed.then_some((0, false, None, None))
    }
    pub fn close(&mut self) -> QueueEffects {
        if self.closed {
            return QueueEffects::default();
        }
        let mut effects = self.clear();
        self.closed = true;
        self.segments.clear();
        effects.reset = true;
        effects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(segment: u64) -> PlaybackSegment {
        PlaybackSegment { attachment: 1, segment }
    }
    fn report(segment: u64) -> Option<AvatarPlaybackSegment> {
        Some(AvatarPlaybackSegment { playback: key(segment), stream_id: None })
    }
    fn frame() -> AudioFrame {
        AudioFrame::new(vec![0; 320], 16000, 1, 160)
    }

    #[test]
    fn delayed_acknowledgement_cannot_finish_the_replacement() {
        let mut queue = Queue::new(false);
        queue.capture(frame(), key(1), 0).unwrap();
        assert!(queue.next_audio().unwrap().1);
        assert!(queue.clear().clear);
        queue.capture(frame(), key(2), 1).unwrap();
        queue.flush().unwrap();
        queue.next_audio().unwrap();
        queue.next_audio().unwrap();
        let old = queue.finished(0.0, true, report(1));
        assert_eq!(old.finished.len(), 1);
        assert_eq!(old.finished[0].segment, key(1));
        assert!(queue.finished(0.0, true, report(1)).finished.is_empty());
        assert_eq!(queue.finished(0.01, false, report(2)).finished[0].segment, key(2));
    }

    #[test]
    fn cut_unread_segments_wait_for_older_delivered_acknowledgements() {
        let mut queue = Queue::new(false);
        queue.capture(frame(), key(1), 0).unwrap();
        queue.flush().unwrap();
        queue.next_audio();
        queue.next_audio();
        queue.capture(frame(), key(2), 0).unwrap();
        queue.flush().unwrap();
        queue.capture(frame(), key(3), 0).unwrap();
        assert!(queue.clear().finished.is_empty());
        let effects = queue.finished(0.02, true, report(1));
        assert_eq!(
            effects.finished.iter().map(|f| f.segment).collect::<Vec<_>>(),
            vec![key(1), key(2), key(3)]
        );
        assert!(effects.finished.iter().all(|f| f.interrupted));
        assert_eq!(effects.finished[1].playback_position, 0.0);
        assert!(queue.next_audio().is_none());
        assert!(!queue.clear().clear);
    }

    #[test]
    fn reader_stamps_epoch_and_start_once_and_close_discards_buffered_frames() {
        let mut queue = Queue::new(true);
        assert!(queue.capture(frame(), key(1), 5).unwrap().started.is_empty());
        queue.capture(frame(), key(1), 5).unwrap();
        let (epoch, starts, _, _) = queue.next_audio().unwrap();
        assert_eq!(epoch, 5);
        assert!(starts);
        assert_eq!(queue.started(None).started, vec![key(1)]);
        assert!(queue.started(None).started.is_empty());
        assert!(!queue.next_audio().unwrap().1);
        queue.capture(frame(), key(1), 5).unwrap();
        assert!(queue.close().reset);
        assert_eq!(queue.next_audio().unwrap(), (0, false, None, None));
        assert!(queue.capture(frame(), key(2), 6).is_err());
        assert!(!queue.close().reset);
    }
}
