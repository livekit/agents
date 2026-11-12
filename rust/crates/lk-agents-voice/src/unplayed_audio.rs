//! What the outputs still have to play, kept so a recorder attached while a segment plays records
//! the rest of it, where it plays. Audio heard before the attach is not the recording's.

use lk_agents_abi::voice::io::{PlaybackEvent, PlaybackSegment};
use lk_agents_abi::{AudioFrame, Ts};
use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

#[derive(Debug, Default)]
pub struct UnplayedAudio {
    /// Captured segments not yet finished, oldest first.
    segments: VecDeque<Unplayed>,
    /// Where each segment joined by the attached recorder began for it: its events are told
    /// from there.
    joined: BTreeMap<PlaybackSegment, Duration>,
}

#[derive(Debug)]
struct Unplayed {
    segment: PlaybackSegment,
    frames: Vec<AudioFrame>,
    started: Option<Ts>,
    /// The end of what the sink reported played; `None` while it reported no progress.
    heard: Option<Duration>,
}

/// What a newly attached recorder is told, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Replay {
    /// The joined segment plays from here on, at the position it joined.
    Started(Ts),
    Capture(AudioFrame),
}

impl UnplayedAudio {
    pub fn captured(&mut self, segment: PlaybackSegment, frame: &AudioFrame) {
        match self.segments.iter_mut().find(|u| u.segment == segment) {
            Some(unplayed) => unplayed.frames.push(frame.clone()),
            None => self.segments.push_back(Unplayed {
                segment,
                frames: vec![frame.clone()],
                started: None,
                heard: None,
            }),
        }
    }

    /// The segments kept, oldest first.
    pub fn segments(&self) -> Vec<PlaybackSegment> {
        self.segments.iter().map(|u| u.segment).collect()
    }

    /// Keeps only `attachment`'s segments its output still owes a report for.
    pub fn retain(&mut self, attachment: u64, pending: impl Fn(PlaybackSegment) -> bool) {
        self.segments.retain(|u| u.segment.attachment != attachment || pending(u.segment));
        self.joined.retain(|s, _| s.attachment != attachment || pending(*s));
    }

    /// The output's playback count was reset: what it held is gone.
    pub fn reset(&mut self, attachment: u64) {
        self.retain(attachment, |_| false);
    }

    pub fn observe(&mut self, ev: &PlaybackEvent) {
        let segment = match ev {
            PlaybackEvent::Started(e) => e.segment,
            PlaybackEvent::Progressed(e) => e.segment,
            PlaybackEvent::Finished { segment, .. } => *segment,
        };
        if let PlaybackEvent::Finished { .. } = ev {
            self.segments.retain(|u| u.segment != segment);
            self.joined.remove(&segment);
            return;
        }
        let Some(unplayed) = self.segments.iter_mut().find(|u| u.segment == segment) else {
            return;
        };
        match ev {
            PlaybackEvent::Started(e) => {
                unplayed.started.get_or_insert(e.created_at);
            }
            PlaybackEvent::Progressed(e) => {
                let end = e.offset + e.duration;
                unplayed.heard = Some(unplayed.heard.map_or(end, |heard| heard.max(end)));
            }
            PlaybackEvent::Finished { .. } => {}
        }
    }

    /// A recorder attaches at `now`: each unfinished segment's audio from its playout position on.
    /// A sink that reported no progress has played since it started.
    pub fn join(&mut self, now: Ts) -> Vec<Replay> {
        self.joined.clear();
        let mut replay = Vec::new();
        for unplayed in &self.segments {
            let captured: Duration =
                unplayed.frames.iter().map(|f| Duration::from_secs_f64(f.duration())).sum();
            let position = match (unplayed.started, unplayed.heard) {
                (_, Some(heard)) => heard,
                (Some(started), None) => now.saturating_duration_since(started),
                (None, None) => Duration::ZERO,
            }
            .min(captured);
            if position >= captured {
                continue;
            }
            self.joined.insert(unplayed.segment, position);
            if let Some(started) = unplayed.started {
                replay.push(Replay::Started(started.saturating_add(position)));
            }
            let mut skip = position;
            for frame in &unplayed.frames {
                let length = Duration::from_secs_f64(frame.duration());
                if skip >= length {
                    skip -= length;
                    continue;
                }
                replay.push(Replay::Capture(tail(frame, skip)));
                skip = Duration::ZERO;
            }
        }
        replay
    }

    /// `ev` as the recorder hears it: a joined segment's audio begins where it joined, and what
    /// played before is not told.
    pub fn told(&self, ev: &PlaybackEvent) -> Option<PlaybackEvent> {
        let segment = match ev {
            PlaybackEvent::Started(e) => e.segment,
            PlaybackEvent::Progressed(e) => e.segment,
            PlaybackEvent::Finished { segment, .. } => *segment,
        };
        let Some(&base) = self.joined.get(&segment) else { return Some(ev.clone()) };
        Some(match ev {
            PlaybackEvent::Started(e) => {
                let mut e = *e;
                e.created_at = e.created_at.saturating_add(base);
                PlaybackEvent::Started(e)
            }
            PlaybackEvent::Progressed(e) => {
                if e.offset + e.duration <= base {
                    return None;
                }
                let skip = base.saturating_sub(e.offset);
                let mut e = *e;
                e.started_at = e.started_at.saturating_add(skip);
                e.offset = e.offset + skip - base;
                e.duration -= skip;
                PlaybackEvent::Progressed(e)
            }
            PlaybackEvent::Finished { segment, event } => {
                let mut event = event.clone();
                event.playback_position = event.playback_position.saturating_sub(base);
                PlaybackEvent::Finished { segment: *segment, event }
            }
        })
    }
}

/// `frame` without its first `skip`, cut on a sample.
fn tail(frame: &AudioFrame, skip: Duration) -> AudioFrame {
    let samples = ((skip.as_secs_f64() * f64::from(frame.sample_rate)).round() as u32)
        .min(frame.samples_per_channel);
    let bytes = samples as usize * usize::from(frame.num_channels) * 2;
    AudioFrame {
        data: frame.data.slice(bytes.min(frame.data.len())..),
        samples_per_channel: frame.samples_per_channel - samples,
        ..frame.clone()
    }
}

#[cfg(test)]
#[path = "unplayed_audio_tests.rs"]
mod tests;
