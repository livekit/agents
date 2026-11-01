//! RTC frame pacing. Time is supplied by the executor; this module owns no source or timer.

use lk_agents_abi::Ts;
use lk_agents_abi::voice::avatar::AvatarOptions;
use lk_agents_abi::voice::rtc::{TrackPublishOptions, TrackSource, media::VideoTrackOptions};
use std::collections::VecDeque;
use std::time::Duration;

pub struct MediaPlan {
    pub audio: TrackPublishOptions,
    pub video: VideoTrackOptions,
    pub capacity: usize,
    pub pacer: FramePacer,
}

impl MediaPlan {
    pub fn new(options: AvatarOptions, queue_size_ms: u32) -> Result<Self, &'static str> {
        if options.video_width == 0 || options.video_height == 0 {
            return Err("avatar video dimensions must be positive");
        }
        if options.audio_sample_rate == 0 || options.audio_channels == 0 {
            return Err("avatar audio rate and channels must be positive");
        }
        let pacer = FramePacer::new(options.video_fps)?;
        // Bound retained video even when the requested latency is shorter than a frame.
        let capacity = (options.video_fps * f64::from(queue_size_ms) / 1000.0)
            .floor()
            .max(1.0)
            .min(u32::MAX as f64) as usize;
        Ok(Self {
            audio: TrackPublishOptions {
                name: "avatar_audio".into(),
                source: TrackSource::Microphone,
                sample_rate: options.audio_sample_rate,
                num_channels: options.audio_channels,
                queue_size_ms,
            },
            video: VideoTrackOptions {
                name: "avatar_video".into(),
                source: TrackSource::Camera,
                width: options.video_width,
                height: options.video_height,
            },
            capacity,
            pacer,
        })
    }
}

pub struct FramePacer {
    interval: Duration,
    next: Option<Ts>,
    captures: VecDeque<Ts>,
}

impl FramePacer {
    pub fn new(fps: f64) -> Result<Self, &'static str> {
        if !fps.is_finite() || fps <= 0.0 {
            return Err("video_fps must be finite and positive");
        }
        let interval = Duration::try_from_secs_f64(1.0 / fps)
            .map_err(|_| "video_fps is outside the supported range")?;
        if interval.is_zero() {
            return Err("video_fps is outside the supported range");
        }
        Ok(Self { interval, next: None, captures: VecDeque::new() })
    }

    pub fn deadline(&mut self, now: Ts) -> Ts {
        let next = self.next.get_or_insert(now);
        if now.saturating_duration_since(*next) > Duration::from_millis(300) {
            *next = now;
        }
        *next
    }

    pub fn captured(&mut self, now: Ts) {
        self.next = Some(self.next.unwrap_or(now).saturating_add(self.interval));
        if self.captures.len() == 100 {
            self.captures.pop_front();
        }
        self.captures.push_back(now);
    }

    pub fn reset(&mut self) {
        self.next = None;
        self.captures.clear();
    }

    pub fn actual_fps(&self) -> f64 {
        let (Some(first), Some(last)) = (self.captures.front(), self.captures.back()) else {
            return 0.0;
        };
        let elapsed = last.saturating_duration_since(*first).as_secs_f64();
        if elapsed == 0.0 { 0.0 } else { (self.captures.len() - 1) as f64 / elapsed }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(ms: u64) -> Ts {
        Ts::from_nanos(ms * 1_000_000)
    }

    #[test]
    fn pacing_recovers_after_stall_without_a_burst() {
        let mut p = FramePacer::new(25.0).unwrap();
        assert_eq!(p.deadline(at(0)), at(0));
        p.captured(at(0));
        assert_eq!(p.deadline(at(1)), at(40));
        p.captured(at(40));
        assert_eq!(p.actual_fps(), 25.0);
        assert_eq!(p.deadline(at(1000)), at(1000));
        p.captured(at(1000));
        assert_eq!(p.deadline(at(1000)), at(1040));
        p.reset();
        assert_eq!(p.actual_fps(), 0.0);
        assert_eq!(p.deadline(at(1100)), at(1100));
    }

    #[test]
    fn invalid_formats_are_rejected_before_source_creation() {
        for fps in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MAX, f64::MIN_POSITIVE] {
            assert!(FramePacer::new(fps).is_err());
        }
        let options = AvatarOptions {
            video_width: 16,
            video_height: 16,
            video_fps: 25.0,
            audio_sample_rate: 16000,
            audio_channels: 1,
        };
        assert_eq!(MediaPlan::new(options.clone(), 1).unwrap().capacity, 1);
        assert_eq!(MediaPlan::new(options, 100).unwrap().capacity, 2);
    }
}
