//! Shared console audio accounting and microphone projection. The CLI acknowledges the last
//! flush once it has played it and drops the acknowledgement on a clear, so its unkeyed receipt
//! settles the flushed segment; a receipt before the flush belongs to a segment already cleared.

use super::proto::agent_session_message::{Message, console_io};
use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::io::{IoError, PlaybackFinished, PlaybackSegment};
use std::time::Duration;

/// The rate the CLI captures and plays at.
pub const CONSOLE_SAMPLE_RATE: u32 = 48_000;

/// One admitted local operation, the audio captured into it and whether it was flushed.
#[derive(Debug, Default)]
pub struct ConsolePlayback {
    segment: Option<PlaybackSegment>,
    captured: Duration,
    flushed: bool,
}

impl ConsolePlayback {
    /// The message for a captured frame and whether it opens this admitted operation.
    /// A different operation cannot borrow the previous one's pending physical buffer.
    pub fn capture(
        &mut self,
        frame: &AudioFrame,
        segment: PlaybackSegment,
    ) -> Result<(Message, bool), IoError> {
        if self.segment.is_some_and(|pending| pending != segment) {
            return Err(IoError::PlaybackSegmentUnavailable);
        }
        let opened = self.segment.is_none();
        self.segment = Some(segment);
        self.captured += Duration::from_secs_f64(frame.duration());
        let message = Message::AudioOutput(console_io::AudioFrame {
            data: frame.data.to_vec(),
            sample_rate: frame.sample_rate,
            num_channels: u32::from(frame.num_channels),
            samples_per_channel: frame.samples_per_channel,
        });
        Ok((message, opened))
    }

    pub fn flush(&mut self) {
        self.flushed = self.segment.is_some();
    }

    /// The CLI's receipt: the flushed segment played to its end.
    pub fn finished(&mut self) -> Option<(PlaybackSegment, PlaybackFinished)> {
        if !self.flushed {
            return None;
        }
        let played = self.captured;
        self.settle(played, false)
    }

    /// Local cleanup settles its own operation without inventing heard audio from elapsed time.
    pub fn clear(&mut self) -> Option<(PlaybackSegment, PlaybackFinished)> {
        self.settle(Duration::ZERO, true)
    }

    fn settle(
        &mut self,
        playback_position: Duration,
        interrupted: bool,
    ) -> Option<(PlaybackSegment, PlaybackFinished)> {
        let segment = self.segment.take()?;
        self.captured = Duration::ZERO;
        self.flushed = false;
        Some((
            segment,
            PlaybackFinished { playback_position, interrupted, synchronized_transcript: None },
        ))
    }
}

/// A microphone frame from the CLI, as the session's audio input reads it.
pub fn microphone_frame(frame: &console_io::AudioFrame) -> AudioFrame {
    AudioFrame::new(
        frame.data.clone(),
        frame.sample_rate,
        frame.num_channels as u16,
        frame.samples_per_channel,
    )
}
