//! `lk agent console`'s audio legs as the session accounts for them (`TcpAudioOutput`,
//! `TcpAudioInput`). The CLI plays what it is sent and acks each flushed segment once played, so a
//! segment's position is all it was sent, or, when a clear cuts it, what had time to play.

use super::proto::agent_session_message::{Message, console_io};
use lk_agents_abi::AudioFrame;
use lk_agents_abi::voice::io::PlaybackFinished;
use std::time::Duration;

/// The rate the CLI captures and plays at.
pub const CONSOLE_SAMPLE_RATE: u32 = 48_000;

/// The segment the CLI is playing. Times are the caller's wall clock, in Unix seconds.
#[derive(Debug, Default)]
pub struct ConsolePlayback {
    /// When the segment's first frame went out.
    started: Option<f64>,
    sent: Duration,
}

impl ConsolePlayback {
    /// `frame` goes to the CLI at `now`: its message, and whether it opened the segment (whose
    /// playback started at `now`).
    pub fn capture(&mut self, frame: &AudioFrame, now: f64) -> (Message, bool) {
        self.sent += Duration::from_secs_f64(frame.duration());
        let opened = self.started.is_none();
        self.started.get_or_insert(now);
        let message = Message::AudioOutput(console_io::AudioFrame {
            data: frame.data.to_vec(),
            sample_rate: frame.sample_rate,
            num_channels: u32::from(frame.num_channels),
            samples_per_channel: frame.samples_per_channel,
        });
        (message, opened)
    }

    /// The segment ends at `now`, played out (the CLI's ack) or cut by a clear; `None` when it
    /// held no audio.
    pub fn finish(&mut self, interrupted: bool, now: f64) -> Option<PlaybackFinished> {
        let started = self.started.take()?;
        let sent = std::mem::take(&mut self.sent);
        let played = Duration::try_from_secs_f64(now - started).unwrap_or_default();
        Some(PlaybackFinished {
            playback_position: if interrupted { played.min(sent) } else { sent },
            interrupted,
            synchronized_transcript: None,
        })
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
