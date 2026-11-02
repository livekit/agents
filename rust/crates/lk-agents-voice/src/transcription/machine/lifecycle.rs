//! Transcript synchronization lifecycle decisions, with physical outcomes supplied as events.

pub use super::pacer::{TranscriptPacerOptions, TranscriptPacerWarning};
use super::synchronizer::{TranscriptSyncInput, TranscriptSyncSegments};
use crate::transcription::effects::TranscriptPacerEffect;
use crate::transcription::effects::TranscriptSyncEffect;
use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::{TranscriptChunk, Ts};
use std::time::Duration;

pub enum TranscriptOutput {
    Word(TranscriptChunk),
    Flush,
    Detached,
}

pub enum TranscriptSyncEvent {
    /// Count the frame before capture: the sink may report playback inside its capture call.
    AudioCaptureStarted {
        secs: f64,
    },
    AudioEnded,
    AudioCleared,
    PlaybackStarted,
    PlaybackProgressed(Duration),
    PlaybackPaused,
    PlaybackFinished {
        position: Duration,
        interrupted: bool,
    },
    PlaybackReset,
    Capture {
        text: TranscriptChunk,
        aligned: bool,
    },
    FlushText,
    AttachText,
    AttachmentChanged {
        audio: Option<bool>,
        text: Option<bool>,
    },
    DetachText,
    TextRetired {
        admitted: bool,
    },
    Output(TranscriptOutput),
    Rotate,
    Close,
    WaitStarted(Ts),
    WaitFinished,
}

#[derive(Clone, Copy)]
enum ClosePhase {
    Open,
    Closing,
    Waiting,
    Closed,
}

/// One owner for the transcript gate and its segment pacing state. Both legs start detached: a
/// session attaches an output when a segment first activates on it, and the gate opens only with
/// both attached, so words with no synchronised audio pass straight through.
pub struct TranscriptSync {
    segments: TranscriptSyncSegments,
    audio_attached: bool,
    text_attached: bool,
    text_attachments: usize,
    capturing_text: bool,
    close: ClosePhase,
}

impl TranscriptSync {
    pub fn new(options: TranscriptPacerOptions) -> Self {
        Self {
            segments: TranscriptSyncSegments::new(options),
            audio_attached: false,
            text_attached: false,
            text_attachments: 0,
            capturing_text: false,
            close: ClosePhase::Open,
        }
    }

    pub fn handle(&mut self, event: TranscriptSyncEvent) -> Vec<TranscriptSyncEffect> {
        match event {
            TranscriptSyncEvent::AudioCaptureStarted { secs } => {
                self.step(TranscriptSyncInput::Audio { secs })
            }
            TranscriptSyncEvent::AudioEnded => self.step(TranscriptSyncInput::AudioEnd),
            TranscriptSyncEvent::AudioCleared => self.step(TranscriptSyncInput::Clear),
            TranscriptSyncEvent::PlaybackStarted => self.step(TranscriptSyncInput::Started),
            TranscriptSyncEvent::PlaybackProgressed(position) => {
                self.step(TranscriptSyncInput::Progressed(position))
            }
            TranscriptSyncEvent::PlaybackPaused => self.step(TranscriptSyncInput::Paused),
            TranscriptSyncEvent::PlaybackFinished { position, interrupted } => {
                self.step(TranscriptSyncInput::Finished { position, interrupted })
            }
            TranscriptSyncEvent::PlaybackReset => self.step(TranscriptSyncInput::Reset),
            TranscriptSyncEvent::Capture { text, aligned } => {
                if !self.segments.enabled() {
                    return vec![TranscriptSyncEffect::Capture { text, aligned }];
                }
                self.capturing_text = true;
                let (start_time, end_time) =
                    if aligned { (text.start_time, text.end_time) } else { (None, None) };
                self.step(TranscriptSyncInput::Text { raw: text.text, start_time, end_time })
            }
            TranscriptSyncEvent::FlushText => {
                if !self.segments.enabled() {
                    vec![TranscriptSyncEffect::FlushText]
                } else if std::mem::take(&mut self.capturing_text) {
                    self.step(TranscriptSyncInput::TextEnd)
                } else {
                    Vec::new()
                }
            }
            TranscriptSyncEvent::AttachText => {
                self.text_attachments += 1;
                vec![TranscriptSyncEffect::AttachText]
            }
            TranscriptSyncEvent::AttachmentChanged { audio, text } => {
                self.audio_attached = audio.unwrap_or(self.audio_attached);
                self.text_attached = text.unwrap_or(self.text_attached);
                self.step(TranscriptSyncInput::Enabled(self.audio_attached && self.text_attached))
            }
            TranscriptSyncEvent::DetachText => {
                let attachments = self.text_attachments;
                let mut effects = Vec::new();
                if attachments <= 1 {
                    self.text_attached = false;
                    effects.extend(self.step(TranscriptSyncInput::Enabled(false)));
                }
                // The final rotation still belongs to its admitted physical attachment.
                effects.push(TranscriptSyncEffect::RetireText { admitted: attachments > 0 });
                effects
            }
            TranscriptSyncEvent::TextRetired { admitted } => {
                if admitted {
                    self.text_attachments -= 1;
                    vec![TranscriptSyncEffect::Queue(TranscriptOutput::Detached)]
                } else {
                    Vec::new()
                }
            }
            TranscriptSyncEvent::Output(output) => {
                if matches!(output, TranscriptOutput::Word(_) | TranscriptOutput::Flush)
                    && self.text_attachments == 0
                {
                    Vec::new()
                } else {
                    vec![TranscriptSyncEffect::Queue(output)]
                }
            }
            TranscriptSyncEvent::Rotate => match self.close {
                ClosePhase::Open => self.step(TranscriptSyncInput::Rotate),
                _ => Vec::new(),
            },
            TranscriptSyncEvent::Close => {
                if !matches!(self.close, ClosePhase::Open) {
                    return Vec::new();
                }
                self.close = ClosePhase::Closing;
                let mut effects = if self.text_attachments > 0 {
                    self.step(TranscriptSyncInput::Rotate)
                } else {
                    Vec::new()
                };
                effects
                    .extend([TranscriptSyncEffect::CloseSender, TranscriptSyncEffect::BeginWait]);
                effects
            }
            TranscriptSyncEvent::WaitStarted(at) => {
                if !matches!(self.close, ClosePhase::Closing) {
                    return Vec::new();
                }
                self.close = ClosePhase::Waiting;
                vec![TranscriptSyncEffect::WaitUntil(at.saturating_add(HOST_TEARDOWN_TIMEOUT))]
            }
            TranscriptSyncEvent::WaitFinished => {
                self.close = ClosePhase::Closed;
                Vec::new()
            }
        }
    }

    fn step(&mut self, input: TranscriptSyncInput) -> Vec<TranscriptSyncEffect> {
        self.segments
            .step(input)
            .into_iter()
            .map(|effect| match effect {
                TranscriptPacerEffect::Forward { word, start_time, end_time } => {
                    TranscriptSyncEffect::PublishWord { word, start_time, end_time }
                }
                TranscriptPacerEffect::Flush => TranscriptSyncEffect::FlushTranscript,
                TranscriptPacerEffect::Heard(heard) => TranscriptSyncEffect::ReportHeard(heard),
                TranscriptPacerEffect::Warn(warning) => TranscriptSyncEffect::Warn(warning),
            })
            .collect()
    }
}
