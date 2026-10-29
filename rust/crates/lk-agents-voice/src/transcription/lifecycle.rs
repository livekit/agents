//! Transcript synchronization lifecycle decisions, with physical outcomes supplied as events.

use super::pacer::{PaceEffect, PacerOptions};
use super::synchronizer::{SyncInput, SyncSegments};
use lk_agents_abi::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::{TimedString, Ts};

pub enum TranscriptOutput {
    Word(TimedString),
    Flush,
    Detached,
}

pub enum SyncEvent {
    Input(SyncInput),
    Capture { text: TimedString, aligned: bool },
    FlushText,
    AttachText,
    AttachmentChanged { audio: Option<bool>, text: Option<bool> },
    DetachText,
    TextRetired { admitted: bool },
    Output(TranscriptOutput),
    Rotate,
    Close,
    WaitStarted(Ts),
    WaitFinished,
}

/// Physical operations and pacing effects, in their required order.
pub enum SyncEffect {
    Pace(PaceEffect),
    Capture { text: TimedString, aligned: bool },
    FlushText,
    AttachText,
    RetireText { admitted: bool },
    Queue(TranscriptOutput),
    CloseSender,
    BeginWait,
    WaitUntil(Ts),
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
pub struct SyncLifecycle {
    pub segments: SyncSegments,
    audio_attached: bool,
    text_attached: bool,
    text_attachments: usize,
    capturing_text: bool,
    close: ClosePhase,
}

impl SyncLifecycle {
    pub fn new(options: PacerOptions) -> Self {
        Self {
            segments: SyncSegments::new(options),
            audio_attached: false,
            text_attached: false,
            text_attachments: 0,
            capturing_text: false,
            close: ClosePhase::Open,
        }
    }

    pub fn handle(&mut self, event: SyncEvent) -> Vec<SyncEffect> {
        match event {
            SyncEvent::Input(input) => self.step(input),
            SyncEvent::Capture { text, aligned } => {
                if !self.segments.enabled() {
                    return vec![SyncEffect::Capture { text, aligned }];
                }
                self.capturing_text = true;
                let (start_time, end_time) =
                    if aligned { (text.start_time, text.end_time) } else { (None, None) };
                self.step(SyncInput::Text { raw: text.text, start_time, end_time })
            }
            SyncEvent::FlushText => {
                if !self.segments.enabled() {
                    vec![SyncEffect::FlushText]
                } else if std::mem::take(&mut self.capturing_text) {
                    self.step(SyncInput::TextEnd)
                } else {
                    Vec::new()
                }
            }
            SyncEvent::AttachText => {
                self.text_attachments += 1;
                vec![SyncEffect::AttachText]
            }
            SyncEvent::AttachmentChanged { audio, text } => {
                self.audio_attached = audio.unwrap_or(self.audio_attached);
                self.text_attached = text.unwrap_or(self.text_attached);
                self.step(SyncInput::Enabled(self.audio_attached && self.text_attached))
            }
            SyncEvent::DetachText => {
                let attachments = self.text_attachments;
                let mut effects = Vec::new();
                if attachments <= 1 {
                    self.text_attached = false;
                    effects.extend(self.step(SyncInput::Enabled(false)));
                }
                // The final rotation still belongs to its admitted physical attachment.
                effects.push(SyncEffect::RetireText { admitted: attachments > 0 });
                effects
            }
            SyncEvent::TextRetired { admitted } => {
                if admitted {
                    self.text_attachments -= 1;
                    vec![SyncEffect::Queue(TranscriptOutput::Detached)]
                } else {
                    Vec::new()
                }
            }
            SyncEvent::Output(output) => {
                if matches!(output, TranscriptOutput::Word(_) | TranscriptOutput::Flush)
                    && self.text_attachments == 0
                {
                    Vec::new()
                } else {
                    vec![SyncEffect::Queue(output)]
                }
            }
            SyncEvent::Rotate => match self.close {
                ClosePhase::Open => self.step(SyncInput::Rotate),
                _ => Vec::new(),
            },
            SyncEvent::Close => {
                if !matches!(self.close, ClosePhase::Open) {
                    return Vec::new();
                }
                self.close = ClosePhase::Closing;
                let mut effects = if self.text_attachments > 0 {
                    self.step(SyncInput::Rotate)
                } else {
                    Vec::new()
                };
                effects.extend([SyncEffect::CloseSender, SyncEffect::BeginWait]);
                effects
            }
            SyncEvent::WaitStarted(at) => {
                if !matches!(self.close, ClosePhase::Closing) {
                    return Vec::new();
                }
                self.close = ClosePhase::Waiting;
                vec![SyncEffect::WaitUntil(at.saturating_add(HOST_TEARDOWN_TIMEOUT))]
            }
            SyncEvent::WaitFinished => {
                self.close = ClosePhase::Closed;
                Vec::new()
            }
        }
    }

    fn step(&mut self, input: SyncInput) -> Vec<SyncEffect> {
        self.segments.step(input).into_iter().map(SyncEffect::Pace).collect()
    }
}
