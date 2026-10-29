//! Streaming synthesis decisions: feed ordering, completion and aligned captions.

use crate::transcription::Visible;
use lk_agents_abi::TimedString;
use lk_agents_abi::errors::ProviderError;

/// About five seconds of speculative 20 ms frames. Runtime backpressures the provider at this bound.
pub const MAX_BUFFERED_TTS_FRAMES: usize = 256;

/// The ordered operations caused by one text input.
pub enum TextEffect {
    Capture(Visible),
    Push(String),
    EndInput,
    FlushText,
}

/// Whether a subsequent source read is still needed.
pub enum Read {
    Provider,
    End,
    Failed(ProviderError),
}

pub struct StreamingTts {
    pub words: Visible,
    pub input_done: bool,
    ended: bool,
    send_error: Option<ProviderError>,
}

impl StreamingTts {
    pub fn new(words: Visible, input_done: bool) -> Self {
        Self { words, input_done, ended: false, send_error: None }
    }

    pub fn before_read(&mut self) -> Read {
        if self.ended {
            Read::End
        } else if let Some(error) = self.send_error.take() {
            Read::Failed(error)
        } else {
            Read::Provider
        }
    }

    /// The transcript delta precedes the same delta's synthesis input.
    pub fn text(&mut self, chunk: Option<Visible>) -> Vec<TextEffect> {
        match chunk {
            Some(chunk) => vec![TextEffect::Capture(chunk)],
            None => {
                self.input_done = true;
                vec![TextEffect::EndInput, TextEffect::FlushText]
            }
        }
    }

    /// Capturing a transcript delta completed; only now does the feed include its words.
    pub fn captured(&mut self, chunk: Visible) -> TextEffect {
        self.words.raw.push_str(&chunk.raw);
        self.words.text.push_str(&chunk.text);
        self.words.tags.extend(chunk.tags);
        TextEffect::Push(chunk.raw)
    }

    /// A rejected text push ends feeding. Its error follows the source read already in flight.
    pub fn push_failed(&mut self, error: ProviderError) {
        self.send_error = Some(error);
        self.input_done = true;
    }

    /// A rejected opening feed is reported before the first source read.
    pub fn opening_pushed(&mut self, error: Option<ProviderError>) -> bool {
        let accepted = error.is_none();
        self.send_error = error;
        self.input_done && accepted
    }

    pub fn source_ended(&mut self) {
        self.ended = true;
    }

    /// A pooled source can stay open after its final frame. End audio once input is complete.
    pub fn frame(&mut self, is_final: bool) -> bool {
        if is_final && self.input_done {
            self.ended = true;
            true
        } else {
            false
        }
    }
}

pub enum TranscriptEffect {
    CaptureTimed(TimedString),
    Capture(String),
    WarnUnaligned,
    Flush,
}

/// Stamped words own an aligned transcript after the first stamp. An unstamped segment falls back
/// to generated text. Closing flushes the transcript exactly once.
#[derive(Default)]
pub struct AlignedTranscript {
    pub words: String,
    pub stamped: bool,
    pub closed: bool,
}

impl AlignedTranscript {
    pub fn word(&mut self, word: &TimedString) -> TranscriptEffect {
        self.stamped = true;
        self.words.push_str(&word.text);
        TranscriptEffect::CaptureTimed(word.clone())
    }

    pub fn end(&mut self, fallback: &str) -> Vec<TranscriptEffect> {
        if std::mem::replace(&mut self.closed, true) {
            return Vec::new();
        }
        let mut effects = Vec::new();
        if !self.stamped {
            self.words.push_str(fallback);
            effects.push(TranscriptEffect::WarnUnaligned);
            effects.push(TranscriptEffect::Capture(fallback.to_owned()));
        }
        effects.push(TranscriptEffect::Flush);
        effects
    }
}
