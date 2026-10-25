//! `session.output` as the core decides with it: the outputs by id and their switches. The data
//! plane's copy (`SessionShared::output`) changes only by the effects these changes raise.

use lk_agents_abi::resources::{AudioOutputRef, TextOutputRef};
use lk_agents_abi::voice::io::AudioOutputCapabilities;

#[derive(Debug)]
pub(crate) struct Outputs {
    pub(crate) audio: Option<AudioOutputRef>,
    /// `output.audio_enabled`: the switch, which a speech freezes at its voice leg.
    pub(crate) audio_enabled: bool,
    pub(crate) transcription: Option<TextOutputRef>,
    /// `output.transcription_enabled`.
    pub(crate) transcription_enabled: bool,
}

/// No output, both switches on.
impl Default for Outputs {
    fn default() -> Self {
        Outputs::new(None, None)
    }
}

impl Outputs {
    pub(crate) fn new(audio: Option<AudioOutputRef>, transcription: Option<TextOutputRef>) -> Self {
        Outputs { audio, audio_enabled: true, transcription, transcription_enabled: true }
    }

    /// Audio plays: a sink, switched on.
    pub(crate) fn plays_audio(&self) -> bool {
        self.audio.is_some() && self.audio_enabled
    }

    /// Transcription shows: a sink, switched on.
    pub(crate) fn shows_transcription(&self) -> bool {
        self.transcription.is_some() && self.transcription_enabled
    }

    /// The pause facts: what the sink can do, and whether audio plays.
    pub(crate) fn pause_facts(&self) -> (AudioOutputCapabilities, bool) {
        let capabilities = self.audio.as_ref().map(|a| a.capabilities).unwrap_or_default();
        (capabilities, self.plays_audio())
    }
}
