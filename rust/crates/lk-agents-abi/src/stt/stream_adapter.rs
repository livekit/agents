//! VAD-to-batch STT adaptation. Host callbacks execute the returned recognition request.
use super::{
    events::{SpeechEvent, SpeechEventType},
    stt::SttCapabilities,
};
use crate::vad::events::VadEventType;

/// A recognize-only STT behind `stt.StreamAdapter`: streaming, no interim results.
#[crate::host(export(name = "stt_adapter_capabilities", pid = false))]
pub fn capabilities(inner: SttCapabilities) -> SttCapabilities {
    SttCapabilities {
        streaming: true,
        interim_results: false,
        diarization: false,
        keyterms: inner.keyterms,
        chat_context: inner.chat_context,
        ..SttCapabilities::default()
    }
}

/// The speech event a VAD event stands for, and whether it closes an utterance to recognize.
#[crate::host(export(name = "stt_adapter_vad", pid = false))]
pub fn vad_event(kind: VadEventType) -> (Option<SpeechEventType>, bool) {
    match kind {
        VadEventType::StartOfSpeech => (Some(SpeechEventType::StartOfSpeech), false),
        VadEventType::EndOfSpeech => (Some(SpeechEventType::EndOfSpeech), true),
        VadEventType::InferenceDone => (None, false),
    }
}

/// The final transcript a batch recognition becomes, or `None` for an empty one.
#[crate::host(export(name = "stt_adapter_recognized", pid = false))]
pub fn recognized(event: SpeechEvent) -> Option<SpeechEvent> {
    let best = event.alternatives.into_iter().next()?;
    if best.text.is_empty() {
        return None;
    }
    let mut final_event = SpeechEvent::new(SpeechEventType::FinalTranscript);
    final_event.alternatives.push(best);
    Some(final_event)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recognize_only_at_end_and_discard_empty_alternatives() {
        assert_eq!(
            vad_event(VadEventType::EndOfSpeech),
            (Some(SpeechEventType::EndOfSpeech), true)
        );
        assert_eq!(vad_event(VadEventType::InferenceDone), (None, false));
        assert!(recognized(SpeechEvent::new(SpeechEventType::FinalTranscript)).is_none());
    }
}
