//! What the fakes can say, against what the core understands.
//!
//! Each provider-facing type the core consumes is a `#[derive(Vocabulary)]` enum, and each table
//! here implements `Covers` for every one of its variants: `Covered` with what produces it, or
//! `Skipped` with why no fake does. A variant the core gains is a compile error here until
//! someone decides how the fakes produce it (TEST_STRATEGY.md section 5, for provider events as
//! for commands).
use lk_agents_abi::{
    llm::realtime::{assert_event_is_covered, event_coverage, event_vocabulary as event},
    stt::events::{assert_speech_event_type_is_covered, speech_event_type_vocabulary as stt},
    utils::vocabulary::{Coverage, Covers},
    vad::events::{assert_vad_event_type_is_covered, vad_event_type_vocabulary as vad},
};

macro_rules! table {
    ($table:ident: $($vocab:ident :: $leaf:ident => $how:expr;)*) => {$(
        impl Covers<$vocab::$leaf> for $table {
            type With = &'static str;
            fn coverage() -> Coverage<&'static str> {
                $how
            }
        }
    )*};
}

use Coverage::{Covered, Skipped};

/// The recogniser's event kinds, by the input that makes each.
pub struct Stt;
table! { Stt:
    stt::StartOfSpeech => Covered("SttStart");
    stt::InterimTranscript => Covered("SttInterim");
    stt::PreflightTranscript => Covered("SttPreflight");
    stt::FinalTranscript => Covered("SttFinal");
    stt::RecognitionUsage => Covered("SttUsage");
    stt::EndOfSpeech => Covered("SttEnd");
}

/// The detector's event kinds.
pub struct Vad;
table! { Vad:
    vad::StartOfSpeech => Covered("VadStart");
    vad::InferenceDone => Covered("VadInference");
    vad::EndOfSpeech => Covered("VadEnd");
}

/// The realtime events the driver consumes (`lk_agents_abi::llm::realtime::Event`), as the
/// realtime fake reports them.
pub struct RealtimeEvents;
table! { RealtimeEvents:
    event::InputSpeechStarted => Covered("input_audio_buffer.speech_started");
    event::InputSpeechStopped => Covered("input_audio_buffer.speech_stopped");
    event::InputTranscription => Covered("conversation.item.input_audio_transcription.delta/completed");
    event::GenerationCreated => Covered("response.created");
    event::GenerateReplyFailed => Covered("a response.create refused or never answered within its timeout");
    event::ServerGeneration => Skipped("a host session's; the native session reports a server response as GenerationCreated with no req_id");
    event::MessageStarted => Covered("response.output_item.added (message)");
    event::TextDelta => Covered("response.output_audio_transcript.delta");
    event::AudioDelta => Covered("response.output_audio.delta");
    event::MessageModalities => Covered("response.content_part.added");
    event::MessageDone => Covered("response.output_item.done / response.done");
    event::FunctionCall => Covered("response.output_item.done (function_call)");
    event::GenerationDone => Covered("response.done");
    event::RemoteItemAdded => Covered("conversation.item.added");
    event::SessionReconnected => Covered("a connection the session opened again");
    event::SessionSwapped => Covered("a fallback setup's FatalError: the model swaps to a fresh one");
    event::Metrics => Covered("response.done usage, transcription usage");
    event::Error => Covered("error, a failed connection");
    event::HostError => Skipped("a host session's wall-clock error");
    event::TranscriptAttribution => Covered("a duplex model's last words arriving after its utterance ended");
}

/// Compiles only while every variant of every provider vocabulary is covered or skipped.
pub fn every_provider_event_is_producible() {
    assert_speech_event_type_is_covered::<Stt, &'static str>();
    assert_vad_event_type_is_covered::<Vad, &'static str>();
    assert_event_is_covered::<RealtimeEvents, &'static str>();
}

/// The driver-facing realtime events, covered or skipped.
pub fn realtime_event_claims() -> Vec<(&'static str, Coverage<&'static str>)> {
    event_coverage::<RealtimeEvents, &'static str>()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a provider event the core consumes that no fake can produce (a new variant fails to
    // compile before this runs; a skip without a reason is caught here).
    #[test]
    fn every_skip_says_why() {
        every_provider_event_is_producible();
        for (name, coverage) in realtime_event_claims() {
            if let Skipped(why) = coverage {
                assert!(why.len() > 10, "{name}: {why}");
            }
        }
    }
}
