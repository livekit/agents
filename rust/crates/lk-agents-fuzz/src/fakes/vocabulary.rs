//! What the fakes can say, against what the core understands.
//!
//! Each provider-facing type the core consumes is a `#[derive(Vocabulary)]` enum, and each table
//! here implements `Covers` for every one of its variants: `Covered` with what produces it, or
//! `Skipped` with why no fake does. A variant the core gains is a compile error here until
//! someone decides how the fakes produce it (TEST_STRATEGY.md section 5, for provider events as
//! for commands). `tests/conformance.rs` checks the realtime claims against what the simulated
//! server actually emits.
use lk_agents_abi::{
    llm::realtime::{assert_event_is_covered, event_coverage, event_vocabulary as event},
    stt::events::{assert_speech_event_type_is_covered, speech_event_type_vocabulary as stt},
    utils::vocabulary::{Coverage, Covers},
    vad::events::{assert_vad_event_type_is_covered, vad_event_type_vocabulary as vad},
};
use lk_agents_inference::realtime::wire::{
    assert_incoming_is_covered, incoming_coverage, incoming_vocabulary as wire,
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

/// The realtime wire the core decodes (`lk_agents_inference::realtime::wire::Incoming`), by the
/// server event type the simulated server sends for it.
pub struct RealtimeWire;
table! { RealtimeWire:
    wire::SpeechStarted => Covered("input_audio_buffer.speech_started");
    wire::SpeechStopped => Covered("input_audio_buffer.speech_stopped");
    wire::ResponseCreated => Covered("response.created");
    wire::OutputItemAdded => Covered("response.output_item.added");
    wire::ContentPartAdded => Covered("response.content_part.added");
    wire::ItemAdded => Covered("conversation.item.added");
    wire::ItemDeleted => Covered("conversation.item.deleted");
    wire::TranscriptionDelta => Covered("conversation.item.input_audio_transcription.delta");
    wire::TranscriptionCompleted => Covered("conversation.item.input_audio_transcription.completed");
    wire::TranscriptionFailed => Covered("conversation.item.input_audio_transcription.failed");
    wire::TranscriptionUpdated => Skipped("xAI only; the fake server speaks OpenAI's dialect");
    wire::TextDelta => Skipped("text output only; the native session always asks for audio (`output_modalities: [audio]`)");
    wire::AudioTranscriptDelta => Covered("response.output_audio_transcript.delta");
    wire::AudioDelta => Covered("response.output_audio.delta");
    wire::OutputItemDone => Covered("response.output_item.done");
    wire::ResponseDone => Covered("response.done");
    wire::Error => Covered("error");
    wire::IdleTimeoutTriggered => Skipped("xAI only; the fake server speaks OpenAI's dialect");
    wire::SessionCreated => Covered("session.created");
    wire::Unhandled => Covered("session.updated");
}

/// The realtime events the driver consumes (`lk_agents_abi::llm::realtime::Event`), as the native
/// session derives them from the simulated server's traffic.
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
    event::ChatCtxUpdated => Covered("conversation.item.added / deleted answering an update");
    event::Disconnected => Skipped("a host session's; the native session reconnects and reports a lost connection as Error");
    event::SessionReconnected => Covered("a connection the session opened again");
    event::SessionSwapped => Skipped("the fallback adapter's; the fakes run one native session");
    event::Metrics => Covered("response.done usage, transcription usage");
    event::ConnectionAcquired => Covered("a connection opened");
    event::Error => Covered("error, a failed connection");
    event::HostError => Skipped("a host session's wall-clock error");
}

/// Compiles only while every variant of every provider vocabulary is covered or skipped.
pub fn every_provider_event_is_producible() {
    assert_speech_event_type_is_covered::<Stt, &'static str>();
    assert_vad_event_type_is_covered::<Vad, &'static str>();
    assert_incoming_is_covered::<RealtimeWire, &'static str>();
    assert_event_is_covered::<RealtimeEvents, &'static str>();
}

/// The wire events the simulated server must be seen to send, by the server event type.
pub fn realtime_wire_claims() -> Vec<(&'static str, Coverage<&'static str>)> {
    incoming_coverage::<RealtimeWire, &'static str>()
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
        for (name, coverage) in realtime_wire_claims().into_iter().chain(realtime_event_claims()) {
            if let Skipped(why) = coverage {
                assert!(why.len() > 10, "{name}: {why}");
            }
        }
    }
}
