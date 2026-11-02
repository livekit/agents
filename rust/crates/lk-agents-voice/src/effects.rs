//! Effect ownership overview for the voice crate.
//!
//! Each row names the producer, its effect contract, and the runtime executor. Indentation is
//! ownership, not a global effect-enum nesting scheme. Actual definitions live in category files.
//!
//! | Producer | Effect contract | Executor |
//! | --- | --- | --- |
//! | [`VoiceDriver`](crate::agent_session::machine::VoiceDriver) | [`VoiceEffect`] | VoiceRuntime |
//! | [`AudioRecognition`](crate::audio_recognition::AudioRecognition) | [`AudioRecognitionEffect`] | VoiceDriver |
//! | [`AudioInputRouter`](crate::data_plane::machine::AudioInputRouter) | [`AudioInputRouterEffect`] | voice frame routing |
//! | [`AudioPlayback`](crate::generation::machine::AudioPlayback) | [`AudioPlaybackEffect`] | generation playout |
//! | [`GenerationPreparation`](crate::generation::machine::GenerationPreparation) | [`GenerationPreparationEffect`] | generation startup |
//! | [`RealtimeGeneration`](crate::generation::machine::RealtimeGeneration) | [`RealtimeGenerationEffect`] | realtime generation |
//! | [`TranscriptSync`](crate::transcription::machine::TranscriptSync) | [`TranscriptSyncEffect`] | transcript synchronization |
//! | [`RoomTranscriptionWriter`](crate::room_io::machine::RoomTranscriptionWriter) | [`RoomTranscriptionEffect`] | room transcription publication |
//! | [`RunnerCore`](crate::avatar::machine::runner::RunnerCore) | [`AvatarEffect`] | avatar transport |
//! | AMD classification | [`AmdEffect`] | answering-machine detection |
//!
//! AudioPlayback's forwarding/receipt/finish/accounting effects and TranscriptSync's pacing
//! effects remain private. Other generation, IVR, keyterm, and workflow protocols are listed
//! below. RoomIo has child protocols; there is no synthetic RoomIoEffect wrapper here.

pub use crate::amd::effects::AmdEffect;
pub use crate::audio_recognition::effects::AudioRecognitionEffect;
pub use crate::avatar::effects::AvatarEffect;
pub use crate::generation::effects::{
    AlignedTranscriptEffect, AudioPlaybackEffect, GenerationPreparationEffect,
    RealtimeGenerationEffect, ReplyStartupEffect, StreamingTtsEffect, SynthesisLossEffect,
    TurnAccumulatorEffect,
};
pub use crate::ivr::effects::IvrEffect;
pub use crate::keyterm_detection::effects::KeytermEffect;
pub use crate::room_io::effects::RoomTranscriptionEffect;
pub use crate::transcription::effects::TranscriptSyncEffect;
pub use crate::workflows::effects::{TransferEffects, TransferToolEffect};

pub use crate::agent_session::effects::VoiceEffect;
pub use crate::data_plane::effects::AudioInputRouterEffect;
