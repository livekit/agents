//! Measured provider timing: deciles (0th to 100th percentile, in ms unless named otherwise)
//! of live sessions. The fakes draw their delays from these, so a run's
//! races happen at the timings production sees.

use crate::fakes::Measured;

/// Cartesia sonic-3.
pub mod tts {
    use super::Measured;

    pub const FIRST_AUDIO: Measured = [183, 190, 196, 200, 205, 214, 217, 227, 240, 250, 658];
    pub const MS_PER_WORD: Measured = [197, 233, 248, 258, 274, 295, 311, 348, 415, 462, 557];
}

/// OpenAI gpt-4.1-mini.
pub mod llm {
    use super::Measured;

    pub const LLM_FIRST_CHUNK: Measured = [372, 445, 479, 519, 547, 597, 694, 746, 769, 999, 3117];
    pub const LLM_CHUNK_GAP: Measured = [0, 0, 0, 2, 4, 6, 8, 11, 18, 27, 172];
    pub const LLM_CHARS_PER_CHUNK: Measured = [1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 18];
}

/// The OpenAI realtime API.
pub mod realtime {
    use super::Measured;

    pub const SPEECH_STARTED_LAG_SEMANTIC: Measured =
        [238, 278, 299, 316, 329, 335, 345, 385, 404, 451, 1327];
    pub const SPEECH_STARTED_LAG_SERVER: Measured =
        [262, 307, 319, 335, 365, 399, 1704, 1803, 2409, 2693, 2728];
    pub const SPEECH_STOPPED_LAG_SEMANTIC: Measured =
        [671, 729, 737, 759, 787, 828, 1168, 1421, 1668, 1853, 3450];
    pub const SPEECH_STOPPED_LAG_SERVER: Measured =
        [582, 614, 628, 632, 653, 686, 725, 1606, 3016, 3820, 5682];
    pub const CANCEL_AFTER_SPEECH_STARTED: Measured = [3, 8, 9, 13, 17, 24, 27, 29, 35, 39, 246];
    pub const CREATED_AFTER_STOPPED: Measured = [0, 4, 6, 6, 7, 7, 8, 9, 9, 10, 15];
    pub const CREATED_AFTER_CREATE: Measured =
        [104, 105, 107, 111, 118, 120, 129, 134, 134, 148, 345];
    pub const TRANSCRIPTION_AFTER_COMMITTED: Measured =
        [278, 340, 362, 383, 391, 408, 453, 513, 599, 683, 3248];
    pub const ITEM_AFTER_CREATED: Measured =
        [164, 214, 290, 349, 386, 419, 475, 536, 572, 723, 1463];
    pub const FIRST_AUDIO_AFTER_ITEM: Measured =
        [62, 209, 235, 248, 268, 305, 317, 329, 345, 401, 1482];
    pub const AUDIO_GAP: Measured = [0, 23, 53, 60, 65, 70, 74, 81, 104, 159, 1108];
    pub const DONE_AFTER_LAST_AUDIO: Measured = [0, 28, 30, 33, 36, 38, 42, 44, 73, 187, 568];
    pub const ERROR_AFTER_CLIENT: Measured = [83, 88, 89, 90, 91, 93, 96, 99, 100, 103, 110];
    pub const DONE_AFTER_CANCEL: Measured = [115, 120, 124, 129, 131, 133, 135, 138, 144, 150, 156];
    pub const AUDIO_MS_PER_CHAR: Measured = [27, 40, 48, 50, 52, 56, 58, 58, 60, 61, 75];
    /// Replies that called a function, of all measured.
    pub const FUNCTION_CALL_SHARE: (u64, u64) = (8, 94);
}

/// The recognisers: delays in ms, how each cuts a turn, and its quirks. A quantity a model
/// never showed is all zeros and never drawn.
pub mod stt {
    use super::Measured;

    pub struct SttModel {
        pub name: &'static str,
        pub(crate) first_after_onset: Measured,
        pub(crate) interim_gap: Measured,
        pub(crate) final_after_end: Measured,
        pub(crate) finals_per_utterance: Measured,
        pub(crate) preflights_per_utterance: Measured,
        pub(crate) final_after_preflight: Measured,
        /// Utterances it finalised twice, of all measured.
        pub(crate) duplicate_share: (u64, u64),
        pub(crate) duplicate_final_after: Measured,
        /// It opens an utterance over noise and closes it with an empty final.
        pub empty_finals: bool,
    }

    pub const STT_MODELS: [SttModel; 3] = [
        SttModel {
            name: "deepgram/nova-3",
            first_after_onset: [660, 761, 774, 782, 817, 829, 836, 1016, 1391, 1738, 1903],
            interim_gap: [802, 950, 968, 970, 972, 978, 984, 996, 1020, 1020, 1023],
            final_after_end: [57, 245, 270, 281, 306, 325, 332, 344, 353, 365, 546],
            finals_per_utterance: [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4],
            preflights_per_utterance: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            final_after_preflight: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            duplicate_share: (3, 28),
            duplicate_final_after: [82, 101, 120, 140, 159, 178, 185, 193, 200, 208, 215],
            empty_finals: true,
        },
        SttModel {
            name: "assemblyai/universal-streaming",
            first_after_onset: [765, 837, 880, 898, 922, 1023, 1032, 1133, 1186, 1572, 1823],
            interim_gap: [40, 46, 92, 103, 138, 180, 218, 269, 349, 554, 967],
            final_after_end: [357, 370, 376, 499, 826, 931, 945, 981, 1031, 1046, 1073],
            finals_per_utterance: [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
            preflights_per_utterance: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            final_after_preflight: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            duplicate_share: (5, 22),
            duplicate_final_after: [138, 139, 140, 149, 168, 187, 200, 213, 221, 225, 229],
            empty_finals: false,
        },
        SttModel {
            name: "deepgram/flux-general-en",
            first_after_onset: [425, 507, 588, 607, 626, 736, 845, 866, 886, 977, 1068],
            interim_gap: [747, 751, 755, 758, 762, 766, 799, 832, 864, 897, 930],
            final_after_end: [154, 184, 213, 239, 265, 386, 507, 734, 960, 983, 1006],
            finals_per_utterance: [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            preflights_per_utterance: [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4],
            final_after_preflight: [0, 0, 2, 6, 12, 19, 56, 121, 240, 312, 359],
            duplicate_share: (0, 7),
            duplicate_final_after: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            empty_finals: false,
        },
    ];
}
