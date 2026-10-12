//! Zeroed records the tests fill in field by field.
//!
//! The reference's models are pydantic with per-field defaults, so a Python test writes
//! `LLMMetrics(prompt_tokens=7, ...)` and lets the rest default. Rust has no such thing for
//! structs with required fields, and spelling fifteen zeros at every call site would bury the one
//! field a test is actually about -- which is precisely how a rollup test stops catching anything.

use lk_agents_abi::metrics::base::{EotInferenceMetrics, InterruptionMetrics, Metadata};

use crate::metrics::base::{
    AvatarMetrics, EouMetrics, InputTokenDetails, LlmMetrics, OutputTokenDetails,
    RealtimeModelMetrics, SttMetrics, TtsMetrics, VadMetrics,
};

pub(crate) fn meta(provider: &str, model: &str) -> Option<Metadata> {
    Some(Metadata { model_name: Some(model.to_owned()), model_provider: Some(provider.to_owned()) })
}

pub(crate) fn llm() -> LlmMetrics {
    LlmMetrics {
        label: String::new(),
        request_id: String::new(),
        timestamp: 0.0,
        duration: 0.0,
        ttft: 0.0,
        cancelled: false,
        completion_tokens: 0,
        prompt_tokens: 0,
        prompt_cached_tokens: 0,
        cache_creation_tokens: 0,
        total_tokens: 0,
        tokens_per_second: 0.0,
        speech_id: None,
        metadata: None,
    }
}

pub(crate) fn stt() -> SttMetrics {
    SttMetrics {
        label: String::new(),
        request_id: String::new(),
        timestamp: 0.0,
        duration: 0.0,
        audio_duration: 0.0,
        input_tokens: 0,
        output_tokens: 0,
        streamed: false,
        acquire_time: 0.0,
        connection_reused: false,
        metadata: None,
    }
}

pub(crate) fn tts() -> TtsMetrics {
    TtsMetrics {
        label: String::new(),
        request_id: String::new(),
        timestamp: 0.0,
        ttfb: 0.0,
        duration: 0.0,
        audio_duration: 0.0,
        cancelled: false,
        characters_count: 0,
        input_tokens: 0,
        output_tokens: 0,
        streamed: false,
        acquire_time: 0.0,
        connection_reused: false,
        segment_id: None,
        speech_id: None,
        metadata: None,
    }
}

pub(crate) fn vad() -> VadMetrics {
    VadMetrics {
        label: String::new(),
        timestamp: 0.0,
        idle_time: 0.0,
        inference_duration_total: 0.0,
        inference_count: 0,
        metadata: None,
    }
}

pub(crate) fn eou() -> EouMetrics {
    EouMetrics {
        timestamp: 0.0,
        end_of_utterance_delay: 0.0,
        transcription_delay: 0.0,
        on_user_turn_completed_delay: 0.0,
        speech_id: None,
        metadata: None,
    }
}

pub(crate) fn realtime() -> RealtimeModelMetrics {
    RealtimeModelMetrics {
        label: String::new(),
        request_id: String::new(),
        timestamp: 0.0,
        duration: 0.0,
        session_duration: 0.0,
        ttft: RealtimeModelMetrics::NO_TTFT,
        cancelled: false,
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        tokens_per_second: 0.0,
        input_token_details: InputTokenDetails::default(),
        output_token_details: OutputTokenDetails::default(),
        acquire_time: 0.0,
        connection_reused: false,
        metadata: None,
    }
}

pub(crate) fn avatar() -> AvatarMetrics {
    AvatarMetrics {
        timestamp: 0.0,
        playback_latency: 0.0,
        session_started_time: None,
        avatar_joined_time: None,
        metadata: None,
    }
}

pub(crate) fn interruption() -> InterruptionMetrics {
    InterruptionMetrics {
        timestamp: 0.0,
        total_duration: 0.0,
        prediction_duration: 0.0,
        detection_delay: 0.0,
        num_interruptions: 0,
        num_backchannels: 0,
        num_requests: 0,
        metadata: None,
    }
}

pub(crate) fn eot() -> EotInferenceMetrics {
    EotInferenceMetrics {
        timestamp: 0.0,
        total_duration: 0.0,
        detection_delay: 0.0,
        prediction_duration: 0.0,
        num_requests: EotInferenceMetrics::DEFAULT_NUM_REQUESTS,
        metadata: None,
    }
}
