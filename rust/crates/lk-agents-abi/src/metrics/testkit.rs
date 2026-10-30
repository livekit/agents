//! Zeroed records the tests fill in field by field.
//!
//! The reference's models are pydantic with per-field defaults, so a Python test writes
//! `LLMMetrics(duration=0.5, ...)` and lets the rest default. Rust has no such thing for
//! structs with required fields, and spelling fifteen zeros at every call site would bury the one
//! field a test is actually about -- which is precisely how a rollup test stops catching anything.

use lk_agents_abi::metrics::base::{EotInferenceMetrics, InterruptionMetrics, Metadata};

use crate::metrics::base::{
    AvatarMetrics, EouMetrics, LlmMetrics, RealtimeModelMetrics, SttMetrics, TtsMetrics, VadMetrics,
};

pub fn meta(provider: &str, model: &str) -> Option<Metadata> {
    Some(Metadata { model_name: Some(model.to_owned()), model_provider: Some(provider.to_owned()) })
}

pub fn llm() -> LlmMetrics {
    LlmMetrics {
        label: String::new(),
        request_id: String::new(),
        timestamp: 0.0,
        duration: 0.0,
        ttft: None,
        cancelled: false,
        tokens: Default::default(),
        tokens_per_second: 0.0,
        speech_id: None,
        metadata: None,
    }
}

pub fn stt() -> SttMetrics {
    SttMetrics {
        label: String::new(),
        request_id: String::new(),
        timestamp: 0.0,
        duration: 0.0,
        audio_duration: None,
        streamed: false,
        tokens: Default::default(),
        acquire_time: 0.0,
        connection_reused: false,
        metadata: None,
    }
}

pub fn tts() -> TtsMetrics {
    TtsMetrics {
        label: String::new(),
        request_id: String::new(),
        timestamp: 0.0,
        ttfb: None,
        duration: 0.0,
        audio_duration: 0.0,
        cancelled: false,
        characters_count: 0,
        tokens: Default::default(),
        streamed: false,
        acquire_time: 0.0,
        connection_reused: false,
        segment_id: None,
        speech_id: None,
        metadata: None,
    }
}

pub fn vad() -> VadMetrics {
    VadMetrics {
        label: String::new(),
        timestamp: 0.0,
        idle_time: 0.0,
        inference_duration_total: 0.0,
        inference_count: 0,
        metadata: None,
    }
}

pub fn eou() -> EouMetrics {
    EouMetrics {
        timestamp: 0.0,
        end_of_utterance_delay: Some(0.0),
        transcription_delay: Some(0.0),
        on_user_turn_completed_delay: 0.0,
        speech_id: None,
        metadata: None,
    }
}

pub fn realtime() -> RealtimeModelMetrics {
    RealtimeModelMetrics {
        label: String::new(),
        request_id: String::new(),
        timestamp: 0.0,
        duration: 0.0,
        connection_duration: None,
        ttft: None,
        cancelled: false,
        tokens: Default::default(),
        tokens_per_second: 0.0,
        acquire_time: 0.0,
        connection_reused: false,
        metadata: None,
    }
}

pub fn avatar() -> AvatarMetrics {
    AvatarMetrics {
        timestamp: 0.0,
        playback_latency: 0.0,
        session_started_time: None,
        avatar_joined_time: None,
        metadata: None,
    }
}

pub fn interruption() -> InterruptionMetrics {
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

pub fn eot() -> EotInferenceMetrics {
    EotInferenceMetrics {
        timestamp: 0.0,
        total_duration: 0.0,
        detection_delay: 0.0,
        prediction_duration: 0.0,
        num_requests: EotInferenceMetrics::DEFAULT_NUM_REQUESTS,
        metadata: None,
    }
}
