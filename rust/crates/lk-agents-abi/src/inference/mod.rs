//! The inference gateway's typing vocabulary (`inference/{stt,tts,llm,avatar,vad}.py`). No Rust
//! reads it yet. The reference's comments stay `//`. The server-side fallback entries are in
//! [`fallback`].

pub mod fallback;

use serde_json::Value;
use std::collections::HashMap;

pub use gateway::*;

#[crate::host(group(kind = "typed_dict", typed_dict_from = "typing", total = "False", node))]
pub mod gateway {
    use super::*;

    #[crate::host(group(path = "livekit.agents.inference.stt"))]
    pub mod stt {
        use super::*;

        pub const DEEPGRAM_MODELS: &[&str] = &[
            "deepgram/nova-3",
            "deepgram/nova-3-medical",
            "deepgram/nova-2",
            "deepgram/nova-2-medical",
            "deepgram/nova-2-conversationalai",
            "deepgram/nova-2-phonecall",
        ];
        pub const DEEPGRAM_FLUX_MODELS: &[&str] = &[
            "deepgram/flux-general",
            "deepgram/flux-general-en",
            "deepgram/flux-general-multi",
        ];
        pub const CARTESIA_MODELS: &[&str] = &["cartesia/ink-whisper", "cartesia/ink-2"];
        #[host(name = "AssemblyAIModels")]
        pub const ASSEMBLY_AI_MODELS: &[&str] = &[
            "assemblyai/universal-streaming",
            "assemblyai/universal-streaming-multilingual",
            "assemblyai/u3-rt-pro",
            "assemblyai/universal-3-5-pro",
        ];
        pub const ELEVENLABS_MODELS: &[&str] = &["elevenlabs/scribe_v2_realtime"];
        pub const XAI_MODELS: &[&str] = &["xai/stt-1"];
        pub const SPEECHMATICS_MODELS: &[&str] =
            &["speechmatics/enhanced", "speechmatics/standard"];
        pub const INWORLD_MODELS: &[&str] = &["inworld/inworld-stt-1"];

        pub struct CartesiaOptions {
            pub min_volume: Option<f64>, // ink-whisper only; default: not specified
            pub max_silence_duration_secs: Option<f64>, // ink-whisper only; default: not specified
            // Turn-detection tuning for turn-detecting models (e.g. ink-2). The gateway
            // validates these against Cartesia's documented ranges.
            pub turn_start_threshold: Option<f64>, // range 0.5-0.9, default 0.8
            pub turn_eager_end_threshold: Option<f64>, // range 0.3-0.6, default 0.4
            pub turn_end_threshold: Option<f64>,   // range 0.05-0.5, default 0.2
            pub turn_end_timeout_ms: Option<i64>,  // range 640-11200, default 5600
            #[python(ty = "str | list[str]")]
            pub keyterm: Option<Value>, // up to 100 terms totaling 1200 characters
        }

        pub struct DeepgramOptions {
            pub filler_words: Option<bool>,    // default: True
            pub interim_results: Option<bool>, // default: True
            pub endpointing: Option<i64>,      // default: 25 (ms)
            pub punctuate: Option<bool>,       // default: True
            pub smart_format: Option<bool>,
            #[python(ty = "list[tuple[str, float]]")]
            pub keywords: Option<Value>,
            #[python(ty = "str | list[str]")]
            pub keyterm: Option<Value>,
            pub profanity_filter: Option<bool>,
            pub numerals: Option<bool>,
            pub mip_opt_out: Option<bool>, // default: False
            pub vad_events: Option<bool>,  // default: False
            pub diarize: Option<bool>,     // when True, enables speaker diarization (default off)
            pub dictation: Option<bool>,
            pub detect_language: Option<bool>,
            pub no_delay: Option<bool>, // default: True
            pub utterance_end: Option<bool>,
            #[python(ty = "str | list[str]")]
            pub redact: Option<Value>,
            #[python(ty = "str | list[str]")]
            pub replace: Option<Value>,
            #[python(ty = "str | list[str]")]
            pub search: Option<Value>,
            #[python(ty = "str | list[str]")]
            pub tag: Option<Value>,
            pub channels: Option<i64>,
            pub version: Option<String>,
            pub callback: Option<String>,
            pub callback_method: Option<String>,
            pub extra: Option<String>,
        }

        pub struct DeepgramFluxOptions {
            pub eager_eot_threshold: Option<f64>, // range 0.3-0.9, default: 0.5
            pub eot_threshold: Option<f64>,       // range 0.5-0.9
            pub eot_timeout_ms: Option<i64>,
            #[python(ty = "str | list[str]")]
            pub keyterm: Option<Value>,
            pub mip_opt_out: Option<bool>, // default: False
            #[python(ty = "str | list[str]")]
            pub tag: Option<Value>,
            pub detect_language: Option<bool>,
        }

        pub struct AssemblyaiOptions {
            pub format_turns: Option<bool>,                    // default: False
            pub end_of_turn_confidence_threshold: Option<f64>, // default: 0.01
            pub min_end_of_turn_silence_when_confident: Option<i64>, // default: 0
            pub max_turn_silence: Option<i64>,                 // default: not specified
            pub keyterms_prompt: Option<Vec<String>>,          // default: not specified
            pub language_detection: Option<bool>,
            pub inactivity_timeout: Option<f64>,       // seconds
            pub prompt: Option<String>, // default: not specified (u3-rt-pro only, mutually exclusive with keyterms_prompt)
            pub speaker_labels: Option<bool>, // when True, enables speaker diarization (default off)
            pub agent_context: Option<String>, // context to bias recognition (u3-rt-pro only, max 1750 chars)
            pub previous_context_n_turns: Option<i64>, // prior turns carried as context; 0 disables (u3-rt-pro only)
            #[python(ty = "Literal[\"near-field\", \"far-field\"]")]
            pub voice_focus: Option<Value>, // isolate primary voice (u3-rt-pro only)
            pub voice_focus_threshold: Option<f64>, // background suppression strength (u3-rt-pro only)
            #[python(ty = "Literal[\"min_latency\", \"balanced\", \"max_accuracy\"]")]
            pub mode: Option<Value>, // accuracy/latency preset (u3-rt-pro)
        }

        pub struct ElevenlabsOptions {
            #[python(ty = "Literal[\"manual\", \"vad\"]")]
            pub commit_strategy: Option<Value>,
            pub include_timestamps: Option<bool>,
            pub vad_silence_threshold_secs: Option<f64>,
            pub vad_threshold: Option<f64>,
            pub min_speech_duration_ms: Option<i64>,
            pub min_silence_duration_ms: Option<i64>,
            pub language_code: Option<String>,
        }

        pub struct SpeechmaticsOptions {
            pub domain: Option<String>,           // e.g. "finance"
            pub output_locale: Option<String>,    // BCP-47 locale for output formatting
            pub max_delay: Option<f64>,           // 0.7-4.0 seconds, default 1.0
            pub max_delay_mode: Option<String>,   // "flexible" | "fixed"
            pub diarization: Option<String>, // "none" | "speaker" | "channel" | "channel_and_speaker_change" | "speaker_change"; non-"none" enables diarization
            pub speaker_sensitivity: Option<f64>, // 0.0-1.0
            pub max_speakers: Option<i64>,
            pub prefer_current_speaker: Option<bool>,
            pub enable_partials: Option<bool>, // default True (overridden by gateway)
            pub enable_entities: Option<bool>,
            pub punctuation_overrides: Option<HashMap<String, Value>>,
            pub additional_vocab: Option<Vec<HashMap<String, Value>>>,
            pub end_of_utterance_silence_trigger: Option<f64>, // seconds of silence before final
            pub audio_filtering_config: Option<HashMap<String, Value>>,
            pub transcript_filtering_config: Option<HashMap<String, Value>>,
        }

        pub struct XaiOptions {
            pub diarize: Option<bool>, // when True, enables speaker diarization (default off)
            pub endpointing: Option<i64>, // silence duration in ms before utterance-final (0-5000)
            pub format: Option<bool>, // enables Inverse Text Normalization (e.g. "one hundred dollars" -> "$100"); requires language
            pub interim_results: Option<bool>, // default True; set False to opt out of interim transcripts
        }

        pub struct InworldOptions {
            pub enable_voice_profile: Option<bool>,    // default: True
            pub voice_profile_top_n: Option<i64>,      // range 1-20, default 10
            pub include_word_timestamps: Option<bool>, // default: True
            #[python(ty = "Literal[\"LINEAR16\", \"AUTO_DETECT\"]")]
            pub audio_encoding: Option<Value>, // default: LINEAR16
            pub inactivity_timeout_seconds: Option<i64>, // >= 0; 0 disables
            pub end_of_turn_confidence_threshold: Option<f64>, // range 0.0-1.0, default 0.5
            pub min_end_of_turn_silence_when_confident: Option<i64>, // >= 0 (ms)
            pub prompts: Option<Vec<String>>,
            pub vad_threshold: Option<f64>, // range 0.0-1.0, default 0.5
        }

        #[host(name = "STTLanguages")]
        pub const STT_LANGUAGES: &[&str] =
            &["multi", "en", "de", "es", "fr", "ja", "pt", "zh", "hi"];
        #[host(union = "FallbackModel, str")]
        pub const FALLBACK_MODEL_TYPE: &[&str] = &[];
        #[host(
            name = "STTModels",
            union = "DeepgramModels, DeepgramFluxModels, CartesiaModels, AssemblyAIModels, ElevenlabsModels, XaiModels, SpeechmaticsModels, InworldModels"
        )] // automatically select a provider based on the language
        pub const STT_MODELS: &[&str] = &["auto"];
        #[host(name = "STTEncoding")]
        pub const STT_ENCODING: &[&str] = &["pcm_s16le"];
    }

    #[crate::host(group(path = "livekit.agents.inference.tts"))]
    pub mod tts {
        use super::*;

        pub const CARTESIA_MODELS: &[&str] = &[
            "cartesia",
            "cartesia/sonic-3.5",
            "cartesia/sonic-3",
            "cartesia/sonic-2",
            "cartesia/sonic-turbo",
            "cartesia/sonic",
            "cartesia/sonic-3-latest",
            "cartesia/sonic-latest",
        ];
        pub const DEEPGRAM_MODELS: &[&str] = &["deepgram", "deepgram/aura", "deepgram/aura-2"];
        pub const ELEVENLABS_MODELS: &[&str] = &[
            "elevenlabs",
            "elevenlabs/eleven_flash_v2",
            "elevenlabs/eleven_flash_v2_5",
            "elevenlabs/eleven_turbo_v2",
            "elevenlabs/eleven_turbo_v2_5",
            "elevenlabs/eleven_multilingual_v2",
            "elevenlabs/eleven_v3",
        ];
        pub const RIME_MODELS: &[&str] = &[
            "rime",
            "rime/coda",
            "rime/mistv2",
            "rime/mistv3",
            "rime/mist",
        ];
        pub const INWORLD_MODELS: &[&str] = &[
            "inworld",
            "inworld/inworld-tts-2",
            "inworld/inworld-tts-1.5-max",
            "inworld/inworld-tts-1.5-mini",
            "inworld/inworld-tts-1.5",
            "inworld/inworld-tts-1-max",
            "inworld/inworld-tts-1",
        ];
        pub const XAI_MODELS: &[&str] = &["xai/tts-1"];
        pub const FISH_AUDIO_MODELS: &[&str] = &[
            "fishaudio",
            "fishaudio/s2.1-pro",
            "fishaudio/s2.1-pro-free",
            "fishaudio/s2-pro",
        ];
        #[host(
            name = "TTSModels",
            union = "CartesiaModels, DeepgramModels, ElevenlabsModels, RimeModels, InworldModels, XaiModels, FishAudioModels"
        )]
        pub const TTS_MODELS: &[&str] = &[];
        #[host(union = "FallbackModel, str")]
        pub const FALLBACK_MODEL_TYPE: &[&str] = &[];

        pub struct CartesiaOptions {
            pub emotion: Option<String>,
            #[python(ty = "Literal[\"slow\", \"normal\", \"fast\"] | float")]
            pub speed: Option<Value>,
            pub volume: Option<f64>,
            pub duration: Option<f64>,
            pub max_buffer_delay_ms: Option<i64>,
            pub add_timestamps: Option<bool>,
            pub add_phoneme_timestamps: Option<bool>,
            pub use_normalized_timestamps: Option<bool>,
        }

        pub struct DeepgramOptions {
            pub mip_opt_out: Option<bool>, // default: False
        }

        pub struct ElevenlabsOptions {
            pub inactivity_timeout: Option<i64>, // default: 60, range 5-180
            #[python(ty = "Literal[\"auto\", \"off\", \"on\"]")]
            pub apply_text_normalization: Option<Value>, // default: "auto"
            pub auto_mode: Option<bool>,
            pub enable_logging: Option<bool>,
            pub enable_ssml_parsing: Option<bool>,
            pub sync_alignment: Option<bool>,
            pub language_code: Option<String>,
            pub stability: Option<f64>,        // range 0-1
            pub similarity_boost: Option<f64>, // range 0-1
            pub style: Option<f64>,            // range 0-1
            pub speed: Option<f64>,            // range 0.25-4
            pub use_speaker_boost: Option<bool>,
            pub chunk_length_schedule: Option<Vec<f64>>,
            pub preferred_alignment: Option<String>,
        }

        /// Rime-specific WebSocket JSON query parameters.
        /// See: https://docs.rime.ai/api-reference/endpoint/websockets-json
        pub struct RimeOptions {
            pub speed_alpha: Option<f64>, // default 1.0, <1 = faster, >1 = slower
            pub pause_between_brackets: Option<bool>, // default False
            pub phonemize_between_brackets: Option<bool>, // default False
            pub inline_speed_alpha: Option<String>, // comma-separated speed factors for [bracketed] words
            pub no_text_normalization: Option<bool>, // default False
        }

        pub struct InworldOptions {
            pub speaking_rate: Option<f64>, // range >0.5, <=1.5
            pub temperature: Option<f64>,   // range 0-2
            // inworld-tts-2 only; temperature is ignored on that model, use this to steer variation
            #[python(
                ty = "Literal[\"DELIVERY_MODE_UNSPECIFIED\", \"STABLE\", \"BALANCED\", \"CREATIVE\"]"
            )]
            pub delivery_mode: Option<Value>,
            #[python(ty = "Literal[\"TIMESTAMP_TYPE_UNSPECIFIED\", \"WORD\", \"CHARACTER\"]")]
            pub timestamp_type: Option<Value>,
            #[python(ty = "Literal[\"APPLY_TEXT_NORMALIZATION_UNSPECIFIED\", \"ON\", \"OFF\"]")]
            pub apply_text_normalization: Option<Value>,
        }

        pub struct XaiOptions {
            #[python(ty = "Literal[32000, 64000, 96000, 128000, 192000]")]
            pub bit_rate: Option<Value>,
            pub speed: Option<f64>, // speaking-rate multiplier, default 1.0
            #[python(ty = "Literal[0, 1, 2]")]
            pub optimize_streaming_latency: Option<Value>, // latency optimization level, default 0
        }

        /// See https://docs.fish.audio/api-reference/endpoint/websocket/tts-live
        pub struct FishAudioOptions {
            pub speed: Option<f64>,  // prosody speaking-rate multiplier, 1.0 is natural
            pub volume: Option<f64>, // prosody loudness adjustment in dB, 0 is natural
            pub normalize_loudness: Option<bool>, // consistent output loudness; S2-Pro family only
            pub temperature: Option<f64>, // range 0-1, higher is more varied/expressive
            pub top_p: Option<f64>,  // nucleus sampling probability mass, range 0-1
            pub normalize: Option<bool>, // normalize numbers/dates/abbreviations before synthesis
            #[python(ty = "Literal[\"normal\", \"balanced\", \"low\"]")]
            pub latency: Option<Value>, // streaming latency mode
            pub chunk_length: Option<i64>, // characters buffered before auto-synthesis, range 100-300
            pub max_new_tokens: Option<i64>, // max audio tokens per text chunk, default 1024
            pub min_chunk_length: Option<i64>, // min characters before splitting a new chunk, range 0-100
            pub condition_on_previous_chunks: Option<bool>, // use prior audio as context for consistency
            pub early_stop_threshold: Option<f64>, // early-stopping threshold for batching, range 0-1
        }

        #[host(name = "TTSEncoding")]
        pub const TTS_ENCODING: &[&str] = &["pcm_s16le"];
    }

    #[crate::host(group(path = "livekit.agents.inference.llm"))]
    pub mod llm {
        use super::*;

        #[host(name = "OpenAIModels")]
        pub const OPEN_AI_MODELS: &[&str] = &[
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4.1",
            "openai/gpt-4.1-mini",
            "openai/gpt-4.1-nano",
            "openai/gpt-5",
            "openai/gpt-5-mini",
            "openai/gpt-5-nano",
            "openai/gpt-5.1",
            "openai/gpt-5.1-chat-latest",
            "openai/gpt-5.2",
            "openai/gpt-5.2-chat-latest",
            "openai/gpt-5.3-chat-latest",
            "openai/gpt-5.4",
            "openai/gpt-5.4-mini",
            "openai/gpt-5.4-nano",
            "openai/gpt-5.5",
            "openai/chat-latest",
            "openai/gpt-oss-120b",
        ];
        pub const GOOGLE_MODELS: &[&str] = &[
            "google/gemini-3.1-pro",
            "google/gemini-3-flash",
            "google/gemini-3.1-flash-lite",
            "google/gemini-3.5-flash",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-flash-lite",
        ];
        pub const KIMI_MODELS: &[&str] = &["moonshotai/kimi-k2.5", "moonshotai/kimi-k2.6"];
        pub const DEEP_SEEK_MODELS: &[&str] =
            &["deepseek-ai/deepseek-v3", "deepseek-ai/deepseek-v3.2"];
        #[host(name = "ZAIModels")]
        pub const ZAI_MODELS: &[&str] = &["zai/glm-5.1"];
        #[host(name = "XAIModels")]
        pub const XAI_MODELS: &[&str] = &[
            "xai/grok-4-1-fast-non-reasoning",
            "xai/grok-4-1-fast-reasoning",
            "xai/grok-4.20-0309-non-reasoning",
            "xai/grok-4.20-0309-reasoning",
            "xai/grok-4.20-multi-agent-0309",
            "xai/grok-4.3",
            "xai/grok-4.5",
        ];
        #[host(
            name = "LLMModels",
            union = "OpenAIModels, GoogleModels, KimiModels, DeepSeekModels, ZAIModels, XAIModels"
        )]
        pub const LLM_MODELS: &[&str] = &[];
        /// Scheduling class for a request. ``low`` yields to voice traffic, so it is only
        /// appropriate for work no caller is waiting on.
        pub const INFERENCE_CLASS: &[&str] = &["priority", "standard", "low"];

        #[python(typed_dict, typed_dict_from = "typing_extensions")]
        pub struct ChatCompletionOptions {
            pub frequency_penalty: Option<Option<f64>>,
            pub logit_bias: Option<Option<HashMap<String, i64>>>,
            pub logprobs: Option<Option<bool>>,
            pub max_completion_tokens: Option<Option<i64>>,
            pub max_tokens: Option<Option<i64>>,
            #[python(
                ty = "Metadata | None",
                type_import = "openai.types.shared_params:Metadata"
            )]
            #[node(ty = "Record<string, string> | null")]
            pub metadata: Option<Value>,
            #[python(ty = "list[Literal[\"text\", \"audio\"]] | None")]
            pub modalities: Option<Value>,
            pub n: Option<Option<i64>>,
            pub parallel_tool_calls: Option<bool>,
            #[python(
                ty = "ChatCompletionPredictionContentParam | None",
                type_import = "openai.types.chat:ChatCompletionPredictionContentParam"
            )]
            #[node(ty = "unknown")]
            pub prediction: Option<Value>,
            pub presence_penalty: Option<Option<f64>>,
            pub prompt_cache_key: Option<String>,
            #[python(ty = "Literal[\"in_memory\", \"24h\"] | None")]
            pub prompt_cache_retention: Option<Value>,
            #[python(
                ty = "ReasoningEffort | None",
                type_import = "openai.types.shared.reasoning_effort:ReasoningEffort"
            )]
            #[node(ty = "unknown")]
            pub reasoning_effort: Option<Value>,
            pub safety_identifier: Option<String>,
            pub seed: Option<Option<i64>>,
            #[python(
                ty = "Literal[\"auto\", \"default\", \"flex\", \"scale\", \"priority\"] | None"
            )]
            pub service_tier: Option<Value>,
            #[python(ty = "str | None | list[str] | None")]
            pub stop: Option<Value>,
            pub store: Option<Option<bool>>,
            pub temperature: Option<Option<f64>>,
            pub top_logprobs: Option<Option<i64>>,
            pub top_p: Option<Option<f64>>,
            pub user: Option<String>,
            #[python(ty = "Literal[\"low\", \"medium\", \"high\"] | None")]
            pub verbosity: Option<Value>,
            #[python(
                ty = "completion_create_params.WebSearchOptions",
                type_import = "openai.types.chat:completion_create_params"
            )]
            #[node(ty = "unknown")]
            pub web_search_options: Option<Value>,
            // livekit-typed arguments
            #[python(ty = "ToolChoice", type_import = "..llm:ToolChoice")]
            #[node(ty = "unknown")]
            pub tool_choice: Option<Value>,
            // TODO(theomonnomn): support repsonse format
            // response_format: completion_create_params.ResponseFormat
        }
    }

    #[crate::host(group(path = "livekit.agents.inference.avatar"))]
    pub mod avatar {

        /// Provider-specific options for a LemonSlice inference avatar.
        ///
        /// Mirrors the per-provider option dicts in :mod:`inference.stt` /
        /// :mod:`inference.tts` (e.g. ``CartesiaOptions``); passed via ``extra_kwargs``.
        /// All keys are optional.
        pub struct LemonSliceOptions {
            pub image_url: Option<String>, // appearance source; mutually exclusive with a model-string agent id
            pub prompt: Option<String>, // speaking prompt (mapped to the provider's agent_prompt)
            pub idle_prompt: Option<String>, // idle prompt (mapped to the provider's agent_idle_prompt)
            pub idle_timeout: Option<i64>, // provider idle timeout in seconds; the gateway clamps it
        }
    }

    #[crate::host(group(path = "livekit.agents.inference.vad"))]
    pub mod vad {

        #[host(name = "VADModels")]
        pub const VAD_MODELS: &[&str] = &["silero"];
    }
}
