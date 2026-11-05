//! The model names LiveKit Inference serves, by modality: the type hints of `inference.STT`,
//! `inference.TTS`, `inference.LLM` and the local models. Model parameters stay `String`: a model
//! the gateway serves before it is listed here is still accepted.

// Speech-to-text.

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "DeepgramModels"
))]
pub enum SttDeepgramModel {
    #[host(rename = "deepgram/nova-3")]
    Nova3,
    #[host(rename = "deepgram/nova-3-medical")]
    Nova3Medical,
    #[host(rename = "deepgram/nova-2")]
    Nova2,
    #[host(rename = "deepgram/nova-2-medical")]
    Nova2Medical,
    #[host(rename = "deepgram/nova-2-conversationalai")]
    Nova2Conversationalai,
    #[host(rename = "deepgram/nova-2-phonecall")]
    Nova2Phonecall,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "DeepgramFluxModels"
))]
pub enum SttDeepgramFluxModel {
    #[host(rename = "deepgram/flux-general")]
    FluxGeneral,
    #[host(rename = "deepgram/flux-general-en")]
    FluxGeneralEn,
    #[host(rename = "deepgram/flux-general-multi")]
    FluxGeneralMulti,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "CartesiaModels"
))]
pub enum SttCartesiaModel {
    #[host(rename = "cartesia/ink-whisper")]
    InkWhisper,
    #[host(rename = "cartesia/ink-2")]
    Ink2,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "AssemblyAIModels"
))]
pub enum SttAssemblyAiModel {
    #[host(rename = "assemblyai/universal-streaming")]
    UniversalStreaming,
    #[host(rename = "assemblyai/universal-streaming-multilingual")]
    UniversalStreamingMultilingual,
    #[host(rename = "assemblyai/u3-rt-pro")]
    U3RtPro,
    #[host(rename = "assemblyai/universal-3-5-pro")]
    Universal35Pro,
    #[host(rename = "assemblyai/universal-3-6-pro")]
    Universal36Pro,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "XaiModels"
))]
pub enum SttXaiModel {
    #[host(rename = "xai/stt-1")]
    Stt1,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "SpeechmaticsModels"
))]
pub enum SttSpeechmaticsModel {
    #[host(rename = "speechmatics/enhanced")]
    Enhanced,
    #[host(rename = "speechmatics/standard")]
    Standard,
    #[host(rename = "speechmatics/linden-1")]
    Linden1,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "InworldModels"
))]
pub enum SttInworldModel {
    #[host(rename = "inworld/inworld-stt-1")]
    InworldStt1,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "GoogleModels"
))]
pub enum SttGoogleModel {
    #[host(rename = "google/gemini-3.5-transcribe-live")]
    Gemini35TranscribeLive,
}

/// The gateway picks the model.
#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "AutoModels"
))]
pub enum SttAutoModel {
    #[host(rename = "auto")]
    Auto,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "STTModels",
    transparent
))]
pub enum SttModel {
    Deepgram(SttDeepgramModel),
    DeepgramFlux(SttDeepgramFluxModel),
    Cartesia(SttCartesiaModel),
    AssemblyAi(SttAssemblyAiModel),
    Xai(SttXaiModel),
    Speechmatics(SttSpeechmaticsModel),
    Inworld(SttInworldModel),
    Google(SttGoogleModel),
    Auto(SttAutoModel),
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "STTLanguages"
))]
pub enum SttLanguage {
    #[host(rename = "multi")]
    Multi,
    #[host(rename = "en")]
    En,
    #[host(rename = "de")]
    De,
    #[host(rename = "es")]
    Es,
    #[host(rename = "fr")]
    Fr,
    #[host(rename = "ja")]
    Ja,
    #[host(rename = "pt")]
    Pt,
    #[host(rename = "zh")]
    Zh,
    #[host(rename = "hi")]
    Hi,
}

#[crate::host(record(
    module = "inference_stt",
    path = "livekit.agents.inference.stt",
    name = "STTEncoding"
))]
pub enum SttEncoding {
    #[host(rename = "pcm_s16le")]
    PcmS16le,
}

// Text-to-speech.

#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "CartesiaModels"
))]
pub enum TtsCartesiaModel {
    #[host(rename = "cartesia")]
    Cartesia,
    #[host(rename = "cartesia/sonic-3.5")]
    Sonic35,
    #[host(rename = "cartesia/sonic-3")]
    Sonic3,
    #[host(rename = "cartesia/sonic-2")]
    Sonic2,
    #[host(rename = "cartesia/sonic-turbo")]
    SonicTurbo,
    #[host(rename = "cartesia/sonic")]
    Sonic,
    #[host(rename = "cartesia/sonic-3-latest")]
    Sonic3Latest,
    #[host(rename = "cartesia/sonic-latest")]
    SonicLatest,
}

#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "DeepgramModels"
))]
pub enum TtsDeepgramModel {
    #[host(rename = "deepgram")]
    Deepgram,
    #[host(rename = "deepgram/aura")]
    Aura,
    #[host(rename = "deepgram/aura-2")]
    Aura2,
}

#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "RimeModels"
))]
pub enum TtsRimeModel {
    #[host(rename = "rime")]
    Rime,
    #[host(rename = "rime/coda")]
    Coda,
    #[host(rename = "rime/mistv2")]
    Mistv2,
    #[host(rename = "rime/mistv3")]
    Mistv3,
    #[host(rename = "rime/mist")]
    Mist,
}

#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "InworldModels"
))]
pub enum TtsInworldModel {
    #[host(rename = "inworld")]
    Inworld,
    #[host(rename = "inworld/inworld-tts-2")]
    InworldTts2,
    #[host(rename = "inworld/inworld-tts-1.5-max")]
    InworldTts15Max,
    #[host(rename = "inworld/inworld-tts-1.5-mini")]
    InworldTts15Mini,
    #[host(rename = "inworld/inworld-tts-1.5")]
    InworldTts15,
    #[host(rename = "inworld/inworld-tts-1-max")]
    InworldTts1Max,
    #[host(rename = "inworld/inworld-tts-1")]
    InworldTts1,
}

#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "XaiModels"
))]
pub enum TtsXaiModel {
    #[host(rename = "xai/tts-1")]
    Tts1,
}

#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "FishAudioModels"
))]
pub enum TtsFishAudioModel {
    #[host(rename = "fishaudio")]
    Fishaudio,
    #[host(rename = "fishaudio/s2.1-pro")]
    S21Pro,
    #[host(rename = "fishaudio/s2.1-pro-free")]
    S21ProFree,
    #[host(rename = "fishaudio/s2-pro")]
    S2Pro,
}

#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "TTSModels",
    transparent
))]
pub enum TtsModel {
    Cartesia(TtsCartesiaModel),
    Deepgram(TtsDeepgramModel),
    Rime(TtsRimeModel),
    Inworld(TtsInworldModel),
    Xai(TtsXaiModel),
    FishAudio(TtsFishAudioModel),
}

#[crate::host(record(
    module = "inference_tts",
    path = "livekit.agents.inference.tts",
    name = "TTSEncoding"
))]
pub enum TtsEncoding {
    #[host(rename = "pcm_s16le")]
    PcmS16le,
}

// Language models.

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "ReasoningEffort"
))]
pub enum ReasoningEffort {
    #[host(rename = "none")]
    None,
    #[host(rename = "minimal")]
    Minimal,
    #[host(rename = "low")]
    Low,
    #[host(rename = "medium")]
    Medium,
    #[host(rename = "high")]
    High,
    #[host(rename = "xhigh")]
    Xhigh,
    #[host(rename = "max")]
    Max,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "OpenAIModels"
))]
pub enum LlmOpenAiModel {
    #[host(rename = "openai/gpt-4o")]
    Gpt4o,
    #[host(rename = "openai/gpt-4o-mini")]
    Gpt4oMini,
    #[host(rename = "openai/gpt-4.1")]
    Gpt41,
    #[host(rename = "openai/gpt-4.1-mini")]
    Gpt41Mini,
    #[host(rename = "openai/gpt-4.1-nano")]
    Gpt41Nano,
    #[host(rename = "openai/gpt-5")]
    Gpt5,
    #[host(rename = "openai/gpt-5-mini")]
    Gpt5Mini,
    #[host(rename = "openai/gpt-5-nano")]
    Gpt5Nano,
    #[host(rename = "openai/gpt-5.1")]
    Gpt51,
    #[host(rename = "openai/gpt-5.1-chat-latest")]
    Gpt51ChatLatest,
    #[host(rename = "openai/gpt-5.2")]
    Gpt52,
    #[host(rename = "openai/gpt-5.2-chat-latest")]
    Gpt52ChatLatest,
    #[host(rename = "openai/gpt-5.3-chat-latest")]
    Gpt53ChatLatest,
    #[host(rename = "openai/gpt-5.4")]
    Gpt54,
    #[host(rename = "openai/gpt-5.4-mini")]
    Gpt54Mini,
    #[host(rename = "openai/gpt-5.4-nano")]
    Gpt54Nano,
    #[host(rename = "openai/gpt-5.5")]
    Gpt55,
    #[host(rename = "openai/gpt-5.6-luna")]
    Gpt56Luna,
    #[host(rename = "openai/gpt-5.6-sol")]
    Gpt56Sol,
    #[host(rename = "openai/gpt-5.6-terra")]
    Gpt56Terra,
    #[host(rename = "openai/chat-latest")]
    ChatLatest,
    #[host(rename = "openai/gpt-oss-120b")]
    GptOss120b,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "GoogleModels"
))]
pub enum LlmGoogleModel {
    #[host(rename = "google/gemini-3.1-pro")]
    Gemini31Pro,
    #[host(rename = "google/gemini-3-flash")]
    Gemini3Flash,
    #[host(rename = "google/gemini-3.1-flash-lite")]
    Gemini31FlashLite,
    #[host(rename = "google/gemini-3.5-flash")]
    Gemini35Flash,
    #[host(rename = "google/gemini-2.5-pro")]
    Gemini25Pro,
    #[host(rename = "google/gemini-2.5-flash")]
    Gemini25Flash,
    #[host(rename = "google/gemini-2.5-flash-lite")]
    Gemini25FlashLite,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "KimiModels"
))]
pub enum LlmKimiModel {
    #[host(rename = "moonshotai/kimi-k2.5")]
    KimiK25,
    #[host(rename = "moonshotai/kimi-k2.6")]
    KimiK26,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "DeepSeekModels"
))]
pub enum LlmDeepSeekModel {
    #[host(rename = "deepseek-ai/deepseek-v3")]
    DeepseekV3,
    #[host(rename = "deepseek-ai/deepseek-v3.2")]
    DeepseekV32,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "ZAIModels"
))]
pub enum LlmZaiModel {
    #[host(rename = "zai/glm-5.1")]
    Glm51,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "XAIModels"
))]
pub enum LlmXaiModel {
    #[host(rename = "xai/grok-4-1-fast-non-reasoning")]
    Grok41FastNonReasoning,
    #[host(rename = "xai/grok-4-1-fast-reasoning")]
    Grok41FastReasoning,
    #[host(rename = "xai/grok-4.20-0309-non-reasoning")]
    Grok4200309NonReasoning,
    #[host(rename = "xai/grok-4.20-0309-reasoning")]
    Grok4200309Reasoning,
    #[host(rename = "xai/grok-4.20-multi-agent-0309")]
    Grok420MultiAgent0309,
    #[host(rename = "xai/grok-4.3")]
    Grok43,
    #[host(rename = "xai/grok-4.5")]
    Grok45,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "LLMModels",
    transparent
))]
pub enum LlmModel {
    OpenAi(LlmOpenAiModel),
    Google(LlmGoogleModel),
    Kimi(LlmKimiModel),
    DeepSeek(LlmDeepSeekModel),
    Zai(LlmZaiModel),
    Xai(LlmXaiModel),
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "OpenAIRealtimeModels"
))]
pub enum RealtimeOpenAiModel {
    #[host(rename = "openai/gpt-realtime")]
    GptRealtime,
    #[host(rename = "openai/gpt-realtime-mini")]
    GptRealtimeMini,
    #[host(rename = "openai/gpt-realtime-1.5")]
    GptRealtime15,
    #[host(rename = "openai/gpt-realtime-2")]
    GptRealtime2,
    #[host(rename = "openai/gpt-realtime-2.1")]
    GptRealtime21,
    #[host(rename = "openai/gpt-realtime-2.1-mini")]
    GptRealtime21Mini,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "XAIRealtimeModels"
))]
pub enum RealtimeXaiModel {
    #[host(rename = "xai/grok-voice")]
    GrokVoice,
    #[host(rename = "xai/grok-voice-latest")]
    GrokVoiceLatest,
    #[host(rename = "xai/grok-voice-think-fast-2.0")]
    GrokVoiceThinkFast20,
}

#[crate::host(record(
    module = "inference_llm",
    path = "livekit.agents.inference.llm",
    name = "RealtimeModels",
    transparent
))]
pub enum RealtimeModel {
    OpenAi(RealtimeOpenAiModel),
    Xai(RealtimeXaiModel),
}

// Local models and request scheduling.

#[crate::host(record(
    module = "inference_models",
    path = "livekit.agents.inference",
    name = "VADModels"
))]
pub enum VadModel {
    #[host(rename = "silero")]
    Silero,
}

#[crate::host(record(
    module = "inference_models",
    path = "livekit.agents.inference",
    name = "TurnDetectorModels"
))]
pub enum TurnDetectorModel {
    #[host(rename = "turn-detector-v1")]
    TurnDetectorV1,
    #[host(rename = "turn-detector-v1-mini")]
    TurnDetectorV1Mini,
}

#[crate::host(record(
    module = "inference_models",
    path = "livekit.agents.inference",
    name = "TurnDetectorVersions"
))]
pub enum TurnDetectorVersion {
    #[host(rename = "v1")]
    V1,
    #[host(rename = "v1-mini")]
    V1Mini,
}

/// Scheduling class for a request. ``low`` yields to voice traffic, so it is only
/// appropriate for work no caller is waiting on.
#[crate::host(record(
    module = "inference_models",
    path = "livekit.agents.inference",
    name = "InferenceClass"
))]
pub enum InferenceClass {
    #[host(rename = "priority")]
    Priority,
    #[host(rename = "standard")]
    Standard,
    #[host(rename = "low")]
    Low,
}
