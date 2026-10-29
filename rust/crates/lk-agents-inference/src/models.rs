//! `_core`'s inference lookups ([`InferenceModels`]): the gateway's model tables, request headers
//! and local model registry, answered by the modules that own them.

use lk_agents_abi::inference::fallback::{SttFallbackModel, TtsFallbackModel};
use lk_agents_abi::inference::{InferenceModels, InvalidOption, VadResolution};
use lk_agents_abi::llm::realtime::RealtimeCapabilities;
use lk_agents_abi::tokenize::language::TurnDetectorModel;
use lk_agents_abi::{SttCapabilities, TtsCapabilities};
use serde_json::{Map, Value};

pub struct Models;

impl InferenceModels for Models {
    fn stt_capabilities(
        model: Option<String>,
        fallback: Option<Vec<String>>,
        extra_kwargs: Option<Map<String, Value>>,
    ) -> SttCapabilities {
        // Only each fallback entry's model is read.
        let fallback: Vec<crate::stt::options::FallbackModel> = fallback
            .unwrap_or_default()
            .into_iter()
            .map(|model| crate::stt::options::FallbackModel { model, extra_kwargs: None })
            .collect();
        crate::stt::options::capabilities(model.as_deref(), &fallback, extra_kwargs.as_ref())
    }

    fn stt_fallback_capabilities(entries: Vec<SttCapabilities>) -> SttCapabilities {
        crate::fallback::merge_stt_capabilities(&entries)
    }

    fn tts_fallback_capabilities(entries: Vec<TtsCapabilities>) -> TtsCapabilities {
        crate::fallback::merge_tts_capabilities(&entries)
    }

    fn merge_realtime_capabilities(
        caps: Vec<RealtimeCapabilities>,
    ) -> Result<RealtimeCapabilities, InvalidOption> {
        crate::fallback::merge_realtime_capabilities(&caps)
            .map_err(|mismatch| InvalidOption(mismatch.to_string()))
    }

    fn inference_quota_usage(headers: Vec<(String, String)>) -> Vec<(String, String)> {
        crate::headers::quota_usage(headers)
            .into_iter()
            .map(|(field, value)| (field.to_owned(), value))
            .collect()
    }

    fn inference_default_url(server_url: Option<String>) -> String {
        crate::headers::default_inference_url(
            std::env::var("LIVEKIT_INFERENCE_URL").ok().as_deref(),
            server_url.or_else(|| std::env::var("LIVEKIT_URL").ok()).as_deref(),
        )
    }

    fn local_models_init(vad: bool, eot: bool) -> bool {
        crate::local_models::init(vad, eot)
    }

    fn tts_parse_model_string(model: String) -> (String, Option<String>) {
        let (model, voice) = crate::tts::wire::parse_model_string(&model);
        (model.to_owned(), voice.map(str::to_owned))
    }

    fn tts_has_aligned_transcript(model: String, extra_kwargs: Map<String, Value>) -> bool {
        crate::tts::wire::has_aligned_transcript(&model, &extra_kwargs)
    }

    fn tts_markup_provider_key(model: String) -> String {
        crate::tts::wire::markup_provider_key(&model).to_owned()
    }

    fn stt_parse_model_string(model: String) -> (String, Option<String>) {
        let (model, language) = crate::stt::options::parse_model_string(&model);
        (model.to_owned(), language.map(str::to_owned))
    }

    fn min_reasoning_effort(model: String) -> Option<String> {
        crate::llm::models::min_reasoning_effort(&model).map(str::to_owned)
    }

    fn amd_auto_select_models(livekit_url: String, api_key: String, api_secret: String) -> bool {
        crate::amd::auto_select_models(&livekit_url, &api_key, &api_secret)
    }

    fn amd_is_evaluated_model(model: String, kind: String) -> Result<bool, InvalidOption> {
        crate::amd::is_evaluated_model_of_kind(&model, &kind).map_err(InvalidOption)
    }

    fn llm_unsupported_params(
        model: String,
        has_tools: bool,
        reasoning_effort: Option<String>,
        reasoning: Option<String>,
    ) -> Vec<String> {
        crate::llm::models::unsupported_params(
            &model,
            has_tools,
            reasoning_effort.as_deref(),
            reasoning.as_deref(),
        )
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    fn eot_select(
        version: Option<String>,
        hosted: bool,
        dev: bool,
        url: String,
        key: String,
        secret: String,
        local_fallback: bool,
    ) -> Result<(TurnDetectorModel, bool, Vec<String>), InvalidOption> {
        crate::eot::options::select(
            version.as_deref(),
            hosted,
            dev,
            &url,
            &key,
            &secret,
            local_fallback,
        )
        .map_err(InvalidOption)
    }

    fn stt_resolve_vad_for_model(model: Option<String>, has_vad: bool) -> VadResolution {
        crate::stt::options::resolve_vad_for_model(model.as_deref(), has_vad)
    }

    fn stt_normalize_fallback_str(model: String) -> SttFallbackModel {
        crate::stt::options::normalize_fallback_str(&model)
    }

    fn tts_normalize_fallback_str(model: String) -> TtsFallbackModel {
        crate::tts::wire::normalize_fallback_str(&model)
    }
}
