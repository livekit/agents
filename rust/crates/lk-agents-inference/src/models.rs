use lk_agents_abi::errors::Invalid;
use lk_agents_abi::inference::InferenceModels;
use lk_agents_abi::{SttCapabilities, TtsCapabilities};

pub struct Models;

impl InferenceModels for Models {
    fn stt_fallback_capabilities(entries: Vec<SttCapabilities>) -> SttCapabilities {
        crate::fallback::merge_stt_capabilities(&entries)
    }

    fn tts_fallback_capabilities(entries: Vec<TtsCapabilities>) -> TtsCapabilities {
        crate::fallback::merge_tts_capabilities(&entries)
    }

    fn inference_quota_usage(headers: crate::headers::Headers) -> Vec<(String, String)> {
        crate::headers::quota_usage(headers)
            .into_iter()
            .map(|(field, value)| (field.to_owned(), value))
            .collect()
    }

    fn local_models_init(vad: bool, eot: bool) -> bool {
        crate::local_models::init(vad, eot)
    }

    fn tts_markup_provider_key(model: String) -> String {
        crate::tts::wire::markup_provider(&model).map_or("", |p| p.as_str()).to_owned()
    }

    fn min_reasoning_effort(model: String) -> Option<String> {
        crate::llm::models::min_reasoning_effort(&model).map(str::to_owned)
    }

    fn amd_auto_select_models(livekit_url: String, api_key: String, api_secret: String) -> bool {
        crate::amd::auto_select_models(&livekit_url, &api_key, &api_secret)
    }

    fn amd_is_evaluated_model(model: String, kind: String) -> Result<bool, Invalid> {
        crate::amd::is_evaluated_model_of_kind(&model, &kind)
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
    ) -> Result<lk_agents_abi::inference::EotSelection, Invalid> {
        crate::eot::select(version.as_deref(), hosted, dev, &url, &key, &secret, local_fallback)
    }
}
