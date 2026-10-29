//! Answering-machine detection: the provider adaptation (model selection, the verdict
//! vocabulary). The machine is `lk_agents_voice::amd::classifier`. Nothing under `amd/`
//! may `await`, spawn, or name a provider trait.

mod detector;

pub use detector::{
    DEFAULT_LLM_MODEL, DEFAULT_STT_MODEL, EVALUATED_LLM_MODELS, EVALUATED_STT_MODELS,
    auto_select_models, is_evaluated_model, is_evaluated_model_of_kind,
};
pub use lk_agents_abi::voice::amd::{AMD_PROMPT, AmdCategory, AmdPrediction, AmdReason};
