//! Answering-machine detection: the provider adaptation (model selection, the setup anchor, the
//! verdict vocabulary). The machine is `lk_agents_voice::amd::classifier`. Nothing under `amd/`
//! may `await`, spawn, or name a provider trait.

mod detector;

pub use detector::{
    DEFAULT_LLM_MODEL, DEFAULT_STT_MODEL, EVALUATED_LLM_MODELS, EVALUATED_STT_MODELS,
    SIP_CALL_STATUS_ACTIVE, SIP_CALL_STATUS_ATTR, TRACK_PUBLICATION_TIMEOUT, auto_select_models,
    category_is_human, category_is_machine, decide_setup_step_named, is_evaluated_model,
    is_evaluated_model_of_kind,
};
pub use lk_agents_abi::voice::amd::{AMD_PROMPT, AmdCategory, AmdPrediction, AmdReason};
