use lk_agents_abi::inference::models::InferenceProvider;
use lk_agents_abi::{AlignedTranscript, SttCapabilities};
use serde_json::{Map, Value};

// Several options are truthiness tests on values providers send empty or `false`; getting one
// wrong silently flips a capability.

/// Models verified to carry word timings. Unknown models stay disabled until verified.
const WORD_ALIGNED_MODELS: &[&str] = &[
    "deepgram/nova-3",
    "deepgram/nova-3-medical",
    "deepgram/nova-2",
    "deepgram/nova-2-medical",
    "deepgram/nova-2-conversationalai",
    "deepgram/nova-2-phonecall",
    "deepgram/flux-general",
    "deepgram/flux-general-en",
    "deepgram/flux-general-multi",
    "cartesia/ink-whisper",
    "assemblyai/universal-streaming",
    "assemblyai/universal-streaming-multilingual",
    "assemblyai/u3-rt-pro",
    "assemblyai/universal-3-5-pro",
    "assemblyai/universal-3-6-pro",
    "xai/stt-1",
    "speechmatics/enhanced",
    "speechmatics/standard",
];

/// Models that natively carry `agent_context` (the AssemblyAI U3 Pro family).
const ASSEMBLYAI_CARRYOVER_MODELS: &[&str] =
    &["assemblyai/u3-rt-pro", "assemblyai/universal-3-5-pro", "assemblyai/universal-3-6-pro"];

/// Speechmatics' model that detects turns server-side: no client VAD, no keyterm prompting.
const SPEECHMATICS_SERVER_TURNS: &str = "speechmatics/linden-1";

const DIARIZATION_EXTRA_KEYS: &[&str] = &[
    "diarize",        // Deepgram, xAI
    "speaker_labels", // AssemblyAI
    "diarization",    // Speechmatics
];

pub const DEFAULT_ENCODING: &str = "pcm_s16le";
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;
/// The only rate OpenAI's live transcription takes.
pub const OPENAI_SAMPLE_RATE: u32 = 24_000;

/// `"provider/model:language"` -> `("provider/model", Some("language"))`, split at the last colon.
pub fn parse_model_string(model: &str) -> (&str, Option<&str>) {
    match model.rfind(':') {
        Some(idx) => (&model[..idx], Some(&model[idx + 1..])),
        None => (model, None),
    }
}

/// True if any known provider diarization flag is truthy.
pub fn diarization_enabled(extra_kwargs: Option<&Map<String, Value>>) -> bool {
    let Some(extra) = extra_kwargs else { return false };
    for key in DIARIZATION_EXTRA_KEYS {
        let Some(value) = extra.get(*key) else { continue };
        // `if not value: continue` -- `false`, `0` and `""` do not turn diarization on.
        if !lk_agents_core::llm::provider_format::is_truthy(value) {
            continue;
        }
        // Speechmatics' `diarization` accepts the string "none" to mean off.
        if value.as_str().is_some_and(|s| s.eq_ignore_ascii_case("none")) {
            continue;
        }
        return true;
    }
    false
}

/// `None` means the model has no keyterm prompting (also the capability check).
pub fn keyterms_extra_for_model(
    model: Option<&str>,
    extra_kwargs: Option<&Map<String, Value>>,
    session_keyterms: &[String],
) -> Option<Map<String, Value>> {
    let model = model?;
    let empty = Map::new();
    let extra = extra_kwargs.unwrap_or(&empty);

    let key = match InferenceProvider::of_model(model)? {
        _ if model == SPEECHMATICS_SERVER_TURNS => return None,
        InferenceProvider::Deepgram => "keyterm",
        InferenceProvider::Cartesia if model == "cartesia/ink-2" => "keyterm",
        InferenceProvider::Assemblyai => "keyterms_prompt",
        InferenceProvider::Openai => "keywords",
        InferenceProvider::Speechmatics => {
            // Keep existing entries as-is (they may carry `sounds_like` etc.) and append new terms.
            let mut existing: Vec<super::provider_options::Vocabulary> = extra
                .get("additional_vocab")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok())
                .unwrap_or_default();
            for term in session_keyterms {
                if existing.iter().any(|entry| &entry.content == term) {
                    continue;
                }
                existing.push(super::provider_options::Vocabulary {
                    content: term.clone(),
                    ..Default::default()
                });
            }
            let mut out = Map::new();
            out.insert(
                "additional_vocab".into(),
                serde_json::to_value(existing).expect("STT vocabulary serialization"),
            );
            return Some(out);
        }
        _ => return None,
    };

    // Deepgram's `keyterm` may be a bare string; it must become a one-element list.
    let mut merged: Vec<String> =
        match extra.get(key).cloned().and_then(|value| serde_json::from_value(value).ok()) {
            Some(super::provider_options::OneOrMany::One(term)) => vec![term],
            Some(super::provider_options::OneOrMany::Many(terms)) => terms,
            _ => Vec::new(),
        };
    // `dict.fromkeys([...])`: order-preserving deduplication.
    for term in session_keyterms {
        if !merged.contains(term) {
            merged.push(term.clone());
        }
    }
    let mut out = Map::new();
    out.insert(key.into(), Value::Array(merged.into_iter().map(Value::String).collect()));
    Some(out)
}

/// True if the model natively carries `agent_context`.
pub fn supports_agent_context_carryover(model: Option<&str>) -> bool {
    model.is_some_and(|m| ASSEMBLYAI_CARRYOVER_MODELS.contains(&m))
}

/// Word-level alignment for the model; `false` for `"auto"`, since claiming alignment is the costlier error.
pub fn aligned_transcript_for_model(model: Option<&str>) -> bool {
    model.is_some_and(|m| WORD_ALIGNED_MODELS.contains(&m))
}

pub use lk_agents_abi::inference::fallback::SttFallbackModel as FallbackModel;

/// `_normalize_fallback`: a bare string becomes `FallbackModel(model=<name>)`, with the
/// `:language` suffix parsed off and discarded.
pub fn normalize_fallback_str(model: &str) -> lk_agents_abi::inference::fallback::SttFallbackModel {
    let (name, _language) = parse_model_string(model);
    FallbackModel { model: name.to_owned(), extra_kwargs: None }
}

/// `aligned_transcript` requires every link of the chain to be aligned.
pub fn capabilities(
    model: Option<&str>,
    fallback: &[FallbackModel],
    extra_kwargs: Option<&Map<String, Value>>,
) -> SttCapabilities {
    let aligned = aligned_transcript_for_model(model)
        && fallback.iter().all(|f| aligned_transcript_for_model(Some(&f.model)));
    SttCapabilities {
        streaming: true,
        interim_results: true,
        diarization: diarization_enabled(extra_kwargs),
        // `Literal["word", False]` -- the gateway never reports chunk-level alignment.
        aligned_transcript: if aligned { AlignedTranscript::Word } else { AlignedTranscript::None },
        offline_recognize: false,
        keyterms: keyterms_extra_for_model(model, None, &[]).is_some(),
        chat_context: supports_agent_context_carryover(model),
    }
}

/// What a model does with the caller's VAD (`_resolve_vad_for_model`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadResolution {
    /// Keep the caller's instance (Speechmatics, which endpoints client-side).
    Keep,
    /// Ignore the caller's instance and warn: the model endpoints server-side.
    DropAndWarn,
    /// A model that endpoints client-side with no VAD supplied: the default local one.
    CreateDefault,
    /// Nothing to do.
    None,
}

/// `_resolve_vad_for_model`: what the host does with the caller's VAD for this model.
pub fn resolve_vad_for_model(model: Option<&str>, has_vad: bool) -> VadResolution {
    let needs_client_vad = model.is_some_and(|model| {
        matches!(InferenceProvider::of_model(model), Some(InferenceProvider::Openai))
            || (InferenceProvider::of_model(model) == Some(InferenceProvider::Speechmatics)
                && model != SPEECHMATICS_SERVER_TURNS)
    });
    if has_vad && !needs_client_vad {
        return VadResolution::DropAndWarn;
    }
    if needs_client_vad && !has_vad {
        return VadResolution::CreateDefault;
    }
    if has_vad { VadResolution::Keep } else { VadResolution::None }
}

#[cfg(test)]
#[path = "options_tests.rs"]
mod tests;
