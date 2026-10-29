//! What real providers did, as the fakes and the test peers draw from it.
//!
//! `corpus/record.py` records real traffic through LiveKit Inference (the realtime model's raw
//! socket, streaming STT, TTS and LLM) as event kinds and relative timing, nothing anybody said;
//! `corpus/derive.py` turns the recordings into `corpus/calibration.json`, one list of observed
//! values per quantity. A draw here is uniform over a list, so a delay is distributed as the real
//! provider's was. The DST's fakes (lk-agents-fuzz) and inference's simulated realtime server
//! both draw from it, and both are checked against the recordings themselves.
//!
//! The lists are read once per process.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use fastrand::Rng;
use serde_json::Value;

/// The calibration, as `derive.py` wrote it.
pub const CALIBRATION: &str = include_str!("../corpus/calibration.json");

/// The corpus directory: its `recordings/` and `probes/` are what conformance tests replay.
pub fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus")
}

fn calibration() -> &'static Value {
    static PARSED: OnceLock<Value> = OnceLock::new();
    PARSED.get_or_init(|| serde_json::from_str(CALIBRATION).expect("corpus/calibration.json"))
}

/// One list of observed values, by its path in the calibration: `["realtime", "item_after_created"]`.
/// Panics on a path the calibration does not have, which is a fake asking for a quantity nobody
/// measured.
pub fn samples(path: &[&str]) -> &'static [f64] {
    static CACHE: OnceLock<std::sync::Mutex<HashMap<String, &'static [f64]>>> = OnceLock::new();
    let key = path.join("/");
    let cache = CACHE.get_or_init(Default::default);
    if let Some(found) = cache.lock().unwrap().get(&key) {
        return found;
    }
    let mut node = calibration();
    for step in path {
        node = node.get(step).unwrap_or_else(|| panic!("calibration has no {key}"));
    }
    let list: Vec<f64> = node
        .as_array()
        .unwrap_or_else(|| panic!("calibration {key} is not a list"))
        .iter()
        .filter_map(Value::as_f64)
        .collect();
    assert!(!list.is_empty(), "calibration {key} is empty");
    let leaked: &'static [f64] = Box::leak(list.into_boxed_slice());
    cache.lock().unwrap().insert(key, leaked);
    leaked
}

/// One observed value, drawn uniformly.
pub fn draw(rng: &mut Rng, path: &[&str]) -> f64 {
    let list = samples(path);
    list[rng.usize(0..list.len())]
}

/// One observed delay in milliseconds, drawn uniformly.
pub fn draw_ms(rng: &mut Rng, path: &[&str]) -> u64 {
    draw(rng, path).max(0.0).round() as u64
}

/// The least and the most a quantity was observed at.
pub fn range(path: &[&str]) -> (f64, f64) {
    let list = samples(path);
    let min = list.iter().copied().fold(f64::INFINITY, f64::min);
    let max = list.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

/// A ratio the calibration states as `[hits, total]`.
pub fn share(path: &[&str]) -> (u64, u64) {
    let list = samples(path);
    (list[0] as u64, list.get(1).copied().unwrap_or(1.0).max(1.0) as u64)
}

/// The STT models the corpus has, by the name a setup draws them by.
pub const STT_MODELS: &[&str] =
    &["deepgram/nova-3", "assemblyai/universal-streaming", "deepgram/flux-general-en"];

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a calibration the fakes cannot read, or a quantity they draw that derive.py
    // stopped writing.
    #[test]
    fn every_quantity_the_fakes_draw_is_calibrated() {
        for path in [
            &["realtime", "speech_started_lag_semantic"][..],
            &["realtime", "speech_started_lag_server"],
            &["realtime", "speech_stopped_lag_semantic"],
            &["realtime", "speech_stopped_lag_server"],
            &["realtime", "created_after_stopped"],
            &["realtime", "created_after_create"],
            &["realtime", "item_after_created"],
            &["realtime", "first_audio_after_item"],
            &["realtime", "audio_gap"],
            &["realtime", "done_after_last_audio"],
            &["realtime", "transcription_after_committed"],
            &["realtime", "cancel_after_speech_started"],
            &["realtime", "audio_ms_per_char"],
            &["realtime", "error_after_client"],
            &["realtime", "function_call_share"],
        ] {
            assert!(!samples(path).is_empty(), "{path:?}");
        }
        for model in STT_MODELS {
            for q in ["first_after_onset", "final_after_end", "finals_per_utterance"] {
                assert!(!samples(&["stt", model, q]).is_empty(), "{model} {q}");
            }
        }
    }
}
