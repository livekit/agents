//! Provider distributions are interpolated deciles of the measured samples.
use std::{env, fs, path::Path};

use serde_json::Value;

fn deciles(samples: &Value) -> [u64; 11] {
    let mut samples: Vec<f64> = samples
        .as_array()
        .expect("calibration samples")
        .iter()
        .map(|v| v.as_f64().expect("numeric sample"))
        .collect();
    samples.sort_by(f64::total_cmp);
    std::array::from_fn(|i| {
        if samples.is_empty() {
            return 0;
        }
        let position = i as f64 * (samples.len() - 1) as f64 / 10.0;
        let low = position.floor() as usize;
        let high = position.ceil() as usize;
        (samples[low] + (samples[high] - samples[low]) * position.fract()).round() as u64
    })
}

fn distribution(out: &mut String, name: &str, samples: &Value) {
    out.push_str(&format!("pub const {name}: [u64; 11] = {:?};\n", deciles(samples)));
}

pub fn generate(root: &Path) {
    println!("cargo:rerun-if-changed=calibration.json");
    let data: Value =
        serde_json::from_slice(&fs::read(root.join("calibration.json")).unwrap()).unwrap();
    let mut out = String::from("// Generated from calibration.json. Do not edit.\n");
    out.push_str("pub mod tts {\n");
    for field in ["first_audio", "ms_per_word"] {
        distribution(&mut out, &field.to_uppercase(), &data["tts"]["cartesia/sonic-3"][field]);
    }
    out.push_str("}\npub mod llm {\n");
    for field in ["first_chunk", "chunk_gap", "chars_per_chunk"] {
        distribution(
            &mut out,
            &format!("LLM_{}", field.to_uppercase()),
            &data["llm"]["openai/gpt-4.1-mini"][field],
        );
    }
    out.push_str("}\npub mod realtime {\n");
    for field in [
        "speech_started_lag_semantic",
        "speech_started_lag_server",
        "speech_stopped_lag_semantic",
        "speech_stopped_lag_server",
        "cancel_after_speech_started",
        "created_after_stopped",
        "created_after_create",
        "transcription_after_committed",
        "item_after_created",
        "first_audio_after_item",
        "audio_gap",
        "done_after_last_audio",
        "error_after_client",
        "done_after_cancel",
        "audio_ms_per_char",
    ] {
        distribution(&mut out, &field.to_uppercase(), &data["realtime"][field]);
    }
    let share = &data["realtime"]["function_call_share"];
    out.push_str(&format!(
        "pub const FUNCTION_CALL_SHARE: (u64, u64) = ({}, {});\n",
        share[0], share[1]
    ));
    out.push_str("}\n");
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("calibration.rs"), out).unwrap();
    let mut stt = String::from(
        "// Generated from calibration.json. Do not edit.\npub const STT_MODELS: [SttModel; 3] = [\n",
    );
    for name in ["deepgram/nova-3", "assemblyai/universal-streaming", "deepgram/flux-general-en"] {
        let model = &data["stt"][name];
        stt.push_str(&format!("SttModel {{ name: {name:?},\n"));
        for field in [
            "first_after_onset",
            "interim_gap",
            "final_after_end",
            "finals_per_utterance",
            "preflights_per_utterance",
            "final_after_preflight",
            "duplicate_final_after",
        ] {
            stt.push_str(&format!("{field}: {:?},\n", deciles(&model[field])));
        }
        stt.push_str(&format!(
            "duplicate_share: ({}, {}), empty_finals: {} }},\n",
            model["duplicate_share"][0],
            model["duplicate_share"][1],
            model["empty_finals_per_minute"]
                .as_array()
                .expect("empty final rates")
                .iter()
                .any(|v| v.as_f64().expect("numeric rate") > 0.0)
        ));
    }
    stt.push_str("];\n");
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("stt_calibration.rs"), stt).unwrap();
}
