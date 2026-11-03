//! The `lk.*_metrics` span attributes: a metrics record as the JSON object 1.8.10's pydantic
//! models dumped (`model_dump_json()`), field for field, which Cloud's consumers parse. An
//! unmeasured duration is `-1`, as the reference's defaults.

use lk_agents_abi::metrics::base::{LlmMetrics, Metadata, RealtimeModelMetrics, TtsMetrics};
use lk_agents_abi::usage::TokenUsage;
use serde_json::{Value, json};

fn metadata(m: Option<&Metadata>) -> Value {
    m.map_or(
        Value::Null,
        |m| json!({"model_name": m.model_name, "model_provider": m.model_provider}),
    )
}

fn or_unmeasured(v: Option<f64>) -> f64 {
    v.unwrap_or(-1.0)
}

/// `lk.llm_metrics`: `LLMMetrics`.
pub fn llm(m: &LlmMetrics) -> Value {
    let t = &m.tokens;
    json!({
        "type": "llm_metrics",
        "label": m.label,
        "request_id": m.request_id,
        "timestamp": m.timestamp,
        "duration": m.duration,
        "ttft": or_unmeasured(m.ttft),
        "cancelled": m.cancelled,
        "completion_tokens": t.output_tokens,
        "prompt_tokens": t.input_tokens,
        "prompt_cached_tokens": t.cache_read_tokens.unwrap_or(0),
        "cache_creation_tokens": t.cache_write_tokens.unwrap_or(0),
        "reasoning_tokens": t.output_reasoning_tokens.unwrap_or(0),
        "total_tokens": t.total_tokens(),
        "tokens_per_second": m.tokens_per_second,
        "speech_id": m.speech_id,
        "metadata": metadata(m.metadata.as_ref()),
    })
}

/// `lk.tts_metrics`: `TTSMetrics`.
pub fn tts(m: &TtsMetrics) -> Value {
    json!({
        "type": "tts_metrics",
        "label": m.label,
        "request_id": m.request_id,
        "timestamp": m.timestamp,
        "ttfb": or_unmeasured(m.ttfb),
        "duration": m.duration,
        "audio_duration": m.audio_duration,
        "cancelled": m.cancelled,
        "characters_count": m.characters_count,
        "input_tokens": m.tokens.input_tokens,
        "output_tokens": m.tokens.output_tokens,
        "streamed": m.streamed,
        "acquire_time": m.acquire_time.unwrap_or(0.0),
        "connection_reused": m.connection_reused,
        "segment_id": m.segment_id,
        "speech_id": m.speech_id,
        "metadata": metadata(m.metadata.as_ref()),
    })
}

/// `lk.realtime_model_metrics`: `RealtimeModelMetrics`, its token details nested.
pub fn realtime(m: &RealtimeModelMetrics) -> Value {
    let t: &TokenUsage = &m.tokens;
    let n = |v: Option<u64>| v.unwrap_or(0);
    let cached_details = (t.cache_read_text_tokens.is_some()
        || t.cache_read_audio_tokens.is_some()
        || t.cache_read_image_tokens.is_some())
    .then(|| {
        json!({
            "audio_tokens": n(t.cache_read_audio_tokens),
            "text_tokens": n(t.cache_read_text_tokens),
            "image_tokens": n(t.cache_read_image_tokens),
        })
    });
    json!({
        "type": "realtime_model_metrics",
        "label": m.label,
        "request_id": m.request_id,
        "timestamp": m.timestamp,
        "duration": m.duration,
        "session_duration": m.connection_duration.unwrap_or(0.0),
        "ttft": or_unmeasured(m.ttft),
        "cancelled": m.cancelled,
        "input_tokens": t.input_tokens,
        "output_tokens": t.output_tokens,
        "total_tokens": t.total_tokens(),
        "tokens_per_second": m.tokens_per_second,
        "input_token_details": {
            "audio_tokens": n(t.input_audio_tokens),
            "text_tokens": n(t.input_text_tokens),
            "image_tokens": n(t.input_image_tokens),
            "cached_tokens": n(t.cache_read_tokens),
            "cached_tokens_details": cached_details,
        },
        "output_token_details": {
            "text_tokens": n(t.output_text_tokens),
            "audio_tokens": n(t.output_audio_tokens),
            "image_tokens": n(t.output_image_tokens),
        },
        "acquire_time": m.acquire_time.unwrap_or(0.0),
        "connection_reused": m.connection_reused,
        "metadata": metadata(m.metadata.as_ref()),
    })
}

/// `langfuse.observation.completion_start_time`: the first chunk's instant as ISO 8601 UTC.
pub fn iso8601(unix_seconds: f64) -> String {
    let micros = (unix_seconds * 1e6).round() as i64;
    let (secs, micros) = (micros.div_euclid(1_000_000), micros.rem_euclid(1_000_000));
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, mo, d) = civil(days);
    let (h, mi, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{micros:06}+00:00")
}

/// Days since the epoch as a proleptic Gregorian date (Howard Hinnant's `civil_from_days`).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::metrics::testkit as tk;

    // Catches: a key Cloud parses renamed or dropped, or an unmeasured ttft written as null.
    #[test]
    fn blobs_keep_the_reference_field_sets() {
        let mut m = tk::llm();
        m.ttft = None;
        let v = llm(&m);
        let keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        for key in [
            "type",
            "label",
            "request_id",
            "timestamp",
            "duration",
            "ttft",
            "cancelled",
            "completion_tokens",
            "prompt_tokens",
            "prompt_cached_tokens",
            "cache_creation_tokens",
            "reasoning_tokens",
            "total_tokens",
            "tokens_per_second",
            "speech_id",
            "metadata",
        ] {
            assert!(keys.iter().any(|k| k == key), "{key} missing from {v}");
        }
        assert_eq!(v["ttft"], -1.0);
        let rt = realtime(&tk::realtime());
        assert_eq!(rt["type"], "realtime_model_metrics");
        assert!(rt["input_token_details"]["cached_tokens"].is_u64(), "{rt}");
        assert_eq!(tts(&tk::tts())["type"], "tts_metrics");
    }

    // Catches: a date or fraction misformatted for Langfuse.
    #[test]
    fn instants_are_iso_8601_utc() {
        assert_eq!(iso8601(0.0), "1970-01-01T00:00:00.000000+00:00");
        assert_eq!(iso8601(1_700_000_000.25), "2023-11-14T22:13:20.250000+00:00");
    }
}
