//! `metrics/utils.py`'s `log_metrics`: Rust decides what to record, the host owns the logger.

use crate::metrics::base::AgentMetrics;

/// Ints and floats stay distinct, as in the reference.
#[derive(Debug, Clone, PartialEq)]
pub enum LogValue {
    Str(String),
    Float(f64),
    Int(u64),
}

/// `fields` keeps insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsLog {
    pub message: &'static str,
    pub fields: Vec<(&'static str, LogValue)>,
}

/// `round(x, n)`, half-to-even as Python's (`round(0.125, 2)` is `0.12`).
fn round_to(x: f64, digits: u32) -> f64 {
    let scale = 10f64.powi(digits as i32);
    (x * scale).round_ties_even() / scale
}

/// `utils.py:22`. Presence outside, `or "unknown"` inside; absent metadata adds neither key.
fn metadata_fields(m: &AgentMetrics) -> Vec<(&'static str, LogValue)> {
    match m.metadata() {
        None => Vec::new(),
        Some(md) => {
            let or_unknown = |s: &Option<String>| {
                LogValue::Str(match s {
                    Some(v) if !v.is_empty() => v.clone(),
                    _ => "unknown".to_owned(),
                })
            };
            vec![
                ("model_name", or_unknown(&md.model_name)),
                ("model_provider", or_unknown(&md.model_provider)),
            ]
        }
    }
}

/// `None` for `VADMetrics` and `EOTInferenceMetrics`, which the reference silently ignores.
pub fn log_fields(metrics: &AgentMetrics) -> Option<MetricsLog> {
    let mut fields = metadata_fields(metrics);
    let message = match metrics {
        AgentMetrics::Llm(m) => {
            fields.extend([
                ("ttft", LogValue::Float(round_to(m.ttft, 2))),
                ("prompt_tokens", LogValue::Int(u64::from(m.prompt_tokens))),
                ("prompt_cached_tokens", LogValue::Int(u64::from(m.prompt_cached_tokens))),
                ("cache_creation_tokens", LogValue::Int(u64::from(m.cache_creation_tokens))),
                ("completion_tokens", LogValue::Int(u64::from(m.completion_tokens))),
                ("tokens_per_second", LogValue::Float(round_to(m.tokens_per_second, 2))),
            ]);
            "LLM metrics"
        }
        AgentMetrics::RealtimeModel(m) => {
            let input = &m.input_token_details;
            let cached = input.cached_tokens_details.unwrap_or_default();
            fields.extend([
                ("ttft", LogValue::Float(round_to(m.ttft, 2))),
                ("input_tokens", LogValue::Int(u64::from(m.input_tokens))),
                // Spelled the other way round from its siblings, as the reference.
                ("cached_input_tokens", LogValue::Int(u64::from(input.cached_tokens))),
                ("input_text_tokens", LogValue::Int(u64::from(input.text_tokens))),
                ("input_cached_text_tokens", LogValue::Int(u64::from(cached.text_tokens))),
                ("input_image_tokens", LogValue::Int(u64::from(input.image_tokens))),
                ("input_cached_image_tokens", LogValue::Int(u64::from(cached.image_tokens))),
                ("input_audio_tokens", LogValue::Int(u64::from(input.audio_tokens))),
                ("input_cached_audio_tokens", LogValue::Int(u64::from(cached.audio_tokens))),
                ("output_tokens", LogValue::Int(u64::from(m.output_tokens))),
                (
                    "output_text_tokens",
                    LogValue::Int(u64::from(m.output_token_details.text_tokens)),
                ),
                (
                    "output_audio_tokens",
                    LogValue::Int(u64::from(m.output_token_details.audio_tokens)),
                ),
                (
                    "output_image_tokens",
                    LogValue::Int(u64::from(m.output_token_details.image_tokens)),
                ),
                ("total_tokens", LogValue::Int(u64::from(m.total_tokens))),
                ("tokens_per_second", LogValue::Float(round_to(m.tokens_per_second, 2))),
            ]);
            "RealtimeModel metrics"
        }
        AgentMetrics::Tts(m) => {
            fields.extend([
                // `ttfb` is the one duration in the whole function that is *not* rounded.
                ("ttfb", LogValue::Float(m.ttfb)),
                ("audio_duration", LogValue::Float(round_to(m.audio_duration, 2))),
            ]);
            "TTS metrics"
        }
        AgentMetrics::Eou(m) => {
            fields.extend([
                ("end_of_utterance_delay", LogValue::Float(round_to(m.end_of_utterance_delay, 2))),
                ("transcription_delay", LogValue::Float(round_to(m.transcription_delay, 2))),
            ]);
            // `on_user_turn_completed_delay` is on the record but not in the log line.
            "EOU metrics"
        }
        AgentMetrics::Stt(m) => {
            fields.push(("audio_duration", LogValue::Float(round_to(m.audio_duration, 2))));
            "STT metrics"
        }
        AgentMetrics::Interruption(m) => {
            fields.extend([
                ("total_duration", LogValue::Float(round_to(m.total_duration, 2))),
                ("prediction_duration", LogValue::Float(round_to(m.prediction_duration, 2))),
                ("detection_delay", LogValue::Float(round_to(m.detection_delay, 2))),
                ("num_interruptions", LogValue::Int(u64::from(m.num_interruptions))),
                ("num_backchannels", LogValue::Int(u64::from(m.num_backchannels))),
                ("num_requests", LogValue::Int(u64::from(m.num_requests))),
            ]);
            "Interruption metrics"
        }
        AgentMetrics::Avatar(m) => {
            // Truthiness (a `0.0` is omitted); three digits here, not two.
            let truthy = |v: Option<f64>| v.filter(|f| *f != 0.0);
            if let (Some(started), Some(joined)) =
                (truthy(m.session_started_time), truthy(m.avatar_joined_time))
            {
                fields
                    .push(("avatar_join_latency", LogValue::Float(round_to(joined - started, 3))));
            }
            if m.playback_latency != 0.0 {
                fields.push(("playback_latency", LogValue::Float(round_to(m.playback_latency, 3))));
            }
            "Avatar metrics"
        }
        // No branch in the reference's chain: nothing is logged for either.
        AgentMetrics::Vad(_) | AgentMetrics::EotInference(_) => return None,
    };
    Some(MetricsLog { message, fields })
}

/// `log_metrics(metrics)` minus the logging: `(message, extra)`, or `None` for do not log
/// (`VADMetrics` and `EOTInferenceMetrics` have no branch in the reference).
#[cfg(feature = "python")]
#[pyo3::pyfunction]
pub fn metrics_log_fields<'py>(
    py: pyo3::Python<'py>,
    metrics: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<Option<(&'static str, pyo3::Bound<'py, pyo3::types::PyDict>)>> {
    use pyo3::types::PyDictMethods as _;
    let Some(record) = AgentMetrics::extract_known(metrics)? else { return Ok(None) };
    let Some(out) = log_fields(&record) else { return Ok(None) };
    let fields = pyo3::types::PyDict::new(py);
    for (key, value) in out.fields {
        match value {
            LogValue::Str(s) => fields.set_item(key, pyo3::types::PyString::new(py, &s))?,
            LogValue::Float(f) => fields.set_item(key, f)?,
            LogValue::Int(i) => fields.set_item(key, i)?,
        }
    }
    Ok(Some((out.message, fields)))
}

#[cfg(feature = "python")]
crate::register_host!(fn metrics_log_fields);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::base::{CachedTokenDetails, InputTokenDetails, OutputTokenDetails};
    use crate::metrics::testkit as tk;
    use lk_agents_abi::metrics::base::Metadata;

    fn keys(l: &MetricsLog) -> Vec<&'static str> {
        l.fields.iter().map(|(k, _)| *k).collect()
    }

    fn get(l: &MetricsLog, k: &str) -> Option<LogValue> {
        l.fields.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone())
    }

    // Catches: logging VAD or EOT inference metrics (they would flood the log).
    #[test]
    fn vad_and_eot_inference_metrics_are_never_logged() {
        assert!(log_fields(&AgentMetrics::Vad(tk::vad())).is_none());
        assert!(log_fields(&AgentMetrics::EotInference(tk::eot())).is_none());
    }

    // Catches: `"unknown"` placeholders with no metadata at all, or a reported name replaced by
    // `"unknown"`.
    #[test]
    fn absent_metadata_omits_both_keys_while_empty_strings_become_unknown() {
        let mut m = tk::stt();
        let none = log_fields(&AgentMetrics::Stt(m.clone())).unwrap();
        assert_eq!(keys(&none), ["audio_duration"]);

        m.metadata = Some(Metadata { model_name: Some(String::new()), model_provider: None });
        let empty = log_fields(&AgentMetrics::Stt(m.clone())).unwrap();
        assert_eq!(get(&empty, "model_name"), Some(LogValue::Str("unknown".into())));
        assert_eq!(get(&empty, "model_provider"), Some(LogValue::Str("unknown".into())));

        m.metadata = tk::meta("deepgram", "nova-3");
        let named = log_fields(&AgentMetrics::Stt(m)).unwrap();
        assert_eq!(get(&named, "model_name"), Some(LogValue::Str("nova-3".into())));
        assert_eq!(get(&named, "model_provider"), Some(LogValue::Str("deepgram".into())));
    }


    // Catches: normalising `cached_input_tokens` to `input_cached_tokens`.
    #[test]
    fn the_realtime_cached_input_key_keeps_its_odd_spelling() {
        let mut m = tk::realtime();
        m.input_token_details = InputTokenDetails { cached_tokens: 9, ..Default::default() };
        let l = log_fields(&AgentMetrics::RealtimeModel(m)).unwrap();
        assert_eq!(get(&l, "cached_input_tokens"), Some(LogValue::Int(9)));
        assert_eq!(get(&l, "input_cached_tokens"), None);
    }

    // Catches: reading the cached breakdown unconditionally.
    #[test]
    fn realtime_cached_breakdown_defaults_to_zero_per_modality() {
        let mut m = tk::realtime();
        m.input_token_details =
            InputTokenDetails { cached_tokens: 9, text_tokens: 4, ..Default::default() };
        let l = log_fields(&AgentMetrics::RealtimeModel(m.clone())).unwrap();
        assert_eq!(get(&l, "input_cached_text_tokens"), Some(LogValue::Int(0)));
        assert_eq!(get(&l, "input_text_tokens"), Some(LogValue::Int(4)));

        m.input_token_details.cached_tokens_details =
            Some(CachedTokenDetails { text_tokens: 3, ..Default::default() });
        let l = log_fields(&AgentMetrics::RealtimeModel(m)).unwrap();
        assert_eq!(get(&l, "input_cached_text_tokens"), Some(LogValue::Int(3)));
    }

    // Catches: rounding `ttfb`, the one duration the reference leaves raw.
    #[test]
    fn tts_ttfb_is_the_one_unrounded_duration() {
        let mut m = tk::tts();
        m.ttfb = 0.18437;
        m.audio_duration = 1.23456;
        let l = log_fields(&AgentMetrics::Tts(m)).unwrap();
        assert_eq!(get(&l, "ttfb"), Some(LogValue::Float(0.18437)));
        assert_eq!(get(&l, "audio_duration"), Some(LogValue::Float(1.23)));
    }

    // Catches: `(x * 100.0).round()`, which rounds the 0.125 tie away from zero.
    #[test]
    fn rounding_breaks_ties_to_even_like_python() {
        let mut m = tk::stt();
        m.audio_duration = 0.125;
        assert_eq!(
            get(&log_fields(&AgentMetrics::Stt(m)).unwrap(), "audio_duration"),
            Some(LogValue::Float(0.12))
        );
        // A non-tie, so "always round down" fails.
        let mut m = tk::stt();
        m.audio_duration = 0.135;
        assert_eq!(
            get(&log_fields(&AgentMetrics::Stt(m)).unwrap(), "audio_duration"),
            Some(LogValue::Float(0.14))
        );
    }

    // Catches: `is_some()` for the truthiness tests on playback latency and avatar timestamps.
    #[test]
    fn zero_avatar_measurements_are_omitted_not_logged_as_zero() {
        let mut m = tk::avatar();
        m.playback_latency = 0.0;
        m.session_started_time = Some(0.0);
        m.avatar_joined_time = Some(4.0);
        let l = log_fields(&AgentMetrics::Avatar(m)).unwrap();
        assert!(l.fields.is_empty(), "got {:?}", l.fields);
    }

    // Catches: two-digit avatar fields, or the join latency computed backwards.
    #[test]
    fn avatar_join_latency_is_joined_minus_started_at_three_digits() {
        let mut m = tk::avatar();
        m.session_started_time = Some(10.0);
        m.avatar_joined_time = Some(12.3456);
        m.playback_latency = 0.98765;
        let l = log_fields(&AgentMetrics::Avatar(m)).unwrap();
        assert_eq!(get(&l, "avatar_join_latency"), Some(LogValue::Float(2.346)));
        assert_eq!(get(&l, "playback_latency"), Some(LogValue::Float(0.988)));
    }

    // Catches: `on_user_turn_completed_delay` in the EOU line.
    #[test]
    fn the_eou_line_carries_only_its_two_delays() {
        let mut eou = tk::eou();
        eou.on_user_turn_completed_delay = 9.0;
        let l = log_fields(&AgentMetrics::Eou(eou)).unwrap();
        assert_eq!(l.message, "EOU metrics");
        assert_eq!(keys(&l), ["end_of_utterance_delay", "transcription_delay"]);
    }

    // Catches: an output-token key wired to the wrong source in the fifteen-field realtime block.
    #[test]
    fn realtime_output_keys_read_their_own_fields() {
        let mut m = tk::realtime();
        m.output_tokens = 10;
        m.total_tokens = 30;
        m.output_token_details =
            OutputTokenDetails { text_tokens: 6, audio_tokens: 3, image_tokens: 1 };
        let l = log_fields(&AgentMetrics::RealtimeModel(m)).unwrap();
        assert_eq!(get(&l, "output_tokens"), Some(LogValue::Int(10)));
        assert_eq!(get(&l, "output_text_tokens"), Some(LogValue::Int(6)));
        assert_eq!(get(&l, "output_audio_tokens"), Some(LogValue::Int(3)));
        assert_eq!(get(&l, "output_image_tokens"), Some(LogValue::Int(1)));
        assert_eq!(get(&l, "total_tokens"), Some(LogValue::Int(30)));
    }
}
