//! `telemetry/otel_metrics.py`'s decisions. Gates differ and must not be collapsed: usage is
//! truthiness (a DELTA zero would still be a data point), turns are key presence, `_model_attrs`
//! presence outside and truthiness inside. `tests/test_otel_metrics_seam.py` checks every record
//! names one of the host's instruments, as the right kind.

use lk_agents_abi::MetricsReport;
use lk_agents_abi::metrics::base::Metadata;
use lk_agents_abi::MetricsMetadata;

use crate::metrics::base::AgentMetrics;

/// A counter takes `.add`, a histogram `.record`.
#[crate::host(wire_strs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentKind {
    Counter,
    Histogram,
}

/// Token counts stay integers: the exported data point type follows the first value seen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OtelValue {
    Int(u64),
    Float(f64),
}

/// `attributes` keeps the reference's insertion order.
#[derive(Debug, Clone, PartialEq)]
pub struct OtelRecord {
    pub instrument: &'static str,
    pub kind: InstrumentKind,
    pub value: OtelValue,
    pub attributes: Vec<(&'static str, String)>,
}

impl OtelRecord {
    /// `int` and `float` stay distinct: an OTel series takes its type from the first value.
    pub fn wire(self) -> (&'static str, &'static str, serde_json::Value, serde_json::Map<String, serde_json::Value>) {
        let value = match self.value {
            OtelValue::Int(i) => i.into(),
            OtelValue::Float(f) => f.into(),
        };
        let attrs = self.attributes.into_iter().map(|(k, v)| (k.to_owned(), v.into())).collect();
        (self.instrument, self.kind.as_str(), value, attrs)
    }

    fn counter(
        instrument: &'static str,
        value: OtelValue,
        attributes: Vec<(&'static str, String)>,
    ) -> Self {
        OtelRecord { instrument, kind: InstrumentKind::Counter, value, attributes }
    }

    fn histogram(instrument: &'static str, value: f64, attributes: Vec<(&'static str, String)>) -> Self {
        OtelRecord {
            instrument,
            kind: InstrumentKind::Histogram,
            value: OtelValue::Float(value),
            attributes,
        }
    }
}

// ---------------------------------------------------------------------------- instrument names

// Not `pub`: [`INSTRUMENTS`] is the one public surface. `otel_metrics.py:36`.
const TURN_E2E_LATENCY: &str = "lk.agents.turn.e2e_latency";
const TURN_LLM_TTFT: &str = "lk.agents.turn.llm_ttft";
const TURN_TTS_TTFB: &str = "lk.agents.turn.tts_ttfb";
const TURN_TRANSCRIPTION_DELAY: &str = "lk.agents.turn.transcription_delay";
const TURN_END_OF_TURN_DELAY: &str = "lk.agents.turn.end_of_turn_delay";
const TURN_ON_USER_TURN_COMPLETED_DELAY: &str = "lk.agents.turn.on_user_turn_completed_delay";

// `otel_metrics.py:68`.
const LLM_INPUT_TOKENS: &str = "lk.agents.usage.llm_input_tokens";
const LLM_INPUT_CACHED_TOKENS: &str = "lk.agents.usage.llm_input_cached_tokens";
const LLM_OUTPUT_TOKENS: &str = "lk.agents.usage.llm_output_tokens";
const LLM_INPUT_AUDIO_TOKENS: &str = "lk.agents.usage.llm_input_audio_tokens";
const LLM_INPUT_TEXT_TOKENS: &str = "lk.agents.usage.llm_input_text_tokens";
const LLM_OUTPUT_AUDIO_TOKENS: &str = "lk.agents.usage.llm_output_audio_tokens";
const LLM_OUTPUT_TEXT_TOKENS: &str = "lk.agents.usage.llm_output_text_tokens";
const LLM_SESSION_DURATION: &str = "lk.agents.usage.llm_session_duration";
const TTS_CHARACTERS: &str = "lk.agents.usage.tts_characters";
const TTS_AUDIO_DURATION: &str = "lk.agents.usage.tts_audio_duration";
const STT_AUDIO_DURATION: &str = "lk.agents.usage.stt_audio_duration";
const INTERRUPTION_NUM_REQUESTS: &str = "lk.agents.usage.interruption_num_requests";

// `otel_metrics.py:90`.
const CONNECTION_ACQUIRE_TIME: &str = "lk.agents.connection.acquire_time";

pub const INSTRUMENTS: &[(&str, InstrumentKind)] = &[
    (TURN_E2E_LATENCY, InstrumentKind::Histogram),
    (TURN_LLM_TTFT, InstrumentKind::Histogram),
    (TURN_TTS_TTFB, InstrumentKind::Histogram),
    (TURN_TRANSCRIPTION_DELAY, InstrumentKind::Histogram),
    (TURN_END_OF_TURN_DELAY, InstrumentKind::Histogram),
    (TURN_ON_USER_TURN_COMPLETED_DELAY, InstrumentKind::Histogram),
    (LLM_INPUT_TOKENS, InstrumentKind::Counter),
    (LLM_INPUT_CACHED_TOKENS, InstrumentKind::Counter),
    (LLM_OUTPUT_TOKENS, InstrumentKind::Counter),
    (LLM_INPUT_AUDIO_TOKENS, InstrumentKind::Counter),
    (LLM_INPUT_TEXT_TOKENS, InstrumentKind::Counter),
    (LLM_OUTPUT_AUDIO_TOKENS, InstrumentKind::Counter),
    (LLM_OUTPUT_TEXT_TOKENS, InstrumentKind::Counter),
    (LLM_SESSION_DURATION, InstrumentKind::Counter),
    (TTS_CHARACTERS, InstrumentKind::Counter),
    (TTS_AUDIO_DURATION, InstrumentKind::Counter),
    (STT_AUDIO_DURATION, InstrumentKind::Counter),
    (INTERRUPTION_NUM_REQUESTS, InstrumentKind::Counter),
    (CONNECTION_ACQUIRE_TIME, InstrumentKind::Histogram),
];

// ---------------------------------------------------------------------------- attributes

/// `otel_metrics.py:99`: provider then name; an empty name is dropped, not emitted as `""`.
fn model_attrs(metadata: Option<&Metadata>) -> Vec<(&'static str, String)> {
    let mut attrs = Vec::new();
    let Some(md) = metadata else { return attrs };
    if let Some(p) = md.model_provider.as_ref().filter(|s| !s.is_empty()) {
        attrs.push(("model_provider", p.clone()));
    }
    if let Some(n) = md.model_name.as_ref().filter(|s| !s.is_empty()) {
        attrs.push(("model_name", n.clone()));
    }
    attrs
}

/// `otel_metrics.py:112`: key presence, so a present empty string is kept. Name then provider, the
/// reverse of `model_attrs`.
fn turn_metadata_attrs(metadata: Option<&MetricsMetadata>) -> Vec<(&'static str, String)> {
    let mut attrs = Vec::new();
    let Some(md) = metadata else { return attrs };
    if let Some(n) = md.model_name.as_ref() {
        attrs.push(("model_name", n.clone()));
    }
    if let Some(p) = md.model_provider.as_ref() {
        attrs.push(("model_provider", p.clone()));
    }
    attrs
}

// ---------------------------------------------------------------------------- the two folds

/// `otel_metrics.py:145`. A zero produces no record; the connection block is gated on
/// `acquire_time > 0` (strict, so a negative clock delta is not recorded).
pub fn usage_records(ev: &AgentMetrics) -> Vec<OtelRecord> {
    let mut out = Vec::new();
    let attrs = || model_attrs(ev.metadata());

    let mut count = |instrument: &'static str, n: u32| {
        if n != 0 {
            out.push(OtelRecord::counter(instrument, OtelValue::Int(u64::from(n)), attrs()));
        }
    };

    // Each arm's seconds counter comes after its token counts.
    let seconds = match ev {
        AgentMetrics::Llm(m) => {
            count(LLM_INPUT_TOKENS, m.prompt_tokens);
            count(LLM_INPUT_CACHED_TOKENS, m.prompt_cached_tokens);
            count(LLM_OUTPUT_TOKENS, m.completion_tokens);
            None
        }
        AgentMetrics::RealtimeModel(m) => {
            count(LLM_INPUT_TOKENS, m.input_tokens);
            count(LLM_INPUT_CACHED_TOKENS, m.input_token_details.cached_tokens);
            count(LLM_OUTPUT_TOKENS, m.output_tokens);
            count(LLM_INPUT_AUDIO_TOKENS, m.input_token_details.audio_tokens);
            count(LLM_INPUT_TEXT_TOKENS, m.input_token_details.text_tokens);
            count(LLM_OUTPUT_AUDIO_TOKENS, m.output_token_details.audio_tokens);
            count(LLM_OUTPUT_TEXT_TOKENS, m.output_token_details.text_tokens);
            Some((LLM_SESSION_DURATION, m.session_duration))
        }
        AgentMetrics::Tts(m) => {
            count(TTS_CHARACTERS, m.characters_count);
            Some((TTS_AUDIO_DURATION, m.audio_duration))
        }
        AgentMetrics::Stt(m) => Some((STT_AUDIO_DURATION, m.audio_duration)),
        AgentMetrics::Interruption(m) => {
            count(INTERRUPTION_NUM_REQUESTS, m.num_requests);
            None
        }
        // VAD, EOU, EOT-inference and avatar metrics produce nothing, as in the reference.
        _ => None,
    };
    if let Some((instrument, secs)) = seconds.filter(|&(_, secs)| secs != 0.0) {
        out.push(OtelRecord::counter(instrument, OtelValue::Float(secs), attrs()));
    }

    // `connection_reused` is `str(True).lower()`, a string attribute, not a bool.
    let conn = match ev {
        AgentMetrics::Stt(m) => Some((m.acquire_time, m.connection_reused)),
        AgentMetrics::Tts(m) => Some((m.acquire_time, m.connection_reused)),
        AgentMetrics::RealtimeModel(m) => Some((m.acquire_time, m.connection_reused)),
        _ => None,
    };
    if let Some((acquire_time, reused)) = conn {
        if acquire_time > 0.0 {
            let mut conn_attrs = model_attrs(ev.metadata());
            conn_attrs.push(("connection_reused", if reused { "true" } else { "false" }.to_owned()));
            out.push(OtelRecord::histogram(CONNECTION_ACQUIRE_TIME, acquire_time, conn_attrs));
        }
    }

    out
}

/// `_record_turn_metrics` (`otel_metrics.py:124`), as the list of histogram observations.
///
/// Key presence: a measured `0.0` is recorded. All three user-turn delays carry `stt_metadata`,
/// including `on_user_turn_completed_delay`; reproduced, not corrected.
/// `_record_turn_metrics(report)` minus the recording, as [`OtelRecord::wire`] tuples. `None` and
/// `{}` yield nothing; a present `0.0` yields a record.
#[crate::host(export(name = "otel_turn_records"))]
pub fn turn_records_wire(report: Option<MetricsReport>) -> Vec<(&'static str, &'static str, serde_json::Value, serde_json::Map<String, serde_json::Value>)> {
    turn_records(&report.unwrap_or_default()).into_iter().map(OtelRecord::wire).collect()
}

pub fn turn_records(report: &MetricsReport) -> Vec<OtelRecord> {
    let llm = turn_metadata_attrs(report.llm_metadata.as_ref());
    let tts = turn_metadata_attrs(report.tts_metadata.as_ref());
    let stt = turn_metadata_attrs(report.stt_metadata.as_ref());

    let mut out = Vec::new();
    let mut record = |instrument: &'static str, value: Option<f64>, attrs: &Vec<(&'static str, String)>| {
        if let Some(v) = value {
            out.push(OtelRecord::histogram(instrument, v, attrs.clone()));
        }
    };

    record(TURN_E2E_LATENCY, report.e2e_latency, &llm);
    record(TURN_LLM_TTFT, report.llm_node_ttft, &llm);
    record(TURN_TTS_TTFB, report.tts_node_ttfb, &tts);
    record(TURN_TRANSCRIPTION_DELAY, report.transcription_delay, &stt);
    record(TURN_END_OF_TURN_DELAY, report.end_of_turn_delay, &stt);
    record(TURN_ON_USER_TURN_COMPLETED_DELAY, report.on_user_turn_completed_delay, &stt);
    out
}

/// `collect_usage(ev)` minus the recording: the counter increments it would make. Empty for
/// unhandled types, unknown types and honest zeros alike (the reference gates on truthiness).
/// `otel_turn_records` is generated.
#[cfg(feature = "python")]
#[pyo3::pyfunction]
#[allow(clippy::type_complexity)]
pub fn otel_usage_records<'py>(
    py: pyo3::Python<'py>,
    metrics: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::PyResult<Vec<(&'static str, &'static str, pyo3::Bound<'py, pyo3::PyAny>, pyo3::Bound<'py, pyo3::types::PyDict>)>> {
    let Some(record) = AgentMetrics::extract_known(metrics)? else { return Ok(Vec::new()) };
    usage_records(&record)
        .into_iter()
        .map(|r| {
            let (instrument, kind, value, attrs) = r.wire();
            Ok((instrument, kind, crate::python::value_to_py(py, &value)?, crate::python::extra_to_py(py, &attrs)?))
        })
        .collect()
}

#[cfg(feature = "python")]
crate::register_host!(fn otel_usage_records);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::testkit as tk;

    fn names(records: &[OtelRecord]) -> Vec<&'static str> {
        records.iter().map(|r| r.instrument).collect()
    }

    // Catches: gating `collect_usage` on presence instead of truthiness.
    #[test]
    fn a_genuine_zero_token_count_records_nothing() {
        let mut m = tk::llm();
        m.prompt_tokens = 0;
        m.completion_tokens = 4;
        let out = usage_records(&AgentMetrics::Llm(m));
        assert_eq!(names(&out), vec![LLM_OUTPUT_TOKENS]);
    }

    // Catches: gating `_record_turn_metrics` on truthiness instead of key presence.
    #[test]
    fn a_measured_zero_latency_is_still_recorded() {
        let report = MetricsReport { e2e_latency: Some(0.0), ..MetricsReport::default() };
        let out = turn_records(&report);
        assert_eq!(names(&out), vec![TURN_E2E_LATENCY]);
        assert_eq!(out[0].value, OtelValue::Float(0.0));
    }

    // Catches: emitting `model_name=""` for a provider that reports an empty name.
    #[test]
    fn an_empty_model_name_is_dropped_but_an_empty_metadata_is_not_invented() {
        let md = Metadata { model_name: Some(String::new()), model_provider: Some("openai".into()) };
        assert_eq!(model_attrs(Some(&md)), vec![("model_provider", "openai".to_owned())]);
        assert_eq!(model_attrs(None), Vec::new());
    }

    // Catches: reading `MetricsMetadata` with `Metadata`'s truthiness rule.
    #[test]
    fn turn_metadata_keeps_a_present_empty_string() {
        let md = MetricsMetadata { model_name: Some(String::new()), model_provider: None };
        assert_eq!(turn_metadata_attrs(Some(&md)), vec![("model_name", String::new())]);
    }

    // Catches: making the connection block another `elif`.
    #[test]
    fn connection_timing_is_additive_not_an_alternative() {
        let mut m = tk::tts();
        m.characters_count = 12;
        m.acquire_time = 0.25;
        m.connection_reused = true;
        let out = usage_records(&AgentMetrics::Tts(m));
        assert_eq!(names(&out), vec![TTS_CHARACTERS, CONNECTION_ACQUIRE_TIME]);
        assert_eq!(
            out[1].attributes,
            vec![("connection_reused", "true".to_owned())]
        );
    }

    // Catches: spelling the connection gate as truthiness (a negative acquire time is truthy).
    #[test]
    fn a_negative_acquire_time_is_not_a_latency() {
        let mut m = tk::stt();
        m.acquire_time = -1.0;
        assert_eq!(usage_records(&AgentMetrics::Stt(m)), Vec::new());
    }

    // Catches: pairing a turn latency with the wrong metadata.
    #[test]
    fn the_user_turn_callback_delay_carries_stt_metadata() {
        let report = MetricsReport {
            on_user_turn_completed_delay: Some(0.5),
            stt_metadata: Some(MetricsMetadata {
                model_name: Some("nova".into()),
                model_provider: None,
            }),
            llm_metadata: Some(MetricsMetadata {
                model_name: Some("gpt".into()),
                model_provider: None,
            }),
            ..MetricsReport::default()
        };
        let out = turn_records(&report);
        assert_eq!(out[0].attributes, vec![("model_name", "nova".to_owned())]);
    }

    // Catches: an instrument named by a fold and missing from `INSTRUMENTS`.
    #[test]
    fn every_emitted_instrument_is_declared() {
        let declared: Vec<&str> = INSTRUMENTS.iter().map(|(n, _)| *n).collect();

        let mut realtime = tk::realtime();
        realtime.input_tokens = 1;
        realtime.output_tokens = 1;
        realtime.session_duration = 1.0;
        realtime.acquire_time = 1.0;
        realtime.input_token_details.cached_tokens = 1;
        realtime.input_token_details.audio_tokens = 1;
        realtime.input_token_details.text_tokens = 1;
        realtime.output_token_details.audio_tokens = 1;
        realtime.output_token_details.text_tokens = 1;
        let mut tts = tk::tts();
        tts.characters_count = 1;
        tts.audio_duration = 1.0;
        let mut stt = tk::stt();
        stt.audio_duration = 1.0;
        let mut interruption = tk::interruption();
        interruption.num_requests = 1;
        let mut llm = tk::llm();
        llm.prompt_tokens = 1;
        llm.prompt_cached_tokens = 1;
        llm.completion_tokens = 1;

        let mut emitted: Vec<&str> = Vec::new();
        for ev in [
            AgentMetrics::Llm(llm),
            AgentMetrics::RealtimeModel(realtime),
            AgentMetrics::Tts(tts),
            AgentMetrics::Stt(stt),
            AgentMetrics::Interruption(interruption),
        ] {
            emitted.extend(names(&usage_records(&ev)));
        }
        emitted.extend(names(&turn_records(&MetricsReport {
            started_speaking_at: None,
            stopped_speaking_at: None,
            transcription_delay: Some(0.0),
            end_of_turn_delay: Some(0.0),
            on_user_turn_completed_delay: Some(0.0),
            llm_node_ttft: Some(0.0),
            llm_node_tps: None,
            llm_node_ttfs: None,
            tts_node_ttfb: Some(0.0),
            playback_latency: None,
            e2e_latency: Some(0.0),
            provider_request_ids: None,
            llm_metadata: None,
            tts_metadata: None,
            stt_metadata: None,
        })));

        for name in &emitted {
            assert!(declared.contains(name), "{name} is emitted but not in INSTRUMENTS");
        }
        for (name, _) in INSTRUMENTS {
            assert!(emitted.contains(name), "{name} is declared but nothing emits it");
        }
    }
}
