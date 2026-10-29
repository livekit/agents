//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `telemetry` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/telemetry/policy.rs.
type _core_TelemetryPolicyImpl = <crate::Selected as super::Implementations>::TelemetryPolicy;

fn _extract_telemetry_policy_export_action_held_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "held_count")
}

fn _extract_telemetry_policy_value_bytes_length(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "length")
}

fn _extract_telemetry_policy_value_bytes_depth(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "depth")
}

fn _extract_telemetry_policy_value_bytes_children(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<usize>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<usize>(&item, "children") }).collect::<PyResult<Vec<_>>>())
}

fn _extract_telemetry_policy_record_bytes_fields(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<usize>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<usize>(&item, "fields") }).collect::<PyResult<Vec<_>>>())
}

fn _extract_telemetry_policy_record_bytes_event_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "event_count")
}

fn _extract_telemetry_policy_record_bytes_link_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "link_count")
}

/// The host supplies SDK fields and lengths; shared policy decides what may be retained/exported.
#[pyclass(frozen, name = "_core_TelemetryPolicy", module = "livekit.agents._core")]
pub struct _core_TelemetryPolicy(::std::sync::Arc<_core_TelemetryPolicyImpl>);

#[pymethods]
impl _core_TelemetryPolicy {
    #[new]
    #[pyo3(signature = (allow_pii, capture_content, project_redaction))]
    fn new(allow_pii: bool, capture_content: bool, project_redaction: bool) -> PyResult<_core_TelemetryPolicy> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::new(allow_pii, capture_content, project_redaction);
        Ok(_core_TelemetryPolicy(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn redacted(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::redacted(&self.0);
        Ok(out)
    }

    /// The destination may export, hold, drop, refuse capacity or stop after recording is disabled.
    #[pyo3(signature = (registered, signal_enabled, job_open, held_count, recording_disabled))]
    fn export_action<'py>(&self, py: Python<'py>, registered: bool, signal_enabled: bool, job_open: bool, #[pyo3(from_py_with = _extract_telemetry_policy_export_action_held_count)] held_count: usize, recording_disabled: bool) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::export_action(&self.0, registered, signal_enabled, job_open, held_count, recording_disabled);
        Ok(crate::generated::records::telemetry::telemetry_export_action_to_py(py, out)?)
    }

    /// Whether to keep the field and, if present, the replacement for its value.
    #[pyo3(signature = (parent, key, destination))]
    fn attribute_action(&self, parent: ::std::option::Option<::std::string::String>, key: ::std::string::String, destination: bool) -> PyResult<(bool, ::std::option::Option<::std::string::String>)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::attribute_action(&self.0, parent, key, destination);
        Ok(out)
    }

    #[pyo3(signature = (name))]
    fn keep_event(&self, name: ::std::string::String) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::keep_event(&self.0, name);
        Ok(out)
    }

    /// A replacement body, or None to snapshot the original SDK value.
    #[pyo3(signature = (safe_body, body))]
    fn log_body(&self, safe_body: bool, body: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::log_body(&self.0, safe_body, body);
        Ok(out)
    }

    #[pyo3(signature = (is_error, description))]
    fn status_description(&self, is_error: bool, description: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::status_description(&self.0, is_error, description);
        Ok(out)
    }

    /// Conservative retained bytes for an SDK scalar or collection. Child charges are already
    /// bounded verdicts; unsupported kinds, depth and collection overflow refuse the record.
    #[pyo3(signature = (kind, length, depth, children))]
    fn value_bytes(&self, kind: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_telemetry_policy_value_bytes_length)] length: usize, #[pyo3(from_py_with = _extract_telemetry_policy_value_bytes_depth)] depth: usize, #[pyo3(from_py_with = _extract_telemetry_policy_value_bytes_children)] children: ::std::vec::Vec<usize>) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let kind = crate::generated::records::telemetry::telemetry_value_kind_from_py(&kind)?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::value_bytes(&self.0, kind, length, depth, children);
        Ok(out)
    }

    /// SDK record overhead plus field charges. Counts are supplied before traversing events/links.
    #[pyo3(signature = (kind, fields, event_count, link_count))]
    fn record_bytes(&self, kind: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_telemetry_policy_record_bytes_fields)] fields: ::std::vec::Vec<usize>, #[pyo3(from_py_with = _extract_telemetry_policy_record_bytes_event_count)] event_count: usize, #[pyo3(from_py_with = _extract_telemetry_policy_record_bytes_link_count)] link_count: usize) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let kind = crate::generated::records::telemetry::telemetry_record_kind_from_py(&kind)?;
        let out = <_core_TelemetryPolicyImpl as ::lk_agents_abi::telemetry::policy::TelemetryPolicy>::record_bytes(&self.0, kind, fields, event_count, link_count);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/policy.rs.
type _core_MetricSeriesImpl = <crate::Selected as super::Implementations>::MetricSeries;

/// The framework's metric series. Subject folds decide measurements; this refuses unbounded labels
/// and series before the host SDK creates a new one.
#[pyclass(name = "_core_MetricSeries", module = "livekit.agents._core")]
pub struct _core_MetricSeries(_core_MetricSeriesImpl);

#[pymethods]
impl _core_MetricSeries {
    #[new]
    fn new() -> PyResult<_core_MetricSeries> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_MetricSeriesImpl as ::lk_agents_abi::telemetry::policy::MetricSeries>::new();
        Ok(_core_MetricSeries(out))
    }

    #[pyo3(signature = (name, attributes))]
    fn admit(&mut self, name: ::std::string::String, attributes: ::std::vec::Vec<(::std::string::String, ::std::string::String)>) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_MetricSeriesImpl as ::lk_agents_abi::telemetry::policy::MetricSeries>::admit(&mut self.0, name, attributes);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/provider.rs.
type ProviderMetricsImpl = <crate::Selected as super::Implementations>::ProviderMetrics;

fn _extract_provider_metrics_stt_batch_metrics_start(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "start")
}

fn _extract_provider_metrics_stt_batch_metrics_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_provider_metrics_stt_batch_metrics_audio_durations(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<f64>> {
    obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<f64>(&item, "audio_durations") }).collect::<PyResult<Vec<_>>>())
}

fn _extract_provider_metrics_stt_batch_metrics_host(obj: &Bound<'_, PyAny>) -> PyResult<(f64, ::std::string::String, bool, ::std::string::String, ::std::string::String)> {
    (|| -> ::pyo3::PyResult<_> { let tuple = obj.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "host"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((::lk_agents_abi::python::number_from_py::<f64>(&item_0, "host"))?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<bool>())?, (item_3.extract::<::std::string::String>())?, (item_4.extract::<::std::string::String>())?,)) })()
}

fn _extract_provider_metrics_stt_stream_metrics_host(obj: &Bound<'_, PyAny>) -> PyResult<(f64, ::std::string::String, bool, ::std::string::String, ::std::string::String)> {
    (|| -> ::pyo3::PyResult<_> { let tuple = obj.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "host"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((::lk_agents_abi::python::number_from_py::<f64>(&item_0, "host"))?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<bool>())?, (item_3.extract::<::std::string::String>())?, (item_4.extract::<::std::string::String>())?,)) })()
}

/// `STT.recognize`'s metrics for one batch recognition.
#[pyfunction]
#[pyo3(signature = (event, start, now, audio_durations, host))]
pub fn stt_batch_metrics(event: ::lk_agents_abi::SpeechEvent, #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_start)] start: f64, #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_now)] now: f64, #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_audio_durations)] audio_durations: ::std::vec::Vec<f64>, #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_host)] host: (f64, ::std::string::String, bool, ::std::string::String, ::std::string::String)) -> PyResult<::lk_agents_abi::metrics::SttMetrics> {
    let out = <ProviderMetricsImpl as ::lk_agents_abi::telemetry::provider::ProviderMetrics>::stt_batch_metrics(event, start, now, audio_durations, host);
    Ok(out)
}

/// `RecognizeStream`'s metrics for one event, and whether it resets the retry count.
#[pyfunction]
#[pyo3(signature = (event, host))]
pub fn stt_stream_metrics(event: ::lk_agents_abi::SpeechEvent, #[pyo3(from_py_with = _extract_provider_metrics_stt_stream_metrics_host)] host: (f64, ::std::string::String, bool, ::std::string::String, ::std::string::String)) -> PyResult<(::std::option::Option<::lk_agents_abi::metrics::SttMetrics>, bool)> {
    let out = <ProviderMetricsImpl as ::lk_agents_abi::telemetry::provider::ProviderMetrics>::stt_stream_metrics(event, host).map_err(|e: ::lk_agents_abi::telemetry::provider::RecognitionUsageMissing| <::lk_agents_abi::telemetry::provider::RecognitionUsageMissing as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/provider.rs.
type LlmMetricStateImpl = <crate::Selected as super::Implementations>::LlmMetricState;

fn _extract_llm_metric_state_new_start(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "start")
}

fn _extract_llm_metric_state_observe_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_llm_metric_state_finish_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_llm_metric_state_finish_host(obj: &Bound<'_, PyAny>) -> PyResult<(f64, ::std::string::String, bool, ::std::string::String, ::std::string::String)> {
    (|| -> ::pyo3::PyResult<_> { let tuple = obj.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "host"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((::lk_agents_abi::python::number_from_py::<f64>(&item_0, "host"))?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<bool>())?, (item_3.extract::<::std::string::String>())?, (item_4.extract::<::std::string::String>())?,)) })()
}

/// `LLMStream`'s metric fold as the host drives it.
#[pyclass(name = "LlmMetricState", module = "livekit.agents._core")]
pub struct LlmMetricState(LlmMetricStateImpl);

#[pymethods]
impl LlmMetricState {
    #[new]
    #[pyo3(signature = (start))]
    fn new(#[pyo3(from_py_with = _extract_llm_metric_state_new_start)] start: f64) -> PyResult<LlmMetricState> {
        let out = <LlmMetricStateImpl as ::lk_agents_abi::telemetry::provider::LlmMetricState>::new(start);
        Ok(LlmMetricState(out))
    }

    /// Whether `chunk` is the first with a response.
    #[pyo3(signature = (now, chunk))]
    fn observe(&mut self, #[pyo3(from_py_with = _extract_llm_metric_state_observe_now)] now: f64, chunk: ::lk_agents_abi::LlmChunk) -> PyResult<bool> {
        let out = <LlmMetricStateImpl as ::lk_agents_abi::telemetry::provider::LlmMetricState>::observe(&mut self.0, now, chunk);
        Ok(out)
    }

    /// `None` for a failed attempt or one that yielded no chunk.
    #[pyo3(signature = (now, failed, host))]
    fn finish(&self, #[pyo3(from_py_with = _extract_llm_metric_state_finish_now)] now: f64, failed: bool, #[pyo3(from_py_with = _extract_llm_metric_state_finish_host)] host: (f64, ::std::string::String, bool, ::std::string::String, ::std::string::String)) -> PyResult<::std::option::Option<::lk_agents_abi::metrics::LlmMetrics>> {
        let out = <LlmMetricStateImpl as ::lk_agents_abi::telemetry::provider::LlmMetricState>::finish(&self.0, now, failed, host);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/provider.rs.
type TtsMetricStateImpl = <crate::Selected as super::Implementations>::TtsMetricState;

fn _extract_tts_metric_state_observe_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_tts_metric_state_observe_start(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "start")
}

fn _extract_tts_metric_state_finish_now(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "now")
}

fn _extract_tts_metric_state_finish_start(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "start")
}

fn _extract_tts_metric_state_finish_input_tokens(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "input_tokens")
}

fn _extract_tts_metric_state_finish_output_tokens(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    ::lk_agents_abi::python::number_from_py::<u32>(&obj, "output_tokens")
}

fn _extract_tts_metric_state_finish_acquire_time(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "acquire_time")
}

fn _extract_tts_metric_state_finish_host(obj: &Bound<'_, PyAny>) -> PyResult<(f64, ::std::string::String, bool, ::std::string::String, ::std::string::String)> {
    (|| -> ::pyo3::PyResult<_> { let tuple = obj.cast::<::pyo3::types::PyTuple>()?; if tuple.len() != 5 { return Err(::pyo3::exceptions::PyTypeError::new_err(format!("{}: expected a tuple of 5 items", "host"))); } let item_0 = tuple.get_item(0)?; let item_1 = tuple.get_item(1)?; let item_2 = tuple.get_item(2)?; let item_3 = tuple.get_item(3)?; let item_4 = tuple.get_item(4)?; Ok(((::lk_agents_abi::python::number_from_py::<f64>(&item_0, "host"))?, (item_1.extract::<::std::string::String>())?, (item_2.extract::<bool>())?, (item_3.extract::<::std::string::String>())?, (item_4.extract::<::std::string::String>())?,)) })()
}

/// The TTS streams' metric fold as the host drives it.
#[pyclass(name = "TtsMetricState", module = "livekit.agents._core")]
pub struct TtsMetricState(TtsMetricStateImpl);

#[pymethods]
impl TtsMetricState {
    #[new]
    #[pyo3(signature = (streamed))]
    fn new(streamed: bool) -> PyResult<TtsMetricState> {
        let out = <TtsMetricStateImpl as ::lk_agents_abi::telemetry::provider::TtsMetricState>::new(streamed);
        Ok(TtsMetricState(out))
    }

    /// A streamed segment's text, in the order its audio is finished.
    #[pyo3(signature = (text))]
    fn queue_text(&mut self, text: ::std::string::String) -> PyResult<()> {
        <TtsMetricStateImpl as ::lk_agents_abi::telemetry::provider::TtsMetricState>::queue_text(&mut self.0, text);
        Ok(())
    }

    /// Whether `audio` ends its segment.
    #[pyo3(signature = (now, start, audio))]
    fn observe(&mut self, #[pyo3(from_py_with = _extract_tts_metric_state_observe_now)] now: f64, #[pyo3(from_py_with = _extract_tts_metric_state_observe_start)] start: f64, audio: ::lk_agents_abi::SynthesizedAudio) -> PyResult<bool> {
        let out = <TtsMetricStateImpl as ::lk_agents_abi::telemetry::provider::TtsMetricState>::observe(&mut self.0, now, start, audio);
        Ok(out)
    }

    /// The record of a finished request or segment, then the figures reset.
    #[pyo3(signature = (now, start, failed, batch_text, input_tokens, output_tokens, acquire_time, connection_reused, host))]
    fn finish(&mut self, #[pyo3(from_py_with = _extract_tts_metric_state_finish_now)] now: f64, #[pyo3(from_py_with = _extract_tts_metric_state_finish_start)] start: f64, failed: bool, batch_text: ::std::string::String, #[pyo3(from_py_with = _extract_tts_metric_state_finish_input_tokens)] input_tokens: u32, #[pyo3(from_py_with = _extract_tts_metric_state_finish_output_tokens)] output_tokens: u32, #[pyo3(from_py_with = _extract_tts_metric_state_finish_acquire_time)] acquire_time: f64, connection_reused: bool, #[pyo3(from_py_with = _extract_tts_metric_state_finish_host)] host: (f64, ::std::string::String, bool, ::std::string::String, ::std::string::String)) -> PyResult<::std::option::Option<::lk_agents_abi::metrics::TtsMetrics>> {
        let out = <TtsMetricStateImpl as ::lk_agents_abi::telemetry::provider::TtsMetricState>::finish(&mut self.0, now, start, failed, batch_text, input_tokens, output_tokens, acquire_time, connection_reused, host);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
type TaggerImpl = <crate::Selected as super::Implementations>::Tagger;

/// Tag sessions with metadata for observability.
///
/// The Tagger allows adding string tags (key:value format) with optional
/// structured metadata to sessions. Tags and metadata are uploaded to
/// LiveKit Cloud at session end.
///
/// Example:
/// ```python
/// # Mark session as successful
/// ctx.tagger.success(reason="Task completed successfully")
///
/// # Mark session as failed
/// ctx.tagger.fail(reason="User hung up before completing booking")
///
/// # Add custom tags
/// ctx.tagger.add("voicemail:true")
/// ctx.tagger.add("language:es")
///
/// # Add tags with structured metadata
/// ctx.tagger.add(
///     "appointment:booked",
///     metadata={"slot_id": "abc123", "calendar": "cal.com"},
/// )
///
/// # Remove a tag
/// ctx.tagger.remove("voicemail:true")
/// ```
#[pyclass(name = "Tagger", module = "livekit.agents._core")]
pub struct Tagger(TaggerImpl);

#[pymethods]
impl Tagger {
    #[new]
    fn new() -> PyResult<Tagger> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::new(&env);
        Ok(Tagger(out))
    }

    /// Mark the session as successful; ``reason`` is kept apart from the tag.
    #[pyo3(signature = (reason = None))]
    fn success(&mut self, reason: ::std::option::Option<::std::string::String>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::success(&mut self.0, reason);
        Ok(())
    }

    /// Mark the session as failed; ``reason`` is kept apart from the tag.
    #[pyo3(signature = (reason = None))]
    fn fail(&mut self, reason: ::std::option::Option<::std::string::String>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::fail(&mut self.0, reason);
        Ok(())
    }

    /// Add a ``key:value`` tag, with optional structured metadata.
    #[pyo3(signature = (tag, *, metadata = None))]
    fn add(&mut self, tag: ::std::string::String, metadata: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        let metadata = metadata.map(|v| ::lk_agents_abi::python::value_from_py(&v, "metadata")).transpose()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::add(&mut self.0, tag, metadata);
        Ok(())
    }

    /// Remove a tag; removing an absent one does nothing.
    #[pyo3(signature = (tag))]
    fn remove(&mut self, tag: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::remove(&mut self.0, tag);
        Ok(())
    }

    /// The session's tags.
    #[getter]
    fn tags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::tags(&self.0);
        Ok(::pyo3::types::PyFrozenSet::new(py, out.into_iter().map(|x| PyResult::Ok(::pyo3::IntoPyObjectExt::into_bound_py_any(x, py)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    /// ``sorted(tagger.tags)``.
    fn sorted_tags(&self) -> PyResult<::std::vec::Vec<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::sorted_tags(&self.0);
        Ok(out)
    }

    /// ``(tag, metadata, epoch seconds)``, in insertion order.
    fn entries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::entries(&self.0);
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok({ let (x0, x1, x2,) = x; ::pyo3::types::PyTuple::new(py, [::pyo3::IntoPyObjectExt::into_bound_py_any(x0, py)?, match x1 { None => py.None().into_bound(py), Some(x) => ::lk_agents_abi::python::value_to_py(py, &x)? }, ::pyo3::IntoPyObjectExt::into_bound_py_any(x2, py)?])?.into_any() })).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    fn is_empty(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::is_empty(&self.0);
        Ok(out)
    }

    /// The judgments recorded by judge groups.
    #[getter]
    fn evaluations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::evaluations(&self.0);
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(crate::generated::records::telemetry::evaluation_record_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }

    /// ``"success"``, ``"fail"`` or ``None``.
    #[getter]
    fn outcome<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::outcome(&self.0);
        Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::telemetry::outcome_to_py(py, value)?)).transpose()?)
    }

    /// The reason given with the outcome.
    #[getter]
    fn outcome_reason(&self) -> PyResult<::std::option::Option<::std::string::String>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::outcome_reason(&self.0);
        Ok(out)
    }

    /// One judgment of a judge group's result: its ``lk.judge.<name>:<verdict>`` tag and record.
    #[pyo3(signature = (name, verdict, reasoning, instructions))]
    fn _core_record_judgment(&mut self, name: ::std::string::String, verdict: ::std::string::String, reasoning: ::std::string::String, instructions: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::_core_record_judgment(&mut self.0, name, verdict, reasoning, instructions);
        Ok(())
    }

    /// Whether the session has judgments or an outcome to upload.
    fn has_evals(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::has_evals(&self.0);
        Ok(out)
    }

    /// The session report's upload records, in the reference's order; none once uploads are
    /// ``disabled``. A raise in ``project_item`` (a chat item's span attribute) is re-raised once
    /// the records are built.
    #[pyo3(signature = (report, disabled, project_item, capture_content = false))]
    fn _core_session_records<'py>(&self, py: Python<'py>, report: Bound<'_, PyAny>, disabled: bool, project_item: Bound<'_, PyAny>, capture_content: bool) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let report = crate::generated::records::telemetry::session_report_input_from_py(&report)?;
        let project_item = crate::generated::telemetry::ChatItemProjectionProxy::from_host(&project_item)?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::_core_session_records(&self.0, report, disabled, project_item, capture_content).map_err(|e: ::lk_agents_abi::errors::HostError| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(::pyo3::types::PyList::new(py, out.into_iter().map(|x| PyResult::Ok(crate::generated::records::telemetry::log_record_wire_to_py(py, x)?)).collect::<PyResult<Vec<_>>>()?)?.into_any())
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
type UploadGateImpl = <crate::Selected as super::Implementations>::UploadGate;

fn _extract_upload_gate_is_disabled_response_status_code(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    ::lk_agents_abi::python::number_from_py::<u16>(&obj, "status_code")
}

fn _extract_upload_gate_is_disabled_response_body(obj: &Bound<'_, PyAny>) -> PyResult<::std::vec::Vec<u8>> {
    { if obj.is_instance_of::<::pyo3::types::PyBytes>() || obj.is_instance_of::<::pyo3::types::PyByteArray>() || obj.is_instance_of::<::pyo3::types::PyMemoryView>() { ::lk_agents_abi::python::owned_bytes(&obj, "body") } else { obj.try_iter().and_then(|items| items.map(|item| { let item = item?; ::lk_agents_abi::python::number_from_py::<u8>(&item, "body") }).collect::<PyResult<Vec<_>>>()) } }
}

/// Per-session upload gate: the "data recording is disabled" latch, one transition, one warning.
#[pyclass(name = "UploadGate", module = "livekit.agents._core")]
pub struct UploadGate(UploadGateImpl);

#[pymethods]
impl UploadGate {
    #[new]
    fn new() -> PyResult<UploadGate> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <UploadGateImpl as ::lk_agents_abi::telemetry::report::UploadGate>::new();
        Ok(UploadGate(out))
    }

    /// Called when a session re-initialises recording.
    fn reset(&mut self) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <UploadGateImpl as ::lk_agents_abi::telemetry::report::UploadGate>::reset(&mut self.0);
        Ok(())
    }

    #[getter]
    fn disabled(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <UploadGateImpl as ::lk_agents_abi::telemetry::report::UploadGate>::disabled(&self.0);
        Ok(out)
    }

    /// Returns ``True`` on the transition only, where the reference logs its warning.
    fn disable(&mut self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <UploadGateImpl as ::lk_agents_abi::telemetry::report::UploadGate>::disable(&mut self.0);
        Ok(out)
    }

    /// Whether an upload response is the project's "recording disabled" 401.
    #[staticmethod]
    #[pyo3(signature = (status_code, body))]
    fn is_disabled_response(#[pyo3(from_py_with = _extract_upload_gate_is_disabled_response_status_code)] status_code: u16, #[pyo3(from_py_with = _extract_upload_gate_is_disabled_response_body)] body: ::std::vec::Vec<u8>) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <UploadGateImpl as ::lk_agents_abi::telemetry::report::UploadGate>::is_disabled_response(status_code, body);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.
type ReportRecordsImpl = <crate::Selected as super::Implementations>::ReportRecords;

/// Whether recording is enabled for the job.
#[pyfunction]
#[pyo3(signature = (options))]
pub fn recording_enabled(options: ::lk_agents_abi::telemetry::report::RecordingOptions) -> PyResult<bool> {
    let out = <ReportRecordsImpl as ::lk_agents_abi::telemetry::report::ReportRecords>::recording_enabled(options);
    Ok(out)
}

/// One chat item's upload record; one at a time lets a host SDK emit a large report without
/// another whole-history vector under its interpreter lock. ``None`` once uploads are
/// ``disabled`` or the transcript is not recorded.
#[pyfunction]
#[pyo3(signature = (disabled, options, item, projected, capture_content, identity))]
pub fn _core_session_chat_record<'py>(py: Python<'py>, disabled: bool, options: ::lk_agents_abi::telemetry::report::RecordingOptions, item: ::lk_agents_abi::ChatItem, projected: Bound<'_, PyAny>, capture_content: bool, identity: (::std::string::String, ::std::string::String, ::std::string::String)) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    let projected = ::lk_agents_abi::python::value_from_py(&projected, "projected")?;
    let out = <ReportRecordsImpl as ::lk_agents_abi::telemetry::report::ReportRecords>::_core_session_chat_record(disabled, options, item, projected, capture_content, identity);
    Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::telemetry::log_record_wire_to_py(py, value)?)).transpose()?)
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report/projection.rs.
type _core_SessionReportProjectionImpl = <crate::Selected as super::Implementations>::SessionReportProjection;

fn _extract_session_report_projection_event_limit_items(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "items")
}

fn _extract_session_report_projection_accepts_event_retained_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "retained_bytes")
}

fn _extract_session_report_projection_accepts_event_event_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "event_bytes")
}

fn _extract_session_report_projection_project_event_retained_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "retained_bytes")
}

fn _extract_session_report_projection_event_record_timestamp(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    ::lk_agents_abi::python::number_from_py::<f64>(&obj, "timestamp")
}

#[pyclass(frozen, name = "_core_SessionReportProjection", module = "livekit.agents._core")]
pub struct _core_SessionReportProjection(::std::sync::Arc<_core_SessionReportProjectionImpl>);

#[pymethods]
impl _core_SessionReportProjection {
    #[new]
    #[pyo3(signature = (capture_content, options, project_redaction))]
    fn new(capture_content: bool, options: ::lk_agents_abi::telemetry::report::RecordingOptions, project_redaction: bool) -> PyResult<_core_SessionReportProjection> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::new(capture_content, &options, project_redaction);
        Ok(_core_SessionReportProjection(::std::sync::Arc::new(out)))
    }

    #[getter]
    fn content(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::content(&self.0);
        Ok(out)
    }

    #[getter]
    fn transcript(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::transcript(&self.0);
        Ok(out)
    }

    #[getter]
    fn audio(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::audio(&self.0);
        Ok(out)
    }

    /// Optional observations beyond this prefix are omitted without failing the transcript.
    #[pyo3(signature = (items))]
    fn event_limit(&self, #[pyo3(from_py_with = _extract_session_report_projection_event_limit_items)] items: usize) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::event_limit(&self.0, items);
        Ok(out)
    }

    /// Admission precedes host SDK serialization and native JSON extraction.
    #[pyo3(signature = (retained_bytes, event_bytes))]
    fn accepts_event(&self, #[pyo3(from_py_with = _extract_session_report_projection_accepts_event_retained_bytes)] retained_bytes: usize, #[pyo3(from_py_with = _extract_session_report_projection_accepts_event_event_bytes)] event_bytes: usize) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::accepts_event(&self.0, retained_bytes, event_bytes);
        Ok(out)
    }

    #[pyo3(signature = (report))]
    fn project<'py>(&self, py: Python<'py>, report: Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let report = crate::generated::records::telemetry::session_report_snapshot_from_py(&report)?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::project(&self.0, report);
        Ok(crate::generated::records::telemetry::session_report_snapshot_to_py(py, out)?)
    }

    #[pyo3(signature = (event, retained_bytes))]
    fn project_event<'py>(&self, py: Python<'py>, event: Bound<'_, PyAny>, #[pyo3(from_py_with = _extract_session_report_projection_project_event_retained_bytes)] retained_bytes: usize) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        ::lk_agents_abi::python::check_pid()?;
        let event = ::lk_agents_abi::python::value_from_py(&event, "event")?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::project_event(&self.0, event, retained_bytes);
        Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::telemetry::projected_report_event_to_py(py, value)?)).transpose()?)
    }

    #[pyo3(signature = (event, identity, timestamp, disabled))]
    fn event_record<'py>(&self, py: Python<'py>, event: Bound<'_, PyAny>, identity: (::std::string::String, ::std::string::String, ::std::string::String), #[pyo3(from_py_with = _extract_session_report_projection_event_record_timestamp)] timestamp: f64, disabled: bool) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        ::lk_agents_abi::python::check_pid()?;
        let event = crate::generated::records::telemetry::projected_report_event_from_py(&event)?;
        let out = <_core_SessionReportProjectionImpl as ::lk_agents_abi::telemetry::report::projection::SessionReportProjection>::event_record(&self.0, event, identity, timestamp, disabled);
        Ok(out.map(|value| Ok::<_, ::pyo3::PyErr>(crate::generated::records::telemetry::log_record_wire_to_py(py, value)?)).transpose()?)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/observation.rs.
type _core_EventRetentionImpl = <crate::Selected as super::Implementations>::EventRetention;

fn _extract_event_retention_admit_retained_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "retained_bytes")
}

fn _extract_event_retention_admit_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

fn _extract_event_retention_expire_now_ns(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "now_ns")
}

/// Bounded retention of a session report's host event snapshots. The native policy retains only
/// byte charges and supplied monotonic timestamps; the adapter owns independent model copies.
#[pyclass(name = "_core_EventRetention", module = "livekit.agents._core")]
pub struct _core_EventRetention(_core_EventRetentionImpl);

#[pymethods]
impl _core_EventRetention {
    #[new]
    #[pyo3(signature = (capture_content))]
    fn new(capture_content: bool) -> PyResult<_core_EventRetention> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::new(&env, capture_content).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_EventRetention(out))
    }

    /// Bind report access to the first job that starts the session. Ownership cannot transfer.
    #[pyo3(signature = (job_id))]
    fn claim(&mut self, job_id: ::std::string::String) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::claim(&mut self.0, job_id);
        Ok(())
    }

    /// Sessions started outside a job retain their existing application-owned access contract.
    #[pyo3(signature = (job_id))]
    fn allows_read(&self, job_id: ::std::string::String) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::allows_read(&self.0, job_id);
        Ok(out)
    }

    /// Immutable consent captured when this session's report retention was created.
    #[getter]
    fn capture_content(&self) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::capture_content(&self.0);
        Ok(out)
    }

    /// Reject content without creation-time consent, unknown schemas and live-object events.
    #[pyo3(signature = (kind, redaction))]
    fn accepts(&self, kind: ::std::string::String, redaction: bool) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::accepts(&self.0, kind, redaction);
        Ok(out)
    }

    /// Expire old entries, then admit only within the item and byte budgets. The adapter removes
    /// exactly the returned prefix before appending the newly admitted snapshot.
    #[pyo3(signature = (retained_bytes, now_ns))]
    fn admit(&mut self, #[pyo3(from_py_with = _extract_event_retention_admit_retained_bytes)] retained_bytes: usize, #[pyo3(from_py_with = _extract_event_retention_admit_now_ns)] now_ns: u64) -> PyResult<(usize, bool)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::admit(&mut self.0, retained_bytes, now_ns);
        Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn expire(&mut self, #[pyo3(from_py_with = _extract_event_retention_expire_now_ns)] now_ns: u64) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::expire(&mut self.0, now_ns);
        Ok(out)
    }

    #[getter]
    fn diagnostics(&self) -> PyResult<::std::vec::Vec<(::std::string::String, u64)>> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_EventRetentionImpl as ::lk_agents_abi::telemetry::observation::EventRetention>::diagnostics(&self.0);
        Ok(out)
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/transport.rs.
type TraceParentsImpl = <crate::Selected as super::Implementations>::TraceParents;

fn _extract_trace_parents__core_format_trace_parent_trace_id(obj: &Bound<'_, PyAny>) -> PyResult<u128> {
    ::lk_agents_abi::python::number_from_py::<u128>(&obj, "trace_id")
}

fn _extract_trace_parents__core_format_trace_parent_span_id(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    ::lk_agents_abi::python::number_from_py::<u64>(&obj, "span_id")
}

/// Wire values from the SDK's integer identifiers. Invalid parents are omitted.
#[pyfunction]
#[pyo3(signature = (trace_id, span_id, sampled))]
pub fn _core_format_trace_parent(#[pyo3(from_py_with = _extract_trace_parents__core_format_trace_parent_trace_id)] trace_id: u128, #[pyo3(from_py_with = _extract_trace_parents__core_format_trace_parent_span_id)] span_id: u64, sampled: bool) -> PyResult<::std::option::Option<::std::string::String>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <TraceParentsImpl as ::lk_agents_abi::telemetry::transport::TraceParents>::_core_format_trace_parent(trace_id, span_id, sampled);
    Ok(out)
}

/// SDK construction values from a validated parent. Invalid context never refuses the
/// request.
#[pyfunction]
#[pyo3(signature = (traceparent = None))]
pub fn _core_trace_parent(traceparent: ::std::option::Option<::std::string::String>) -> PyResult<::std::option::Option<(u128, u64, bool)>> {
    ::lk_agents_abi::python::check_pid()?;
    let out = <TraceParentsImpl as ::lk_agents_abi::telemetry::transport::TraceParents>::_core_trace_parent(traceparent);
    Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/upload.rs.
type _core_SessionReportUploadImpl = <crate::Selected as super::Implementations>::SessionReportUpload;

fn _extract_session_report_upload_upload_start_time_millis(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    ::lk_agents_abi::python::number_from_py::<i64>(&obj, "start_time_millis")
}

/// Credentials for the session recording endpoint and host OTLP exporters.
#[pyclass(frozen, name = "_core_SessionReportUpload", module = "livekit.agents._core")]
pub struct _core_SessionReportUpload(::std::sync::Arc<_core_SessionReportUploadImpl>);

#[pymethods]
impl _core_SessionReportUpload {
    #[new]
    #[pyo3(signature = (api_key, api_secret))]
    fn new(api_key: ::std::string::String, api_secret: ::std::string::String) -> PyResult<_core_SessionReportUpload> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::new(&env, api_key, api_secret).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_SessionReportUpload(::std::sync::Arc::new(out)))
    }

    /// A fresh LiveKit observability credential. No token policy lives in the host adapter.
    fn bearer(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::bearer(&self.0).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Returns false only when the project disabled recording. Export failures are reported once
    /// to the caller, which decides where to surface them without failing the conversation.
    #[pyo3(signature = (base_url, room_id, job_id, simulated, redaction_enabled, start_time_millis, transcript, audio_path, disabled))]
    fn upload<'py>(&self, py: Python<'py>, base_url: ::std::string::String, room_id: ::std::string::String, job_id: ::std::string::String, simulated: bool, redaction_enabled: bool, #[pyo3(from_py_with = _extract_session_report_upload_upload_start_time_millis)] start_time_millis: i64, transcript: ::std::option::Option<::std::string::String>, audio_path: ::std::option::Option<::std::string::String>, disabled: bool) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_SessionReportUploadImpl as ::lk_agents_abi::telemetry::upload::SessionReportUpload>::upload(&this, base_url, room_id, job_id, simulated, redaction_enabled, start_time_millis, transcript, audio_path, disabled)
                .await.map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/upload.rs.
type _core_PreparedTranscriptImpl = <crate::Selected as super::Implementations>::PreparedTranscript;

fn _extract_prepared_transcript_check_input_items(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "items")
}

fn _extract_prepared_transcript_check_input_item_bytes(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    ::lk_agents_abi::python::number_from_py::<usize>(&obj, "item_bytes")
}

/// Incremental report preparation. Host adapters admit one item's shape before conversion and
/// append it off their event loop; no whole-history extraction crosses a binding call.
#[pyclass(frozen, name = "_core_PreparedTranscript", module = "livekit.agents._core")]
pub struct _core_PreparedTranscript(::std::sync::Arc<_core_PreparedTranscriptImpl>);

#[pymethods]
impl _core_PreparedTranscript {
    #[new]
    fn new() -> PyResult<_core_PreparedTranscript> {
        ::lk_agents_abi::python::check_pid()?;
        let env = crate::process::env();
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::new(&env).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(_core_PreparedTranscript(::std::sync::Arc::new(out)))
    }

    /// Maximum items per append batch before the adapter returns control to its event loop.
    #[getter]
    fn input_batch_size(&self) -> PyResult<usize> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::input_batch_size(&self.0);
        Ok(out)
    }

    /// Which media parts belong to the transcript's native projection (image, audio).
    fn content_policy(&self) -> PyResult<(bool, bool)> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::content_policy(&self.0);
        Ok(out)
    }

    /// Validate host collection lengths and the shared retained-byte verdict before converting an
    /// item's fields. These scalar facts cannot allocate an unbounded native item.
    #[pyo3(signature = (items, item_bytes))]
    fn check_input(&self, #[pyo3(from_py_with = _extract_prepared_transcript_check_input_items)] items: usize, #[pyo3(from_py_with = _extract_prepared_transcript_check_input_item_bytes)] item_bytes: usize) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::check_input(&self.0, items, item_bytes).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    #[pyo3(signature = (item))]
    fn append(&self, item: ::lk_agents_abi::ChatItem) -> PyResult<()> {
        ::lk_agents_abi::python::check_pid()?;
        <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::append(&self.0, item).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(())
    }

    fn finish(&self) -> PyResult<::std::string::String> {
        ::lk_agents_abi::python::check_pid()?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::finish(&self.0).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Append one already projected event to the same bounded recording preparation.
    #[pyo3(signature = (event))]
    fn append_event(&self, event: Bound<'_, PyAny>) -> PyResult<bool> {
        ::lk_agents_abi::python::check_pid()?;
        let event = ::lk_agents_abi::python::value_from_py(&event, "event")?;
        let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::append_event(&self.0, event).map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
        Ok(out)
    }

    /// Serialize and write a console report without converting its history back into host objects.
    #[pyo3(signature = (report, path))]
    fn finish_report<'py>(&self, py: Python<'py>, report: Bound<'_, PyAny>, path: ::std::string::String) -> PyResult<Bound<'py, PyAny>> {
        ::lk_agents_abi::python::check_pid()?;
        let report = crate::generated::records::telemetry::session_report_snapshot_from_py(&report)?;
        let this = ::std::sync::Arc::clone(&self.0);
        crate::host::future_into_py(py, async move {
            let out = <_core_PreparedTranscriptImpl as ::lk_agents_abi::telemetry::upload::PreparedTranscript>::finish_report(&this, report, path)
                .await.map_err(|e: ::lk_agents_abi::telemetry::upload::UploadError| <::lk_agents_abi::telemetry::upload::UploadError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?;
            Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any(out, py))
        })
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report.rs.

/// [`::lk_agents_abi::telemetry::report::ChatItemProjection`] over a host `collections.abc.Callable`.
#[derive(Clone)]
pub struct ChatItemProjectionProxy {
    host: ::std::sync::Arc<Py<PyAny>>,
}

impl ChatItemProjectionProxy {
    /// A registered native class as itself, any other host object behind this proxy.
    pub fn from_host(obj: &Bound<'_, PyAny>) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::telemetry::report::ChatItemProjection>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::telemetry::report::ChatItemProjection>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, None)?))
    }

    /// A host object a call returned, its calls run in a copy of the context that call ran in.
    pub fn in_context(obj: &Bound<'_, PyAny>, context: &::lk_agents_abi::utils::host_ref::HostRef) -> PyResult<::std::sync::Arc<dyn ::lk_agents_abi::telemetry::report::ChatItemProjection>> {
        if let Some(native) = ::lk_agents_abi::python::native::<dyn ::lk_agents_abi::telemetry::report::ChatItemProjection>(obj) {
            return Ok(native);
        }
        Ok(::std::sync::Arc::new(Self::take(obj, Some(context))?))
    }

    /// A host object the core owns (`Box<dyn ..>`), always behind this proxy.
    pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::telemetry::report::ChatItemProjection>> {
        Ok(::std::boxed::Box::new(Self::take(obj, None)?))
    }

    fn take(obj: &Bound<'_, PyAny>, context: ::std::option::Option<&::lk_agents_abi::utils::host_ref::HostRef>) -> PyResult<Self> {
        let _ = context;
        Ok(ChatItemProjectionProxy { host: ::std::sync::Arc::new(obj.clone().unbind()) })
    }
}

::lk_agents_abi::python::inventory::submit! {
    ::lk_agents_abi::python::DeclaredProxy::of::<dyn ::lk_agents_abi::telemetry::report::ChatItemProjection>(|obj| ChatItemProjectionProxy::from_host(obj).map(|p| ::std::boxed::Box::new(p) as ::std::boxed::Box<dyn ::std::any::Any>))
}

impl ::lk_agents_abi::telemetry::report::ChatItemProjection for ChatItemProjectionProxy {
    fn project(&self, item: ::lk_agents_abi::ChatItem) -> ::std::result::Result<::serde_json::Value, ::lk_agents_abi::errors::HostError> {
        let host = ::std::sync::Arc::clone(&self.host);
        let item_arg = item;
        let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ::std::boxed::Box::new(move |py| {
                let args: [::pyo3::Bound<'_, ::pyo3::PyAny>; 1] = [::pyo3::IntoPyObjectExt::into_bound_py_any(item_arg, py)?];
                Ok((host.bind(py).getattr("__call__")?, ::pyo3::types::PyTuple::new(py, args)?))
            }));
        ::pyo3::Python::attach(|py| {
            let out = match out {
                Ok(out) => out,
                Err(e) => {
                    return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));
                }
            };
            let o = out.bind(py);
            let out = (::lk_agents_abi::python::value_from_py(&o, "ChatItemProjection.__call__")).map_err(|e| <::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e))?;
            Ok(out)
        })
    }
}
