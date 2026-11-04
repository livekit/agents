//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `metrics` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/metrics/mod.rs.
type RuntimeLoadImpl =
    <crate::generated::implementations::Selected as super::Implementations>::RuntimeLoad;
type RuntimeLoadRuntimeBusySecondsOutput = ::std::vec::Vec<f64>;

/// Each async worker's total busy seconds so far; empty before the runtime started.
#[pyfunction]
pub fn runtime_busy_seconds(py: Python<'_>) -> PyResult<RuntimeLoadRuntimeBusySecondsOutput> {
    let out =
        py.detach(<RuntimeLoadImpl as ::lk_agents_abi::metrics::RuntimeLoad>::runtime_busy_seconds);
    ::pyo3::PyResult::Ok(out)
}

// Declared in rust/crates/lk-agents-abi/src/metrics/mod.rs.
type MetricsRecordsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::MetricsRecords;

/// `log_metrics(metrics)` minus the logging, or `None` for do not log (`VADMetrics` and
/// `EOTInferenceMetrics` have no branch in the reference).
#[pyfunction]
#[pyo3(signature = (metrics))]
pub fn metrics_log_fields<'py>(
    py: Python<'py>,
    metrics: Bound<'_, PyAny>,
) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
    crate::py_calls::check_pid()?;
    let metrics = crate::generated::records::metrics::agent_metrics_from_py(metrics.as_any())?;
    let out = py.detach(|| {
        <MetricsRecordsImpl as ::lk_agents_abi::metrics::MetricsRecords>::metrics_log_fields(
            metrics,
        )
    });
    out.map(|value| crate::generated::records::metrics::metrics_log_to_py(py, value)).transpose()
}

// Declared in rust/crates/lk-agents-abi/src/metrics/provider.rs.
type ProviderMetricsImpl =
    <crate::generated::implementations::Selected as super::Implementations>::ProviderMetrics;

fn _extract_provider_metrics_stt_batch_metrics_start(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderMetricsSttBatchMetricsStartInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "start")
}
type ProviderMetricsSttBatchMetricsStartInput = f64;

fn _extract_provider_metrics_stt_batch_metrics_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderMetricsSttBatchMetricsNowInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "now")
}
type ProviderMetricsSttBatchMetricsNowInput = f64;

fn _extract_provider_metrics_stt_batch_metrics_audio_durations(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderMetricsSttBatchMetricsAudioDurationsInput> {
    obj.try_iter().and_then(|items| {
        items
            .map(|item| {
                let item = item?;
                crate::py_values::number_from_py::<f64>(item.as_any(), "audio_durations")
            })
            .collect::<PyResult<Vec<_>>>()
    })
}
type ProviderMetricsSttBatchMetricsAudioDurationsInput = ::std::vec::Vec<f64>;

fn _extract_provider_metrics_stt_batch_metrics_timestamp(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderMetricsSttBatchMetricsTimestampInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "timestamp")
}
type ProviderMetricsSttBatchMetricsTimestampInput = f64;

fn _extract_provider_metrics_stt_stream_metrics_timestamp(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderMetricsSttStreamMetricsTimestampInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "timestamp")
}
type ProviderMetricsSttStreamMetricsTimestampInput = f64;

/// `STT.recognize`'s metrics for one batch recognition, stamped at `timestamp` (unix seconds)
/// with the provider's label, model and provider name.
#[pyfunction]
#[pyo3(signature = (event, start, now, audio_durations, timestamp, label, model, provider))]
pub fn stt_batch_metrics<'py>(
    py: Python<'py>,
    event: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_start)]
    start: ProviderMetricsSttBatchMetricsStartInput,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_now)]
    now: ProviderMetricsSttBatchMetricsNowInput,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_audio_durations)]
    audio_durations: ProviderMetricsSttBatchMetricsAudioDurationsInput,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_timestamp)]
    timestamp: ProviderMetricsSttBatchMetricsTimestampInput,
    label: ::std::string::String,
    model: ::std::string::String,
    provider: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    let event = crate::generated::records::stt::speech_event_from_py(event.as_any())?;
    let out = py.detach(|| <ProviderMetricsImpl as ::lk_agents_abi::metrics::provider::ProviderMetrics>::stt_batch_metrics(event, start, now, audio_durations, timestamp, label, model, provider));
    crate::generated::records::metrics::stt_metrics_to_py(py, out)
}

/// `RecognizeStream`'s metrics for one event, and whether it resets the retry count.
#[pyfunction]
#[pyo3(signature = (event, timestamp, label, model, provider))]
pub fn stt_stream_metrics<'py>(
    py: Python<'py>,
    event: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_stream_metrics_timestamp)]
    timestamp: ProviderMetricsSttStreamMetricsTimestampInput,
    label: ::std::string::String,
    model: ::std::string::String,
    provider: ::std::string::String,
) -> PyResult<Bound<'py, PyAny>> {
    let event = crate::generated::records::stt::speech_event_from_py(event.as_any())?;
    let out = py.detach(|| <ProviderMetricsImpl as ::lk_agents_abi::metrics::provider::ProviderMetrics>::stt_stream_metrics(event, timestamp, label, model, provider)).map_err(|e: ::lk_agents_abi::metrics::provider::RecognitionUsageMissing| <::lk_agents_abi::metrics::provider::RecognitionUsageMissing as crate::py_errors::HostException>::to_py_err(&e))?;
    ::pyo3::PyResult::Ok({
        let (x0, x1) = out;
        ::pyo3::types::PyTuple::new(
            py,
            [
                match x0 {
                    None => py.None().into_bound(py),
                    Some(x) => crate::generated::records::metrics::stt_metrics_to_py(py, x)?,
                },
                ::pyo3::IntoPyObjectExt::into_bound_py_any(x1, py)?,
            ],
        )?
        .into_any()
    })
}

// Declared in rust/crates/lk-agents-abi/src/metrics/provider.rs.
type LlmMetricStateImpl =
    <crate::generated::implementations::Selected as super::Implementations>::LlmMetricState;

fn _extract_llm_metric_state_new_start(
    obj: &Bound<'_, PyAny>,
) -> PyResult<LlmMetricStateNewStartInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "start")
}
type LlmMetricStateNewStartInput = f64;

fn _extract_llm_metric_state_observe_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<LlmMetricStateObserveNowInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "now")
}
type LlmMetricStateObserveNowInput = f64;

fn _extract_llm_metric_state_finish_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<LlmMetricStateFinishNowInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "now")
}
type LlmMetricStateFinishNowInput = f64;

fn _extract_llm_metric_state_finish_timestamp(
    obj: &Bound<'_, PyAny>,
) -> PyResult<LlmMetricStateFinishTimestampInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "timestamp")
}
type LlmMetricStateFinishTimestampInput = f64;

/// `LLMStream`'s metric fold as the host drives it.
#[pyclass(name = "LlmMetricState", module = "livekit.agents._core")]
pub struct LlmMetricState(pub(crate) LlmMetricStateImpl);

#[pymethods]
impl LlmMetricState {
    /// `label`, `model` and `provider`: the provider's, stamped on the record.
    #[new]
    #[pyo3(signature = (start, label, model, provider))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_llm_metric_state_new_start)]
        start: LlmMetricStateNewStartInput,
        label: ::std::string::String,
        model: ::std::string::String,
        provider: ::std::string::String,
    ) -> PyResult<LlmMetricState> {
        let out = py.detach(|| {
            <LlmMetricStateImpl as ::lk_agents_abi::metrics::provider::LlmMetricState>::new(
                start, label, model, provider,
            )
        });
        ::pyo3::PyResult::Ok(LlmMetricState(out))
    }

    /// Whether `chunk` is the first with a response.
    #[pyo3(signature = (now, chunk))]
    fn observe(
        &mut self,
        #[pyo3(from_py_with = _extract_llm_metric_state_observe_now)]
        now: LlmMetricStateObserveNowInput,
        chunk: Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let chunk = crate::generated::records::llm::chat_chunk_from_py(chunk.as_any())?;
        let out =
            <LlmMetricStateImpl as ::lk_agents_abi::metrics::provider::LlmMetricState>::observe(
                &mut self.0,
                now,
                chunk,
            );
        ::pyo3::PyResult::Ok(out)
    }

    /// `None` for a failed attempt or one that yielded no chunk; `timestamp`: unix seconds.
    #[pyo3(signature = (now, failed, timestamp))]
    fn finish<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_llm_metric_state_finish_now)]
        now: LlmMetricStateFinishNowInput,
        failed: bool,
        #[pyo3(from_py_with = _extract_llm_metric_state_finish_timestamp)]
        timestamp: LlmMetricStateFinishTimestampInput,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        let out =
            <LlmMetricStateImpl as ::lk_agents_abi::metrics::provider::LlmMetricState>::finish(
                &self.0, now, failed, timestamp,
            );
        out.map(|value| crate::generated::records::metrics::llm_metrics_to_py(py, value))
            .transpose()
    }
}

// Declared in rust/crates/lk-agents-abi/src/metrics/provider.rs.
type TtsMetricStateImpl =
    <crate::generated::implementations::Selected as super::Implementations>::TtsMetricState;

fn _extract_tts_metric_state_observe_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TtsMetricStateObserveNowInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "now")
}
type TtsMetricStateObserveNowInput = f64;

fn _extract_tts_metric_state_observe_start(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TtsMetricStateObserveStartInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "start")
}
type TtsMetricStateObserveStartInput = f64;

fn _extract_tts_metric_state_finish_now(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TtsMetricStateFinishNowInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "now")
}
type TtsMetricStateFinishNowInput = f64;

fn _extract_tts_metric_state_finish_start(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TtsMetricStateFinishStartInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "start")
}
type TtsMetricStateFinishStartInput = f64;

fn _extract_tts_metric_state_finish_timestamp(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TtsMetricStateFinishTimestampInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "timestamp")
}
type TtsMetricStateFinishTimestampInput = f64;

/// The TTS streams' metric fold as the host drives it.
#[pyclass(name = "TtsMetricState", module = "livekit.agents._core")]
pub struct TtsMetricState(pub(crate) TtsMetricStateImpl);

#[pymethods]
impl TtsMetricState {
    /// `label`, `model` and `provider`: the provider's, stamped on each record.
    #[new]
    #[pyo3(signature = (streamed, label, model, provider))]
    fn new(
        py: Python<'_>,
        streamed: bool,
        label: ::std::string::String,
        model: ::std::string::String,
        provider: ::std::string::String,
    ) -> PyResult<TtsMetricState> {
        let out = py.detach(|| {
            <TtsMetricStateImpl as ::lk_agents_abi::metrics::provider::TtsMetricState>::new(
                streamed, label, model, provider,
            )
        });
        ::pyo3::PyResult::Ok(TtsMetricState(out))
    }

    /// A streamed segment's text, in the order its audio is finished.
    #[pyo3(signature = (text))]
    fn queue_text(&mut self, text: ::std::string::String) -> PyResult<()> {
        <TtsMetricStateImpl as ::lk_agents_abi::metrics::provider::TtsMetricState>::queue_text(
            &mut self.0,
            text,
        );
        Ok(())
    }

    /// Whether `audio` ends its segment.
    #[pyo3(signature = (now, start, audio))]
    fn observe(
        &mut self,
        #[pyo3(from_py_with = _extract_tts_metric_state_observe_now)]
        now: TtsMetricStateObserveNowInput,
        #[pyo3(from_py_with = _extract_tts_metric_state_observe_start)]
        start: TtsMetricStateObserveStartInput,
        audio: Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let audio = crate::generated::records::tts::synthesized_audio_from_py(audio.as_any())?;
        let out =
            <TtsMetricStateImpl as ::lk_agents_abi::metrics::provider::TtsMetricState>::observe(
                &mut self.0,
                now,
                start,
                audio,
            );
        ::pyo3::PyResult::Ok(out)
    }

    /// The record of a finished request or segment, then the figures reset; `timestamp`: unix
    /// seconds.
    #[pyo3(signature = (now, start, failed, batch_text, timestamp))]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_tts_metric_state_finish_now)]
        now: TtsMetricStateFinishNowInput,
        #[pyo3(from_py_with = _extract_tts_metric_state_finish_start)]
        start: TtsMetricStateFinishStartInput,
        failed: bool,
        batch_text: ::std::string::String,
        #[pyo3(from_py_with = _extract_tts_metric_state_finish_timestamp)]
        timestamp: TtsMetricStateFinishTimestampInput,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        let out =
            <TtsMetricStateImpl as ::lk_agents_abi::metrics::provider::TtsMetricState>::finish(
                &mut self.0,
                now,
                start,
                failed,
                batch_text,
                timestamp,
            );
        out.map(|value| crate::generated::records::metrics::tts_metrics_to_py(py, value))
            .transpose()
    }
}

impl crate::py_errors::HostException
    for ::lk_agents_abi::metrics::provider::RecognitionUsageMissing
{
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyAssertionError::new_err(::std::string::ToString::to_string(self))
    }
}
