//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `telemetry` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

// Declared in rust/crates/lk-agents-abi/src/telemetry/mod.rs.
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

fn _extract_provider_metrics_stt_batch_metrics_host(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderMetricsSttBatchMetricsHostInput> {
    (|| -> ::pyo3::PyResult<_> {
        let tuple = obj.cast::<::pyo3::types::PyTuple>()?;
        if tuple.len() != 5 {
            return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                "{}: expected a tuple of 5 items",
                "host"
            )));
        }
        let item_0 = tuple.get_item(0)?;
        let item_1 = tuple.get_item(1)?;
        let item_2 = tuple.get_item(2)?;
        let item_3 = tuple.get_item(3)?;
        let item_4 = tuple.get_item(4)?;
        Ok((
            (crate::py_values::number_from_py::<f64>(item_0.as_any(), "host"))?,
            (item_1.extract::<::std::string::String>())?,
            (item_2.extract::<bool>())?,
            (item_3.extract::<::std::string::String>())?,
            (item_4.extract::<::std::string::String>())?,
        ))
    })()
}
type ProviderMetricsSttBatchMetricsHostInput =
    (f64, ::std::string::String, bool, ::std::string::String, ::std::string::String);

fn _extract_provider_metrics_stt_stream_metrics_host(
    obj: &Bound<'_, PyAny>,
) -> PyResult<ProviderMetricsSttStreamMetricsHostInput> {
    (|| -> ::pyo3::PyResult<_> {
        let tuple = obj.cast::<::pyo3::types::PyTuple>()?;
        if tuple.len() != 5 {
            return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                "{}: expected a tuple of 5 items",
                "host"
            )));
        }
        let item_0 = tuple.get_item(0)?;
        let item_1 = tuple.get_item(1)?;
        let item_2 = tuple.get_item(2)?;
        let item_3 = tuple.get_item(3)?;
        let item_4 = tuple.get_item(4)?;
        Ok((
            (crate::py_values::number_from_py::<f64>(item_0.as_any(), "host"))?,
            (item_1.extract::<::std::string::String>())?,
            (item_2.extract::<bool>())?,
            (item_3.extract::<::std::string::String>())?,
            (item_4.extract::<::std::string::String>())?,
        ))
    })()
}
type ProviderMetricsSttStreamMetricsHostInput =
    (f64, ::std::string::String, bool, ::std::string::String, ::std::string::String);

/// `STT.recognize`'s metrics for one batch recognition.
#[pyfunction]
#[pyo3(signature = (event, start, now, audio_durations, host))]
pub fn stt_batch_metrics<'py>(
    py: Python<'py>,
    event: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_start)]
    start: ProviderMetricsSttBatchMetricsStartInput,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_now)]
    now: ProviderMetricsSttBatchMetricsNowInput,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_audio_durations)]
    audio_durations: ProviderMetricsSttBatchMetricsAudioDurationsInput,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_batch_metrics_host)]
    host: ProviderMetricsSttBatchMetricsHostInput,
) -> PyResult<Bound<'py, PyAny>> {
    let event = crate::generated::records::stt::speech_event_from_py(event.as_any())?;
    let out = py.detach(|| <ProviderMetricsImpl as ::lk_agents_abi::telemetry::provider::ProviderMetrics>::stt_batch_metrics(event, start, now, audio_durations, host));
    crate::generated::records::metrics::stt_metrics_to_py(py, out)
}

/// `RecognizeStream`'s metrics for one event, and whether it resets the retry count.
#[pyfunction]
#[pyo3(signature = (event, host))]
pub fn stt_stream_metrics<'py>(
    py: Python<'py>,
    event: Bound<'_, PyAny>,
    #[pyo3(from_py_with = _extract_provider_metrics_stt_stream_metrics_host)]
    host: ProviderMetricsSttStreamMetricsHostInput,
) -> PyResult<Bound<'py, PyAny>> {
    let event = crate::generated::records::stt::speech_event_from_py(event.as_any())?;
    let out = py.detach(|| <ProviderMetricsImpl as ::lk_agents_abi::telemetry::provider::ProviderMetrics>::stt_stream_metrics(event, host)).map_err(|e: ::lk_agents_abi::telemetry::provider::RecognitionUsageMissing| <::lk_agents_abi::telemetry::provider::RecognitionUsageMissing as crate::py_errors::HostException>::to_py_err(&e))?;
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

// Declared in rust/crates/lk-agents-abi/src/telemetry/mod.rs.
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

fn _extract_llm_metric_state_finish_host(
    obj: &Bound<'_, PyAny>,
) -> PyResult<LlmMetricStateFinishHostInput> {
    (|| -> ::pyo3::PyResult<_> {
        let tuple = obj.cast::<::pyo3::types::PyTuple>()?;
        if tuple.len() != 5 {
            return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                "{}: expected a tuple of 5 items",
                "host"
            )));
        }
        let item_0 = tuple.get_item(0)?;
        let item_1 = tuple.get_item(1)?;
        let item_2 = tuple.get_item(2)?;
        let item_3 = tuple.get_item(3)?;
        let item_4 = tuple.get_item(4)?;
        Ok((
            (crate::py_values::number_from_py::<f64>(item_0.as_any(), "host"))?,
            (item_1.extract::<::std::string::String>())?,
            (item_2.extract::<bool>())?,
            (item_3.extract::<::std::string::String>())?,
            (item_4.extract::<::std::string::String>())?,
        ))
    })()
}
type LlmMetricStateFinishHostInput =
    (f64, ::std::string::String, bool, ::std::string::String, ::std::string::String);

/// `LLMStream`'s metric fold as the host drives it.
#[pyclass(name = "LlmMetricState", module = "livekit.agents._core")]
pub struct LlmMetricState(pub(crate) LlmMetricStateImpl);

#[pymethods]
impl LlmMetricState {
    #[new]
    #[pyo3(signature = (start))]
    fn new(
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_llm_metric_state_new_start)]
        start: LlmMetricStateNewStartInput,
    ) -> PyResult<LlmMetricState> {
        let out = py.detach(|| {
            <LlmMetricStateImpl as ::lk_agents_abi::telemetry::provider::LlmMetricState>::new(start)
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
            <LlmMetricStateImpl as ::lk_agents_abi::telemetry::provider::LlmMetricState>::observe(
                &mut self.0,
                now,
                chunk,
            );
        ::pyo3::PyResult::Ok(out)
    }

    /// `None` for a failed attempt or one that yielded no chunk.
    #[pyo3(signature = (now, failed, host))]
    fn finish<'py>(
        &self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_llm_metric_state_finish_now)]
        now: LlmMetricStateFinishNowInput,
        failed: bool,
        #[pyo3(from_py_with = _extract_llm_metric_state_finish_host)]
        host: LlmMetricStateFinishHostInput,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        let out =
            <LlmMetricStateImpl as ::lk_agents_abi::telemetry::provider::LlmMetricState>::finish(
                &self.0, now, failed, host,
            );
        out.map(|value| crate::generated::records::metrics::llm_metrics_to_py(py, value))
            .transpose()
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/mod.rs.
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

fn _extract_tts_metric_state_finish_host(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TtsMetricStateFinishHostInput> {
    (|| -> ::pyo3::PyResult<_> {
        let tuple = obj.cast::<::pyo3::types::PyTuple>()?;
        if tuple.len() != 5 {
            return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                "{}: expected a tuple of 5 items",
                "host"
            )));
        }
        let item_0 = tuple.get_item(0)?;
        let item_1 = tuple.get_item(1)?;
        let item_2 = tuple.get_item(2)?;
        let item_3 = tuple.get_item(3)?;
        let item_4 = tuple.get_item(4)?;
        Ok((
            (crate::py_values::number_from_py::<f64>(item_0.as_any(), "host"))?,
            (item_1.extract::<::std::string::String>())?,
            (item_2.extract::<bool>())?,
            (item_3.extract::<::std::string::String>())?,
            (item_4.extract::<::std::string::String>())?,
        ))
    })()
}
type TtsMetricStateFinishHostInput =
    (f64, ::std::string::String, bool, ::std::string::String, ::std::string::String);

/// The TTS streams' metric fold as the host drives it.
#[pyclass(name = "TtsMetricState", module = "livekit.agents._core")]
pub struct TtsMetricState(pub(crate) TtsMetricStateImpl);

#[pymethods]
impl TtsMetricState {
    #[new]
    #[pyo3(signature = (streamed))]
    fn new(py: Python<'_>, streamed: bool) -> PyResult<TtsMetricState> {
        let out = py.detach(|| {
            <TtsMetricStateImpl as ::lk_agents_abi::telemetry::provider::TtsMetricState>::new(
                streamed,
            )
        });
        ::pyo3::PyResult::Ok(TtsMetricState(out))
    }

    /// A streamed segment's text, in the order its audio is finished.
    #[pyo3(signature = (text))]
    fn queue_text(&mut self, text: ::std::string::String) -> PyResult<()> {
        <TtsMetricStateImpl as ::lk_agents_abi::telemetry::provider::TtsMetricState>::queue_text(
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
            <TtsMetricStateImpl as ::lk_agents_abi::telemetry::provider::TtsMetricState>::observe(
                &mut self.0,
                now,
                start,
                audio,
            );
        ::pyo3::PyResult::Ok(out)
    }

    /// The record of a finished request or segment, then the figures reset.
    #[pyo3(signature = (now, start, failed, batch_text, host))]
    fn finish<'py>(
        &mut self,
        py: Python<'py>,
        #[pyo3(from_py_with = _extract_tts_metric_state_finish_now)]
        now: TtsMetricStateFinishNowInput,
        #[pyo3(from_py_with = _extract_tts_metric_state_finish_start)]
        start: TtsMetricStateFinishStartInput,
        failed: bool,
        batch_text: ::std::string::String,
        #[pyo3(from_py_with = _extract_tts_metric_state_finish_host)]
        host: TtsMetricStateFinishHostInput,
    ) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        let out =
            <TtsMetricStateImpl as ::lk_agents_abi::telemetry::provider::TtsMetricState>::finish(
                &mut self.0,
                now,
                start,
                failed,
                batch_text,
                host,
            );
        out.map(|value| crate::generated::records::metrics::tts_metrics_to_py(py, value))
            .transpose()
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/host.rs.
type TelemetrySpanImpl =
    <crate::generated::implementations::Selected as super::Implementations>::HostSpan;

fn _extract_host_span_new_parent(obj: &Bound<'_, PyAny>) -> PyResult<HostSpanNewParentInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        ((|| -> ::pyo3::PyResult<_> {
            let tuple = obj.cast::<::pyo3::types::PyTuple>()?;
            if tuple.len() != 3 {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "{}: expected a tuple of 3 items",
                    "parent"
                )));
            }
            let item_0 = tuple.get_item(0)?;
            let item_1 = tuple.get_item(1)?;
            let item_2 = tuple.get_item(2)?;
            Ok((
                (crate::py_values::number_from_py::<u128>(item_0.as_any(), "parent"))?,
                (crate::py_values::number_from_py::<u64>(item_1.as_any(), "parent"))?,
                (item_2.extract::<bool>())?,
            ))
        })())
        .map(Some)
    }
}
type HostSpanNewParentInput = ::std::option::Option<(u128, u64, bool)>;

fn _extract_host_span_new_kind(obj: &Bound<'_, PyAny>) -> PyResult<HostSpanNewKindInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "kind")
}
type HostSpanNewKindInput = i64;

fn _extract_host_span_new_start_time(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HostSpanNewStartTimeInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u64>(obj.as_any(), "start_time")).map(Some)
    }
}
type HostSpanNewStartTimeInput = ::std::option::Option<u64>;

fn _extract_host_span_new_ids(obj: &Bound<'_, PyAny>) -> PyResult<HostSpanNewIdsInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        ((|| -> ::pyo3::PyResult<_> {
            let tuple = obj.cast::<::pyo3::types::PyTuple>()?;
            if tuple.len() != 2 {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "{}: expected a tuple of 2 items",
                    "ids"
                )));
            }
            let item_0 = tuple.get_item(0)?;
            let item_1 = tuple.get_item(1)?;
            Ok((
                (crate::py_values::number_from_py::<u128>(item_0.as_any(), "ids"))?,
                (crate::py_values::number_from_py::<u64>(item_1.as_any(), "ids"))?,
            ))
        })())
        .map(Some)
    }
}
type HostSpanNewIdsInput = ::std::option::Option<(u128, u64)>;
type HostSpanContextOutput = (u128, u64, bool);

fn _extract_host_span_add_event_timestamp(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HostSpanAddEventTimestampInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u64>(obj.as_any(), "timestamp")).map(Some)
    }
}
type HostSpanAddEventTimestampInput = ::std::option::Option<u64>;

fn _extract_host_span_set_status_code(
    obj: &Bound<'_, PyAny>,
) -> PyResult<HostSpanSetStatusCodeInput> {
    crate::py_values::number_from_py::<i64>(obj.as_any(), "code")
}
type HostSpanSetStatusCodeInput = i64;

fn _extract_host_span_end_end_time(obj: &Bound<'_, PyAny>) -> PyResult<HostSpanEndEndTimeInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        (crate::py_values::number_from_py::<u64>(obj.as_any(), "end_time")).map(Some)
    }
}
type HostSpanEndEndTimeInput = ::std::option::Option<u64>;

/// A span started through the host's OpenTelemetry API (an application span, or a framework span
/// around host-only code). Ended once; recorded by the process's exporter.
#[pyclass(frozen, name = "TelemetrySpan", module = "livekit.agents._core")]
pub struct TelemetrySpan(pub(crate) ::std::sync::Arc<TelemetrySpanImpl>);

#[pymethods]
impl TelemetrySpan {
    /// `parent`: `(trace_id, span_id, sampled)` of the current span, `None` for a new trace.
    /// `kind`: the OpenTelemetry `SpanKind` value. `ids`: `(trace_id, span_id)` of a span the
    /// host already started (an SDK span the bridge forwards). `job`: the job it belongs to.
    #[new]
    #[pyo3(signature = (name, scope, *, parent = None, kind = 0, start_time = None, ids = None, job = None))]
    fn new(
        py: Python<'_>,
        name: ::std::string::String,
        scope: ::std::string::String,
        #[pyo3(from_py_with = _extract_host_span_new_parent)] parent: HostSpanNewParentInput,
        #[pyo3(from_py_with = _extract_host_span_new_kind)] kind: HostSpanNewKindInput,
        #[pyo3(from_py_with = _extract_host_span_new_start_time)]
        start_time: HostSpanNewStartTimeInput,
        #[pyo3(from_py_with = _extract_host_span_new_ids)] ids: HostSpanNewIdsInput,
        job: ::std::option::Option<::std::string::String>,
    ) -> PyResult<TelemetrySpan> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| {
            <TelemetrySpanImpl as ::lk_agents_abi::telemetry::host::HostSpan>::new(
                &env, name, scope, parent, kind, start_time, ids, job,
            )
        });
        ::pyo3::PyResult::Ok(TelemetrySpan(::std::sync::Arc::new(out)))
    }

    /// `(trace_id, span_id, sampled)`.
    #[getter]
    fn context(&self, py: Python<'_>) -> PyResult<HostSpanContextOutput> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TelemetrySpanImpl as ::lk_agents_abi::telemetry::host::HostSpan>::context(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// Whether the span has not ended.
    fn is_recording(&self, py: Python<'_>) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = py.detach(|| {
            <TelemetrySpanImpl as ::lk_agents_abi::telemetry::host::HostSpan>::is_recording(&self.0)
        });
        ::pyo3::PyResult::Ok(out)
    }

    /// A JSON scalar or array; `None` removes nothing and is ignored.
    #[pyo3(signature = (key, value))]
    fn set_attribute(
        &self,
        py: Python<'_>,
        key: ::std::string::String,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let value = crate::py_values::value_from_py(value.as_any(), "value")?;
        py.detach(|| {
            <TelemetrySpanImpl as ::lk_agents_abi::telemetry::host::HostSpan>::set_attribute(
                &self.0, key, value,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (name, attributes = None, timestamp = None))]
    fn add_event(
        &self,
        py: Python<'_>,
        name: ::std::string::String,
        attributes: Option<Bound<'_, PyAny>>,
        #[pyo3(from_py_with = _extract_host_span_add_event_timestamp)]
        timestamp: HostSpanAddEventTimestampInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let attributes = attributes
            .map(|v| {
                v.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| {
                    dict.iter()
                        .map(|(key, item)| {
                            Ok((
                                key.extract::<::std::string::String>()?,
                                crate::py_values::value_from_py(item.as_any(), "attributes")?,
                            ))
                        })
                        .collect::<PyResult<::std::collections::BTreeMap<_, _>>>()
                })
            })
            .transpose()?;
        py.detach(|| {
            <TelemetrySpanImpl as ::lk_agents_abi::telemetry::host::HostSpan>::add_event(
                &self.0, name, attributes, timestamp,
            )
        });
        Ok(())
    }

    /// `code`: the OpenTelemetry `StatusCode` value.
    #[pyo3(signature = (code, description = None))]
    fn set_status(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_host_span_set_status_code)] code: HostSpanSetStatusCodeInput,
        description: ::std::option::Option<::std::string::String>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <TelemetrySpanImpl as ::lk_agents_abi::telemetry::host::HostSpan>::set_status(
                &self.0,
                code,
                description,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (name))]
    fn update_name(&self, py: Python<'_>, name: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <TelemetrySpanImpl as ::lk_agents_abi::telemetry::host::HostSpan>::update_name(
                &self.0, name,
            )
        });
        Ok(())
    }

    /// Ends the span; later calls do nothing.
    #[pyo3(signature = (end_time = None))]
    fn end(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = _extract_host_span_end_end_time)] end_time: HostSpanEndEndTimeInput,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        py.detach(|| {
            <TelemetrySpanImpl as ::lk_agents_abi::telemetry::host::HostSpan>::end(
                &self.0, end_time,
            )
        });
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/host.rs.
type TelemetryImpl =
    <crate::generated::implementations::Selected as super::Implementations>::Telemetry;
type TelemetryCoreTelemetryDiagnosticsOutput =
    ::std::collections::BTreeMap<::std::string::String, u64>;

fn _extract_telemetry_core_event_loop_blocked_duration(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TelemetryCoreEventLoopBlockedDurationInput> {
    crate::py_values::number_from_py::<f64>(obj.as_any(), "duration")
}
type TelemetryCoreEventLoopBlockedDurationInput = f64;

fn _extract_telemetry_core_event_loop_blocked_parent(
    obj: &Bound<'_, PyAny>,
) -> PyResult<TelemetryCoreEventLoopBlockedParentInput> {
    if obj.is_none() {
        Ok(None)
    } else {
        ((|| -> ::pyo3::PyResult<_> {
            let tuple = obj.cast::<::pyo3::types::PyTuple>()?;
            if tuple.len() != 3 {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "{}: expected a tuple of 3 items",
                    "parent"
                )));
            }
            let item_0 = tuple.get_item(0)?;
            let item_1 = tuple.get_item(1)?;
            let item_2 = tuple.get_item(2)?;
            Ok((
                (crate::py_values::number_from_py::<u128>(item_0.as_any(), "parent"))?,
                (crate::py_values::number_from_py::<u64>(item_1.as_any(), "parent"))?,
                (item_2.extract::<bool>())?,
            ))
        })())
        .map(Some)
    }
}
type TelemetryCoreEventLoopBlockedParentInput = ::std::option::Option<(u128, u64, bool)>;

/// Starts the process's exporter, stamping `sdk_version`, with the application endpoint from
/// the `OTEL_EXPORTER_OTLP_*` environment. Later calls do nothing.
#[pyfunction]
#[pyo3(signature = (sdk_version))]
pub fn _core_telemetry_install(py: Python<'_>, sdk_version: ::std::string::String) -> PyResult<()> {
    py.detach(|| {
        <TelemetryImpl as ::lk_agents_abi::telemetry::host::Telemetry>::_core_telemetry_install(
            sdk_version,
        )
    });
    Ok(())
}

/// Replaces the application endpoint; arguments override the environment.
#[pyfunction]
#[pyo3(signature = (*, endpoint = None, headers = None, protocol = None, metadata = None, allow_pii = None))]
pub fn _core_telemetry_configure(
    py: Python<'_>,
    endpoint: ::std::option::Option<::std::string::String>,
    headers: ::std::option::Option<
        ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
    >,
    protocol: ::std::option::Option<::std::string::String>,
    metadata: Option<Bound<'_, PyAny>>,
    allow_pii: ::std::option::Option<bool>,
) -> PyResult<()> {
    let metadata = metadata
        .map(|v| {
            v.cast::<::pyo3::types::PyDict>().map_err(::pyo3::PyErr::from).and_then(|dict| {
                dict.iter()
                    .map(|(key, item)| {
                        Ok((
                            key.extract::<::std::string::String>()?,
                            crate::py_values::value_from_py(item.as_any(), "metadata")?,
                        ))
                    })
                    .collect::<PyResult<::std::collections::BTreeMap<_, _>>>()
            })
        })
        .transpose()?;
    py.detach(|| <TelemetryImpl as ::lk_agents_abi::telemetry::host::Telemetry>::_core_telemetry_configure(endpoint, headers, protocol, metadata, allow_pii)).map_err(|e: ::lk_agents_abi::telemetry::host::TelemetryConfigError| <::lk_agents_abi::telemetry::host::TelemetryConfigError as crate::py_errors::HostException>::to_py_err(&e))?;
    Ok(())
}

/// The export counters: `dropped`, `failed`, `conflicts`, `incomplete_cleanup`, `swept_spans`,
/// `abandoned_sessions`.
#[pyfunction]
pub fn _core_telemetry_diagnostics(
    py: Python<'_>,
) -> PyResult<TelemetryCoreTelemetryDiagnosticsOutput> {
    let out = py.detach(
        <TelemetryImpl as ::lk_agents_abi::telemetry::host::Telemetry>::_core_telemetry_diagnostics,
    );
    ::pyo3::PyResult::Ok(out)
}

/// Counts a configuration conflict the host found (a tracer provider it cannot bridge).
#[pyfunction]
pub fn _core_telemetry_conflict(py: Python<'_>) -> PyResult<()> {
    py.detach(
        <TelemetryImpl as ::lk_agents_abi::telemetry::host::Telemetry>::_core_telemetry_conflict,
    );
    Ok(())
}

/// GenAI message capture for spans recorded afterwards.
#[pyfunction]
#[pyo3(signature = (enabled))]
pub fn _core_set_capture_content(py: Python<'_>, enabled: bool) -> PyResult<()> {
    py.detach(|| {
        <TelemetryImpl as ::lk_agents_abi::telemetry::host::Telemetry>::_core_set_capture_content(
            enabled,
        )
    });
    Ok(())
}

/// The process-exit flush: every destination's queue, bounded.
#[pyfunction]
pub fn _core_telemetry_shutdown<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    crate::py_calls::future_into_py(py, async move {
        let () = <TelemetryImpl as ::lk_agents_abi::telemetry::host::Telemetry>::_core_telemetry_shutdown().await;
        Python::attach(|py| ::pyo3::IntoPyObjectExt::into_py_any((), py))
    })
}

/// A stall of the host's event loop, `duration` seconds long, reported once it ended: its
/// measurement, and an `event_loop_blocked` span under `parent` (`(trace_id, span_id,
/// sampled)`), at most 30 a minute. `task`: the task that was running.
#[pyfunction]
#[pyo3(signature = (duration, *, parent = None, job = None, task = None))]
pub fn _core_event_loop_blocked(
    py: Python<'_>,
    #[pyo3(from_py_with = _extract_telemetry_core_event_loop_blocked_duration)]
    duration: TelemetryCoreEventLoopBlockedDurationInput,
    #[pyo3(from_py_with = _extract_telemetry_core_event_loop_blocked_parent)]
    parent: TelemetryCoreEventLoopBlockedParentInput,
    job: ::std::option::Option<::std::string::String>,
    task: ::std::option::Option<::std::string::String>,
) -> PyResult<()> {
    py.detach(|| {
        <TelemetryImpl as ::lk_agents_abi::telemetry::host::Telemetry>::_core_event_loop_blocked(
            duration, parent, job, task,
        )
    });
    Ok(())
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report/mod.rs.
type TaggerImpl = <crate::generated::implementations::Selected as super::Implementations>::Tagger;
type TaggerOutcomeReasonOutput = ::std::option::Option<::std::string::String>;

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
pub struct Tagger(pub(crate) TaggerImpl);

#[pymethods]
impl Tagger {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Tagger> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out =
            py.detach(|| <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::new(&env));
        ::pyo3::PyResult::Ok(Tagger(out))
    }

    /// Mark the session as successful; ``reason`` is kept apart from the tag.
    #[pyo3(signature = (reason = None))]
    fn success(&mut self, reason: ::std::option::Option<::std::string::String>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::success(&mut self.0, reason);
        Ok(())
    }

    /// Mark the session as failed; ``reason`` is kept apart from the tag.
    #[pyo3(signature = (reason = None))]
    fn fail(&mut self, reason: ::std::option::Option<::std::string::String>) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::fail(&mut self.0, reason);
        Ok(())
    }

    /// Add a ``key:value`` tag, with optional structured metadata.
    #[pyo3(signature = (tag, *, metadata = None))]
    fn add(
        &mut self,
        tag: ::std::string::String,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        let metadata = metadata
            .map(|v| crate::py_values::value_from_py(v.as_any(), "metadata"))
            .transpose()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::add(&mut self.0, tag, metadata);
        Ok(())
    }

    /// Remove a tag; removing an absent one does nothing.
    #[pyo3(signature = (tag))]
    fn remove(&mut self, tag: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::remove(&mut self.0, tag);
        Ok(())
    }

    /// The session's tags.
    #[getter]
    fn tags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::tags(&self.0);
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyFrozenSet::new(
                py,
                out.into_iter()
                    .map(|x| ::pyo3::IntoPyObjectExt::into_bound_py_any(x, py))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// The tags in insertion order, with their metadata and when they were added.
    fn entries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::entries(&self.0);
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::telemetry::report_tag_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// The judgments recorded by judge groups.
    #[getter]
    fn evaluations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::py_calls::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::evaluations(&self.0);
        ::pyo3::PyResult::Ok(
            ::pyo3::types::PyList::new(
                py,
                out.into_iter()
                    .map(|x| crate::generated::records::telemetry::evaluation_record_to_py(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
            )?
            .into_any(),
        )
    }

    /// ``"success"``, ``"fail"`` or ``None``.
    #[getter]
    fn outcome<'py>(&self, py: Python<'py>) -> PyResult<::std::option::Option<Bound<'py, PyAny>>> {
        crate::py_calls::check_pid()?;
        let out = <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::outcome(&self.0);
        out.map(|value| crate::generated::records::telemetry::outcome_to_py(py, value)).transpose()
    }

    /// The reason given with the outcome.
    #[getter]
    fn outcome_reason(&self) -> PyResult<TaggerOutcomeReasonOutput> {
        crate::py_calls::check_pid()?;
        let out =
            <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::outcome_reason(&self.0);
        ::pyo3::PyResult::Ok(out)
    }

    /// One judgment of a judge group's result: its ``lk.judge.<name>:<verdict>`` tag and record.
    #[pyo3(signature = (name, verdict, reasoning, instructions))]
    fn _core_record_judgment(
        &mut self,
        name: ::std::string::String,
        verdict: ::std::string::String,
        reasoning: ::std::string::String,
        instructions: ::std::string::String,
    ) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <TaggerImpl as ::lk_agents_abi::telemetry::report::Tagger>::_core_record_judgment(
            &mut self.0,
            name,
            verdict,
            reasoning,
            instructions,
        );
        Ok(())
    }
}

// Declared in rust/crates/lk-agents-abi/src/telemetry/report/mod.rs.
type CoreEventRetentionImpl =
    <crate::generated::implementations::Selected as super::Implementations>::EventRetention;
type EventRetentionAdmitOutput = (usize, bool);

fn _extract_event_retention_admit_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<EventRetentionAdmitNowNsInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type EventRetentionAdmitNowNsInput = u64;

fn _extract_event_retention_expire_now_ns(
    obj: &Bound<'_, PyAny>,
) -> PyResult<EventRetentionExpireNowNsInput> {
    crate::py_values::number_from_py::<u64>(obj.as_any(), "now_ns")
}
type EventRetentionExpireNowNsInput = u64;

/// Bounded retention of a session report's host event snapshots. The native policy retains only
/// byte charges and supplied monotonic timestamps; the adapter owns independent model copies.
#[pyclass(name = "_core_EventRetention", module = "livekit.agents._core")]
pub struct CoreEventRetention(pub(crate) CoreEventRetentionImpl);

#[pymethods]
impl CoreEventRetention {
    /// Takes the process's GenAI message capture setting as it is now.
    #[new]
    fn new(py: Python<'_>) -> PyResult<CoreEventRetention> {
        crate::py_calls::check_pid()?;
        let env = crate::process::env();
        let out = py.detach(|| {
            <CoreEventRetentionImpl as ::lk_agents_abi::telemetry::report::EventRetention>::new(
                &env,
            )
        });
        ::pyo3::PyResult::Ok(CoreEventRetention(out))
    }

    /// Bind report access to the first job that starts the session. Ownership cannot transfer.
    #[pyo3(signature = (job_id))]
    fn claim(&mut self, job_id: ::std::string::String) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <CoreEventRetentionImpl as ::lk_agents_abi::telemetry::report::EventRetention>::claim(
            &mut self.0,
            job_id,
        );
        Ok(())
    }

    /// Sessions started outside a job retain their existing application-owned access contract.
    #[pyo3(signature = (job_id))]
    fn allows_read(&self, job_id: ::std::string::String) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out = <CoreEventRetentionImpl as ::lk_agents_abi::telemetry::report::EventRetention>::allows_read(&self.0, job_id);
        ::pyo3::PyResult::Ok(out)
    }

    /// Reject content without creation-time consent, unknown schemas and live-object events.
    #[pyo3(signature = (kind, redaction))]
    fn accepts(&self, kind: ::std::string::String, redaction: bool) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out =
            <CoreEventRetentionImpl as ::lk_agents_abi::telemetry::report::EventRetention>::accepts(
                &self.0, kind, redaction,
            );
        ::pyo3::PyResult::Ok(out)
    }

    /// Expire old entries, then admit `event` only within the item and byte budgets. The adapter
    /// removes exactly the returned prefix before appending its snapshot of an admitted event.
    #[pyo3(signature = (event, now_ns))]
    fn admit(
        &mut self,
        event: Bound<'_, PyAny>,
        #[pyo3(from_py_with = _extract_event_retention_admit_now_ns)]
        now_ns: EventRetentionAdmitNowNsInput,
    ) -> PyResult<EventRetentionAdmitOutput> {
        crate::py_calls::check_pid()?;
        let event = crate::py_values::telemetry_value_from_py(event.as_any())?;
        let out =
            <CoreEventRetentionImpl as ::lk_agents_abi::telemetry::report::EventRetention>::admit(
                &mut self.0,
                event,
                now_ns,
            );
        ::pyo3::PyResult::Ok(out)
    }

    #[pyo3(signature = (now_ns))]
    fn expire(
        &mut self,
        #[pyo3(from_py_with = _extract_event_retention_expire_now_ns)]
        now_ns: EventRetentionExpireNowNsInput,
    ) -> PyResult<usize> {
        crate::py_calls::check_pid()?;
        let out =
            <CoreEventRetentionImpl as ::lk_agents_abi::telemetry::report::EventRetention>::expire(
                &mut self.0,
                now_ns,
            );
        ::pyo3::PyResult::Ok(out)
    }
}

impl crate::py_errors::HostException
    for ::lk_agents_abi::telemetry::provider::RecognitionUsageMissing
{
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyAssertionError::new_err(::std::string::ToString::to_string(self))
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::telemetry::host::TelemetryConfigError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}
