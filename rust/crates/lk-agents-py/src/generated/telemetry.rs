//! GENERATED FILE -- DO NOT EDIT.
//!
//! Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
//! `#[host(class)]` and declared `#[host(proxy)]` traits of `lk-agents-abi`.
//!
//! The `telemetry` classes.

// A host signature takes the parameters it declares.
#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

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
#[pyo3(signature = (*, endpoint = None, headers = None, metadata = None, allow_pii = None))]
pub fn _core_telemetry_configure(
    py: Python<'_>,
    endpoint: ::std::option::Option<::std::string::String>,
    headers: ::std::option::Option<
        ::std::collections::BTreeMap<::std::string::String, ::std::string::String>,
    >,
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
    py.detach(|| <TelemetryImpl as ::lk_agents_abi::telemetry::host::Telemetry>::_core_telemetry_configure(endpoint, headers, metadata, allow_pii)).map_err(|e: ::lk_agents_abi::telemetry::host::TelemetryConfigError| <::lk_agents_abi::telemetry::host::TelemetryConfigError as crate::py_errors::HostException>::to_py_err(&e))?;
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

/// A session report's host event snapshots: which are accepted and how many are kept. The adapter
/// owns the snapshots.
#[pyclass(name = "_core_EventRetention", module = "livekit.agents._core")]
pub struct CoreEventRetention(pub(crate) CoreEventRetentionImpl);

#[pymethods]
impl CoreEventRetention {
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

    /// Reject content of a redacted session, unknown schemas and live-object events.
    #[pyo3(signature = (kind, redaction))]
    fn accepts(&self, kind: ::std::string::String, redaction: bool) -> PyResult<bool> {
        crate::py_calls::check_pid()?;
        let out =
            <CoreEventRetentionImpl as ::lk_agents_abi::telemetry::report::EventRetention>::accepts(
                &self.0, kind, redaction,
            );
        ::pyo3::PyResult::Ok(out)
    }

    /// Counts one retained event; every accepted event is kept.
    fn admit(&mut self) -> PyResult<()> {
        crate::py_calls::check_pid()?;
        <CoreEventRetentionImpl as ::lk_agents_abi::telemetry::report::EventRetention>::admit(
            &mut self.0,
        );
        Ok(())
    }
}

impl crate::py_errors::HostException for ::lk_agents_abi::telemetry::host::TelemetryConfigError {
    fn to_py_err(&self) -> ::pyo3::PyErr {
        ::pyo3::exceptions::PyValueError::new_err(::std::string::ToString::to_string(self))
    }
}
