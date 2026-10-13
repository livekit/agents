//! Worker host contracts. Runtime behavior belongs to `lk-agents-worker`.

use lk_agents_host_derive::Host;

// For the generated reader's glob import. `RecordingOptions` is aliased: the crate root exports
// another type of that name.
pub use crate::ChatItem;
pub use crate::metrics::usage::records::ModelUsage;
pub use crate::telemetry::report::RecordingOptions as ReportRecordingOptions;

/// Current worker state, available independently of lifecycle notification consumption.
#[derive(Debug, Clone, Default, Host)]
#[binding(emit)]
pub struct Snapshot {
    pub worker_id: String,
    pub draining: bool,
    pub active_jobs: usize,
    pub reserved_slots: usize,
    pub worker_load: f32,
    pub target_idle_processes: Option<u32>,
    /// `finished` or `timed_out`, once the drain completes.
    pub drain_outcome: Option<String>,
    /// `run()` has returned. `Some` carries the reason, `None` while still running.
    pub finished: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerType {
    Room,
    Publisher,
}

/// `WorkerPermissions`, field-for-field, including the defaults.
// Every key optional: the `#[binding(default = ..)]` values and the record's `default` are `Default`'s.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, default)]
pub struct WorkerPermissions {
    #[binding(default = true)]
    pub can_publish: bool,
    #[binding(default = true)]
    pub can_subscribe: bool,
    #[binding(default = true)]
    pub can_publish_data: bool,
    #[binding(default = true)]
    pub can_update_metadata: bool,
    /// `models.TrackSource` values, as raw enum numbers.
    pub can_publish_sources: Vec<i32>,
    pub hidden: bool,
}

impl Default for WorkerPermissions {
    fn default() -> Self {
        WorkerPermissions {
            can_publish: true,
            can_subscribe: true,
            can_publish_data: true,
            can_update_metadata: true,
            can_publish_sources: Vec::new(),
            hidden: false,
        }
    }
}

/// The scalar half of a `livekit.agent.Job`, as handed to the user's `request_fnc`.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit)]
pub struct JobInfo {
    pub id: String,
    pub dispatch_id: String,
    #[host(rename = "type")]
    pub ty: i32,
    /// The room's name, flattened out of `Job.room` so a `request_fnc` need not parse a protobuf.
    pub room: Option<String>,
    pub room_sid: Option<String>,
    pub agent_name: String,
    pub metadata: String,
    pub deployment: String,
    pub enable_recording: bool,
    pub enable_redaction: bool,
    /// The exact `agent.Job` bytes: the fields above are lossy, and this crosses IPC.
    pub proto: Vec<u8>,
}

/// A job offered to the host. The host returns a [`JobDecision`].
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(emit)]
pub struct JobRequest {
    pub job: JobInfo,
    /// `AvailabilityRequest.resuming` -- the job is migrating from another worker.
    pub resuming: bool,
}

impl JobRequest {
    pub fn room_name(&self) -> &str {
        self.job.room.as_deref().unwrap_or("")
    }
}

/// `RunningJobInfo`: what a launched job is handed.
#[derive(Debug, Clone, PartialEq, Host)]
// `accept_arguments` nests rather than converting: `AcceptArguments` already declares both directions.
#[binding(emit)]
pub struct RunningJobInfo {
    pub accept_arguments: AcceptArguments,
    pub job: JobInfo,
    pub url: String,
    pub token: String,
    pub worker_id: String,
    pub fake_job: bool,
}

/// Acceptance returns before assignment and launch; the worker owns the assignment deadline.
#[derive(Debug, Clone, PartialEq)]
pub enum JobDecision {
    Accept(AcceptArguments),
    Reject { terminate: bool },
    /// The host returned without a decision: reject without terminating.
    Unanswered,
    /// Logged, then rejected without terminating.
    Failed(crate::errors::HostError),
}

impl JobDecision {
    /// The one table of `accept`, `reject`, `terminate`, `unanswered`, `failed`; `terminate` also
    /// asks the server to end the job. An unknown word is refused with the binding's message.
    pub fn from_wire(
        word: &str,
        accept: impl FnOnce() -> Result<AcceptArguments, String>,
        error: impl FnOnce() -> String,
    ) -> Result<JobDecision, String> {
        Ok(match word {
            "accept" => JobDecision::Accept(accept()?),
            "reject" => JobDecision::Reject { terminate: false },
            "terminate" => JobDecision::Reject { terminate: true },
            "unanswered" => JobDecision::Unanswered,
            "failed" => JobDecision::Failed(crate::errors::HostError::new(error())),
            other => {
                return Err(format!(
                    "unknown decision {other:?}: expected accept, reject, terminate, unanswered or failed"
                ));
            }
        })
    }
}

/// The dict a `request_fnc` answers with, before it is a [`JobDecision`]: the `decision` word and
/// the `error` only `failed` carries.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items)]
pub struct RawJobDecision {
    pub decision: String,
    #[binding(default)]
    pub error: Option<String>,
}

/// `attributes` crosses as a dict out, a list of pairs (or `None`) in.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AcceptArguments {
    pub name: String,
    pub identity: String,
    pub metadata: String,
    /// Merged after `lk.agent.name`, so a user attribute of that name wins.
    pub attributes: Vec<(String, String)>,
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for &AcceptArguments {
    type Target = pyo3::types::PyDict;
    type Output = pyo3::Bound<'py, pyo3::types::PyDict>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::PyDictMethods as _;
        let attributes = pyo3::types::PyDict::new(py);
        for (k, v) in &self.attributes {
            attributes.set_item(k, v)?;
        }
        let d = pyo3::types::PyDict::new(py);
        d.set_item("name", &self.name)?;
        d.set_item("identity", &self.identity)?;
        d.set_item("metadata", &self.metadata)?;
        d.set_item("attributes", attributes)?;
        Ok(d)
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for AcceptArguments {
    type Target = pyo3::types::PyDict;
    type Output = pyo3::Bound<'py, pyo3::types::PyDict>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
        (&self).into_pyobject(py)
    }
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for AcceptArguments {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::{PyAnyMethods as _, PyDictMethods as _};
        let d = obj.cast::<pyo3::types::PyDict>()?;
        let text = |key: &str| -> pyo3::PyResult<String> {
            Ok(match d.get_item(key)? { Some(v) => v.extract()?, None => String::new() })
        };
        Ok(AcceptArguments {
            name: text("name")?,
            identity: text("identity")?,
            metadata: text("metadata")?,
            attributes: match d.get_item("attributes")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => Vec::new(),
            },
        })
    }
}

// --- `next_notification()`'s four dicts ------------------------------------------------------
//
// `worker.py` matches on `event["kind"]`, so the tag values are API.

/// The server accepted the registration. `kind: "registered"`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "registered")]
pub struct WorkerRegistered {
    /// The id the server issued.
    pub worker_id: String,
    pub region: String,
    /// The protocol version the server chose.
    pub protocol: i32,
    pub deployment: String,
    pub agent_name: String,
}

/// `drain()` finished within its deadline. `kind: "drain_finished"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "drain_finished")]
pub struct WorkerDrainFinished {}

/// `drain()` hit its deadline; the caller is still expected to close the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "drain_timed_out")]
pub struct WorkerDrainTimedOut {}

/// `run()` returned, and why. `kind: "stopped"`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(emit, tag = "kind", tag_value = "stopped")]
pub struct WorkerStopped {
    /// `closed`, `worker runtime cancelled`, or the runtime error's own sentence.
    pub reason: String,
}

// --- the session report, as `_core.session_records` reads it -----------------------------------

/// `SessionReport._core_view()`'s flat dict. Every value tolerant: a malformed field is a blank
/// dashboard column, not a lost upload.
#[derive(Debug, Clone, Default)]
pub struct SessionReportInput {
    pub job_id: String,
    pub room_id: String,
    pub room: String,
    pub agent_name: String,
    pub sdk_version: String,
    pub started_at: Option<f64>,
    pub timestamp: f64,
    /// Absent is an empty object, not `null`.
    pub options: serde_json::Value,
    pub recording_options: ReportRecordingOptions,
    /// `None` when the session used nothing.
    pub model_usage: Vec<ModelUsage>,
    pub chat_history: Vec<ChatItem>,
}

/// A scalar that does not read is its default; an unknown `model_usage` type raises rather than
/// silently leaving the bill. Absent keys are defaults.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for SessionReportInput {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::{PyAnyMethods as _, PyDictMethods as _};
        let d = obj.cast::<pyo3::types::PyDict>()?;
        fn lenient<'py, T: for<'x> pyo3::FromPyObject<'x, 'py> + Default>(v: Option<pyo3::Bound<'py, pyo3::PyAny>>) -> T {
            v.and_then(|v| v.extract().ok()).unwrap_or_default()
        }
        Ok(SessionReportInput {
            job_id: lenient(d.get_item("job_id")?),
            room_id: lenient(d.get_item("room_id")?),
            room: lenient(d.get_item("room")?),
            agent_name: lenient(d.get_item("agent_name")?),
            sdk_version: lenient(d.get_item("sdk_version")?),
            started_at: lenient(d.get_item("started_at")?),
            timestamp: lenient(d.get_item("timestamp")?),
            options: match d.get_item("options")? {
                Some(v) if !v.is_none() => crate::python::value_from_py(&v, "report.options")?,
                _ => serde_json::Value::Object(Default::default()),
            },
            recording_options: match d.get_item("recording_options")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => ReportRecordingOptions::default(),
            },
            model_usage: match d.get_item("model_usage")? {
                Some(v) if !v.is_none() => v.try_iter()?.map(|u| u?.extract::<ModelUsage>()).collect::<pyo3::PyResult<_>>()?,
                _ => Vec::new(),
            },
            chat_history: match d.get_item("chat_history")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => Vec::new(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{AcceptArguments, JobDecision};

    // Catches: a decision word read differently, e.g. `terminate` as a plain reject.
    #[test]
    fn every_decision_word_maps_to_its_decision_and_an_unknown_one_is_refused() {
        let accept = || -> Result<AcceptArguments, String> {
            Ok(AcceptArguments { name: "agent".into(), ..AcceptArguments::default() })
        };
        let message = || "boom".to_owned();
        assert_eq!(
            JobDecision::from_wire("accept", accept, message),
            Ok(JobDecision::Accept(AcceptArguments {
                name: "agent".into(),
                ..AcceptArguments::default()
            }))
        );
        assert_eq!(
            JobDecision::from_wire("reject", accept, message),
            Ok(JobDecision::Reject { terminate: false })
        );
        assert_eq!(
            JobDecision::from_wire("terminate", accept, message),
            Ok(JobDecision::Reject { terminate: true })
        );
        assert_eq!(JobDecision::from_wire("unanswered", accept, message), Ok(JobDecision::Unanswered));
        assert_eq!(
            JobDecision::from_wire("failed", accept, message),
            Ok(JobDecision::Failed(crate::errors::HostError::new("boom")))
        );
        // A malformed `accept` payload is the caller's error, not an unknown word.
        assert_eq!(
            JobDecision::from_wire("accept", || Err::<AcceptArguments, _>("bad name".to_owned()), message),
            Err("bad name".to_owned())
        );
        assert!(JobDecision::from_wire("maybe", accept, message).is_err());
    }
}

#[cfg(all(test, feature = "python"))]
mod python_tests {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use super::SessionReportInput;

    // Catches: the report reader losing its tolerance for absent/`None` keys, or accepting an
    // unknown `model_usage` type.
    #[test]
    fn the_report_reader_tolerates_absent_and_none_and_refuses_an_unknown_usage() {
        Python::initialize();
        Python::attach(|py| {
            let empty: SessionReportInput = PyDict::new(py).as_any().extract().unwrap();
            assert_eq!(empty.job_id, "");
            assert_eq!(empty.started_at, None);
            assert_eq!(empty.timestamp, 0.0);
            assert!(empty.options.is_object(), "absent options are `{{}}`, not null");
            assert!(empty.model_usage.is_empty() && empty.chat_history.is_empty());

            let nones = py
                .eval(
                    c"{'job_id': None, 'room': 7, 'started_at': None, 'timestamp': 'late', \
                      'options': None, 'recording_options': None, 'model_usage': None, \
                      'chat_history': None}",
                    None,
                    None,
                )
                .unwrap();
            let read: SessionReportInput = nones.extract().unwrap();
            assert_eq!((read.job_id.as_str(), read.room.as_str()), ("", ""));
            assert_eq!((read.started_at, read.timestamp), (None, 0.0));
            assert_eq!(read.recording_options, Default::default());

            let usage = py
                .eval(
                    c"{'model_usage': [{'type': 'tts_usage', 'provider': 'p', 'model': 'm', \
                      'characters_count': 12}]}",
                    None,
                    None,
                )
                .unwrap();
            let read: SessionReportInput = usage.extract().unwrap();
            match &read.model_usage[..] {
                [crate::metrics::ModelUsage::Tts(t)] => {
                    assert_eq!((t.characters_count, t.audio_duration), (12, 0.0), "absent is zero")
                }
                other => panic!("{other:?}"),
            }

            let unknown = py.eval(c"{'model_usage': [{'type': 'nope'}]}", None, None).unwrap();
            let err = unknown.extract::<SessionReportInput>().map(|_| ()).expect_err("an unknown usage type must raise");
            assert!(err.to_string().contains("nope"), "{err}");
        });
    }
}
