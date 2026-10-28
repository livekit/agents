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
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[python(dataclass, module = "worker_options")]
#[binding(extract)]
pub struct WorkerPermissions {
    #[python(default = "True")]
    #[binding(default = true)]
    pub can_publish: bool,
    #[python(default = "True")]
    #[binding(default = true)]
    pub can_subscribe: bool,
    #[python(default = "True")]
    #[binding(default = true)]
    pub can_publish_data: bool,
    #[python(default = "True")]
    #[binding(default = true)]
    pub can_update_metadata: bool,
    /// `models.TrackSource` values, as raw enum numbers.
    #[python(default_factory = "list")]
    pub can_publish_sources: Vec<i32>,
    #[python(default = "False")]
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
#[crate::class(from_py_object)]
#[binding(class)]
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
    #[python(ty = "bytes")]
    pub proto: Vec<u8>,
}

/// A job offered to the host. The host returns a [`JobDecision`].
#[derive(Debug, Clone, PartialEq, Host)]
#[crate::class(from_py_object)]
#[binding(class)]
pub struct JobRequest {
    #[python(ty = "JobInfo")]
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
#[crate::class(from_py_object)]
#[binding(class)]
pub struct RunningJobInfo {
    #[python(ty = "AcceptArguments")]
    pub accept_arguments: AcceptArguments,
    #[python(ty = "JobInfo")]
    pub job: JobInfo,
    pub url: String,
    pub token: String,
    pub worker_id: String,
    pub fake_job: bool,
    /// The dispatch timeline in unix seconds, 0.0 for a job no dispatch offered (simulated):
    /// the `AvailabilityRequest` arrived,
    pub received_at: f64,
    /// the request was accepted,
    pub accepted_at: f64,
    /// and the `JobAssignment` arrived.
    pub assigned_at: f64,
}

/// Acceptance returns before assignment and launch; the worker owns the assignment deadline.
#[derive(Debug, Clone, PartialEq)]
pub enum JobDecision {
    Accept(AcceptArguments),
    Reject {
        terminate: bool,
    },
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

/// Accepted participant fields; application attributes retain their own string keys.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[crate::class]
#[binding(class)]
pub struct AcceptArguments {
    pub name: String,
    pub identity: String,
    pub metadata: String,
    /// Merged after `lk.agent.name`, so a user attribute of that name wins.
    pub attributes: Vec<(String, String)>,
}

#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for AcceptArguments {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::{PyAnyMethods as _, PyDictMethods as _};
        let d = obj.cast::<pyo3::types::PyDict>()?;
        let text = |key: &str| -> pyo3::PyResult<String> {
            Ok(match d.get_item(key)? {
                Some(v) => v.extract()?,
                None => String::new(),
            })
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

/// Registration accepted by the server, delivered as a typed record.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "worker_server", emit))]
pub struct RegisteredInfo {
    /// The id the server issued.
    pub worker_id: String,
    pub region: String,
    /// The protocol version the server chose.
    pub protocol: i32,
    pub deployment: String,
    pub agent_name: String,
}

/// The worker notification and server callback share one registration schema.
pub type WorkerRegistered = RegisteredInfo;

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
    pub events: Vec<serde_json::Value>,
    pub omitted_events: usize,
}

/// A scalar that does not read is its default; an unknown `model_usage` type raises rather than
/// silently leaving the bill. Absent keys are defaults.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for SessionReportInput {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::{PyAnyMethods as _, PyDictMethods as _};
        let d = obj.cast::<pyo3::types::PyDict>()?;
        fn lenient<'py, T: for<'x> pyo3::FromPyObject<'x, 'py> + Default>(
            v: Option<pyo3::Bound<'py, pyo3::PyAny>>,
        ) -> T {
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
                Some(v) if !v.is_none() => v
                    .try_iter()?
                    .map(|u| u?.extract::<ModelUsage>())
                    .collect::<pyo3::PyResult<_>>()?,
                _ => Vec::new(),
            },
            chat_history: match d.get_item("chat_history")? {
                Some(v) if !v.is_none() => v.extract()?,
                _ => Vec::new(),
            },
            omitted_events: lenient(d.get_item("omitted_events")?),
            events: match d.get_item("events")? {
                Some(v) if !v.is_none() => match crate::python::value_from_py(&v, "report.events")?
                {
                    serde_json::Value::Array(events) => events,
                    _ => {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "report.events must be an array",
                        ));
                    }
                },
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
        assert_eq!(
            JobDecision::from_wire("unanswered", accept, message),
            Ok(JobDecision::Unanswered)
        );
        assert_eq!(
            JobDecision::from_wire("failed", accept, message),
            Ok(JobDecision::Failed(crate::errors::HostError::new("boom")))
        );
        // A malformed `accept` payload is the caller's error, not an unknown word.
        assert_eq!(
            JobDecision::from_wire(
                "accept",
                || Err::<AcceptArguments, _>("bad name".to_owned()),
                message
            ),
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
            let err = unknown
                .extract::<SessionReportInput>()
                .map(|_| ())
                .expect_err("an unknown usage type must raise");
            assert!(err.to_string().contains("nope"), "{err}");
        });
    }
}

// --- server launch configuration -----------------------------------------------------------

/// Process-pool settings resolved before the host launch callback.
#[derive(Debug, Clone, PartialEq, Host)]
#[crate::class(from_py_object)]
#[binding(class, new)]
pub struct ProcessPoolOptions {
    pub num_idle_processes: u32,
    pub initialize_timeout: f64,
    pub close_timeout: f64,
    pub session_end_timeout: f64,
    pub memory_warn_mb: f64,
    pub memory_limit_mb: f64,
    pub http_proxy: Option<String>,
}

/// Resolved server settings handed to the process launch adapter.
#[derive(Debug, Clone, PartialEq, Host)]
#[crate::class(from_py_object)]
#[binding(class, new)]
pub struct ProcessStartOptions {
    pub devmode: bool,
    pub job_processes: bool,
    pub env: Vec<(String, String)>,
    pub job_executor_type: String,
    pub multiprocessing_context: String,
    #[python(ty = "ProcessPoolOptions")]
    pub pool: ProcessPoolOptions,
}

/// A sparse update to the server configuration; omission and an explicit null remain distinct.
#[derive(Clone, Default, Host)]
#[python(dataclass, module = "worker_options")]
#[binding(extract)]
pub struct ServerOptions {
    #[python(
        ty = "NotGivenOr[str]",
        default = "NOT_GIVEN",
        import = "..types:NOT_GIVEN, NotGivenOr"
    )]
    pub ws_url: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub api_key: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub api_secret: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub agent_name: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub host: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[int]", default = "NOT_GIVEN")]
    pub port: crate::Maybe<u16>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub load_threshold: crate::Maybe<f32>,
    #[python(ty = "NotGivenOr[int]", default = "NOT_GIVEN")]
    pub num_idle_processes: crate::Maybe<u32>,
    #[python(ty = "NotGivenOr[float | None]", default = "NOT_GIVEN")]
    pub drain_timeout: crate::Maybe<Option<f64>>,
    #[python(ty = "NotGivenOr[int]", default = "NOT_GIVEN")]
    pub max_retry: crate::Maybe<u32>,
    #[python(ty = "NotGivenOr[WorkerPermissions]", default = "NOT_GIVEN")]
    pub permissions: crate::Maybe<WorkerPermissions>,
    #[python(ty = "NotGivenOr[str | None]", default = "NOT_GIVEN")]
    pub http_proxy: crate::Maybe<Option<String>>,
    #[python(ty = "NotGivenOr[int | None]", default = "NOT_GIVEN")]
    pub prometheus_port: crate::Maybe<Option<u16>>,
    #[python(ty = "NotGivenOr[str | None]", default = "NOT_GIVEN")]
    pub prometheus_multiproc_dir: crate::Maybe<Option<String>>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub version: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub job_executor_type: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[str]", default = "NOT_GIVEN")]
    pub multiprocessing_context: crate::Maybe<String>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub job_memory_warn_mb: crate::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub job_memory_limit_mb: crate::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub shutdown_process_timeout: crate::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub session_end_timeout: crate::Maybe<f64>,
    #[python(ty = "NotGivenOr[float]", default = "NOT_GIVEN")]
    pub initialize_process_timeout: crate::Maybe<f64>,
}

/// What the agent server reads from where it runs: the CPU count its defaults are sized by, and
/// the base URL its text endpoints are reached at.
#[crate::host(functions)]
pub trait ServerEnvironment {
    /// Logical CPUs, fractional under a cgroup quota. Reads, starts nothing.
    #[host(function)]
    fn cpu_count() -> f64;

    /// `https://<project>/agents/<agent>/<deployment>` (`AgentServer._endpoint_url_base`), when
    /// the URL and the agent are known; a worker registered without a deployment is `default`.
    #[host(function)]
    fn endpoint_url_base(ws_url: String, agent_name: String, deployment: String) -> Option<String>;
}
