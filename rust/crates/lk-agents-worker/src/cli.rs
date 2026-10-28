//! What `lk agent dev|start|console` drive (`python myagent.py start [--dev] ...`): the argument
//! contract with the Go CLI, finding the entrypoint's module, the log lines (JSON in production,
//! colored in dev), the dev channel to the CLI (server info, running jobs kept across reloads), and
//! the exit policy on SIGINT/SIGTERM. Python keeps the event loop and imports the user's module.

#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use prost::Message as _;
use serde_json::{Map, Value};

use crate::LOG;

pub use lk_agents_abi::worker::cli::{CliArgs, CliError};

pub struct WorkerCli;

impl lk_agents_abi::worker::cli::WorkerCli for WorkerCli {
    fn parse_cli(argv: Vec<String>) -> Result<CliArgs, CliError> {
        parse_cli(argv)
    }
    fn entrypoint_module(path: Option<String>) -> Result<(String, String), CliError> {
        entrypoint_module(path)
    }
}

// ------------------------------------------------------------------------------------ arguments

/// The typed command line `lk agent` passes to the host launcher.
pub fn parse_cli(argv: Vec<String>) -> Result<CliArgs, CliError> {
    let usage = |m: String| CliError::Usage(m);
    let mut args = argv.into_iter();
    let command = args.next().ok_or_else(|| usage("a command is required".into()))?;
    let (flags, switches): (&[&str], &[&str]) = match command.as_str() {
        "start" => (
            &[
                "--log-level",
                "--log-format",
                "--url",
                "--api-key",
                "--api-secret",
                "--cli-addr",
                "--reload-addr",
            ],
            &["--dev", "--simulation"],
        ),
        "console" => (&["--connect-addr"], &["--record"]),
        "download-files" => (&[], &[]),
        other => return Err(usage(format!("invalid command: {other:?}"))),
    };
    let mut values: HashMap<String, String> = HashMap::new();
    let mut on: Vec<String> = Vec::new();
    let mut entrypoint = None;
    while let Some(arg) = args.next() {
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) if k.starts_with("--") => (k.to_owned(), Some(v.to_owned())),
            _ => (arg.clone(), None),
        };
        if flags.contains(&key.as_str()) {
            let value = match inline {
                Some(v) => v,
                None => args
                    .next()
                    .ok_or_else(|| usage(format!("argument {key}: expected one argument")))?,
            };
            values.insert(key, value);
        } else if switches.contains(&key.as_str()) && inline.is_none() {
            on.push(key);
        } else if !arg.starts_with('-') && entrypoint.is_none() && command != "download-files" {
            entrypoint = Some(arg);
        } else {
            return Err(usage(format!("unrecognized arguments: {arg}")));
        }
    }
    let get = |k: &str| values.get(k).cloned();
    let flag = |k: &str| on.iter().any(|f| f == k);
    Ok(match command.as_str() {
        "start" => {
            let log_format = get("--log-format").unwrap_or_else(|| "json".into());
            if !["json", "colored"].contains(&log_format.as_str()) {
                return Err(usage(format!(
                    "argument --log-format: invalid choice: {log_format:?} (choose from 'json', 'colored')"
                )));
            }
            let cli_addr = get("--cli-addr").or_else(|| get("--reload-addr"));
            if cli_addr.is_some() && !flag("--dev") {
                return Err(usage("--cli-addr requires --dev".into()));
            }
            CliArgs::Start {
                entrypoint,
                log_level: get("--log-level").unwrap_or_else(|| "INFO".into()),
                log_style: match (log_format.as_str(), flag("--dev"), flag("--simulation")) {
                    ("json", false, _) => "json",
                    (_, _, true) => "compact",
                    _ => "dev",
                }
                .into(),
                log_format,
                dev: flag("--dev"),
                cli_addr,
                simulation: flag("--simulation"),
                ws_url: get("--url").filter(|value| !value.is_empty()),
                api_key: get("--api-key").filter(|value| !value.is_empty()),
                api_secret: get("--api-secret").filter(|value| !value.is_empty()),
            }
        }
        "console" => CliArgs::Console {
            entrypoint,
            connect_addr: get("--connect-addr").ok_or_else(|| {
                usage("the following arguments are required: --connect-addr".into())
            })?,
            record: flag("--record"),
        },
        _ => CliArgs::DownloadFiles,
    })
}

// ----------------------------------------------------------------------------------- entrypoint

/// Where the default entrypoint is looked for, in order.
const DEFAULT_PATHS: [&str; 6] =
    ["main.py", "app.py", "agent.py", "app/main.py", "app/app.py", "app/agent.py"];

/// The entrypoint's module, as `(import string, directory to put on sys.path)`: `path`, or the
/// first default file in the working directory; a file inside packages imports through them.
pub fn entrypoint_module(path: Option<String>) -> Result<(String, String), CliError> {
    let path = match path {
        Some(p) => PathBuf::from(p),
        None => DEFAULT_PATHS
            .iter()
            .map(PathBuf::from)
            .find(|p| p.is_file())
            .ok_or(CliError::NoDefaultFile)?,
    };
    if !path.exists() {
        return Err(CliError::NoPath(path.display().to_string()));
    }
    let absolute =
        std::path::absolute(&path).map_err(|_| CliError::NoPath(path.display().to_string()))?;
    let module = match absolute.file_stem() {
        Some(stem) if stem == "__init__" && absolute.is_file() => {
            absolute.parent().unwrap_or(&absolute).to_path_buf()
        }
        _ => absolute.clone(),
    };
    let mut parts = vec![stem(&module)];
    let mut sys_path = module.parent().unwrap_or(Path::new("/")).to_path_buf();
    for parent in module.ancestors().skip(1) {
        if !parent.join("__init__.py").is_file() {
            break;
        }
        parts.insert(0, stem(parent));
        sys_path = parent.parent().unwrap_or(Path::new("/")).to_path_buf();
    }
    Ok((parts.join("."), sys_path.display().to_string()))
}

fn stem(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

// ------------------------------------------------------------------------------------ log lines

/// What a log line is made of: the record's level, logger, process and time, a `dict` message
/// when one was logged, and its extra fields (the record's attributes outside the standard ones).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LogFields {
    pub level: String,
    pub name: String,
    pub pid: Option<i64>,
    pub created: f64,
    pub msg: Option<Map<String, Value>>,
    pub extra: Map<String, Value>,
}

/// `logging.LogRecord`'s own attributes, which a line never repeats as extras.
const RESERVED: [&str; 23] = [
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
];

/// One log line. `style` is `json` (production: one object per line, Python's `json.dumps`
/// spelling), or `console`, `compact` or `dev` (colored, the console's line reset first, the
/// compact one without dashes). `message` is the formatted message, `asctime` the record's time as
/// the host formats it, `exception` and `stack` the tracebacks.
#[lk_agents_abi::host(export(pid = false))]
pub fn format_log_record(
    style: &str,
    record: LogFields,
    message: String,
    asctime: String,
    exception: Option<String>,
    stack: Option<String>,
    ensure_ascii: bool,
) -> String {
    if style == "json" {
        return json_line(record, message, exception, stack, ensure_ascii);
    }
    let esc = |codes: &str| format!("\x1b[{codes}m");
    let color = match record.level.as_str() {
        "DEBUG" => esc("36"),
        "INFO" => esc("32"),
        "WARNING" => esc("33"),
        "ERROR" => esc("31"),
        "CRITICAL" => esc("1;31"),
        "DEV" => esc("35"),
        _ => String::new(),
    };
    let (reset, gray) = (esc("0"), esc("90"));
    let extra = if record.extra.is_empty() {
        String::new()
    } else {
        py_json(&Value::Object(record.extra), false)
    };
    let (level, name) = (format!("{:<4}", record.level), record.name);
    let mut line = match style {
        "console" => {
            format!("\r{asctime} {color}{level}{reset} {name} {message} {gray}{extra}{reset}")
        }
        "compact" => {
            format!("{asctime} {color}{level}{reset} {name} {message} {gray}{extra}{reset}")
        }
        _ => format!("{asctime} - {color}{level}{reset} {name} - {message} {gray}{extra}{reset}"),
    };
    for trace in [exception, stack].into_iter().flatten() {
        line.push('\n');
        line.push_str(&trace);
    }
    line
}

fn json_line(
    record: LogFields,
    message: String,
    exception: Option<String>,
    stack: Option<String>,
    ensure_ascii: bool,
) -> String {
    let mut line = Map::new();
    let mut fields = match record.msg {
        Some(dict) => {
            line.insert("message".into(), "".into());
            dict
        }
        None => {
            line.insert("message".into(), message.into());
            Map::from_iter([
                ("level".into(), record.level.into()),
                ("name".into(), record.name.into()),
            ])
        }
    };
    if let Some(exc) = exception.filter(|_| fields.get("exc_info").is_none_or(falsy)) {
        fields.insert("exc_info".into(), exc.into());
    }
    if let Some(stack) = stack.filter(|_| fields.get("stack_info").is_none_or(falsy)) {
        fields.insert("stack_info".into(), stack.into());
    }
    line.extend(fields);
    line.extend(record.extra);
    if !line.contains_key("pid") {
        line.insert("pid".into(), record.pid.map_or(Value::Null, Value::from));
    }
    line.insert("timestamp".into(), iso_utc(record.created).into());
    py_json(&Value::Object(line), ensure_ascii)
}

fn falsy(v: &Value) -> bool {
    matches!(v, Value::Null | Value::Bool(false))
        || v.as_str() == Some("")
        || v.as_f64() == Some(0.0)
}

/// `datetime.fromtimestamp(created, tz=timezone.utc).isoformat()`.
fn iso_utc(created: f64) -> String {
    let micros = (created * 1e6).round_ties_even() as i64;
    let Some(at) = chrono::DateTime::from_timestamp_micros(micros) else { return String::new() };
    if micros.rem_euclid(1_000_000) == 0 {
        at.format("%Y-%m-%dT%H:%M:%S+00:00").to_string()
    } else {
        at.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()
    }
}

/// `json.dumps(value, ensure_ascii=..)`: `", "` and `": "` separators, Python's float spelling.
pub fn py_json(value: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_json(value, ensure_ascii, &mut out);
    out
}

fn write_json(value: &Value, ascii: bool, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => out.push_str(&i.to_string()),
            (_, Some(u), _) => out.push_str(&u.to_string()),
            (_, _, Some(f)) => out.push_str(&py_float(f)),
            _ => out.push_str(&n.to_string()),
        },
        Value::String(s) => write_str(s, ascii, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(item, ascii, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(k, ascii, out);
                out.push_str(": ");
                write_json(v, ascii, out);
            }
            out.push('}');
        }
    }
}

fn write_str(s: &str, ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (ascii && !c.is_ascii()) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `repr(float)`: the shortest round trip, in exponent form outside `1e-4 <= |f| < 1e16`.
fn py_float(f: f64) -> String {
    if !f.is_finite() {
        return if f.is_nan() {
            "NaN".into()
        } else if f > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    let abs = f.abs();
    if abs != 0.0 && !(1e-4..1e16).contains(&abs) {
        let s = format!("{f:e}");
        let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
        let exp: i32 = exp.parse().unwrap_or(0);
        return format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let s = f.to_string();
    if s.contains('.') { s } else { format!("{s}.0") }
}

#[cfg(feature = "python")]
mod host {
    use super::{LogFields, RESERVED};
    use pyo3::prelude::*;
    use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
    use serde_json::{Map, Value};

    /// A record's attribute as JSON, as the reference's encoder writes it: a date by its
    /// `isoformat()`, anything else not JSON by its `str()`.
    fn lenient(obj: &Bound<'_, PyAny>) -> Value {
        if obj.is_none() {
            return Value::Null;
        }
        if let Ok(b) = obj.cast::<PyBool>() {
            return Value::Bool(b.is_true());
        }
        if obj.is_instance_of::<PyInt>() {
            if let Ok(i) = obj.extract::<i64>() {
                return i.into();
            }
        }
        if obj.is_instance_of::<PyFloat>() {
            if let Some(n) = obj.extract::<f64>().ok().and_then(serde_json::Number::from_f64) {
                return Value::Number(n);
            }
        }
        if let Ok(s) = obj.cast::<PyString>() {
            return Value::String(s.to_string());
        }
        if let Ok(d) = obj.cast::<PyDict>() {
            return Value::Object(d.iter().map(|(k, v)| (key(&k), lenient(&v))).collect());
        }
        if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
            if let Ok(items) = obj.try_iter() {
                return Value::Array(items.flatten().map(|i| lenient(&i)).collect());
            }
        }
        if let Ok(iso) = obj.call_method0("isoformat") {
            return Value::String(key(&iso));
        }
        obj.str().map(|s| Value::String(s.to_string())).unwrap_or(Value::Null)
    }

    fn key(obj: &Bound<'_, PyAny>) -> String {
        obj.str().map(|s| s.to_string()).unwrap_or_default()
    }

    /// A `logging.LogRecord`.
    impl<'a, 'py> FromPyObject<'a, 'py> for LogFields {
        type Error = PyErr;

        fn extract(record: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let msg = record.getattr("msg")?;
            let mut extra = Map::new();
            for (k, v) in record.getattr("__dict__")?.cast::<PyDict>()?.iter() {
                let k = key(&k);
                if !RESERVED.contains(&k.as_str()) && !k.starts_with('_') {
                    extra.insert(k, lenient(&v));
                }
            }
            Ok(LogFields {
                level: record.getattr("levelname")?.extract()?,
                name: record.getattr("name")?.extract()?,
                pid: record.getattr("process")?.extract()?,
                created: record.getattr("created")?.extract()?,
                msg: msg.cast::<PyDict>().ok().map(|d| match lenient(d.as_any()) {
                    Value::Object(map) => map,
                    _ => Map::new(),
                }),
                extra,
            })
        }
    }
}

// ---------------------------------------------------------------------------------- dev channel

/// A job running under the dev server, kept across reloads: the `Job` protobuf as bytes, the
/// accept arguments (name, identity, metadata), the server URL, the token, the worker id, and
/// whether it is a fake job.
pub type DevJob = (Vec<u8>, String, String, String, String, String, String, bool);

/// The agent server the dev channel reports on.
#[lk_agents_abi::host(proxy(live))]
#[async_trait::async_trait]
pub trait DevServer: Send + Sync {
    /// The jobs running now.
    fn running_jobs(&self) -> Vec<DevJob>;
    /// Resume the jobs the CLI kept from before the reload, on this server's URL and a fresh token.
    async fn reload_jobs(&self, jobs: Vec<DevJob>) -> Result<(), HostError>;
}

fn to_proto(job: DevJob) -> livekit_protocol::agent::RunningAgentJobInfo {
    let (job, accept_name, accept_identity, accept_metadata, url, token, worker_id, mock_job) = job;
    livekit_protocol::agent::RunningAgentJobInfo {
        job,
        accept_name,
        accept_identity,
        accept_metadata,
        url,
        token,
        worker_id,
        mock_job,
    }
}

fn from_proto(pb: livekit_protocol::agent::RunningAgentJobInfo) -> DevJob {
    (
        pb.job,
        pb.accept_name,
        pb.accept_identity,
        pb.accept_metadata,
        pb.url,
        pb.token,
        pb.worker_id,
        pb.mock_job,
    )
}

/// The dev channel to `lk agent dev` at `cli_addr` (`host:port`), length-prefixed
/// `AgentDevMessage`s: the server's name and URL first, then the jobs kept from before the reload
/// asked for and resumed (on `url`, each with its token extended by the server's API key and
/// secret), then every running-jobs request answered, until the CLI closes it. A failure is
/// logged: the worker runs on without the channel.
#[lk_agents_abi::host(export)]
pub async fn dev_channel(
    cli_addr: String,
    agent_name: String,
    url: String,
    api_key: String,
    api_secret: String,
    server: Arc<dyn DevServer>,
) {
    if let Err(err) =
        serve_dev_channel(cli_addr, agent_name, url, (api_key, api_secret), server).await
    {
        log::error!(target: LOG, "dev channel failed: {err}");
    }
}

async fn serve_dev_channel(
    cli_addr: String,
    agent_name: String,
    url: String,
    (api_key, api_secret): (String, String),
    server: Arc<dyn DevServer>,
) -> Result<(), HostError> {
    use livekit_protocol::agent::{
        AgentDevMessage, GetRunningAgentJobsRequest, GetRunningAgentJobsResponse, ServerInfo,
        agent_dev_message::Message,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let io = |e: std::io::Error| HostError::new(format!("dev channel: {e}"));
    let mut stream = tokio::net::TcpStream::connect(&cli_addr).await.map_err(io)?;
    async fn send(stream: &mut tokio::net::TcpStream, message: Message) -> std::io::Result<()> {
        let bytes = AgentDevMessage { message: Some(message) }.encode_to_vec();
        stream.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
        stream.write_all(&bytes).await?;
        stream.flush().await
    }
    async fn recv(stream: &mut tokio::net::TcpStream) -> std::io::Result<Option<Message>> {
        let mut len = [0u8; 4];
        stream.read_exact(&mut len).await?;
        let mut bytes = vec![0u8; u32::from_be_bytes(len) as usize];
        stream.read_exact(&mut bytes).await?;
        let decoded = AgentDevMessage::decode(bytes.as_slice())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(decoded.message)
    }

    send(&mut stream, Message::ServerInfo(ServerInfo { agent_name, url: url.clone() }))
        .await
        .map_err(io)?;
    send(&mut stream, Message::GetRunningJobsRequest(GetRunningAgentJobsRequest {}))
        .await
        .map_err(io)?;
    let Ok(first) = recv(&mut stream).await else { return Ok(()) };
    if let Some(Message::GetRunningJobsResponse(kept)) = first {
        if !kept.jobs.is_empty() {
            log::info!(target: LOG, "reloading {} job(s)", kept.jobs.len());
            let mut jobs = Vec::with_capacity(kept.jobs.len());
            for mut job in kept.jobs {
                job.token = crate::auth::extend_job_token(&job.token, &api_key, &api_secret)
                    .map_err(|e| HostError::new(format!("cannot reload a job: {e}")))?;
                job.url = url.clone();
                jobs.push(from_proto(job));
            }
            server.reload_jobs(jobs).await?;
        }
    }
    // Asked before the CLI restarts this process: what runs now, to resume after the reload.
    while let Ok(message) = recv(&mut stream).await {
        if let Some(Message::GetRunningJobsRequest(_)) = message {
            let jobs = server.running_jobs().into_iter().map(to_proto).collect();
            let reply = Message::GetRunningJobsResponse(GetRunningAgentJobsResponse { jobs });
            if send(&mut stream, reply).await.is_err() {
                break;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------- exit policy

/// How long an exit scheduled on the event loop may go unserved before the loop is taken to be
/// blocked by synchronous code and the exit is raised inside it. Well under the ~30 s an
/// orchestrator leaves between SIGTERM and SIGKILL, so the drain still gets its time.
pub const EXIT_ESCALATION_TIMEOUT: Duration = Duration::from_secs(3);

/// Final event-loop cleanup runs after the server has released its owned processes.
pub const LOOP_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// No exit asked yet.
    Running,
    /// The first signal scheduled the exit on the loop.
    Exiting,
    /// The worker drains and closes: a signal is the operator forcing the exit.
    Draining,
    /// The host is cancelling tasks, finalizing generators and closing its executor.
    Closing,
    /// Every host cleanup operation settled and the loop closed.
    Closed,
}

#[derive(Debug)]
pub struct ExitSignals {
    phase: Phase,
    /// The exit is served on the loop (the watchdog stands down).
    served: bool,
    /// The watchdog re-signalled the main thread: the next signal is its own.
    escalating: bool,
}

impl ExitSignals {
    pub fn new() -> Self {
        Self { phase: Phase::Running, served: false, escalating: false }
    }

    /// A signal arrived; `exit_served` says whether the loop already served the exit. Answers
    /// `"schedule"` (schedule the exit on the loop), `"raise"` (raise it here: the loop is
    /// blocked), `"force"` (exit the process now) or `"ignore"`.
    pub fn signal(&mut self, exit_served: bool) -> &'static str {
        if std::mem::take(&mut self.escalating) {
            // The watchdog's own signal: a raise only while the exit is still unserved, else the
            // loop resumed and won the race and raising could skip the drain.
            return if exit_served || self.served { "ignore" } else { "raise" };
        }
        match self.phase {
            Phase::Running => {
                self.phase = Phase::Exiting;
                "schedule"
            }
            Phase::Exiting | Phase::Draining | Phase::Closing => "force",
            Phase::Closed => "ignore",
        }
    }

    /// The loop served the exit: the watchdog stands down.
    pub fn served(&mut self) {
        self.served = true;
    }

    /// The worker drains and closes: from now on a signal forces the exit.
    pub fn draining(&mut self) {
        self.phase = Phase::Draining;
        self.served = true;
    }

    pub fn escalation_due(&mut self) -> bool {
        if !self.escalation_pending() {
            return false;
        }
        self.escalating = true;
        true
    }

    pub fn escalation_pending(&self) -> bool {
        !self.served && self.phase == Phase::Exiting
    }

    /// Returns true once, so repeated closing callers share one watchdog.
    pub fn closing(&mut self) -> bool {
        if matches!(self.phase, Phase::Closing | Phase::Closed) {
            return false;
        }
        self.phase = Phase::Closing;
        self.served = true;
        true
    }

    pub fn closed(&mut self) {
        self.phase = Phase::Closed;
    }

    pub fn cleanup_due(&self) -> bool {
        self.phase == Phase::Closing
    }
}

impl Default for ExitSignals {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn the_go_cli_contract() -> Result<(), CliError> {
        let CliArgs::Start { entrypoint, cli_addr, log_format, log_style, ws_url, .. } = parse_cli(
            argv("start agent.py --dev --log-level DEBUG --reload-addr 127.0.0.1:9 --url wss://x"),
        )?
        else {
            panic!("expected start")
        };
        assert_eq!(entrypoint.as_deref(), Some("agent.py"));
        assert_eq!(cli_addr.as_deref(), Some("127.0.0.1:9"));
        assert_eq!(log_format, "json");
        assert_eq!(log_style, "dev", "colored in development");
        assert_eq!(ws_url.as_deref(), Some("wss://x"));
        let CliArgs::Start { log_style, .. } = parse_cli(argv("start"))? else {
            panic!("expected start")
        };
        assert_eq!(log_style, "json");
        let CliArgs::Start { log_style, .. } =
            parse_cli(argv("start --log-format colored --simulation"))?
        else {
            panic!("expected start")
        };
        assert_eq!(log_style, "compact");
        assert!(parse_cli(argv("start --cli-addr x")).is_err(), "the dev channel needs --dev");
        let CliArgs::Console { connect_addr, record, .. } =
            parse_cli(argv("console --connect-addr=127.0.0.1:1 --record"))?
        else {
            panic!("expected console")
        };
        assert_eq!(connect_addr, "127.0.0.1:1");
        assert!(record);
        assert!(matches!(parse_cli(argv("download-files"))?, CliArgs::DownloadFiles));
        assert!(parse_cli(argv("console")).is_err());
        assert!(parse_cli(argv("start --log-format xml")).is_err());
        assert!(parse_cli(argv("start --nope")).is_err());
        Ok(())
    }

    #[test]
    fn json_lines_read_as_the_reference() {
        let record = LogFields {
            level: "INFO".into(),
            name: "livekit.agents".into(),
            pid: Some(7),
            created: 1_700_000_000.5,
            msg: None,
            extra: Map::from_iter([
                ("job_id".to_owned(), "AJ_1".into()),
                ("lag".to_owned(), 0.00001.into()),
            ]),
        };
        assert_eq!(
            format_log_record("json", record, "héllo".into(), String::new(), None, None, true),
            r#"{"message": "h\u00e9llo", "level": "INFO", "name": "livekit.agents", "job_id": "AJ_1", "lag": 1e-05, "pid": 7, "timestamp": "2023-11-14T22:13:20.500000+00:00"}"#
        );
    }

    #[test]
    fn colored_lines_carry_their_extras() {
        let record = LogFields {
            level: "WARNING".into(),
            name: "x".into(),
            extra: Map::from_iter([("a".to_owned(), 1.into())]),
            ..Default::default()
        };
        let line = format_log_record(
            "compact",
            record,
            "hi".into(),
            "12:00:00".into(),
            Some("Traceback".into()),
            None,
            false,
        );
        assert_eq!(
            line,
            "12:00:00 \x1b[33mWARNING\x1b[0m x hi \x1b[90m{\"a\": 1}\x1b[0m\nTraceback"
        );
    }

    #[test]
    fn the_exit_policy() {
        let mut exit = ExitSignals::new();
        assert_eq!(exit.signal(false), "schedule");
        exit.served();
        assert_eq!(exit.signal(true), "force", "a second signal forces the exit");
        exit.draining();
        assert_eq!(exit.signal(true), "force");
    }

    #[test]
    fn final_loop_cleanup_remains_owned_until_the_loop_closed() {
        let mut exit = ExitSignals::new();
        assert_eq!(exit.signal(false), "schedule");
        assert!(exit.closing());
        assert!(!exit.escalation_due(), "a stale first-signal watchdog cannot skip cleanup");
        assert!(!exit.closing(), "repeated cleanup callers share the same deadline");
        assert!(exit.cleanup_due(), "draining alone did not close the host loop");
        assert_eq!(exit.signal(true), "force");
        exit.closed();
        assert!(!exit.cleanup_due());
        assert!(!exit.closing(), "a completed cleanup cannot arm another watchdog");
        assert_eq!(exit.signal(true), "ignore");
    }
}
