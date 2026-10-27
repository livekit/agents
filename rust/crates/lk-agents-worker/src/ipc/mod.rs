//! The job process IPC: the worker and each job or inference process over one socket.
//!
//! - The codec: a frame is a 4-byte big-endian length, then the message: a big-endian `i32`
//!   `MSG_ID`, then its fields in [`MESSAGES`] order (the reference's `ipc/proto.py` wire).
//! - [`ProcSupervisor`]: the parent's side of one job or inference process. It runs the
//!   initialize handshake, the ping/pong and memory monitoring, the graceful shutdown and the kill
//!   sequence. Spawning the process stays with the host language.
//! - [`IpcClient`]: the child's side of the socket, with the self health check. The child answers
//!   pings from its own event loop, so a pong still measures that loop's responsiveness.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{Instant, MissedTickBehavior};

pub mod pool;
#[cfg(feature = "python")]
pub mod python;
#[cfg(test)]
mod tests;

// ------------------------------------------------------------------------------------ codec

pub const INITIALIZE_REQUEST: i32 = 0;
pub const INITIALIZE_RESPONSE: i32 = 1;
pub const PING_REQUEST: i32 = 2;
pub const PONG_RESPONSE: i32 = 3;
pub const START_JOB_REQUEST: i32 = 4;
pub const SHUTDOWN_REQUEST: i32 = 5;
pub const EXITING: i32 = 6;
pub const INFERENCE_REQUEST: i32 = 7;
pub const INFERENCE_RESPONSE: i32 = 8;
pub const DUMP_STACK_TRACE_REQUEST: i32 = 9;
pub const SHUTDOWN_REQUEST_ACK: i32 = 10;
pub const SHUTTING_DOWN: i32 = 11;

/// A field's wire type, as `channel.py` writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    /// One byte, nonzero is true.
    Bool,
    /// `struct "f"`: a native-endian `f32`.
    Float,
    /// `struct "d"`: a native-endian `f64`.
    Double,
    /// An 8-byte big-endian unsigned integer.
    Long,
    /// A 4-byte big-endian length, then UTF-8.
    Str,
    /// A 4-byte big-endian length, then the bytes.
    Bytes,
    /// A bool, then the bytes when it is true.
    OptBytes,
}

use FieldKind as K;

type Schema = &'static [(&'static str, FieldKind)];

/// Each message's name and fields in wire order; the index is its `MSG_ID`.
pub const MESSAGES: [(&str, Schema); 12] = [
    (
        "InitializeRequest",
        &[
            ("asyncio_debug", K::Bool),
            ("ping_interval", K::Float),
            ("ping_timeout", K::Float),
            ("high_ping_threshold", K::Float),
            ("http_proxy", K::Str),
        ],
    ),
    ("InitializeResponse", &[("error", K::Str)]),
    ("PingRequest", &[("timestamp", K::Long)]),
    ("PongResponse", &[("last_timestamp", K::Long), ("timestamp", K::Long)]),
    (
        "StartJobRequest",
        &[
            ("job", K::Bytes),
            ("name", K::Str),
            ("identity", K::Str),
            ("metadata", K::Str),
            ("url", K::Str),
            ("token", K::Str),
            ("worker_id", K::Str),
            ("fake_job", K::Bool),
            ("received_at", K::Double),
            ("accepted_at", K::Double),
            ("assigned_at", K::Double),
            ("launched_at", K::Double),
        ],
    ),
    ("ShutdownRequest", &[("reason", K::Str)]),
    ("Exiting", &[("reason", K::Str)]),
    ("InferenceRequest", &[("method", K::Str), ("request_id", K::Str), ("data", K::Bytes)]),
    ("InferenceResponse", &[("request_id", K::Str), ("data", K::OptBytes), ("error", K::Str)]),
    ("DumpStackTraceRequest", &[]),
    ("ShutdownRequestAck", &[]),
    ("ShuttingDown", &[]),
];

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Float(f32),
    Double(f64),
    Long(u64),
    Str(String),
    Bytes(Vec<u8>),
    /// An [`FieldKind::OptBytes`] without data.
    Absent,
}

impl Value {
    fn fits(&self, kind: FieldKind) -> bool {
        matches!(
            (self, kind),
            (Value::Bool(_), K::Bool)
                | (Value::Float(_), K::Float)
                | (Value::Double(_), K::Double)
                | (Value::Long(_), K::Long)
                | (Value::Str(_), K::Str)
                | (Value::Bytes(_), K::Bytes)
                | (Value::Bytes(_) | Value::Absent, K::OptBytes)
        )
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CodecError {
    #[error("unknown IPC message id {0}")]
    UnknownId(i32),
    #[error("truncated IPC message")]
    Truncated,
    #[error("IPC message field {0} is not valid UTF-8")]
    Utf8(&'static str),
    #[error("IPC message {0} takes {1} fields")]
    Arity(&'static str, usize),
    #[error("IPC message field {0} has the wrong type")]
    Type(&'static str),
}

fn schema(id: i32) -> Result<(&'static str, Schema), CodecError> {
    usize::try_from(id).ok().and_then(|i| MESSAGES.get(i)).copied().ok_or(CodecError::UnknownId(id))
}

/// One message: its id and its fields in [`MESSAGES`] order.
#[derive(Clone, Debug, PartialEq)]
pub struct IpcMessage {
    id: i32,
    fields: Vec<Value>,
}

impl IpcMessage {
    pub fn new(id: i32, fields: Vec<Value>) -> Result<Self, CodecError> {
        let (name, schema) = schema(id)?;
        if fields.len() != schema.len() {
            return Err(CodecError::Arity(name, schema.len()));
        }
        for ((field, kind), value) in schema.iter().zip(&fields) {
            if !value.fits(*kind) {
                return Err(CodecError::Type(field));
            }
        }
        Ok(Self { id, fields })
    }

    pub fn ping(timestamp: u64) -> Self {
        Self { id: PING_REQUEST, fields: vec![Value::Long(timestamp)] }
    }

    pub fn shutdown_request() -> Self {
        Self { id: SHUTDOWN_REQUEST, fields: vec![Value::Str(String::new())] }
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn name(&self) -> &'static str {
        MESSAGES[self.id as usize].0
    }

    pub fn fields(&self) -> impl Iterator<Item = (&'static str, &Value)> {
        MESSAGES[self.id as usize].1.iter().map(|(name, _)| *name).zip(&self.fields)
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.fields().find(|(field, _)| *field == name).map(|(_, value)| value)
    }

    fn get_str(&self, name: &str) -> &str {
        match self.get(name) {
            Some(Value::Str(s)) => s,
            _ => "",
        }
    }

    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    fn get_bytes(&self, name: &str) -> Option<&[u8]> {
        match self.get(name) {
            Some(Value::Bytes(b)) => Some(b),
            _ => None,
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        fn sized(b: &mut Vec<u8>, data: &[u8]) {
            b.extend_from_slice(&(data.len() as u32).to_be_bytes());
            b.extend_from_slice(data);
        }
        let mut b = self.id.to_be_bytes().to_vec();
        for ((_, kind), value) in MESSAGES[self.id as usize].1.iter().zip(&self.fields) {
            match value {
                Value::Bool(v) => b.push(u8::from(*v)),
                Value::Float(v) => b.extend_from_slice(&v.to_ne_bytes()),
                Value::Double(v) => b.extend_from_slice(&v.to_ne_bytes()),
                Value::Long(v) => b.extend_from_slice(&v.to_be_bytes()),
                Value::Str(s) => sized(&mut b, s.as_bytes()),
                Value::Bytes(data) => {
                    if *kind == K::OptBytes {
                        b.push(1);
                    }
                    sized(&mut b, data)
                }
                Value::Absent => b.push(0),
            }
        }
        b
    }

    pub fn decode(data: &[u8]) -> Result<Self, CodecError> {
        let mut r = Reader(data);
        let id = i32::from_be_bytes(r.array()?);
        let (_, schema) = schema(id)?;
        let fields =
            schema.iter().map(|(name, kind)| r.value(name, *kind)).collect::<Result<_, _>>()?;
        Ok(Self { id, fields })
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        if self.0.len() < n {
            return Err(CodecError::Truncated);
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], CodecError> {
        Ok(self.take(N)?.try_into().unwrap_or([0; N]))
    }

    fn sized(&mut self) -> Result<&'a [u8], CodecError> {
        let len = u32::from_be_bytes(self.array()?) as usize;
        self.take(len)
    }

    fn value(&mut self, name: &'static str, kind: FieldKind) -> Result<Value, CodecError> {
        Ok(match kind {
            K::Bool => Value::Bool(self.array::<1>()?[0] != 0),
            K::Float => Value::Float(f32::from_ne_bytes(self.array()?)),
            K::Double => Value::Double(f64::from_ne_bytes(self.array()?)),
            K::Long => Value::Long(u64::from_be_bytes(self.array()?)),
            K::Str => Value::Str(
                std::str::from_utf8(self.sized()?).map_err(|_| CodecError::Utf8(name))?.to_owned(),
            ),
            K::Bytes => Value::Bytes(self.sized()?.to_vec()),
            K::OptBytes => match self.array::<1>()?[0] != 0 {
                true => Value::Bytes(self.sized()?.to_vec()),
                false => Value::Absent,
            },
        })
    }
}

/// One frame's payload.
pub async fn read_frame<R: AsyncRead + Unpin + ?Sized>(r: &mut R) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let mut buf = vec![0; u32::from_be_bytes(len) as usize];
    r.read_exact(&mut buf).await?;
    Ok(buf)
}

/// One frame.
pub async fn write_frame<W: AsyncWrite + Unpin + ?Sized>(w: &mut W, data: &[u8]) -> io::Result<()> {
    let len = u32::try_from(data.len()).map_err(|_| io::ErrorKind::InvalidInput)?;
    let mut buf = Vec::with_capacity(4 + data.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(data);
    w.write_all(&buf).await?;
    w.flush().await
}

/// `utils.time_ms()`: wall-clock milliseconds, rounded.
pub fn time_ms() -> i64 {
    (lk_agents_abi::unix_now() * 1000.0 + 0.5) as i64
}

// --------------------------------------------------------------------------------- plumbing

type BoxRead = Box<dyn AsyncRead + Send + Unpin>;
type BoxWrite = Box<dyn AsyncWrite + Send + Unpin>;

/// The socket is closed.
#[derive(Debug, thiserror::Error)]
#[error("duplex closed")]
pub struct Closed;

use lk_agents_abi::lock;

/// A one-way latch to wait on.
struct Flag(watch::Sender<bool>);

impl Flag {
    fn new() -> Self {
        Self(watch::channel(false).0)
    }

    fn set(&self) {
        self.0.send_replace(true);
    }

    async fn wait(&self) {
        let _ = self.0.subscribe().wait_for(|set| *set).await;
    }
}

/// The far future a non-finite timeout stands for.
const FOREVER: Duration = Duration::from_secs(1 << 30);

/// Seconds as a `Duration`: zero when negative or NaN.
fn secs(s: f64) -> Duration {
    if s.is_nan() || s <= 0.0 { Duration::ZERO } else { Duration::from_secs_f64(s.min(1e9)) }
}

/// `round(x, digits)`: Python rounds the exact binary value half to even, as formatting does.
fn round(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// The writing half, shared by every sender.
struct Writer(tokio::sync::Mutex<Option<BoxWrite>>);

impl Writer {
    async fn send(&self, data: &[u8]) -> Result<(), Closed> {
        let mut writer = self.0.lock().await;
        let w = writer.as_mut().ok_or(Closed)?;
        write_frame(w, data).await.map_err(|_| Closed)
    }

    async fn close(&self) {
        if let Some(mut w) = self.0.lock().await.take() {
            let _ = w.shutdown().await;
        }
    }
}

// ------------------------------------------------------------------------------- supervisor

/// How often the memory monitor samples the process.
pub const MEMORY_MONITOR_INTERVAL: Duration = Duration::from_secs(5);
/// The least time between two "high memory usage" warnings of one process.
pub const MEMORY_WARN_COOLDOWN: Duration = Duration::from_secs(120);
/// Warn again before the cooldown when usage grew this much since the last warning.
pub const MEMORY_WARN_RESET_DELTA_MB: f64 = 50.0;
/// How long a closing client writes what is still queued for the parent.
pub const FLUSH_TIMEOUT: Duration = Duration::from_secs(2);
/// Between SIGUSR1 (the child dumps its stacks) and SIGKILL, and after a DumpStackTraceRequest.
pub const KILL_GRACE: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExtraValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

/// A log record's extra fields; the host adds the process's own after them.
pub type Extra = Vec<(&'static str, ExtraValue)>;

/// Why a memory sample failed.
pub enum MemoryError<E> {
    /// The process is gone or not ours to read (`psutil.NoSuchProcess`, `psutil.AccessDenied`):
    /// warn once and stop monitoring.
    Unavailable(String),
    /// Anything else: log it and try again on the next sample.
    Failed(E),
    /// An interruption rather than an error (a `BaseException` such as `SystemExit`): stop
    /// monitoring without a log.
    Interrupted,
}

/// What the supervisor needs from the host language: logging and the process handle.
pub trait ProcHost: Send + Sync + 'static {
    type Error: Send + 'static;

    fn log(&self, level: Level, msg: &str, extra: Extra);
    /// An error log with the failure's traceback.
    fn log_exception(&self, msg: &str, err: Self::Error);
    fn is_alive(&self) -> bool;
    /// SIGKILL (`Process.kill`; a terminate on Windows).
    fn kill(&self);
    /// The process's footprint in MB and the metric it reflects (pss, uss or rss). Blocking: the
    /// supervisor calls it off the runtime.
    fn sample_memory(&self) -> Result<(f64, String), MemoryError<Self::Error>>;
    fn has_running_job(&self) -> bool;
    /// The process exited and was reaped: the host's handle can go.
    fn release(&self) {}
}

#[derive(Clone, Debug)]
pub struct SupervisorOptions {
    pub pid: u32,
    /// `"job"` or `"inference"`, the start of the memory logs.
    pub process_kind: String,
    pub initialize_timeout: f64,
    pub close_timeout: f64,
    pub memory_warn_mb: f64,
    pub memory_limit_mb: f64,
    pub ping_interval: f64,
    pub ping_timeout: f64,
    pub high_ping_threshold: f64,
    /// A process can be killed; a job thread (`ThreadJobExecutor`) cannot: its late
    /// initialization, ack or exit is logged and waited for instead, and `kill` asks it to shut
    /// down.
    pub killable: bool,
}

impl SupervisorOptions {
    /// What the logs call the supervised runner.
    fn runner(&self) -> &'static str {
        if self.killable { "process" } else { "job runner" }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[error("duplex closed")]
    Closed,
    #[error("first message must be InitializeResponse")]
    Unexpected,
    #[error("process initialization failed: {0}")]
    Failed(String),
    #[error(transparent)]
    Codec(#[from] CodecError),
}

/// The parent's side of one supervised process. The host spawns the process, hands over its end
/// of the socket, calls [`initialize`](Self::initialize) then [`supervise`](Self::supervise), and
/// reports the exit with [`exited`](Self::exited).
pub struct ProcSupervisor<H: ProcHost> {
    inner: Arc<Inner<H>>,
}

impl<H: ProcHost> Clone for ProcSupervisor<H> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

struct Inner<H: ProcHost> {
    host: H,
    opts: SupervisorOptions,
    spawned: Instant,
    reader: tokio::sync::Mutex<Option<BoxRead>>,
    writer: Writer,
    /// Every message read once supervising, for the host's main task.
    messages: tokio::sync::Mutex<mpsc::UnboundedReceiver<Vec<u8>>>,
    forward: Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>,
    closing: AtomicBool,
    killed: AtomicBool,
    shutdown_ack: Flag,
    shutting_down: Flag,
    exited: Flag,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl<H: ProcHost> ProcSupervisor<H> {
    pub fn new<S>(stream: S, host: H, opts: SupervisorOptions) -> Self
    where
        S: AsyncRead + AsyncWrite + Send + 'static,
    {
        let (reader, writer) = tokio::io::split(stream);
        let (forward, messages) = mpsc::unbounded_channel();
        Self {
            inner: Arc::new(Inner {
                host,
                opts,
                spawned: Instant::now(),
                reader: tokio::sync::Mutex::new(Some(Box::new(reader))),
                writer: Writer(tokio::sync::Mutex::new(Some(Box::new(writer)))),
                messages: tokio::sync::Mutex::new(messages),
                forward: Mutex::new(Some(forward)),
                closing: AtomicBool::new(false),
                killed: AtomicBool::new(false),
                shutdown_ack: Flag::new(),
                shutting_down: Flag::new(),
                exited: Flag::new(),
                tasks: Mutex::new(Vec::new()),
            }),
        }
    }

    /// The kill sequence ran on a live process.
    pub fn killed(&self) -> bool {
        self.inner.killed.load(Ordering::SeqCst)
    }

    pub async fn send(&self, data: &[u8]) -> Result<(), Closed> {
        self.inner.writer.send(data).await
    }

    /// The next message read while supervising; [`Closed`] once the socket closed and every
    /// message was received.
    pub async fn recv(&self) -> Result<Vec<u8>, Closed> {
        self.inner.messages.lock().await.recv().await.ok_or(Closed)
    }

    /// The InitializeRequest to send the process: the supervision's ping settings.
    pub fn initialize_request(&self, asyncio_debug: bool, http_proxy: &str) -> IpcMessage {
        let opts = &self.inner.opts;
        let fields = vec![
            Value::Bool(asyncio_debug),
            Value::Float(opts.ping_interval as f32),
            Value::Float(opts.ping_timeout as f32),
            Value::Float(opts.high_ping_threshold as f32),
            Value::Str(http_proxy.to_owned()),
        ];
        IpcMessage::new(INITIALIZE_REQUEST, fields).expect("the schema's fields")
    }

    /// Waits for the InitializeResponse to the InitializeRequest the host sent. `None` when it
    /// timed out: the process was then killed. Returns how long the process took.
    pub async fn initialize(&self) -> Result<Option<Duration>, InitError> {
        let inner = &self.inner;
        inner.host.log(Level::Info, &format!("initializing {}", inner.opts.runner()), Vec::new());
        let start = Instant::now();
        let frame = {
            let mut reader = inner.reader.lock().await;
            let r = reader.as_mut().ok_or(InitError::Closed)?;
            tokio::time::timeout(secs(inner.opts.initialize_timeout), read_frame(r)).await
        };
        let frame = match frame {
            Err(_) => {
                if inner.opts.killable {
                    inner.dump_and_kill().await;
                }
                return Ok(None);
            }
            Ok(Err(_)) => return Err(InitError::Closed),
            Ok(Ok(frame)) => frame,
        };
        let msg = IpcMessage::decode(&frame)?;
        if msg.id != INITIALIZE_RESPONSE {
            return Err(InitError::Unexpected);
        }
        let error = msg.get_str("error");
        if !error.is_empty() {
            return Err(InitError::Failed(error.to_owned()));
        }
        let elapsed = start.elapsed();
        let elapsed_time = ExtraValue::Float(round(elapsed.as_secs_f64(), 2));
        let msg = format!("{} initialized", inner.opts.runner());
        inner.host.log(Level::Info, &msg, vec![("elapsed_time", elapsed_time)]);
        Ok(Some(elapsed))
    }

    /// Starts reading, pinging and (with a memory threshold) the memory monitor. Called once the
    /// initialization settled, whatever its outcome. Needs a runtime context.
    pub fn supervise(&self) {
        let inner = &self.inner;
        let mut tasks = lock(&inner.tasks);
        if inner.exited_set() || !tasks.is_empty() {
            return;
        }
        // the process is killed when no pong arrives within ping_timeout; each pong moves this
        let (deadline, deadline_rx) =
            watch::channel(Instant::now() + secs(inner.opts.ping_timeout));
        tasks.push(tokio::spawn(read_loop(inner.clone(), deadline)));
        tasks.push(tokio::spawn(ping_loop(inner.clone())));
        // a job thread has no pong deadline (`ping_timeout` 0): it is only reported late
        if inner.opts.ping_timeout > 0.0 {
            tasks.push(tokio::spawn(pong_timeout(inner.clone(), deadline_rx)));
        }
        if inner.opts.memory_limit_mb > 0.0 || inner.opts.memory_warn_mb > 0.0 {
            tasks.push(tokio::spawn(memory_loop(inner.clone())));
        }
    }

    /// The graceful close of an initialized process: ShutdownRequest, then the ack and the exit
    /// within `close_timeout` each, killing the process when either is late.
    pub async fn shutdown(&self) {
        let inner = &self.inner;
        inner.closing.store(true, Ordering::SeqCst);
        let _ = inner.writer.send(&IpcMessage::shutdown_request().encode()).await;

        let close_timeout = secs(inner.opts.close_timeout);
        let killable = inner.opts.killable;
        if tokio::time::timeout(close_timeout, inner.shutdown_ack.wait()).await.is_err() {
            if killable {
                let msg = "process did not ack shutdown in time, killing process";
                inner.host.log(Level::Error, msg, Vec::new());
                inner.dump_and_kill().await;
            } else {
                inner.host.log(Level::Error, "job did not ack shutdown in time", Vec::new());
            }
        }
        inner.shutting_down.wait().await;
        if tokio::time::timeout(close_timeout, inner.exited.wait()).await.is_err() {
            if killable {
                let msg = "process did not exit in time, killing process";
                inner.host.log(Level::Error, msg, Vec::new());
                inner.dump_and_kill().await;
            } else {
                let msg = "job shutdown is taking too much time..";
                inner.host.log(Level::Error, msg, Vec::new());
                inner.exited.wait().await;
            }
        }
    }

    /// The kill sequence; a runner that cannot be killed is asked to shut down instead.
    pub async fn kill(&self) {
        if !self.inner.opts.killable {
            return self.shutdown().await;
        }
        self.inner.closing.store(true, Ordering::SeqCst);
        self.inner.dump_and_kill().await;
    }

    /// The process exited and was reaped with `exitcode`: stops supervising and closes the socket.
    pub async fn exited(&self, exitcode: Option<i64>) {
        let inner = &self.inner;
        inner.closing.store(true, Ordering::SeqCst);
        let tasks = std::mem::take(&mut *lock(&inner.tasks));
        for task in &tasks {
            task.abort();
        }
        for task in tasks {
            let _ = task.await;
        }
        lock(&inner.forward).take();
        inner.shutdown_ack.set();
        inner.shutting_down.set();
        inner.writer.close().await;
        inner.reader.lock().await.take();

        if exitcode != Some(0) && !self.killed() {
            let code = exitcode.map_or("None".to_owned(), |c| c.to_string());
            let msg = format!("process exited with non-zero exit code {code}");
            inner.host.log(Level::Error, &msg, Vec::new());
        }
        inner.exited.set();
        inner.host.release();
    }
}

impl<H: ProcHost> Inner<H> {
    fn exited_set(&self) -> bool {
        *self.exited.0.borrow()
    }

    fn stopping(&self) -> bool {
        self.closing.load(Ordering::SeqCst) || self.killed.load(Ordering::SeqCst)
    }

    async fn dump_and_kill(&self) {
        self.send_dump_signal().await;
        self.send_kill_signal().await;
    }

    /// Where SIGUSR1 does not exist (Windows), ask the child to dump its stacks over the socket.
    async fn send_dump_signal(&self) {
        let enabled = std::env::var("LK_DUMP_STACK_TRACES")
            .map(|v| !matches!(v.to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(false);
        if cfg!(unix) || !enabled {
            return;
        }
        let msg = "sending DumpStackTraceRequest message to process";
        self.host.log(Level::Info, msg, Vec::new());
        let dump = IpcMessage { id: DUMP_STACK_TRACE_REQUEST, fields: Vec::new() };
        if self.writer.send(&dump.encode()).await.is_ok() {
            tokio::time::sleep(KILL_GRACE).await;
        }
    }

    /// SIGUSR1 (the child dumps its stacks when enabled), then SIGKILL half a second later.
    async fn send_kill_signal(&self) {
        if !self.host.is_alive() {
            return;
        }
        self.host.log(Level::Info, "killing process", Vec::new());
        #[cfg(unix)]
        if let Ok(pid) = libc::pid_t::try_from(self.opts.pid)
            && pid > 0
        {
            self.host.log(Level::Info, "sending SIGUSR1 signal to process", Vec::new());
            // SAFETY: kill(2) takes plain integers.
            if unsafe { libc::kill(pid, libc::SIGUSR1) } == 0 {
                tokio::time::sleep(KILL_GRACE).await;
            }
        }
        if self.host.is_alive() {
            self.host.kill();
        }
        self.killed.store(true, Ordering::SeqCst);
    }

    fn memory_extra(&self, memory_mb: f64, metric: &str, baseline_mb: f64) -> Extra {
        vec![
            ("memory_usage_mb", ExtraValue::Float(round(memory_mb, 1))),
            ("memory_metric", ExtraValue::Str(metric.to_owned())),
            ("memory_warn_mb", ExtraValue::Float(self.opts.memory_warn_mb)),
            ("memory_limit_mb", ExtraValue::Float(self.opts.memory_limit_mb)),
            ("uptime", ExtraValue::Float(round(self.spawned.elapsed().as_secs_f64(), 1))),
            ("has_running_job", ExtraValue::Bool(self.host.has_running_job())),
            ("baseline_memory_mb", ExtraValue::Float(round(baseline_mb, 1))),
            ("growth_memory_mb", ExtraValue::Float(round(memory_mb - baseline_mb, 1))),
        ]
    }
}

/// The sole reader once supervising: tracks pongs and the shutdown handshake, forwards every
/// message to [`ProcSupervisor::recv`]. However it ends (closed socket, bad frame, aborted), the
/// shutdown waits resolve, so a close never waits for an ack that cannot arrive.
async fn read_loop<H: ProcHost>(inner: Arc<Inner<H>>, pong_deadline: watch::Sender<Instant>) {
    struct Done<'a, H: ProcHost>(&'a Inner<H>);
    impl<H: ProcHost> Drop for Done<'_, H> {
        fn drop(&mut self) {
            lock(&self.0.forward).take();
            self.0.shutdown_ack.set();
            self.0.shutting_down.set();
        }
    }
    let _done = Done(&inner);
    let Some(mut reader) = inner.reader.lock().await.take() else { return };
    let forward = lock(&inner.forward).clone();
    let ping_timeout = secs(inner.opts.ping_timeout);

    while let Ok(frame) = read_frame(&mut reader).await {
        let msg = match IpcMessage::decode(&frame) {
            Ok(msg) => msg,
            Err(e) => {
                let msg = format!("invalid IPC message: {e}");
                inner.host.log(Level::Error, &msg, Vec::new());
                break;
            }
        };
        match msg.id {
            PONG_RESPONSE => {
                let Some(Value::Long(timestamp)) = msg.get("timestamp") else { unreachable!() };
                let delay = time_ms() - *timestamp as i64;
                if delay as f64 > inner.opts.high_ping_threshold * 1000.0 {
                    let extra = vec![("delay", ExtraValue::Int(delay))];
                    let msg =
                        if inner.opts.killable { "process is unresponsive" } else { "job executor is unresponsive" };
                    inner.host.log(Level::Warning, msg, extra);
                }
                // no receiver once the timeout fired: a late pong does not revive it
                let _ = pong_deadline.send(Instant::now() + ping_timeout);
            }
            SHUTDOWN_REQUEST_ACK => inner.shutdown_ack.set(),
            SHUTTING_DOWN => inner.shutting_down.set(),
            EXITING => {
                let extra = vec![("reason", ExtraValue::Str(msg.get_str("reason").to_owned()))];
                if inner.opts.killable {
                    inner.host.log(Level::Info, "process exiting", extra);
                } else {
                    inner.host.log(Level::Debug, "job exiting", extra);
                }
            }
            _ => {}
        }
        if let Some(forward) = &forward {
            let _ = forward.send(frame);
        }
    }
}

async fn ping_loop<H: ProcHost>(inner: Arc<Inner<H>>) {
    let period = secs(inner.opts.ping_interval).max(Duration::from_millis(1));
    let mut interval = tokio::time::interval(period);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        let ping = IpcMessage::ping(time_ms().max(0) as u64);
        if inner.writer.send(&ping.encode()).await.is_err() {
            break;
        }
    }
}

async fn pong_timeout<H: ProcHost>(inner: Arc<Inner<H>>, mut deadline: watch::Receiver<Instant>) {
    loop {
        let at = (*deadline.borrow_and_update()).min(Instant::now() + FOREVER);
        tokio::select! {
            () = tokio::time::sleep_until(at) => break,
            changed = deadline.changed() => if changed.is_err() {
                // the reader ended; the last deadline still stands
                tokio::time::sleep_until(at).await;
                break;
            },
        }
    }
    drop(deadline);
    inner.host.log(Level::Error, "process is unresponsive, killing process", Vec::new());
    inner.dump_and_kill().await;
}

/// Rate-limits the high memory warning: once per cooldown, or sooner when usage grew.
#[derive(Default)]
struct WarnGate {
    last: Option<(Instant, f64)>,
}

impl WarnGate {
    fn should_warn(&mut self, memory_mb: f64, now: Instant) -> bool {
        let emit = match self.last {
            None => true,
            Some((at, mb)) => {
                now.duration_since(at) >= MEMORY_WARN_COOLDOWN
                    || memory_mb - mb >= MEMORY_WARN_RESET_DELTA_MB
            }
        };
        if emit {
            self.last = Some((now, memory_mb));
        }
        emit
    }
}

async fn memory_loop<H: ProcHost>(inner: Arc<Inner<H>>) {
    let (warn_mb, limit_mb) = (inner.opts.memory_warn_mb, inner.opts.memory_limit_mb);
    let kind = inner.opts.process_kind.clone();
    // the first sample, taken just after initialization, is the post-prewarm baseline
    let mut baseline_mb = None;
    let mut gate = WarnGate::default();
    while !inner.stopping() {
        let host = inner.clone();
        let sample = tokio::task::spawn_blocking(move || host.host.sample_memory()).await;
        match sample {
            Ok(Ok((memory_mb, metric))) => {
                let baseline = *baseline_mb.get_or_insert(memory_mb);
                if limit_mb > 0.0 && memory_mb > limit_mb {
                    let msg = format!("{kind} process exceeded memory limit, killing it");
                    let extra = inner.memory_extra(memory_mb, &metric, baseline);
                    inner.host.log(Level::Error, &msg, extra);
                    inner.dump_and_kill().await;
                } else if warn_mb > 0.0
                    && memory_mb > warn_mb
                    && gate.should_warn(memory_mb, Instant::now())
                {
                    // without a hard limit nothing is terminated: say so
                    let advisory = match limit_mb <= 0.0 {
                        true => " (advisory only, the process will not be terminated)",
                        false => "",
                    };
                    let msg = format!(
                        "{kind} process memory usage is above the warning threshold{advisory}"
                    );
                    let extra = inner.memory_extra(memory_mb, &metric, baseline);
                    inner.host.log(Level::Warning, &msg, extra);
                }
            }
            Ok(Err(MemoryError::Unavailable(e))) => {
                if !inner.stopping() {
                    let msg = format!("failed to get memory info for process: {e}");
                    inner.host.log(Level::Warning, &msg, Vec::new());
                }
                return;
            }
            Ok(Err(MemoryError::Interrupted)) | Err(_) => return,
            Ok(Err(MemoryError::Failed(e))) => {
                if inner.stopping() {
                    return;
                }
                inner.host.log_exception("Error in memory monitoring task", e);
            }
        }
        tokio::time::sleep(MEMORY_MONITOR_INTERVAL).await;
    }
}

// ----------------------------------------------------------------------------------- client

/// The child's grace on top of the parent's `ping_timeout` before it calls itself orphaned.
pub const HEALTH_CHECK_GRACE: f64 = 10.0;

/// The child's side of the socket: the initialize handshake ([`handshake`](Self::handshake)),
/// then the parent's messages, read once [`serve`](Self::serve)d. What the child sends goes out in
/// order through one queue ([`queue`](Self::queue)). With a health timeout, it prints to stderr and stops reading when the
/// parent sends nothing for that long, whatever the state of the child's event loop.
pub struct IpcClient {
    outbox: Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>,
    writer: Mutex<Option<JoinHandle<()>>>,
    /// The reading half until it is served.
    unread: tokio::sync::Mutex<Option<BoxRead>>,
    forward: Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>,
    messages: tokio::sync::Mutex<mpsc::UnboundedReceiver<Vec<u8>>>,
    reader: Mutex<Option<JoinHandle<()>>>,
}

impl IpcClient {
    /// Needs a runtime context.
    pub fn new<S>(stream: S) -> Self
    where
        S: AsyncRead + AsyncWrite + Send + 'static,
    {
        let (reader, mut writer) = tokio::io::split(stream);
        let (outbox, mut queued) = mpsc::unbounded_channel::<Vec<u8>>();
        let writer = tokio::spawn(async move {
            while let Some(data) = queued.recv().await {
                if write_frame(&mut writer, &data).await.is_err() {
                    return;
                }
            }
            let _ = writer.shutdown().await;
        });
        let (forward, messages) = mpsc::unbounded_channel();
        Self {
            outbox: Mutex::new(Some(outbox)),
            writer: Mutex::new(Some(writer)),
            unread: tokio::sync::Mutex::new(Some(Box::new(reader))),
            forward: Mutex::new(Some(forward)),
            messages: tokio::sync::Mutex::new(messages),
            reader: Mutex::new(None),
        }
    }

    /// The parent's first message, its InitializeRequest.
    pub async fn handshake(&self) -> Result<Vec<u8>, Closed> {
        let mut unread = self.unread.lock().await;
        read_frame(unread.as_mut().ok_or(Closed)?).await.map_err(|_| Closed)
    }

    /// Reads the parent's messages from now on, for [`recv`](Self::recv). Needs a runtime context.
    pub fn serve(&self, health_timeout: Option<Duration>) {
        let reader = self.unread.try_lock().ok().and_then(|mut r| r.take());
        let forward = lock(&self.forward).take();
        if let (Some(reader), Some(forward)) = (reader, forward) {
            *lock(&self.reader) = Some(tokio::spawn(client_read_loop(reader, forward, health_timeout)));
        }
    }

    /// Queues a message for the parent.
    pub fn queue(&self, data: Vec<u8>) -> Result<(), Closed> {
        lock(&self.outbox).as_ref().ok_or(Closed)?.send(data).map_err(|_| Closed)
    }

    pub async fn recv(&self) -> Result<Vec<u8>, Closed> {
        self.messages.lock().await.recv().await.ok_or(Closed)
    }

    /// Stops reading, writes what is queued (for up to [`FLUSH_TIMEOUT`]) and closes the socket.
    pub async fn close(&self) {
        if let Some(reader) = lock(&self.reader).take() {
            reader.abort();
        }
        lock(&self.outbox).take();
        let writer = lock(&self.writer).take();
        if let Some(mut writer) = writer {
            if tokio::time::timeout(FLUSH_TIMEOUT, &mut writer).await.is_err() {
                writer.abort();
            }
        }
    }
}

async fn client_read_loop(
    mut reader: BoxRead,
    forward: mpsc::UnboundedSender<Vec<u8>>,
    health_timeout: Option<Duration>,
) {
    let health_timeout = health_timeout.map(|t| t.min(FOREVER));
    let mut deadline = health_timeout.map(|t| Instant::now() + t);
    loop {
        let expired = async {
            match deadline {
                Some(at) => tokio::time::sleep_until(at).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            frame = read_frame(&mut reader) => {
                let Ok(frame) = frame else { break };
                deadline = health_timeout.map(|t| Instant::now() + t);
                if forward.send(frame).is_err() {
                    break;
                }
            }
            () = expired => {
                eprintln!("worker process is not responding.. worker crashed?");
                break;
            }
        }
    }
}
