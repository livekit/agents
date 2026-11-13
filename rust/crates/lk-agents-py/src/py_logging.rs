/// Interpreter interface namespaces. Generated inherent methods perform the declared
/// imports and member access; they accept Python objects without asserting their runtime type.
pub mod interfaces {
    #[lk_agents_host_derive::host(interface(
        module = "logging",
        attrs(get_logger(name = "getLogger"))
    ))]
    pub struct LoggingModule;

    #[lk_agents_host_derive::host(interface(
        host = "logging.Logger",
        attrs(
            get_effective_level(name = "getEffectiveLevel"),
            is_enabled_for(name = "isEnabledFor"),
            make_record(name = "makeRecord"),
            handle
        )
    ))]
    pub struct Logger;

    #[lk_agents_host_derive::host(interface(
        host = "logging.LogRecord",
        attrs(created(write), msecs(write), exc_text(write))
    ))]
    pub struct LogRecord;

    #[lk_agents_host_derive::host(interface(module = "atexit", attrs(register)))]
    pub struct AtExitModule;

    #[lk_agents_host_derive::host(interface(module = "livekit.agents._core", attrs(flush_logs)))]
    pub struct CoreModule;
}

use crate::generated::accessors as access;
use log::Level;
use log::LevelFilter;
use log::Log;
use log::Metadata;
use log::Record;
use parking_lot::Condvar;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::SyncSender;
use std::sync::mpsc::TrySendError;
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

const QUEUE: usize = 4096;
const FLUSH_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a logger's cached level filters lines before one line is let through to reread it.
const LEVEL_TTL: Duration = Duration::from_secs(1);

struct Line {
    level: Level,
    target: String,
    message: String,
    file: Option<String>,
    line: Option<u32>,
    key_values: Vec<(String, Field)>,
    at: SystemTime,
}

/// A key-value as Python's `extra` holds it.
#[derive(Clone, Debug, PartialEq)]
enum Field {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

/// The key-value [`emit`] sets as the record's `exc_text`: a failure's formatted traceback.
const EXC_TEXT: &str = "exc_text";

/// The queue's sender, for the process that started its drain (a fork starts its own).
struct Queue {
    pid: u32,
    sender: SyncSender<Line>,
    progress: Arc<Progress>,
}

/// The lines queued and the lines the drain has forwarded since: [`flush`] waits for the second
/// to reach the first.
#[derive(Default)]
struct Progress {
    queued: AtomicU64,
    forwarded: Mutex<u64>,
    advanced: Condvar,
}

impl Progress {
    /// Waits, up to `timeout`, until `target` lines were forwarded.
    fn wait(&self, target: u64, timeout: Duration) {
        let mut forwarded = self.forwarded.lock();
        self.advanced.wait_while_for(&mut forwarded, |forwarded| *forwarded < target, timeout);
    }

    fn forwarded(&self) {
        *self.forwarded.lock() += 1;
        self.advanced.notify_all();
    }
}

struct Forwarder {
    queue: Mutex<Option<Queue>>,
    dropped: AtomicU64,
    /// Which Rust lines reach Python at all: [`DEFAULT_DIRECTIVES`], then `RUST_LOG`'s.
    directives: env_filter::Filter,
    /// Each logger's effective level and when the drain last read it: `enabled` reads it without
    /// the GIL. A target not read yet, or read more than [`LEVEL_TTL`] ago, is enabled, so the
    /// next line refreshes a level the host raised since.
    levels: Mutex<BTreeMap<String, (LevelFilter, Instant)>>,
}

static FORWARDER: OnceLock<Forwarder> = OnceLock::new();
thread_local! {
    // A Python handler can call flush_logs from the drain itself.
    static IN_DRAIN: Cell<bool> = const { Cell::new(false) };
}

fn forwarder() -> &'static Forwarder {
    FORWARDER.get_or_init(|| Forwarder {
        queue: Mutex::new(None),
        dropped: AtomicU64::new(0),
        directives: directives(std::env::var("RUST_LOG").ok().as_deref()),
        levels: Mutex::new(BTreeMap::new()),
    })
}

/// Routes the core's `log` lines to `_core`'s Python process, and flushes them at its exit.
/// Starts nothing: the drain starts on the first line.
pub fn install(module: &Bound<'_, PyModule>) -> PyResult<()> {
    install_logger();
    let flush_logs = access::CoreModule::flush_logs(module)?;
    access::AtExitModule::register(&(access::AtExitModule::import(module.py())?))?
        .call1((flush_logs,))?;
    Ok(())
}

/// Installs the forwarder as the `log` logger.
pub fn install_logger() {
    struct Installed;
    impl Log for Installed {
        fn enabled(&self, metadata: &Metadata) -> bool {
            forwarder().enabled(metadata)
        }
        fn log(&self, record: &Record) {
            forwarder().queue(record);
        }
        fn flush(&self) {
            flush();
        }
    }
    if log::set_logger(&Installed).is_ok() {
        // `enabled` filters: the directives first (no lock), then Python's levels.
        log::set_max_level(LevelFilter::Trace);
    }
}

/// The declared `_core.flush_logs`.
pub struct Logging;

impl lk_agents_abi::invoke::Logging for Logging {
    fn flush_logs() {
        flush();
    }
}

/// Waits (up to [`FLUSH_TIMEOUT`], without the GIL) until every line queued before the call has
/// reached Python's `logging`, then reads the loggers' levels again. A test boundary and the
/// interpreter's exit call it.
pub fn flush() {
    // A logger's level may have changed since the drain read it (a test's `caplog`).
    forwarder().levels.lock().clear();
    if IN_DRAIN.get() {
        return;
    }
    let pid = std::process::id();
    let queued = forwarder()
        .queue
        .lock()
        .as_ref()
        .filter(|queue| queue.pid == pid)
        .map(|queue| queue.progress.clone());
    let Some(progress) = queued else { return };
    let target = progress.queued.load(Ordering::SeqCst);
    let wait = move || progress.wait(target, FLUSH_TIMEOUT);
    if crate::py_calls::holds_gil() {
        Python::attach(|py| py.detach(wait));
    } else {
        wait();
    }
}

/// LiveKit's own lines pass at any level (Python's logger levels decide); other crates' only from
/// warnings up. The stall detector's sampler (py-spy, its process reader and binary parser) logs
/// internals; the detector logs what matters (`event-loop stall stacks unavailable`).
const DEFAULT_DIRECTIVES: &str =
    "warn,livekit=trace,lk_agents=trace,py_spy=error,remoteprocess=error,goblin=error";

/// The default directives, overridden by `rust_log`'s (`RUST_LOG` syntax: `rustls=debug`).
fn directives(rust_log: Option<&str>) -> env_filter::Filter {
    let mut builder = env_filter::Builder::new();
    builder.parse(DEFAULT_DIRECTIVES);
    if let Some(rust_log) = rust_log {
        builder.parse(rust_log);
    }
    builder.build()
}

impl Forwarder {
    fn enabled(&self, metadata: &Metadata) -> bool {
        if !self.directives.enabled(metadata) {
            return false;
        }
        let levels = self.levels.lock();
        levels
            .get(metadata.target())
            .is_none_or(|(filter, read)| metadata.level() <= *filter || read.elapsed() > LEVEL_TTL)
    }

    fn queue(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let mut key_values = Vec::new();
        let _ = record.key_values().visit(&mut KeyValues(&mut key_values));
        let line = Line {
            level: record.level(),
            target: record.target().to_owned(),
            message: record.args().to_string(),
            file: record.file().map(str::to_owned),
            line: record.line(),
            key_values,
            at: SystemTime::now(),
        };
        let Some((sender, progress)) = self.sender() else { return };
        match sender.try_send(line) {
            Ok(()) => {
                progress.queued.fetch_add(1, Ordering::SeqCst);
            }
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    /// This process's sender, its drain started on first use.
    fn sender(&self) -> Option<(SyncSender<Line>, Arc<Progress>)> {
        let pid = std::process::id();
        let mut queue = self.queue.lock();
        if let Some(queue) = queue.as_ref().filter(|queue| queue.pid == pid) {
            return Some((queue.sender.clone(), queue.progress.clone()));
        }
        let (sender, lines) = sync_channel(QUEUE);
        let progress = Arc::new(Progress::default());
        let drained = progress.clone();
        std::thread::Builder::new()
            .name("lk-log-drain".into())
            .spawn(move || drain(lines, &drained))
            .ok()?;
        let parents =
            queue.replace(Queue { pid, sender: sender.clone(), progress: progress.clone() });
        // A forked child never drops its parent's sender: disconnecting would wake the parent's
        // drain, which does not exist here (on macOS, a trap in libdispatch).
        std::mem::forget(parents);
        Some((sender, progress))
    }
}

/// A line's key-values, typed as Python's `extra` takes them; a null (`None`) one is left out.
struct KeyValues<'a>(&'a mut Vec<(String, Field)>);

impl<'kvs> log::kv::VisitSource<'kvs> for KeyValues<'_> {
    fn visit_pair(
        &mut self,
        key: log::kv::Key<'kvs>,
        value: log::kv::Value<'kvs>,
    ) -> Result<(), log::kv::Error> {
        let mut field = None;
        value.visit(Typed(&mut field))?;
        if let Some(field) = field {
            self.0.push((key.as_str().to_owned(), field));
        }
        Ok(())
    }
}

struct Typed<'a>(&'a mut Option<Field>);

impl<'v> log::kv::VisitValue<'v> for Typed<'_> {
    fn visit_any(&mut self, value: log::kv::Value<'_>) -> Result<(), log::kv::Error> {
        *self.0 = Some(Field::Str(value.to_string()));
        Ok(())
    }
    fn visit_null(&mut self) -> Result<(), log::kv::Error> {
        Ok(())
    }
    fn visit_u64(&mut self, value: u64) -> Result<(), log::kv::Error> {
        *self.0 = Some(i64::try_from(value).map_or(Field::Float(value as f64), Field::Int));
        Ok(())
    }
    fn visit_i64(&mut self, value: i64) -> Result<(), log::kv::Error> {
        *self.0 = Some(Field::Int(value));
        Ok(())
    }
    fn visit_f64(&mut self, value: f64) -> Result<(), log::kv::Error> {
        *self.0 = Some(Field::Float(value));
        Ok(())
    }
    fn visit_bool(&mut self, value: bool) -> Result<(), log::kv::Error> {
        *self.0 = Some(Field::Bool(value));
        Ok(())
    }
    fn visit_str(&mut self, value: &str) -> Result<(), log::kv::Error> {
        *self.0 = Some(Field::Str(value.to_owned()));
        Ok(())
    }
}

/// When the drain began forwarding the line in hand (nanoseconds since [`DRAIN_EPOCH`], plus
/// one): from taking it, through waiting for the GIL, to Python's handlers returning; 0 between
/// lines.
static FORWARDING_SINCE: AtomicU64 = AtomicU64::new(0);
static DRAIN_EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn since_epoch() -> u64 {
    u64::try_from(DRAIN_EPOCH.elapsed().as_nanos()).unwrap_or(u64::MAX).saturating_add(1)
}

/// Whether one line has been on its way to Python's `logging` for `at_least`, waiting for the GIL
/// or in a handler: the core's log lines are not getting through (the event-loop stall detector
/// writes to stderr then).
pub fn forwarding_blocked(at_least: Duration) -> bool {
    let since = FORWARDING_SINCE.load(Ordering::Relaxed);
    since != 0 && since_epoch().saturating_sub(since) >= at_least.as_nanos() as u64
}

fn drain(lines: Receiver<Line>, progress: &Progress) {
    IN_DRAIN.set(true);
    for line in lines {
        FORWARDING_SINCE.store(since_epoch(), Ordering::Relaxed);
        Python::attach(|py| forward_line(py, &line));
        FORWARDING_SINCE.store(0, Ordering::Relaxed);
        progress.forwarded();
    }
}

/// Emits the line, preceded by a note of the lines dropped since the last one.
fn forward_line(py: Python<'_>, line: &Line) {
    let dropped = forwarder().dropped.swap(0, Ordering::Relaxed);
    if dropped > 0 {
        let note = format!("{dropped} log lines were dropped: the queue was full");
        let _ = emit(py, &Line { level: Level::Warn, message: note, ..line.clone_meta() });
    }
    if let Err(error) = emit(py, line) {
        error.print(py);
    }
}

impl Line {
    fn clone_meta(&self) -> Line {
        Line {
            level: self.level,
            target: self.target.clone(),
            message: String::new(),
            file: self.file.clone(),
            line: self.line,
            key_values: Vec::new(),
            at: self.at,
        }
    }
}

/// `logging.getLogger(target)` handles the line, stamped when it was logged.
fn emit(py: Python<'_>, line: &Line) -> PyResult<()> {
    let name = line.target.replace("::", ".");
    let logger = access::LoggingModule::get_logger(&(access::LoggingModule::import(py)?))?
        .call1((&name,))?;
    let level = match line.level {
        Level::Error => 40,
        Level::Warn => 30,
        Level::Info => 20,
        Level::Debug => 10,
        Level::Trace => 5,
    };
    let effective: u32 = access::Logger::get_effective_level(&logger)?.call0()?.extract()?;
    forwarder().levels.lock().insert(line.target.clone(), (filter(effective), Instant::now()));
    if !access::Logger::is_enabled_for(&logger)?.call1((level,))?.extract::<bool>()? {
        return Ok(());
    }
    let extra = PyDict::new(py);
    let mut exc_text = None;
    for (key, value) in &line.key_values {
        match value {
            Field::Str(text) if key == EXC_TEXT => exc_text = Some(text),
            Field::Int(value) => extra.set_item(key, value)?,
            Field::Float(value) => extra.set_item(key, value)?,
            Field::Bool(value) => extra.set_item(key, value)?,
            Field::Str(value) => extra.set_item(key, value)?,
        }
    }
    let record = access::Logger::make_record(&logger)?.call1((
        &name,
        level,
        line.file.as_deref().unwrap_or(""),
        line.line.unwrap_or(0),
        &line.message,
        pyo3::types::PyTuple::empty(py),
        py.None(),
        py.None(),
        extra,
    ))?;
    let created = line.at.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs_f64();
    access::LogRecord::set_created(record.as_any(), created)?;
    access::LogRecord::set_msecs(record.as_any(), (created.fract() * 1000.0).floor())?;
    if let Some(text) = exc_text {
        access::LogRecord::set_exc_text(record.as_any(), text)?;
    }
    access::Logger::handle(&logger)?.call1((record,))?;
    Ok(())
}

/// The most verbose `log` level a Python logger at `effective` passes.
fn filter(effective: u32) -> LevelFilter {
    match effective {
        0..=5 => LevelFilter::Trace,
        6..=10 => LevelFilter::Debug,
        11..=20 => LevelFilter::Info,
        21..=30 => LevelFilter::Warn,
        31..=40 => LevelFilter::Error,
        _ => LevelFilter::Off,
    }
}

#[cfg(test)]
mod logging_tests {
    use pyo3::prelude::*;
    use std::sync::mpsc;
    use std::time::Duration;

    // Catches: a log line waiting for the GIL while the logging thread holds what a GIL holder
    // waits for (a lock, here a reply).
    #[test]
    fn a_line_logged_while_a_gil_holder_waits_on_the_logger_completes_and_arrives() {
        Python::initialize();
        crate::py_logging::install_logger();
        let seen = Python::attach(|py| {
            let globals = crate::py_test_support::fixture(py, "logging_handler");
            globals.getattr("seen").unwrap().unbind()
        });
        let (attached, attaching) = mpsc::channel();
        let (logged, logging) = mpsc::channel::<()>();
        let waiter = std::thread::spawn(move || {
            Python::attach(|_| {
                attached.send(()).unwrap();
                logging.recv_timeout(Duration::from_secs(10)).is_ok()
            })
        });
        attaching.recv().unwrap();
        log::warn!(target: "livekit.agents.logtest", "under a lock");
        logged.send(()).unwrap();
        assert!(waiter.join().unwrap(), "the line waited for the GIL");
        crate::py_logging::flush();
        Python::attach(|py| {
            let messages: Vec<String> = seen.bind(py).extract().unwrap();
            assert_eq!(messages, ["under a lock"]);
            let logging = py.import("logging").unwrap();
            let previous = logging
                .getattr("root")
                .unwrap()
                .getattr("manager")
                .unwrap()
                .getattr("disable")
                .unwrap()
                .extract::<i32>()
                .unwrap();
            logging.call_method1("disable", (50,)).unwrap();
            let line = super::Line {
                level: log::Level::Warn,
                target: "livekit.agents.logtest".into(),
                message: "globally disabled".into(),
                file: None,
                line: None,
                key_values: Vec::new(),
                at: std::time::SystemTime::now(),
            };
            let emitted = super::emit(py, &line);
            logging.call_method1("disable", (previous,)).unwrap();
            emitted.unwrap();
            let messages: Vec<String> = seen.bind(py).extract().unwrap();
            assert_eq!(messages, ["under a lock"], "logging.disable must be respected");
        });
    }

    #[test]
    fn a_flush_waits_for_the_lines_queued_before_it_up_to_its_timeout() {
        let progress = std::sync::Arc::new(super::Progress::default());
        let started = std::time::Instant::now();
        progress.wait(1, Duration::from_millis(30));
        assert!(started.elapsed() < Duration::from_secs(1), "the wait exceeded its timeout");
        let drain = progress.clone();
        let forwarding = std::thread::spawn(move || drain.forwarded());
        progress.wait(1, Duration::from_secs(10));
        forwarding.join().unwrap();
        assert_eq!(*progress.forwarded.lock(), 1);
    }

    // Catches: a key-value reaching Python's `extra` as its text rather than its value, or a
    // `None` one present.
    #[test]
    fn key_values_keep_their_type_and_a_null_one_is_left_out() {
        let pairs = [
            ("pid", log::kv::Value::from(7_u32)),
            ("uptime", log::kv::Value::from(1.5_f64)),
            ("has_running_job", log::kv::Value::from(true)),
            ("room", log::kv::Value::from("logging")),
            ("room_id", log::kv::Value::null()),
        ];
        let record = log::Record::builder().key_values(&pairs).build();
        let mut fields = Vec::new();
        record.key_values().visit(&mut super::KeyValues(&mut fields)).unwrap();
        use super::Field;
        assert_eq!(
            fields,
            [
                ("pid".to_owned(), Field::Int(7)),
                ("uptime".to_owned(), Field::Float(1.5)),
                ("has_running_job".to_owned(), Field::Bool(true)),
                ("room".to_owned(), Field::Str("logging".into())),
            ]
        );
    }

    #[test]
    fn a_drain_handler_can_flush_without_waiting_for_itself() {
        Python::initialize();
        // The drain cannot acquire the GIL here, so a barrier would time out if flush waited.
        Python::attach(|_| {
            super::forwarder().sender().unwrap();
            let started = std::time::Instant::now();
            super::IN_DRAIN.set(true);
            super::flush();
            super::IN_DRAIN.set(false);
            assert!(started.elapsed() < Duration::from_secs(1));
        });
    }

    // Catches: other crates' debug lines (py-spy's Mach-O parser, TLS handshakes) flooding the
    // console, and `RUST_LOG` failing to bring one back or to quiet LiveKit's own.
    #[test]
    fn only_livekit_lines_pass_below_warn_unless_rust_log_says_otherwise() {
        let passes = |rust_log, target, level| {
            let metadata = log::Metadata::builder().target(target).level(level).build();
            super::directives(rust_log).enabled(&metadata)
        };
        use log::Level::{Debug, Error, Info, Warn};
        assert!(passes(None, lk_agents_abi::LOG_TARGET, Debug));
        assert!(passes(None, "livekit::room", Debug));
        assert!(!passes(None, "goblin::mach", Debug));
        assert!(!passes(None, "rustls::client::hs", Debug));
        assert!(passes(None, "rustls::client::hs", Warn));
        assert!(!passes(None, "py_spy::python_spy", Warn));
        assert!(passes(None, "py_spy::python_spy", Error));

        assert!(passes(Some("rustls=debug"), "rustls::client::hs", Debug));
        assert!(!passes(Some("rustls=debug"), "goblin::mach", Debug));
        assert!(!passes(Some("livekit=info"), lk_agents_abi::LOG_TARGET, Debug));
        assert!(passes(Some("livekit=info"), lk_agents_abi::LOG_TARGET, Info));
    }
}
