//! Rust log records forwarded to Python without waiting for the GIL.

/// Interpreter interface namespaces. Generated inherent methods perform the declared
/// imports and member access; they accept Python objects without asserting their runtime type.
pub mod interfaces {
    #[lk_agents_abi::host(interface(module = "logging", attrs(get_logger(name = "getLogger"))))]
    pub struct LoggingModule;

    #[lk_agents_abi::host(interface(
        host = "logging.Logger",
        attrs(
            get_effective_level(name = "getEffectiveLevel"),
            is_enabled_for(name = "isEnabledFor"),
            make_record(name = "makeRecord"),
            handle
        )
    ))]
    pub struct Logger;

    #[lk_agents_abi::host(interface(
        host = "logging.LogRecord",
        attrs(created(write), msecs(write))
    ))]
    pub struct LogRecord;

    #[lk_agents_abi::host(interface(module = "atexit", attrs(register)))]
    pub struct AtExitModule;

    #[lk_agents_abi::host(interface(module = "livekit.agents._core", attrs(flush_logs)))]
    pub struct CoreModule;
}

use crate::generated::accessors as access;
use log::Level;
use log::LevelFilter;
use log::Log;
use log::Metadata;
use log::Record;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::SyncSender;
use std::sync::mpsc::TrySendError;
use std::sync::mpsc::sync_channel;
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
    key_values: Vec<(String, String)>,
    at: SystemTime,
}

enum Entry {
    Line(Line),
    Flush(SyncSender<()>),
}

/// The queue's sender, for the process that started its drain (a fork starts its own).
struct Queue {
    pid: u32,
    sender: SyncSender<Entry>,
}

struct Forwarder {
    queue: Mutex<Option<Queue>>,
    dropped: AtomicU64,
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
        log::set_max_level(LevelFilter::Debug);
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
        .map(|queue| queue.sender.clone());
    let Some(sender) = queued else { return };
    let wait = move || flush_sender(&sender, FLUSH_TIMEOUT);
    if crate::py_calls::holds_gil() {
        Python::attach(|py| py.detach(wait));
    } else {
        wait();
    }
}

fn flush_sender(sender: &SyncSender<Entry>, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    let (done, flushed) = sync_channel(1);
    let mut entry = Entry::Flush(done);
    loop {
        match sender.try_send(entry) {
            Ok(()) => break,
            Err(TrySendError::Disconnected(_)) => return,
            Err(TrySendError::Full(pending)) => entry = pending,
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return;
        }
        std::thread::sleep(remaining.min(Duration::from_millis(1)));
    }
    let _ = flushed.recv_timeout(deadline.saturating_duration_since(Instant::now()));
}

impl Forwarder {
    fn enabled(&self, metadata: &Metadata) -> bool {
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
        let Some(sender) = self.sender() else { return };
        if let Err(TrySendError::Full(_)) = sender.try_send(Entry::Line(line)) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// This process's sender, its drain started on first use.
    fn sender(&self) -> Option<SyncSender<Entry>> {
        let pid = std::process::id();
        let mut queue = self.queue.lock();
        if let Some(queue) = queue.as_ref().filter(|queue| queue.pid == pid) {
            return Some(queue.sender.clone());
        }
        let (sender, entries) = sync_channel(QUEUE);
        std::thread::Builder::new()
            .name("lk-log-drain".into())
            .spawn(move || drain(entries))
            .ok()?;
        *queue = Some(Queue { pid, sender: sender.clone() });
        Some(sender)
    }
}

struct KeyValues<'a>(&'a mut Vec<(String, String)>);

impl<'kvs> log::kv::VisitSource<'kvs> for KeyValues<'_> {
    fn visit_pair(
        &mut self,
        key: log::kv::Key<'kvs>,
        value: log::kv::Value<'kvs>,
    ) -> Result<(), log::kv::Error> {
        self.0.push((key.as_str().to_owned(), value.to_string()));
        Ok(())
    }
}

fn drain(entries: Receiver<Entry>) {
    IN_DRAIN.set(true);
    for entry in entries {
        match entry {
            Entry::Line(line) => Python::attach(|py| forward_line(py, &line)),
            Entry::Flush(done) => {
                let _ = done.send(());
            }
        }
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
    for (key, value) in &line.key_values {
        extra.set_item(key, value)?;
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

    // Catches: a log line waiting for the GIL. The thread logging holds what a GIL holder waits
    // for (a lock, here a reply), as review14's storage_lost warn under the Requests lock did.
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
    fn flush_timeout_includes_waiting_for_space_in_the_queue() {
        let (sender, _receiver) = mpsc::sync_channel(1);
        let (done, _flushed) = mpsc::sync_channel(1);
        assert!(sender.try_send(super::Entry::Flush(done)).is_ok());
        let started = std::time::Instant::now();
        super::flush_sender(&sender, Duration::from_millis(30));
        assert!(started.elapsed() < Duration::from_secs(1), "enqueue exceeded the flush deadline");
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
}
